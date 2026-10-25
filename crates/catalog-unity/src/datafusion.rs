//! Datafusion integration for UnityCatalog

use dashmap::DashMap;
use datafusion::catalog::SchemaProvider;
use datafusion::catalog::{CatalogProvider, CatalogProviderList};
use datafusion::common::DataFusionError;
use datafusion::datasource::TableProvider;
use object_store_unity::UnityStoreFactory;
use std::sync::Arc;
use tracing::{Level, error};

use super::models::{
    GetTableResponse, ListCatalogsResponse, ListSchemasResponse, ListTableSummariesResponse, Table,
    TableType,
};
use super::{DataCatalogResult, UnityCatalog, UnityCatalogError};
use deltalake_core::{DeltaTableBuilder, Path};
use reqwest::Url;

/// In-memory list of catalogs populated by unity catalog
#[derive(Debug)]
pub struct UnityCatalogList {
    /// Collection of catalogs containing schemas and ultimately TableProviders
    pub catalogs: DashMap<String, Arc<dyn CatalogProvider>>,
}

impl UnityCatalogList {
    /// Create a new instance of [`UnityCatalogList`]
    pub async fn try_new(client: Arc<UnityCatalog>) -> DataCatalogResult<Self> {
        let catalogs = match client.list_catalogs().await? {
            ListCatalogsResponse::Success { catalogs, .. } => {
                let mut providers = Vec::new();
                for catalog in catalogs {
                    let provider =
                        UnityCatalogProvider::try_new(client.clone(), &catalog.name).await?;
                    providers.push((catalog.name, Arc::new(provider) as Arc<dyn CatalogProvider>));
                }
                providers
            }
            _ => vec![],
        };
        Ok(Self {
            catalogs: catalogs.into_iter().collect(),
        })
    }
}

impl CatalogProviderList for UnityCatalogList {
    fn register_catalog(
        &self,
        name: String,
        catalog: Arc<dyn CatalogProvider>,
    ) -> Option<Arc<dyn CatalogProvider>> {
        self.catalogs.insert(name, catalog)
    }

    fn catalog_names(&self) -> Vec<String> {
        self.catalogs.iter().map(|c| c.key().clone()).collect()
    }

    fn catalog(&self, name: &str) -> Option<Arc<dyn CatalogProvider>> {
        self.catalogs.get(name).map(|c| c.value().clone())
    }
}

/// A datafusion [`CatalogProvider`] backed by Databricks UnityCatalog
#[derive(Debug)]
pub struct UnityCatalogProvider {
    /// Parent catalog for schemas of interest.
    pub schemas: DashMap<String, Arc<dyn SchemaProvider>>,
}

impl UnityCatalogProvider {
    /// Create a new instance of [`UnityCatalogProvider`]
    pub async fn try_new(
        client: Arc<UnityCatalog>,
        catalog_name: impl Into<String>,
    ) -> DataCatalogResult<Self> {
        let catalog_name = catalog_name.into();
        let schemas = match client.list_schemas(&catalog_name).await? {
            ListSchemasResponse::Success { schemas } => {
                let mut providers = Vec::new();
                for schema in schemas {
                    let provider =
                        UnitySchemaProvider::try_new(client.clone(), &catalog_name, &schema.name)
                            .await?;
                    providers.push((schema.name, Arc::new(provider) as Arc<dyn SchemaProvider>));
                }
                providers
            }
            _ => vec![],
        };
        Ok(Self {
            schemas: schemas.into_iter().collect(),
        })
    }
}

impl CatalogProvider for UnityCatalogProvider {
    fn schema_names(&self) -> Vec<String> {
        self.schemas.iter().map(|c| c.key().clone()).collect()
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        self.schemas.get(name).map(|c| c.value().clone())
    }
}

/// The location the table's routes are keyed on.
///
/// Local paths are canonicalized, as delta-rs resolves them, so a location
/// behind a symlink (e.g. macOS `/tmp`) still matches its route.
fn canonical_location(url: &Url) -> Url {
    if url.scheme() != "file" {
        return url.clone();
    }
    url.to_file_path()
        .ok()
        .and_then(|path| std::fs::canonicalize(path).ok())
        .and_then(|path| Url::from_directory_path(path).ok())
        .unwrap_or_else(|| url.clone())
}

/// A datafusion [`SchemaProvider`] backed by Databricks UnityCatalog
#[derive(Debug)]
pub struct UnitySchemaProvider {
    client: Arc<UnityCatalog>,
    catalog_name: String,
    schema_name: String,

    /// Parent catalog for schemas of interest.
    table_names: Vec<String>,
    table_cache: DashMap<String, Arc<dyn TableProvider>>,
    store_factory: UnityStoreFactory,
}

impl UnitySchemaProvider {
    /// Create a new instance of [`UnitySchemaProvider`]
    pub async fn try_new(
        client: Arc<UnityCatalog>,
        catalog_name: impl Into<String>,
        schema_name: impl Into<String>,
    ) -> DataCatalogResult<Self> {
        let catalog_name = catalog_name.into();
        let schema_name = schema_name.into();
        let table_names = match client
            .list_table_summaries(&catalog_name, &schema_name)
            .await?
        {
            ListTableSummariesResponse::Success { tables, .. } => tables
                .into_iter()
                .filter(|t| {
                    t.table_type == TableType::Managed || t.table_type == TableType::External
                })
                .filter_map(|t| t.full_name.split('.').next_back().map(|n| n.into()))
                .collect(),
            ListTableSummariesResponse::Error(_) => vec![],
        };
        let store_factory = client.store_factory()?;
        Ok(Self {
            client,
            table_names,
            catalog_name,
            schema_name,
            table_cache: DashMap::new(),
            store_factory,
        })
    }
}

#[async_trait::async_trait]
impl SchemaProvider for UnitySchemaProvider {
    fn table_names(&self) -> Vec<String> {
        self.table_names.clone()
    }

    async fn table(
        &self,
        name: &str,
    ) -> datafusion::common::Result<Option<Arc<dyn TableProvider>>> {
        if self.table_cache.contains_key(name) {
            tracing::event!(Level::DEBUG, "Cache hit: {name}");
            return Ok(Some(self.table_cache.get(name).unwrap().clone()));
        }
        let maybe_table = self
            .client
            .get_table(&self.catalog_name, &self.schema_name, &name.to_string())
            .await
            .map_err(|err| DataFusionError::External(Box::new(err)))?;

        match maybe_table {
            GetTableResponse::Success(Table {
                storage_location: Some(_),
                ..
            }) => {
                let full_path = format!("{}.{}.{}", self.catalog_name, self.schema_name, name);
                let store = crate::table_store(&self.store_factory, &full_path, None)
                    .await
                    .map_err(|err| DataFusionError::External(Box::new(err)))?;

                // Every Unity Catalog table in a bucket shares one routing
                // store as its root, so whichever instance DataFusion registers
                // for the bucket still sends each table's requests to the store
                // holding that table's credentials (delta-io/delta-rs#4425).
                let location = canonical_location(store.url());
                let prefix = Path::from_url_path(location.path())
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                let routing = self.client.routing_store(&location);
                routing.register(prefix, store.root());
                let table = DeltaTableBuilder::from_url(location.clone())
                    .map_err(|e| DataFusionError::External(Box::new(e)))?
                    .with_storage_backend(routing, location)
                    .load()
                    .await?;
                let provider = table.table_provider().await?;
                self.table_cache
                    .insert(name.to_string(), Arc::clone(&provider));
                Ok(Some(provider))
            }
            GetTableResponse::Success(Table {
                storage_location: None,
                name,
                table_type,
                ..
            }) => Err(DataFusionError::External(Box::new(
                UnityCatalogError::NotATable(name, table_type),
            ))),
            GetTableResponse::Error(err) => {
                error!("failed to fetch table from unity catalog: {}", err.message);
                Err(DataFusionError::External(Box::new(err)))
            }
        }
    }

    fn table_exist(&self, name: &str) -> bool {
        self.table_names.contains(&name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::{Int64Array, RecordBatch};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::prelude::SessionContext;
    use deltalake_core::{DeltaTable, ensure_table_uri};
    use httpmock::prelude::*;

    use super::*;
    use crate::UnityCatalogBuilder;
    use crate::client::ClientOptions;

    fn catalog(server: &MockServer) -> Arc<UnityCatalog> {
        Arc::new(
            UnityCatalogBuilder::builder()
                .workspace_url(server.url(""))
                .bearer_token("pat")
                .client_options(ClientOptions::builder().allow_http(true).build())
                .build()
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn routing_store_is_shared_per_bucket() {
        let server = MockServer::start();
        let uc = catalog(&server);
        let url = |s: &str| Url::parse(s).unwrap();
        let t1 = uc.routing_store(&url("abfss://container@account.dfs.core.windows.net/t1"));
        let t2 = uc.routing_store(&url("abfss://container@account.dfs.core.windows.net/t2"));
        let other = uc.routing_store(&url("abfss://other@account.dfs.core.windows.net/t1"));
        assert!(
            Arc::ptr_eq(&t1, &t2),
            "tables in one container share a store"
        );
        assert!(!Arc::ptr_eq(&t1, &other), "containers do not");
    }

    async fn write_table(location: &str, ids: Vec<i64>) {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(ids))]).unwrap();
        DeltaTable::try_from_url(ensure_table_uri(location).unwrap())
            .await
            .unwrap()
            .write(vec![batch])
            .await
            .unwrap();
    }

    async fn mock_table(server: &MockServer, name: &str, location: &str) {
        let body = serde_json::json!({
            "name": name,
            "catalog_name": "main",
            "schema_name": "default",
            "table_type": "MANAGED",
            "data_source_format": "DELTA",
            "columns": [],
            "properties": {},
            "storage_location": location,
            "created_at": 0,
            "updated_at": 0,
            "table_id": format!("id-{name}"),
        });
        server
            .mock_async(|when, then| {
                when.method(GET)
                    .path(format!("/api/2.1/unity-catalog/tables/main.default.{name}"));
                then.json_body(body);
            })
            .await;
    }

    /// Two tables under one storage root, queried in one session: each reads
    /// its own data through the shared routing store (delta-io/delta-rs#4425).
    #[tokio::test(flavor = "multi_thread")]
    async fn tables_sharing_a_bucket_query_in_one_session() {
        let dir = tempfile::tempdir().unwrap();
        // Deliberately not canonicalized: on macOS the temp dir sits behind the
        // `/var` -> `/private/var` symlink, which the routing must tolerate.
        let root = Url::from_directory_path(dir.path()).unwrap();
        let t1 = root.join("tables/t1/").unwrap();
        let t2 = root.join("tables/t2/").unwrap();
        write_table(t1.as_str(), vec![1, 2]).await;
        write_table(t2.as_str(), vec![1, 2, 3]).await;

        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/api/2.1/unity-catalog/table-summaries");
                then.json_body(serde_json::json!({"tables": [
                    {"full_name": "main.default.t1", "table_type": "MANAGED"},
                    {"full_name": "main.default.t2", "table_type": "MANAGED"},
                ]}));
            })
            .await;
        mock_table(&server, "t1", t1.as_str()).await;
        mock_table(&server, "t2", t2.as_str()).await;

        let provider = UnitySchemaProvider::try_new(catalog(&server), "main", "default")
            .await
            .unwrap();
        let ctx = SessionContext::new();
        for name in ["t1", "t2"] {
            let table = provider.table(name).await.unwrap().unwrap();
            ctx.register_table(name, table).unwrap();
        }

        for (name, expected) in [("t1", 2), ("t2", 3), ("t1", 2)] {
            let batches = ctx
                .sql(&format!("SELECT COUNT(*) AS n FROM {name}"))
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            let count = batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0);
            assert_eq!(count, expected, "rows in {name}");
        }
    }
}
