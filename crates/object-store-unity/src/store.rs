//! Building credential-scoped object stores for Unity Catalog securables.

use std::fmt;
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::prefix::PrefixStore;
use tokio::runtime::Handle;
use url::Url;

use crate::builder::UnityStoreFactoryBuilder;
use crate::client::UnityClient;
use crate::credential::{CredentialVendor, PathVendor, TableVendor, VolumeVendor};
use crate::error::{Error, Result};
use crate::models::{
    CredentialKind, Operation, PathOperation, TableOperation, TemporaryCredentials, VolumeOperation,
};
use crate::reference::UnityReference;

/// An object store scoped to the storage of one Unity Catalog securable.
///
/// Vended credentials are scoped to a sub-path of a bucket or container — the
/// securable's storage root. [`as_dyn`](Self::as_dyn) returns a store rooted
/// there; [`root`](Self::root) returns the underlying bucket-rooted store.
#[derive(Clone)]
pub struct ScopedStore {
    root: Arc<dyn ObjectStore>,
    url: Url,
    prefix: Path,
}

impl ScopedStore {
    /// A store whose paths are relative to the securable's storage root.
    pub fn as_dyn(&self) -> Arc<dyn ObjectStore> {
        if self.prefix.as_ref().is_empty() {
            self.root.clone()
        } else {
            Arc::new(PrefixStore::new(self.root.clone(), self.prefix.clone()))
        }
    }

    /// The bucket- (or container-) rooted store.
    ///
    /// The credential may not authorize anything outside [`prefix`](Self::prefix);
    /// callers using this store must stay beneath it.
    pub fn root(&self) -> Arc<dyn ObjectStore> {
        self.root.clone()
    }

    /// The storage URL of the securable root, e.g. `s3://bucket/tables/t1/`.
    pub fn url(&self) -> &Url {
        &self.url
    }

    /// The securable root's path within [`root`](Self::root).
    pub fn prefix(&self) -> &Path {
        &self.prefix
    }
}

impl fmt::Debug for ScopedStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScopedStore")
            .field("root", &self.root.to_string())
            .field("url", &self.url.as_str())
            .field("prefix", &self.prefix.as_ref())
            .finish()
    }
}

/// Creates [`ScopedStore`]s for Unity Catalog securables, backed by vended
/// credentials that refresh themselves.
///
/// Each store is bound to the securable and operation it was created for: a
/// refresh re-vends exactly that, so it can never widen the store's access.
///
/// ```no_run
/// # use object_store::ObjectStoreExt;
/// # use object_store_unity::{Operation, UnityStoreFactory};
/// # async fn run() -> object_store_unity::Result<()> {
/// let factory = UnityStoreFactory::builder()
///     .with_url("https://my-workspace.cloud.databricks.com/api/2.1/unity-catalog/")
///     .with_token("<personal access token>")
///     .build()?;
///
/// let volume = factory
///     .for_url("uc:///Volumes/main/default/landing/raw/", Operation::Read)
///     .await?;
/// let object = volume.as_dyn().get(&"file.parquet".into()).await?;
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct UnityStoreFactory {
    client: UnityClient,
    aws_region: Option<String>,
    io_runtime: Option<Handle>,
}

impl UnityStoreFactory {
    pub(crate) fn new(
        client: UnityClient,
        aws_region: Option<String>,
        io_runtime: Option<Handle>,
    ) -> Self {
        Self {
            client,
            aws_region,
            io_runtime,
        }
    }

    /// Create a [`UnityStoreFactoryBuilder`].
    pub fn builder() -> UnityStoreFactoryBuilder {
        UnityStoreFactoryBuilder::new()
    }

    /// The Unity Catalog client this factory vends through.
    pub fn client(&self) -> &UnityClient {
        &self.client
    }

    /// Create a store for any URL [`UnityReference`] understands.
    ///
    /// A volume URL with a sub-path yields a store rooted at that sub-path.
    pub async fn for_url(&self, url: &str, operation: Operation) -> Result<ScopedStore> {
        match UnityReference::parse(url)? {
            UnityReference::Table {
                catalog,
                schema,
                table,
            } => {
                self.for_table(&format!("{catalog}.{schema}.{table}"), operation.into())
                    .await
            }
            UnityReference::Volume {
                catalog,
                schema,
                volume,
                path,
            } => {
                let store = self
                    .for_volume(&format!("{catalog}.{schema}.{volume}"), operation.into())
                    .await?;
                Ok(extend_prefix(store, &path))
            }
            UnityReference::Path(url) => self.for_path(&url, operation.into()).await,
        }
    }

    /// Create a store for a table, by its `<catalog>.<schema>.<table>` name.
    pub async fn for_table(
        &self,
        full_name: &str,
        operation: TableOperation,
    ) -> Result<ScopedStore> {
        let table = self.client.get_table(full_name).await?;
        let location = parse_location(table.storage_location.as_deref())?;
        if let Some(url) = location.as_ref().filter(|url| url.scheme() == "file") {
            return local_store(url, operation == TableOperation::ReadWrite);
        }
        let vendor = TableVendor {
            client: self.client.clone(),
            table_id: table.table_id,
            operation,
        };
        self.store_for_vendor(Arc::new(vendor), location).await
    }

    /// Create a store for a volume, by its `<catalog>.<schema>.<volume>` name.
    pub async fn for_volume(
        &self,
        full_name: &str,
        operation: VolumeOperation,
    ) -> Result<ScopedStore> {
        let volume = self.client.get_volume(full_name).await?;
        let location = parse_location(volume.storage_location.as_deref())?;
        if let Some(url) = location.as_ref().filter(|url| url.scheme() == "file") {
            return local_store(url, operation == VolumeOperation::ReadWrite);
        }
        let vendor = VolumeVendor {
            client: self.client.clone(),
            volume_id: volume.volume_id,
            operation,
        };
        self.store_for_vendor(Arc::new(vendor), location).await
    }

    /// Create a store for a raw storage URL (`s3://`, `abfss://`, `gs://`, …),
    /// governed by an external location.
    ///
    /// `file://` URLs are served from the local filesystem without vending.
    pub async fn for_path(&self, url: &Url, operation: PathOperation) -> Result<ScopedStore> {
        self.path_store(url, operation, false).await
    }

    /// Like [`for_path`](Self::for_path), but only checks that the credential
    /// could be issued (`dry_run`): the vended credential is not usable for I/O.
    pub async fn dry_run_path(&self, url: &Url, operation: PathOperation) -> Result<ScopedStore> {
        self.path_store(url, operation, true).await
    }

    /// Create a store backed by any [`CredentialVendor`].
    ///
    /// The vendor is asked for a credential now, and again whenever the cached
    /// one nears expiry. The vended `url` must name the storage location.
    pub async fn for_vendor(&self, vendor: Arc<dyn CredentialVendor>) -> Result<ScopedStore> {
        self.store_for_vendor(vendor, None).await
    }

    async fn path_store(
        &self,
        url: &Url,
        operation: PathOperation,
        dry_run: bool,
    ) -> Result<ScopedStore> {
        if url.scheme() == "file" {
            return local_store(url, operation != PathOperation::Read);
        }
        let vendor = PathVendor {
            client: self.client.clone(),
            url: url.clone(),
            operation,
            dry_run,
        };
        self.store_for_vendor(Arc::new(vendor), Some(url.clone()))
            .await
    }

    /// Vend once and build the store, falling back to `location` when the
    /// response carries no `url`.
    async fn store_for_vendor(
        &self,
        vendor: Arc<dyn CredentialVendor>,
        location: Option<Url>,
    ) -> Result<ScopedStore> {
        let credentials = vendor.vend().await?;
        let url = match parse_location(credentials.url.as_deref())? {
            Some(url) => url,
            None => location.ok_or_else(|| {
                Error::InvalidResponse("vended credential names no storage location".to_string())
            })?,
        };
        let prefix = match parse_azurite(&url) {
            Some(azurite) => Path::from(azurite.prefix),
            None => Path::from_url_path(url.path())
                .map_err(|e| Error::InvalidUrl(format!("{url}: {e}")))?,
        };
        let root = self.build_root(vendor, &credentials, &url).await?;
        Ok(ScopedStore { root, url, prefix })
    }

    async fn build_root(
        &self,
        vendor: Arc<dyn CredentialVendor>,
        credentials: &TemporaryCredentials,
        url: &Url,
    ) -> Result<Arc<dyn ObjectStore>> {
        match credentials.kind().ok_or(Error::MissingCredential)? {
            CredentialKind::Azure => self.azure_store(vendor, credentials, url).await,
            CredentialKind::Aws => self.aws_store(vendor, credentials, url).await,
            CredentialKind::Gcp => self.gcp_store(vendor, credentials, url).await,
        }
    }

    #[cfg(feature = "azure")]
    async fn azure_store(
        &self,
        vendor: Arc<dyn CredentialVendor>,
        credentials: &TemporaryCredentials,
        url: &Url,
    ) -> Result<Arc<dyn ObjectStore>> {
        use object_store::azure::{AzureConfigKey, MicrosoftAzureBuilder};

        use crate::credential::{UnityCredentialProvider, azure_credential};

        let mut builder = match parse_azurite(url) {
            // The emulator's URLs are path-style, which `with_url` does not
            // understand, and in emulator mode the builder ignores credential
            // providers: configure account and container, and pass the SAS.
            Some(azurite) => {
                let sas = credentials
                    .azure_user_delegation_sas
                    .as_ref()
                    .ok_or_else(|| {
                        Error::CredentialMismatch(
                            "the Azurite emulator needs a SAS credential".into(),
                        )
                    })?;
                MicrosoftAzureBuilder::new()
                    .with_use_emulator(true)
                    .with_account(azurite.account)
                    .with_container_name(azurite.container)
                    .with_config(AzureConfigKey::SasKey, &sas.sas_token)
            }
            None => {
                let provider =
                    UnityCredentialProvider::new(vendor, credentials, azure_credential).await?;
                MicrosoftAzureBuilder::new()
                    .with_url(url.as_str())
                    .with_credentials(Arc::new(provider))
            }
        };
        if let Some(connector) = self.connector() {
            builder = builder.with_http_connector(connector);
        }
        Ok(Arc::new(builder.build()?))
    }

    #[cfg(not(feature = "azure"))]
    async fn azure_store(
        &self,
        _: Arc<dyn CredentialVendor>,
        _: &TemporaryCredentials,
        _: &Url,
    ) -> Result<Arc<dyn ObjectStore>> {
        Err(Error::FeatureDisabled {
            cloud: "Azure",
            feature: "azure",
        })
    }

    #[cfg(feature = "aws")]
    async fn aws_store(
        &self,
        vendor: Arc<dyn CredentialVendor>,
        credentials: &TemporaryCredentials,
        url: &Url,
    ) -> Result<Arc<dyn ObjectStore>> {
        use crate::credential::{UnityCredentialProvider, aws_credential};

        let access_point = credentials
            .aws_temp_credentials
            .as_ref()
            .and_then(|aws| aws.access_point.clone())
            .filter(|ap| !ap.is_empty());
        let provider = UnityCredentialProvider::new(vendor, credentials, aws_credential).await?;
        let mut builder = s3_builder(url, self.aws_region.clone(), access_point)?
            .with_credentials(Arc::new(provider));
        if let Some(connector) = self.connector() {
            builder = builder.with_http_connector(connector);
        }
        Ok(Arc::new(builder.build()?))
    }

    #[cfg(not(feature = "aws"))]
    async fn aws_store(
        &self,
        _: Arc<dyn CredentialVendor>,
        _: &TemporaryCredentials,
        _: &Url,
    ) -> Result<Arc<dyn ObjectStore>> {
        Err(Error::FeatureDisabled {
            cloud: "AWS",
            feature: "aws",
        })
    }

    #[cfg(feature = "gcp")]
    async fn gcp_store(
        &self,
        vendor: Arc<dyn CredentialVendor>,
        credentials: &TemporaryCredentials,
        url: &Url,
    ) -> Result<Arc<dyn ObjectStore>> {
        use object_store::gcp::GoogleCloudStorageBuilder;

        use crate::credential::{UnityCredentialProvider, gcp_credential};

        let provider = UnityCredentialProvider::new(vendor, credentials, gcp_credential).await?;
        let mut builder = GoogleCloudStorageBuilder::new()
            .with_url(url.as_str())
            .with_credentials(Arc::new(provider));
        if let Some(connector) = self.connector() {
            builder = builder.with_http_connector(connector);
        }
        Ok(Arc::new(builder.build()?))
    }

    #[cfg(not(feature = "gcp"))]
    async fn gcp_store(
        &self,
        _: Arc<dyn CredentialVendor>,
        _: &TemporaryCredentials,
        _: &Url,
    ) -> Result<Arc<dyn ObjectStore>> {
        Err(Error::FeatureDisabled {
            cloud: "GCP",
            feature: "gcp",
        })
    }

    /// A connector spawning object-store requests on the configured I/O runtime.
    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    fn connector(&self) -> Option<object_store::client::SpawnedReqwestConnector> {
        self.io_runtime
            .clone()
            .map(object_store::client::SpawnedReqwestConnector::new)
    }
}

impl fmt::Debug for UnityStoreFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnityStoreFactory")
            .field("client", &self.client)
            .field("aws_region", &self.aws_region)
            .field("io_runtime", &self.io_runtime.is_some())
            .finish()
    }
}

fn parse_location(location: Option<&str>) -> Result<Option<Url>> {
    location
        .filter(|l| !l.is_empty())
        .map(|l| Url::parse(l).map_err(|e| Error::InvalidUrl(format!("{l}: {e}"))))
        .transpose()
}

/// Start an S3 builder for a vended S3-compatible location.
///
/// Cloudflare R2 locations arrive as
/// `r2://<bucket>@<account>.r2.cloudflarestorage.com/<path>`, a scheme
/// object_store does not parse, so the bucket, the account endpoint, and R2's
/// `auto` region are set explicitly — an AWS region override must not clobber
/// the latter.
///
/// For S3 the region is `aws_region`, else `AWS_REGION`, else object_store's
/// default. A vended S3 access-point ARN is used as the bucket, so SigV4
/// signatures match what STS authorized.
#[cfg(feature = "aws")]
fn s3_builder(
    url: &Url,
    aws_region: Option<String>,
    access_point: Option<String>,
) -> Result<object_store::aws::AmazonS3Builder> {
    use object_store::aws::AmazonS3Builder;

    if url.scheme() == "r2" {
        let bucket = url.username();
        let host = url.host_str().unwrap_or_default();
        if bucket.is_empty() || !host.ends_with(".r2.cloudflarestorage.com") {
            return Err(Error::InvalidUrl(format!(
                "expected r2://<bucket>@<account>.r2.cloudflarestorage.com/..., got {url}"
            )));
        }
        return Ok(AmazonS3Builder::new()
            .with_bucket_name(bucket)
            .with_endpoint(format!("https://{host}"))
            .with_region("auto"));
    }

    let mut builder = AmazonS3Builder::new().with_url(url.as_str());
    if let Some(region) = aws_region.or_else(|| std::env::var("AWS_REGION").ok()) {
        builder = builder.with_region(region);
    }
    if let Some(access_point) = access_point {
        builder = builder.with_bucket_name(access_point);
    }
    Ok(builder)
}

/// A store for a `file://` location: local storage needs no vended credential.
///
/// Mirrors the bucket-root + prefix model of cloud stores: `root` is an unrooted
/// [`LocalFileSystem`] and `prefix` is the directory, so both
/// [`ScopedStore::as_dyn`] (relative paths) and [`ScopedStore::root`] (full
/// paths) resolve correctly. With `create`, a missing directory is created.
///
/// POSIX only: object_store paths cannot express Windows absolute paths.
fn local_store(url: &Url, create: bool) -> Result<ScopedStore> {
    if cfg!(windows) {
        return Err(Error::InvalidConfig(format!(
            "local (file://) storage is not supported on Windows: {url}"
        )));
    }
    let dir = url
        .to_file_path()
        .map_err(|_| Error::InvalidUrl(format!("not a local file path: {url}")))?;
    if create {
        std::fs::create_dir_all(&dir).map_err(|e| {
            Error::InvalidConfig(format!("failed to create local directory {dir:?}: {e}"))
        })?;
    }
    let prefix =
        Path::from_url_path(url.path()).map_err(|e| Error::InvalidUrl(format!("{url}: {e}")))?;
    Ok(ScopedStore {
        root: Arc::new(LocalFileSystem::new()),
        url: url.clone(),
        prefix,
    })
}

/// A store rooted `extra` below `store`'s securable root.
fn extend_prefix(store: ScopedStore, extra: &str) -> ScopedStore {
    let parts: Vec<&str> = extra.split('/').filter(|p| !p.is_empty()).collect();
    if parts.is_empty() {
        return store;
    }
    let mut url = store.url.clone();
    // A directory URL keeps its trailing slash.
    let directory = url.path().ends_with('/');
    if let Ok(mut segments) = url.path_segments_mut() {
        segments.pop_if_empty().extend(&parts);
        if directory {
            segments.push("");
        }
    }
    let prefix = store
        .prefix
        .parts()
        .chain(parts.iter().map(|p| object_store::path::PathPart::from(*p)))
        .collect();
    ScopedStore {
        root: store.root,
        url,
        prefix,
    }
}

/// Components of an Azurite (Azure Blob emulator) location.
#[cfg_attr(not(feature = "azure"), allow(dead_code))]
struct AzuriteLocation {
    account: String,
    container: String,
    prefix: String,
}

/// Detect an Azurite location: path-style
/// `http://127.0.0.1:10000/<account>/<container>/<prefix>` (or `localhost`),
/// or `azurite://<container>/<prefix>` with the well-known emulator account.
fn parse_azurite(url: &Url) -> Option<AzuriteLocation> {
    const EMULATOR_ACCOUNT: &str = "devstoreaccount1";

    if url.scheme() == "azurite" {
        return Some(AzuriteLocation {
            account: EMULATOR_ACCOUNT.to_string(),
            container: url.host_str().filter(|h| !h.is_empty())?.to_string(),
            prefix: url.path().trim_start_matches('/').to_string(),
        });
    }

    let is_localhost = matches!(url.host_str(), Some("localhost" | "127.0.0.1"));
    if url.scheme() == "http" && is_localhost && url.port() == Some(10000) {
        let mut segments = url.path_segments()?;
        let account = segments.next().filter(|s| !s.is_empty())?.to_string();
        let container = segments.next().filter(|s| !s.is_empty())?.to_string();
        return Some(AzuriteLocation {
            account,
            container,
            prefix: segments.collect::<Vec<_>>().join("/"),
        });
    }

    None
}

#[cfg(all(test, feature = "aws", feature = "azure", feature = "gcp"))]
mod tests {
    use futures::TryStreamExt;
    use httpmock::prelude::*;
    use object_store::{ObjectStoreExt, PutPayload};
    use serde_json::json;

    use super::*;
    use crate::credential::azure_credential;

    const API: &str = "/api/2.1/unity-catalog";

    fn factory(server: &MockServer) -> UnityStoreFactory {
        UnityStoreFactory::builder()
            .with_url(server.url(API))
            .with_token("pat")
            .with_aws_region("us-east-1")
            .build()
            .unwrap()
    }

    /// A factory whose Unity Catalog endpoint is unreachable: any vend fails.
    fn offline_factory() -> UnityStoreFactory {
        UnityStoreFactory::builder()
            .with_url("http://127.0.0.1:1/api/2.1/unity-catalog/")
            .with_allow_unauthenticated(true)
            .build()
            .unwrap()
    }

    fn aws_credentials(url: &str) -> serde_json::Value {
        json!({
            "expiration_time": chrono::Utc::now().timestamp_millis() + 3_600_000,
            "url": url,
            "aws_temp_credentials": {
                "access_key_id": "ak", "secret_access_key": "sk", "session_token": "st"
            }
        })
    }

    async fn mock_table(server: &MockServer, location: &str) {
        server
            .mock_async(|when, then| {
                when.method(GET)
                    .path(format!("{API}/tables/main.default.orders"));
                then.status(200)
                    .json_body(json!({"table_id": "t-1", "storage_location": location}));
            })
            .await;
    }

    #[tokio::test]
    async fn table_store_vends_read_and_scopes_to_the_table() {
        let server = MockServer::start_async().await;
        mock_table(&server, "s3://bucket/tables/orders").await;
        let vend = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path(format!("{API}/temporary-table-credentials"))
                    .header("authorization", "Bearer pat")
                    .json_body(json!({"table_id": "t-1", "operation": "READ"}));
                then.status(200)
                    .json_body(aws_credentials("s3://bucket/tables/orders/"));
            })
            .await;

        // Both the canonical and the legacy deltalake URL resolve to the table.
        for url in [
            "uc:///Tables/main/default/orders",
            "uc://main.default.orders",
        ] {
            let store = factory(&server)
                .for_url(url, Operation::Read)
                .await
                .unwrap();
            assert_eq!(store.url().as_str(), "s3://bucket/tables/orders/");
            assert_eq!(store.prefix(), &Path::from("tables/orders"));
        }
        vend.assert_calls_async(2).await;
    }

    #[tokio::test]
    async fn volume_url_with_sub_path_extends_the_prefix() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(GET)
                    .path(format!("{API}/volumes/main.default.landing"));
                then.status(200).json_body(
                    json!({"volume_id": "v-1", "storage_location": "s3://bucket/volumes/landing"}),
                );
            })
            .await;
        let vend = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path(format!("{API}/temporary-volume-credentials"))
                    .json_body(json!({"volume_id": "v-1", "operation": "WRITE_VOLUME"}));
                then.status(200)
                    .json_body(aws_credentials("s3://bucket/volumes/landing/"));
            })
            .await;

        let store = factory(&server)
            .for_url(
                "uc:///Volumes/main/default/landing/raw/2024/",
                Operation::ReadWrite,
            )
            .await
            .unwrap();
        assert_eq!(store.prefix(), &Path::from("volumes/landing/raw/2024"));
        assert_eq!(
            store.url().as_str(),
            "s3://bucket/volumes/landing/raw/2024/"
        );
        vend.assert_async().await;
    }

    #[tokio::test]
    async fn table_with_local_storage_skips_vending() {
        let server = MockServer::start_async().await;
        let dir = tempfile::tempdir().unwrap();
        let location = Url::from_directory_path(dir.path()).unwrap();
        mock_table(&server, location.as_str()).await;
        let vend = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path(format!("{API}/temporary-table-credentials"));
                then.status(500);
            })
            .await;

        let store = factory(&server)
            .for_table("main.default.orders", TableOperation::Read)
            .await
            .unwrap();
        assert_eq!(store.url(), &location);
        vend.assert_calls_async(0).await;
    }

    #[cfg(not(windows))]
    #[tokio::test]
    async fn file_paths_round_trip_without_vending() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("fresh");
        let url = Url::from_directory_path(&root).unwrap();

        let store = offline_factory()
            .for_path(&url, PathOperation::ReadWrite)
            .await
            .unwrap();
        assert!(root.exists(), "read-write creates the directory");

        let path = Path::from("hello.txt");
        store
            .as_dyn()
            .put(&path, PutPayload::from_static(b"world"))
            .await
            .unwrap();
        let listing: Vec<_> = store.as_dyn().list(None).try_collect().await.unwrap();
        assert_eq!(listing.len(), 1);
        assert_eq!(listing[0].location, path);
        // The root store resolves the full path beneath the prefix.
        let full = Path::from(format!("{}/hello.txt", store.prefix()));
        let bytes = store
            .root()
            .get(&full)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(&bytes[..], b"world");
    }

    #[tokio::test]
    async fn azurite_location_is_scoped_within_the_container() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST)
                    .path(format!("{API}/temporary-path-credentials"));
                then.status(200).json_body(json!({
                    "url": "http://127.0.0.1:10000/devstoreaccount1/container/tbl/data",
                    "azure_user_delegation_sas": {"sas_token": "sv=2021-08-06&sp=rl&sig=AAAA"}
                }));
            })
            .await;

        let url = Url::parse("abfss://container@devstoreaccount1/tbl/data").unwrap();
        let store = factory(&server)
            .for_path(&url, PathOperation::Read)
            .await
            .unwrap();
        assert_eq!(store.prefix(), &Path::from("tbl/data"));
    }

    #[tokio::test]
    async fn vended_credential_without_any_cloud_is_an_error() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST)
                    .path(format!("{API}/temporary-path-credentials"));
                then.status(200).json_body(json!({"url": "s3://bucket/p/"}));
            })
            .await;

        let url = Url::parse("s3://bucket/p/").unwrap();
        let err = factory(&server)
            .for_path(&url, PathOperation::Read)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::MissingCredential), "unexpected: {err}");
    }

    /// The vended (URL-encoded) SAS signature reaches Azure encoded exactly once.
    #[tokio::test]
    async fn azure_sas_signature_is_not_double_encoded() {
        use object_store::azure::MicrosoftAzureBuilder;

        let server = MockServer::start_async().await;
        let get = server
            .mock_async(|when, then| {
                // `query_param` matches the decoded value: a double-encoded
                // `%252B` would decode to `%2B` and miss.
                when.method(GET).query_param("sig", "ab+cd/ef=");
                then.status(404);
            })
            .await;

        let credentials = TemporaryCredentials {
            azure_user_delegation_sas: Some(crate::models::AzureUserDelegationSas {
                sas_token: "sv=2021-08-06&sp=rl&sig=ab%2Bcd%2Fef%3D".to_string(),
            }),
            ..Default::default()
        };
        let store = MicrosoftAzureBuilder::new()
            .with_account("account")
            .with_container_name("container")
            .with_endpoint(server.base_url())
            .with_allow_http(true)
            .with_retry(object_store::RetryConfig {
                max_retries: 0,
                ..Default::default()
            })
            .with_credentials(Arc::new(object_store::StaticCredentialProvider::new(
                azure_credential(&credentials).unwrap(),
            )))
            .build()
            .unwrap();

        let _ = store.get(&Path::from("prefix/file")).await;
        get.assert_async().await;
    }

    /// OAuth M2M options authenticate the vend: the token comes from the
    /// workspace host derived from the Unity Catalog URL.
    #[tokio::test]
    async fn oauth_m2m_authenticates_vending() {
        let server = MockServer::start_async().await;
        let token = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/oidc/v1/token")
                    .form_urlencoded_tuple("grant_type", "client_credentials")
                    .form_urlencoded_tuple("client_id", "sp-id");
                then.status(200).json_body(json!({
                    "access_token": "m2m-token", "token_type": "Bearer", "expires_in": 3600
                }));
            })
            .await;
        let vend = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path(format!("{API}/temporary-path-credentials"))
                    .header("authorization", "Bearer m2m-token");
                then.status(200)
                    .json_body(aws_credentials("s3://bucket/p/"));
            })
            .await;

        let factory = UnityStoreFactory::builder()
            .with_url(server.url(API))
            .with_options([
                ("databricks_client_id", "sp-id"),
                ("databricks_client_secret", "sp-secret"),
            ])
            .build()
            .unwrap();
        let url = Url::parse("s3://bucket/p/").unwrap();
        factory.for_path(&url, PathOperation::Read).await.unwrap();

        token.assert_async().await;
        vend.assert_async().await;
    }

    async fn mock_vend<'a>(server: &'a MockServer, bearer: &str) -> httpmock::Mock<'a> {
        let bearer = format!("Bearer {bearer}");
        server
            .mock_async(|when, then| {
                when.method(POST)
                    .path(format!("{API}/temporary-path-credentials"))
                    .header("authorization", bearer);
                then.status(200)
                    .json_body(aws_credentials("s3://bucket/p/"));
            })
            .await
    }

    async fn vend_through(factory: UnityStoreFactory) {
        let url = Url::parse("s3://bucket/p/").unwrap();
        factory.for_path(&url, PathOperation::Read).await.unwrap();
    }

    /// OSS Unity Catalog: a token file (e.g. a projected service-account
    /// token) is exchanged at the server's token endpoint, and the issued
    /// token signs the vend.
    #[tokio::test]
    async fn token_file_is_exchanged_for_a_unity_catalog_token() {
        let server = MockServer::start_async().await;
        let dir = tempfile::tempdir().unwrap();
        let token_file = dir.path().join("token");
        std::fs::write(&token_file, "idp-id-token").unwrap();
        let exchange = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/api/1.0/unity-control/auth/tokens")
                    .form_urlencoded_tuple(
                        "grant_type",
                        "urn:ietf:params:oauth:grant-type:token-exchange",
                    )
                    .form_urlencoded_tuple("subject_token", "idp-id-token")
                    .form_urlencoded_tuple(
                        "subject_token_type",
                        "urn:ietf:params:oauth:token-type:id_token",
                    );
                then.json_body(json!({"access_token": "uc-token", "expires_in": 3600}));
            })
            .await;
        let vend = mock_vend(&server, "uc-token").await;

        let factory = UnityStoreFactory::builder()
            .with_url(server.url(API))
            .with_aws_region("us-east-1")
            .with_options([
                ("unity_token_file", token_file.to_string_lossy().as_ref()),
                ("unity_token_exchange", "true"),
            ])
            .build()
            .unwrap();
        vend_through(factory).await;
        exchange.assert_async().await;
        vend.assert_async().await;
    }

    /// Client credentials against the deployment's own identity provider.
    #[tokio::test]
    async fn client_credentials_against_any_identity_provider() {
        let server = MockServer::start_async().await;
        let token = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/idp/oauth2/token")
                    .form_urlencoded_tuple("client_id", "app")
                    .form_urlencoded_tuple("scope", "catalog");
                then.json_body(json!({"access_token": "idp-token", "expires_in": 3600}));
            })
            .await;
        let vend = mock_vend(&server, "idp-token").await;

        let factory = UnityStoreFactory::builder()
            .with_url(server.url(API))
            .with_aws_region("us-east-1")
            .with_options([
                ("unity_client_id", "app"),
                ("unity_client_secret", "secret"),
                (
                    "unity_oauth_token_url",
                    server.url("/idp/oauth2/token").as_str(),
                ),
                ("unity_oauth_scope", "catalog"),
            ])
            .build()
            .unwrap();
        vend_through(factory).await;
        token.assert_async().await;
        vend.assert_async().await;
    }

    /// An Azure service principal for the deployment's own Entra app.
    #[tokio::test]
    async fn azure_service_principal_for_a_custom_scope() {
        let server = MockServer::start_async().await;
        let aad = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/tenant-1/oauth2/v2.0/token")
                    .form_urlencoded_tuple("scope", "api://my-catalog/.default");
                then.json_body(json!({"access_token": "aad-token", "expires_in": 3600}));
            })
            .await;
        let vend = mock_vend(&server, "aad-token").await;

        let factory = UnityStoreFactory::builder()
            .with_url(server.url(API))
            .with_aws_region("us-east-1")
            .with_options([
                ("unity_client_id", "app"),
                ("unity_client_secret", "secret"),
                ("unity_azure_tenant_id", "tenant-1"),
                ("unity_azure_authority_host", server.base_url().as_str()),
                ("unity_oauth_scope", "api://my-catalog/.default"),
            ])
            .build()
            .unwrap();
        vend_through(factory).await;
        aad.assert_async().await;
        vend.assert_async().await;
    }

    #[test]
    fn s3_builder_maps_r2_locations() {
        use object_store::aws::AmazonS3ConfigKey;

        let url = Url::parse("r2://my-bucket@acct.r2.cloudflarestorage.com/tables/t1/").unwrap();
        let builder = s3_builder(&url, Some("eu-west-1".to_string()), None).unwrap();
        let get = |key| builder.get_config_value(&key);
        assert_eq!(
            get(AmazonS3ConfigKey::Endpoint).as_deref(),
            Some("https://acct.r2.cloudflarestorage.com")
        );
        assert_eq!(get(AmazonS3ConfigKey::Bucket).as_deref(), Some("my-bucket"));
        assert_eq!(get(AmazonS3ConfigKey::Region).as_deref(), Some("auto"));

        let no_bucket = Url::parse("r2://acct.r2.cloudflarestorage.com/tables/").unwrap();
        assert!(s3_builder(&no_bucket, None, None).is_err());
    }

    #[test]
    fn s3_builder_applies_region_and_access_point() {
        use object_store::aws::AmazonS3ConfigKey;

        let url = Url::parse("s3://bucket/prefix/").unwrap();
        let access_point = "arn:aws:s3:eu-north-1:123:accesspoint/ap";
        let builder = s3_builder(
            &url,
            Some("eu-north-1".to_string()),
            Some(access_point.to_string()),
        )
        .unwrap();
        assert_eq!(
            builder
                .get_config_value(&AmazonS3ConfigKey::Region)
                .as_deref(),
            Some("eu-north-1")
        );
        assert_eq!(
            builder
                .get_config_value(&AmazonS3ConfigKey::Bucket)
                .as_deref(),
            Some(access_point)
        );
    }

    #[test]
    fn parse_azurite_forms() {
        let url = Url::parse("http://localhost:10000/acct/cont/a/b").unwrap();
        let loc = parse_azurite(&url).unwrap();
        assert_eq!(
            (loc.account.as_str(), loc.container.as_str()),
            ("acct", "cont")
        );
        assert_eq!(loc.prefix, "a/b");

        let loc = parse_azurite(&Url::parse("azurite://cont/a").unwrap()).unwrap();
        assert_eq!(loc.account, "devstoreaccount1");

        for url in [
            "https://acct.blob.core.windows.net/cont/a",
            "http://127.0.0.1:9000/acct/cont/a",
        ] {
            assert!(parse_azurite(&Url::parse(url).unwrap()).is_none(), "{url}");
        }
    }
}
