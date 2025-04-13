//! Object storage backend abstraction layer for Delta Table transaction logs and data
use std::sync::{Arc, LazyLock, OnceLock};

use dashmap::DashMap;
use object_store::limit::LimitStore;
use object_store::local::LocalFileSystem;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::prefix::PrefixStore;
use object_store::{DynObjectStore, ObjectStore};
use url::Url;

use super::{config, StorageConfig};
use crate::{DeltaResult, DeltaTableError};

pub use retry_ext::ObjectStoreRetryExt;
pub use runtime::{DeltaIOStorageBackend, IORuntime};

#[cfg(feature = "delta-cache")]
pub(super) mod cache;
pub(super) mod retry_ext;
pub(super) mod runtime;
pub(super) mod utils;

static DELTA_LOG_PATH: LazyLock<Path> = LazyLock::new(|| Path::from("_delta_log"));

/// Sharable reference to [`ObjectStore`]
pub type ObjectStoreRef = Arc<DynObjectStore>;

/// Factory trait for creating [ObjectStoreRef] instances at runtime
pub trait ObjectStoreFactory: Send + Sync {
    #[allow(missing_docs)]
    fn parse_url_opts(
        &self,
        url: &Url,
        options: &StorageConfig,
    ) -> DeltaResult<(ObjectStoreRef, Path)>;
}

#[derive(Clone, Debug, Default)]
pub(crate) struct DefaultObjectStoreFactory {}

impl ObjectStoreFactory for DefaultObjectStoreFactory {
    fn parse_url_opts(
        &self,
        url: &Url,
        options: &StorageConfig,
    ) -> DeltaResult<(ObjectStoreRef, Path)> {
        let (store, path) = match url.scheme() {
            "memory" => {
                let path = Path::from_url_path(url.path())?;
                let store = Box::new(PrefixStore::new(InMemory::new(), path.clone()))
                    as Box<dyn ObjectStore>;
                (store, path)
            }
            "file" => {
                let store = Box::new(LocalFileSystem::new_with_prefix(
                    url.to_file_path().unwrap(),
                )?) as Box<dyn ObjectStore>;
                (store, Path::from("/"))
            }
            _ => return Err(DeltaTableError::InvalidTableLocation(url.clone().into())),
        };

        if let Some(limit) = &options.limit {
            Ok((limit_store_handler(store, limit), path))
        } else {
            Ok((Arc::new(store), path))
        }
    }
}

pub trait ObjectStoreRegistry: Send + Sync + std::fmt::Debug + 'static + Clone {
    /// If a store with the same key existed before, it is replaced and returned
    fn register_store(
        &self,
        url: &Url,
        store: Arc<dyn ObjectStore>,
    ) -> Option<Arc<dyn ObjectStore>>;

    /// Get a suitable store for the provided URL. For example:
    /// If no [`ObjectStore`] found for the `url`, ad-hoc discovery may be executed depending on
    /// the `url` and [`ObjectStoreRegistry`] implementation. An [`ObjectStore`] may be lazily
    /// created and registered.
    fn get_store(&self, url: &Url) -> DeltaResult<Arc<dyn ObjectStore>>;

    fn all_stores(&self) -> &DashMap<String, Arc<dyn ObjectStore>>;
}

/// The default [`ObjectStoreRegistry`]
#[derive(Clone)]
pub struct DefaultObjectStoreRegistry {
    /// A map from scheme to object store that serve list / read operations for the store
    object_stores: DashMap<String, Arc<dyn ObjectStore>>,
}

impl Default for DefaultObjectStoreRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultObjectStoreRegistry {
    pub fn new() -> Self {
        let object_stores: DashMap<String, Arc<dyn ObjectStore>> = DashMap::new();
        Self { object_stores }
    }
}

impl std::fmt::Debug for DefaultObjectStoreRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("DefaultObjectStoreRegistry")
            .field(
                "schemes",
                &self
                    .object_stores
                    .iter()
                    .map(|o| o.key().clone())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl ObjectStoreRegistry for DefaultObjectStoreRegistry {
    fn register_store(
        &self,
        url: &Url,
        store: Arc<dyn ObjectStore>,
    ) -> Option<Arc<dyn ObjectStore>> {
        self.object_stores.insert(url.to_string(), store)
    }

    fn get_store(&self, url: &Url) -> DeltaResult<Arc<dyn ObjectStore>> {
        self.object_stores
            .get(&url.to_string())
            .map(|o| Arc::clone(o.value()))
            .ok_or_else(|| {
                DeltaTableError::generic(format!(
                    "No suitable object store found for {url}. See `RuntimeEnv::register_object_store`"
                ))
            })
    }

    fn all_stores(&self) -> &DashMap<String, Arc<dyn ObjectStore>> {
        &self.object_stores
    }
}

/// TODO
pub type FactoryRegistry = Arc<DashMap<Url, Arc<dyn ObjectStoreFactory>>>;

/// TODO
pub fn factories() -> FactoryRegistry {
    static REGISTRY: OnceLock<FactoryRegistry> = OnceLock::new();
    REGISTRY
        .get_or_init(|| {
            let registry = FactoryRegistry::default();
            registry.insert(
                Url::parse("memory://").unwrap(),
                Arc::new(DefaultObjectStoreFactory::default()),
            );
            registry.insert(
                Url::parse("file://").unwrap(),
                Arc::new(DefaultObjectStoreFactory::default()),
            );
            registry
        })
        .clone()
}

/// Simpler access pattern for the [FactoryRegistry] to get a single store
pub fn store_for<K, V, I>(url: &Url, options: I) -> DeltaResult<ObjectStoreRef>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str> + Into<String>,
    V: AsRef<str> + Into<String>,
{
    let scheme = Url::parse(&format!("{}://", url.scheme())).unwrap();
    let storage_config = StorageConfig::parse_options(options)?;
    if let Some(factory) = factories().get(&scheme) {
        let (store, _prefix) = factory.parse_url_opts(url, &storage_config)?;
        Ok(store)
    } else {
        Err(DeltaTableError::InvalidTableLocation(url.clone().into()))
    }
}

/// Simple function to wrap the given [ObjectStore] in a [PrefixStore] if necessary
///
/// This simplifies the use of the storage since it ensures that list/get/etc operations
/// start from the prefix in the object storage rather than from the root configured URI of the
/// [ObjectStore]
pub fn url_prefix_handler<T: ObjectStore>(store: T, prefix: Path) -> ObjectStoreRef {
    if prefix != Path::from("/") {
        Arc::new(PrefixStore::new(store, prefix))
    } else {
        Arc::new(store)
    }
}

#[derive(Debug, Clone, Default)]
pub struct LimitConfig {
    pub max_concurrency: Option<usize>,
}

impl config::TryUpdateKey for LimitConfig {
    fn try_update_key(&mut self, key: &str, v: &str) -> DeltaResult<Option<()>> {
        match key {
            // The number of concurrent connections the underlying object store can create
            // Reference [LimitStore](https://docs.rs/object_store/latest/object_store/limit/struct.LimitStore.html)
            // for more information
            "OBJECT_STORE_CONCURRENCY_LIMIT" | "concurrency_limit" => {
                self.max_concurrency = Some(config::parse_usize(v)?);
            }
            _ => return Ok(None),
        }
        Ok(Some(()))
    }
}

/// Wrap the given [ObjectStore] in a [LimitStore] if configured
///
/// Limits the number of concurrent connections the underlying object store
/// Reference [LimitStore](https://docs.rs/object_store/latest/object_store/limit/struct.LimitStore.html) for more information
pub fn limit_store_handler<T: ObjectStore>(store: T, options: &LimitConfig) -> ObjectStoreRef {
    if let Some(limit) = options.max_concurrency {
        Arc::new(LimitStore::new(store, limit))
    } else {
        Arc::new(store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_url_prefix_handler() {
        let store = InMemory::new();
        let path = Path::parse("/databases/foo/bar").expect("Failed to parse path");

        let prefixed = url_prefix_handler(store, path.clone());

        assert_eq!(
            String::from("PrefixObjectStore(databases/foo/bar)"),
            format!("{prefixed}")
        );
    }

    #[test]
    fn test_limit_store_handler() {
        let store = InMemory::new();

        let options = StorageConfig::parse_options(HashMap::<&str, &str>::from_iter(vec![(
            "OBJECT_STORE_CONCURRENCY_LIMIT",
            "500",
        )]))
        .unwrap();

        let limited = limit_store_handler(store, options.limit.as_ref().unwrap());

        assert_eq!(
            String::from("LimitStore(500, InMemory)"),
            format!("{limited}")
        );
    }
}
