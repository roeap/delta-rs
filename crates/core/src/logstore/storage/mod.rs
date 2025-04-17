//! Object storage backend abstraction layer for Delta Table transaction logs and data
use std::sync::{Arc, LazyLock};

use dashmap::DashMap;
use object_store::path::Path;
use object_store::{DynObjectStore, ObjectStore};
use url::Url;

use crate::{DeltaResult, DeltaTableError};
use deltalake_derive::DeltaConfig;

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

#[derive(Debug, Clone, Default, DeltaConfig)]
pub struct LimitConfig {
    #[delta(alias = "concurrency_limit", env = "OBJECT_STORE_CONCURRENCY_LIMIT")]
    pub max_concurrency: Option<usize>,
}

#[cfg(test)]
#[allow(drop_bounds, unused)]
mod tests {
    use std::collections::HashMap;
    use std::env;

    use rstest::*;

    use super::*;
    use crate::logstore::config::TryUpdateKey;

    #[fixture]
    pub fn with_env(#[default(vec![])] vars: Vec<(&str, &str)>) -> impl Drop {
        // Store the original values before modifying
        let original_values: HashMap<String, Option<String>> = vars
            .iter()
            .map(|(key, _)| (key.to_string(), std::env::var(key).ok()))
            .collect();

        // Set all the new environment variables
        for (key, value) in vars {
            std::env::set_var(key, value);
        }

        // Create a cleanup struct that will restore original values when dropped
        struct EnvCleanup(HashMap<String, Option<String>>);

        impl Drop for EnvCleanup {
            fn drop(&mut self) {
                for (key, maybe_value) in self.0.iter() {
                    match maybe_value {
                        Some(value) => env::set_var(key, value),
                        None => env::remove_var(key),
                    }
                }
            }
        }

        EnvCleanup(original_values)
    }

    #[rstest]
    fn test_api_with_env(
        #[with(vec![("API_KEY", "test_key"), ("API_URL", "http://test.example.com")])]
        with_env: impl Drop,
    ) {
        // Test code using these environment variables
        assert_eq!(env::var("API_KEY").unwrap(), "test_key");
        assert_eq!(env::var("API_URL").unwrap(), "http://test.example.com");

        drop(with_env);

        assert!(env::var("API_KEY").is_err());
        assert!(env::var("API_URL").is_err());
    }

    #[test]
    fn test_limit_config() {
        let mut config = LimitConfig::default();
        assert!(config.max_concurrency.is_none());

        config.try_update_key("concurrency_limit", "10").unwrap();
        assert_eq!(config.max_concurrency, Some(10));

        config.try_update_key("max_concurrency", "20").unwrap();
        assert_eq!(config.max_concurrency, Some(20));
    }

    #[rstest]
    fn test_limit_config_env(
        #[with(vec![("OBJECT_STORE_CONCURRENCY_LIMIT", "100")])] with_env: impl Drop,
    ) {
        let mut config = LimitConfig::default();
        assert!(config.max_concurrency.is_none());

        config.load_from_environment().unwrap();
        assert_eq!(config.max_concurrency, Some(100));
    }
}
