//! [`PrimedStore`]: the composite object store that satisfies the wasm engine's
//! "ready futures only" contract.
//!
//! Kernel handler traits are synchronous, and on wasm the [`InlineExecutor`] backing them
//! can only complete futures that are already ready. `PrimedStore` makes that true for the
//! `_delta_log` metadata path: an async [`PrimedStore::prime`] call — driven by the JS event
//! loop, so free to fetch — copies the log tail into an [`InMemory`] cache *before* any sync
//! kernel call. Table-data reads pass through to the inner store and are driven end-to-end
//! by DataFusion's own async execution, which never blocks.
//!
//! A log read that misses the cache falls through to the inner store; on wasm that surfaces
//! as the inline executor's "would block — not primed" error (never a hang), telling the
//! caller to re-prime.
//!
//! [`InlineExecutor`]: deltalake_core::kernel::InlineExecutor

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    ObjectStoreExt as _, PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use url::Url;

use deltalake_core::{DeltaResult, DeltaTableError};

/// Name of the checkpoint hint file inside `_delta_log/`.
const LAST_CHECKPOINT: &str = "_last_checkpoint";

/// How many log files [`PrimedStore::prime`] fetches concurrently.
const PRIME_CONCURRENCY: usize = 8;

/// Where [`PrimedStore::prime`] learns the set of `_delta_log` files to cache.
#[derive(Debug, Clone)]
pub enum LogSource {
    /// List the log via the inner store (requires a listing-capable store).
    List,
    /// Host-supplied file manifest (e.g. a catalog's commit-tail response).
    ///
    /// Locations are interpreted relative to the table root unless they already start with
    /// the table's path. The manifest must include checkpoint parts and v2 sidecars
    /// (`_delta_log/_sidecars/…`) explicitly; `_last_checkpoint` itself is always fetched
    /// and need not be listed.
    Manifest(Vec<ObjectMeta>),
}

/// Guardrails on how much data a single [`PrimedStore::prime`] call may cache.
#[derive(Debug, Clone, Copy)]
pub struct PrimeLimits {
    /// Maximum number of log files fetched per prime.
    pub max_files: usize,
    /// Maximum total bytes cached per prime.
    pub max_bytes: u64,
}

impl Default for PrimeLimits {
    fn default() -> Self {
        Self {
            max_files: 512,
            max_bytes: 256 * 1024 * 1024,
        }
    }
}

/// Telemetry from a completed [`PrimedStore::prime`] call.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrimeReport {
    /// Number of log files cached (including `_last_checkpoint` when present).
    pub files: usize,
    /// Total bytes cached.
    pub bytes: u64,
    /// Checkpoint version from `_last_checkpoint`, if the hint exists.
    pub checkpoint_version: Option<u64>,
}

/// Minimal projection of `_last_checkpoint` — only the fields priming needs.
#[derive(serde::Deserialize)]
struct LastCheckpointHint {
    version: u64,
}

/// Composite read-only [`ObjectStore`]: an [`InMemory`] cache for everything under
/// `_delta_log/`, pass-through to the inner store for table data.
#[derive(Debug)]
pub struct PrimedStore {
    inner: Arc<dyn ObjectStore>,
    cache: InMemory,
    log_prefix: Path,
    table_prefix: Path,
    limits: PrimeLimits,
    /// Set when priming observed that `_last_checkpoint` does not exist, so the sync
    /// kernel read of it gets an immediate `NotFound` instead of falling through to a
    /// would-block fetch.
    last_checkpoint_absent: AtomicBool,
}

impl std::fmt::Display for PrimedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PrimedStore({})", self.table_prefix)
    }
}

impl PrimedStore {
    /// Create a store for the table at `table_url`, backed by `inner`.
    ///
    /// `inner` must resolve the same paths the table URL implies: the store registered (or
    /// registrable) for `table_url`'s scheme/authority, addressed by absolute paths.
    pub fn try_new(inner: Arc<dyn ObjectStore>, table_url: &Url) -> DeltaResult<Self> {
        let table_prefix = Path::from_url_path(table_url.path())
            .map_err(|err| DeltaTableError::generic(err.to_string()))?;
        let log_prefix = table_prefix.clone().join("_delta_log");
        Ok(Self {
            inner,
            cache: InMemory::new(),
            log_prefix,
            table_prefix,
            limits: PrimeLimits::default(),
            last_checkpoint_absent: AtomicBool::new(false),
        })
    }

    /// Replace the priming guardrails.
    pub fn with_limits(mut self, limits: PrimeLimits) -> Self {
        self.limits = limits;
        self
    }

    fn is_log_path(&self, location: &Path) -> bool {
        location.prefix_matches(&self.log_prefix)
    }

    fn last_checkpoint_path(&self) -> Path {
        self.log_prefix.clone().join(LAST_CHECKPOINT)
    }

    /// Prefetch the `_delta_log` tail into the cache.
    ///
    /// Async and driven by the host event loop, so free to fetch. After it returns, every
    /// log read the sync kernel engine issues is served from memory. May be called again
    /// (e.g. after a "not primed" error when the table changed); the cache is additive —
    /// log files are immutable — and `_last_checkpoint` is refreshed.
    pub async fn prime(&self, source: LogSource) -> DeltaResult<PrimeReport> {
        let mut report = PrimeReport::default();

        // 1. `_last_checkpoint`: always fetched; its absence is remembered so the kernel's
        //    sync read gets an immediate NotFound rather than a would-block fetch.
        let last_checkpoint = self.last_checkpoint_path();
        let checkpoint_version = match self.inner.get(&last_checkpoint).await {
            Ok(result) => {
                let bytes = result.bytes().await.map_err(DeltaTableError::from)?;
                let hint: Option<LastCheckpointHint> = serde_json::from_slice(&bytes).ok();
                report.files += 1;
                report.bytes += bytes.len() as u64;
                self.cache
                    .put(&last_checkpoint, bytes.into())
                    .await
                    .map_err(DeltaTableError::from)?;
                self.last_checkpoint_absent.store(false, Ordering::Relaxed);
                hint.map(|h| h.version)
            }
            Err(object_store::Error::NotFound { .. }) => {
                self.last_checkpoint_absent.store(true, Ordering::Relaxed);
                None
            }
            Err(err) => return Err(err.into()),
        };
        report.checkpoint_version = checkpoint_version;

        // 2. Determine the file set. Note `_delta_log/_sidecars/…` sorts *after* numeric
        //    commit names (`'_' > '9'`), so the offset listing captures v2 sidecars too.
        let mut files: Vec<ObjectMeta> = match source {
            LogSource::List => {
                let listing = match checkpoint_version {
                    Some(version) => {
                        let offset = self.log_prefix.clone().join(format!("{version:020}"));
                        self.inner.list_with_offset(Some(&self.log_prefix), &offset)
                    }
                    None => self.inner.list(Some(&self.log_prefix)),
                };
                listing.try_collect().await.map_err(DeltaTableError::from)?
            }
            LogSource::Manifest(entries) => entries
                .into_iter()
                .map(|mut meta| {
                    if !meta.location.prefix_matches(&self.table_prefix) {
                        meta.location =
                            Path::from_iter(self.table_prefix.parts().chain(meta.location.parts()));
                    }
                    meta
                })
                .collect(),
        };
        files.retain(|meta| meta.location != last_checkpoint);

        // 3. Guardrails, checked against listed sizes before fetching a single byte.
        let total: u64 = files.iter().map(|meta| meta.size).sum();
        if report.files + files.len() > self.limits.max_files {
            return Err(DeltaTableError::generic(format!(
                "priming would cache {} log files, exceeding the configured cap of {} \
                 (PrimeLimits::max_files)",
                report.files + files.len(),
                self.limits.max_files
            )));
        }
        if report.bytes + total > self.limits.max_bytes {
            return Err(DeltaTableError::generic(format!(
                "priming would cache {} bytes of log data, exceeding the configured cap of {} \
                 (PrimeLimits::max_bytes)",
                report.bytes + total,
                self.limits.max_bytes
            )));
        }

        // 4. Fetch concurrently into the cache, re-checking the byte cap against actual sizes.
        let inner = &self.inner;
        let mut fetched = futures::stream::iter(files)
            .map(|meta| async move {
                let bytes = inner.get(&meta.location).await?.bytes().await?;
                Ok::<_, object_store::Error>((meta.location, bytes))
            })
            .buffered(PRIME_CONCURRENCY);
        while let Some((location, bytes)) = fetched.try_next().await? {
            report.files += 1;
            report.bytes += bytes.len() as u64;
            if report.bytes > self.limits.max_bytes {
                return Err(DeltaTableError::generic(format!(
                    "priming exceeded the configured cap of {} bytes at {location} \
                     (PrimeLimits::max_bytes)",
                    self.limits.max_bytes
                )));
            }
            self.cache
                .put(&location, bytes.into())
                .await
                .map_err(DeltaTableError::from)?;
        }

        Ok(report)
    }

    fn read_only_error(&self, op: &'static str) -> object_store::Error {
        object_store::Error::NotSupported {
            source: format!("PrimedStore is read-only: {op} is not supported").into(),
        }
    }
}

#[async_trait]
impl ObjectStore for PrimedStore {
    async fn put_opts(
        &self,
        _location: &Path,
        _payload: PutPayload,
        _opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        Err(self.read_only_error("put"))
    }

    async fn put_multipart_opts(
        &self,
        _location: &Path,
        _opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        Err(self.read_only_error("put_multipart"))
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        if !self.is_log_path(location) {
            return self.inner.get_opts(location, options).await;
        }
        if location == &self.last_checkpoint_path()
            && self.last_checkpoint_absent.load(Ordering::Relaxed)
        {
            return Err(object_store::Error::NotFound {
                path: location.to_string(),
                source: "not present when the delta log was primed".into(),
            });
        }
        match self.cache.get_opts(location, options.clone()).await {
            // Cache miss: fall through to the inner store. On wasm this is what turns
            // an unprimed read into the inline executor's diagnosable error.
            Err(object_store::Error::NotFound { .. }) => {
                self.inner.get_opts(location, options).await
            }
            other => other,
        }
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        locations
            .map(|_| {
                Err(object_store::Error::NotSupported {
                    source: "PrimedStore is read-only: delete is not supported".into(),
                })
            })
            .boxed()
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        match prefix {
            Some(prefix) if self.is_log_path(prefix) => self.cache.list(Some(prefix)),
            _ => self.inner.list(prefix),
        }
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        match prefix {
            Some(prefix) if self.is_log_path(prefix) => {
                self.cache.list_with_offset(Some(prefix), offset)
            }
            _ => self.inner.list_with_offset(prefix, offset),
        }
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        match prefix {
            Some(prefix) if self.is_log_path(prefix) => {
                self.cache.list_with_delimiter(Some(prefix)).await
            }
            _ => self.inner.list_with_delimiter(prefix).await,
        }
    }

    async fn copy_opts(
        &self,
        _from: &Path,
        _to: &Path,
        _options: CopyOptions,
    ) -> object_store::Result<()> {
        Err(self.read_only_error("copy"))
    }
}
