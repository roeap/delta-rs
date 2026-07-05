use deltalake_derive::DeltaConfig;
use serde::{Deserialize, Serialize};

// The dedicated IO runtime and its storage-backend wrapper build a tokio
// multi-threaded runtime and spawn onto it — neither works on wasm. Everything
// except the plain `RuntimeConfig` struct below is therefore native-only.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod native {
    use std::ops::Range;
    use std::sync::OnceLock;

    use bytes::Bytes;
    use futures::FutureExt;
    use futures::StreamExt;
    use futures::TryFutureExt;
    use futures::future::BoxFuture;
    use futures::stream::BoxStream;
    use object_store::path::Path;
    use object_store::{
        CopyOptions, Error as ObjectStoreError, GetOptions, GetResult, ListResult, ObjectMeta,
        ObjectStore, ObjectStoreExt, PutOptions, PutPayload, PutResult, RenameOptions,
        Result as ObjectStoreResult,
    };
    use object_store::{MultipartUpload, PutMultipartOptions};
    use tokio::runtime::{Builder as RuntimeBuilder, Handle, Runtime};
    use tracing::Instrument as _;

    use super::RuntimeConfig;

    /// Creates static IO Runtime with optional configuration
    fn io_rt(config: Option<&RuntimeConfig>) -> &Runtime {
        static IO_RT: OnceLock<Runtime> = OnceLock::new();
        IO_RT.get_or_init(|| {
            let rt = match config {
                Some(config) => {
                    let mut builder = if let Some(true) = config.multi_threaded {
                        RuntimeBuilder::new_multi_thread()
                    } else {
                        RuntimeBuilder::new_current_thread()
                    };

                    if let Some(threads) = config.worker_threads {
                        builder.worker_threads(threads);
                    }

                    match (config.enable_io, config.enable_time) {
                        (Some(true), Some(true)) => {
                            builder.enable_all();
                        }
                        (Some(false), Some(true)) => {
                            builder.enable_time();
                        }
                        _ => (),
                    };

                    #[cfg(unix)]
                    {
                        if let (Some(true), Some(false)) = (config.enable_io, config.enable_time) {
                            builder.enable_io();
                        }
                    }
                    builder
                        .thread_name(
                            config
                                .thread_name
                                .clone()
                                .unwrap_or("IO-runtime".to_string()),
                        )
                        .build()
                }
                _ => Runtime::new(),
            };
            rt.expect("Failed to create a tokio runtime for IO.")
        })
    }

    /// Provide custom Tokio RT or a runtime config
    #[derive(Debug, Clone)]
    pub enum IORuntime {
        /// Tokio RT handle
        RT(Handle),
        /// Configuration for tokio runtime
        Config(RuntimeConfig),
    }

    impl Default for IORuntime {
        fn default() -> Self {
            IORuntime::RT(io_rt(None).handle().clone())
        }
    }

    impl IORuntime {
        /// Retrieves the Tokio runtime for IO bound operations
        pub fn get_handle(&self) -> Handle {
            match self {
                IORuntime::RT(handle) => handle,
                IORuntime::Config(config) => io_rt(Some(config)).handle(),
            }
            .clone()
        }
    }

    /// Wraps any object store and runs IO in it's own runtime [EXPERIMENTAL]
    #[derive(Clone)]
    pub struct DeltaIOStorageBackend<T: ObjectStore + Clone> {
        /// The wrapped object store that performs the actual IO.
        pub inner: T,
        rt: IORuntime,
    }

    impl<T> DeltaIOStorageBackend<T>
    where
        T: ObjectStore + Clone,
    {
        /// Wrap `store` so that its IO operations are executed on the dedicated runtime `rt`.
        pub fn new(store: T, rt: IORuntime) -> Self {
            Self { inner: store, rt }
        }
    }

    impl<T: ObjectStore + Clone> DeltaIOStorageBackend<T> {
        /// spawn tasks on IO runtime
        pub fn spawn_io_rt<F, O>(
            &self,
            f: F,
            store: &T,
            path: Path,
        ) -> BoxFuture<'_, ObjectStoreResult<O>>
        where
            F: for<'a> FnOnce(&'a T, &'a Path) -> BoxFuture<'a, ObjectStoreResult<O>>
                + Send
                + 'static,
            O: Send + 'static,
        {
            let store = store.clone();
            // Propagate the caller's tracing context across the runtime boundary.
            // The span is built in the caller's dispatcher context, so it captures
            // the active subscriber; `.instrument` re-enters it on every poll on the
            // IO runtime, nesting the work under the originating span rather than
            // leaving it as a disconnected root.
            let span = tracing::debug_span!(
                "io_rt::object_store_op",
                path = %path,
                "mlflow.spanType" = crate::kernel::mlflow::SPAN_TYPE_TOOL,
                "delta.zone" = crate::kernel::mlflow::ZONE_DELTA_RS,
            );
            let fut = self
                .rt
                .get_handle()
                .spawn(async move { f(&store, &path).await }.instrument(span));
            fut.unwrap_or_else(|e| match e.try_into_panic() {
                Ok(p) => std::panic::resume_unwind(p),
                Err(e) => Err(ObjectStoreError::JoinError { source: e }),
            })
            .boxed()
        }

        /// spawn tasks on IO runtime
        pub fn spawn_io_rt_from_to<F, O>(
            &self,
            f: F,
            store: &T,
            from: Path,
            to: Path,
        ) -> BoxFuture<'_, ObjectStoreResult<O>>
        where
            F: for<'a> FnOnce(&'a T, &'a Path, &'a Path) -> BoxFuture<'a, ObjectStoreResult<O>>
                + Send
                + 'static,
            O: Send + 'static,
        {
            let store = store.clone();
            let span = tracing::debug_span!(
                "io_rt::object_store_op",
                from = %from,
                to = %to,
                "mlflow.spanType" = crate::kernel::mlflow::SPAN_TYPE_TOOL,
                "delta.zone" = crate::kernel::mlflow::ZONE_DELTA_RS,
            );
            let fut = self
                .rt
                .get_handle()
                .spawn(async move { f(&store, &from, &to).await }.instrument(span));
            fut.unwrap_or_else(|e| match e.try_into_panic() {
                Ok(p) => std::panic::resume_unwind(p),
                Err(e) => Err(ObjectStoreError::JoinError { source: e }),
            })
            .boxed()
        }
    }

    impl<T: ObjectStore + Clone> std::fmt::Debug for DeltaIOStorageBackend<T> {
        fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(fmt, "DeltaIOStorageBackend({:?})", self.inner)
        }
    }

    impl<T: ObjectStore + Clone> std::fmt::Display for DeltaIOStorageBackend<T> {
        fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(fmt, "DeltaIOStorageBackend({})", self.inner)
        }
    }

    #[async_trait::async_trait]
    impl<T: ObjectStore + Clone> ObjectStore for DeltaIOStorageBackend<T> {
        async fn put_opts(
            &self,
            location: &Path,
            bytes: PutPayload,
            options: PutOptions,
        ) -> ObjectStoreResult<PutResult> {
            self.spawn_io_rt(
                move |store, path| store.put_opts(path, bytes, options).boxed(),
                &self.inner,
                location.clone(),
            )
            .await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> ObjectStoreResult<GetResult> {
            self.spawn_io_rt(
                move |store, path| store.get_opts(path, options).boxed(),
                &self.inner,
                location.clone(),
            )
            .await
        }

        async fn get_ranges(
            &self,
            location: &Path,
            ranges: &[Range<u64>],
        ) -> ObjectStoreResult<Vec<Bytes>> {
            let ranges = ranges.to_vec();
            self.spawn_io_rt(
                move |store, path| {
                    let ranges = ranges.clone();
                    async move { store.get_ranges(path, &ranges).await }.boxed()
                },
                &self.inner,
                location.clone(),
            )
            .await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, ObjectStoreResult<Path>>,
        ) -> BoxStream<'static, ObjectStoreResult<Path>> {
            let store = self.clone();
            locations
                .map(move |location| {
                    let store = store.clone();
                    async move {
                        let location = location?;
                        store
                            .spawn_io_rt(
                                move |inner, path| inner.delete(path).boxed(),
                                &store.inner,
                                location.clone(),
                            )
                            .await?;
                        Ok(location)
                    }
                })
                .buffered(10)
                .boxed()
        }

        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, ObjectStoreResult<ObjectMeta>> {
            self.inner.list(prefix)
        }

        fn list_with_offset(
            &self,
            prefix: Option<&Path>,
            offset: &Path,
        ) -> BoxStream<'static, ObjectStoreResult<ObjectMeta>> {
            self.inner.list_with_offset(prefix, offset)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> ObjectStoreResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: CopyOptions,
        ) -> ObjectStoreResult<()> {
            self.spawn_io_rt_from_to(
                move |store, from_path, to_path| {
                    store.copy_opts(from_path, to_path, options).boxed()
                },
                &self.inner,
                from.clone(),
                to.clone(),
            )
            .await
        }

        async fn rename_opts(
            &self,
            from: &Path,
            to: &Path,
            options: RenameOptions,
        ) -> ObjectStoreResult<()> {
            self.spawn_io_rt_from_to(
                move |store, from_path, to_path| {
                    store.rename_opts(from_path, to_path, options).boxed()
                },
                &self.inner,
                from.clone(),
                to.clone(),
            )
            .await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            options: PutMultipartOptions,
        ) -> ObjectStoreResult<Box<dyn MultipartUpload>> {
            self.spawn_io_rt(
                move |store, path| store.put_multipart_opts(path, options).boxed(),
                &self.inner,
                location.clone(),
            )
            .await
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[tokio::test]
        async fn test_ioruntime_default() {
            let _ = IORuntime::default();
        }
    }
} // mod native

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use native::{DeltaIOStorageBackend, IORuntime};

/// Configuration for the (native-only) dedicated Tokio IO runtime.
///
/// This is a plain (de)serializable config struct with no tokio dependency, so it
/// is available on all targets (it is referenced by `StorageConfig`); it only has
/// an effect where the tokio-backed [`IORuntime`] exists (native).
#[derive(Debug, Clone, Serialize, Deserialize, Default, DeltaConfig)]
pub struct RuntimeConfig {
    /// Whether to use a multi-threaded runtime
    pub(crate) multi_threaded: Option<bool>,
    /// Number of worker threads to use
    pub(crate) worker_threads: Option<usize>,
    /// Name of the thread
    pub(crate) thread_name: Option<String>,
    /// Whether to enable IO
    pub(crate) enable_io: Option<bool>,
    /// Whether to enable time
    pub(crate) enable_time: Option<bool>,
}
