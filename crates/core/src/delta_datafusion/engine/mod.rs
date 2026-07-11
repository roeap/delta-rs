use std::sync::Arc;

use datafusion::catalog::Session;
use datafusion::execution::TaskContext;
use delta_kernel::{
    DeltaResult, Engine, EvaluationHandler, JsonHandler, ParquetHandler, StorageHandler,
};
use futures::{StreamExt as _, stream::BoxStream};
use url::Url;

pub(crate) use self::expressions::*;
// Public DataFusion `Expr` → kernel `Predicate` converter (checkpoint row-group
// skipping); see `expressions::to_delta_predicate`.
pub use self::expressions::to_delta_predicate;
use self::file_formats::DataFusionFileFormatHandler;
pub use self::storage::{AsObjectStoreUrl, DataFusionStorageHandler};
use crate::kernel::ARROW_HANDLER;
pub use crate::kernel::executor::{ExecutorHandle, InlineExecutor, TracedHandle};

mod expressions;
mod file_formats;
mod storage;

/// A Datafusion based Kernel Engine
#[derive(Clone, Debug)]
pub struct DataFusionEngine {
    storage: Arc<DataFusionStorageHandler>,
    formats: Arc<DataFusionFileFormatHandler>,
}

impl DataFusionEngine {
    /// Create an engine from a DataFusion [`Session`], reusing its task context and the
    /// ambient executor (the current Tokio runtime natively, the inline executor on wasm).
    /// This is the convenient entry point when wiring the kernel engine into an active
    /// query session.
    pub fn new_from_session(session: &dyn Session) -> Arc<Self> {
        Self::new(session.task_ctx(), ExecutorHandle::current()).into()
    }

    /// Create an engine directly from a DataFusion [`TaskContext`], using the ambient
    /// executor. Useful inside physical operators where only the task context is available.
    pub fn new_from_context(ctx: Arc<TaskContext>) -> Arc<Self> {
        Self::new(ctx, ExecutorHandle::current()).into()
    }

    /// Create an engine from an explicit [`TaskContext`] and executor.
    ///
    /// The other constructors delegate here; call this directly when you need to bind the
    /// engine to a specific executor rather than the ambient one — e.g. a particular Tokio
    /// runtime [`Handle`](tokio::runtime::Handle) (which converts into an
    /// [`ExecutorHandle`]), or [`InlineExecutor`] to force the wasm execution model in
    /// native tests.
    pub fn new(ctx: Arc<TaskContext>, executor: impl Into<ExecutorHandle>) -> Self {
        let executor = executor.into();
        let storage = Arc::new(DataFusionStorageHandler::new(ctx.clone(), executor.clone()));
        let formats = Arc::new(DataFusionFileFormatHandler::new(ctx, executor));
        Self { storage, formats }
    }
}

impl Engine for DataFusionEngine {
    fn evaluation_handler(&self) -> Arc<dyn EvaluationHandler> {
        ARROW_HANDLER.clone()
    }

    fn storage_handler(&self) -> Arc<dyn StorageHandler> {
        self.storage.clone()
    }

    fn json_handler(&self) -> Arc<dyn JsonHandler> {
        self.formats.clone()
    }

    fn parquet_handler(&self) -> Arc<dyn ParquetHandler> {
        self.formats.clone()
    }
}

/// Converts a Stream-producing future to a synchronous iterator.
///
/// This method performs the initial blocking call to extract the stream from the future, and each
/// subsequent call to `next` on the iterator translates to a blocking `stream.next()` call, using
/// the provided `executor`. Buffered streams allow concurrency in the form of prefetching,
/// because that initial call will attempt to populate the N buffer slots; every call to
/// `stream.next()` leaves an empty slot (out of N buffer slots) that the stream immediately
/// attempts to fill by launching another future that can make progress in the background while we
/// block on and consume each of the N-1 entries that precede it.
///
/// This is an internal utility for bridging object_store's async API to
/// Delta Kernel's synchronous handler traits.
pub(crate) fn stream_future_to_iter<T: Send + 'static>(
    executor: ExecutorHandle,
    stream_future: impl Future<Output = DeltaResult<BoxStream<'static, DeltaResult<T>>>>
    + Send
    + 'static,
) -> DeltaResult<Box<dyn Iterator<Item = DeltaResult<T>> + Send>> {
    Ok(Box::new(BlockingStreamIterator {
        stream: Some(executor.try_block_on(stream_future)??),
        executor,
    }))
}

/// Bridges an async stream of `DeltaResult` items to a synchronous iterator by driving one
/// item per `next()` call on `executor`. When the executor itself fails (the inline executor's
/// "would block" case), the iterator yields that error as its final item and fuses.
pub(crate) struct BlockingStreamIterator<T: Send + 'static> {
    pub(crate) stream: Option<BoxStream<'static, DeltaResult<T>>>,
    pub(crate) executor: ExecutorHandle,
}

impl<T: Send + 'static> Iterator for BlockingStreamIterator<T> {
    type Item = DeltaResult<T>;

    fn next(&mut self) -> Option<Self::Item> {
        // Move the stream into the future so we can block on it.
        let mut stream = self.stream.take()?;
        let task = async move { (stream.next().await, stream) };
        match self.executor.try_block_on(task) {
            Ok((item, stream)) => {
                // We must not poll an exhausted stream after it returned None.
                if item.is_some() {
                    self.stream = Some(stream);
                }
                item
            }
            // Executor failure: yield the error; `self.stream` stays `None`, fusing the iterator.
            Err(err) => Some(Err(err)),
        }
    }
}

pub(crate) trait UrlExt {
    // Check if a given url is a presigned url and can be used
    // to access the object store via simple http requests
    fn is_presigned(&self) -> bool;
}

impl UrlExt for Url {
    fn is_presigned(&self) -> bool {
        matches!(self.scheme(), "http" | "https")
            && (
                // https://docs.aws.amazon.com/AmazonS3/latest/API/sigv4-query-string-auth.html
                // https://developers.cloudflare.com/r2/api/s3/presigned-urls/
                self
                .query_pairs()
                .any(|(k, _)| k.eq_ignore_ascii_case("X-Amz-Signature")) ||
                // https://learn.microsoft.com/en-us/rest/api/storageservices/create-user-delegation-sas#version-2020-12-06-and-later
                // note signed permission (sp) must always be present
                self
                .query_pairs().any(|(k, _)| k.eq_ignore_ascii_case("sp")) ||
                // https://cloud.google.com/storage/docs/authentication/signatures
                self
                .query_pairs().any(|(k, _)| k.eq_ignore_ascii_case("X-Goog-Credential")) ||
                // https://www.alibabacloud.com/help/en/oss/user-guide/upload-files-using-presigned-urls
                self
                .query_pairs().any(|(k, _)| k.eq_ignore_ascii_case("X-OSS-Credential"))
            )
    }
}
