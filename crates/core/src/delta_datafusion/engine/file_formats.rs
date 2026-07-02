use std::sync::Arc;

use dashmap::{DashMap, mapref::one::Ref};
use datafusion::execution::{
    TaskContext,
    object_store::{ObjectStoreRegistry, ObjectStoreUrl},
};
use delta_kernel::engine::parse_json as arrow_parse_json;
use delta_kernel::{
    EngineData, FileDataReadResultIterator, FileMeta, FilteredEngineData, JsonHandler,
    ParquetHandler, PredicateRef, error::DeltaResult as KernelResult, schema::SchemaRef,
};
// kernel v0.25.0 relocated the tokio DefaultEngine out of `delta_kernel::engine::default`
// into the separate `delta_kernel_default_engine` crate (native-only).
use delta_kernel_default_engine::{
    executor::tokio::{TokioBackgroundExecutor, TokioMultiThreadExecutor},
    json::DefaultJsonHandler,
    parquet::DefaultParquetHandler,
};
use itertools::Itertools;
use tokio::runtime::{Handle, RuntimeFlavor};

use super::storage::{AsObjectStoreUrl, group_by_store};

#[derive(Clone)]
pub struct DataFusionFileFormatHandler {
    ctx: Arc<TaskContext>,
    pq_registry: Arc<DashMap<ObjectStoreUrl, Arc<dyn ParquetHandler>>>,
    json_registry: Arc<DashMap<ObjectStoreUrl, Arc<dyn JsonHandler>>>,
    handle: Handle,
    /// The span current when this engine was constructed (typically the
    /// operation / scan span). The kernel drives JSON/parquet reads from its own
    /// task executor, which carries no `tracing` context, so without re-entering
    /// this span the callbacks would start a brand-new disconnected root trace.
    /// See [`Self::read_json_files`].
    span: tracing::Span,
}

impl DataFusionFileFormatHandler {
    /// Create a new [`DatafusionParquetHandler`] instance.
    pub fn new(ctx: Arc<TaskContext>, handle: Handle) -> Self {
        Self {
            ctx,
            pq_registry: DashMap::new().into(),
            json_registry: DashMap::new().into(),
            handle,
            span: tracing::Span::current(),
        }
    }

    fn registry(&self) -> Arc<dyn ObjectStoreRegistry> {
        self.ctx.runtime_env().object_store_registry.clone()
    }

    fn get_or_create_pq(
        &self,
        url: ObjectStoreUrl,
    ) -> KernelResult<Ref<'_, ObjectStoreUrl, Arc<dyn ParquetHandler>>> {
        if let Some(handler) = self.pq_registry.get(&url) {
            return Ok(handler);
        }
        let store = self
            .registry()
            .get_store(url.as_ref())
            .map_err(delta_kernel::Error::generic_err)?;

        let handler: Arc<dyn ParquetHandler> = match self.handle.runtime_flavor() {
            RuntimeFlavor::MultiThread => Arc::new(DefaultParquetHandler::new(
                store,
                Arc::new(TokioMultiThreadExecutor::new(self.handle.clone())),
            )),
            RuntimeFlavor::CurrentThread => Arc::new(DefaultParquetHandler::new(
                store,
                Arc::new(TokioBackgroundExecutor::new()),
            )),
            _ => panic!("unsupported runtime flavor"),
        };

        self.pq_registry.insert(url.clone(), handler);
        Ok(self.pq_registry.get(&url).unwrap())
    }

    fn get_or_create_json(
        &self,
        url: ObjectStoreUrl,
    ) -> KernelResult<Ref<'_, ObjectStoreUrl, Arc<dyn JsonHandler>>> {
        if let Some(handler) = self.json_registry.get(&url) {
            return Ok(handler);
        }
        let store = self
            .registry()
            .get_store(url.as_ref())
            .map_err(delta_kernel::Error::generic_err)?;

        let handler: Arc<dyn JsonHandler> = match self.handle.runtime_flavor() {
            RuntimeFlavor::MultiThread => Arc::new(DefaultJsonHandler::new(
                store,
                Arc::new(TokioMultiThreadExecutor::new(self.handle.clone())),
            )),
            RuntimeFlavor::CurrentThread => Arc::new(DefaultJsonHandler::new(
                store,
                Arc::new(TokioBackgroundExecutor::new()),
            )),
            _ => panic!("unsupported runtime flavor"),
        };

        self.json_registry.insert(url.clone(), handler);
        Ok(self.json_registry.get(&url).unwrap())
    }
}

impl ParquetHandler for DataFusionFileFormatHandler {
    fn read_parquet_files(
        &self,
        files: &[FileMeta],
        physical_schema: SchemaRef,
        predicate: Option<PredicateRef>,
    ) -> KernelResult<FileDataReadResultIterator> {
        // See `read_json_files`: re-enter the originating span so this callback
        // nests under the scan rather than orphaning into its own trace.
        let _parent = self.span.enter();
        let span = tracing::debug_span!(
            "engine::read_parquet_files",
            num_files = files.len(),
            "mlflow.spanType" = crate::kernel::mlflow::SPAN_TYPE_TOOL,
            "delta.zone" = crate::kernel::mlflow::ZONE_ENGINE,
        );
        let _enter = span.enter();
        let grouped_files = group_by_store(files.to_vec());
        Ok(Box::new(
            grouped_files
                .into_iter()
                .map(|(url, files)| {
                    self.get_or_create_pq(url)?.read_parquet_files(
                        &files.to_vec(),
                        physical_schema.clone(),
                        predicate.clone(),
                    )
                })
                // TODO: this should not do any blocking operations, since this should
                // happen when the iterators are polled and we are just creating a vec of iterators.
                // Is this correct?
                .try_collect::<_, Vec<_>, _>()?
                .into_iter()
                .flatten(),
        ))
    }

    fn write_parquet_file(
        &self,
        _location: url::Url,
        _data: Box<dyn Iterator<Item = KernelResult<Box<dyn EngineData>>> + Send>,
    ) -> KernelResult<()> {
        todo!("write parquet file")
    }

    fn read_parquet_footer(&self, file: &FileMeta) -> KernelResult<delta_kernel::ParquetFooter> {
        let _parent = self.span.enter();
        let span = tracing::trace_span!(
            "engine::read_parquet_footer",
            path = %file.location,
            "mlflow.spanType" = crate::kernel::mlflow::SPAN_TYPE_TOOL,
            "delta.zone" = crate::kernel::mlflow::ZONE_ENGINE,
        );
        let _enter = span.enter();
        self.get_or_create_pq(file.as_object_store_url())?
            .read_parquet_footer(file)
    }
}

impl JsonHandler for DataFusionFileFormatHandler {
    fn parse_json(
        &self,
        json_strings: Box<dyn EngineData>,
        output_schema: SchemaRef,
    ) -> KernelResult<Box<dyn EngineData>> {
        arrow_parse_json(json_strings, output_schema)
    }

    fn read_json_files(
        &self,
        files: &[FileMeta],
        physical_schema: SchemaRef,
        predicate: Option<PredicateRef>,
    ) -> KernelResult<FileDataReadResultIterator> {
        // Re-enter the engine's originating span before opening the callback span
        // so this kernel→engine callback nests under the scan/operation that
        // triggered it rather than starting a disconnected root trace.
        let _parent = self.span.enter();
        let span = tracing::debug_span!(
            "engine::read_json_files",
            num_files = files.len(),
            "mlflow.spanType" = crate::kernel::mlflow::SPAN_TYPE_TOOL,
            "delta.zone" = crate::kernel::mlflow::ZONE_ENGINE,
        );
        let _enter = span.enter();
        let grouped_files = group_by_store(files.to_vec());
        Ok(Box::new(
            grouped_files
                .into_iter()
                .map(|(url, files)| {
                    self.get_or_create_json(url)?.read_json_files(
                        &files.to_vec(),
                        physical_schema.clone(),
                        predicate.clone(),
                    )
                })
                // TODO: this should not do any blocking operations, since this should
                // happen when the iterators are polled and we are just creating a vec of iterators.
                // Is this correct?
                .try_collect::<_, Vec<_>, _>()?
                .into_iter()
                .flatten(),
        ))
    }

    fn write_json_file(
        &self,
        path: &url::Url,
        data: Box<dyn Iterator<Item = KernelResult<FilteredEngineData>> + Send + '_>,
        overwrite: bool,
    ) -> KernelResult<()> {
        self.get_or_create_json(path.as_object_store_url())?
            .write_json_file(path, data, overwrite)
    }
}
