use std::sync::Arc;

use arrow_schema::{DataType as ArrowDataType, Field, Fields, Schema, SchemaRef as ArrowSchemaRef};
use datafusion::{
    config::TableParquetOptions,
    datasource::physical_plan::{
        JsonSource, ParquetSource, parquet::CachedParquetFileReaderFactory,
    },
    execution::{TaskContext, object_store::ObjectStoreUrl},
    physical_expr::planner::logical2physical,
    physical_plan::{ExecutionPlan, execute_stream},
    prelude::Expr,
};
use datafusion_datasource::{
    PartitionedFile, TableSchema, file_groups::FileGroup, file_scan_config::FileScanConfigBuilder,
    source::DataSourceExec,
};
use datafusion_physical_expr_adapter::DefaultPhysicalExprAdapterFactory;
use delta_kernel::{
    DeltaResult, EngineData, Error, FileDataReadResultIterator, FileMeta, FilteredEngineData,
    JsonHandler, ParquetHandler, PredicateRef,
    engine::{
        arrow_conversion::{TryFromArrow as _, TryIntoArrow as _},
        arrow_data::ArrowEngineData,
        parse_json as arrow_parse_json, to_json_bytes,
    },
    schema::{DataType, SchemaRef, StructType},
};
use futures::TryStreamExt;
use itertools::Itertools as _;
use object_store::{PutMode, path::Path};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::arrow::async_reader::ParquetObjectReader;
use url::Url;

use crate::delta_datafusion::engine::{
    AsObjectStoreUrl as _, BlockingStreamIterator, TracedHandle, UrlExt as _, predicate_to_df,
};

#[derive(Debug, Clone)]
pub struct DataFusionFileFormatHandler {
    handle: TracedHandle,
    ctx: Arc<TaskContext>,
}

impl DataFusionFileFormatHandler {
    /// Create a new [`DataFusionFileFormatHandler`] instance.
    pub fn new(ctx: Arc<TaskContext>, handle: TracedHandle) -> Self {
        Self { handle, ctx }
    }
}

impl ParquetHandler for DataFusionFileFormatHandler {
    fn read_parquet_files(
        &self,
        files: &[FileMeta],
        physical_schema: SchemaRef,
        predicate: Option<PredicateRef>,
    ) -> DeltaResult<FileDataReadResultIterator> {
        if files.is_empty() {
            return Ok(Box::new(std::iter::empty()));
        }

        // Re-enter an mlflow-annotated span so this kernel→engine callback nests under the
        // originating scan rather than starting a disconnected root trace.
        let span = tracing::debug_span!(
            "engine::read_parquet_files",
            num_files = files.len(),
            "mlflow.spanType" = crate::kernel::mlflow::SPAN_TYPE_TOOL,
            "delta.zone" = crate::kernel::mlflow::ZONE_ENGINE,
        );
        let _enter = span.enter();

        let store_url = files
            .first()
            .expect("files is not empty")
            .location
            .as_object_store_url();
        // A single file group keeps files in input order and reads them sequentially, satisfying
        // the kernel ordering contract (batches emitted in input-file order, no cross-file merge).
        let files = to_partitioned_files((store_url.clone(), files.iter().collect()))?.1;

        let arrow_schema: Schema = physical_schema.as_ref().try_into_arrow()?;
        let arrow_schema: ArrowSchemaRef = Arc::new(relax_nullability(&arrow_schema));
        let exec = parquet_exec(
            &self.ctx,
            files,
            arrow_schema,
            predicate
                .map(|p| predicate_to_df(p.as_ref(), &DataType::BOOLEAN))
                .transpose()
                .map_err(Error::generic_err)?,
            store_url,
        )?;

        execute_iter(exec, self.ctx.clone(), self.handle.clone())
    }

    fn read_parquet_footer(&self, file: &FileMeta) -> DeltaResult<delta_kernel::ParquetFooter> {
        let span = tracing::trace_span!(
            "engine::read_parquet_footer",
            path = %file.location,
            "mlflow.spanType" = crate::kernel::mlflow::SPAN_TYPE_TOOL,
            "delta.zone" = crate::kernel::mlflow::ZONE_ENGINE,
        );
        let _enter = span.enter();

        let store = self
            .ctx
            .runtime_env()
            .object_store(file.location.as_object_store_url())
            .map_err(Error::generic_err)?;
        let location = file.location.clone();
        let size = file.size;

        // Skip the embedded arrow IPC schema so the derived kernel schema matches what the
        // default engine produces (`reader_options()` sets exactly this option, but it is
        // `pub(crate)` in the kernel, so inline it here).
        let reader_options = ArrowReaderOptions::new().with_skip_arrow_metadata(true);

        let metadata = self.handle.block_on(async move {
            if location.is_presigned() {
                let resp = reqwest::get(location).await.map_err(Error::generic_err)?;
                let bytes = resp.bytes().await.map_err(Error::generic_err)?;
                ArrowReaderMetadata::load(&bytes, reader_options).map_err(Error::from)
            } else {
                let path = Path::from_url_path(location.path())?;
                let mut reader = ParquetObjectReader::new(store, path).with_file_size(size);
                ArrowReaderMetadata::load_async(&mut reader, reader_options)
                    .await
                    .map_err(Error::from)
            }
        })?;

        let schema = StructType::try_from_arrow(metadata.schema().as_ref())
            .map(Arc::new)
            .map_err(Error::Arrow)?;
        Ok(delta_kernel::ParquetFooter { schema })
    }

    fn write_parquet_file(
        &self,
        _location: url::Url,
        _data: Box<dyn Iterator<Item = DeltaResult<Box<dyn EngineData>>> + Send>,
    ) -> DeltaResult<()> {
        todo!("write parquet file")
    }
}

impl JsonHandler for DataFusionFileFormatHandler {
    fn parse_json(
        &self,
        json_strings: Box<dyn EngineData>,
        output_schema: SchemaRef,
    ) -> DeltaResult<Box<dyn EngineData>> {
        arrow_parse_json(json_strings, output_schema)
    }

    fn read_json_files(
        &self,
        files: &[FileMeta],
        physical_schema: SchemaRef,
        _predicate: Option<PredicateRef>,
    ) -> DeltaResult<FileDataReadResultIterator> {
        if files.is_empty() {
            return Ok(Box::new(std::iter::empty()));
        }

        // Re-enter an mlflow-annotated span so this kernel→engine callback nests under the
        // originating scan/operation trace rather than starting a disconnected root trace.
        let span = tracing::debug_span!(
            "engine::read_json_files",
            num_files = files.len(),
            "mlflow.spanType" = crate::kernel::mlflow::SPAN_TYPE_TOOL,
            "delta.zone" = crate::kernel::mlflow::ZONE_ENGINE,
        );
        let _enter = span.enter();

        let store_url = files
            .first()
            .expect("files is not empty")
            .location
            .as_object_store_url();
        // Single file group => input-order, sequential reads (kernel ordering contract).
        let files = to_partitioned_files((store_url.clone(), files.iter().collect()))?.1;

        let arrow_schema: Schema = physical_schema.as_ref().try_into_arrow()?;
        let arrow_schema: ArrowSchemaRef = Arc::new(relax_nullability(&arrow_schema));
        let exec = json_exec(store_url, files, arrow_schema);

        execute_iter(exec, self.ctx.clone(), self.handle.clone())
    }

    // note: for now we just buffer all the data and write it out all at once
    fn write_json_file(
        &self,
        path: &Url,
        data: Box<dyn Iterator<Item = DeltaResult<FilteredEngineData>> + Send + '_>,
        overwrite: bool,
    ) -> DeltaResult<()> {
        let span = tracing::debug_span!(
            "engine::write_json_file",
            path = %path,
            overwrite,
            "mlflow.spanType" = crate::kernel::mlflow::SPAN_TYPE_TOOL,
            "delta.zone" = crate::kernel::mlflow::ZONE_ENGINE,
        );
        let _enter = span.enter();

        let buffer = to_json_bytes(data)?;
        let put_mode = if overwrite {
            PutMode::Overwrite
        } else {
            PutMode::Create
        };

        let store_url = path.as_object_store_url();
        let store = self
            .ctx
            .runtime_env()
            .object_store(store_url)
            .map_err(Error::generic_err)?;

        let path = Path::from_url_path(path.path())?;
        let path_str = path.to_string();
        self.handle
            .block_on(async move { store.put_opts(&path, buffer.into(), put_mode.into()).await })
            .map_err(|e| match e {
                object_store::Error::AlreadyExists { .. } => Error::FileAlreadyExists(path_str),
                e => e.into(),
            })?;

        Ok(())
    }
}

fn execute_iter(
    exec: Arc<dyn ExecutionPlan>,
    ctx: Arc<TaskContext>,
    task_executor: TracedHandle,
) -> DeltaResult<FileDataReadResultIterator> {
    let stream = execute_stream(exec, ctx)
        .map_err(Error::generic_err)?
        .map_err(Error::generic_err)
        .map_ok(|batch| Box::new(ArrowEngineData::new(batch)) as Box<dyn EngineData>);

    Ok(Box::new(BlockingStreamIterator {
        stream: Some(Box::pin(stream)),
        handle: task_executor,
    }))
}

fn parquet_exec(
    ctx: &Arc<TaskContext>,
    files: Vec<PartitionedFile>,
    read_schema: ArrowSchemaRef,
    predicate: Option<Expr>,
    store_url: ObjectStoreUrl,
) -> DeltaResult<Arc<dyn ExecutionPlan>> {
    let pq_options = TableParquetOptions {
        global: ctx.session_config().options().execution.parquet.clone(),
        ..Default::default()
    };

    let reader_factory = Arc::new(CachedParquetFileReaderFactory::new(
        ctx.runtime_env()
            .object_store(&store_url)
            .map_err(Error::generic_err)?,
        ctx.runtime_env().cache_manager.get_file_metadata_cache(),
    ));
    // The read schema lives on the source's `TableSchema`; Delta partition columns are injected
    // by kernel transforms above the parquet scan, so there are no parquet partition fields.
    let table_schema = TableSchema::from(read_schema.clone());
    let mut file_source = ParquetSource::new(table_schema)
        .with_table_parquet_options(pq_options)
        .with_parquet_file_reader_factory(reader_factory);

    if let Some(pred) = predicate {
        let physical = logical2physical(&pred, read_schema.as_ref());
        file_source = file_source
            .with_predicate(physical)
            .with_pushdown_filters(true);
    }

    // One file group keeps files in input order and reads them sequentially (kernel contract).
    let file_group: FileGroup = files.into_iter().collect();

    // Reconcile the file's on-disk schema with the kernel's requested `physical_schema`
    // (e.g. nullable→non-nullable struct fields, missing feature-list columns in older
    // checkpoints) the way the default engine did via projection+cast. Without this, the
    // parquet scan's strict logical→physical cast rejects checkpoints like `protocol`.
    let config = FileScanConfigBuilder::new(store_url, Arc::new(file_source))
        .with_file_groups(vec![file_group])
        .with_expr_adapter(Some(Arc::new(DefaultPhysicalExprAdapterFactory) as _))
        .build();

    Ok(DataSourceExec::from_data_source(config) as Arc<dyn ExecutionPlan>)
}

fn json_exec(
    store_url: ObjectStoreUrl,
    files: Vec<PartitionedFile>,
    arrow_schema: ArrowSchemaRef,
) -> Arc<dyn ExecutionPlan> {
    let file_group: FileGroup = files.into_iter().collect();
    let config = FileScanConfigBuilder::new(
        store_url,
        Arc::new(JsonSource::new(TableSchema::from(arrow_schema))),
    )
    .with_file_groups(vec![file_group])
    // Reconcile per-file JSON schemas to the requested schema (nullability, missing columns
    // across commit versions) rather than failing on a strict cast.
    .with_expr_adapter(Some(Arc::new(DefaultPhysicalExprAdapterFactory) as _))
    .build();
    DataSourceExec::from_data_source(config)
}

/// Recursively mark every field (including nested struct/list/map children) as nullable.
///
/// The kernel hands us its *logical* schema, whose nested struct fields (e.g. `protocol`) are
/// often non-nullable, while the on-disk checkpoint stores them as nullable. DataFusion's parquet
/// scan refuses to cast a nullable file field to a non-nullable logical field, so we read against a
/// fully-nullable view of the schema — matching the default engine, which never enforces
/// non-nullability on read and reconciles types itself afterwards.
fn relax_nullability(schema: &Schema) -> Schema {
    Schema::new(relax_fields(schema.fields()))
}

fn relax_fields(fields: &Fields) -> Fields {
    fields
        .iter()
        .map(|f| Arc::new(relax_field(f)))
        .collect::<Vec<_>>()
        .into()
}

fn relax_field(field: &Field) -> Field {
    let data_type = match field.data_type() {
        ArrowDataType::Struct(children) => ArrowDataType::Struct(relax_fields(children)),
        ArrowDataType::List(child) => ArrowDataType::List(Arc::new(relax_field(child))),
        ArrowDataType::LargeList(child) => ArrowDataType::LargeList(Arc::new(relax_field(child))),
        // Arrow requires the map's `entries` struct field and its `key` child to stay
        // non-nullable, so we relax only the map *value*, leaving the entries wrapper intact.
        ArrowDataType::Map(entries, sorted) => {
            ArrowDataType::Map(Arc::new(relax_map_entries(entries)), *sorted)
        }
        other => other.clone(),
    };
    Field::new(field.name(), data_type, true).with_metadata(field.metadata().clone())
}

/// Relax a map's `entries` struct: keep the entries field and its `key` child non-nullable
/// (an Arrow invariant), but relax the `value` child so nested nullability differences don't
/// reject the read.
fn relax_map_entries(entries: &Field) -> Field {
    let ArrowDataType::Struct(children) = entries.data_type() else {
        return entries.clone();
    };
    let new_children: Fields = children
        .iter()
        .enumerate()
        .map(|(idx, child)| {
            // Field 0 is the key (must stay non-nullable); field 1 is the value.
            if idx == 0 {
                child.clone()
            } else {
                Arc::new(relax_field(child))
            }
        })
        .collect::<Vec<_>>()
        .into();
    Field::new(
        entries.name(),
        ArrowDataType::Struct(new_children),
        entries.is_nullable(),
    )
    .with_metadata(entries.metadata().clone())
}

fn to_partitioned_files(
    arg: (ObjectStoreUrl, Vec<&FileMeta>),
) -> DeltaResult<(ObjectStoreUrl, Vec<PartitionedFile>)> {
    let (url, files) = arg;
    let part_files = files
        .into_iter()
        .map(|f| {
            let path = Path::from_url_path(f.location.path())?;
            let mut partitioned_file = PartitionedFile::new(path.to_string(), f.size);
            // NB: we need to reassign the location since the 'new' method does
            // incorrect or inconsistent encoding internally.
            partitioned_file.object_meta.location = path;
            Ok::<_, Error>(partitioned_file)
        })
        .try_collect::<_, Vec<_>, _>()?;
    Ok::<_, Error>((url, part_files))
}
