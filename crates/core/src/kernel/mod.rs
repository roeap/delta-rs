//! Delta Kernel module
//!
//! The Kernel module contains all the logic for reading and processing the Delta Lake transaction log.

use delta_kernel::engine::arrow_expression::ArrowEvaluationHandler;
use std::sync::{Arc, LazyLock};
use tokio::task::JoinHandle;
use tracing::Span;
use tracing::dispatcher;

pub mod arrow;
pub mod error;
/// Core Delta log action models (Add, Remove, Metadata, Protocol, ...) and related types.
pub mod models;
pub mod scalars;
/// Delta and Arrow schema types, conversions, casting and partition handling.
pub mod schema;
pub(crate) mod snapshot;
pub mod transaction;

pub use arrow::engine_ext::StructDataExt;
pub use delta_kernel::Version;
pub use delta_kernel::engine;
pub use error::*;
pub use models::*;
pub use schema::*;
pub use snapshot::*;

pub(crate) static ARROW_HANDLER: LazyLock<Arc<ArrowEvaluationHandler>> =
    LazyLock::new(|| Arc::new(ArrowEvaluationHandler {}));

/// Run `f` on a blocking thread, propagating the current tracing dispatcher
/// across the thread boundary and entering `span` for the duration of `f`.
///
/// `span` should be constructed on the calling thread (where the parent span is
/// current) so it captures its parent linkage; it is then moved into the
/// blocking task and entered there. This makes the work performed by `f` (e.g.
/// a call into Delta Kernel) appear as a child of `span` in the trace tree
/// rather than as a disconnected root.
pub(crate) fn spawn_blocking_in_span<F, R>(span: Span, f: F) -> JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    // Capture the current dispatcher so the spawned thread reports to the same
    // subscriber as the caller.
    let dispatch = dispatcher::get_default(|d| d.clone());

    tokio::task::spawn_blocking(move || dispatcher::with_default(&dispatch, || span.in_scope(f)))
}

/// Like [`spawn_blocking_in_span`], but re-enters the *current* span instead of
/// a caller-supplied child span. Prefer [`spawn_blocking_in_span`] with an
/// explicit span so the blocking work gets its own node in the trace tree.
pub(crate) fn spawn_blocking_with_span<F, R>(f: F) -> JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    spawn_blocking_in_span(Span::current(), f)
}

/// Helpers for emitting OpenTelemetry span attributes that MLflow's trace UI
/// recognizes when traces are ingested over OTLP.
///
/// `tracing-opentelemetry` copies each span field into an OTel attribute using
/// the field name verbatim, so a field literally named `mlflow.spanType`
/// becomes the OTel attribute `mlflow.spanType` that MLflow translates into a
/// span-type chip. Since Rust identifiers cannot contain dots, the field must
/// be written with the quoted syntax at the span site, e.g.
/// `info_span!("kernel::scan_metadata", "mlflow.spanType" = MLFLOW_SPAN_TYPE_RETRIEVER)`.
///
/// `delta.zone` is a custom attribute (`delta-rs` / `kernel` / `engine`) MLflow
/// keeps as searchable span metadata, used to distinguish the three sides of the
/// delta-rs ↔ Delta Kernel handoff in the trace tree.
pub(crate) mod mlflow {
    use serde::Serialize;
    use tracing::Span;

    /// OTel attribute field names MLflow recognizes / the demo groups on.
    ///
    /// `info_span!` / `debug_span!` accept quoted dotted field names directly
    /// (`"mlflow.spanType" = ...`). The `#[instrument]` attribute macro in our
    /// `tracing-attributes` version does **not** — it requires an identifier or a
    /// const expression in braces. These constants let `#[instrument]` sites write
    /// `fields({FIELD_SPAN_TYPE} = SPAN_TYPE_WORKFLOW, ...)`.
    pub(crate) const FIELD_SPAN_TYPE: &str = "mlflow.spanType";
    pub(crate) const FIELD_ZONE: &str = "delta.zone";
    pub(crate) const FIELD_SPAN_INPUTS: &str = "mlflow.spanInputs";
    pub(crate) const FIELD_SPAN_OUTPUTS: &str = "mlflow.spanOutputs";

    /// MLflow span-type values we map Delta concepts onto. See the tracing
    /// section of `CONTRIBUTING.md` for the full mapping rationale.
    pub(crate) const SPAN_TYPE_WORKFLOW: &str = "WORKFLOW";
    pub(crate) const SPAN_TYPE_TASK: &str = "TASK";
    pub(crate) const SPAN_TYPE_AGENT: &str = "AGENT";
    pub(crate) const SPAN_TYPE_RETRIEVER: &str = "RETRIEVER";
    pub(crate) const SPAN_TYPE_TOOL: &str = "TOOL";

    /// `delta.zone` values identifying which side of the kernel handoff a span
    /// belongs to.
    pub(crate) const ZONE_DELTA_RS: &str = "delta-rs";
    pub(crate) const ZONE_KERNEL: &str = "kernel";
    pub(crate) const ZONE_ENGINE: &str = "engine";

    /// Serialize `value` as JSON and record it into the current span's `field`.
    ///
    /// Intended for the `mlflow.spanInputs` / `mlflow.spanOutputs` fields, which
    /// MLflow renders as the span's Inputs/Outputs panels (and, on the root
    /// span, as the trace-level request/response preview). The target field must
    /// have been declared as [`tracing::field::Empty`] at the span site.
    ///
    /// Serialization failures are intentionally swallowed: telemetry must never
    /// affect the operation it observes.
    pub(crate) fn record_json(field: &'static str, value: &impl Serialize) {
        if let Ok(json) = serde_json::to_string(value) {
            Span::current().record(field, json.as_str());
        }
    }

    /// Like [`record_json`], but records into the supplied `span` rather than the
    /// current one. Used when the inputs are known on the calling thread before a
    /// `spawn_blocking_in_span` closure enters the span on a blocking thread.
    pub(crate) fn record_json_in(span: &Span, field: &'static str, value: &impl Serialize) {
        if let Ok(json) = serde_json::to_string(value) {
            span.record(field, json.as_str());
        }
    }
}
