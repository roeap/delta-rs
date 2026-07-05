//! Opaque predicate bridge: carry untranslatable DataFusion filter expressions
//! through the kernel predicate seam.
//!
//! DataFusion filter expressions the kernel predicate model cannot represent
//! (UDFs, `LIKE`, `CASE`, regex, most scalar functions…) would otherwise die in
//! [`to_delta_predicate`](super::to_delta_predicate)'s catch-all and never reach
//! the kernel scan — losing partition pruning and file skipping for them
//! (results stay correct; DataFusion re-applies the filters post-scan).
//!
//! [`DataFusionOpaquePredicateOp`] wraps such an `Expr` as a kernel
//! [`Predicate::Opaque`](delta_kernel::expressions::Predicate::Opaque) op that
//! evaluates the original DataFusion expression, extending **partition pruning**
//! to arbitrary DataFusion predicates.
//!
//! # Evaluation model
//!
//! The op carries the DataFusion `Expr` and the ordered list of leaf column
//! names it references. It is embedded with kernel column expressions (`exprs`),
//! one per referenced column. At evaluation time the op does *not* look columns
//! up by name in the batch; instead it evaluates each embedded `Expression`
//! against the batch (letting the kernel resolve nested paths such as
//! `partitionValues_parsed.<col>`), then rebuilds a batch keyed by the original
//! DataFusion column names and runs the `Expr`. The same code therefore serves
//! two callers with different batch shapes:
//!
//! - **Partition pruning / data skipping.** [`Self::as_data_skipping_predicate`]
//!   rewrites the op over the `partitionValues_parsed.<col>` stat columns and
//!   the kernel's `DataSkippingFilter` evaluates it columnarly via
//!   [`Self::eval_pred`]. Only *partition* columns participate in v1 (their
//!   partition value is exact — min == max); any non-partition reference
//!   disqualifies the op, so it never wrongly prunes on approximate stats.
//! - **Direct data.** The embedded `exprs` are plain `Column` references
//!   evaluated against a data batch.
//!
//! # Invariant — opaque ops never wrongly prune
//!
//! Scalar evaluation degrades to "don't know" (`Ok(None)`) on any missing column
//! or evaluation error rather than failing the scan; the stats rewrite is
//! produced only for exact partition columns. A conservative opaque op can
//! therefore never *remove* a file/partition that might match; at worst it
//! declines to prune and DataFusion re-applies the filter post-scan.

use datafusion::arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch};
use datafusion::arrow::compute;
use datafusion::arrow::datatypes::{Field, Schema};
use datafusion::logical_expr::Expr;
use datafusion::physical_expr::planner::logical2physical;
use delta_kernel::DeltaResult;
use delta_kernel::engine::arrow_expression::evaluate_expression::evaluate_expression;
use delta_kernel::engine::arrow_expression::opaque::ArrowOpaquePredicateOp;
use delta_kernel::expressions::{Expression, ScalarExpressionEvaluator};
use delta_kernel::kernel_predicates::{
    DirectDataSkippingPredicateEvaluator, DirectPredicateEvaluator,
    IndirectDataSkippingPredicateEvaluator,
};
use delta_kernel::schema::DataType as KernelDataType;
use delta_kernel::{Error as KernelError, Predicate};
use std::sync::Arc;

/// An engine-defined kernel predicate op that evaluates an arbitrary DataFusion
/// [`Expr`] the kernel predicate model cannot otherwise represent.
///
/// Constructed via [`arrow_opaque_predicate`]. The `exprs` handed to the kernel
/// are the referenced columns (as kernel expressions), aligned by position with
/// [`Self::columns`], the original DataFusion leaf column names.
#[derive(Debug, Clone)]
pub(crate) struct DataFusionOpaquePredicateOp {
    /// The DataFusion logical predicate; references input columns by name.
    expr: Expr,
    /// The DataFusion leaf column names referenced by `expr`, in the same order
    /// as the embedded kernel column expressions.
    columns: Vec<String>,
    /// Cached display of `expr`, used as the op `name()`.
    name: String,
}

impl DataFusionOpaquePredicateOp {
    /// Builds the op from a DataFusion predicate and the ordered leaf names of
    /// the columns it references.
    pub(crate) fn new(expr: Expr, columns: Vec<String>) -> Self {
        let name = expr.to_string();
        Self {
            expr,
            columns,
            name,
        }
    }

    /// The wrapped DataFusion expression (used to recover the original filter on
    /// the `to_datafusion` round-trip).
    pub(crate) fn expr(&self) -> &Expr {
        &self.expr
    }

    /// Evaluate the wrapped expression, applying `inverted`.
    ///
    /// Each embedded `arg` is first evaluated against `batch` (so the kernel
    /// resolves nested paths like `partitionValues_parsed.<col>`), then the
    /// resulting arrays are reassembled into a batch keyed by the original
    /// DataFusion column names, against which the `Expr` is planned and run.
    fn eval_bool(
        &self,
        args: &[Expression],
        batch: &RecordBatch,
        inverted: bool,
    ) -> DeltaResult<BooleanArray> {
        if args.len() != self.columns.len() {
            return Err(KernelError::generic(format!(
                "opaque predicate arg/column count mismatch: {} args, {} columns",
                args.len(),
                self.columns.len()
            )));
        }

        // Evaluate each embedded column expression against the input batch.
        let arrays: Vec<ArrayRef> = args
            .iter()
            .map(|arg| evaluate_expression(arg, batch, None))
            .collect::<DeltaResult<_>>()?;
        let fields: Vec<Field> = self
            .columns
            .iter()
            .zip(&arrays)
            .map(|(name, array)| Field::new(name, array.data_type().clone(), true))
            .collect();
        let df_batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
            .map_err(|e| KernelError::generic(format!("opaque predicate batch: {e}")))?;

        let physical = logical2physical(&self.expr, df_batch.schema().as_ref());
        let values = physical
            .evaluate(&df_batch)
            .map_err(|e| KernelError::generic(format!("opaque predicate eval: {e}")))?;
        let array = values
            .into_array(df_batch.num_rows())
            .map_err(|e| KernelError::generic(format!("opaque predicate eval: {e}")))?;
        let bools = array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| KernelError::generic("opaque predicate did not evaluate to a boolean"))?
            .clone();
        Ok(if inverted {
            compute::not(&bools)?
        } else {
            bools
        })
    }
}

impl PartialEq for DataFusionOpaquePredicateOp {
    fn eq(&self, other: &Self) -> bool {
        // Structural equality of the underlying DataFusion expression: two
        // ops built from structurally equal `Expr`s must compare equal so the
        // kernel can dedup/compare predicate trees.
        self.expr == other.expr
    }
}

impl ArrowOpaquePredicateOp for DataFusionOpaquePredicateOp {
    fn name(&self) -> &str {
        &self.name
    }

    fn eval_pred(
        &self,
        args: &[Expression],
        batch: &RecordBatch,
        inverted: bool,
    ) -> DeltaResult<BooleanArray> {
        self.eval_bool(args, batch, inverted)
    }

    fn eval_pred_scalar(
        &self,
        eval_expr: &ScalarExpressionEvaluator<'_>,
        _eval_pred: &DirectPredicateEvaluator<'_>,
        exprs: &[Expression],
        inverted: bool,
    ) -> DeltaResult<Option<bool>> {
        // Scalar evaluation for a single partition tuple. Resolve every embedded
        // column to a scalar via the kernel-supplied evaluator, build a one-row
        // batch keyed by the column expressions, and reuse the columnar path.
        // Any missing column or evaluation failure resolves to "don't know"
        // (`Ok(None)`) — an opaque op must never fail the scan or wrongly prune.
        match self.eval_scalar_inner(eval_expr, exprs, inverted) {
            Ok(value) => Ok(value),
            Err(_) => Ok(None),
        }
    }

    fn eval_as_data_skipping_predicate(
        &self,
        _predicate_evaluator: &DirectDataSkippingPredicateEvaluator<'_>,
        _exprs: &[Expression],
        _inverted: bool,
    ) -> Option<bool> {
        // Direct data-skipping evaluation over min/max stats is not supported in
        // v1; partition pruning is handled by `as_data_skipping_predicate`.
        None
    }

    fn as_data_skipping_predicate(
        &self,
        predicate_evaluator: &IndirectDataSkippingPredicateEvaluator<'_>,
        exprs: &[Expression],
        inverted: bool,
    ) -> Option<Predicate> {
        // Partition pruning: rewrite this op over the exact partition-value stat
        // columns (`partitionValues_parsed.<col>`). Only partition columns
        // qualify — their stat is exact (min == max). A non-partition reference
        // (min != max, or no stat) disqualifies the rewrite so we never prune on
        // approximate stats. The kernel guards the result against Remove rows.
        let mut stat_exprs = Vec::with_capacity(exprs.len());
        for expr in exprs {
            let Expression::Column(col) = expr else {
                return None;
            };
            // Type is only used to gate min/max eligibility; partition columns
            // are always eligible, so any type works. Use a broad default.
            let min = predicate_evaluator.get_min_stat(col, &KernelDataType::STRING)?;
            let max = predicate_evaluator.get_max_stat(col, &KernelDataType::STRING)?;
            // Exact only when min and max resolve to the same stat expression —
            // true precisely for partition columns.
            if min != max {
                return None;
            }
            stat_exprs.push(min);
        }
        // Re-wrap the same op over the partition-value stat expressions. When the
        // DataSkippingFilter evaluates this opaque predicate columnarly it calls
        // back into `eval_pred`, which resolves each `partitionValues_parsed.<col>`
        // arg against the stats batch.
        Some(arrow_opaque_predicate(self.clone(), stat_exprs, inverted))
    }
}

impl DataFusionOpaquePredicateOp {
    /// Fallible core of [`Self::eval_pred_scalar`]; any `Err`/`None` from column
    /// resolution short-circuits to "don't know".
    fn eval_scalar_inner(
        &self,
        eval_expr: &ScalarExpressionEvaluator<'_>,
        exprs: &[Expression],
        inverted: bool,
    ) -> DeltaResult<Option<bool>> {
        if exprs.len() != self.columns.len() {
            return Ok(None);
        }
        let mut fields = Vec::with_capacity(exprs.len());
        let mut columns = Vec::with_capacity(exprs.len());

        for (expr, df_name) in exprs.iter().zip(&self.columns) {
            let Some(scalar) = eval_expr(expr) else {
                // Kernel could not supply this column — decline to prune.
                return Ok(None);
            };
            let array = scalar.to_array(1)?;
            fields.push(Field::new(df_name, array.data_type().clone(), true));
            columns.push(array);
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, columns)
            .map_err(|e| KernelError::generic(format!("opaque scalar batch: {e}")))?;
        // The one-row batch is already keyed by DataFusion column names, so
        // evaluate the expr directly rather than re-extracting via `eval_bool`.
        let physical = logical2physical(&self.expr, batch.schema().as_ref());
        let values = physical
            .evaluate(&batch)
            .map_err(|e| KernelError::generic(format!("opaque scalar eval: {e}")))?;
        let array = values
            .into_array(1)
            .map_err(|e| KernelError::generic(format!("opaque scalar eval: {e}")))?;
        let bools = array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| KernelError::generic("opaque predicate not boolean"))?;
        let bools = if inverted {
            compute::not(bools)?
        } else {
            bools.clone()
        };
        if bools.is_null(0) {
            Ok(None)
        } else {
            Ok(Some(bools.value(0)))
        }
    }
}

/// Wraps `expr` as a kernel opaque predicate over the given referenced columns.
///
/// The predicate is always constructed via the arrow adaptor
/// (`Predicate::arrow_opaque`); the kernel `ARROW_HANDLER` evaluator only
/// recognizes adaptor-wrapped ops. When `inverted` is set the wrapping applies
/// a `NOT`, preserving the semantics of an inverted rewrite.
fn arrow_opaque_predicate(
    op: DataFusionOpaquePredicateOp,
    columns: Vec<Expression>,
    inverted: bool,
) -> Predicate {
    use delta_kernel::engine::arrow_expression::opaque::ArrowOpaquePredicate;
    let pred = Predicate::arrow_opaque(op, columns);
    if inverted { Predicate::not(pred) } else { pred }
}

/// Builds an opaque predicate wrapping `op` over its referenced columns (not
/// inverted). Used by the `to_kernel` conversion catch-all.
pub(crate) fn build_opaque_predicate(
    op: DataFusionOpaquePredicateOp,
    columns: Vec<Expression>,
) -> Predicate {
    arrow_opaque_predicate(op, columns, false)
}

/// Attempts to recover a [`DataFusionOpaquePredicateOp`] from a kernel
/// [`Predicate::Opaque`] built via [`build_opaque_predicate`].
///
/// Returns the original DataFusion [`Expr`] when the op is one of ours;
/// `None` for opaque predicates produced by any other engine.
pub(crate) fn recover_datafusion_expr(predicate: &Predicate) -> Option<Expr> {
    use delta_kernel::engine::arrow_expression::opaque::ArrowOpaquePredicateOpAdaptor;

    let Predicate::Opaque(opaque) = predicate else {
        return None;
    };
    // Two-step downcast: kernel stores `Arc<dyn OpaquePredicateOp>`; our op is
    // wrapped in the arrow adaptor, whose inner `dyn ArrowOpaquePredicateOp` is
    // our concrete type.
    let adaptor = opaque
        .op
        .any_ref()
        .downcast_ref::<ArrowOpaquePredicateOpAdaptor>()?;
    let op = adaptor
        .op()
        .any_ref()
        .downcast_ref::<DataFusionOpaquePredicateOp>()?;
    Some(op.expr().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::StringArray;
    use datafusion::arrow::datatypes::DataType as ArrowDataType;
    use datafusion::logical_expr::{col, lit};
    use delta_kernel::expressions::{ColumnName, Scalar};
    use delta_kernel::{AsAny, DynPartialEq};

    fn string_batch(name: &str, values: &[&str]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            name,
            ArrowDataType::Utf8,
            true,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(values.to_vec()))]).unwrap()
    }

    // A predicate DataFusion can express but `to_kernel` cannot:
    // starts_with(part, 'ab').
    fn starts_with_expr() -> Expr {
        use datafusion::functions::expr_fn::starts_with;
        starts_with(col("part"), lit("ab"))
    }

    fn op() -> DataFusionOpaquePredicateOp {
        DataFusionOpaquePredicateOp::new(starts_with_expr(), vec!["part".to_string()])
    }

    #[test]
    fn v5_spike_downcast_round_trip() {
        // The V5 validation: build an opaque predicate wrapping a DF expr and
        // recover the original expr back through the kernel seam via downcast.
        let predicate =
            build_opaque_predicate(op(), vec![Expression::Column(ColumnName::new(["part"]))]);
        let recovered = recover_datafusion_expr(&predicate).expect("recover our op");
        assert_eq!(recovered, starts_with_expr());
    }

    #[test]
    fn recover_returns_none_for_non_opaque() {
        let pred = Predicate::literal(true);
        assert!(recover_datafusion_expr(&pred).is_none());
    }

    #[test]
    fn eval_pred_columnar() {
        let op = op();
        let args = [Expression::Column(ColumnName::new(["part"]))];
        let batch = string_batch("part", &["abc", "xyz", "abz"]);

        let result = op.eval_pred(&args, &batch, false).expect("eval");
        assert_eq!(result, BooleanArray::from(vec![true, false, true]));

        let inverted = op.eval_pred(&args, &batch, true).expect("eval inverted");
        assert_eq!(inverted, BooleanArray::from(vec![false, true, false]));
    }

    #[test]
    fn eval_pred_scalar_matches_columnar() {
        let op = op();
        let columns = [Expression::Column(ColumnName::new(["part"]))];

        let eval = |expr: &Expression| match expr {
            Expression::Column(_) => Some(Scalar::String("abc".to_string())),
            _ => None,
        };
        let eval_expr: &ScalarExpressionEvaluator<'_> = &eval;
        let result = op
            .eval_scalar_inner(eval_expr, &columns, false)
            .expect("scalar eval");
        assert_eq!(result, Some(true));

        let inverted = op
            .eval_scalar_inner(eval_expr, &columns, true)
            .expect("scalar eval inverted");
        assert_eq!(inverted, Some(false));
    }

    #[test]
    fn eval_pred_scalar_missing_column_is_none() {
        let op = op();
        let columns = [Expression::Column(ColumnName::new(["part"]))];
        let eval = |_: &Expression| None;
        let eval_expr: &ScalarExpressionEvaluator<'_> = &eval;
        let result = op
            .eval_scalar_inner(eval_expr, &columns, false)
            .expect("scalar eval");
        assert_eq!(result, None);
    }

    #[test]
    fn eval_pred_resolves_nested_partition_column() {
        // Simulates the partition-stats rewrite: the embedded arg is a nested
        // `partitionValues_parsed.part` reference and the batch carries that
        // struct column. The op must still evaluate `starts_with(part, 'ab')`.
        use datafusion::arrow::array::StructArray;
        use datafusion::arrow::datatypes::Fields;

        let part = Arc::new(StringArray::from(vec!["abc", "xyz"])) as ArrayRef;
        let struct_fields = Fields::from(vec![Field::new("part", ArrowDataType::Utf8, true)]);
        let struct_arr = StructArray::new(struct_fields.clone(), vec![part], None);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "partitionValues_parsed",
            ArrowDataType::Struct(struct_fields),
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(struct_arr)]).unwrap();

        let args = [Expression::Column(ColumnName::new([
            "partitionValues_parsed",
            "part",
        ]))];
        let result = op().eval_pred(&args, &batch, false).expect("eval nested");
        assert_eq!(result, BooleanArray::from(vec![true, false]));
    }

    #[test]
    fn partial_eq_structural() {
        let a = op();
        let b = op();
        assert_eq!(a, b);
        assert!(a.dyn_eq(b.any_ref()));

        let c =
            DataFusionOpaquePredicateOp::new(col("part").eq(lit("ab")), vec!["part".to_string()]);
        assert_ne!(a, c);
    }
}
