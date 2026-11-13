//! Casts a projected value to its declared type where only nested nullability differs.
//!
//! Kernel declares nested fields as non-nullable under a nullable parent, for example the
//! `deletionVector` struct in the metadata scan's `file_action_key`, whose children are set
//! whenever it is. DataFusion builds such values with nullable children (`named_struct`, `CASE`)
//! and its `CAST` refuses to tighten a nested field's nullability. Arrow's cast allows it and
//! checks the data instead, so [`declared_cast`] wraps that cast as a scalar function: a NULL
//! under a non-nullable nested field is an execution error, not a refused plan. Struct NULLs are
//! unioned into their children first, as [`super::scan::declared_type`] does after a parquet
//! read.

use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, AsArray as _};
use datafusion::arrow::compute::{CastOptions, cast_with_options};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::{Result as DFResult, exec_err};
use datafusion::logical_expr::{
    ColumnarValue, Expr, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use delta_kernel::engine::arrow_data::fix_nested_null_masks;

use super::scan::expr_adapter::relax_nested_nullability;

/// Whether casting `from` to `to` only tightens or loosens nested nullability.
pub(crate) fn differs_only_in_nested_nullability(from: &DataType, to: &DataType) -> bool {
    from != to && relax_nested_nullability(from) == relax_nested_nullability(to)
}

/// `expr` cast to `target` with arrow's cast; see the module docs.
pub(crate) fn declared_cast(expr: Expr, target: DataType) -> Expr {
    ScalarUDF::new_from_impl(DeclaredCast::new(target)).call(vec![expr])
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct DeclaredCast {
    target: DataType,
    signature: Signature,
}

impl DeclaredCast {
    fn new(target: DataType) -> Self {
        Self {
            target,
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for DeclaredCast {
    fn name(&self) -> &str {
        "declared_cast"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(self.target.clone())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let [input] = args.args.as_slice() else {
            return exec_err!(
                "declared_cast expects one argument, got {}",
                args.args.len()
            );
        };
        let array = input.to_array(args.number_rows)?;
        let array = match array.as_struct_opt() {
            Some(array) => Arc::new(fix_nested_null_masks(array.clone())) as ArrayRef,
            None => array,
        };
        let options = CastOptions {
            safe: false,
            ..Default::default()
        };
        Ok(ColumnarValue::Array(cast_with_options(
            &array,
            &self.target,
            &options,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::{Field, Fields};

    use super::*;

    fn dv(children_nullable: bool) -> DataType {
        DataType::Struct(Fields::from(vec![
            Field::new("storageType", DataType::Utf8, children_nullable),
            Field::new("offset", DataType::Int32, true),
        ]))
    }

    #[test]
    fn detects_nested_nullability_only_differences() {
        assert!(differs_only_in_nested_nullability(&dv(true), &dv(false)));
        assert!(!differs_only_in_nested_nullability(&dv(false), &dv(false)));
        assert!(!differs_only_in_nested_nullability(
            &DataType::Int32,
            &DataType::Int64
        ));
        let renamed = DataType::Struct(Fields::from(vec![
            Field::new("type", DataType::Utf8, false),
            Field::new("offset", DataType::Int32, true),
        ]));
        assert!(!differs_only_in_nested_nullability(&dv(true), &renamed));
    }
}
