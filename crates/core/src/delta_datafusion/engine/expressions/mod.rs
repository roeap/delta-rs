pub(crate) use self::to_datafusion::*;
pub(crate) use self::to_kernel::*;

mod opaque;
mod to_datafusion;
mod to_json;
mod to_kernel;

#[cfg(test)]
mod tests {
    use std::ops::Not;

    use datafusion::logical_expr::{col, lit};
    use delta_kernel::schema::DataType;

    use super::*;

    #[test]
    fn test_roundtrip_simple_and() {
        let df_expr = col("a").eq(lit(1)).and(col("b").eq(lit(2)));
        let delta_expr = to_delta_expression(&df_expr).unwrap();
        let df_expr_roundtrip = to_datafusion_expr(&delta_expr, &DataType::BOOLEAN).unwrap();
        assert_eq!(df_expr, df_expr_roundtrip);
    }

    #[test]
    fn test_roundtrip_nested_and() {
        let df_expr = col("a")
            .eq(lit(1))
            .and(col("b").eq(lit(2)))
            .and(col("c").eq(lit(3)))
            .and(col("d").eq(lit(4)));
        let delta_expr = to_delta_expression(&df_expr).unwrap();
        let df_expr_roundtrip = to_datafusion_expr(&delta_expr, &DataType::BOOLEAN).unwrap();
        assert_eq!(df_expr, df_expr_roundtrip);
    }

    #[test]
    fn test_roundtrip_mixed_and_or() {
        let df_expr = col("a")
            .eq(lit(1))
            .and(col("b").eq(lit(2)))
            .or(col("c").eq(lit(3)).and(col("d").eq(lit(4))));
        let delta_expr = to_delta_expression(&df_expr).unwrap();
        let df_expr_roundtrip = to_datafusion_expr(&delta_expr, &DataType::BOOLEAN).unwrap();
        assert_eq!(df_expr, df_expr_roundtrip);
    }

    #[test]
    fn test_roundtrip_unary() {
        let df_expr = !col("a").eq(lit(1));
        let delta_expr = to_delta_expression(&df_expr).unwrap();
        let df_expr_roundtrip = to_datafusion_expr(&delta_expr, &DataType::BOOLEAN).unwrap();
        assert_eq!(df_expr, df_expr_roundtrip);
    }

    #[test]
    fn test_roundtrip_is_null() {
        let df_expr = col("a").is_null();
        let delta_expr = to_delta_expression(&df_expr).unwrap();
        let df_expr_roundtrip = to_datafusion_expr(&delta_expr, &DataType::BOOLEAN).unwrap();
        assert_eq!(df_expr, df_expr_roundtrip);
    }

    #[test]
    fn test_roundtrip_binary_ops() {
        let df_expr = col("a") + col("b") * col("c");
        let delta_expr = to_delta_expression(&df_expr).unwrap();
        let df_expr_roundtrip = to_datafusion_expr(&delta_expr, &DataType::BOOLEAN).unwrap();
        assert_eq!(df_expr, df_expr_roundtrip);
    }

    #[test]
    fn test_roundtrip_comparison_ops() {
        let df_expr = col("a").gt(col("b")).and(col("c").gt(col("d")).not());
        let delta_expr = to_delta_expression(&df_expr).unwrap();
        let df_expr_roundtrip = to_datafusion_expr(&delta_expr, &DataType::BOOLEAN).unwrap();
        assert_eq!(df_expr, df_expr_roundtrip);
    }

    #[test]
    fn test_roundtrip_opaque_predicate() {
        use datafusion::functions::expr_fn::starts_with;
        // An untranslatable boolean predicate round-trips through the opaque
        // seam: DF expr -> kernel opaque predicate -> original DF expr.
        let df_expr = starts_with(col("part"), lit("ab"));
        let delta_pred = try_opaque_predicate(&df_expr).expect("opaque wrap");
        let df_expr_roundtrip = to_datafusion_expr(
            &delta_kernel::expressions::Expression::Predicate(Box::new(delta_pred)),
            &DataType::BOOLEAN,
        )
        .unwrap();
        assert_eq!(df_expr, df_expr_roundtrip);
    }
}
