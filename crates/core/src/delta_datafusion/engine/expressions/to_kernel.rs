use datafusion::common::tree_node::{Transformed, TreeNode as _};
use datafusion::common::{Result, ScalarValue, plan_datafusion_err, plan_err};
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::utils::{conjunction, disjunction};
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use delta_kernel::expressions::{
    BinaryExpression, BinaryExpressionOp, BinaryPredicate, BinaryPredicateOp, ColumnName,
    DecimalData, Expression, JunctionPredicate, JunctionPredicateOp, Predicate, Scalar,
    UnaryPredicate, UnaryPredicateOp,
};
use delta_kernel::schema::{DataType, DecimalType, PrimitiveType};

use crate::delta_datafusion::engine::expressions::opaque::{
    DataFusionOpaquePredicateOp, build_opaque_predicate,
};
use crate::kernel::scalars::ScalarExt;

/// Converts a DataFusion expression to a Delta predicate.
///
/// If the expression converts to a Delta predicate, returns it directly.
/// Otherwise, wraps the expression as a boolean expression predicate.
///
/// This is the *structural* conversion only. Expressions the kernel predicate
/// model cannot represent (UDFs, `LIKE`, `CASE`, regex, most scalar functions…)
/// are an error here; callers that want partition pruning for such predicates
/// invoke [`try_opaque_predicate`] as an explicit fallback *after* their own
/// safety guards (e.g. schema-override type-mismatch checks) — opaque wrapping
/// is deliberately not folded into this function so it cannot bypass them.
pub(crate) fn to_delta_predicate(expr: &Expr) -> Result<Predicate> {
    match to_delta_expression(&normalize_delta_predicate_expr(expr)?)? {
        Expression::Predicate(pred) => Ok(pred.as_ref().clone()),
        expr => Ok(Predicate::BooleanExpression(expr)),
    }
}

/// Attempts to wrap an untranslatable boolean DataFusion predicate as a kernel
/// opaque predicate that evaluates the original `Expr`, extending partition
/// pruning to arbitrary DataFusion predicates while never wrongly pruning.
///
/// Returns `None` (leaving the caller to treat the predicate as unsupported)
/// unless ALL of the following hold:
/// - the node is plausibly boolean-valued (see [`is_probably_boolean`]);
/// - the expression is non-volatile (`random()` etc. must never prune);
/// - every column reference converts to a kernel [`ColumnName`].
///
/// Callers MUST run any correctness guards (e.g. schema-override type mismatch)
/// before invoking this — the op evaluates the DataFusion `Expr` verbatim, so a
/// predicate that is unsafe to push structurally is equally unsafe to push as
/// opaque. Opaque predicates are always constructed via the arrow adaptor so the
/// kernel `ARROW_HANDLER` evaluator recognizes them.
pub(crate) fn try_opaque_predicate(expr: &Expr) -> Option<Predicate> {
    let expr = normalize_delta_predicate_expr(expr).ok()?;
    let expr = &expr;
    if !is_probably_boolean(expr) || expr.is_volatile() {
        return None;
    }

    // Collect the referenced columns as (DataFusion leaf name, kernel column
    // expression) pairs, kept in a stable order so the op can align embedded
    // args with the names it rebuilds the evaluation batch under. Reuse the
    // structural column-conversion path so nested/field access stays consistent.
    let mut column_refs: Vec<_> = expr.column_refs().into_iter().collect();
    column_refs.sort_by(|a, b| a.name.cmp(&b.name));

    let mut names = Vec::with_capacity(column_refs.len());
    let mut columns = Vec::with_capacity(column_refs.len());
    for column in column_refs {
        match to_delta_expression(&Expr::Column(column.clone())) {
            Ok(kernel_expr @ Expression::Column(_)) => {
                names.push(column.name.clone());
                columns.push(kernel_expr);
            }
            _ => return None,
        }
    }

    let op = DataFusionOpaquePredicateOp::new(expr.clone(), names);
    Some(build_opaque_predicate(op, columns))
}

/// Heuristic: is this DataFusion expression plausibly boolean-valued, so that
/// wrapping it as a boolean opaque predicate is meaningful?
///
/// We only have the logical `Expr` here (no schema), so this restricts to node
/// kinds that are obviously boolean-returning. Non-boolean nodes (arithmetic,
/// projections, bare columns) are NOT wrapped — they belong in expression
/// position, which v1 leaves as a hard error.
fn is_probably_boolean(expr: &Expr) -> bool {
    match expr {
        Expr::Like(_)
        | Expr::SimilarTo(_)
        | Expr::IsTrue(_)
        | Expr::IsFalse(_)
        | Expr::IsUnknown(_)
        | Expr::IsNotTrue(_)
        | Expr::IsNotFalse(_)
        | Expr::IsNotUnknown(_)
        | Expr::IsNull(_)
        | Expr::IsNotNull(_)
        | Expr::Between(_)
        | Expr::InList(_)
        | Expr::InSubquery(_) => true,
        Expr::Not(inner) => is_probably_boolean(inner),
        Expr::BinaryExpr(BinaryExpr { op, .. }) => matches!(
            op,
            Operator::Eq
                | Operator::NotEq
                | Operator::Lt
                | Operator::LtEq
                | Operator::Gt
                | Operator::GtEq
                | Operator::And
                | Operator::Or
                | Operator::IsDistinctFrom
                | Operator::IsNotDistinctFrom
                | Operator::RegexMatch
                | Operator::RegexIMatch
                | Operator::RegexNotMatch
                | Operator::RegexNotIMatch
                | Operator::LikeMatch
                | Operator::ILikeMatch
                | Operator::NotLikeMatch
                | Operator::NotILikeMatch
        ),
        // A boolean-returning scalar function (`starts_with`, `ends_with`,
        // `contains`, `regexp_like`, custom boolean UDFs, …). We cannot check
        // the return type without a schema, so accept scalar functions and rely
        // on the columnar evaluator: a non-boolean result errors and, via the
        // "never fail the scan" contract, resolves to "don't know".
        Expr::ScalarFunction(_) => true,
        // A CASE with a boolean output can only be told apart with a schema;
        // accept it for the same reason as scalar functions.
        Expr::Case(_) => true,
        _ => false,
    }
}

fn normalize_delta_predicate_expr(expr: &Expr) -> Result<Expr> {
    let transformed = expr.clone().transform_up(|expr| match expr {
        Expr::InList(in_list) => Ok(match rewrite_in_list_expr_for_kernel(&in_list) {
            Some(lowered) => Transformed::yes(lowered),
            None => Transformed::no(Expr::InList(in_list)),
        }),
        other => Ok(Transformed::no(other)),
    })?;

    Ok(transformed.data)
}

/// Rewrites supported `IN` / `NOT IN` expressions into conjunctions or
/// disjunctions that can be converted into Delta kernel predicates.
///
/// Returns `None` when any list item cannot be rewritten without changing the
/// meaning of the predicate. In that case callers should leave the original
/// `Expr::InList` unchanged and let downstream conversion decide how to handle it.
fn rewrite_in_list_expr_for_kernel(in_list: &InList) -> Option<Expr> {
    if in_list.list.is_empty() {
        return Some(Expr::Literal(
            ScalarValue::Boolean(Some(in_list.negated)),
            None,
        ));
    }

    let list = in_list
        .list
        .iter()
        .map(|item| match item {
            Expr::Literal(value, _) if !value.is_null() => Some(item.clone()),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;

    let lowered_terms = list.into_iter().map(|item| {
        if in_list.negated {
            in_list.expr.as_ref().clone().not_eq(item)
        } else {
            in_list.expr.as_ref().clone().eq(item)
        }
    });

    if in_list.negated {
        conjunction(lowered_terms)
    } else {
        disjunction(lowered_terms)
    }
}

/// Converts a DataFusion expression to a Delta kernel expression.
pub(crate) fn to_delta_expression(expr: &Expr) -> Result<Expression> {
    match expr {
        Expr::Column(column) => Ok(Expression::Column(ColumnName::new([column.name.as_str()]))),
        Expr::Literal(scalar, _meta) => {
            Ok(Expression::Literal(datafusion_scalar_to_scalar(scalar)?))
        }
        Expr::BinaryExpr(BinaryExpr {
            op: op @ (Operator::And | Operator::Or),
            ..
        }) => {
            let preds = flatten_junction_expr(expr, *op)?;
            Ok(Expression::Predicate(Box::new(Predicate::Junction(
                JunctionPredicate {
                    op: to_junction_op(*op),
                    preds,
                },
            ))))
        }
        Expr::BinaryExpr(BinaryExpr {
            op: op @ (Operator::Eq | Operator::Lt | Operator::Gt | Operator::IsDistinctFrom),
            left,
            right,
        }) => Ok(Expression::Predicate(Box::new(Predicate::Binary(
            BinaryPredicate {
                left: Box::new(to_delta_expression(left.as_ref())?),
                op: to_binary_predicate_op(*op)?,
                right: Box::new(to_delta_expression(right.as_ref())?),
            },
        )))),
        Expr::BinaryExpr(BinaryExpr {
            op: op @ (Operator::NotEq | Operator::LtEq | Operator::GtEq),
            left,
            right,
        }) => {
            let inverted = match op {
                Operator::NotEq => Operator::Eq,
                Operator::LtEq => Operator::Gt,
                Operator::GtEq => Operator::Lt,
                _ => unreachable!(),
            };
            Ok(Expression::Predicate(Box::new(Predicate::Not(Box::new(
                Predicate::Binary(BinaryPredicate {
                    left: Box::new(to_delta_expression(left.as_ref())?),
                    op: to_binary_predicate_op(inverted)?,
                    right: Box::new(to_delta_expression(right.as_ref())?),
                }),
            )))))
        }
        Expr::BinaryExpr(BinaryExpr {
            op: Operator::IsNotDistinctFrom,
            left,
            right,
        }) => Ok(Expression::Predicate(Box::new(Predicate::Not(Box::new(
            Predicate::Binary(BinaryPredicate {
                left: Box::new(to_delta_expression(left.as_ref())?),
                op: to_binary_predicate_op(Operator::IsDistinctFrom)?,
                right: Box::new(to_delta_expression(right.as_ref())?),
            }),
        ))))),
        Expr::BinaryExpr(BinaryExpr { op, left, right }) => {
            Ok(Expression::Binary(BinaryExpression {
                left: Box::new(to_delta_expression(left.as_ref())?),
                op: to_binary_op(*op)?,
                right: Box::new(to_delta_expression(right.as_ref())?),
            }))
        }
        Expr::IsNull(expr) => Ok(Expression::Predicate(Box::new(Predicate::Unary(
            UnaryPredicate {
                op: UnaryPredicateOp::IsNull,
                expr: Box::new(to_delta_expression(expr.as_ref())?),
            },
        )))),
        Expr::IsNotNull(expr) => Ok(Expression::Predicate(Box::new(Predicate::Not(Box::new(
            Predicate::Unary(UnaryPredicate {
                op: UnaryPredicateOp::IsNull,
                expr: Box::new(to_delta_expression(expr.as_ref())?),
            }),
        ))))),
        Expr::Not(expr) => Ok(Expression::Predicate(Box::new(Predicate::Not(Box::new(
            Predicate::BooleanExpression(to_delta_expression(expr.as_ref())?),
        ))))),
        Expr::Between(between) => {
            let expr = to_delta_expression(&between.expr)?;
            let expression = Predicate::Junction(JunctionPredicate {
                op: JunctionPredicateOp::Or,
                preds: vec![
                    Predicate::Binary(BinaryPredicate {
                        left: Box::new(expr.clone()),
                        op: BinaryPredicateOp::LessThan,
                        right: Box::new(to_delta_expression(&between.low)?),
                    }),
                    Predicate::Binary(BinaryPredicate {
                        left: Box::new(expr),
                        op: BinaryPredicateOp::GreaterThan,
                        right: Box::new(to_delta_expression(&between.high)?),
                    }),
                ],
            });
            if between.negated {
                Ok(Expression::Predicate(Box::new(expression)))
            } else {
                Ok(Expression::Predicate(Box::new(Predicate::Not(Box::new(
                    expression,
                )))))
            }
        }
        Expr::ScalarFunction(scalar_fn) => {
            if scalar_fn.name() == "get_field" {
                if scalar_fn.args.len() != 2 {
                    return plan_err!(
                        "get_field function requires exactly 2 arguments, got {}",
                        scalar_fn.args.len()
                    );
                }

                let field_name = match &scalar_fn.args[1] {
                    Expr::Literal(name, _) => name.to_string(),
                    other => other.schema_name().to_string(),
                };

                if let Expression::Column(ref col_name) = to_delta_expression(&scalar_fn.args[0])? {
                    return Ok(Expression::Column(
                        col_name.join(&ColumnName::new([field_name.as_str()])),
                    ));
                }
            }
            plan_err!(
                "Scalar function not supported in Delta Kernel expressions: {:?}",
                scalar_fn
            )
        }
        _ => plan_err!("Cannot convert to kernel expression: {:?}", expr),
    }
}

pub(crate) fn datafusion_scalar_to_scalar(scalar: &ScalarValue) -> Result<Scalar> {
    match scalar {
        ScalarValue::Boolean(maybe_value) => match maybe_value {
            Some(value) => Ok(Scalar::Boolean(*value)),
            None => Ok(Scalar::Null(DataType::BOOLEAN)),
        },
        ScalarValue::Utf8(maybe_value)
        | ScalarValue::LargeUtf8(maybe_value)
        | ScalarValue::Utf8View(maybe_value) => match maybe_value {
            Some(value) => Ok(Scalar::String(value.clone())),
            None => Ok(Scalar::Null(DataType::STRING)),
        },
        ScalarValue::Int8(maybe_value) => match maybe_value {
            Some(value) => Ok(Scalar::Byte(*value)),
            None => Ok(Scalar::Null(DataType::BYTE)),
        },
        ScalarValue::Int16(maybe_value) => match maybe_value {
            Some(value) => Ok(Scalar::Short(*value)),
            None => Ok(Scalar::Null(DataType::SHORT)),
        },
        ScalarValue::Int32(maybe_value) => match maybe_value {
            Some(value) => Ok(Scalar::Integer(*value)),
            None => Ok(Scalar::Null(DataType::INTEGER)),
        },
        ScalarValue::Int64(maybe_value) => match maybe_value {
            Some(value) => Ok(Scalar::Long(*value)),
            None => Ok(Scalar::Null(DataType::LONG)),
        },
        ScalarValue::Float32(maybe_value) => match maybe_value {
            Some(value) => Ok(Scalar::Float(*value)),
            None => Ok(Scalar::Null(DataType::FLOAT)),
        },
        ScalarValue::Float64(maybe_value) => match maybe_value {
            Some(value) => Ok(Scalar::Double(*value)),
            None => Ok(Scalar::Null(DataType::DOUBLE)),
        },
        ScalarValue::TimestampMicrosecond(maybe_value, Some(_)) => match maybe_value {
            Some(value) => Ok(Scalar::Timestamp(*value)),
            None => Ok(Scalar::Null(DataType::TIMESTAMP)),
        },
        #[cfg(feature = "nanosecond-timestamps")]
        ScalarValue::TimestampNanosecond(maybe_value, Some(_)) => match maybe_value {
            Some(value) => Ok(Scalar::TimestampNanos(*value)),
            None => Ok(Scalar::Null(DataType::TIMESTAMP_NANOS)),
        },
        ScalarValue::TimestampMicrosecond(maybe_value, None) => match maybe_value {
            Some(value) => Ok(Scalar::TimestampNtz(*value)),
            None => Ok(Scalar::Null(DataType::TIMESTAMP_NTZ)),
        },
        ScalarValue::Date32(maybe_value) => match maybe_value {
            Some(value) => Ok(Scalar::Date(*value)),
            None => Ok(Scalar::Null(DataType::DATE)),
        },
        ScalarValue::Binary(maybe_value)
        | ScalarValue::LargeBinary(maybe_value)
        | ScalarValue::BinaryView(maybe_value)
        | ScalarValue::FixedSizeBinary(_, maybe_value) => match maybe_value {
            Some(value) => Ok(Scalar::Binary(value.clone())),
            None => Ok(Scalar::Null(DataType::BINARY)),
        },
        ScalarValue::Decimal128(maybe_value, precision, scale) => match maybe_value {
            Some(value) => Ok(Scalar::Decimal(
                DecimalData::try_new(
                    *value,
                    DecimalType::try_new(*precision, *scale as u8)
                        .map_err(|e| plan_datafusion_err!("{e}"))?,
                )
                .map_err(|e| plan_datafusion_err!("{e}"))?,
            )),
            None => Ok(Scalar::Null(DataType::Primitive(PrimitiveType::Decimal(
                DecimalType::try_new(*precision, *scale as u8)
                    .map_err(|e| plan_datafusion_err!("{e}"))?,
            )))),
        },
        ScalarValue::Struct(data) => Ok(Scalar::from_array(data.as_ref(), 0)
            .ok_or_else(|| plan_datafusion_err!("Struct to kernel scalar conversion failed."))?),
        ScalarValue::Dictionary(_, value) => datafusion_scalar_to_scalar(value.as_ref()),
        _ => plan_err!("Cannot convert to kernel scalar value: {:?}", scalar),
    }
}

fn to_binary_predicate_op(op: Operator) -> Result<BinaryPredicateOp> {
    match op {
        Operator::Eq => Ok(BinaryPredicateOp::Equal),
        Operator::Lt => Ok(BinaryPredicateOp::LessThan),
        Operator::Gt => Ok(BinaryPredicateOp::GreaterThan),
        Operator::IsDistinctFrom => Ok(BinaryPredicateOp::Distinct),
        _ => plan_err!("Operator not supported in Delta Kernel: {:?}", op),
    }
}

fn to_binary_op(op: Operator) -> Result<BinaryExpressionOp> {
    match op {
        Operator::Plus => Ok(BinaryExpressionOp::Plus),
        Operator::Minus => Ok(BinaryExpressionOp::Minus),
        Operator::Multiply => Ok(BinaryExpressionOp::Multiply),
        Operator::Divide => Ok(BinaryExpressionOp::Divide),
        _ => plan_err!("Operator not supported in Delta Kernel: {:?}", op),
    }
}

/// Helper function to flatten nested AND/OR expressions into a single junction expression
fn flatten_junction_expr(expr: &Expr, target_op: Operator) -> Result<Vec<Predicate>> {
    match expr {
        Expr::BinaryExpr(BinaryExpr { op, left, right }) if *op == target_op => {
            let mut left_exprs = flatten_junction_expr(left.as_ref(), target_op)?;
            let mut right_exprs = flatten_junction_expr(right.as_ref(), target_op)?;
            left_exprs.append(&mut right_exprs);
            Ok(left_exprs)
        }
        _ => {
            let delta_expr = to_delta_predicate(expr)?;
            Ok(vec![delta_expr])
        }
    }
}

fn to_junction_op(op: Operator) -> JunctionPredicateOp {
    match op {
        Operator::And => JunctionPredicateOp::And,
        Operator::Or => JunctionPredicateOp::Or,
        _ => unimplemented!("Unsupported operator: {:?}", op),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::{
        common::Column,
        functions::core::expr_ext::FieldAccessor,
        logical_expr::{col, lit},
    };
    use delta_kernel::expressions::{BinaryExpressionOp, JunctionPredicateOp, Scalar};

    fn assert_junction_expr(
        expr: &Expr,
        expected_op: JunctionPredicateOp,
        expected_children: usize,
    ) {
        let delta_expr = to_delta_expression(expr).unwrap();
        match delta_expr {
            Expression::Predicate(predicate) => match predicate.as_ref() {
                Predicate::Junction(junction) => {
                    assert_eq!(junction.op, expected_op);
                    assert_eq!(junction.preds.len(), expected_children);
                }
                _ => panic!("Expected Junction predicate, got {:?}", predicate),
            },
            _ => panic!("Expected Junction expression, got {:?}", delta_expr),
        }
    }

    fn assert_string_equality_predicate(
        predicate: &Predicate,
        expected_column: &str,
        expected: &str,
    ) {
        match predicate {
            Predicate::Binary(binary) => {
                assert_eq!(binary.op, BinaryPredicateOp::Equal);
                match binary.left.as_ref() {
                    Expression::Column(name) => assert_eq!(name.to_string(), expected_column),
                    other => panic!("Expected column expression, got {:?}", other),
                }
                match binary.right.as_ref() {
                    Expression::Literal(Scalar::String(value)) => assert_eq!(value, expected),
                    other => panic!("Expected string literal, got {:?}", other),
                }
            }
            other => panic!("Expected binary predicate, got {:?}", other),
        }
    }

    fn assert_string_inequality_predicate(
        predicate: &Predicate,
        expected_column: &str,
        expected: &str,
    ) {
        match predicate {
            Predicate::Not(inner) => {
                assert_string_equality_predicate(inner.as_ref(), expected_column, expected)
            }
            other => panic!("Expected NOT predicate, got {:?}", other),
        }
    }

    #[test]
    fn test_simple_and() {
        let expr = col("a").eq(lit(1)).and(col("b").eq(lit(2)));
        assert_junction_expr(&expr, JunctionPredicateOp::And, 2);
    }

    #[test]
    fn test_simple_or() {
        let expr = col("a").eq(lit(1)).or(col("b").eq(lit(2)));
        assert_junction_expr(&expr, JunctionPredicateOp::Or, 2);
    }

    #[test]
    fn test_in_list_rewrites_to_or_of_equalities() {
        let expr = col("part").in_list(vec![lit("a"), lit("c")], false);
        let delta_predicate = to_delta_predicate(&expr).unwrap();

        match delta_predicate {
            Predicate::Junction(junction) => {
                assert_eq!(junction.op, JunctionPredicateOp::Or);
                assert_eq!(junction.preds.len(), 2);
                assert_string_equality_predicate(&junction.preds[0], "part", "a");
                assert_string_equality_predicate(&junction.preds[1], "part", "c");
            }
            other => panic!("Expected OR junction, got {:?}", other),
        }
    }

    #[test]
    fn test_not_in_list_rewrites_to_and_of_inequalities() {
        let expr = col("part").in_list(vec![lit("a"), lit("c")], true);
        let delta_predicate = to_delta_predicate(&expr).unwrap();

        match delta_predicate {
            Predicate::Junction(junction) => {
                assert_eq!(junction.op, JunctionPredicateOp::And);
                assert_eq!(junction.preds.len(), 2);
                assert_string_inequality_predicate(&junction.preds[0], "part", "a");
                assert_string_inequality_predicate(&junction.preds[1], "part", "c");
            }
            other => panic!("Expected AND junction, got {:?}", other),
        }
    }

    #[test]
    fn test_empty_in_list_rewrites_to_boolean_constant() {
        let expr = col("part").in_list(Vec::<Expr>::new(), false);
        assert_eq!(
            to_delta_predicate(&expr).unwrap(),
            Predicate::BooleanExpression(Expression::Literal(Scalar::Boolean(false)))
        );

        let negated = col("part").in_list(Vec::<Expr>::new(), true);
        assert_eq!(
            to_delta_predicate(&negated).unwrap(),
            Predicate::BooleanExpression(Expression::Literal(Scalar::Boolean(true)))
        );
    }

    #[test]
    fn test_normalize_delta_predicate_expr_leaves_non_literal_in_list_unchanged() {
        let expr = col("part").in_list(vec![lit("a"), col("other")], false);

        assert_eq!(normalize_delta_predicate_expr(&expr).unwrap(), expr);
    }

    #[test]
    fn test_normalize_delta_predicate_expr_leaves_not_in_list_with_null_unchanged() {
        let expr = col("part").in_list(vec![lit("a"), lit(ScalarValue::Utf8(None))], true);

        assert_eq!(normalize_delta_predicate_expr(&expr).unwrap(), expr);
    }

    #[test]
    fn test_structural_conversion_rejects_untranslatable() {
        use datafusion::functions::expr_fn::starts_with;
        // The structural conversion no longer auto-wraps; untranslatable
        // predicates are an error here (opaque wrapping is an explicit caller
        // fallback via `try_opaque_predicate`).
        let expr = starts_with(col("part"), lit("ab"));
        assert!(to_delta_predicate(&expr).is_err());
    }

    #[test]
    fn test_opaque_wraps_untranslatable_boolean_predicate() {
        use datafusion::functions::expr_fn::starts_with;
        // `starts_with(part, 'ab')` has no kernel structural representation, so
        // `try_opaque_predicate` wraps it as an opaque predicate.
        let expr = starts_with(col("part"), lit("ab"));
        let predicate = try_opaque_predicate(&expr).expect("opaque wrap");
        assert!(matches!(predicate, Predicate::Opaque(_)));
    }

    #[test]
    fn test_opaque_wraps_comparison_over_untranslatable_scalar_fn() {
        use datafusion::functions::expr_fn::substr;
        // `substr(part, 1) = 'ab'` — the comparison is structural but its left
        // operand is an untranslatable scalar function, so the whole predicate
        // becomes opaque.
        let expr = substr(col("part"), lit(1i64)).eq(lit("ab"));
        let predicate = try_opaque_predicate(&expr).expect("opaque wrap");
        assert!(matches!(predicate, Predicate::Opaque(_)));
    }

    #[test]
    fn test_volatile_expr_not_wrapped() {
        use datafusion::functions::expr_fn::random;
        // `random() > 0.5` is boolean but volatile; it must never prune.
        let expr = random().gt(lit(0.5));
        assert!(to_delta_predicate(&expr).is_err());
        assert!(try_opaque_predicate(&expr).is_none());
    }

    #[test]
    fn test_non_boolean_node_kind_not_wrapped() {
        // A non-boolean node kind (arithmetic) is never wrapped as an opaque
        // *predicate*, even if it happens to reference untranslatable children.
        // (`try_opaque_predicate` is only ever called from predicate position;
        // this guards the boolean-kind heuristic itself.)
        let expr = col("a") + col("b");
        assert!(try_opaque_predicate(&expr).is_none());
    }

    #[test]
    fn test_boolean_scalar_fn_wrapped_relies_on_evaluator() {
        use datafusion::functions::expr_fn::substr;
        // A bare scalar function in predicate position IS wrapped: we cannot
        // prove its return type without a schema, so we accept it and rely on
        // the columnar evaluator to degrade a non-boolean result to "don't
        // know" (documented contract).
        let expr = substr(col("part"), lit(1i64));
        let predicate = try_opaque_predicate(&expr).expect("opaque wrap");
        assert!(matches!(predicate, Predicate::Opaque(_)));
    }

    #[test]
    fn test_field_access() {
        let expr = col("a").field("b");
        assert_eq!(
            to_delta_expression(&expr).unwrap(),
            Expression::Column(ColumnName::new(["a", "b"]))
        );
        let expr = col("a").field("b").field("c");
        assert_eq!(
            to_delta_expression(&expr).unwrap(),
            Expression::Column(ColumnName::new(["a", "b", "c"]))
        );

        let expr = col("a").field("b").field("c").eq(lit(10));
        let delta_expr = to_delta_expression(&expr).unwrap();
        match delta_expr {
            Expression::Predicate(predicate) => match predicate.as_ref() {
                Predicate::Binary(binary) => {
                    assert_eq!(binary.op, BinaryPredicateOp::Equal);
                    match binary.left.as_ref() {
                        Expression::Column(name) => {
                            assert_eq!(name.to_string(), "a.b.c")
                        }
                        _ => panic!("Expected Column expression in left operand"),
                    }
                    match *binary.right.as_ref() {
                        Expression::Literal(Scalar::Integer(value)) => assert_eq!(value, 10),
                        _ => panic!("Expected Integer literal in right operand"),
                    }
                }
                _ => panic!("Expected Binary predicate, got {:?}", predicate),
            },
            _ => panic!("Expected Binary expression, got {:?}", delta_expr),
        }
    }

    #[test]
    fn test_nested_and() {
        let expr = col("a")
            .eq(lit(1))
            .and(col("b").eq(lit(2)))
            .and(col("c").eq(lit(3)))
            .and(col("d").eq(lit(4)));
        assert_junction_expr(&expr, JunctionPredicateOp::And, 4);
    }

    #[test]
    fn test_nested_or() {
        let expr = col("a")
            .eq(lit(1))
            .or(col("b").eq(lit(2)))
            .or(col("c").eq(lit(3)))
            .or(col("d").eq(lit(4)));
        assert_junction_expr(&expr, JunctionPredicateOp::Or, 4);
    }

    #[test]
    fn test_mixed_nested_and_or() {
        // (a AND b) OR (c AND d)
        let left = col("a").eq(lit(1)).and(col("b").eq(lit(2)));
        let right = col("c").eq(lit(3)).and(col("d").eq(lit(4)));
        let expr = left.or(right);

        let delta_expr = to_delta_expression(&expr).unwrap();
        match delta_expr {
            Expression::Predicate(predicate) => match predicate.as_ref() {
                Predicate::Junction(junction) => {
                    assert_eq!(junction.op, JunctionPredicateOp::Or);
                    assert_eq!(junction.preds.len(), 2);

                    // Check that both children are AND junctions
                    for child in &junction.preds {
                        match child {
                            Predicate::Junction(binary) => {
                                assert_eq!(binary.op, JunctionPredicateOp::And);
                            }
                            _ => panic!("Expected Binary expression in child: {:?}", child),
                        }
                    }
                }
                _ => panic!("Expected Junction predicate, got {:?}", predicate),
            },
            _ => panic!("Expected Junction expression"),
        }
    }

    #[test]
    fn test_deeply_nested_and() {
        // (((a AND b) AND c) AND d)
        let expr = col("a")
            .eq(lit(1))
            .and(col("b").eq(lit(2)))
            .and(col("c").eq(lit(3)))
            .and(col("d").eq(lit(4)));
        assert_junction_expr(&expr, JunctionPredicateOp::And, 4);
    }

    #[test]
    fn test_complex_expression() {
        // (a AND b) OR ((c AND d) AND e)
        let left = col("a").eq(lit(1)).and(col("b").eq(lit(2)));
        let right = col("c")
            .eq(lit(3))
            .and(col("d").eq(lit(4)))
            .and(col("e").eq(lit(5)));
        let expr = left.or(right);

        let delta_expr = to_delta_expression(&expr).unwrap();
        match delta_expr {
            Expression::Predicate(predicate) => match predicate.as_ref() {
                Predicate::Junction(junction) => {
                    assert_eq!(junction.op, JunctionPredicateOp::Or);
                    assert_eq!(junction.preds.len(), 2);

                    // First child should be an AND with 2 expressions
                    match &junction.preds[0] {
                        Predicate::Junction(child_junction) => {
                            assert_eq!(child_junction.op, JunctionPredicateOp::And);
                            assert_eq!(child_junction.preds.len(), 2);
                        }
                        _ => panic!("Expected Junction expression in first child"),
                    }

                    // Second child should be an AND with 3 expressions
                    match &junction.preds[1] {
                        Predicate::Junction(child_junction) => {
                            assert_eq!(child_junction.op, JunctionPredicateOp::And);
                            assert_eq!(child_junction.preds.len(), 3);
                        }
                        _ => panic!("Expected Junction expression in second child"),
                    }
                }
                _ => panic!("Expected Junction predicate, got {:?}", predicate),
            },
            _ => panic!("Expected Junction expression"),
        }
    }

    #[test]
    fn test_column_expression() {
        let expr = col("test_column");
        let delta_expr = to_delta_expression(&expr).unwrap();
        match delta_expr {
            Expression::Column(name) => assert_eq!(&name.to_string(), "test_column"),
            _ => panic!("Expected Column expression, got {:?}", delta_expr),
        }
    }

    #[test]
    fn test_column_expression_preserves_dots_in_name() {
        let expr = Expr::Column(Column::from_name("a.b"));
        assert_eq!(
            to_delta_expression(&expr).unwrap(),
            Expression::Column(ColumnName::new(["a.b"]))
        );
    }

    #[test]
    fn test_field_access_preserves_dots_in_field_name_segment() {
        let expr = col("a").field("b.c");
        assert_eq!(
            to_delta_expression(&expr).unwrap(),
            Expression::Column(ColumnName::new(["a", "b.c"]))
        );
    }

    #[test]
    fn test_literal_expressions() {
        // Test boolean literal
        let expr = lit(true);
        let delta_expr = to_delta_expression(&expr).unwrap();
        match delta_expr {
            Expression::Literal(Scalar::Boolean(value)) => assert!(value),
            _ => panic!("Expected Boolean literal, got {:?}", delta_expr),
        }

        // Test string literal
        let expr = lit("test");
        let delta_expr = to_delta_expression(&expr).unwrap();
        match delta_expr {
            Expression::Literal(Scalar::String(value)) => assert_eq!(value, "test"),
            _ => panic!("Expected String literal, got {:?}", delta_expr),
        }

        // Test integer literal
        let expr = lit(42i32);
        let delta_expr = to_delta_expression(&expr).unwrap();
        match delta_expr {
            Expression::Literal(Scalar::Integer(value)) => assert_eq!(value, 42),
            _ => panic!("Expected Integer literal, got {:?}", delta_expr),
        }

        // Test decimal literal
        let expr = lit(ScalarValue::Decimal128(Some(12345), 10, 2));
        let delta_expr = to_delta_expression(&expr).unwrap();
        match delta_expr {
            Expression::Literal(Scalar::Decimal(data)) => {
                assert_eq!(data.bits(), 12345);
                assert_eq!(data.precision(), 10);
                assert_eq!(data.scale(), 2);
            }
            _ => panic!("Expected Decimal literal, got {:?}", delta_expr),
        }
    }

    #[test]
    fn test_binary_expressions() {
        // Test comparison operators
        let test_cases = vec![
            (col("a").eq(lit(1)), BinaryPredicateOp::Equal),
            (col("a").lt(lit(1)), BinaryPredicateOp::LessThan),
            (col("a").gt(lit(1)), BinaryPredicateOp::GreaterThan),
        ];

        for (expr, expected_op) in test_cases {
            let delta_expr = to_delta_expression(&expr).unwrap();
            match delta_expr {
                Expression::Predicate(predicate) => match predicate.as_ref() {
                    Predicate::Binary(binary) => {
                        assert_eq!(binary.op, expected_op);
                        match binary.left.as_ref() {
                            Expression::Column(name) => assert_eq!(name.to_string(), "a"),
                            _ => panic!("Expected Column expression in left operand"),
                        }
                        match *binary.right.as_ref() {
                            Expression::Literal(Scalar::Integer(value)) => assert_eq!(value, 1),
                            _ => panic!("Expected Integer literal in right operand"),
                        }
                    }
                    _ => panic!("Expected Binary predicate, got {:?}", predicate),
                },
                _ => panic!("Expected Binary expression, got {:?}", delta_expr),
            }
        }

        // Test arithmetic operators
        let test_cases = vec![
            (col("a") + lit(1), BinaryExpressionOp::Plus),
            (col("a") - lit(1), BinaryExpressionOp::Minus),
            (col("a") * lit(1), BinaryExpressionOp::Multiply),
            (col("a") / lit(1), BinaryExpressionOp::Divide),
        ];

        for (expr, expected_op) in test_cases {
            let delta_expr = to_delta_expression(&expr).unwrap();
            match delta_expr {
                Expression::Binary(binary) => {
                    assert_eq!(binary.op, expected_op);
                    match binary.left.as_ref() {
                        Expression::Column(name) => assert_eq!(name.to_string(), "a"),
                        _ => panic!("Expected Column expression in left operand"),
                    }
                    match *binary.right.as_ref() {
                        Expression::Literal(Scalar::Integer(value)) => assert_eq!(value, 1),
                        _ => panic!("Expected Integer literal in right operand"),
                    }
                }
                _ => panic!("Expected Binary expression, got {:?}", delta_expr),
            }
        }
    }

    #[test]
    fn test_unary_expressions() {
        // Test IS NULL
        let expr = col("a").is_null();
        let delta_expr = to_delta_expression(&expr).unwrap();
        match delta_expr {
            Expression::Predicate(predicate) => match predicate.as_ref() {
                Predicate::Unary(unary) => {
                    assert_eq!(unary.op, UnaryPredicateOp::IsNull);
                    match unary.expr.as_ref() {
                        Expression::Column(name) => assert_eq!(name.to_string(), "a"),
                        _ => panic!("Expected Column expression in operand"),
                    }
                }
                _ => panic!("Expected Unary predicate, got {:?}", predicate),
            },
            _ => panic!("Expected Unary expression, got {:?}", delta_expr),
        }

        // Test NOT
        let expr = !col("a");
        let delta_expr = to_delta_expression(&expr).unwrap();
        match delta_expr {
            Expression::Predicate(predicate) => match predicate.as_ref() {
                Predicate::Not(unary) => match unary.as_ref() {
                    Predicate::BooleanExpression(expr) => match expr {
                        Expression::Column(name) => assert_eq!(name.to_string(), "a"),
                        _ => panic!("Expected Column expression in operand"),
                    },
                    _ => panic!("Expected Boolean expression in operand"),
                },
                _ => panic!("Expected Unary predicate, got {:?}", predicate),
            },
            _ => panic!("Expected Unary expression, got {:?}", delta_expr),
        }
    }

    #[test]
    fn test_null_literals() {
        let test_cases = vec![
            (lit(ScalarValue::Boolean(None)), DataType::BOOLEAN),
            (lit(ScalarValue::Utf8(None)), DataType::STRING),
            (lit(ScalarValue::Int32(None)), DataType::INTEGER),
            (lit(ScalarValue::Float64(None)), DataType::DOUBLE),
        ];

        for (expr, expected_type) in test_cases {
            let delta_expr = to_delta_expression(&expr).unwrap();
            match delta_expr {
                Expression::Literal(Scalar::Null(data_type)) => {
                    assert_eq!(data_type, expected_type);
                }
                _ => panic!("Expected Null literal, got {:?}", delta_expr),
            }
        }
    }

    #[test]
    fn test_between_expressions() {
        // Test BETWEEN (not negated) - should be equivalent to: NOT (x < low OR x > high)
        let expr = col("x").between(lit(10), lit(20));
        let delta_expr = to_delta_expression(&expr).unwrap();

        match delta_expr {
            Expression::Predicate(predicate) => match predicate.as_ref() {
                Predicate::Not(not_pred) => match not_pred.as_ref() {
                    Predicate::Junction(junction) => {
                        assert_eq!(junction.op, JunctionPredicateOp::Or);
                        assert_eq!(junction.preds.len(), 2);

                        // First predicate should be x < 10
                        match &junction.preds[0] {
                            Predicate::Binary(binary) => {
                                assert_eq!(binary.op, BinaryPredicateOp::LessThan);
                                match binary.left.as_ref() {
                                    Expression::Column(name) => assert_eq!(name.to_string(), "x"),
                                    _ => panic!("Expected Column expression in left operand"),
                                }
                                match binary.right.as_ref() {
                                    Expression::Literal(Scalar::Integer(value)) => {
                                        assert_eq!(*value, 10)
                                    }
                                    _ => panic!("Expected Integer literal in right operand"),
                                }
                            }
                            _ => panic!("Expected Binary predicate for first condition"),
                        }

                        // Second predicate should be x > 20
                        match &junction.preds[1] {
                            Predicate::Binary(binary) => {
                                assert_eq!(binary.op, BinaryPredicateOp::GreaterThan);
                                match binary.left.as_ref() {
                                    Expression::Column(name) => assert_eq!(name.to_string(), "x"),
                                    _ => panic!("Expected Column expression in left operand"),
                                }
                                match binary.right.as_ref() {
                                    Expression::Literal(Scalar::Integer(value)) => {
                                        assert_eq!(*value, 20)
                                    }
                                    _ => panic!("Expected Integer literal in right operand"),
                                }
                            }
                            _ => panic!("Expected Binary predicate for second condition"),
                        }
                    }
                    _ => panic!("Expected Junction predicate inside NOT"),
                },
                _ => panic!("Expected NOT predicate for BETWEEN, got {:?}", predicate),
            },
            _ => panic!("Expected Predicate expression, got {:?}", delta_expr),
        }
    }

    #[test]
    fn test_not_between_expressions() {
        // Test NOT BETWEEN (negated) - should be equivalent to: x < low OR x > high
        let expr = col("y").not_between(lit(5), lit(15));
        let delta_expr = to_delta_expression(&expr).unwrap();

        match delta_expr {
            Expression::Predicate(predicate) => match predicate.as_ref() {
                Predicate::Junction(junction) => {
                    assert_eq!(junction.op, JunctionPredicateOp::Or);
                    assert_eq!(junction.preds.len(), 2);

                    // First predicate should be y < 5
                    match &junction.preds[0] {
                        Predicate::Binary(binary) => {
                            assert_eq!(binary.op, BinaryPredicateOp::LessThan);
                            match binary.left.as_ref() {
                                Expression::Column(name) => assert_eq!(name.to_string(), "y"),
                                _ => panic!("Expected Column expression in left operand"),
                            }
                            match binary.right.as_ref() {
                                Expression::Literal(Scalar::Integer(value)) => {
                                    assert_eq!(*value, 5)
                                }
                                _ => panic!("Expected Integer literal in right operand"),
                            }
                        }
                        _ => panic!("Expected Binary predicate for first condition"),
                    }

                    // Second predicate should be y > 15
                    match &junction.preds[1] {
                        Predicate::Binary(binary) => {
                            assert_eq!(binary.op, BinaryPredicateOp::GreaterThan);
                            match binary.left.as_ref() {
                                Expression::Column(name) => assert_eq!(name.to_string(), "y"),
                                _ => panic!("Expected Column expression in left operand"),
                            }
                            match binary.right.as_ref() {
                                Expression::Literal(Scalar::Integer(value)) => {
                                    assert_eq!(*value, 15)
                                }
                                _ => panic!("Expected Integer literal in right operand"),
                            }
                        }
                        _ => panic!("Expected Binary predicate for second condition"),
                    }
                }
                _ => panic!(
                    "Expected Junction predicate for NOT BETWEEN, got {:?}",
                    predicate
                ),
            },
            _ => panic!("Expected Predicate expression, got {:?}", delta_expr),
        }
    }

    #[test]
    fn test_between_with_expressions() {
        // Test BETWEEN with expressions as bounds: col("a") + 1 BETWEEN col("low") AND col("high")
        let expr = (col("a") + lit(1)).between(col("low"), col("high"));
        let delta_expr = to_delta_expression(&expr).unwrap();

        match delta_expr {
            Expression::Predicate(predicate) => match predicate.as_ref() {
                Predicate::Not(not_pred) => match not_pred.as_ref() {
                    Predicate::Junction(junction) => {
                        assert_eq!(junction.op, JunctionPredicateOp::Or);
                        assert_eq!(junction.preds.len(), 2);

                        // Verify the expression being tested is (a + 1)
                        for pred in &junction.preds {
                            match pred {
                                Predicate::Binary(binary) => match binary.left.as_ref() {
                                    Expression::Binary(bin_expr) => {
                                        assert_eq!(bin_expr.op, BinaryExpressionOp::Plus);
                                        match bin_expr.left.as_ref() {
                                            Expression::Column(name) => {
                                                assert_eq!(name.to_string(), "a")
                                            }
                                            _ => panic!("Expected Column 'a' in binary expression"),
                                        }
                                    }
                                    _ => panic!("Expected Binary expression for (a + 1)"),
                                },
                                _ => panic!("Expected Binary predicate"),
                            }
                        }
                    }
                    _ => panic!("Expected Junction predicate inside NOT"),
                },
                _ => panic!("Expected NOT predicate for BETWEEN"),
            },
            _ => panic!("Expected Predicate expression"),
        }
    }
}
