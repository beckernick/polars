use std::sync::Arc;

use polars_core::prelude::*;
use polars_ops::prelude::{JoinType, MaintainOrderJoin};
use polars_utils::arena::{Arena, Node};

use crate::plans::aexpr::AExpr;
use crate::plans::ir::IR;
use crate::plans::optimizer::OptimizationRule;
use crate::plans::schema::det_join_schema;
use crate::plans::{ExprIR, JoinOptionsIR};
use crate::utils::aexpr_to_leaf_names_iter;

/// Reorders chains of inner joins to minimise the size of intermediate results.
///
/// For a plan of the form `(A INNER JOIN B) INNER JOIN C` where:
/// - both joins are inner joins without `maintain_order`,
/// - the outer join's left-side keys reference only columns from `A`, and
/// - `C` has fewer estimated rows than `B`,
///
/// the plan is reordered to `(A INNER JOIN C) INNER JOIN B`, which produces a
/// smaller first intermediate and therefore a faster overall query.  A
/// `SimpleProjection` is appended to restore the original output column order.
pub(super) struct JoinReorder;

impl JoinReorder {
    /// Estimate the row count of a plan node.  Returns `usize::MAX` when the
    /// size is not known.
    fn estimate_rows(node: Node, ir_arena: &Arena<IR>) -> usize {
        match ir_arena.get(node) {
            IR::DataFrameScan { df, .. } => df.height(),
            IR::Scan { file_info, .. } => {
                let (known, estimated) = file_info.row_estimation;
                known.unwrap_or(estimated)
            },
            IR::Filter { input, .. } => {
                // A filter reduces rows — use half as a conservative heuristic.
                Self::estimate_rows(*input, ir_arena).saturating_div(2)
            },
            IR::Slice { len, .. } => *len as usize,
            other => {
                // For joins, aggregations, etc. walk inputs and take the max.
                let mut inputs = Vec::new();
                other.copy_inputs(&mut inputs);
                inputs
                    .iter()
                    .map(|&n| Self::estimate_rows(n, ir_arena))
                    .max()
                    .unwrap_or(usize::MAX)
            },
        }
    }

    /// Returns `true` when every leaf column name referenced by `exprs` is
    /// present in `schema`.
    fn all_leaf_names_in_schema(
        exprs: &[ExprIR],
        schema: &Schema,
        expr_arena: &Arena<AExpr>,
    ) -> bool {
        exprs.iter().all(|expr| {
            aexpr_to_leaf_names_iter(expr.node(), expr_arena)
                .all(|name| schema.contains(name.as_str()))
        })
    }

    /// Returns `true` when `options` describes an inner join that has no
    /// `maintain_order` requirement (i.e. the join is freely reorderable).
    fn is_reorderable_inner(options: &JoinOptionsIR) -> bool {
        options.args.how == JoinType::Inner
            && options.args.maintain_order == MaintainOrderJoin::None
    }
}

impl OptimizationRule for JoinReorder {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        // ── 1. Match the outer join ──────────────────────────────────────────
        let (inner_join_node, c_node, outer_schema, outer_left_on, outer_right_on, outer_options) = {
            let IR::Join {
                input_left,
                input_right,
                schema,
                left_on,
                right_on,
                options,
            } = lp_arena.get(node)
            else {
                return Ok(None);
            };
            if !Self::is_reorderable_inner(options) {
                return Ok(None);
            }
            (
                *input_left,
                *input_right,
                Arc::clone(schema),
                left_on.clone(),
                right_on.clone(),
                Arc::clone(options),
            )
        };

        // ── 2. Match the inner join (left input of the outer join) ───────────
        let (a_node, b_node, inner_left_on, inner_right_on, inner_options) = {
            let IR::Join {
                input_left,
                input_right,
                left_on,
                right_on,
                options,
                ..
            } = lp_arena.get(inner_join_node)
            else {
                return Ok(None);
            };
            if !Self::is_reorderable_inner(options) {
                return Ok(None);
            }
            (
                *input_left,
                *input_right,
                left_on.clone(),
                right_on.clone(),
                Arc::clone(options),
            )
        };

        // ── 3. Guard: outer left keys must come from A, not from B ───────────
        //
        // If any key is only provided by B, moving C above B would produce a
        // plan where that key does not yet exist when A JOIN C is evaluated.
        let schema_a = lp_arena.get(a_node).schema(lp_arena).into_owned();
        if !Self::all_leaf_names_in_schema(&outer_left_on, &schema_a, expr_arena) {
            return Ok(None);
        }

        // ── 4. Guard: only reorder when C is smaller than B ─────────────────
        let b_rows = Self::estimate_rows(b_node, lp_arena);
        let c_rows = Self::estimate_rows(c_node, lp_arena);
        if c_rows >= b_rows {
            return Ok(None);
        }

        // ── 5. Build the reordered plan ──────────────────────────────────────
        //
        // New inner join: A JOIN C  (using the original outer join's keys)
        let schema_c = lp_arena.get(c_node).schema(lp_arena).into_owned();
        let new_inner_schema = det_join_schema(
            &schema_a,
            &schema_c,
            &outer_left_on,
            &outer_right_on,
            &outer_options,
            expr_arena,
        )?;

        let new_inner_node = lp_arena.add(IR::Join {
            input_left: a_node,
            input_right: c_node,
            schema: new_inner_schema.clone(),
            left_on: outer_left_on,
            right_on: outer_right_on,
            options: outer_options,
        });

        // New outer join: (A JOIN C) JOIN B  (using the original inner join's keys)
        let schema_b = lp_arena.get(b_node).schema(lp_arena).into_owned();
        let new_outer_schema = det_join_schema(
            &new_inner_schema,
            &schema_b,
            &inner_left_on,
            &inner_right_on,
            &inner_options,
            expr_arena,
        )?;

        let new_outer_node = lp_arena.add(IR::Join {
            input_left: new_inner_node,
            input_right: b_node,
            schema: new_outer_schema,
            left_on: inner_left_on,
            right_on: inner_right_on,
            options: inner_options,
        });

        // ── 6. Restore original column order ────────────────────────────────
        //
        // After reordering, B's columns appear after C's columns in the output
        // rather than before.  A SimpleProjection restores the original schema
        // order with no data copy (it calls `df.select_unchecked`).
        Ok(Some(IR::SimpleProjection {
            input: new_outer_node,
            columns: outer_schema,
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use polars_core::prelude::*;
    use polars_ops::prelude::{JoinArgs, JoinType};
    use polars_utils::arena::Arena;

    use super::*;
    use crate::plans::ir::IR;
    use crate::plans::optimizer::OptimizationRule;
    use crate::plans::{ExprIR, JoinOptionsIR, OutputName};
    use crate::plans::aexpr::AExpr;

    /// Create an `IR::DataFrameScan` node whose `df.height()` equals `n_rows`.
    fn make_df_scan(
        schema: Schema,
        n_rows: usize,
        ir_arena: &mut Arena<IR>,
    ) -> Node {
        let columns: Vec<Column> = schema
            .iter_fields()
            .map(|f| Column::new_scalar(f.name().clone(), Scalar::null(f.dtype.clone()), n_rows))
            .collect();
        let df = DataFrame::new(n_rows, columns).unwrap();
        ir_arena.add(IR::DataFrameScan {
            df: Arc::new(df),
            schema: Arc::new(schema.clone()),
            output_schema: None,
        })
    }

    fn col_expr(name: &str, expr_arena: &mut Arena<AExpr>) -> ExprIR {
        let node = expr_arena.add(AExpr::Column(name.into()));
        ExprIR::new(node, OutputName::ColumnLhs(name.into()))
    }

    fn inner_join_options() -> Arc<JoinOptionsIR> {
        Arc::new(JoinOptionsIR {
            allow_parallel: true,
            force_parallel: false,
            args: JoinArgs {
                how: JoinType::Inner,
                ..Default::default()
            },
            options: None,
        })
    }

    /// Verifies that `(A JOIN B) JOIN C` is reordered to `(A JOIN C) JOIN B`
    /// when C is smaller than B.
    #[test]
    fn test_join_reorder_basic() {
        let mut ir_arena: Arena<IR> = Arena::new();
        let mut expr_arena: Arena<AExpr> = Arena::new();

        // A: events — 5 000 rows, keys: id, b_key, c_key
        let schema_a = Schema::from_iter([
            Field::new("id".into(), DataType::Int64),
            Field::new("b_key".into(), DataType::Int64),
            Field::new("c_key".into(), DataType::Int64),
        ]);
        // B: large dimension — 1 000 rows
        let schema_b = Schema::from_iter([
            Field::new("b_key".into(), DataType::Int64),
            Field::new("b_val".into(), DataType::String),
        ]);
        // C: small dimension — 10 rows
        let schema_c = Schema::from_iter([
            Field::new("c_key".into(), DataType::Int64),
            Field::new("c_val".into(), DataType::String),
        ]);

        let a = make_df_scan(schema_a.clone(), 5_000, &mut ir_arena);
        let b = make_df_scan(schema_b.clone(), 1_000, &mut ir_arena);
        let c = make_df_scan(schema_c.clone(), 10, &mut ir_arena);

        let options = inner_join_options();
        let schema_a_ref = Arc::new(schema_a.clone());
        let schema_b_ref = Arc::new(schema_b.clone());
        let schema_c_ref = Arc::new(schema_c.clone());

        // Inner join schema: A JOIN B
        let inner_schema = det_join_schema(
            &schema_a_ref,
            &schema_b_ref,
            &[col_expr("b_key", &mut expr_arena)],
            &[col_expr("b_key", &mut expr_arena)],
            &options,
            &expr_arena,
        )
        .unwrap();

        let inner_node = ir_arena.add(IR::Join {
            input_left: a,
            input_right: b,
            schema: inner_schema.clone(),
            left_on: vec![col_expr("b_key", &mut expr_arena)],
            right_on: vec![col_expr("b_key", &mut expr_arena)],
            options: Arc::clone(&options),
        });

        // Outer join schema: (A JOIN B) JOIN C
        let outer_schema = det_join_schema(
            &inner_schema,
            &schema_c_ref,
            &[col_expr("c_key", &mut expr_arena)],
            &[col_expr("c_key", &mut expr_arena)],
            &options,
            &expr_arena,
        )
        .unwrap();

        let outer_node = ir_arena.add(IR::Join {
            input_left: inner_node,
            input_right: c,
            schema: Arc::clone(&outer_schema),
            left_on: vec![col_expr("c_key", &mut expr_arena)],
            right_on: vec![col_expr("c_key", &mut expr_arena)],
            options: Arc::clone(&options),
        });

        let mut rule = JoinReorder;
        let result = rule
            .optimize_plan(&mut ir_arena, &mut expr_arena, outer_node)
            .unwrap();

        // The optimizer should have returned Some(SimpleProjection { ... })
        assert!(
            result.is_some(),
            "join reorder should have triggered for C (10 rows) < B (1000 rows)"
        );

        let IR::SimpleProjection { input, columns } = result.unwrap() else {
            panic!("expected SimpleProjection wrapper");
        };

        // The projection must restore the original schema.
        assert_eq!(
            columns.iter_names().collect::<Vec<_>>(),
            outer_schema.iter_names().collect::<Vec<_>>(),
            "SimpleProjection must restore original column order"
        );

        // The new outer join should use B as right input.
        let IR::Join {
            input_left: new_inner,
            input_right: new_b,
            ..
        } = ir_arena.get(input)
        else {
            panic!("expected outer Join node");
        };
        assert_eq!(*new_b, b, "B must be the right input of the new outer join");

        // The new inner join should use C as right input.
        let IR::Join {
            input_left: new_a,
            input_right: new_c,
            ..
        } = ir_arena.get(*new_inner)
        else {
            panic!("expected inner Join node");
        };
        assert_eq!(*new_a, a, "A must remain the left input");
        assert_eq!(*new_c, c, "C must be the right input of the new inner join");
    }

    /// When C is larger than B the optimisation must NOT fire.
    #[test]
    fn test_join_reorder_no_swap_when_b_smaller() {
        let mut ir_arena: Arena<IR> = Arena::new();
        let mut expr_arena: Arena<AExpr> = Arena::new();

        let schema_a = Schema::from_iter([
            Field::new("id".into(), DataType::Int64),
            Field::new("b_key".into(), DataType::Int64),
            Field::new("c_key".into(), DataType::Int64),
        ]);
        let schema_b = Schema::from_iter([
            Field::new("b_key".into(), DataType::Int64),
            Field::new("b_val".into(), DataType::String),
        ]);
        let schema_c = Schema::from_iter([
            Field::new("c_key".into(), DataType::Int64),
            Field::new("c_val".into(), DataType::String),
        ]);

        let a = make_df_scan(schema_a.clone(), 5_000, &mut ir_arena);
        // B is small (10), C is large (1 000) — no swap beneficial
        let b = make_df_scan(schema_b.clone(), 10, &mut ir_arena);
        let c = make_df_scan(schema_c.clone(), 1_000, &mut ir_arena);

        let options = inner_join_options();
        let schema_a_ref = Arc::new(schema_a);
        let schema_b_ref = Arc::new(schema_b);
        let schema_c_ref = Arc::new(schema_c);

        let inner_schema = det_join_schema(
            &schema_a_ref,
            &schema_b_ref,
            &[col_expr("b_key", &mut expr_arena)],
            &[col_expr("b_key", &mut expr_arena)],
            &options,
            &expr_arena,
        )
        .unwrap();

        let inner_node = ir_arena.add(IR::Join {
            input_left: a,
            input_right: b,
            schema: inner_schema.clone(),
            left_on: vec![col_expr("b_key", &mut expr_arena)],
            right_on: vec![col_expr("b_key", &mut expr_arena)],
            options: Arc::clone(&options),
        });

        let outer_schema = det_join_schema(
            &inner_schema,
            &schema_c_ref,
            &[col_expr("c_key", &mut expr_arena)],
            &[col_expr("c_key", &mut expr_arena)],
            &options,
            &expr_arena,
        )
        .unwrap();

        let outer_node = ir_arena.add(IR::Join {
            input_left: inner_node,
            input_right: c,
            schema: outer_schema,
            left_on: vec![col_expr("c_key", &mut expr_arena)],
            right_on: vec![col_expr("c_key", &mut expr_arena)],
            options,
        });

        let mut rule = JoinReorder;
        let result = rule
            .optimize_plan(&mut ir_arena, &mut expr_arena, outer_node)
            .unwrap();

        assert!(
            result.is_none(),
            "join reorder must not fire when B is already the smaller table"
        );
    }
}
