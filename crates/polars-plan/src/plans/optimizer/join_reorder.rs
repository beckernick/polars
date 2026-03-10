use std::sync::Arc;

use polars_core::prelude::*;
use polars_ops::prelude::{JoinType, MaintainOrderJoin};
use polars_utils::arena::{Arena, Node};

use crate::plans::aexpr::AExpr;
use crate::plans::ir::IR;
use crate::plans::optimizer::OptimizationRule;
use crate::plans::schema::det_join_schema;
use crate::plans::{ExprIR, JoinOptionsIR};
use crate::utils::check_input_node;

/// One right-side input extracted from a left-deep inner-join chain.
struct JoinEntry {
    right_node: Node,
    left_on: Vec<ExprIR>,
    right_on: Vec<ExprIR>,
    options: Arc<JoinOptionsIR>,
    right_schema: SchemaRef,
    estimated_rows: usize,
}

/// Reorders chains of inner joins to minimise the size of intermediate results.
///
/// For a left-deep chain  `(((base JOIN t1) JOIN t2) … JOIN tn)`  the rule:
///
/// 1. Extracts all `n` right-side tables together with their join keys.
/// 2. Runs a greedy topological sort: repeatedly pick the table with the
///    fewest estimated rows whose join keys are already available in the
///    accumulated left-side schema.
/// 3. Rebuilds the left-deep tree in the new order.
/// 4. Wraps the result in a `SimpleProjection` to preserve the original
///    output column order.
///
/// The optimisation fires only when all joins in the chain are inner joins
/// with no `maintain_order` requirement.
pub(super) struct JoinReorder;

impl JoinReorder {
    /// Estimate the row count of a plan node. Returns `usize::MAX` when
    /// the size is not known.
    fn estimate_rows(node: Node, ir_arena: &Arena<IR>) -> usize {
        match ir_arena.get(node) {
            IR::DataFrameScan { df, .. } => df.height(),
            IR::Scan { file_info, .. } => {
                let (known, estimated) = file_info.row_estimation;
                known.unwrap_or(estimated)
            },
            IR::Filter { input, .. } => Self::estimate_rows(*input, ir_arena).saturating_div(2),
            IR::Slice { len, .. } => *len as usize,
            other => other
                .inputs()
                .map(|n| Self::estimate_rows(n, ir_arena))
                .max()
                .unwrap_or(usize::MAX),
        }
    }

    fn is_reorderable_inner(options: &JoinOptionsIR) -> bool {
        options.args.how == JoinType::Inner
            && options.args.maintain_order == MaintainOrderJoin::None
    }

    /// Walk the left-deep join tree rooted at `node` and collect:
    /// - `base_node`: the leftmost non-join leaf.
    /// - `entries`: one `JoinEntry` per right-side input, in original
    ///   left-to-right join order (innermost join first).
    ///
    /// Returns `None` if any join in the chain is not a reorderable inner join.
    fn extract_chain(
        node: Node,
        ir_arena: &Arena<IR>,
        expr_arena: &Arena<AExpr>,
    ) -> Option<(Node, Vec<JoinEntry>)> {
        let mut entries: Vec<JoinEntry> = Vec::new();
        let mut current = node;

        loop {
            let IR::Join {
                input_left,
                input_right,
                left_on,
                right_on,
                options,
                ..
            } = ir_arena.get(current)
            else {
                // Reached the base (non-join) node.
                // We collected entries outermost-first; reverse to get B, C, D order.
                entries.reverse();
                return Some((current, entries));
            };

            if !Self::is_reorderable_inner(options) {
                return None;
            }

            entries.push(JoinEntry {
                right_node: *input_right,
                left_on: left_on.clone(),
                right_on: right_on.clone(),
                options: Arc::clone(options),
                right_schema: ir_arena.get(*input_right).schema(ir_arena).into_owned(),
                estimated_rows: Self::estimate_rows(*input_right, ir_arena),
            });

            current = *input_left;
        }
    }

    /// Greedy topological sort over `entries`.
    ///
    /// At each step, from the set of entries whose `left_on` keys are all
    /// present in the current accumulated schema, pick the one with the
    /// fewest estimated rows (ties broken by original index for stability).
    /// Then advance the schema by computing the join output with that entry.
    ///
    /// Returns:
    /// - `Ok(Some((order, schemas)))` — a valid reordering was found. `order`
    ///   contains indices into `entries`; `schemas[i]` is the accumulated join
    ///   output schema after placing `order[i]`, ready for reuse when
    ///   rebuilding the tree.
    /// - `Ok(None)` — no valid ordering could be constructed (bail out safely).
    fn greedy_order(
        base_schema: SchemaRef,
        entries: &[JoinEntry],
        expr_arena: &Arena<AExpr>,
    ) -> PolarsResult<Option<(Vec<usize>, Vec<SchemaRef>)>> {
        let n = entries.len();
        let mut placed = vec![false; n];
        let mut order = Vec::with_capacity(n);
        let mut schemas = Vec::with_capacity(n);
        let mut current_schema = base_schema;

        for _ in 0..n {
            let Some(next) = (0..n)
                .filter(|&i| !placed[i])
                .filter(|&i| {
                    entries[i]
                        .left_on
                        .iter()
                        .all(|expr| check_input_node(expr.node(), &current_schema, expr_arena))
                })
                .min_by_key(|&i| (entries[i].estimated_rows, i))
            else {
                // No entry can be placed — dependency cannot be satisfied.
                return Ok(None);
            };

            placed[next] = true;
            order.push(next);

            // Advance the schema to reflect this join's output, so subsequent
            // entries can correctly check column availability.
            current_schema = det_join_schema(
                &current_schema,
                &entries[next].right_schema,
                &entries[next].left_on,
                &entries[next].right_on,
                &entries[next].options,
                expr_arena,
            )?;
            schemas.push(Arc::clone(&current_schema));
        }

        Ok(Some((order, schemas)))
    }
}

impl OptimizationRule for JoinReorder {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        // Fast pre-check: only process Join nodes.
        if !matches!(lp_arena.get(node), IR::Join { .. }) {
            return Ok(None);
        }

        // ── 1. Extract the full left-deep join chain ─────────────────────────
        let Some((base_node, entries)) = Self::extract_chain(node, lp_arena, expr_arena) else {
            return Ok(None);
        };

        // Need at least two joins for reordering to be meaningful.
        if entries.len() < 2 {
            return Ok(None);
        }

        // ── 2. Find the optimal ordering ─────────────────────────────────────
        let original_schema = lp_arena.get(node).schema(lp_arena).into_owned();
        let base_schema = lp_arena.get(base_node).schema(lp_arena).into_owned();

        let Some((order, schemas)) = Self::greedy_order(base_schema, &entries, expr_arena)? else {
            return Ok(None);
        };

        // If the greedy order matches the original, nothing to do.
        let is_identity = order.iter().enumerate().all(|(pos, &orig)| pos == orig);
        if is_identity {
            return Ok(None);
        }

        // ── 3. Build the reordered left-deep join tree ───────────────────────
        let mut current_left = base_node;

        for (step, &idx) in order.iter().enumerate() {
            let entry = &entries[idx];
            current_left = lp_arena.add(IR::Join {
                input_left: current_left,
                input_right: entry.right_node,
                schema: Arc::clone(&schemas[step]),
                left_on: entry.left_on.clone(),
                right_on: entry.right_on.clone(),
                options: Arc::clone(&entry.options),
            });
        }

        // ── 4. Restore original column order ─────────────────────────────────
        //
        // Reordering changes the order in which columns appear in the output.
        // A SimpleProjection restores the original schema order at negligible
        // cost (it calls `df.select_unchecked`).
        Ok(Some(IR::SimpleProjection {
            input: current_left,
            columns: original_schema,
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
    use crate::plans::aexpr::AExpr;
    use crate::plans::ir::IR;
    use crate::plans::optimizer::OptimizationRule;
    use crate::plans::schema::det_join_schema;
    use crate::plans::{ExprIR, JoinOptionsIR, OutputName};

    fn make_df_scan(schema: Schema, n_rows: usize, ir_arena: &mut Arena<IR>) -> Node {
        let columns: Vec<Column> = schema
            .iter_fields()
            .map(|f| Column::new_scalar(f.name().clone(), Scalar::null(f.dtype.clone()), n_rows))
            .collect();
        let df = DataFrame::new(n_rows, columns).unwrap();
        ir_arena.add(IR::DataFrameScan {
            df: Arc::new(df),
            schema: Arc::new(schema),
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

    /// 3-table case: `(A JOIN B) JOIN C` → `(A JOIN C) JOIN B` when C < B.
    #[test]
    fn test_three_tables_reorder() {
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
        let b = make_df_scan(schema_b.clone(), 1_000, &mut ir_arena);
        let c = make_df_scan(schema_c.clone(), 10, &mut ir_arena);

        let options = inner_join_options();
        let sa = Arc::new(schema_a);
        let sb = Arc::new(schema_b);
        let sc = Arc::new(schema_c);

        let inner_schema = det_join_schema(
            &sa,
            &sb,
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
            &sc,
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

        let result = JoinReorder
            .optimize_plan(&mut ir_arena, &mut expr_arena, outer_node)
            .unwrap();

        let IR::SimpleProjection { input, columns } = result.unwrap() else {
            panic!("expected SimpleProjection");
        };
        assert_eq!(
            columns.iter_names().collect::<Vec<_>>(),
            outer_schema.iter_names().collect::<Vec<_>>(),
        );

        // New outer join should have B as right input.
        let IR::Join {
            input_right: new_b,
            input_left: new_inner,
            ..
        } = ir_arena.get(input)
        else {
            panic!("expected outer Join");
        };
        assert_eq!(*new_b, b);

        // New inner join should have C as right input.
        let IR::Join {
            input_left: new_a,
            input_right: new_c,
            ..
        } = ir_arena.get(*new_inner)
        else {
            panic!("expected inner Join");
        };
        assert_eq!(*new_a, a);
        assert_eq!(*new_c, c);
    }

    /// Already-optimal order should produce no change.
    #[test]
    fn test_no_swap_when_already_optimal() {
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
        // B is already smaller than C → optimal order is already (A JOIN B) JOIN C
        let b = make_df_scan(schema_b.clone(), 10, &mut ir_arena);
        let c = make_df_scan(schema_c.clone(), 1_000, &mut ir_arena);

        let options = inner_join_options();
        let sa = Arc::new(schema_a);
        let sb = Arc::new(schema_b);
        let sc = Arc::new(schema_c);

        let inner_schema = det_join_schema(
            &sa,
            &sb,
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
            &sc,
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

        let result = JoinReorder
            .optimize_plan(&mut ir_arena, &mut expr_arena, outer_node)
            .unwrap();
        assert!(result.is_none(), "no reorder needed when already optimal");
    }

    /// 5-table case: base JOIN t1(1000) JOIN t2(500) JOIN t3(10) JOIN t4(5)
    /// Optimal: base JOIN t4 JOIN t3 JOIN t2 JOIN t1
    #[test]
    fn test_five_tables_reorder() {
        let mut ir_arena: Arena<IR> = Arena::new();
        let mut expr_arena: Arena<AExpr> = Arena::new();

        // Base table with keys for all four right-side tables.
        let base_schema = Schema::from_iter([
            Field::new("id".into(), DataType::Int64),
            Field::new("k1".into(), DataType::Int64),
            Field::new("k2".into(), DataType::Int64),
            Field::new("k3".into(), DataType::Int64),
            Field::new("k4".into(), DataType::Int64),
        ]);
        let mk_right = |key: &str| {
            Schema::from_iter([
                Field::new(key.into(), DataType::Int64),
                Field::new(format!("{key}_val").into(), DataType::String),
            ])
        };

        let base = make_df_scan(base_schema.clone(), 10_000, &mut ir_arena);
        let t1 = make_df_scan(mk_right("k1"), 1_000, &mut ir_arena);
        let t2 = make_df_scan(mk_right("k2"), 500, &mut ir_arena);
        let t3 = make_df_scan(mk_right("k3"), 10, &mut ir_arena);
        let t4 = make_df_scan(mk_right("k4"), 5, &mut ir_arena);

        // Build worst-case chain: base JOIN t1(1000) JOIN t2(500) JOIN t3(10) JOIN t4(5)
        let options = inner_join_options();
        let sb = Arc::new(base_schema.clone());

        let mut left = base;
        let mut left_schema = sb.clone();
        let chain = [
            ("k1", t1, 1_000usize),
            ("k2", t2, 500),
            ("k3", t3, 10),
            ("k4", t4, 5),
        ];

        for (key, right, _rows) in &chain {
            let right_schema = ir_arena.get(*right).schema(&ir_arena).into_owned();
            let join_schema = det_join_schema(
                &left_schema,
                &right_schema,
                &[col_expr(key, &mut expr_arena)],
                &[col_expr(key, &mut expr_arena)],
                &options,
                &expr_arena,
            )
            .unwrap();
            left = ir_arena.add(IR::Join {
                input_left: left,
                input_right: *right,
                schema: join_schema.clone(),
                left_on: vec![col_expr(key, &mut expr_arena)],
                right_on: vec![col_expr(key, &mut expr_arena)],
                options: Arc::clone(&options),
            });
            left_schema = join_schema;
        }
        let root = left;
        let original_schema = ir_arena.get(root).schema(&ir_arena).into_owned();

        let result = JoinReorder
            .optimize_plan(&mut ir_arena, &mut expr_arena, root)
            .unwrap();
        let IR::SimpleProjection {
            input: new_root,
            columns,
        } = result.unwrap()
        else {
            panic!("expected SimpleProjection");
        };

        // Output schema must be preserved.
        assert_eq!(
            columns.iter_names().collect::<Vec<_>>(),
            original_schema.iter_names().collect::<Vec<_>>(),
        );

        // Collect the right-side nodes in the new order (outermost to innermost).
        let mut rights = Vec::new();
        let mut cur = new_root;
        loop {
            let IR::Join {
                input_left,
                input_right,
                ..
            } = ir_arena.get(cur)
            else {
                break;
            };
            rights.push(*input_right);
            cur = *input_left;
        }
        rights.reverse(); // now innermost-first

        // Expected order by ascending row count: t4(5), t3(10), t2(500), t1(1000)
        assert_eq!(
            rights,
            vec![t4, t3, t2, t1],
            "tables should be joined in ascending row-count order"
        );
    }

    /// A table whose left_on key was introduced by a previous right-side join
    /// must not be moved before that join.
    #[test]
    fn test_dependency_respected() {
        let mut ir_arena: Arena<IR> = Arena::new();
        let mut expr_arena: Arena<AExpr> = Arena::new();

        // base has id and b_key.
        // B adds column c_key (not in base).
        // C must join on c_key, so C depends on B.
        let base_schema = Schema::from_iter([
            Field::new("id".into(), DataType::Int64),
            Field::new("b_key".into(), DataType::Int64),
        ]);
        let b_schema = Schema::from_iter([
            Field::new("b_key".into(), DataType::Int64),
            Field::new("b_val".into(), DataType::String),
            Field::new("c_key".into(), DataType::Int64), // C depends on this
        ]);
        let c_schema = Schema::from_iter([
            Field::new("c_key".into(), DataType::Int64),
            Field::new("c_val".into(), DataType::String),
        ]);

        let base = make_df_scan(base_schema.clone(), 5_000, &mut ir_arena);
        // B is large (10 000 rows), C is tiny (5 rows) — naïve reorder would put C first,
        // but C depends on c_key which only B provides.
        let b = make_df_scan(b_schema.clone(), 10_000, &mut ir_arena);
        let c = make_df_scan(c_schema.clone(), 5, &mut ir_arena);

        let options = inner_join_options();
        let sb = Arc::new(base_schema);
        let sbb = Arc::new(b_schema);
        let sc = Arc::new(c_schema);

        // base JOIN B on b_key
        let inner_schema = det_join_schema(
            &sb,
            &sbb,
            &[col_expr("b_key", &mut expr_arena)],
            &[col_expr("b_key", &mut expr_arena)],
            &options,
            &expr_arena,
        )
        .unwrap();
        let inner_node = ir_arena.add(IR::Join {
            input_left: base,
            input_right: b,
            schema: inner_schema.clone(),
            left_on: vec![col_expr("b_key", &mut expr_arena)],
            right_on: vec![col_expr("b_key", &mut expr_arena)],
            options: Arc::clone(&options),
        });

        // (base JOIN B) JOIN C on c_key  — c_key comes from B
        let outer_schema = det_join_schema(
            &inner_schema,
            &sc,
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

        let result = JoinReorder
            .optimize_plan(&mut ir_arena, &mut expr_arena, outer_node)
            .unwrap();
        // Dependency prevents moving C before B, so order is unchanged → no rewrite.
        assert!(
            result.is_none(),
            "dependency on B must prevent C from being moved first"
        );
    }
}
