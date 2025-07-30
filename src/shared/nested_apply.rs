use replace_with::replace_with_or_default;
use rustc_hash::FxHashSet;

use crate::{
    bdd_node::BddNodeAny,
    boolean_operators::{self, BooleanOperator},
    conversion::{UncheckedFrom, UncheckedInto},
    node_id::{NodeId, NodeId16, NodeId32, NodeId64, NodeIdAny},
    node_table::{NodeTable, NodeTable16, NodeTable32, NodeTable64, NodeTableAny},
    split::nested_apply::{InnerApplyState, inner_apply_any},
    task_cache::{TaskCache16, TaskCache32, TaskCache64, TaskCacheAny},
    variable_id::{VarIdPacked16, VarIdPacked32, VarIdPacked64, VarIdPackedAny, VariableId},
};

use super::{Bdd, manager::BddManager};

impl BddManager {
    /// Applies the `outer_op` to the [`Bdd`]s. On each node of the resulting `Bdd`,
    /// if its variable is in `variables`, the node is replaced with the result
    /// of applying `inner_op` to its low and high children.
    ///
    /// This function is useful for implementing combinations of applying binary
    /// operators and quantification. For example, if `outer_op` is [`boolean_operators::And`]
    /// and `inner_op` is [`boolean_operators::Or`], this combination corresponds to the
    /// "relational product" operation.
    pub fn nested_apply<TOuterOp: BooleanOperator, TInnerOp: BooleanOperator>(
        &mut self,
        left: &Bdd,
        right: &Bdd,
        outer_op: TOuterOp,
        inner_op: TInnerOp,
        variables: &[VariableId],
    ) -> Bdd {
        self.maybe_collect_garbage();

        let mut bdd_root = NodeId::undefined();

        replace_with_or_default(&mut self.unique_table, |table| match table {
            NodeTable::Size16(table) => {
                let (root, table) = nested_apply_16_bit(
                    table,
                    left.root.get().unchecked_into(),
                    right.root.get().unchecked_into(),
                    outer_op,
                    inner_op,
                    variables,
                );
                bdd_root = root;
                table
            }
            NodeTable::Size32(table) => {
                let (root, table) = nested_apply_32_bit(
                    table,
                    left.root.get().unchecked_into(),
                    right.root.get().unchecked_into(),
                    outer_op,
                    inner_op,
                    variables,
                );
                bdd_root = root;
                table
            }
            NodeTable::Size64(table) => {
                let (root, table) = nested_apply_64_bit(
                    table,
                    left.root.get().unchecked_into(),
                    right.root.get().unchecked_into(),
                    outer_op,
                    inner_op,
                    variables,
                );
                bdd_root = root;
                table
            }
        });

        debug_assert!(!bdd_root.is_undefined());

        let bdd = Bdd::new(bdd_root);
        self.roots.push(bdd.root_weak());

        bdd
    }

    /// Calculates a [`Bdd`] with the given `variables` eliminated using existential quantification.
    pub fn exists(&mut self, bdd: &Bdd, variables: &[VariableId]) -> Bdd {
        self.binary_op_with_exists(bdd, bdd, boolean_operators::And, variables)
    }

    /// Calculates a [`Bdd`] with the given `variables` eliminated using universal quantification.
    pub fn for_all(&mut self, bdd: &Bdd, variables: &[VariableId]) -> Bdd {
        self.binary_op_with_for_all(bdd, bdd, boolean_operators::And, variables)
    }

    /// Applies the `operator` to the [`Bdd`]s and eliminates the given `variables` using existential
    /// quantification from the result.
    pub fn binary_op_with_exists<TBooleanOp: BooleanOperator>(
        &mut self,
        left: &Bdd,
        right: &Bdd,
        operator: TBooleanOp,
        variables: &[VariableId],
    ) -> Bdd {
        self.nested_apply(left, right, operator, boolean_operators::Or, variables)
    }

    /// Applies the `operator` to the [`Bdd`]s and eliminates the given `variables` using universal
    /// quantification from the result.
    pub fn binary_op_with_for_all<TBooleanOp: BooleanOperator>(
        &mut self,
        left: &Bdd,
        right: &Bdd,
        operator: TBooleanOp,
        variables: &[VariableId],
    ) -> Bdd {
        self.nested_apply(left, right, operator, boolean_operators::And, variables)
    }
}

#[derive(Debug)]
struct NestedApplyState<TOuterCache, TInnerCache, TNodeTable: NodeTableAny> {
    stack: Vec<(TNodeTable::Id, TNodeTable::Id, TNodeTable::VarId)>,
    results: Vec<TNodeTable::Id>,
    outer_task_cache: TOuterCache,
    inner_task_cache: TInnerCache,
    node_table: TNodeTable,
    inner_state: InnerApplyState<TNodeTable>,
}

macro_rules! impl_nested_apply_state_variant {
    ($variant_name:ident, $constructor:ident, $task_cache:ident, $node_table:ident) => {
        type $variant_name<TOuterCache = $task_cache> =
            NestedApplyState<TOuterCache, $task_cache, $node_table>;

        fn $constructor(
            left: <$node_table as NodeTableAny>::Id,
            right: <$node_table as NodeTableAny>::Id,
            node_table: $node_table,
        ) -> $variant_name {
            let undefined_var = <$node_table as NodeTableAny>::VarId::undefined();
            NestedApplyState {
                stack: vec![(left, right, undefined_var)],
                results: Vec::new(),
                outer_task_cache: $task_cache::default(),
                inner_task_cache: $task_cache::default(),
                node_table,
                inner_state: InnerApplyState::default(),
            }
        }
    };
}

impl_nested_apply_state_variant!(
    NestedApplyState16,
    default_state_16,
    TaskCache16,
    NodeTable16
);

impl_nested_apply_state_variant!(
    NestedApplyState32,
    default_state_32,
    TaskCache32,
    NodeTable32
);

impl_nested_apply_state_variant!(
    NestedApplyState64,
    default_state_64,
    TaskCache64,
    NodeTable64
);

macro_rules! impl_nested_apply_state_conversion_simple {
    ($from_variant:ident, $to_variant:ident) => {
        impl<TCacheIn, TCacheOut: From<TCacheIn>> From<$from_variant<TCacheIn>>
            for $to_variant<TCacheOut>
        {
            fn from(state: $from_variant<TCacheIn>) -> Self {
                Self {
                    stack: state
                        .stack
                        .into_iter()
                        .map(|(left, right, var)| (left.into(), right.into(), var.into()))
                        .collect(),
                    results: state.results.into_iter().map(|id| id.into()).collect(),
                    outer_task_cache: state.outer_task_cache.into(),
                    inner_task_cache: state.inner_task_cache.into(),
                    node_table: state.node_table.into(),
                    inner_state: state.inner_state.into(),
                }
            }
        }
    };
}

impl_nested_apply_state_conversion_simple!(NestedApplyState16, NestedApplyState32);
impl_nested_apply_state_conversion_simple!(NestedApplyState32, NestedApplyState64);

fn nested_apply_any<
    TNodeTable: NodeTableAny,
    TOuterOp: BooleanOperator,
    TInnerOp: BooleanOperator,
    TTrigger: Fn(TNodeTable::VarId) -> bool,
    TOuterCache: TaskCacheAny<ResultId = TNodeTable::Id>,
    TInnerCache: TaskCacheAny<ResultId = TNodeTable::Id>,
>(
    outer_op: TOuterOp,
    inner_op: TInnerOp,
    trigger: TTrigger,
    state: NestedApplyState<TOuterCache, TInnerCache, TNodeTable>,
) -> Result<(TNodeTable::Id, TNodeTable), NestedApplyState<TOuterCache, TInnerCache, TNodeTable>> {
    // The outer operator should only short-circuit due to terminals, otherwise
    // it messes with algorithm logic.
    let outer_op = outer_op.for_split::<TNodeTable::Id, TNodeTable::Id, TNodeTable::Id>();
    let inner_op = inner_op.for_shared::<TNodeTable::Id>();

    let NestedApplyState {
        mut stack,
        mut results,
        mut outer_task_cache,
        mut inner_task_cache,
        mut node_table,
        mut inner_state,
    } = state;

    while let Some((left_id, right_id, variable)) = stack.pop() {
        // Check if the result is known because the operation short-circuited
        // the computation.
        let result = outer_op(left_id, right_id);
        if !result.is_undefined() {
            results.push(result);
            continue;
        }

        if variable.is_undefined() {
            // The task has not been expanded yet

            let left_node: &<TNodeTable as NodeTableAny>::Node =
                unsafe { node_table.get_node_unchecked(left_id) };
            let right_node = unsafe { node_table.get_node_unchecked(right_id) };

            let use_cache = left_node.has_many_parents() || right_node.has_many_parents();

            if use_cache {
                let result = outer_task_cache.get((left_id, right_id));
                if !result.is_undefined() {
                    results.push(result);
                    continue;
                }
            }

            let left_variable = left_node.variable();
            let right_variable = right_node.variable();

            let mut variable = left_variable.min(right_variable);

            let (left_low, left_high) = if variable == left_variable {
                (left_node.low(), left_node.high())
            } else {
                (left_id, left_id)
            };

            let (right_low, right_high) = if variable == right_variable {
                (right_node.low(), right_node.high())
            } else {
                (right_id, right_id)
            };

            variable.set_use_cache(use_cache);
            stack.push((left_id, right_id, variable));
            stack.push((left_high, right_high, TNodeTable::VarId::undefined()));
            stack.push((left_low, right_low, TNodeTable::VarId::undefined()));

            continue;
        }
        let high_result = results.pop().expect("high result present in result stack");
        let low_result = results.pop().expect("low result present in result stack");

        let node_id = if trigger(variable) {
            if inner_state.is_empty() {
                // Construct the starting state for inner `apply`.
                inner_state = InnerApplyState {
                    stack: vec![(low_result, high_result, TNodeTable::VarId::undefined())],
                    results: Vec::new(),
                };
            }

            match inner_apply_any(
                &inner_op,
                // Make sure that in the next iteration, the inner state is
                // empty, so that it is correctly initialized.
                std::mem::take(&mut inner_state),
                &mut inner_task_cache,
                &mut node_table,
            ) {
                Ok(id) => id,
                Err(inner_state) => {
                    results.push(low_result);
                    results.push(high_result);
                    stack.push((left_id, right_id, variable));

                    return Err(NestedApplyState {
                        stack,
                        results,
                        outer_task_cache,
                        inner_task_cache,
                        node_table,
                        inner_state,
                    });
                }
            }
        } else {
            match node_table.ensure_node(variable, low_result, high_result) {
                Ok(id) => id,
                Err(_) => {
                    results.push(low_result);
                    results.push(high_result);
                    stack.push((left_id, right_id, variable));
                    return Err(NestedApplyState {
                        stack,
                        results,
                        outer_task_cache,
                        inner_task_cache,
                        node_table,
                        inner_state: InnerApplyState::default(),
                    });
                }
            }
        };
        if variable.use_cache() {
            outer_task_cache.set((left_id, right_id), node_id);
        }
        results.push(node_id);
    }

    let root = results.pop().expect("root result present in result stack");
    debug_assert!(results.is_empty());

    Ok((root, node_table))
}

fn nested_apply_16_bit<TOuterOp: BooleanOperator, TInnerOp: BooleanOperator>(
    node_table: NodeTable16,
    left: NodeId16,
    right: NodeId16,
    outer_op: TOuterOp,
    inner_op: TInnerOp,
    variables: &[VariableId],
) -> (NodeId, NodeTable) {
    let variable_set: FxHashSet<VarIdPacked16> =
        FxHashSet::from_iter(variables.iter().map(|&v| v.unchecked_into()));

    let trigger = |var: VarIdPacked16| variable_set.contains(&var);
    let state = default_state_16(left, right, node_table);
    let result_16 = nested_apply_any(outer_op, inner_op, trigger, state);
    let state = match result_16 {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size16(table)),
        Err(state) => state,
    };

    let trigger = |var: VarIdPacked32| variable_set.contains(&var.unchecked_into());
    let state: NestedApplyState32<TaskCache16<NodeId32>> = state.into();
    let result_32 = nested_apply_any(outer_op, inner_op, trigger, state);
    let state = match result_32 {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size32(table)),
        Err(state) => state,
    };

    let trigger = |var: VarIdPacked64| variable_set.contains(&var.unchecked_into());
    let state: NestedApplyState64<TaskCache16<NodeId64>> = state.into();
    let result_64 = nested_apply_any(outer_op, inner_op, trigger, state);
    match result_64 {
        Ok((root, table)) => (NodeId::unchecked_from(root), NodeTable::Size64(table)),
        Err(_state) => unreachable!("BDD does not fit into 64-bit bounds."),
    }
}

fn nested_apply_32_bit<TOuterOp: BooleanOperator, TInnerOp: BooleanOperator>(
    node_table: NodeTable32,
    left: NodeId32,
    right: NodeId32,
    outer_op: TOuterOp,
    inner_op: TInnerOp,
    variables: &[VariableId],
) -> (NodeId, NodeTable) {
    let variable_set: FxHashSet<VarIdPacked32> =
        FxHashSet::from_iter(variables.iter().map(|&v| v.unchecked_into()));

    let trigger = |var: VarIdPacked32| variable_set.contains(&var);
    let state = default_state_32(left, right, node_table);
    let result_32 = nested_apply_any(outer_op, inner_op, trigger, state);
    let state = match result_32 {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size32(table)),
        Err(state) => state,
    };

    let trigger = |var: VarIdPacked64| variable_set.contains(&var.unchecked_into());
    let state: NestedApplyState64<TaskCache32<NodeId64>> = state.into();
    let result_64 = nested_apply_any(outer_op, inner_op, trigger, state);
    match result_64 {
        Ok((root, table)) => (NodeId::unchecked_from(root), NodeTable::Size64(table)),
        Err(_state) => unreachable!("BDD does not fit into 64-bit bounds."),
    }
}

fn nested_apply_64_bit<TOuterOp: BooleanOperator, TInnerOp: BooleanOperator>(
    node_table: NodeTable64,
    left: NodeId64,
    right: NodeId64,
    outer_op: TOuterOp,
    inner_op: TInnerOp,
    variables: &[VariableId],
) -> (NodeId, NodeTable) {
    let variable_set: FxHashSet<VarIdPacked64> =
        FxHashSet::from_iter(variables.iter().map(|&v| v.unchecked_into()));

    let trigger = |var: VarIdPacked64| variable_set.contains(&var);
    let state = default_state_64(left, right, node_table);
    let result_64 = nested_apply_any(outer_op, inner_op, trigger, state);
    match result_64 {
        Ok((root, table)) => (NodeId::unchecked_from(root), NodeTable::Size64(table)),
        Err(_state) => unreachable!("BDD does not fit into 64-bit bounds."),
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        boolean_operators,
        shared::{Bdd, manager::BddManager},
        variable_id::VariableId,
    };

    #[test]
    fn basic_nested_apply_invariants() {
        let mut m = BddManager::no_gc();

        let v_a = VariableId::from(1u32);
        let v_b = VariableId::from(2u32);

        let a = m.new_bdd_literal(v_a, true);
        let b = m.new_bdd_literal(v_b, true);
        let tt = m.new_bdd_true();
        let ff = m.new_bdd_false();

        // True / false constants
        assert_eq!(m.exists(&tt, &[v_a]), tt);
        assert_eq!(m.exists(&ff, &[v_a]), ff);
        assert_eq!(m.for_all(&tt, &[v_a]), tt);
        assert_eq!(m.for_all(&ff, &[v_a]), ff);

        // Quantifying over the same variable
        assert_eq!(m.exists(&a, &[v_a]), tt);
        assert_eq!(m.for_all(&a, &[v_a]), ff);

        // Quantifying over independent variables
        assert_eq!(m.exists(&a, &[v_b]), a);
        assert_eq!(m.for_all(&a, &[v_b]), a);

        // Quantifying over an empty set of variables
        let a_and_b = m.and(&a, &b);
        assert_eq!(m.exists(&a_and_b, &[]), a_and_b);
        assert_eq!(m.for_all(&a_and_b, &[]), a_and_b);
    }

    #[test]
    fn nested_apply_quantifier_distributivity() {
        let mut m = BddManager::no_gc();

        let v_x = VariableId::from(1u32);
        let v_y = VariableId::from(2u32);
        let v_z = VariableId::from(3u32);
        let v_w = VariableId::from(4u32);

        let x = m.new_bdd_literal(v_x, true);
        let y = m.new_bdd_literal(v_y, true);
        let z = m.new_bdd_literal(v_z, true);
        let w = m.new_bdd_literal(v_w, true);

        // Create f(x) = x ∧ y (depends on x)
        let f_x = m.and(&x, &y);

        // Create g = z <=> w (independent of x)
        let g = m.iff(&z, &w);

        // Calculate left side: ∃x.(f(x) ∧ g)
        let left_side = m.binary_op_with_exists(&f_x, &g, boolean_operators::And, &[v_x]);

        // Calculate right side: (∃x.f(x)) ∧ g
        let right_exists = m.exists(&f_x, &[v_x]);
        let right_side = m.and(&right_exists, &g);

        assert_eq!(left_side, right_side);

        // Also test for universal quantification: ∀x.(f(x) ∨ g) = (∀x.f(x)) ∨ g
        let left_side_forall = m.binary_op_with_for_all(&f_x, &g, boolean_operators::Or, &[v_x]);

        let right_forall = m.for_all(&f_x, &[v_x]);
        let right_side_forall = m.or(&right_forall, &g);

        assert_eq!(left_side_forall, right_side_forall);
    }

    #[test]
    fn nested_apply_quantifier_commutativity() {
        let mut m = BddManager::no_gc();

        let v_x = VariableId::from(1u32);
        let v_y = VariableId::from(2u32);
        let v_z = VariableId::from(3u32);
        let v_w = VariableId::from(4u32);

        let x = m.new_bdd_literal(v_x, true);
        let y = m.new_bdd_literal(v_y, true);
        let z = m.new_bdd_literal(v_z, true);
        let w = m.new_bdd_literal(v_w, true);

        // Create a formula: ((x ∧ z ∧ w) ∨ (x ∧ z) ∨ w) ∧ (y xor z)
        let xz = m.and(&x, &z);
        let xzw = m.and(&xz, &w);
        let xz_or_w = m.or(&xz, &w);
        let left = m.or(&xzw, &xz_or_w);
        let xor = m.xor(&y, &z);
        let formula = m.and(&left, &xor);

        // Test that the order of quantification doesn't matter for the same quantifier
        let exists_x = m.exists(&formula, &[v_x]);
        let exists_y = m.exists(&formula, &[v_y]);
        let exists_x_then_y = m.exists(&exists_x, &[v_y]);
        let exists_y_then_x = m.exists(&exists_y, &[v_x]);
        let exists_xy = m.exists(&formula, &[v_x, v_y]);

        assert_eq!(exists_x_then_y, exists_y_then_x);
        assert_eq!(exists_x_then_y, exists_xy);

        // Create a formula: (x => z) ∧ (y => w)
        let limp = m.implies(&x, &z);
        let rimp = m.implies(&y, &w);
        let formula = m.and(&limp, &rimp);

        let forall_x = m.for_all(&formula, &[v_x]);
        let forall_y = m.for_all(&formula, &[v_y]);
        let forall_x_then_y = m.for_all(&forall_x, &[v_y]);
        let forall_y_then_x = m.for_all(&forall_y, &[v_x]);
        let forall_xy = m.for_all(&formula, &[v_x, v_y]);

        assert_eq!(forall_x_then_y, forall_y_then_x);
        assert_eq!(forall_x_then_y, forall_xy);
    }

    #[test]
    fn nested_apply_without_quantification_is_equivalent_to_apply() {
        let mut manager = BddManager::no_gc();

        let v1 = VariableId::from(1u32);
        let v2 = VariableId::from(2u32);
        let v3 = VariableId::from(3u32);
        let v4 = VariableId::from(4u32);
        let v5 = VariableId::from(5u32);

        // High variable IDs to force different BDD sizes
        let v_high16 = VariableId::from(1u16 << 8);
        let v_high32 = VariableId::from(1u32 << 24);

        // Create several complex BDDs
        let x1 = manager.new_bdd_literal(v1, true);
        let x2 = manager.new_bdd_literal(v2, true);
        let x3 = manager.new_bdd_literal(v3, true);
        let x4 = manager.new_bdd_literal(v4, true);
        let x5 = manager.new_bdd_literal(v5, true);

        let high16 = manager.new_bdd_literal(v_high16, true);
        let high32 = manager.new_bdd_literal(v_high32, true);

        // Create more complex BDDs
        let x2_implies = manager.implies(&x1, &x2);
        let x4_iff_x5 = manager.iff(&x4, &x5);
        let x3_xor_x4iff = manager.xor(&x3, &x4_iff_x5);
        let e1 = manager.and(&x2_implies, &x3_xor_x4iff);

        let x1_and_x2 = manager.and(&x1, &x2);
        let x1x2_or_x3 = manager.or(&x1_and_x2, &x3);
        let x4_and_x5 = manager.and(&x4, &x5);
        let xor_part = manager.xor(&x1x2_or_x3, &x4_and_x5);
        let e2 = manager.and(&xor_part, &high16);

        let x3_implies_x4 = manager.implies(&x3, &x4);
        let implies_or_x5 = manager.or(&x3_implies_x4, &x5);
        let x1_or_x2 = manager.or(&x1, &x2);
        let iff_part = manager.iff(&implies_or_x5, &x1_or_x2);
        let e3 = manager.and(&iff_part, &high32);

        // Test nested_apply vs. regular apply for AND operation
        let result1 = manager.binary_op_with_exists(&e1, &e2, boolean_operators::And, &[]);
        let result2 = manager.and(&e1, &e2);
        assert_eq!(result1, result2);

        // Test nested_apply vs. regular apply for IFF operation
        let result3 = manager.binary_op_with_for_all(&e1, &e3, boolean_operators::Iff, &[]);
        let result4 = manager.iff(&e1, &e3);
        assert_eq!(result3, result4);
    }

    #[test]
    fn nested_apply_bit_width_variants() {
        // Should test that the manager uses nested apply correctly with different bit widths.
        // We will gradually force the manager to increase its bit-width by adding extra
        // variables of a specific width.

        let mut manager = BddManager::no_gc();

        let v1 = VariableId::from(u16::MAX >> 4);
        let v2 = VariableId::from(u32::MAX >> 4);
        let v3 = VariableId::new_long(u64::MAX >> 4).unwrap();

        for v in [v1, v2, v3] {
            let v_true = manager.new_bdd_literal(v, true);
            let v_false = manager.new_bdd_literal(v, false);

            // exists v1. (v1 & v1) is tautology
            assert!(
                manager
                    .binary_op_with_exists(&v_true, &v_true, boolean_operators::And, &[v])
                    .is_true()
            );
            // exists v1. (v1 & !v1) is contradiction
            assert!(
                manager
                    .binary_op_with_exists(&v_true, &v_false, boolean_operators::And, &[v])
                    .is_false()
            );
            // forall v1. (v1 | !v1) is tautology
            assert!(
                manager
                    .binary_op_with_for_all(&v_true, &v_false, boolean_operators::Or, &[v])
                    .is_true()
            );
            // forall v1. (v1 | v1) is contradiction
            assert!(
                manager
                    .binary_op_with_for_all(&v_true, &v_true, boolean_operators::Or, &[v])
                    .is_false()
            );
        }
    }

    #[test]
    fn nested_apply_growth_test() {
        // Should test that the manager can reuse state when the data structures need to grow
        // during nested apply. To test this, we use a variant of the ripple carry adder from
        // normal BDD tests. This is not particularly relevant for nested apply operations,
        // but we just need to test that the growth phase actually completes successfully.

        pub fn ripple_carry_adder_with_projection(manager: &mut BddManager, num_vars: u32) -> Bdd {
            let mut result = manager.new_bdd_false();
            for x in 0..(num_vars / 2) {
                let x1 = manager.new_bdd_literal(VariableId::new(x), true);
                let x2 = manager.new_bdd_literal(VariableId::new(x + num_vars / 2), true);
                // To make it a bit interesting, we always erase some previously used variable.
                // This means we are not computing ripple carry adder after all, but at least
                // it tests the nested `apply` operator.
                let and = manager.binary_op_with_exists(&x1, &x2, boolean_operators::And, &[]);
                result = manager.binary_op_with_exists(
                    &result,
                    &and,
                    boolean_operators::Or,
                    &[VariableId::new(x / 4)],
                );
            }
            result
        }

        let mut manager = BddManager::no_gc();

        let result = ripple_carry_adder_with_projection(&mut manager, 4);
        assert_eq!(manager.node_count(&result), 6);

        let result = ripple_carry_adder_with_projection(&mut manager, 8);
        assert_eq!(manager.node_count(&result), 24);

        let result = ripple_carry_adder_with_projection(&mut manager, 16);
        assert_eq!(manager.node_count(&result), 256);

        let result = ripple_carry_adder_with_projection(&mut manager, 24);
        assert_eq!(manager.node_count(&result), 2560);

        let result = ripple_carry_adder_with_projection(&mut manager, 32);
        assert_eq!(manager.node_count(&result), 24576);

        let result = ripple_carry_adder_with_projection(&mut manager, 40);
        assert_eq!(manager.node_count(&result), 229376);
    }
}
