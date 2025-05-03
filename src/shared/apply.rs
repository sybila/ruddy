use replace_with::replace_with_or_default;

use crate::{
    bdd_node::BddNodeAny,
    boolean_operators::{self, BooleanOperator},
    conversion::{UncheckedFrom, UncheckedInto},
    node_id::{NodeId, NodeId16, NodeId32, NodeId64, NodeIdAny},
    node_table::{NodeTable, NodeTable16, NodeTable32, NodeTable64, NodeTableAny},
    task_cache::{TaskCache16, TaskCache32, TaskCache64, TaskCacheAny},
    variable_id::VarIdPackedAny,
};

use super::{bdd::Bdd, manager::BddManager};

impl BddManager {
    fn apply<TBooleanOp: BooleanOperator>(
        &mut self,
        left: &Bdd,
        right: &Bdd,
        operator: TBooleanOp,
    ) -> Bdd {
        self.maybe_collect_garbage();

        let mut bdd_root = NodeId::undefined();

        replace_with_or_default(&mut self.unique_table, |table| match table {
            NodeTable::Size16(table) => {
                let (root, table) = apply_16_bit(
                    table,
                    left.root.get().unchecked_into(),
                    right.root.get().unchecked_into(),
                    operator,
                );
                bdd_root = root;
                table
            }
            NodeTable::Size32(table) => {
                let (root, table) = apply_32_bit(
                    table,
                    left.root.get().unchecked_into(),
                    right.root.get().unchecked_into(),
                    operator,
                );
                bdd_root = root;
                table
            }
            NodeTable::Size64(table) => {
                let (root, table) = apply_64_bit(
                    table,
                    left.root.get().unchecked_into(),
                    right.root.get().unchecked_into(),
                    operator,
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

    /// Calculate a [`Bdd`] representing the boolean formula `left && right` (conjunction).
    pub fn and(&mut self, left: &Bdd, right: &Bdd) -> Bdd {
        self.apply(left, right, boolean_operators::And)
    }

    /// Calculate a [`Bdd`] representing the boolean formula `left || right` (disjunction).
    pub fn or(&mut self, left: &Bdd, right: &Bdd) -> Bdd {
        self.apply(left, right, boolean_operators::Or)
    }

    /// Calculate a [`Bdd`] representing the boolean formula `left ^ right` (xor; non-equivalence).
    pub fn xor(&mut self, left: &Bdd, right: &Bdd) -> Bdd {
        self.apply(left, right, boolean_operators::Xor)
    }

    /// Calculate a [`Bdd`] representing the boolean formula `left => right` (implication).
    pub fn implies(&mut self, left: &Bdd, right: &Bdd) -> Bdd {
        self.apply(left, right, boolean_operators::Implies)
    }

    /// Calculate a [`Bdd`] representing the boolean formula `left <=> right` (equivalence).
    pub fn iff(&mut self, left: &Bdd, right: &Bdd) -> Bdd {
        self.apply(left, right, boolean_operators::Iff)
    }
}

#[derive(Debug)]
struct ApplyState<TNodeTable: NodeTableAny, TTaskCache> {
    stack: Vec<(TNodeTable::Id, TNodeTable::Id, TNodeTable::VarId)>,
    results: Vec<TNodeTable::Id>,
    task_cache: TTaskCache,
    node_table: TNodeTable,
}

type NodeTableId<N> = <N as NodeTableAny>::Id;

macro_rules! impl_apply_state_conversion {
    ($from_table:ident, $to_table:ident, $cache:ident) => {
        impl From<ApplyState<$from_table, $cache<NodeTableId<$from_table>>>>
            for ApplyState<$to_table, $cache<NodeTableId<$to_table>>>
        {
            fn from(state: ApplyState<$from_table, $cache<NodeTableId<$from_table>>>) -> Self {
                Self {
                    stack: state
                        .stack
                        .into_iter()
                        .map(|(a, b, c)| (a.into(), b.into(), c.into()))
                        .collect(),
                    results: state.results.into_iter().map(|id| id.into()).collect(),
                    task_cache: state.task_cache.into(),
                    node_table: state.node_table.into(),
                }
            }
        }
    };
}

impl_apply_state_conversion!(NodeTable16, NodeTable32, TaskCache16);
impl_apply_state_conversion!(NodeTable32, NodeTable64, TaskCache16);
impl_apply_state_conversion!(NodeTable32, NodeTable64, TaskCache32);

fn apply_any_default_state<
    TNodeTable: NodeTableAny,
    TBooleanOp: BooleanOperator,
    TTaskCache: TaskCacheAny<ResultId = TNodeTable::Id>,
>(
    left: TNodeTable::Id,
    right: TNodeTable::Id,
    node_table: TNodeTable,
    operator: TBooleanOp,
) -> Result<(TNodeTable::Id, TNodeTable), ApplyState<TNodeTable, TTaskCache>> {
    let state = ApplyState {
        stack: vec![(left, right, TNodeTable::VarId::undefined())],
        results: Vec::new(),
        task_cache: TTaskCache::default(),
        node_table,
    };

    apply_any(operator, state)
}

fn apply_any<
    TNodeTable: NodeTableAny,
    TBooleanOp: BooleanOperator,
    TTaskCache: TaskCacheAny<ResultId = TNodeTable::Id>,
>(
    operator: TBooleanOp,
    state: ApplyState<TNodeTable, TTaskCache>,
) -> Result<(TNodeTable::Id, TNodeTable), ApplyState<TNodeTable, TTaskCache>> {
    let operator = operator.for_shared::<TNodeTable::Id, TNodeTable::Id, TNodeTable::Id>();

    let ApplyState {
        mut stack,
        mut results,
        mut task_cache,
        mut node_table,
    } = state;

    while let Some((left_id, right_id, variable)) = stack.pop() {
        // Check if the result is known because the operation short-circuited
        // the computation.
        let result = operator(left_id, right_id);
        if !result.is_undefined() {
            results.push(result);
            continue;
        }

        if variable.is_undefined() {
            // The task has not been expanded yet.

            let left_node = unsafe { node_table.get_node_unchecked(left_id) };
            let right_node = unsafe { node_table.get_node_unchecked(right_id) };

            let use_cache = left_node.has_many_parents() || right_node.has_many_parents();

            if use_cache {
                let result = task_cache.get((left_id, right_id));
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

        let node_id = match node_table.ensure_node(variable, low_result, high_result) {
            Ok(node_id) => node_id,
            Err(_) => {
                results.push(low_result);
                results.push(high_result);
                stack.push((left_id, right_id, variable));

                return Err(ApplyState {
                    stack,
                    results,
                    task_cache,
                    node_table,
                });
            }
        };

        if variable.use_cache() {
            task_cache.set((left_id, right_id), node_id);
        }
        results.push(node_id);
    }
    let root = results.pop().expect("root result present in result stack");
    debug_assert!(results.is_empty());
    Ok((root, node_table))
}

fn apply_16_bit<TBooleanOp: BooleanOperator>(
    node_table: NodeTable16,
    left: NodeId16,
    right: NodeId16,
    operator: TBooleanOp,
) -> (NodeId, NodeTable) {
    let state = match apply_any_default_state::<_, _, TaskCache16<NodeId16>>(
        left, right, node_table, operator,
    ) {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size16(table)),
        Err(state) => state,
    };

    let state = match apply_any(operator, state.into()) {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size32(table)),
        Err(state) => state,
    };

    let (root, table) = apply_any(operator, state.into()).expect("64-bit operation failed");
    (NodeId::unchecked_from(root), NodeTable::Size64(table))
}

fn apply_32_bit<TBooleanOp: BooleanOperator>(
    node_table: NodeTable32,
    left: NodeId32,
    right: NodeId32,
    operator: TBooleanOp,
) -> (NodeId, NodeTable) {
    let state = match apply_any_default_state::<_, _, TaskCache32<NodeId32>>(
        left, right, node_table, operator,
    ) {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size32(table)),
        Err(state) => state,
    };

    let (root, table) = apply_any(operator, state.into()).expect("64-bit operation failed");
    (NodeId::unchecked_from(root), NodeTable::Size64(table))
}

fn apply_64_bit<TBooleanOp: BooleanOperator>(
    node_table: NodeTable64,
    left: NodeId64,
    right: NodeId64,
    operator: TBooleanOp,
) -> (NodeId, NodeTable) {
    let (root, table) =
        apply_any_default_state::<_, _, TaskCache64<NodeId64>>(left, right, node_table, operator)
            .expect("TODO: 64-bit operation failed");
    (NodeId::unchecked_from(root), NodeTable::Size64(table))
}

#[cfg(test)]
mod tests {
    use crate::{
        shared::manager::BddManager,
        variable_id::{VarIdPacked16, VarIdPacked32, VarIdPacked64, VariableId},
    };

    fn test_basic_apply_invariants(var1: VariableId, var2: VariableId) {
        // These are obviously not all invariants/equalities, but at least something to
        // check that we have the major corner cases covered.
        let mut m = BddManager::no_gc();

        let a = m.new_bdd_literal(var1, true);
        let b = m.new_bdd_literal(var2, true);
        let a_n = m.new_bdd_literal(var1, false);
        let b_n = m.new_bdd_literal(var2, false);
        let tt = m.new_bdd_true();
        let ff = m.new_bdd_false();

        assert_eq!(&m.and(&a, &a), &a);
        assert_eq!(&m.and(&a, &tt), &a);
        assert_eq!(&m.and(&a, &ff), &ff);
        assert_eq!(&m.and(&a, &b), &m.and(&b, &a));

        assert_eq!(&m.or(&a, &a), &a);
        assert_eq!(&m.or(&a, &tt), &tt);
        assert_eq!(&m.or(&a, &ff), &a);
        assert_eq!(&m.or(&a, &b), &m.or(&b, &a));

        assert_eq!(&m.implies(&a, &a), &tt);
        assert_eq!(&m.implies(&a, &tt), &tt);
        assert_eq!(&m.implies(&a, &ff), &a_n);
        assert_eq!(&m.implies(&a, &b), &m.or(&a_n, &b));

        assert_eq!(&m.xor(&a, &a), &ff);
        assert_eq!(&m.xor(&a, &tt), &a_n);
        assert_eq!(&m.xor(&a, &ff), &a);
        let res1 = m.and(&a, &b_n);
        let res2 = m.and(&a_n, &b);
        assert_eq!(&m.xor(&a, &b), &m.or(&res1, &res2));

        assert_eq!(&m.iff(&a, &a), &tt);
        assert_eq!(&m.iff(&a, &tt), &a);
        assert_eq!(&m.iff(&a, &ff), &a_n);
        let res1 = m.and(&a, &b);
        let res2 = m.and(&a_n, &b_n);
        assert_eq!(&m.iff(&a, &b), &m.or(&res1, &res2));
    }

    #[test]
    fn basic_apply_invariants_16() {
        test_basic_apply_invariants(
            VariableId::from(VarIdPacked16::MAX_ID - 2),
            VariableId::from(VarIdPacked16::MAX_ID - 1),
        );
    }

    #[test]
    fn basic_apply_invariants_32() {
        test_basic_apply_invariants(
            VariableId::new(VarIdPacked32::MAX_ID - 2),
            VariableId::new(VarIdPacked32::MAX_ID - 1),
        );
    }

    #[test]
    fn basic_apply_invariants_64() {
        test_basic_apply_invariants(
            VariableId::new_long(VarIdPacked64::MAX_ID - 2).unwrap(),
            VariableId::new_long(VarIdPacked64::MAX_ID - 1).unwrap(),
        );
    }
}
