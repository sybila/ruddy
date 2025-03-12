use std::{
    cell::Cell,
    rc::{Rc, Weak},
};

use crate::{
    bdd_node::BddNodeAny,
    boolean_operators::{lift_operator, TriBool},
    conversion::{UncheckedFrom, UncheckedInto},
    node_id::{NodeId, NodeId16, NodeId32, NodeId64, NodeIdAny},
    node_table::{NodeTable, NodeTable16, NodeTable32, NodeTable64, NodeTableAny},
    task_cache::{TaskCache16, TaskCache32, TaskCache64, TaskCacheAny},
    variable_id::{VarIdPackedAny, VariableId},
};

use crate::node_table::GarbageCollector;

use replace_with::replace_with_or_default;

#[derive(Debug, Default)]
pub struct BddManager {
    unique_table: NodeTable,
    roots: Vec<Weak<Cell<NodeId>>>,
}

#[derive(Debug, Clone)]
pub struct Bdd {
    root: Rc<Cell<NodeId>>,
}

impl Bdd {
    fn new(root: NodeId) -> Self {
        Self {
            root: Rc::new(Cell::new(root)),
        }
    }

    fn root_weak(&self) -> Weak<Cell<NodeId>> {
        Rc::downgrade(&self.root)
    }
}

impl PartialEq for Bdd {
    fn eq(&self, other: &Self) -> bool {
        self.root.get() == other.root.get()
    }
}

impl Eq for Bdd {}

impl BddManager {
    pub fn new() -> Self {
        Self {
            unique_table: NodeTable::Size16(Default::default()),
            roots: Default::default(),
        }
    }

    pub fn new_bdd_false(&self) -> Bdd {
        Bdd::new(NodeId::zero())
    }

    pub fn new_bdd_true(&self) -> Bdd {
        Bdd::new(NodeId::one())
    }

    fn grow(&mut self) {
        replace_with_or_default(&mut self.unique_table, |table| match table {
            NodeTable::Size16(table) => NodeTable::Size32(table.into()),
            NodeTable::Size32(table) => NodeTable::Size64(table.into()),
            table64 => table64,
        });
    }

    fn grow_to_64(&mut self) {
        replace_with_or_default(&mut self.unique_table, |table| match table {
            NodeTable::Size16(table) => NodeTable::Size64(table.into()),
            NodeTable::Size32(table) => NodeTable::Size64(table.into()),
            table64 => table64,
        });
    }

    pub fn new_bdd_literal(&mut self, variable: VariableId, value: bool) -> Bdd {
        match &self.unique_table {
            NodeTable::Size16(_) if variable.fits_only_in_packed64() => {
                self.grow_to_64();
            }
            NodeTable::Size32(_) if variable.fits_only_in_packed64() => {
                self.grow();
            }
            NodeTable::Size16(_) if variable.fits_only_in_packed32() => {
                self.grow();
            }
            _ => {}
        }

        if self.unique_table.is_full() {
            self.grow();
        }

        let root: NodeId = match &mut self.unique_table {
            NodeTable::Size16(table) => {
                let root = table
                    .ensure_literal(variable.unchecked_into(), value)
                    .expect("ensuring literal after growth should always succeed");
                root.unchecked_into()
            }
            NodeTable::Size32(table) => {
                let root = table
                    .ensure_literal(variable.unchecked_into(), value)
                    .expect("ensuring literal after growth should always succeed");
                root.unchecked_into()
            }
            NodeTable::Size64(table) => {
                let root = table
                    .ensure_literal(variable.unchecked_into(), value)
                    .expect("TODO: 64-bit ensure_literal failed");
                root.unchecked_into()
            }
        };

        let bdd = Bdd::new(root);
        self.roots.push(bdd.root_weak());
        bdd
    }

    pub fn collect_garbage(&mut self) {
        replace_with_or_default(&mut self.unique_table, |table| match table {
            NodeTable::Size16(table) => table.collect_garbage(&mut self.roots),
            NodeTable::Size32(table) => table.collect_garbage(&mut self.roots),
            NodeTable::Size64(table) => table.collect_garbage(&mut self.roots),
        });
    }

    fn apply<TTriBoolOp: Fn(TriBool, TriBool) -> TriBool>(
        &mut self,
        left: &Bdd,
        right: &Bdd,
        operator: TTriBoolOp,
    ) -> Bdd {
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
        self.apply(left, right, TriBool::and)
    }

    /// Calculate a [`Bdd`] representing the boolean formula `left || right` (disjunction).
    pub fn or(&mut self, left: &Bdd, right: &Bdd) -> Bdd {
        self.apply(left, right, TriBool::or)
    }

    /// Calculate a [`Bdd`] representing the boolean formula `left ^ right` (xor; non-equivalence).
    pub fn xor(&mut self, left: &Bdd, right: &Bdd) -> Bdd {
        self.apply(left, right, TriBool::xor)
    }

    /// Calculate a [`Bdd`] representing the boolean formula `left => right` (implication).
    pub fn implies(&mut self, left: &Bdd, right: &Bdd) -> Bdd {
        self.apply(left, right, TriBool::implies)
    }

    /// Calculate a [`Bdd`] representing the boolean formula `left <=> right` (equivalence).
    pub fn iff(&mut self, left: &Bdd, right: &Bdd) -> Bdd {
        self.apply(left, right, TriBool::iff)
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
    TBooleanOp: Fn(TNodeTable::Id, TNodeTable::Id) -> TNodeTable::Id,
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
    TBooleanOp: Fn(TNodeTable::Id, TNodeTable::Id) -> TNodeTable::Id,
    TTaskCache: TaskCacheAny<ResultId = TNodeTable::Id>,
>(
    operator: TBooleanOp,
    state: ApplyState<TNodeTable, TTaskCache>,
) -> Result<(TNodeTable::Id, TNodeTable), ApplyState<TNodeTable, TTaskCache>> {
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

fn apply_16_bit<TTriBoolOp: Fn(TriBool, TriBool) -> TriBool>(
    node_table: NodeTable16,
    left: NodeId16,
    right: NodeId16,
    operator: TTriBoolOp,
) -> (NodeId, NodeTable) {
    let state = match apply_any_default_state::<_, _, TaskCache16<NodeId16>>(
        left,
        right,
        node_table,
        lift_operator(&operator),
    ) {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size16(table)),
        Err(state) => state,
    };

    let state = match apply_any(lift_operator(&operator), state.into()) {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size32(table)),
        Err(state) => state,
    };

    let (root, table) =
        apply_any(lift_operator(operator), state.into()).expect("64-bit operation failed");
    (NodeId::unchecked_from(root), NodeTable::Size64(table))
}

fn apply_32_bit<TTriBoolOp: Fn(TriBool, TriBool) -> TriBool>(
    node_table: NodeTable32,
    left: NodeId32,
    right: NodeId32,
    operator: TTriBoolOp,
) -> (NodeId, NodeTable) {
    let state = match apply_any_default_state::<_, _, TaskCache32<NodeId32>>(
        left,
        right,
        node_table,
        lift_operator(&operator),
    ) {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size32(table)),
        Err(state) => state,
    };

    let (root, table) =
        apply_any(lift_operator(operator), state.into()).expect("64-bit operation failed");
    (NodeId::unchecked_from(root), NodeTable::Size64(table))
}

fn apply_64_bit<TTriBoolOp: Fn(TriBool, TriBool) -> TriBool>(
    node_table: NodeTable64,
    left: NodeId64,
    right: NodeId64,
    operator: TTriBoolOp,
) -> (NodeId, NodeTable) {
    let (root, table) = apply_any_default_state::<_, _, TaskCache64<NodeId64>>(
        left,
        right,
        node_table,
        lift_operator(operator),
    )
    .expect("TODO: 64-bit operation failed");
    (NodeId::unchecked_from(root), NodeTable::Size64(table))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        node_id::{NodeId32, NodeId64, NodeIdAny},
        variable_id::{VarIdPacked32, VarIdPacked64, VariableId},
    };

    fn next_ripple_carry_adder(
        manager: &mut BddManager,
        prev: &Bdd,
        low_var: VariableId,
        high_var: VariableId,
    ) -> Bdd {
        let l = manager.new_bdd_literal(low_var, true);
        let r = manager.new_bdd_literal(high_var, true);
        let prod = manager.and(&l, &r);
        manager.or(prev, &prod)
    }

    #[test]
    fn manager_grows_from_16_to_32() {
        // This test is mostly a sanity check to test whether apply correctly handles
        // the boundary case of growing from 16 to 32 bits, when adding a lot of new nodes.

        let mut manager = BddManager::new();
        let n = 16;
        let low_vars: Vec<_> = (1..n).map(VariableId::new).collect();
        let high_vars: Vec<_> = (n + 1..2 * n).map(VariableId::new).collect();

        let mut _bdd = manager.new_bdd_false();

        for i in 0..high_vars.len() - 1 {
            _bdd = next_ripple_carry_adder(&mut manager, &_bdd, low_vars[i], high_vars[i]);

            assert!(matches!(manager.unique_table, NodeTable::Size16(_)));
        }

        // Now, the node table is "close" to the boundary with roughly 50k nodes.

        let last = high_vars.len() - 1;

        // We expect this to essentially double the table size
        _bdd = next_ripple_carry_adder(&mut manager, &_bdd, low_vars[last], high_vars[last]);

        assert!(matches!(manager.unique_table, NodeTable::Size32(_)));
    }

    #[test]
    fn manager_growth_from_16_to_32_interpersed_with_gc() {
        let mut manager = BddManager::new();
        let n = 16;
        let low_vars: Vec<_> = (1..n).map(VariableId::new).collect();
        let high_vars: Vec<_> = (n + 1..2 * n).map(VariableId::new).collect();

        let mut _bdd = manager.new_bdd_false();

        for i in 0..high_vars.len() - 1 {
            _bdd = next_ripple_carry_adder(&mut manager, &_bdd, low_vars[i], high_vars[i]);

            manager.collect_garbage();
            assert_eq!(manager.unique_table.node_count(), 1 << (i + 2));
            assert!(matches!(manager.unique_table, NodeTable::Size16(_)));
        }

        let last = high_vars.len() - 1;

        _bdd = next_ripple_carry_adder(&mut manager, &_bdd, low_vars[last], high_vars[last]);

        manager.collect_garbage();

        assert!(matches!(manager.unique_table, NodeTable::Size32(_)));
        assert_eq!(manager.unique_table.node_count(), 1 << n);
    }

    #[test]
    fn adding_32_bit_variable_to_16_bit_manager_grows_to_32_bit() {
        let mut manager = BddManager::new();
        let var_num = u32::from(u16::MAX);
        let variable = VariableId::new(var_num);
        manager.new_bdd_literal(variable, true);

        assert!(matches!(manager.unique_table, NodeTable::Size32(_)));
        assert_eq!(manager.unique_table.node_count(), 3);

        match &manager.unique_table {
            NodeTable::Size32(table) => {
                let node = table.get_node(NodeId32::new(2)).unwrap();
                assert_eq!(node.low, NodeId32::zero());
                assert_eq!(node.high, NodeId32::one());
                assert_eq!(node.variable, VarIdPacked32::new(var_num));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn adding_64_bit_variable_to_16_bit_manager_grows_to_64_bit() {
        let mut manager = BddManager::new();
        let var_num = u64::from(u32::MAX);
        let variable = VariableId::new_long(var_num).unwrap();
        manager.new_bdd_literal(variable, true);

        assert!(matches!(manager.unique_table, NodeTable::Size64(_)));
        assert_eq!(manager.unique_table.node_count(), 3);

        match &manager.unique_table {
            NodeTable::Size64(table) => {
                let node = table.get_node(NodeId64::new(2)).unwrap();
                assert_eq!(node.low, NodeId64::zero());
                assert_eq!(node.high, NodeId64::one());
                assert_eq!(node.variable, VarIdPacked64::new(var_num));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn adding_64_bit_variable_to_32_bit_manager_grows_to_64_bit() {
        let mut manager = BddManager::new();
        manager.grow();
        assert!(matches!(manager.unique_table, NodeTable::Size32(_)));
        assert_eq!(manager.unique_table.node_count(), 2);

        let var_num = u64::from(u32::MAX);
        let variable = VariableId::new_long(var_num).unwrap();
        manager.new_bdd_literal(variable, true);

        assert!(matches!(manager.unique_table, NodeTable::Size64(_)));
        assert_eq!(manager.unique_table.node_count(), 3);

        match &manager.unique_table {
            NodeTable::Size64(table) => {
                let node = table.get_node(NodeId64::new(2)).unwrap();
                assert_eq!(node.low, NodeId64::zero());
                assert_eq!(node.high, NodeId64::one());
                assert_eq!(node.variable, VarIdPacked64::new(var_num));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn basic_apply_invariants() {
        // These are obviously not all invariants/equalities, but at least something to
        // check that we have the major corner cases covered.
        let mut m = BddManager::new();

        let a = m.new_bdd_literal(VariableId::from(1u32), true);
        let b = m.new_bdd_literal(VariableId::from(2u32), true);
        let a_n = m.new_bdd_literal(VariableId::from(1u32), false);
        let b_n = m.new_bdd_literal(VariableId::from(2u32), false);
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
}
