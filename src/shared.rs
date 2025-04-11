use std::{
    cell::Cell,
    rc::{Rc, Weak},
};

use crate::{
    bdd_node::BddNodeAny,
    boolean_operators::{lift_operator, TriBool},
    conversion::{UncheckedFrom, UncheckedInto},
    nested_apply::{inner_apply_any, InnerApplyState},
    node_id::{NodeId, NodeId16, NodeId32, NodeId64, NodeIdAny},
    node_table::{NodeTable, NodeTable16, NodeTable32, NodeTable64, NodeTableAny},
    task_cache::{TaskCache16, TaskCache32, TaskCache64, TaskCacheAny},
    variable_id::{VarIdPacked16, VarIdPacked32, VarIdPacked64, VarIdPackedAny, VariableId},
};

use crate::node_table::GarbageCollector;

use replace_with::replace_with_or_default;
use rustc_hash::FxHashSet;

#[derive(Debug, Clone, Copy, Default)]
pub enum GarbageCollection {
    Manual,
    #[default]
    Automatic,
}

#[derive(Debug)]
pub struct BddManager {
    unique_table: NodeTable,
    roots: Vec<Weak<Cell<NodeId>>>,
    gc: GarbageCollection,
    nodes_after_last_gc: usize,
}

impl Default for BddManager {
    fn default() -> Self {
        Self {
            unique_table: Default::default(),
            roots: Default::default(),
            gc: Default::default(),
            nodes_after_last_gc: 2,
        }
    }
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

    pub fn is_true(&self) -> bool {
        self.root.get().is_one()
    }

    pub fn is_false(&self) -> bool {
        self.root.get().is_zero()
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
        Self::default()
    }

    pub fn with_gc(self, garbage_collection: GarbageCollection) -> Self {
        Self {
            gc: garbage_collection,
            ..self
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

        self.maybe_collect_garbage();
        bdd
    }

    fn maybe_collect_garbage(&mut self) {
        if !matches!(self.gc, GarbageCollection::Automatic) {
            return;
        }

        const GROWTH_RATIO: usize = 4;

        let nodes_added_since_last_gc = self.unique_table.node_count() - self.nodes_after_last_gc;

        if nodes_added_since_last_gc > self.nodes_after_last_gc.saturating_mul(GROWTH_RATIO) {
            self.collect_garbage();
        }
    }

    pub fn collect_garbage(&mut self) {
        replace_with_or_default(&mut self.unique_table, |table| match table {
            NodeTable::Size16(table) => table.collect_garbage(&mut self.roots),
            NodeTable::Size32(table) => table.collect_garbage(&mut self.roots),
            NodeTable::Size64(table) => table.collect_garbage(&mut self.roots),
        });
        self.nodes_after_last_gc = self.unique_table.node_count();
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

        self.maybe_collect_garbage();
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

    /// Calculate a [`Bdd`] representing the boolean formula `!bdd` (negation).
    pub fn not(&mut self, bdd: &Bdd) -> Bdd {
        let mut bdd_root = NodeId::undefined();

        replace_with_or_default(&mut self.unique_table, |table| match table {
            NodeTable::Size16(table) => {
                let (root, table) = not_16_bit(table, bdd.root.get().unchecked_into());
                bdd_root = root;
                table
            }
            NodeTable::Size32(table) => {
                let (root, table) = not_32_bit(table, bdd.root.get().unchecked_into());
                bdd_root = root;
                table
            }
            NodeTable::Size64(table) => {
                let (root, table) = not_64_bit(table, bdd.root.get().unchecked_into());
                bdd_root = root;
                table
            }
        });

        debug_assert!(!bdd_root.is_undefined());

        let bdd = Bdd::new(bdd_root);
        self.roots.push(bdd.root_weak());

        self.maybe_collect_garbage();
        bdd
    }

    fn nested_apply<
        TTriboolOp1: Fn(TriBool, TriBool) -> TriBool,
        TTriboolOp2: Fn(TriBool, TriBool) -> TriBool,
    >(
        &mut self,
        left: &Bdd,
        right: &Bdd,
        outer_op: TTriboolOp1,
        inner_op: TTriboolOp2,
        variables: &[VariableId],
    ) -> Bdd {
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

        self.maybe_collect_garbage();
        bdd
    }

    /// Eliminates the given `variables` using existential quantification.
    pub fn exists(&mut self, bdd: &Bdd, variables: &[VariableId]) -> Bdd {
        self.binary_op_with_exists(bdd, bdd, TriBool::and, variables)
    }

    /// Eliminates the given `variables` using universal quantification.
    pub fn for_all(&mut self, bdd: &Bdd, variables: &[VariableId]) -> Bdd {
        self.binary_op_with_for_all(bdd, bdd, TriBool::and, variables)
    }

    /// Applies a binary operator to the BDDs and eliminates the given `variables` using existential
    /// quantification from the result.
    pub fn binary_op_with_exists<TTriBoolOp: Fn(TriBool, TriBool) -> TriBool>(
        &mut self,
        left: &Bdd,
        right: &Bdd,
        operator: TTriBoolOp,
        variables: &[VariableId],
    ) -> Bdd {
        self.nested_apply(left, right, operator, TriBool::or, variables)
    }

    /// Applies a binary operator to the BDDs and eliminates the given `variables` using universal
    /// quantification from the result.
    pub fn binary_op_with_for_all<TTriBoolOp: Fn(TriBool, TriBool) -> TriBool>(
        &mut self,
        left: &Bdd,
        right: &Bdd,
        operator: TTriBoolOp,
        variables: &[VariableId],
    ) -> Bdd {
        self.nested_apply(left, right, operator, TriBool::and, variables)
    }
}

fn not_16_bit(node_table: NodeTable16, root: NodeId16) -> (NodeId, NodeTable) {
    let state = match not_default_state(node_table, root) {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size16(table)),
        Err(state) => state,
    };

    let state = match not_any(state.into()) {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size32(table)),
        Err(state) => state,
    };

    let (root, table) = not_any(state.into()).expect("TODO: 64-bit not failed");
    (NodeId::unchecked_from(root), NodeTable::Size64(table))
}

fn not_32_bit(node_table: NodeTable32, root: NodeId32) -> (NodeId, NodeTable) {
    let state = match not_default_state::<_, TaskCache32<NodeId32>>(node_table, root) {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size32(table)),
        Err(state) => state,
    };

    let (root, table) = not_any(state.into()).expect("TODO: 64-bit not failed");
    (NodeId::unchecked_from(root), NodeTable::Size64(table))
}

fn not_64_bit(node_table: NodeTable64, root: NodeId64) -> (NodeId, NodeTable) {
    let (root, table) = (not_default_state::<_, TaskCache32<NodeId64>>(node_table, root))
        .expect("TODO: 64-bit not failed");
    (NodeId::unchecked_from(root), NodeTable::Size64(table))
}

fn not_default_state<
    TNodeTable: NodeTableAny,
    TTaskCache: TaskCacheAny<ResultId = TNodeTable::Id>,
>(
    node_table: TNodeTable,
    root: TNodeTable::Id,
) -> Result<(TNodeTable::Id, TNodeTable), NotState<TNodeTable, TTaskCache>> {
    let stack = vec![(root, TNodeTable::VarId::undefined())];
    let results = Vec::new();
    let task_cache = TTaskCache::default();

    let state = NotState {
        stack,
        results,
        task_cache,
        node_table,
    };

    not_any(state)
}

fn not_any<TNodeTable: NodeTableAny, TTaskCache: TaskCacheAny<ResultId = TNodeTable::Id>>(
    state: NotState<TNodeTable, TTaskCache>,
) -> Result<(TNodeTable::Id, TNodeTable), NotState<TNodeTable, TTaskCache>> {
    let NotState {
        mut stack,
        mut results,
        mut task_cache,
        mut node_table,
    } = state;

    while let Some((id, variable)) = stack.pop() {
        if variable.is_undefined() {
            if id.is_terminal() {
                results.push(id.flipped_if_terminal());
                continue;
            }
            let node = unsafe { node_table.get_node_unchecked(id) };

            let use_cache = node.has_many_parents();

            if use_cache {
                let result = task_cache.get((id, id));
                if !result.is_undefined() {
                    results.push(result);
                    continue;
                }
            }

            let mut variable = node.variable();
            variable.set_use_cache(use_cache);
            stack.push((id, variable));
            stack.push((node.high(), TNodeTable::VarId::undefined()));
            stack.push((node.low(), TNodeTable::VarId::undefined()));
        } else {
            let high_result = results.pop().expect("high result present in result stack");
            let low_result = results.pop().expect("low results present in result stack");

            let new_id = match node_table.ensure_node(variable, low_result, high_result) {
                Ok(id) => id,
                Err(_) => {
                    return {
                        results.push(low_result);
                        results.push(high_result);
                        stack.push((id, variable));
                        Err(NotState {
                            stack,
                            results,
                            task_cache,
                            node_table,
                        })
                    }
                }
            };
            if variable.use_cache() {
                task_cache.set((id, id), new_id);
            }

            results.push(new_id);
        }
    }

    let root = results.pop().expect("root result present in result stack");
    debug_assert!(results.is_empty());
    Ok((root, node_table))
}

#[derive(Debug)]
struct NotState<TNodeTable: NodeTableAny, TTaskCache> {
    stack: Vec<(TNodeTable::Id, TNodeTable::VarId)>,
    results: Vec<TNodeTable::Id>,
    task_cache: TTaskCache,
    node_table: TNodeTable,
}

macro_rules! impl_not_state_conversion {
    ($from_table:ident, $to_table:ident, $cache:ident) => {
        impl From<NotState<$from_table, $cache<NodeTableId<$from_table>>>>
            for NotState<$to_table, $cache<NodeTableId<$to_table>>>
        {
            fn from(state: NotState<$from_table, $cache<NodeTableId<$from_table>>>) -> Self {
                Self {
                    stack: state
                        .stack
                        .into_iter()
                        .map(|(n, v)| (n.into(), v.into()))
                        .collect(),
                    results: state.results.into_iter().map(|n| n.into()).collect(),
                    task_cache: state.task_cache.into(),
                    node_table: state.node_table.into(),
                }
            }
        }
    };
}

impl_not_state_conversion!(NodeTable16, NodeTable32, TaskCache16);
impl_not_state_conversion!(NodeTable32, NodeTable64, TaskCache16);
impl_not_state_conversion!(NodeTable32, NodeTable64, TaskCache32);

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

#[derive(Debug)]
struct NestedApplyState<TTaskCache, TNodeTable: NodeTableAny> {
    stack: Vec<(TNodeTable::Id, TNodeTable::Id, TNodeTable::VarId)>,
    results: Vec<TNodeTable::Id>,
    outer_task_cache: TTaskCache,
    inner_task_cache: TTaskCache,
    node_table: TNodeTable,
    inner_state: InnerApplyState<TNodeTable>,
}

macro_rules! impl_nested_apply_state_conversion {
    ($from_table:ident, $to_table:ident, $cache:ident) => {
        impl From<NestedApplyState<$cache<NodeTableId<$from_table>>, $from_table>>
            for NestedApplyState<$cache<NodeTableId<$to_table>>, $to_table>
        {
            fn from(
                state: NestedApplyState<$cache<NodeTableId<$from_table>>, $from_table>,
            ) -> Self {
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

impl_nested_apply_state_conversion!(NodeTable16, NodeTable32, TaskCache16);
impl_nested_apply_state_conversion!(NodeTable32, NodeTable64, TaskCache16);
impl_nested_apply_state_conversion!(NodeTable32, NodeTable64, TaskCache32);

fn nested_apply_any_default_state<
    TNodeTable: NodeTableAny,
    TOuterOp: Fn(TNodeTable::Id, TNodeTable::Id) -> TNodeTable::Id,
    TInnerOp: Fn(TNodeTable::Id, TNodeTable::Id) -> TNodeTable::Id,
    TTrigger: Fn(TNodeTable::VarId) -> bool,
    TTaskCache: TaskCacheAny<ResultId = TNodeTable::Id>,
>(
    left: TNodeTable::Id,
    right: TNodeTable::Id,
    node_table: TNodeTable,
    outer_op: TOuterOp,
    inner_op: TInnerOp,
    trigger: TTrigger,
) -> Result<(TNodeTable::Id, TNodeTable), NestedApplyState<TTaskCache, TNodeTable>> {
    let state = NestedApplyState {
        stack: vec![(left, right, TNodeTable::VarId::undefined())],
        results: Vec::new(),
        outer_task_cache: TTaskCache::default(),
        inner_task_cache: TTaskCache::default(),
        node_table,
        inner_state: InnerApplyState::default(),
    };

    nested_apply_any(outer_op, inner_op, trigger, state)
}

fn nested_apply_any<
    TNodeTable: NodeTableAny,
    TOuterOp: Fn(TNodeTable::Id, TNodeTable::Id) -> TNodeTable::Id,
    TInnerOp: Fn(TNodeTable::Id, TNodeTable::Id) -> TNodeTable::Id,
    TTrigger: Fn(TNodeTable::VarId) -> bool,
    TTaskCache: TaskCacheAny<ResultId = TNodeTable::Id>,
>(
    outer_op: TOuterOp,
    inner_op: TInnerOp,
    trigger: TTrigger,
    state: NestedApplyState<TTaskCache, TNodeTable>,
) -> Result<(TNodeTable::Id, TNodeTable), NestedApplyState<TTaskCache, TNodeTable>> {
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
                // Construct the starting state for inner apply
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

fn nested_apply_16_bit<
    TTriBoolOp1: Fn(TriBool, TriBool) -> TriBool,
    TTriBoolOp2: Fn(TriBool, TriBool) -> TriBool,
>(
    node_table: NodeTable16,
    left: NodeId16,
    right: NodeId16,
    outer_op: TTriBoolOp1,
    inner_op: TTriBoolOp2,
    variables: &[VariableId],
) -> (NodeId, NodeTable) {
    let variable_set: FxHashSet<VarIdPacked16> =
        FxHashSet::from_iter(variables.iter().map(|&v| v.unchecked_into()));

    let trigger = |var: VarIdPacked16| variable_set.contains(&var);

    let state = match nested_apply_any_default_state(
        left,
        right,
        node_table,
        lift_operator(&outer_op),
        lift_operator(&inner_op),
        trigger,
    ) {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size16(table)),
        Err(state) => state,
    };

    let trigger = |var: VarIdPacked32| variable_set.contains(&var.unchecked_into());

    let state = match nested_apply_any(
        lift_operator(&outer_op),
        lift_operator(&inner_op),
        trigger,
        state.into(),
    ) {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size32(table)),
        Err(state) => state,
    };

    let trigger = |var: VarIdPacked64| variable_set.contains(&var.unchecked_into());

    let (root, table) = nested_apply_any(
        lift_operator(&outer_op),
        lift_operator(&inner_op),
        trigger,
        state.into(),
    )
    .expect("64-bit operation failed");
    (NodeId::unchecked_from(root), NodeTable::Size64(table))
}

fn nested_apply_32_bit<
    TriBoolOp1: Fn(TriBool, TriBool) -> TriBool,
    TriBoolOp2: Fn(TriBool, TriBool) -> TriBool,
>(
    node_table: NodeTable32,
    left: NodeId32,
    right: NodeId32,
    outer_op: TriBoolOp1,
    inner_op: TriBoolOp2,
    variables: &[VariableId],
) -> (NodeId, NodeTable) {
    let variable_set: FxHashSet<VarIdPacked32> =
        FxHashSet::from_iter(variables.iter().map(|&v| v.unchecked_into()));

    let trigger = |var: VarIdPacked32| variable_set.contains(&var);

    let state = match nested_apply_any_default_state::<_, _, _, _, TaskCache32<NodeId32>>(
        left,
        right,
        node_table,
        lift_operator(&outer_op),
        lift_operator(&inner_op),
        trigger,
    ) {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size32(table)),
        Err(state) => state,
    };

    let trigger = |var: VarIdPacked64| variable_set.contains(&var.unchecked_into());

    let (root, table) = nested_apply_any(
        lift_operator(&outer_op),
        lift_operator(&inner_op),
        trigger,
        state.into(),
    )
    .expect("64-bit operation failed");

    (NodeId::unchecked_from(root), NodeTable::Size64(table))
}

fn nested_apply_64_bit<
    TTriBoolOp1: Fn(TriBool, TriBool) -> TriBool,
    TTriBoolOp2: Fn(TriBool, TriBool) -> TriBool,
>(
    node_table: NodeTable64,
    left: NodeId64,
    right: NodeId64,
    outer_op: TTriBoolOp1,
    inner_op: TTriBoolOp2,
    variables: &[VariableId],
) -> (NodeId, NodeTable) {
    let variable_set: FxHashSet<VarIdPacked64> =
        FxHashSet::from_iter(variables.iter().map(|&v| v.unchecked_into()));

    let trigger = |var: VarIdPacked64| variable_set.contains(&var);

    let (root, table) = nested_apply_any_default_state::<_, _, _, _, TaskCache64<NodeId64>>(
        left,
        right,
        node_table,
        lift_operator(outer_op),
        lift_operator(inner_op),
        trigger,
    )
    .expect("64-bit operation failed");

    (NodeId::unchecked_from(root), NodeTable::Size64(table))
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        node_id::{NodeId32, NodeId64, NodeIdAny},
        node_table::NodeTable,
        variable_id::{VarIdPacked16, VarIdPacked32, VarIdPacked64, VariableId},
    };

    impl BddManager {
        fn no_gc() -> Self {
            Self::new().with_gc(GarbageCollection::Manual)
        }
    }

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

        let mut manager = BddManager::no_gc();
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
    fn manager_growth_from_16_to_32_interspersed_with_gc() {
        let mut manager = BddManager::no_gc();
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
    fn maybe_collect_garbage() {
        // A sanity test checking that `maybe_collect_garbage` triggers gc when needed.
        // The exact conditions are not tested right now, as that is subject to change.
        let mut manager = BddManager::no_gc();
        let nodes = 8192;
        // Should be enough to trigger gc in basically every case.
        for i in 2..nodes {
            let var = VariableId::new(u16::MAX as u32 + i);
            let _ = manager.new_bdd_literal(var, true);
        }

        manager.gc = GarbageCollection::Automatic;
        manager.maybe_collect_garbage();

        assert_eq!(manager.unique_table.node_count(), 2);
    }

    #[test]
    fn adding_32_bit_variable_to_16_bit_manager_grows_to_32_bit() {
        let mut manager = BddManager::no_gc();
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
        let mut manager = BddManager::no_gc();
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
        let mut manager = BddManager::no_gc();
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

    #[test]
    fn not() {
        let mut manager = BddManager::no_gc();

        let v1 = VariableId::new(0);
        let v2 = VariableId::new(1);
        let v3 = VariableId::new(2);
        let v4 = VariableId::new(3);

        let a = manager.new_bdd_literal(v1, true);
        let a_n = manager.new_bdd_literal(v1, false);
        let b = manager.new_bdd_literal(v2, true);
        let b_n = manager.new_bdd_literal(v2, false);
        let c = manager.new_bdd_literal(v3, true);
        let c_n = manager.new_bdd_literal(v3, false);
        let d = manager.new_bdd_literal(v4, true);
        let d_n = manager.new_bdd_literal(v4, false);

        let ab = manager.and(&a, &b);
        let cd = manager.and(&c, &d);
        let bdd = manager.or(&ab, &cd);

        let a_nb_n = manager.or(&a_n, &b_n);
        let cd_n = manager.or(&c_n, &d_n);
        let expected = manager.and(&a_nb_n, &cd_n);

        let once = manager.not(&bdd);
        let twice = manager.not(&once);

        assert_eq!(once, expected);
        assert_eq!(bdd, twice);
    }

    #[test]
    fn not_overflow() {
        let mut table = NodeTable16::default();
        let v = VarIdPacked16::new(1000);

        let nodes = u16::MAX - 30000;

        // Only having one variable should not interfere with the not operation
        for i in 2..nodes {
            table
                .ensure_node(v, NodeId16::zero(), NodeId16::new(i - 1))
                .unwrap();
        }

        let root = table
            .ensure_node(v, NodeId16::zero(), NodeId16::new(nodes - 1))
            .unwrap();

        let (not_root, table) = not_16_bit(table, root);

        let table = match table {
            NodeTable::Size32(table) => table,
            _ => panic!("Expected 32-bit table"),
        };

        assert_eq!(table.node_count(), nodes as usize * 2);
        assert_eq!(
            NodeId32::unchecked_from(not_root),
            NodeId32::new(nodes as u32 * 2 - 1)
        );

        for i in 3..nodes {
            let node = table
                .get_node(NodeId32::new(i as u32 + nodes as u32))
                .unwrap();

            assert_eq!(node.variable(), v.into());
            assert_eq!(node.high, NodeId32::new(i as u32 + nodes as u32 - 1));
            assert_eq!(node.low, NodeId32::one());
        }

        let node = table.get_node(NodeId32::new(nodes as u32 + 1)).unwrap();
        assert_eq!(node.variable(), v.into());
        assert_eq!(node.high, NodeId32::zero());
        assert_eq!(node.low, NodeId32::one());
    }

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

        // Quantifying over empty set of variables
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
        let left_side = m.binary_op_with_exists(&f_x, &g, TriBool::and, &[v_x]);

        // Calculate right side: (∃x.f(x)) ∧ g
        let right_exists = m.exists(&f_x, &[v_x]);
        let right_side = m.and(&right_exists, &g);

        assert_eq!(left_side, right_side);

        // Also test for universal quantification: ∀x.(f(x) ∨ g) = (∀x.f(x)) ∨ g
        let left_side_forall = m.binary_op_with_for_all(&f_x, &g, TriBool::or, &[v_x]);

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

        // Test that the order of quantification doesn't matter for same quantifier
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

        // Test nested_apply vs regular apply for AND operation
        let result1 = manager.binary_op_with_exists(&e1, &e2, TriBool::and, &[]);
        let result2 = manager.and(&e1, &e2);
        assert_eq!(result1, result2);

        // Test nested_apply vs regular apply for IFF operation
        let result3 = manager.binary_op_with_for_all(&e1, &e3, TriBool::iff, &[]);
        let result4 = manager.iff(&e1, &e3);
        assert_eq!(result3, result4);
    }

    #[test]
    fn nested_apply_bit_width_variants() {
        let mut manager = BddManager::no_gc();

        let v1 = VariableId::from(u16::MAX >> 4);
        let v2 = VariableId::from(u32::MAX >> 4);
        let v3 = VariableId::new_long(u64::MAX >> 4).unwrap();

        // We will gradually force the manager to increase its bit-width by adding extra
        // variables of a specific width.

        for v in [v1, v2, v3] {
            let v_true = manager.new_bdd_literal(v, true);
            let v_false = manager.new_bdd_literal(v, false);

            // exists v1. (v1 & v1) is tautology
            assert!(manager
                .binary_op_with_exists(&v_true, &v_true, TriBool::and, &[v])
                .is_true());
            // exists v1. (v1 & !v1) is contradiction
            assert!(manager
                .binary_op_with_exists(&v_true, &v_false, TriBool::and, &[v])
                .is_false());
            // forall v1. (v1 | !v1) is tautology
            assert!(manager
                .binary_op_with_for_all(&v_true, &v_false, TriBool::or, &[v])
                .is_true());
            // forall v1. (v1 | v1) is contradiction
            assert!(manager
                .binary_op_with_for_all(&v_true, &v_true, TriBool::or, &[v])
                .is_false());
        }
    }
}
