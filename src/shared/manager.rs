use crate::{
    bdd_node::BddNodeAny,
    conversion::{UncheckedFrom, UncheckedInto},
    node_id::{NodeId, NodeId16, NodeId32, NodeId64, NodeIdAny},
    node_table::{
        GarbageCollector, NodeTable, NodeTable16, NodeTable32, NodeTable64, NodeTableAny,
    },
    task_cache::{TaskCache16, TaskCache32, TaskCache64, TaskCacheAny},
    variable_id::{VarIdPackedAny, VariableId},
};
use std::{
    cell::Cell,
    io::{self, Write},
    rc::{Rc, Weak},
};

use replace_with::replace_with_or_default;
use rustc_hash::FxHashMap;

/// The garbage collection strategy of the [`BddManager`].
#[derive(Debug, Clone, Copy, Default)]
pub enum GarbageCollection {
    /// Garbage collection is only performed when invoked explicitly.
    Manual,
    /// Garbage collection is performed automatically.
    #[default]
    Automatic,
}

/// A type representing a shared binary decision diagram. See [`BddManager`] for more details.
///
/// A shared `Bdd` is effectively just an index to the root node in the `BddManager`'s
/// unique node table. However, it is also reference counted to ensure that, as
/// long as the `Bdd` is alive, its nodes will not be garbage collected. As such,
/// cloning a `Bdd` increments this reference counter.
#[derive(Debug, Clone)]
pub struct Bdd {
    pub(crate) root: Rc<Cell<NodeId>>,
}

impl Bdd {
    pub(crate) fn new(root: NodeId) -> Self {
        Self {
            root: Rc::new(Cell::new(root)),
        }
    }

    pub(crate) fn root_weak(&self) -> Weak<Cell<NodeId>> {
        Rc::downgrade(&self.root)
    }

    /// Returns `true` if the `Bdd` represents the constant boolean function `true`.
    pub fn is_true(&self) -> bool {
        self.root.get().is_one()
    }

    /// Returns `true` if the `Bdd` represents the constant boolean function `false`.
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

/// The main structure for managing *shared* binary decision diagrams.
///
/// In the shared representation, the nodes of shared [`Bdd`]s are all stored in the
/// `BddManager`'s internal unique table. The table ensures that no duplicate
/// nodes are created. This means that if two or more BDDs share a subgraph,
/// that subgraph is stored only once in memory. Consequently, a `Bdd` object
/// is just an index to the root node in the unique table.
///
/// As BDDs are created and dropped, some nodes in the `BddManager`'s unique
/// table may no longer be referenced by any BDD and are considered "dead".
/// To remove these nodes, the manager automatically performs garbage collection by
/// default (configurable via [`BddManager::set_gc`]). This process marks all nodes
/// reachable from currently "live" `Bdd` handles; unmarked nodes are then
/// invalidated, so that their memory can be reused, or completely removed.
///
/// Note that, unlike most other implementations of (shared) BDDs, the `BddManager`
/// currently does not hold a computed cache that is reused between operations.
/// Instead, a new one is created for each operation.
#[derive(Debug)]
pub struct BddManager {
    pub(crate) unique_table: NodeTable,
    pub(crate) roots: Vec<Weak<Cell<NodeId>>>,
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

impl BddManager {
    /// Creates a new [`BddManager`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the [`GarbageCollection`] strategy of the `BddManager`.
    pub fn set_gc(&mut self, garbage_collection: GarbageCollection) {
        self.gc = garbage_collection;
    }

    /// Returns the [`Bdd`] representing the boolean constant `false`.
    pub fn new_bdd_false(&self) -> Bdd {
        Bdd::new(NodeId::zero())
    }

    /// Returns the [`Bdd`] representing the boolean constant `true`.
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

    /// Returns the total number of *used* nodes stored in the `BddManager`.
    pub fn total_node_count(&self) -> usize {
        match &self.unique_table {
            NodeTable::Size16(table) => table.node_count(),
            NodeTable::Size32(table) => table.node_count(),
            NodeTable::Size64(table) => table.node_count(),
        }
    }

    /// Returns the total number of node slots available in the `BddManager`, including
    /// free and full slots.
    pub fn total_capacity(&self) -> usize {
        match &self.unique_table {
            NodeTable::Size16(table) => table.size(),
            NodeTable::Size32(table) => table.size(),
            NodeTable::Size64(table) => table.size(),
        }
    }

    /// Returns the number of nodes in the `bdd`.
    pub fn node_count(&self, bdd: &Bdd) -> usize {
        let root = bdd.root.get();
        // TODO: Maybe this should not use unchecked into?
        match &self.unique_table {
            NodeTable::Size16(table) => table.reachable_node_count(root.unchecked_into()),
            NodeTable::Size32(table) => table.reachable_node_count(root.unchecked_into()),
            NodeTable::Size64(table) => table.reachable_node_count(root.unchecked_into()),
        }
    }

    /// Imports a [`crate::split::bdd::Bdd`], recreating its nodes in the `BddManager`.
    pub fn import_split(&mut self, bdd: &crate::split::bdd::Bdd) -> Bdd {
        let mut equivalent: FxHashMap<NodeId, Bdd> = FxHashMap::default();
        equivalent.insert(NodeId::zero(), self.new_bdd_false());
        equivalent.insert(NodeId::one(), self.new_bdd_true());
        let root = bdd.root();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            let variable = bdd.get_variable(node);
            let (low, high) = bdd.get_links(node);
            let low_replica = equivalent.get(&low);
            let high_replica = equivalent.get(&high);
            match (low_replica, high_replica) {
                (Some(low_replica), Some(high_replica)) => {
                    let node_replica = self.if_then_else_internal(
                        variable,
                        high_replica,
                        low_replica,
                        node == root,
                    );
                    equivalent.insert(node, node_replica);
                }
                _ => {
                    stack.push(node);
                    if low_replica.is_none() {
                        stack.push(low);
                    }
                    if high_replica.is_none() {
                        stack.push(high);
                    }
                }
            }
        }
        equivalent.get(&root).unwrap().clone()
    }

    /// Calculates a [`Bdd`] representing the boolean function `if 'condition' then 'then' else 'else_'`.
    pub fn if_then_else(&mut self, condition: VariableId, then: &Bdd, else_: &Bdd) -> Bdd {
        self.if_then_else_internal(condition, then, else_, true)
    }

    fn if_then_else_internal(
        &mut self,
        condition: VariableId,
        then: &Bdd,
        else_: &Bdd,
        is_root: bool,
    ) -> Bdd {
        if is_root {
            self.maybe_collect_garbage();
        }

        match &self.unique_table {
            NodeTable::Size16(_) if condition.fits_only_in_packed64() => {
                self.grow_to_64();
            }
            NodeTable::Size32(_) if condition.fits_only_in_packed64() => {
                self.grow();
            }
            NodeTable::Size16(_) if condition.fits_only_in_packed32() => {
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
                    .ensure_node(
                        condition.unchecked_into(),
                        else_.root.get().unchecked_into(),
                        then.root.get().unchecked_into(),
                    )
                    .expect("ensuring literal after growth should always succeed");
                root.unchecked_into()
            }
            NodeTable::Size32(table) => {
                let root = table
                    .ensure_node(
                        condition.unchecked_into(),
                        else_.root.get().unchecked_into(),
                        then.root.get().unchecked_into(),
                    )
                    .expect("ensuring literal after growth should always succeed");
                root.unchecked_into()
            }
            NodeTable::Size64(table) => {
                let root = table
                    .ensure_node(
                        condition.unchecked_into(),
                        else_.root.get().unchecked_into(),
                        then.root.get().unchecked_into(),
                    )
                    .expect("TODO: 64-bit ensure_literal failed");
                root.unchecked_into()
            }
        };

        let bdd = Bdd::new(root);
        if is_root {
            // If the created BDD node is a root, we have to save it into the internal pool.
            // This is not always required though, especially when we are creating nodes
            // for internal purposes.
            self.roots.push(bdd.root_weak());
        }

        bdd
    }

    /// Returns the [`Bdd`] representing the boolean function `variable=value`.
    pub fn new_bdd_literal(&mut self, variable: VariableId, value: bool) -> Bdd {
        if value {
            self.if_then_else(variable, &self.new_bdd_true(), &self.new_bdd_false())
        } else {
            self.if_then_else(variable, &self.new_bdd_false(), &self.new_bdd_true())
        }
    }

    pub(crate) fn maybe_collect_garbage(&mut self) {
        if !matches!(self.gc, GarbageCollection::Automatic) {
            return;
        }

        const GROWTH_RATIO: usize = 4;

        let nodes_added_since_last_gc = self.unique_table.node_count() - self.nodes_after_last_gc;

        if nodes_added_since_last_gc > self.nodes_after_last_gc.saturating_mul(GROWTH_RATIO) {
            self.collect_garbage();
        }
    }

    /// Removes all nodes from the `BddManager`'s pool that are not reachable from any
    /// [`Bdd`] created by this `BddManager`.
    pub fn collect_garbage(&mut self) {
        replace_with_or_default(&mut self.unique_table, |table| match table {
            NodeTable::Size16(table) => table.collect_garbage(&mut self.roots),
            NodeTable::Size32(table) => table.collect_garbage(&mut self.roots),
            NodeTable::Size64(table) => table.collect_garbage(&mut self.roots),
        });
        self.nodes_after_last_gc = self.unique_table.node_count();
    }

    /// Approximately counts the number of satisfying paths in the `bdd`.
    pub fn count_satisfying_paths(&self, bdd: &Bdd) -> f64 {
        match &self.unique_table {
            NodeTable::Size16(table) => {
                table.count_satisfying_paths(bdd.root.get().unchecked_into())
            }
            NodeTable::Size32(table) => {
                table.count_satisfying_paths(bdd.root.get().unchecked_into())
            }
            NodeTable::Size64(table) => {
                table.count_satisfying_paths(bdd.root.get().unchecked_into())
            }
        }
    }

    /// Approximately counts the number of satisfying valuations in the `bdd`.
    /// If `largest_variable` is [`Option::Some`], then it is
    /// assumed to be the largest variable. Otherwise, the largest variable residing
    /// in the manager is used.
    ///
    /// # Panics
    ///
    /// Assumes that the given variable is greater than or equal to any
    /// variable in the `bdd`. Otherwise, the function may give unexpected results
    /// in release mode or panic in debug mode.
    pub fn count_satisfying_valuations(
        &self,
        bdd: &Bdd,
        largest_variable: Option<VariableId>,
    ) -> f64 {
        match &self.unique_table {
            NodeTable::Size16(table) => {
                table.count_satisfying_valuations(bdd.root.get().unchecked_into(), largest_variable)
            }
            NodeTable::Size32(table) => {
                table.count_satisfying_valuations(bdd.root.get().unchecked_into(), largest_variable)
            }
            NodeTable::Size64(table) => {
                table.count_satisfying_valuations(bdd.root.get().unchecked_into(), largest_variable)
            }
        }
    }

    /// Calculates a [`Bdd`] representing the boolean formula `!bdd` (negation).
    pub fn not(&mut self, bdd: &Bdd) -> Bdd {
        self.maybe_collect_garbage();

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

        bdd
    }

    /// Writes `bdd` as a DOT graph to the given `output` stream.
    pub fn write_bdd_as_dot(&self, bdd: &Bdd, output: &mut dyn Write) -> io::Result<()> {
        match &self.unique_table {
            NodeTable::Size16(table) => {
                table.write_bdd_as_dot(bdd.root.get().unchecked_into(), output)
            }
            NodeTable::Size32(table) => {
                table.write_bdd_as_dot(bdd.root.get().unchecked_into(), output)
            }
            NodeTable::Size64(table) => {
                table.write_bdd_as_dot(bdd.root.get().unchecked_into(), output)
            }
        }
    }

    /// Converts `bdd` to a DOT graph string.
    pub fn bdd_to_dot_string(&self, bdd: &Bdd) -> String {
        let mut buffer = Vec::new();
        self.write_bdd_as_dot(bdd, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }
}

#[derive(Debug)]
struct NotState<TNodeTable: NodeTableAny, TTaskCache> {
    stack: Vec<(TNodeTable::Id, TNodeTable::VarId)>,
    results: Vec<TNodeTable::Id>,
    task_cache: TTaskCache,
    node_table: TNodeTable,
}

macro_rules! impl_not_state_variant {
    ($variant_name:ident, $constructor:ident, $task_cache:ident, $node_table:ident) => {
        type $variant_name<TCache = $task_cache> = NotState<$node_table, TCache>;

        fn $constructor(
            root: <$node_table as NodeTableAny>::Id,
            node_table: $node_table,
        ) -> $variant_name {
            let undefined_var = <$node_table as NodeTableAny>::VarId::undefined();

            NotState {
                stack: vec![(root, undefined_var)],
                results: Vec::new(),
                task_cache: $task_cache::default(),
                node_table,
            }
        }
    };
}

impl_not_state_variant!(NotState16, default_state_16, TaskCache16, NodeTable16);
impl_not_state_variant!(NotState32, default_state_32, TaskCache32, NodeTable32);
impl_not_state_variant!(NotState64, default_state_64, TaskCache64, NodeTable64);

macro_rules! impl_not_state_conversion {
    ($from_variant:ident, $to_variant:ident) => {
        impl<TCacheIn, TCacheOut: From<TCacheIn>> From<$from_variant<TCacheIn>>
            for $to_variant<TCacheOut>
        {
            fn from(state: $from_variant<TCacheIn>) -> Self {
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

impl_not_state_conversion!(NotState16, NotState32);
impl_not_state_conversion!(NotState32, NotState64);

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

fn not_16_bit(node_table: NodeTable16, root: NodeId16) -> (NodeId, NodeTable) {
    let state: NotState16<TaskCache16> = default_state_16(root, node_table);

    let state: NotState32<TaskCache16<NodeId32>> = match not_any(state) {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size16(table)),
        Err(state) => state.into(),
    };

    let state: NotState64<TaskCache16<NodeId64>> = match not_any(state) {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size32(table)),
        Err(state) => state.into(),
    };

    let (root, table) = not_any(state).expect("64-bit operation failed");
    (NodeId::unchecked_from(root), NodeTable::Size64(table))
}

fn not_32_bit(node_table: NodeTable32, root: NodeId32) -> (NodeId, NodeTable) {
    let state: NotState32<TaskCache32> = default_state_32(root, node_table);

    let state: NotState64<TaskCache32<NodeId64>> = match not_any(state) {
        Ok((root, table)) => return (NodeId::unchecked_from(root), NodeTable::Size32(table)),
        Err(state) => state.into(),
    };

    let (root, table) = not_any(state).expect("64-bit operation failed");
    (NodeId::unchecked_from(root), NodeTable::Size64(table))
}

fn not_64_bit(node_table: NodeTable64, root: NodeId64) -> (NodeId, NodeTable) {
    let state: NotState64<TaskCache64> = default_state_64(root, node_table);
    let (root, table) = not_any(state).expect("64-bit operation failed");
    (NodeId::unchecked_from(root), NodeTable::Size64(table))
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{
        node_id::{NodeId32, NodeId64, NodeIdAny},
        node_table::NodeTable,
        variable_id::{VarIdPacked16, VarIdPacked32, VarIdPacked64, VariableId},
    };

    impl BddManager {
        pub(crate) fn no_gc() -> Self {
            let mut manager = BddManager::new();
            manager.set_gc(GarbageCollection::Manual);
            manager
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

    pub fn ripple_carry_adder(manager: &mut BddManager, num_vars: u32) -> Bdd {
        let mut result = manager.new_bdd_false();
        for x in 0..(num_vars / 2) {
            let x1 = manager.new_bdd_literal(VariableId::new(x), true);
            let x2 = manager.new_bdd_literal(VariableId::new(x + num_vars / 2), true);
            let and = manager.and(&x1, &x2);
            result = manager.or(&result, &and);
        }
        result
    }

    #[test]
    fn standalone_import() {
        let bdd_a = crate::split::apply::tests::ripple_carry_adder(24).unwrap();
        let bdd_a = bdd_a.into();

        let mut manager = BddManager::no_gc();
        let imported = manager.import_split(&bdd_a);
        // First, check that the created BDD will survive GC (i.e., the root is correctly set).
        manager.collect_garbage();
        let op_test = manager.and(&imported, &imported);
        assert_eq!(op_test, imported);

        let expected = ripple_carry_adder(&mut manager, 24);
        assert_eq!(expected, imported);
    }

    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn queens(n: usize) -> (BddManager, Bdd) {
        let mut m = BddManager::no_gc();
        fn mk_negative_literals(n: usize, m: &mut BddManager) -> Vec<Bdd> {
            let mut bdd_literals = Vec::with_capacity(n * n);
            for i in 0..n {
                for j in 0..n {
                    let literal = m.new_bdd_literal(((i * n + j) as u32).into(), false);
                    bdd_literals.push(literal);
                }
            }
            bdd_literals
        }

        fn one_queen(n: usize, i: usize, j: usize, m: &mut BddManager, negative: &[Bdd]) -> Bdd {
            let mut s = m.new_bdd_literal(((i * n + j) as u32).into(), true);

            // no queens in the same row
            for k in 0..n {
                if k != j {
                    s = m.and(&s, &negative[i * n + k]);
                }
            }

            // no queens in the same column
            for k in 0..n {
                if k != i {
                    s = m.and(&s, &negative[k * n + j]);
                }
            }

            // no queens in the main diagonal (top-left to bot-right)
            // r - c = i - j  =>  c = (r + j) - i
            for row in 0..n {
                if let Some(col) = (row + j).checked_sub(i) {
                    if col < n && row != i {
                        s = m.and(&s, &negative[row * n + col]);
                    }
                }
            }

            // no queens in the anti diagonal (top-right to bot-left)
            // r + c = i + j  =>  c = (i + j) - r
            for row in 0..n {
                if let Some(col) = (i + j).checked_sub(row) {
                    if col < n && row != i {
                        s = m.and(&s, &negative[row * n + col]);
                    }
                }
            }

            s
        }

        fn queen_in_row(n: usize, row: usize, m: &mut BddManager, negative: &[Bdd]) -> Bdd {
            let mut r = m.new_bdd_false();
            for col in 0..n {
                let one_queen = one_queen(n, row, col, m, negative);
                r = m.or(&r, &one_queen);
            }
            r
        }

        let negative = mk_negative_literals(n, &mut m);
        let mut result = m.new_bdd_true();
        for row in 0..n {
            let in_row = queen_in_row(n, row, &mut m, &negative);
            result = m.and(&result, &in_row);
        }
        (m, result)
    }

    #[test]
    fn count_sat_valuations() {
        let mut m = BddManager::no_gc();
        let f = m.new_bdd_false();
        let t = m.new_bdd_true();
        assert_eq!(m.count_satisfying_valuations(&t, None), 1.0,);
        let v0 = m.new_bdd_literal(VariableId::new(0), true);
        let v0n = m.new_bdd_literal(VariableId::new(0), false);

        assert_eq!(
            m.count_satisfying_valuations(&f, Some(VariableId::new(0))),
            0.0,
        );

        assert_eq!(
            m.count_satisfying_valuations(&t, Some(VariableId::new(0))),
            2.0,
        );

        assert_eq!(m.count_satisfying_valuations(&t, None), 2.0,);

        assert_eq!(m.count_satisfying_valuations(&v0, None), 1.0,);

        assert_eq!(
            m.count_satisfying_valuations(&v0n, Some(VariableId::new(2))),
            4.0,
        );

        let v1 = m.new_bdd_literal(VariableId::new(1), true);
        let v3 = m.new_bdd_literal(VariableId::new(3), true);
        let or2 = m.or(&v0, &v1);
        let or3 = m.or(&or2, &v3);

        let and2 = m.and(&v0, &v1);
        let and3 = m.and(&and2, &v3);

        assert_eq!(m.count_satisfying_valuations(&or3, None), 14.0);
        assert_eq!(m.count_satisfying_valuations(&and3, None), 2.0);

        let (m, bdd4) = queens(4);
        assert_eq!(m.count_satisfying_valuations(&bdd4, None), 2.0);
        let (m, bdd8) = queens(8);
        assert_eq!(m.count_satisfying_valuations(&bdd8, None), 92.0);
    }

    #[test]
    fn count_sat_paths() {
        let mut m = BddManager::no_gc();
        let f = m.new_bdd_false();
        let t = m.new_bdd_true();
        let v0 = m.new_bdd_literal(VariableId::new(0), true);
        let v0n = m.new_bdd_literal(VariableId::new(0), false);

        assert_eq!(m.count_satisfying_paths(&f), 0.0,);

        assert_eq!(m.count_satisfying_paths(&t), 1.0,);

        assert_eq!(m.count_satisfying_paths(&v0), 1.0,);

        assert_eq!(m.count_satisfying_paths(&v0n), 1.0,);

        let v1 = m.new_bdd_literal(VariableId::new(1), true);
        let v3 = m.new_bdd_literal(VariableId::new(3), true);
        let or2 = m.or(&v0, &v1);
        let or3 = m.or(&or2, &v3);

        let and2 = m.and(&v0, &v1);
        let and3 = m.and(&and2, &v3);

        assert_eq!(m.count_satisfying_paths(&or3), 3.0);
        assert_eq!(m.count_satisfying_paths(&and3), 1.0);

        let (m, bdd4) = queens(6);
        assert_eq!(m.count_satisfying_paths(&bdd4), 4.0);

        let (m, bdd8) = queens(9);
        assert_eq!(m.count_satisfying_paths(&bdd8), 352.0);
    }

    #[test]
    fn bdd_to_dot() {
        let mut manager = BddManager::no_gc();

        let v_1 = VariableId::new(1);
        let v_2 = VariableId::new(2);
        let v_3 = VariableId::new(3);
        let v_4 = VariableId::new(4);

        let x = manager.new_bdd_literal(v_1, true);
        let y = manager.new_bdd_literal(v_2, true);
        let z = manager.new_bdd_literal(v_3, false);

        let xy = manager.and(&x, &y);

        let xyz = manager.and(&xy, &z);

        let _n1 = manager.new_bdd_literal(v_3, true);
        let _n2 = manager.new_bdd_literal(v_4, true);

        let result = manager.bdd_to_dot_string(&xyz);

        let expected = r#"digraph BDD {
  __ruddy_root [label="", style=invis, height=0, width=0];
  __ruddy_root -> 7;

  edge [dir=none];

  0 [label="0", shape=box, width=0.3, height=0.3];
  1 [label="1", shape=box, width=0.3, height=0.3];

  7 [label="1", shape=circle];
  7 -> 0 [style=dashed];
  7 -> 6;
  6 [label="2", shape=circle];
  6 -> 0 [style=dashed];
  6 -> 4;
  4 [label="3", shape=circle];
  4 -> 1 [style=dashed];
  4 -> 0;
}
"#;

        assert_eq!(result, expected);
    }

    #[test]
    fn constant_bdd_to_dot() {
        let mut manager = BddManager::no_gc();

        let f = manager.new_bdd_false();
        let v_1 = VariableId::new(1);
        let v_2 = VariableId::new(2);

        let _x = manager.new_bdd_literal(v_1, true);
        let _y = manager.new_bdd_literal(v_2, true);

        let result = manager.bdd_to_dot_string(&f);

        let expected = r#"digraph BDD {
  __ruddy_root [label="", style=invis, height=0, width=0];
  __ruddy_root -> 0;

  edge [dir=none];

  0 [label="0", shape=box, width=0.3, height=0.3];
  1 [label="1", shape=box, width=0.3, height=0.3];
}
"#;

        assert_eq!(result, expected);

        let t = manager.new_bdd_true();
        let result = manager.bdd_to_dot_string(&t);

        let expected = r#"digraph BDD {
  __ruddy_root [label="", style=invis, height=0, width=0];
  __ruddy_root -> 1;

  edge [dir=none];

  0 [label="0", shape=box, width=0.3, height=0.3];
  1 [label="1", shape=box, width=0.3, height=0.3];
}
"#;

        assert_eq!(result, expected);
    }
}
