use crate::conversion::UncheckedInto;
use crate::node_id::{AsNodeId, NodeId};
use crate::variable_id::{variables_between, Mark, VariableId};
use crate::{
    bdd_node::{BddNode16, BddNode32, BddNode64, BddNodeAny},
    node_id::{NodeId16, NodeId32, NodeId64, NodeIdAny},
    split::bdd::BddAny,
    variable_id::{VarIdPacked16, VarIdPacked32, VarIdPacked64, VarIdPackedAny},
};
use crate::{usize_is_at_least_32_bits, usize_is_at_least_64_bits};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use std::cell::Cell;
use std::cmp::{max, min};
use std::collections::HashSet;
use std::io::Write;
use std::rc::Weak;
use std::{fmt, io};

/// The `NodeTableAny` is a data structure that enforces uniqueness of BDD nodes created
/// during the BDD construction process.
pub(crate) trait NodeTableAny: Default {
    type Id: NodeIdAny;
    type VarId: VarIdPackedAny;
    type Node: BddNodeAny<Id = Self::Id, VarId = Self::VarId>;

    /// Make a new, empty `NodeTableAny` with at least the specified amount of `capacity`.
    fn with_capacity(capacity: usize) -> Self;

    /// Returns the number of nodes in the node table, including the terminal nodes.
    fn node_count(&self) -> usize;

    /// Get a (checked) reference to a node, or `None` if such node does not exist.
    fn get_node(&self, id: Self::Id) -> Option<&Self::Node>;

    /// An unchecked variant of [`NodeTableAny::get_node`].
    ///
    /// # Safety
    ///
    /// Calling this method with an `id` that is not in the table is undefined behavior.
    unsafe fn get_node_unchecked(&self, id: Self::Id) -> &Self::Node;

    /// Searches the `NodeTableAny` for a node matching node `(var, (low, high))`, and returns its
    /// identifier (i.e. the node's variable is `var`, and the node's low and high children
    /// are `low` and `high`, respectively). If such a node is not found, a new node is created
    /// and added to the node table.
    ///
    /// This method should not be used to "create" terminal nodes, i.e. it must hold that
    /// `variable != VarId::undefined`. Furthermore, the method can fail if the node table is
    /// "full", meaning no new nodes can be created. Typically, the table is responsible
    /// for resizing itself, but some implementations can be limited in the number of representable
    /// nodes through other means (e.g. the bit width of the underlying ID types).
    fn ensure_node(
        &mut self,
        variable: Self::VarId,
        low: Self::Id,
        high: Self::Id,
    ) -> Result<Self::Id, NodeTableFullError>;

    /// Create a new [`BddAny`] from `self` rooted in `root`. The conversion preserves
    /// *all* nodes that are present in `self`, not just the ones reachable from the
    /// root node.
    ///
    /// ## Safety
    ///
    /// Similar to [`BddAny::new_unchecked`], this function is unsafe, because it can be used to
    /// create an invariant-breaking BDD. While [`NodeTableAny`] cannot be used (under normal
    /// conditions) to create BDDs with cycles, it can definitely be used to create BDDs with
    /// broken variable ordering.
    unsafe fn into_bdd<TBdd: BddAny<Id = Self::Id, VarId = Self::VarId, Node = Self::Node>>(
        self,
        root: Self::Id,
    ) -> TBdd;

    /// Create a new [`BddAny`] from `self` rooted in `root`. The conversion preserves
    /// only the nodes that are reachable from the root node.
    ///
    /// ## Safety
    ///
    /// Similar to [`BddAny::new_unchecked`], this function is unsafe, because it can be used to
    /// create an invariant-breaking BDD. While [`NodeTableAny`] cannot be used (under normal
    /// conditions) to create BDDs with cycles, it can definitely be used to create BDDs with
    /// broken variable ordering.
    unsafe fn into_reachable_bdd<
        TBdd: BddAny<Id = Self::Id, VarId = Self::VarId, Node = Self::Node>,
    >(
        self,
        root: Self::Id,
    ) -> TBdd;

    /// Deletes the entry with the given `id` from the node table.
    ///
    /// This method must not be used to delete terminal nodes.
    fn delete(&mut self, id: Self::Id);
}

/// An error that is returned when [`NodeTableAny`] is full and a new node cannot be added.
///
/// It carries the bit-width of the current node ID type for which the error was raised.
#[derive(PartialEq, Eq, Clone, Debug)]
pub(crate) struct NodeTableFullError {
    width: usize,
}

impl fmt::Display for NodeTableFullError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ensuring node on a full {}-bit node table", self.width)
    }
}

impl std::error::Error for NodeTableFullError {}

/// An element of the [`NodeTableImpl`]. Consists of a [`BddNodeAny`] node, and three node pointers,
/// referencing the `parent` tree that is rooted in this entry, plus two `next_parent` pointers
/// that define the parent tree which contains the entry itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NodeEntry<
    TNodeId: NodeIdAny,
    TVarId: VarIdPackedAny,
    TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
> {
    pub node: TNode,
    parent: TNodeId,
    next_parent_zero: TNodeId,
    next_parent_one: TNodeId,
}

impl<
        TNodeId: NodeIdAny,
        TVarId: VarIdPackedAny,
        TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
    > From<TNode> for NodeEntry<TNodeId, TVarId, TNode>
{
    fn from(node: TNode) -> Self {
        NodeEntry {
            node,
            parent: TNodeId::undefined(),
            next_parent_zero: TNodeId::undefined(),
            next_parent_one: TNodeId::undefined(),
        }
    }
}

impl<
        TNodeId: NodeIdAny,
        TVarId: VarIdPackedAny,
        TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
    > NodeEntry<TNodeId, TVarId, TNode>
{
    fn zero() -> Self {
        Self::from(TNode::zero())
    }

    fn one() -> Self {
        Self::from(TNode::one())
    }

    fn is_deleted(&self) -> bool {
        self.node.low().is_undefined()
    }

    /// Marks the entry as deleted. We use the `low` field to mark the entry as
    /// deleted by setting it to `undefined`, and the `high` field to store the
    /// next free entry in the free list.
    ///
    /// Nodes with `undefined` children should never be created, so using the
    /// `low` field to store the deletion flag is safe.
    fn mark_as_deleted(&mut self, next_free: TNodeId) {
        *self.node.low_mut() = TNodeId::undefined();
        *self.node.high_mut() = next_free;
    }
}

type NodeEntry16 = NodeEntry<NodeId16, VarIdPacked16, BddNode16>;
type NodeEntry32 = NodeEntry<NodeId32, VarIdPacked32, BddNode32>;
type NodeEntry64 = NodeEntry<NodeId64, VarIdPacked64, BddNode64>;

macro_rules! impl_from_node_entry {
    ($from:ident, $to:ident) => {
        impl From<$from> for $to {
            fn from(entry: $from) -> Self {
                Self {
                    node: entry.node.into(),
                    parent: entry.parent.into(),
                    next_parent_zero: entry.next_parent_zero.into(),
                    next_parent_one: entry.next_parent_one.into(),
                }
            }
        }
    };
}

impl_from_node_entry!(NodeEntry16, NodeEntry32);
impl_from_node_entry!(NodeEntry16, NodeEntry64);
impl_from_node_entry!(NodeEntry32, NodeEntry64);

/// A mutable view into a deleted entry in a [`NodeTableImpl`].
struct DeletedEntryMut<
    'a,
    TNodeId: NodeIdAny,
    TVarId: VarIdPackedAny,
    TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
>(&'a mut NodeEntry<TNodeId, TVarId, TNode>);

impl<
        TNodeId: NodeIdAny,
        TVarId: VarIdPackedAny,
        TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
    > DeletedEntryMut<'_, TNodeId, TVarId, TNode>
{
    fn next_free(&self) -> TNodeId {
        self.0.node.high()
    }

    fn insert(self, entry: NodeEntry<TNodeId, TVarId, TNode>) {
        *self.0 = entry;
    }
}

/// A reasonably random hash of the node data (its minimum child and variable).
fn hash_node_data<TVarId: VarIdPackedAny, TNodeId: NodeIdAny>(
    variable: TVarId,
    low: TNodeId,
    high: TNodeId,
) -> u64 {
    // Inspired by https://stackoverflow.com/a/27952689
    // TODO: is this a good hash function?
    let min_child = min(low, high);
    let hash_child =
        UncheckedInto::<u64>::unchecked_into(min_child).wrapping_mul(14695981039346656039);
    let hash_variable = variable.unpack_u64().wrapping_mul(14695981039346656039);
    hash_child.wrapping_mul(3).wrapping_add(hash_variable)
}

/// Returns `true` if the top bit is one.
fn top_bit_is_one(hash: u64) -> bool {
    hash & (1 << (u64::BITS - 1)) != 0
}

/// A generic implementation of [`NodeTableAny`] backed by [`BddNodeAny`].
///
/// Instead of "normal" hashing, it uses a "tree of parents" scheme, where each node is stored
/// in its maximal child. This is effective because most nodes in a BDD only have one or two
/// parents, so we know the tree will stay shallow and will be searched in `O(1)` time. Only a
/// few nodes have many parents, resulting in average `O(log)` search time (this part does
/// in fact depend on hash collisions).
///
/// (Note that we could obtain a tight `O(log)` bound by using a search tree instead of
/// a hash-prefix tree, but this would require balancing, which adds a lot of overhead)
///
/// The `NodeTableImpl` also supports deletion of nodes. When a node is deleted,
/// it is not actually removed from the table, but only marked as deleted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NodeTableImpl<
    TNodeId: NodeIdAny,
    TVarId: VarIdPackedAny,
    TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
> {
    entries: Vec<NodeEntry<TNodeId, TVarId, TNode>>,
    first_free: TNodeId,
    deleted: usize,
    current_mark: Mark,
}

impl<TNodeId, TVarId, TNode> NodeTableImpl<TNodeId, TVarId, TNode>
where
    TNodeId: NodeIdAny,
    TVarId: VarIdPackedAny,
    TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
{
    /// Make a new [`NodeTableImpl`] containing nodes `0` and `1`.
    pub fn new() -> Self {
        Self {
            entries: vec![NodeEntry::zero(), NodeEntry::one()],
            first_free: TNodeId::undefined(),
            deleted: 0,
            current_mark: Mark::default(),
        }
    }

    /// Returns the total number of entries in the node table,  
    /// including **deleted** entries and entries for the terminal nodes.
    pub fn size(&self) -> usize {
        self.entries.len()
    }

    /// Create a new [`NodeEntry`] in this table (without checking for uniqueness), and increment
    /// the parent counters of its child nodes.
    ///
    /// # Safety
    ///
    /// The function requires that `low` and `high` IDs point to existing entries in this node
    /// table. This requirement is not checked in release mode and if broken results
    /// in undefined behavior.
    unsafe fn push_node(&mut self, variable: TVarId, low: TNodeId, high: TNodeId) {
        debug_assert!(!variable.is_undefined());
        self.get_node_unchecked_mut(low).increment_parent_counter();

        self.get_node_unchecked_mut(high).increment_parent_counter();

        // Reset the parent counter of the new node.
        let mut variable = variable.reset_parents();
        // We want the new node to have the same mark as the rest of the nodes.
        variable.set_mark(self.current_mark);

        let new_entry = TNode::new(variable, low, high).into();

        if self.first_free.is_undefined() {
            self.entries.push(new_entry);
        } else {
            let free_entry = self.get_deleted_entry_unchecked_mut(self.first_free);
            let new_first_free = free_entry.next_free();
            free_entry.insert(new_entry);
            self.first_free = new_first_free;
            self.deleted -= 1;
        }
    }

    /// Get a checked reference to the entry with the given `id`, or `None` if the entry does not exist.
    pub fn get_entry(&self, id: TNodeId) -> Option<&NodeEntry<TNodeId, TVarId, TNode>> {
        self.entries
            .get(id.as_usize())
            .filter(|entry| !entry.is_deleted())
    }

    /// Get a checked mutable reference to the entry with the given `id`, or `None` if the entry does not exist.
    pub fn get_entry_mut(&mut self, id: TNodeId) -> Option<&mut NodeEntry<TNodeId, TVarId, TNode>> {
        self.entries
            .get_mut(id.as_usize())
            .filter(|entry| !entry.is_deleted())
    }

    /// An unchecked variant of [`NodeTableImpl::get_entry`].
    ///
    /// # Safety
    ///
    /// Calling this method with an `id` that is not in the table is undefined behavior.
    unsafe fn get_entry_unchecked(&self, id: TNodeId) -> &NodeEntry<TNodeId, TVarId, TNode> {
        debug_assert!(id.as_usize() < self.size());
        let entry = self.entries.get_unchecked(id.as_usize());
        debug_assert!(!entry.is_deleted());
        entry
    }

    /// An unchecked variant of [`NodeTableImpl::get_entry_mut`].
    ///
    /// # Safety
    ///
    /// Calling this method with an `id` that is not in the table is undefined behavior.
    unsafe fn get_entry_unchecked_mut(
        &mut self,
        id: TNodeId,
    ) -> &mut NodeEntry<TNodeId, TVarId, TNode> {
        debug_assert!(id.as_usize() < self.size());
        let entry = self.entries.get_unchecked_mut(id.as_usize());
        debug_assert!(!entry.is_deleted());
        entry
    }

    /// Get an unchecked mutable reference to the deleted entry with the given `id`.
    ///
    /// # Safety
    ///
    /// Calling this method with an `id` that is not in the table or is not deleted is undefined behavior.
    unsafe fn get_deleted_entry_unchecked_mut(
        &mut self,
        id: TNodeId,
    ) -> DeletedEntryMut<TNodeId, TVarId, TNode> {
        debug_assert!(id.as_usize() < self.size());
        let entry = self.entries.get_unchecked_mut(id.as_usize());
        debug_assert!(entry.is_deleted());
        DeletedEntryMut(entry)
    }

    /// Get an unchecked mutable reference to the node with the given `id`.
    ///
    /// # Safety
    ///
    /// Calling this method with an `id` that is not in the table is undefined behavior.
    pub(crate) unsafe fn get_node_unchecked_mut(&mut self, id: TNodeId) -> &mut TNode {
        &mut self.get_entry_unchecked_mut(id).node
    }

    /// Get an iterator over nodes in the table.
    pub(crate) fn iter_nodes(&self) -> impl Iterator<Item = &TNode> {
        self.entries
            .iter()
            .filter(|entry| !entry.is_deleted())
            .map(|entry| &entry.node)
    }

    /// Find the parent of `node` in the tree rooted in `start` by following the `hash`.
    ///
    /// Returns the parent node and a boolean indicating whether the node is the
    /// one child of the parent. If `start` and `node` are the same (i.e., no
    /// parent can be found), then the parent will be `undefined`.
    ///
    /// The `hash` is rotated left by one bit after each step, hence it can
    /// be used to continue the search from `node` after the function returns.
    ///
    /// # Warning
    ///
    /// If the node is not found in the tree rooted in `start`, the function
    /// will panic in debug mode and loop indefinitely in release mode.
    fn find_parent_of_node(
        &self,
        node: TNodeId,
        start: TNodeId,
        hash: &mut u64,
    ) -> (TNodeId, bool) {
        if start == node {
            return (TNodeId::undefined(), false);
        }

        let mut current_node = start;

        loop {
            let current_entry = unsafe { self.get_entry_unchecked(current_node) };

            let top_bit_is_one = top_bit_is_one(*hash);

            let next_node = if top_bit_is_one {
                current_entry.next_parent_one
            } else {
                current_entry.next_parent_zero
            };

            *hash = hash.rotate_left(1);

            debug_assert!(next_node != TNodeId::undefined());

            if next_node == node {
                return (current_node, top_bit_is_one);
            }

            current_node = next_node;
        }
    }

    /// Find a leaf in the tree rooted in `start` by following the `hash`.
    ///
    /// Returns the leaf node, its parent, and a boolean indicating whether the
    /// leaf is the one child of the parent. If `start` is a leaf, then the parent
    /// will be `undefined`.
    ///
    /// The `hash` is rotated left by one bit after each step.
    ///
    /// Note that it might happen that we encounter a node with only one child, while
    /// the hash would lead us to the non-existent (`undefined`) child. In this case, we
    /// follow this child instead of the hash.
    fn follow_hash_to_leaf(&self, start: TNodeId, hash: &mut u64) -> (TNodeId, TNodeId, bool) {
        let mut current_node = start;

        let mut parent = TNodeId::undefined();
        let mut leaf_is_one_child = false;

        loop {
            let current_entry = unsafe { self.get_entry_unchecked(current_node) };

            let next_parent_one = current_entry.next_parent_one;
            let next_parent_zero = current_entry.next_parent_zero;

            if next_parent_one.is_undefined() && next_parent_zero.is_undefined() {
                return (current_node, parent, leaf_is_one_child);
            }

            parent = current_node;

            const TOP_BIT_ONE: bool = true;
            const TOP_BIT_ZERO: bool = false;
            const ONE_CHILD_UNDEFINED: bool = true;
            const ONE_CHILD_EXISTS: bool = false;
            const ZERO_CHILD_UNDEFINED: bool = true;
            const ZERO_CHILD_EXISTS: bool = false;

            match (
                top_bit_is_one(*hash),
                next_parent_one.is_undefined(),
                next_parent_zero.is_undefined(),
            ) {
                // The hash would lead us to a non-existent child, but the node
                // has a child. We follow the child instead of the hash.
                (TOP_BIT_ONE, ONE_CHILD_UNDEFINED, _) => {
                    current_node = next_parent_zero;
                    leaf_is_one_child = false;
                }
                (TOP_BIT_ZERO, _, ZERO_CHILD_UNDEFINED) => {
                    current_node = next_parent_one;
                    leaf_is_one_child = true;
                }
                // In these cases, we can just follow the child.
                (TOP_BIT_ONE, ONE_CHILD_EXISTS, _) => {
                    current_node = next_parent_one;
                    *hash = hash.rotate_left(1);
                    leaf_is_one_child = true;
                }
                (TOP_BIT_ZERO, _, ZERO_CHILD_EXISTS) => {
                    current_node = next_parent_zero;
                    *hash = hash.rotate_left(1);
                    leaf_is_one_child = false;
                }
            }

            debug_assert!(!current_node.is_undefined());
        }
    }

    /// Replace the one child or the zero child of `node` with `replacement`.
    fn replace_nodes_child(&mut self, node: TNodeId, one_child: bool, replacement: TNodeId) {
        let node_entry = unsafe { self.get_entry_unchecked_mut(node) };
        if one_child {
            node_entry.next_parent_one = replacement;
        } else {
            node_entry.next_parent_zero = replacement;
        }
    }

    /// Replace the children of `node` with `next_parent_one` and `next_parent_zero`.
    fn replace_nodes_children(
        &mut self,
        node: TNodeId,
        next_parent_one: TNodeId,
        next_parent_zero: TNodeId,
    ) {
        let node_entry = unsafe { self.get_entry_unchecked_mut(node) };
        node_entry.next_parent_one = next_parent_one;
        node_entry.next_parent_zero = next_parent_zero;
    }

    /// Mark the node with the given `id` as deleted and add it to the free list.
    /// See [`NodeEntry::mark_as_deleted`] for details.
    fn mark_node_as_deleted(&mut self, id: TNodeId) {
        let next_free = self.first_free;
        unsafe { self.get_entry_unchecked_mut(id) }.mark_as_deleted(next_free);
        self.first_free = id;
        self.deleted += 1;
    }

    /// Searches for a node representing `variable=value`, i.e.,
    /// `(variable, (0, 1))` if `value` is `true` or `(variable, (1, 0))` otherwise,
    /// and returns its identifier. If such a node is not found,
    /// a new node is created and added to the node table.
    ///
    /// This method should not be used to "create" terminal nodes, i.e. it must hold that
    /// `variable != VarId::undefined`.
    // This method is currently not used, but could be useful in
    // the future, so let's not remove it.
    #[allow(dead_code)]
    pub(crate) fn ensure_literal(
        &mut self,
        variable: TVarId,
        value: bool,
    ) -> Result<TNodeId, NodeTableFullError> {
        if value {
            self.ensure_node(variable, TNodeId::zero(), TNodeId::one())
        } else {
            self.ensure_node(variable, TNodeId::one(), TNodeId::zero())
        }
    }

    /// Compute the number of nodes that are reachable from the given `root` node. This
    /// is not very useful for standalone BDDs, but it is used often to compute BDD size
    /// for shared BDDs.
    pub(crate) fn reachable_node_count(&self, root: TNodeId) -> usize {
        debug_assert!(!root.is_undefined());

        // Ensure that the root exists. Transitively, everything reachable from root
        // also exists and we can safely avoid bounds checks.
        assert!(self.get_entry(root).is_some());

        if root.is_zero() {
            return 1;
        }
        if root.is_one() {
            return 2;
        }

        let mut stack = vec![root];
        let mut visited: FxHashSet<TNodeId> =
            HashSet::with_capacity_and_hasher(1024, FxBuildHasher);

        while let Some(node) = stack.pop() {
            if node.is_terminal() || visited.contains(&node) {
                continue;
            }
            visited.insert(node);
            let entry = unsafe { self.get_entry_unchecked(node) };
            stack.push(entry.node.high());
            stack.push(entry.node.low());
        }

        visited.len() + 2 // plus two terminal nodes...
    }

    /// Get the largest [`VariableId`] in the table. Returns [`Option::None`] if
    /// the table contains only terminal nodes.
    pub(crate) fn get_largest_variable(&self) -> Option<VariableId> {
        let variable = self
            .iter_nodes()
            .map(|node| node.variable())
            .reduce(TVarId::max_defined)
            .expect("node table is not empty");
        if variable.is_undefined() {
            None
        } else {
            Some(variable.unchecked_into())
        }
    }

    /// Approximately counts the number of satisfying paths in the BDD rooted
    /// in `root`.
    pub(crate) fn count_satisfying_paths(&self, root: TNodeId) -> f64 {
        debug_assert!(!root.is_undefined());
        assert!(self.get_entry(root).is_some());

        let mut cache: FxHashMap<TNodeId, f64> =
            FxHashMap::with_capacity_and_hasher(1024, FxBuildHasher);

        cache.insert(TNodeId::zero(), 0.0);
        cache.insert(TNodeId::one(), 1.0);

        let mut stack = vec![root];

        while let Some(id) = stack.pop() {
            if cache.contains_key(&id) {
                continue;
            }

            let node = unsafe { self.get_node_unchecked(id) };
            let low = node.low();
            let high = node.high();

            let low_count = cache.get(&low);
            let high_count = cache.get(&high);

            match (low_count, high_count) {
                (Some(low_count), Some(high_count)) => {
                    cache.insert(id, low_count + high_count);
                }
                _ => {
                    stack.push(id);

                    if low_count.is_none() {
                        stack.push(low);
                    }

                    if high_count.is_none() {
                        stack.push(high);
                    }
                }
            };
        }

        *cache.get(&root).expect("count for root present in cache")
    }

    /// Approximately counts the number of satisfying valuations in the BDD
    /// rooted in `root`. If `largest_variable` is [`Option::Some`], then it is
    /// assumed to be the largest variable. Otherwise, the largest variable in the
    /// table is used.
    ///
    /// Assumes that the given variable is greater than or equal to than any
    /// variable in the BDD. Otherwise, the function may give unexpected results
    /// in release mode or panic in debug mode.
    pub(crate) fn count_satisfying_valuations(
        &self,
        root: TNodeId,
        largest_variable: Option<VariableId>,
    ) -> f64 {
        debug_assert!(!root.is_undefined());
        if root.is_zero() {
            return 0.0;
        }

        let largest_variable = largest_variable.or_else(|| self.get_largest_variable());

        if root.is_one() {
            if let Some(largest_variable) = largest_variable {
                let exponent = (Into::<u64>::into(largest_variable) + 1)
                    .try_into()
                    .unwrap_or(f64::MAX_EXP);
                return 2.0f64.powi(exponent);
            }
            return 1.0f64;
        }

        let largest_variable = largest_variable.expect("node table contains non-terminal node");

        let root_variable = match self.get_node(root) {
            Some(node) => node.variable(),
            None => unreachable!(),
        };

        let mut cache: FxHashMap<TNodeId, f64> =
            FxHashMap::with_capacity_and_hasher(1024, FxBuildHasher);

        cache.insert(TNodeId::zero(), 0.0);
        cache.insert(TNodeId::one(), 1.0);

        let mut stack = vec![root];

        while let Some(&id) = stack.last() {
            if cache.contains_key(&id) {
                stack.pop();
                continue;
            }

            let node = unsafe { self.get_node_unchecked(id) };
            let low = node.low();
            let high = node.high();
            let variable = node.variable();
            let low_variable = unsafe { self.get_node_unchecked(low) }.variable();
            let high_variable = unsafe { self.get_node_unchecked(high) }.variable();

            let low_count = cache.get(&low);
            let high_count = cache.get(&high);

            match (low_count, high_count) {
                (Some(low_count), Some(high_count)) => {
                    let skipped = variables_between(low_variable, variable, largest_variable)
                        .try_into()
                        .unwrap_or(f64::MAX_EXP);

                    let low_count = low_count * 2.0f64.powi(skipped);

                    let skipped = variables_between(high_variable, variable, largest_variable)
                        .try_into()
                        .unwrap_or(f64::MAX_EXP);

                    let high_count = high_count * 2.0f64.powi(skipped);

                    cache.insert(id, low_count + high_count);
                }
                _ => {
                    stack.push(id);

                    if low_count.is_none() {
                        stack.push(low);
                    }

                    if high_count.is_none() {
                        stack.push(high);
                    }
                }
            };
        }

        let count = cache.get(&root).expect("count for root present in cache");
        let result = count
            * 2.0f64.powi(
                root_variable
                    .unpack_u64()
                    .try_into()
                    .unwrap_or(f64::MAX_EXP),
            );
        if result.is_nan() {
            f64::INFINITY
        } else {
            result
        }
    }
}

impl<
        TNodeId: NodeIdAny,
        TVarId: VarIdPackedAny,
        TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
    > Default for NodeTableImpl<TNodeId, TVarId, TNode>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
        TNodeId: NodeIdAny,
        TVarId: VarIdPackedAny,
        TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
    > NodeTableAny for NodeTableImpl<TNodeId, TVarId, TNode>
{
    type Id = TNodeId;
    type VarId = TVarId;
    type Node = TNode;

    fn with_capacity(capacity: usize) -> Self {
        let mut entries = Vec::with_capacity(capacity);
        entries.push(NodeEntry::zero());
        entries.push(NodeEntry::one());
        Self {
            entries,
            first_free: TNodeId::undefined(),
            deleted: 0,
            current_mark: Mark::default(),
        }
    }

    fn node_count(&self) -> usize {
        self.entries.len() - self.deleted
    }

    fn get_node(&self, id: TNodeId) -> Option<&TNode> {
        self.get_entry(id).map(|entry| &entry.node)
    }

    unsafe fn get_node_unchecked(&self, id: TNodeId) -> &TNode {
        &self.get_entry_unchecked(id).node
    }

    fn ensure_node(
        &mut self,
        variable: Self::VarId,
        low: Self::Id,
        high: Self::Id,
    ) -> Result<Self::Id, NodeTableFullError> {
        debug_assert!(!variable.is_undefined());

        if low == high {
            // We don't need to create a new node in this case.
            return Ok(low);
        }

        // Save the ID of the new node for later use. Also check that the table
        // is not full.
        let new_node = if self.first_free.is_undefined() {
            debug_assert_eq!(self.size(), self.node_count());
            TNodeId::try_from(self.size()).map_err(|_| NodeTableFullError {
                width: std::mem::size_of::<TNodeId>() * 8,
            })?
        } else {
            self.first_free
        };

        debug_assert!((low.as_usize()) < self.size());
        debug_assert!((high.as_usize()) < self.size());

        // Max child is used to select the tree of nodes into which we are inserting.
        // The reason why `max_child` is used instead of `min_child` is that it avoids terminal
        // nodes which typically have many parents.
        let max_child = max(low, high);

        let mut hash = hash_node_data(variable, low, high);

        let root_node = unsafe { self.get_entry_unchecked_mut(max_child) };
        let mut current_node = root_node.parent;

        if current_node.is_undefined() {
            // This is the first parent for the root node.
            root_node.parent = new_node;
            unsafe {
                self.push_node(variable, low, high);
            }
            return Ok(new_node);
        }

        // There is a larger tree that we need to explore.
        loop {
            debug_assert!((current_node.as_usize()) < self.size());

            let current_entry = unsafe { self.get_entry_unchecked_mut(current_node) };

            if current_entry.node.variable() == variable
                && current_entry.node.low() == low
                && current_entry.node.high() == high
            {
                // The node exists and is stored inside `current_node`.
                return Ok(current_node);
            }

            if top_bit_is_one(hash) {
                if current_entry.next_parent_one.is_undefined() {
                    // Next "one" slot is empty. We can save the node there.
                    current_entry.next_parent_one = new_node;
                    unsafe {
                        self.push_node(variable, low, high);
                    }
                    return Ok(new_node);
                } else {
                    // Next slot is already occupied, which means we should test
                    // it in the next iteration.
                    current_node = current_entry.next_parent_one;
                }
            } else {
                // Top bit is "zero". Do the same thing but with the `next_parent_zero` pointer.
                if current_entry.next_parent_zero.is_undefined() {
                    current_entry.next_parent_zero = new_node;
                    unsafe {
                        self.push_node(variable, low, high);
                    }
                    return Ok(new_node);
                } else {
                    current_node = current_entry.next_parent_zero;
                }
            }

            // Rotate hash so that the next iteration will see the next top-most bit.
            hash = hash.rotate_left(1);
        }
    }

    /// Deletes the entry with the given `id` from the node table.
    ///
    /// The entry must not be a terminal node, but this condition is only checked in debug mode.
    ///
    /// # Warning
    ///
    /// Deleting an entry using this method does not update the parent counter of its children.
    ///
    /// Deleting an entry which has a parent (i.e., its `parent` pointer is not `undefined`)
    /// will result in the parent getting disconnected and unable to be found in
    /// the node table's search trees (such as in [`NodeTableImpl::ensure_node`]). This is
    /// okay as long as the parent is also deleted after.
    fn delete(&mut self, id: TNodeId) {
        debug_assert!(!id.is_terminal());
        debug_assert!(id.as_usize() < self.size());

        let entry = match self.get_entry_mut(id) {
            Some(entry) => entry,
            // The node is already deleted. We don't have to do anything.
            None => return,
        };

        let node = &entry.node;
        let low = node.low();
        let high = node.high();
        let deleted_entry_next_parent_one = entry.next_parent_one;
        let deleted_entry_next_parent_zero = entry.next_parent_zero;

        let max_child = max(low, high);

        let mut hash = hash_node_data(node.variable(), low, high);

        debug_assert!(low.as_usize() < self.size());
        debug_assert!(high.as_usize() < self.size());

        // Max child is the node through which we will enter the tree using its parent pointer.
        let tree_parent = match self.get_entry(max_child) {
            Some(entry) => entry,
            None => {
                // The node's max child is deleted. Call `max_child` x and its
                // entry's parent y (which is the root of the tree that should contain
                // our node). The bdd and the parent tree look the following way:
                //          ...                              ...
                //           |                                |
                //          id                                x
                //         /  \  parents of x                |p
                //        /    \ / /                         y
                //      ...     x                          /0 \1
                //             / \                       parents of x
                //             ...                          ...
                // Hence we expect `id` and all of the other parents of x to be
                // deleted (because it should not happen under normal usage that
                // x is unreachable while its parents are reachable). This means
                // we don't have to do any updates to the parent tree, and can
                // simply mark `id` as deleted.
                self.mark_node_as_deleted(id);
                return;
            }
        };
        // `tree_parent.parent` is the root of the tree that contains the node to be deleted.
        let root = tree_parent.parent;
        debug_assert!(!root.is_undefined());

        // We need to save the parent of the node to be deleted, so that we can update it later.
        // Note that this parent is *inside* the tree and is different from `tree_parent`.
        let (deleted_node_parent, deleted_node_is_one_child) =
            self.find_parent_of_node(id, root, &mut hash);

        // Now follow the hash until we find a leaf.
        // We will use this leaf to replace the deleted node.
        let (leaf, leaf_parent, leaf_is_one_child) = self.follow_hash_to_leaf(id, &mut hash);

        let deleted_node_is_leaf = leaf == id;
        let deleted_node_is_root = deleted_node_parent.is_undefined();
        let deleted_node_is_replacement_leafs_parent = leaf_parent == id;

        const LEAF: bool = true;
        const NOT_LEAF: bool = false;
        const ROOT: bool = true;
        const NOT_ROOT: bool = false;
        const LEAF_PARENT: bool = true;
        const NOT_LEAF_PARENT: bool = false;

        match (
            deleted_node_is_leaf,
            deleted_node_is_root,
            deleted_node_is_replacement_leafs_parent,
        ) {
            (LEAF, ROOT, NOT_LEAF_PARENT) => {
                // The tree contains exactly one node which we are deleting.
                // Update `tree_parent`'s parent pointer to undefined.
                let tree_parent = unsafe { self.get_entry_unchecked_mut(max_child) };
                tree_parent.parent = TNodeId::undefined();
            }
            (LEAF, NOT_ROOT, NOT_LEAF_PARENT) => {
                // The deleted node is a leaf and it is not the root of the tree.

                // Intuitively, `leaf_parent` should in this case be the same as
                // `deleted_node_parent`. However, `find_leaf_with_same_hash`
                // cannot find the parent of the leaf, since it starts at the leaf
                // itself.

                // Update the deleted node's parent's child pointer.
                self.replace_nodes_child(
                    deleted_node_parent,
                    deleted_node_is_one_child,
                    TNodeId::undefined(),
                );
            }
            (NOT_LEAF, NOT_ROOT, LEAF_PARENT) => {
                // The deleted node is not a leaf and it is not the root of the tree.
                // It is the parent of the leaf we are replacing it with.

                // Update the deleted node's parent's child pointer to be the leaf.
                self.replace_nodes_child(deleted_node_parent, deleted_node_is_one_child, leaf);

                // Update the leaf's child pointer to be the deleted node's (other) child.
                self.replace_nodes_child(
                    leaf,
                    !leaf_is_one_child,
                    if !leaf_is_one_child {
                        deleted_entry_next_parent_one
                    } else {
                        deleted_entry_next_parent_zero
                    },
                );
            }
            (NOT_LEAF, ROOT, LEAF_PARENT) => {
                // The deleted node is the root of the tree, but not a leaf.
                // Furthermore, it is the parent of the leaf we are replacing it
                // with, i.e., the leaf is its child.

                // Update `tree_parent`'s parent pointer to point to the leaf.
                let tree_parent = unsafe { self.get_entry_unchecked_mut(max_child) };
                tree_parent.parent = leaf;

                // Update the leaf's child pointer to be the deleted node's (other) child.
                self.replace_nodes_child(
                    leaf,
                    !leaf_is_one_child,
                    if !leaf_is_one_child {
                        deleted_entry_next_parent_one
                    } else {
                        deleted_entry_next_parent_zero
                    },
                );
            }
            (NOT_LEAF, ROOT, NOT_LEAF_PARENT) => {
                // The deleted node is the root of the tree, but not a leaf.
                // It is not the parent of the leaf we are replacing it with.

                // Update `tree_parent`'s parent pointer to point to the leaf.
                let tree_parent = unsafe { self.get_entry_unchecked_mut(max_child) };
                tree_parent.parent = leaf;

                // Update the leaf's parent pointer to undefined.
                self.replace_nodes_child(leaf_parent, leaf_is_one_child, TNodeId::undefined());

                // Update the leaf's child pointers to be the deleted node's children.
                self.replace_nodes_children(
                    leaf,
                    deleted_entry_next_parent_one,
                    deleted_entry_next_parent_zero,
                );
            }
            (NOT_LEAF, NOT_ROOT, NOT_LEAF_PARENT) => {
                // The deleted node is not a leaf and it is not the root of the tree.
                // It is not the parent of the leaf we are replacing it with.

                // Update the deleted node's parent's child pointer to be the leaf.
                self.replace_nodes_child(deleted_node_parent, deleted_node_is_one_child, leaf);

                // Update the leaf's child pointers to be the deleted node's children.
                self.replace_nodes_children(
                    leaf,
                    deleted_entry_next_parent_one,
                    deleted_entry_next_parent_zero,
                );

                // Update the leaf's parent's child pointer to be undefined.
                self.replace_nodes_child(leaf_parent, leaf_is_one_child, TNodeId::undefined());
            }
            (LEAF, _, LEAF_PARENT) => {
                // This case should never happen. If the deleted node is a leaf
                // then it cannot be its parent.
                unreachable!("deleted node is a leaf and also its parent");
            }
        }

        self.mark_node_as_deleted(id);
    }

    unsafe fn into_bdd<TBdd: BddAny<Id = TNodeId, VarId = TVarId, Node = TNode>>(
        self,
        root: TNodeId,
    ) -> TBdd {
        // Zero and one have special cases to always ensure that they are structurally equivalent
        // to the result of [`BddAny::new_false`]/[`BddAny::new_true`], regardless of what's in the
        // provided node table.
        if root.is_zero() {
            return TBdd::new_false();
        }
        if root.is_one() {
            return TBdd::new_true();
        }
        let nodes = self.entries.into_iter().map(|entry| entry.node).collect();
        unsafe { TBdd::new_unchecked(root, nodes) }
    }

    unsafe fn into_reachable_bdd<
        TBdd: BddAny<Id = Self::Id, VarId = Self::VarId, Node = Self::Node>,
    >(
        mut self,
        root: Self::Id,
    ) -> TBdd {
        // Zero and one have special cases to always ensure that they are structurally equivalent
        // to the result of [`BddAny::new_false`]/[`BddAny::new_true`], regardless of what's in the
        // provided node table.
        if root.is_zero() {
            return TBdd::new_false();
        }
        if root.is_one() {
            return TBdd::new_true();
        }

        let mut result = vec![TNode::zero(), TNode::one()];
        let mut stack = vec![root];

        let new_mark = self.current_mark.flipped();

        // `0` and `1` have correct translations.
        unsafe {
            let zero = self.get_entry_unchecked_mut(TNodeId::zero());
            zero.parent = TNodeId::zero();

            let one = self.get_entry_unchecked_mut(TNodeId::one());
            one.parent = TNodeId::one();
        }

        while let Some(id) = stack.pop() {
            let node = unsafe { self.get_node_unchecked(id) };
            if node.has_same_mark(new_mark) {
                continue;
            }

            let low = node.low();
            let high = node.high();
            let variable = node.variable();

            let low_entry = unsafe { self.get_entry_unchecked_mut(low) };
            let low_is_translated = low_entry.node.has_same_mark(new_mark);
            let new_low = low_entry.parent;

            let high_entry = unsafe { self.get_entry_unchecked_mut(high) };
            let high_is_translated = high_entry.node.has_same_mark(new_mark);
            let new_high = high_entry.parent;

            if low_is_translated && high_is_translated {
                result.push(TNode::new(variable.reset(), new_low, new_high));
                unsafe {
                    result
                        .get_unchecked_mut(new_low.as_usize())
                        .increment_parent_counter();
                    result
                        .get_unchecked_mut(new_high.as_usize())
                        .increment_parent_counter();
                }
                let entry = unsafe { self.get_entry_unchecked_mut(id) };
                entry.parent = (result.len() - 1).unchecked_into();
                entry.node.set_mark(new_mark);
                continue;
            }

            stack.push(id);

            if !high_is_translated {
                stack.push(high);
            }

            if !low_is_translated {
                stack.push(low);
            }
        }

        let new_root = unsafe { self.get_entry_unchecked_mut(root) }.parent;
        unsafe { TBdd::new_unchecked(new_root, result) }
    }
}

pub(crate) type NodeTable16 = NodeTableImpl<NodeId16, VarIdPacked16, BddNode16>;
pub(crate) type NodeTable32 = NodeTableImpl<NodeId32, VarIdPacked32, BddNode32>;
pub(crate) type NodeTable64 = NodeTableImpl<NodeId64, VarIdPacked64, BddNode64>;

macro_rules! impl_from_node_table {
    ($from:ident, $to:ident) => {
        impl From<$from> for $to {
            fn from(table: $from) -> Self {
                let entries = table
                    .entries
                    .into_iter()
                    .map(|entry| entry.into())
                    .collect();
                Self {
                    entries,
                    first_free: table.first_free.into(),
                    deleted: table.deleted,
                    current_mark: table.current_mark,
                }
            }
        }
    };
}

impl_from_node_table!(NodeTable16, NodeTable32);
impl_from_node_table!(NodeTable16, NodeTable64);
impl_from_node_table!(NodeTable32, NodeTable64);

const MAX_16_BIT_ID_USIZE: usize = NodeId16::MAX_ID as usize;
const MAX_32_BIT_ID_USIZE: usize = usize_is_at_least_32_bits(NodeId32::MAX_ID);
const MAX_64_BIT_ID_USIZE: usize = usize_is_at_least_64_bits(NodeId64::MAX_ID);

impl NodeTable16 {
    /// Returns `true` if the node table is full and a new node cannot be added.
    pub fn is_full(&self) -> bool {
        let is_full = self.first_free.is_undefined() && self.size() == MAX_16_BIT_ID_USIZE + 1;
        // This assertion is here to ensure that the fullness check in `ensure_node` is
        // conceptually the same as the one here.
        debug_assert_eq!(
            is_full,
            self.first_free.is_undefined() && TryInto::<NodeId16>::try_into(self.size()).is_err()
        );
        is_full
    }
}

impl NodeTable32 {
    /// Returns `true` if the node table is full and a new node cannot be added.
    pub fn is_full(&self) -> bool {
        let is_full = self.first_free.is_undefined() && self.size() == MAX_32_BIT_ID_USIZE + 1;
        // This assertion is here to ensure that the fullness check in `ensure_node` is
        // conceptually the same as the one here.
        debug_assert_eq!(
            is_full,
            self.first_free.is_undefined() && TryInto::<NodeId32>::try_into(self.size()).is_err()
        );
        is_full
    }
}

impl NodeTable64 {
    /// Returns `true` if the node table is full and a new node cannot be added.
    pub fn is_full(&self) -> bool {
        let is_full = self.first_free.is_undefined() && self.size() == MAX_64_BIT_ID_USIZE + 1;
        // This assertion is here to ensure that the fullness check in `ensure_node` is
        // conceptually the same as the one here.
        debug_assert_eq!(
            is_full,
            self.first_free.is_undefined() && TryInto::<NodeId64>::try_into(self.size()).is_err()
        );
        is_full
    }
}

#[derive(Debug)]
pub(crate) enum NodeTable {
    Size16(NodeTable16),
    Size32(NodeTable32),
    Size64(NodeTable64),
}

impl NodeTable {
    /// Returns `true` if the node table is full and a new node cannot be
    /// added until the table is grown.
    pub fn is_full(&self) -> bool {
        match self {
            NodeTable::Size16(table) => table.is_full(),
            NodeTable::Size32(table) => table.is_full(),
            NodeTable::Size64(table) => table.is_full(),
        }
    }

    /// Returns the number of entries in the node table,
    /// including the entries for the terminal nodes.
    pub fn node_count(&self) -> usize {
        match self {
            NodeTable::Size16(table) => table.node_count(),
            NodeTable::Size32(table) => table.node_count(),
            NodeTable::Size64(table) => table.node_count(),
        }
    }
}

impl Default for NodeTable {
    fn default() -> Self {
        NodeTable::Size16(NodeTable16::new())
    }
}

pub(crate) struct MarkPhaseData<TVarId: VarIdPackedAny> {
    reachable_count: usize,
    max_var_id: TVarId,
}

impl<TNodeId, TVarId, TNode> NodeTableImpl<TNodeId, TVarId, TNode>
where
    TNodeId: NodeIdAny,
    TVarId: VarIdPackedAny,
    TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
{
    /// Mark the reachable nodes in the table starting from the roots in `roots`.
    ///
    /// Returns the number of reachable nodes, and the maximum variable id.
    ///
    /// The `roots` vector is modified to only contain the roots that are still alive.
    /// The nodes' parent counters are recalculated to only count the reachable parents.
    fn mark_reachable(&mut self, roots: &mut Vec<Weak<Cell<NodeId>>>) -> MarkPhaseData<TVarId> {
        // Currently, all nodes have their mark set to `self.current_mark`.
        // Flip the mark to avoid having to explicitly reset the mark of all nodes.
        let new_mark = self.current_mark.flipped();
        self.current_mark = new_mark;

        // Terminal nodes are always reachable.
        let mut reachable_count = 2usize;
        let mut max_var_id = TVarId::undefined();

        roots.retain(|root| {
            if let Some(root) = root.upgrade() {
                let root_id: TNodeId = root.get().unchecked_into();
                let root_node = unsafe { self.get_node_unchecked_mut(root_id) };
                if root_node.has_same_mark(new_mark) {
                    return true;
                }
                root_node.set_mark(new_mark);
                root_node.reset_parent_counter();
                reachable_count += 1;

                let mut stack = vec![root_id];

                while let Some(id) = stack.pop() {
                    let node = unsafe { self.get_node_unchecked(id) };

                    let variable = node.variable();
                    max_var_id = max_var_id.max_defined(variable);

                    let high = node.high();
                    let low = node.low();

                    let high_node = unsafe { self.get_node_unchecked_mut(high) };
                    if !high_node.has_same_mark(new_mark) {
                        high_node.reset_parent_counter();
                        high_node.set_mark(new_mark);
                        reachable_count += 1;
                        stack.push(high);
                    }
                    high_node.increment_parent_counter();

                    let low_node = unsafe { self.get_node_unchecked_mut(low) };
                    if !low_node.has_same_mark(new_mark) {
                        low_node.reset_parent_counter();
                        low_node.set_mark(new_mark);
                        reachable_count += 1;
                        stack.push(low);
                    }
                    low_node.increment_parent_counter();
                }

                true
            } else {
                false
            }
        });

        MarkPhaseData {
            reachable_count,
            max_var_id,
        }
    }

    /// Rebuild the node table into a new one containing only the nodes reachable
    /// from `roots`.
    ///
    /// The roots in `roots` are translated to point to the corresponding nodes in the new table.
    ///
    /// # Panics
    ///
    /// This function panics if the variable ids of the nodes in the old table
    /// cannot be converted to variable ids of the new table or if the new table
    /// overflows. These conditions are expected to be checked by the caller.
    fn rebuild<TResultTable: NodeTableAny>(
        mut self,
        roots: &mut Vec<Weak<Cell<NodeId>>>,
        reachable: usize,
    ) -> TResultTable
    where
        TNodeId: UncheckedInto<TResultTable::Id>,
        TResultTable::Id: AsNodeId<TNodeId>,
        TVarId: UncheckedInto<TResultTable::VarId>,
    {
        let mut result = TResultTable::with_capacity(reachable);

        // Here we need to create a translation table from the old node ids to the new node ids.
        // We could use a vector for this, but since the old table is not used anymore, we can
        // reuse it to store the translations. This is safe, because the old table has
        // bit-width equal or higher to the new table, hence we can safely fit the new nodes
        // into it. We will use the entry's `parent` pointer to store the translation of the node.
        // This will not interfere with the traversal of the nodes, because the `parent`
        // pointer is not used during the traversal.
        //
        // Furthermore, we want to avoid initializing all of the `parent` pointers to `undefined`.
        // We can do so by using the mark of the nodes to store whether that
        // entry's parent is a translation or not.

        let new_mark = self.current_mark.flipped();

        // `0` and `1` have correct translations.
        unsafe {
            let zero = self.get_entry_unchecked_mut(TNodeId::zero());
            zero.parent = TNodeId::zero();

            let one = self.get_entry_unchecked_mut(TNodeId::one());
            one.parent = TNodeId::one();
        }

        for root in roots {
            if let Some(root) = root.upgrade() {
                let root_id: TNodeId = root.get().unchecked_into();
                let mut stack = vec![root_id];

                while let Some(id) = stack.pop() {
                    let node = unsafe { self.get_node_unchecked(id) };

                    let variable = node.variable();
                    let low = node.low();
                    let high = node.high();

                    let low_entry = unsafe { self.get_entry_unchecked_mut(low) };
                    let low_is_translated = low_entry.node.has_same_mark(new_mark);
                    let new_low = low_entry.parent;

                    let high_entry = unsafe { self.get_entry_unchecked_mut(high) };
                    let high_is_translated = high_entry.node.has_same_mark(new_mark);
                    let new_high = high_entry.parent;

                    if low_is_translated && high_is_translated {
                        let new_id = result
                            .ensure_node(
                                variable.unchecked_into(),
                                new_low.unchecked_into(),
                                new_high.unchecked_into(),
                            )
                            .expect(
                                "ensuring node while rebuilding node table should not overflow",
                            );
                        let entry = unsafe { self.get_entry_unchecked_mut(id) };
                        entry.parent = new_id.into();
                        entry.node.set_mark(new_mark);
                        continue;
                    }

                    stack.push(id);

                    if !high_is_translated {
                        stack.push(high);
                    }

                    if !low_is_translated {
                        stack.push(low);
                    }
                }

                debug_assert!(self
                    .get_node(root_id)
                    .is_some_and(|node| node.has_same_mark(new_mark)));
                let new_root = unsafe { self.get_entry_unchecked_mut(root_id).parent };
                root.set(new_root.unchecked_into());
            }
        }

        debug_assert_eq!(result.node_count(), reachable);
        result
    }

    /// Delete the nodes that are marked as unreachable from the table.
    fn delete_unreachable(&mut self) {
        for idx in 2..self.size() {
            let id: TNodeId = idx.unchecked_into();
            // Here we can't use `is_node_reachable_unchecked` because it uses
            // `get_entry_unchecked`, which panics if the node is deleted.
            // We want `is_node_reachable_unchecked` to panic, because checking
            // reachability of a deleted node is suspicious and probably a bug.
            // Hence we use `self.entries.get_unchecked` directly.
            // `delete` handles deleted nodes correctly.
            let entry = unsafe { self.entries.get_unchecked(idx) };
            if !entry.node.has_same_mark(self.current_mark) {
                self.delete(id);
            }
        }
    }

    /// Collect garbage in the node table by either deleting unreachable nodes or
    /// rebuilding the table.
    fn collect_garbage(mut self, roots: &mut Vec<Weak<Cell<NodeId>>>) -> NodeTable
    where
        Self: GarbageCollector<Table = Self>,
    {
        let MarkPhaseData {
            reachable_count,
            max_var_id,
        } = self.mark_reachable(roots);

        // Decide which garbage collection strategy to use. We take into account
        // the number of already deleted nodes in the table and the number of nodes
        // that are not reachable, i.e., to be deleted now.
        let total_nodes = self.size();
        let holes = total_nodes - self.node_count();
        let deletions = self.node_count() - reachable_count;
        let potential_fragmentation = holes + deletions;

        if potential_fragmentation.saturating_mul(Self::FRAGMENTATION_FACTOR) >= total_nodes {
            // If there would be a lot of holes in the table, perform a rebuild operation.
            self.rebuild_shrink(roots, reachable_count, max_var_id)
        } else {
            // Otherwise just delete the unreachable nodes.
            self.delete_unreachable()
        }
    }
}

type VarId<N> = <N as NodeTableAny>::VarId;

pub(crate) trait GarbageCollector {
    type Table: NodeTableAny;

    const SHRINK_THRESHOLD_16_BIT: usize = MAX_16_BIT_ID_USIZE >> 1;
    const SHRINK_THRESHOLD_32_BIT: usize = MAX_32_BIT_ID_USIZE >> 1;

    const FRAGMENTATION_FACTOR: usize = 3; // 3 representing about 33% fragmentation

    /// Rebuild the node table into a new one containing only the nodes reachable
    /// from `roots`. The resulting table's bit width is chosen to be as small as
    /// possible based on the number of reachable nodes and the maximum variable id.
    fn rebuild_shrink(
        self,
        roots: &mut Vec<Weak<Cell<NodeId>>>,
        reachable: usize,
        max_var_id: VarId<Self::Table>,
    ) -> NodeTable;

    /// Delete the unreachable nodes from the table.
    ///
    /// Note this would ideally be a function that takes a `&mut self` and modifies
    /// the table in place. However, `collect_garbage`, which uses this function,
    /// needs to return a table, so we have to take `self` by value.
    fn delete_unreachable(self) -> NodeTable;

    /// Collect garbage in the node table by either deleting unreachable nodes or
    /// rebuilding the table.
    fn collect_garbage(self, roots: &mut Vec<Weak<Cell<NodeId>>>) -> NodeTable;
}

impl GarbageCollector for NodeTable16 {
    type Table = NodeTable16;

    fn rebuild_shrink(
        self,
        roots: &mut Vec<Weak<Cell<NodeId>>>,
        reachable: usize,
        _max_var_id: VarId<Self::Table>,
    ) -> NodeTable {
        debug_assert!(reachable <= MAX_16_BIT_ID_USIZE);
        NodeTable::Size16(self.rebuild(roots, reachable))
    }

    fn delete_unreachable(mut self) -> NodeTable {
        Self::delete_unreachable(&mut self);
        NodeTable::Size16(self)
    }

    fn collect_garbage(self, roots: &mut Vec<Weak<Cell<NodeId>>>) -> NodeTable {
        self.collect_garbage(roots)
    }
}

impl GarbageCollector for NodeTable32 {
    type Table = NodeTable32;

    fn rebuild_shrink(
        self,
        roots: &mut Vec<Weak<Cell<NodeId>>>,
        reachable: usize,
        max_var_id: VarId<Self::Table>,
    ) -> NodeTable {
        debug_assert!(reachable <= MAX_32_BIT_ID_USIZE);
        if reachable <= Self::SHRINK_THRESHOLD_16_BIT && max_var_id.fits_in_packed16() {
            NodeTable::Size16(self.rebuild(roots, reachable))
        } else {
            NodeTable::Size32(self.rebuild(roots, reachable))
        }
    }

    fn delete_unreachable(mut self) -> NodeTable {
        Self::delete_unreachable(&mut self);
        NodeTable::Size32(self)
    }

    fn collect_garbage(self, roots: &mut Vec<Weak<Cell<NodeId>>>) -> NodeTable {
        self.collect_garbage(roots)
    }
}

impl GarbageCollector for NodeTable64 {
    type Table = NodeTable64;

    fn rebuild_shrink(
        self,
        roots: &mut Vec<Weak<Cell<NodeId>>>,
        reachable: usize,
        max_var_id: VarId<Self::Table>,
    ) -> NodeTable {
        debug_assert!(reachable <= MAX_64_BIT_ID_USIZE);
        if reachable <= Self::SHRINK_THRESHOLD_16_BIT && max_var_id.fits_in_packed16() {
            NodeTable::Size16(self.rebuild(roots, reachable))
        } else if reachable <= Self::SHRINK_THRESHOLD_32_BIT && max_var_id.fits_in_packed32() {
            NodeTable::Size32(self.rebuild(roots, reachable))
        } else {
            NodeTable::Size64(self.rebuild(roots, reachable))
        }
    }

    fn delete_unreachable(mut self) -> NodeTable {
        Self::delete_unreachable(&mut self);
        NodeTable::Size64(self)
    }

    fn collect_garbage(self, roots: &mut Vec<Weak<Cell<NodeId>>>) -> NodeTable {
        self.collect_garbage(roots)
    }
}

impl<TNodeId, TVarId, TNode> NodeTableImpl<TNodeId, TVarId, TNode>
where
    TNodeId: NodeIdAny,
    TVarId: VarIdPackedAny,
    TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
{
    /// Write the BDD rooted in `root` as a DOT graph to the given `output` stream.
    pub(crate) fn write_bdd_as_dot(&self, root: TNodeId, output: &mut dyn Write) -> io::Result<()> {
        assert!(self.get_entry(root).is_some());

        let mut seen = FxHashSet::default();
        seen.insert(root);
        seen.insert(TNodeId::zero());
        seen.insert(TNodeId::one());
        let mut stack = vec![root];
        writeln!(output, "digraph BDD {{")?;
        writeln!(
            output,
            "  __ruddy_root [label=\"\", style=invis, height=0, width=0];"
        )?;

        writeln!(output, "  __ruddy_root -> {root};")?;
        writeln!(output)?;
        writeln!(output, "  edge [dir=none];")?;
        writeln!(output)?;

        writeln!(
            output,
            "  0 [label=\"0\", shape=box, width=0.3, height=0.3];"
        )?;
        writeln!(
            output,
            "  1 [label=\"1\", shape=box, width=0.3, height=0.3];"
        )?;
        if root.is_terminal() {
            writeln!(output, "}}")?;
            return Ok(());
        }

        writeln!(output)?;

        while let Some(id) = stack.pop() {
            let node = unsafe { self.get_node_unchecked(id) };

            let low = node.low();
            let high = node.high();
            let variable = node.variable();

            writeln!(output, "  {id} [label=\"{variable}\", shape=circle];")?;
            writeln!(output, "  {id} -> {low} [style=dashed];")?;
            writeln!(output, "  {id} -> {high};",)?;

            if !seen.contains(&low) {
                seen.insert(low);
                stack.push(low);
            }

            if !seen.contains(&high) {
                seen.insert(high);
                stack.push(high);
            }
        }

        writeln!(output, "}}")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::{Rc, Weak};

    use crate::bdd_node::{BddNode32, BddNodeAny};
    use crate::conversion::UncheckedInto;
    use crate::node_id::{NodeId, NodeId16, NodeId32, NodeIdAny};
    use crate::node_table::{
        hash_node_data, NodeTable, NodeTable16, NodeTable32, NodeTable64, NodeTableAny,
    };
    use crate::split::bdd::{Bdd32, BddAny};
    use crate::variable_id::{
        VarIdPacked16, VarIdPacked32, VarIdPacked64, VarIdPackedAny, VariableId,
    };

    #[test]
    pub fn node_table_32_basic() {
        let mut table = NodeTable32::new();
        assert_eq!(2, table.node_count());
        assert_eq!(table, NodeTable32::with_capacity(1024));

        // Create some nodes
        let id1 = table
            .ensure_node(VarIdPacked32::new(4), NodeId32::zero(), NodeId32::one())
            .unwrap();
        let id1_p = table
            .ensure_node(VarIdPacked32::new(4), NodeId32::zero(), NodeId32::one())
            .unwrap();
        assert_eq!(id1, id1_p);
        let id2 = table.ensure_node(VarIdPacked32::new(3), id1, NodeId32::one());
        let id2_p = table.ensure_node(VarIdPacked32::new(3), id1, NodeId32::one());
        assert_eq!(id2, id2_p);

        assert_eq!(
            id1,
            table.ensure_node(VarIdPacked32::new(3), id1, id1).unwrap()
        );

        // Make a bunch of nodes that should all "collide" in the parent tree of the `1` node.
        let ids = (0..1000u32)
            .map(|i| table.ensure_node(VarIdPacked32::new(i), NodeId32::zero(), NodeId32::one()))
            .collect::<Vec<_>>();

        for (i, id) in ids.iter().enumerate() {
            let id_p = table.ensure_node(
                VarIdPacked32::new(i.try_into().unwrap()),
                NodeId32::zero(),
                NodeId32::one(),
            );
            assert_eq!(*id, id_p);
        }

        assert_eq!(1003, table.node_count());
    }

    #[test]
    pub fn node_table_conversions() {
        let table = NodeTable16::default();
        let converted = NodeTable32::from(table);
        assert_eq!(converted, NodeTable32::default());
        let converted = NodeTable64::from(converted);
        assert_eq!(converted, NodeTable64::default());
    }

    #[test]
    pub fn node_table_full() {
        let v = VarIdPacked16::new(10);
        let mut table = NodeTable16::default();

        for i in 2..u16::MAX {
            assert!(!table.is_full());
            let j = NodeId16::new(i - 1);
            let k = NodeId16::new(i - 2);
            assert!(table.ensure_node(v, j, k).is_ok());
        }

        assert!(table.is_full());
        let err = table
            .ensure_node(v, NodeId16::zero(), NodeId16::one())
            .unwrap_err();
        println!("{err}");
        assert_eq!(err.width, 16);

        table.delete(NodeId16::new(2));
        assert!(!table.is_full());
        assert!(table
            .ensure_node(VarIdPacked16::new(11), NodeId16::zero(), NodeId16::one())
            .is_ok());
        assert!(table.is_full());
        assert!(table
            .ensure_node(VarIdPacked16::new(12), NodeId16::zero(), NodeId16::one())
            .is_err());
    }

    #[test]
    fn delete_root_in_tree_with_only_root() {
        let mut table = NodeTable32::new();
        let root = table
            .ensure_node(VarIdPacked32::new(0), NodeId32::zero(), NodeId32::one())
            .unwrap();

        let one_entry = &table.entries[1];
        assert_eq!(one_entry.parent, root);

        table.delete(root);

        assert!(table.get_entry(root).is_none());
        assert!(table.entries[root.as_usize()].is_deleted());

        let one_entry = &table.entries[1];
        assert_eq!(one_entry.parent, NodeId32::undefined());
        assert_eq!(one_entry.next_parent_one, NodeId32::undefined());
        assert_eq!(one_entry.next_parent_zero, NodeId32::undefined());
        assert_eq!(table.node_count(), 2);
        assert_eq!(table.first_free, root);
        assert_eq!(table.deleted, 1);
    }

    #[test]
    fn delete_root_with_two_leaf_children() {
        let mut table = NodeTable32::new();
        let root = table
            .ensure_node(VarIdPacked32::new(0), NodeId32::zero(), NodeId32::one())
            .unwrap();

        // Forcefully insert the two nodes instead of using `ensure_node`. This is to
        // make sure that the hash does not interfere with the test.

        let leaf1 = NodeId32::new(3);
        unsafe { table.push_node(VarIdPacked32::new(1), NodeId32::zero(), NodeId32::one()) };
        table.get_entry_mut(root).unwrap().next_parent_one = leaf1;

        let leaf0 = NodeId32::new(4);
        unsafe { table.push_node(VarIdPacked32::new(2), NodeId32::zero(), NodeId32::one()) };
        table.get_entry_mut(root).unwrap().next_parent_zero = leaf0;

        table.delete(root);
        assert!(table.get_entry(root).is_none());
        assert!(table.entries[root.as_usize()].is_deleted());

        let new_root = table.entries[1].parent;
        assert!(new_root != root);
        assert!(new_root == leaf0 || new_root == leaf1);

        let new_root_entry = table.get_entry(new_root).unwrap();
        if new_root == leaf0 {
            assert_eq!(new_root_entry.next_parent_one, leaf1);
            assert_eq!(new_root_entry.next_parent_zero, NodeId32::undefined());
        } else {
            assert_eq!(new_root_entry.next_parent_one, NodeId32::undefined());
            assert_eq!(new_root_entry.next_parent_zero, leaf0);
        }
        assert_eq!(table.node_count(), 4);
        assert_eq!(table.first_free, root);
        assert_eq!(table.deleted, 1);
    }

    #[test]
    fn delete_leafs_parent() {
        // Make a tree which looks like this:
        //    1
        //    |p
        //    2 == root
        //  /0 \1
        // 3    4
        //    /0 \1
        //   5    6
        // and delete the node with id 4.

        let mut table = NodeTable32::new();
        let root = table
            .ensure_node(VarIdPacked32::new(0), NodeId32::zero(), NodeId32::one())
            .unwrap();

        let node3 = NodeId32::new(3);
        unsafe { table.push_node(VarIdPacked32::new(1), NodeId32::zero(), NodeId32::one()) };
        table.get_entry_mut(root).unwrap().next_parent_zero = node3;

        let node4 = NodeId32::new(4);
        unsafe { table.push_node(VarIdPacked32::new(2), NodeId32::zero(), NodeId32::one()) };
        table.get_entry_mut(root).unwrap().next_parent_one = node4;

        let leaf0 = NodeId32::new(5);
        unsafe { table.push_node(VarIdPacked32::new(3), NodeId32::zero(), NodeId32::one()) };
        table.get_entry_mut(node4).unwrap().next_parent_zero = leaf0;

        let leaf1 = NodeId32::new(6);
        unsafe { table.push_node(VarIdPacked32::new(4), NodeId32::zero(), NodeId32::one()) };
        table.get_entry_mut(node4).unwrap().next_parent_one = leaf1;

        table.delete(node4);

        assert!(table.get_entry(node4).is_none());
        assert!(table.entries[node4.as_usize()].is_deleted());

        let replacement = table.get_entry(root).unwrap().next_parent_one;
        assert!(replacement != node4);
        assert!(replacement == leaf0 || replacement == leaf1);

        let replacement_entry = table.get_entry(replacement).unwrap();
        if replacement == leaf0 {
            assert_eq!(replacement_entry.next_parent_one, leaf1);
            assert_eq!(replacement_entry.next_parent_zero, NodeId32::undefined());
        } else {
            assert_eq!(replacement_entry.next_parent_zero, leaf0);
            assert_eq!(replacement_entry.next_parent_one, NodeId32::undefined());
        }
        assert_eq!(table.node_count(), 6);
        assert_eq!(table.first_free, node4);
        assert_eq!(table.deleted, 1);
    }

    #[test]
    fn delete_root_in_tree_with_multiple_nodes() {
        let mut table = NodeTable32::new();
        let ids = (0..1000u32)
            .map(|i| {
                table
                    .ensure_node(VarIdPacked32::new(i), NodeId32::zero(), NodeId32::one())
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let root = ids[0];
        let root_entry = table.get_entry(root).unwrap();
        let old_root_next_parent_zero = root_entry.next_parent_zero;
        let old_root_next_parent_one = root_entry.next_parent_one;

        let one_entry = &table.entries[1];
        assert_eq!(one_entry.parent, root);

        let hash = hash_node_data(
            root_entry.node.variable,
            root_entry.node.low,
            root_entry.node.high,
        );
        let (leaf, _, _) = table.follow_hash_to_leaf(root, &mut hash.clone());
        let (parent_of_leaf, leaf_is_one_child) =
            table.find_parent_of_node(leaf, root, &mut hash.clone());
        let parent_entry = table.get_entry(parent_of_leaf).unwrap();
        let parent_old_next_parent_one = parent_entry.next_parent_one;
        let parent_old_next_parent_zero = parent_entry.next_parent_zero;

        table.delete(root);

        assert!(table.get_entry(root).is_none());
        assert!(table.entries[root.as_usize()].is_deleted());

        let one_entry = &table.entries[1];

        let new_root = one_entry.parent;
        assert_ne!(new_root, root);
        assert!(ids.contains(&new_root));
        assert_eq!(new_root, leaf);

        let new_root_entry = table.get_entry(new_root).unwrap();
        assert_eq!(new_root_entry.next_parent_zero, old_root_next_parent_zero);
        assert_eq!(new_root_entry.next_parent_one, old_root_next_parent_one);
        let parent_entry = table.get_entry(parent_of_leaf).unwrap();
        if leaf_is_one_child {
            assert_eq!(parent_entry.next_parent_one, NodeId32::undefined());
            assert_eq!(parent_entry.next_parent_zero, parent_old_next_parent_zero);
        } else {
            assert_eq!(parent_entry.next_parent_zero, NodeId32::undefined());
            assert_eq!(parent_entry.next_parent_one, parent_old_next_parent_one);
        }
        assert_eq!(table.node_count(), ids.len() - 1 + 2);
        assert_eq!(table.first_free, root);
        assert_eq!(table.deleted, 1);
    }

    #[test]
    fn delete_normal_node_in_tree_with_multiple_nodes() {
        let mut table = NodeTable32::new();
        let ids = (0..1000u32)
            .map(|i| {
                table
                    .ensure_node(VarIdPacked32::new(i), NodeId32::zero(), NodeId32::one())
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let root = ids[0];

        // Find a leaf and then traverse its parents to find a node that is "normal",
        // i.e., not a leaf or a root.
        let &leaf_start = ids.last().unwrap();
        let leaf_entry = table.get_entry(leaf_start).unwrap();
        let hash = hash_node_data(
            leaf_entry.node.variable,
            leaf_entry.node.low,
            leaf_entry.node.high,
        );

        let (parent, _) = table.find_parent_of_node(leaf_start, root, &mut hash.clone());

        let (parent, _) = table.find_parent_of_node(parent, root, &mut hash.clone());
        let (parent, _) = table.find_parent_of_node(parent, root, &mut hash.clone());

        let (node, _) = table.find_parent_of_node(parent, root, &mut hash.clone());
        let node_entry = table.get_entry(node).unwrap();
        let old_next_parent_zero = node_entry.next_parent_zero;
        let old_next_parent_one = node_entry.next_parent_one;
        let mut hash_node = hash_node_data(
            node_entry.node.variable,
            node_entry.node.low,
            node_entry.node.high,
        );

        let (leaf, leaf_parent, leaf_is_one_child) =
            table.follow_hash_to_leaf(root, &mut hash_node);
        let leaf_parent_entry = table.get_entry(leaf_parent).unwrap();
        let leaf_parent_old_next_parent_zero = leaf_parent_entry.next_parent_zero;
        let leaf_parent_old_next_parent_one = leaf_parent_entry.next_parent_one;

        let (parent, node_is_one_child) = table.find_parent_of_node(node, root, &mut hash.clone());
        let parent_entry = table.get_entry(parent).unwrap();
        let parent_old_next_parent_zero = parent_entry.next_parent_zero;
        let parent_old_next_parent_one = parent_entry.next_parent_one;

        // This is not actually a failure, but a precondition for the test.
        // This should not happen unless we get really unlucky with the hash, or
        // the hashing function is bad.
        assert_ne!(node, root);

        table.delete(node);
        assert!(table.get_entry(node).is_none());
        assert!(table.entries[node.as_usize()].is_deleted());

        // Check that its parent is correctly updated.
        let parent_entry = table.get_entry(parent).unwrap();
        if node_is_one_child {
            assert_eq!(parent_entry.next_parent_one, leaf);
            assert_eq!(parent_entry.next_parent_zero, parent_old_next_parent_zero);
        } else {
            assert_eq!(parent_entry.next_parent_zero, leaf);
            assert_eq!(parent_entry.next_parent_one, parent_old_next_parent_one);
        }

        // Check that the leaf's parent is correctly updated.
        let leaf_parent_entry = table.get_entry(leaf_parent).unwrap();
        if leaf_is_one_child {
            assert_eq!(leaf_parent_entry.next_parent_one, NodeId32::undefined());
            assert_eq!(
                leaf_parent_entry.next_parent_zero,
                leaf_parent_old_next_parent_zero
            );
        } else {
            assert_eq!(leaf_parent_entry.next_parent_zero, NodeId32::undefined());
            assert_eq!(
                leaf_parent_entry.next_parent_one,
                leaf_parent_old_next_parent_one
            );
        }

        // Check that the node which replaced the deleted one has correct children.
        let replacement = table.get_entry(leaf).unwrap();
        assert_eq!(replacement.next_parent_one, old_next_parent_one);
        assert_eq!(replacement.next_parent_zero, old_next_parent_zero);

        assert_eq!(table.node_count(), ids.len() - 1 + 2);
        assert_eq!(table.first_free, node);
        assert_eq!(table.deleted, 1);
    }

    #[test]
    fn delete_leaf_in_tree_with_multiple_nodes() {
        let mut table = NodeTable32::new();
        let ids = (0..1000u32)
            .map(|i| {
                table
                    .ensure_node(VarIdPacked32::new(i), NodeId32::zero(), NodeId32::one())
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let root = ids[0];
        let leaf = *ids.last().unwrap();
        let leaf_entry = table.get_entry(leaf).unwrap();
        let hash = hash_node_data(
            leaf_entry.node.variable,
            leaf_entry.node.low,
            leaf_entry.node.high,
        );

        let (leaf_parent, leaf_is_one_child) =
            table.find_parent_of_node(leaf, root, &mut hash.clone());

        let leaf_parent_entry = table.get_entry(leaf_parent).unwrap();
        let leaf_parent_old_next_parent_one = leaf_parent_entry.next_parent_one;
        let leaf_parent_old_next_parent_zero = leaf_parent_entry.next_parent_zero;

        table.delete(leaf);

        assert!(table.get_entry(leaf).is_none());
        assert!(table.entries[leaf.as_usize()].is_deleted());

        let leaf_parent_entry = table.get_entry(leaf_parent).unwrap();

        if leaf_is_one_child {
            assert_eq!(leaf_parent_entry.next_parent_one, NodeId32::undefined());
            assert_eq!(
                leaf_parent_entry.next_parent_zero,
                leaf_parent_old_next_parent_zero
            )
        } else {
            assert_eq!(leaf_parent_entry.next_parent_zero, NodeId32::undefined());
            assert_eq!(
                leaf_parent_entry.next_parent_one,
                leaf_parent_old_next_parent_one
            );
        }
        assert_eq!(table.node_count(), ids.len() - 1 + 2);
        assert_eq!(table.first_free, leaf);
        assert_eq!(table.deleted, 1);
    }

    #[test]
    fn free_list_delete_nodes_then_reinsert() {
        let mut table = NodeTable32::new();
        let nodes = 1001u32;

        let ids = (0..nodes)
            .map(|i| {
                table
                    .ensure_node(VarIdPacked32::new(i), NodeId32::zero(), NodeId32::one())
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let mut table_delete = table.clone();

        // Delete every node with even id.
        for &id in ids.iter().rev() {
            if id.as_usize() % 2 == 0 {
                table_delete.delete(id);
            }
        }

        assert_eq!(table_delete.deleted, (nodes / 2) as usize + 1);

        // Check that the free list is correct.
        assert_eq!(table_delete.first_free, *ids.first().unwrap());
        assert_eq!(
            table_delete.entries[ids.last().unwrap().as_usize()]
                .node
                .high,
            NodeId32::undefined()
        );

        for &id in ids.iter().rev().skip(1) {
            if id.as_usize() % 2 == 0 {
                assert!(table_delete.entries[id.as_usize()].is_deleted());
                assert_eq!(
                    table_delete.entries[id.as_usize()].node.high,
                    NodeId32::new(UncheckedInto::<u32>::unchecked_into(id) + 2)
                );
            }
        }

        // Reinsert the nodes back.
        for &id in ids.iter() {
            let id_u32: u32 = id.unchecked_into();
            if id_u32 % 2 == 0 {
                table_delete
                    .ensure_node(
                        VarIdPacked32::new(id_u32 - 2),
                        NodeId32::zero(),
                        NodeId32::one(),
                    )
                    .unwrap();
            }
        }

        assert_eq!(table_delete.deleted, 0);
        assert_eq!(table_delete.first_free, NodeId32::undefined());

        // The tables will have different trees, but all the nodes should be the same.
        for id in ids {
            assert_eq!(
                table.get_node(id).unwrap(),
                table_delete.get_node(id).unwrap()
            );
        }
    }

    #[test]
    fn delete_deleted_node() {
        let mut table = NodeTable32::new();
        let root = table
            .ensure_node(VarIdPacked32::new(0), NodeId32::zero(), NodeId32::one())
            .unwrap();

        table.delete(root);
        table.delete(root);

        assert!(table.get_entry(root).is_none());
        assert!(table.entries[root.as_usize()].is_deleted());
        assert_eq!(table.node_count(), 2);
        assert_eq!(table.first_free, root);
        assert_eq!(table.deleted, 1);
    }

    #[test]
    fn delete_node_which_is_parent_of_tree_and_then_the_tree() {
        // Make a tree which looks like this:
        //           1
        //           |p
        //           2
        //            \1
        //             3
        //          /0 |p
        //         4   5
        //           /0 \1
        //          6    7
        // and delete the node with id 3, then 6,7, and 8.

        let mut table = NodeTable32::new();

        let node2 = table
            .ensure_node(VarIdPacked32::new(10), NodeId32::zero(), NodeId32::one())
            .unwrap();

        let node3 = NodeId32::new(3);
        unsafe { table.push_node(VarIdPacked32::new(11), NodeId32::zero(), NodeId32::one()) };
        table.get_entry_mut(node2).unwrap().next_parent_one = node3;

        let node4 = NodeId32::new(4);
        unsafe { table.push_node(VarIdPacked32::new(13), NodeId32::zero(), NodeId32::one()) };
        table.get_entry_mut(node3).unwrap().next_parent_zero = node4;

        let node5 = NodeId32::new(5);
        unsafe { table.push_node(VarIdPacked32::new(0), node3, NodeId32::one()) };
        table.get_entry_mut(node3).unwrap().parent = node5;

        let node6 = NodeId32::new(6);
        unsafe { table.push_node(VarIdPacked32::new(1), node3, NodeId32::one()) };
        table.get_entry_mut(node5).unwrap().next_parent_zero = node6;

        let node7 = NodeId32::new(7);
        unsafe { table.push_node(VarIdPacked32::new(2), node3, NodeId32::one()) };
        table.get_entry_mut(node5).unwrap().next_parent_one = node7;

        table.delete(node3);
        assert!(table.get_entry(node3).is_none());
        assert!(table.entries[node3.as_usize()].is_deleted());
        assert_eq!(table.first_free, node3);

        table.delete(node5);
        assert!(table.get_entry(node5).is_none());
        assert!(table.entries[node5.as_usize()].is_deleted());
        assert_eq!(table.first_free, node5);

        table.delete(node6);
        assert!(table.get_entry(node6).is_none());
        assert!(table.entries[node6.as_usize()].is_deleted());
        assert_eq!(table.first_free, node6);

        table.delete(node7);
        assert!(table.get_entry(node7).is_none());
        assert!(table.entries[node7.as_usize()].is_deleted());
        assert_eq!(table.first_free, node7);

        assert_eq!(table.node_count(), 4);
        assert_eq!(table.deleted, 4);

        // Check that the remaining tree is correct.
        let one_entry = table.get_entry(NodeId32::one()).unwrap();
        assert_eq!(one_entry.parent, node2);

        let node2_entry = table.get_entry(node2).unwrap();
        assert_eq!(node2_entry.next_parent_zero, NodeId32::undefined());
        assert_eq!(node2_entry.next_parent_one, node4);
        assert_eq!(node2_entry.parent, NodeId32::undefined());

        let node4_entry = table.get_entry(node4).unwrap();
        assert_eq!(node4_entry.next_parent_zero, NodeId32::undefined());
        assert_eq!(node4_entry.next_parent_one, NodeId32::undefined());
        assert_eq!(node4_entry.parent, NodeId32::undefined());
    }

    #[test]
    fn mark_reachable() {
        let mut table = NodeTable32::new();
        let mut ids_reachable = vec![NodeId32::zero(), NodeId32::one()];

        let mut reachable_nonterminal = 10;

        for i in 0..reachable_nonterminal {
            let id = table
                .ensure_node(
                    VarIdPacked32::new(reachable_nonterminal - i),
                    *ids_reachable.last().unwrap(),
                    NodeId32::zero(),
                )
                .unwrap();
            ids_reachable.push(id);
        }

        let reachable_max_var_id = VarIdPacked32::new(reachable_nonterminal);

        let root_32 = *ids_reachable.last().unwrap();
        let root: NodeId = root_32.unchecked_into();

        let root_subtree_32 = ids_reachable[(reachable_nonterminal / 2) as usize];
        let root_subtree: NodeId = root_subtree_32.unchecked_into();

        let superroot_32 = table
            .ensure_node(VarIdPacked32::new(0), root_32, root_subtree_32)
            .unwrap();
        let superroot: NodeId = superroot_32.unchecked_into();
        reachable_nonterminal += 1;

        let unreachable_nonterminal_1 = 5;
        let mut ids_unreachable_1 = vec![];

        for i in 0..unreachable_nonterminal_1 {
            let id = table
                .ensure_node(
                    VarIdPacked32::new(10 * (unreachable_nonterminal_1 - i)),
                    *ids_unreachable_1.last().unwrap_or(&NodeId32::zero()),
                    root_32,
                )
                .unwrap();
            ids_unreachable_1.push(id);
        }

        let unreachable_root1: NodeId = (*ids_unreachable_1.last().unwrap()).unchecked_into();

        let unreachable_nonterminal_2 = 25;
        let mut ids_unreachable_2 = vec![];

        for i in 0..unreachable_nonterminal_2 {
            let id = table
                .ensure_node(
                    VarIdPacked32::new(100 * (unreachable_nonterminal_2 - i)),
                    *ids_unreachable_2.last().unwrap_or(&NodeId32::zero()),
                    root_subtree_32,
                )
                .unwrap();
            ids_unreachable_2.push(id);
        }

        let unreachable_root2: NodeId = (*ids_unreachable_2.last().unwrap()).unchecked_into();

        assert!(table.get_node(root_32).unwrap().has_many_parents());
        assert!(table.get_node(root_subtree_32).unwrap().has_many_parents());

        assert_eq!(
            table.node_count(),
            (2 + reachable_nonterminal + unreachable_nonterminal_1 + unreachable_nonterminal_2)
                as usize
        );

        let root = Rc::new(Cell::new(root));
        let root_subtree = Rc::new(Cell::new(root_subtree));
        let superroot = Rc::new(Cell::new(superroot));

        let mut roots = vec![
            Rc::downgrade(&root),
            Rc::downgrade(&root_subtree),
            Rc::downgrade(&superroot),
        ];

        {
            let unreachable_root1 = Rc::new(Cell::new(unreachable_root1));
            let unreachable_root2 = Rc::new(Cell::new(unreachable_root2));

            roots.push(Rc::downgrade(&unreachable_root1));
            roots.push(Rc::downgrade(&unreachable_root2));
        }

        let mark_data = table.mark_reachable(&mut roots);
        assert_eq!(roots.len(), 3);
        assert_eq!(roots[0].upgrade().unwrap().get(), root.get());
        assert_eq!(roots[1].upgrade().unwrap().get(), root_subtree.get());
        assert_eq!(roots[2].upgrade().unwrap().get(), superroot.get());

        assert_eq!(
            mark_data.reachable_count,
            2 + reachable_nonterminal as usize
        );
        assert_eq!(mark_data.max_var_id, reachable_max_var_id);

        for &id in ids_reachable.iter() {
            assert!(table
                .get_node(id)
                .unwrap()
                .has_same_mark(table.current_mark));
            if !id.is_terminal() && id != root_32 && id != root_subtree_32 {
                // The reachable nodes should have exactly one parent.
                let node = &mut table.get_entry_mut(id).unwrap().node;
                assert!(!node.has_many_parents());
                node.increment_parent_counter();
                assert!(node.has_many_parents());
            }
        }
        // The root should have one parent
        let root_node = &mut table.get_entry_mut(root_32).unwrap().node;
        assert!(!root_node.has_many_parents());
        root_node.increment_parent_counter();
        assert!(root_node.has_many_parents());

        // The root subtree should have two parents
        let root_subtree = &mut table.get_entry_mut(root_subtree_32).unwrap().node;
        assert!(root_subtree.has_many_parents());

        for &id in ids_unreachable_1.iter() {
            assert!(!table
                .get_node(id)
                .unwrap()
                .has_same_mark(table.current_mark));
        }

        for &id in ids_unreachable_2.iter() {
            assert!(!table
                .get_node(id)
                .unwrap()
                .has_same_mark(table.current_mark));
        }
    }

    #[allow(clippy::type_complexity)]
    fn make_roots<TNodeId: NodeIdAny>(
        strong_ids: &[TNodeId],
        weak_ids: &[TNodeId],
    ) -> (Vec<Rc<Cell<NodeId>>>, Vec<Weak<Cell<NodeId>>>) {
        let mut all_roots = vec![];
        let mut strongs = vec![];

        for &id in strong_ids {
            let strong = Rc::new(Cell::new(id.unchecked_into()));
            all_roots.push(strong.clone());
            strongs.push(strong);
        }

        for &id in weak_ids {
            all_roots.push(Rc::new(Cell::new(id.unchecked_into())));
        }

        let weaks = all_roots.iter().map(Rc::downgrade).collect();

        (strongs, weaks)
    }

    #[allow(clippy::type_complexity)]
    fn init_rebuild<
        FromNodeTable: NodeTableAny<Id = FromNodeId, VarId = FromVarId>,
        ToNodeTable: NodeTableAny<Id = ToNodeId, VarId = ToVarId>,
        FromNodeId: NodeIdAny,
        ToNodeId: NodeIdAny,
        FromVarId: VarIdPackedAny,
        ToVarId: VarIdPackedAny,
    >(
        table: &mut FromNodeTable,
        reachable: usize,
    ) -> (
        ToNodeTable,
        ToNodeId,
        Rc<Cell<NodeId>>,
        Vec<Weak<Cell<NodeId>>>,
        FromVarId,
    ) {
        let var_unreachable_1 = VariableId::new(100);

        let unreachable_1 = 10;
        let mut ids = vec![FromNodeId::one()];
        for _ in 2..unreachable_1 {
            let id = table
                .ensure_node(
                    var_unreachable_1.unchecked_into(),
                    ids[ids.len() - 1],
                    FromNodeId::zero(),
                )
                .unwrap();
            ids.push(id);
        }

        let unreachable_root1 = ids[ids.len() - 1];

        let mut ids = vec![FromNodeId::zero(), FromNodeId::one()];

        let var_reachable = VariableId::new(10);

        for _ in 2..reachable {
            let id = table
                .ensure_node(
                    var_reachable.unchecked_into(),
                    ids[ids.len() - 2],
                    ids[ids.len() - 1],
                )
                .unwrap();
            ids.push(id);
        }

        let root = ids[ids.len() - 1];

        let var_unreachable_2 = VariableId::new(1);
        let unreachable_2 = 11;
        for _ in 2..unreachable_2 {
            let id = table
                .ensure_node(
                    var_unreachable_2.unchecked_into(),
                    ids[ids.len() - 2],
                    ids[ids.len() - 1],
                )
                .unwrap();
            ids.push(id);
        }

        let unreachable_root2 = ids[ids.len() - 1];

        let (strongs, roots) = make_roots(&[root], &[unreachable_root1, unreachable_root2]);
        assert_eq!(strongs.len(), 1);

        let mut expected = ToNodeTable::default();
        let mut ids_to = vec![ToNodeId::zero(), ToNodeId::one()];

        for _ in 2..reachable {
            let id = expected
                .ensure_node(
                    var_reachable.unchecked_into(),
                    ids_to[ids_to.len() - 2],
                    ids_to[ids_to.len() - 1],
                )
                .unwrap();
            ids_to.push(id);
        }

        let expected_root = ids_to[ids_to.len() - 1];

        (
            expected,
            expected_root,
            strongs[0].clone(),
            roots,
            var_reachable.unchecked_into(),
        )
    }

    #[test]
    fn rebuild_32_to_16() {
        let mut table = NodeTable32::new();
        // Make sure that the mark does not interfere with rebuilding.
        table.current_mark = table.current_mark.flipped();

        let reachable = u16::MAX as usize;

        let (expected, expected_root, root, mut roots, _) =
            init_rebuild::<_, NodeTable16, _, _, _, _>(&mut table, reachable);

        let prev_root = root.get();
        let mut rebuilt = table.rebuild::<NodeTable16>(&mut roots, reachable);

        assert_eq!(rebuilt.node_count(), reachable);

        // The root got correctly translated
        assert_ne!(root.get(), prev_root);
        assert_eq!(root.get(), expected_root.unchecked_into());

        assert_eq!(expected, rebuilt);

        assert!(rebuilt
            .ensure_node(VarIdPacked16::new(1000), NodeId16::zero(), NodeId16::one())
            .is_err());
    }

    #[test]
    fn rebuild_64_to_16() {
        let mut table = NodeTable64::new();
        // Make sure that the mark does not interfere with rebuilding.
        table.current_mark = table.current_mark.flipped();

        let reachable = u16::MAX as usize;

        let (expected, expected_root, root, mut roots, _) =
            init_rebuild::<_, NodeTable16, _, _, _, _>(&mut table, reachable);

        let prev_root = root.get();
        let mut rebuilt = table.rebuild::<NodeTable16>(&mut roots, reachable);

        assert_eq!(rebuilt.node_count(), reachable);

        // The root got correctly translated
        assert_ne!(root.get(), prev_root);
        assert_eq!(root.get(), expected_root.unchecked_into());

        assert_eq!(expected, rebuilt);

        assert!(rebuilt
            .ensure_node(VarIdPacked16::new(1000), NodeId16::zero(), NodeId16::one())
            .is_err());
    }

    #[test]
    fn rebuild_64_to_32() {
        let mut table = NodeTable64::new();
        // Make sure that the mark does not interfere with rebuilding.
        table.current_mark = table.current_mark.flipped();

        // Making 2**32 nodes is impractical, so just 2**17 for now.
        let reachable = (u16::MAX as usize) * 2;

        let (expected, expected_root, root, mut roots, _) =
            init_rebuild::<_, NodeTable32, _, _, _, _>(&mut table, reachable);

        let prev_root = root.get();
        let rebuilt = table.rebuild::<NodeTable32>(&mut roots, reachable);

        assert_eq!(rebuilt.node_count(), reachable);

        // The root got correctly translated
        assert_ne!(root.get(), prev_root);
        assert_eq!(root.get(), expected_root.unchecked_into());

        assert_eq!(expected, rebuilt);
    }

    #[test]
    fn delete_unreachable() {
        let mut table = NodeTable32::new();

        let nodes = 500;
        let unreachable_ids = (0..nodes)
            .map(|i| {
                table
                    .ensure_node(VarIdPacked32::new(i), NodeId32::zero(), NodeId32::one())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let reachable_ids = (0..nodes)
            .map(|i| {
                table
                    .ensure_node(
                        VarIdPacked32::new(nodes + i),
                        NodeId32::zero(),
                        NodeId32::one(),
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        // At this point all of the nodes are considered reachable.
        // Flip mark and remark reachable nodes.
        let new_mark = table.current_mark.flipped();
        table.current_mark = new_mark;

        for &id in reachable_ids.iter() {
            table.get_entry_mut(id).unwrap().node.set_mark(new_mark);
        }

        for &id in unreachable_ids.iter() {
            assert!(!table.get_node(id).unwrap().has_same_mark(new_mark));
        }

        table.delete_unreachable();

        assert_eq!(table.node_count(), nodes as usize + 2);
        assert_eq!(table.deleted, nodes as usize);

        for &id in reachable_ids.iter() {
            assert!(table.get_entry(id).is_some());
            assert!(!table.entries[id.as_usize()].is_deleted());
            assert!(table.get_node(id).unwrap().has_same_mark(new_mark));
        }

        for &id in unreachable_ids.iter() {
            assert!(table.get_entry(id).is_none());
            assert!(table.entries[id.as_usize()].is_deleted());
        }

        let reachable_ids_after_delete = (0..nodes)
            .map(|i| {
                table
                    .ensure_node(
                        VarIdPacked32::new(nodes + i),
                        NodeId32::zero(),
                        NodeId32::one(),
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        assert_eq!(reachable_ids, reachable_ids_after_delete);

        // Deleting the same nodes again should not change anything.
        table.delete_unreachable();

        assert_eq!(table.node_count(), nodes as usize + 2);
        assert_eq!(table.deleted, nodes as usize);

        // Delete the rest of the nodes.
        table.current_mark = table.current_mark.flipped();
        table.delete_unreachable();

        assert_eq!(table.node_count(), 2);
        assert_eq!(table.deleted, 2 * nodes as usize);

        for entry in table.entries.iter().skip(2) {
            assert!(entry.is_deleted());
        }
    }

    fn init_rebuild_does_not_shrink<TNodeTable: NodeTableAny>(
        table: &mut TNodeTable,
        big: TNodeTable::VarId,
        small: TNodeTable::VarId,
    ) -> (TNodeTable::Id, TNodeTable::Id) {
        let root_big = table
            .ensure_node(big, TNodeTable::Id::zero(), TNodeTable::Id::one())
            .unwrap();

        let root_small = table
            .ensure_node(small, TNodeTable::Id::zero(), TNodeTable::Id::one())
            .unwrap();

        for i in 0..1000 {
            table
                .ensure_node(
                    VariableId::new(i).unchecked_into(),
                    TNodeTable::Id::zero(),
                    TNodeTable::Id::one(),
                )
                .unwrap();
        }

        (root_big, root_small)
    }

    #[test]
    fn rebuild_table_64_with_64_bit_var_does_not_shrink() {
        let mut table = NodeTable64::new();

        let big = VarIdPacked64::new(VarIdPacked32::MAX_ID as u64 + 1);
        let small = VarIdPacked64::new(0);
        let reachable: usize = 4;

        let (root_big, root_small) = init_rebuild_does_not_shrink(&mut table, big, small);

        let (_strongs, mut roots) = make_roots(&[root_big, root_small], &[]);

        use super::GarbageCollector;
        let rebuilt = match table.rebuild_shrink(&mut roots, reachable, big) {
            NodeTable::Size64(table) => table,
            _ => panic!("expected 64-bit table"),
        };

        assert_eq!(rebuilt.node_count(), reachable);
        assert_eq!(rebuilt.get_node(root_big).unwrap().variable, big);
        assert_eq!(rebuilt.get_node(root_small).unwrap().variable, small);
    }

    #[test]
    fn rebuild_table_32_with_32_bit_var_does_not_shrink() {
        let mut table = NodeTable32::new();

        let big = VarIdPacked32::new(VarIdPacked16::MAX_ID as u32 + 1);
        let small = VarIdPacked32::new(0);
        let reachable: usize = 4;

        let (root_big, root_small) = init_rebuild_does_not_shrink(&mut table, big, small);

        let (_strongs, mut roots) = make_roots(&[root_big, root_small], &[]);

        use super::GarbageCollector;
        let rebuilt = match table.rebuild_shrink(&mut roots, reachable, big) {
            NodeTable::Size32(table) => table,
            _ => panic!("expected 32-bit table"),
        };

        assert_eq!(rebuilt.node_count(), reachable);
        assert_eq!(rebuilt.get_node(root_big).unwrap().variable, big);
        assert_eq!(rebuilt.get_node(root_small).unwrap().variable, small);
    }

    #[test]
    fn rebuild_table_64_with_32_bit_var_does_not_shrink_to_16() {
        let mut table = NodeTable64::new();

        let big_32 = VarIdPacked32::new(VarIdPacked16::MAX_ID as u32 + 1);
        let big = big_32.into();
        let small_32 = VarIdPacked32::new(0);
        let small = small_32.into();
        let reachable: usize = 4;

        let (root_big, root_small) = init_rebuild_does_not_shrink(&mut table, big, small);

        let (_strongs, mut roots) = make_roots(&[root_big, root_small], &[]);

        use super::GarbageCollector;
        let rebuilt = match table.rebuild_shrink(&mut roots, reachable, big) {
            NodeTable::Size32(table) => table,
            _ => panic!("expected 64-bit table"),
        };

        assert_eq!(rebuilt.node_count(), reachable);
        assert_eq!(
            rebuilt
                .get_node(root_big.unchecked_into())
                .unwrap()
                .variable,
            big_32
        );
        assert_eq!(
            rebuilt
                .get_node(root_small.unchecked_into())
                .unwrap()
                .variable,
            small_32
        );
    }

    #[test]
    fn rebuild_shrink_64_to_16() {
        use super::GarbageCollector;
        let mut table = NodeTable64::new();
        // Make sure that the mark does not interfere with rebuilding.
        table.current_mark = table.current_mark.flipped();

        let reachable = NodeTable64::SHRINK_THRESHOLD_16_BIT;

        let (expected, expected_root, root, mut roots, max_var) =
            init_rebuild::<_, NodeTable16, _, _, _, _>(&mut table, reachable);

        let prev_root = root.get();
        let rebuilt = match table.rebuild_shrink(&mut roots, reachable, max_var) {
            NodeTable::Size16(table) => table,
            _ => panic!("expected 16-bit table"),
        };

        assert_eq!(rebuilt.node_count(), reachable);

        // The root got correctly translated
        assert_ne!(root.get(), prev_root);
        assert_eq!(root.get(), expected_root.unchecked_into());

        assert_eq!(expected, rebuilt);
    }

    #[test]
    fn rebuild_shrink_64_to_32() {
        use super::GarbageCollector;
        let mut table = NodeTable64::new();
        // Make sure that the mark does not interfere with rebuilding.
        table.current_mark = table.current_mark.flipped();

        let reachable = NodeTable64::SHRINK_THRESHOLD_16_BIT + 1;

        let (expected, expected_root, root, mut roots, max_var) =
            init_rebuild::<_, NodeTable32, _, _, _, _>(&mut table, reachable);

        let prev_root = root.get();
        let rebuilt = match table.rebuild_shrink(&mut roots, reachable, max_var) {
            NodeTable::Size32(table) => table,
            _ => panic!("expected 32-bit table"),
        };

        assert_eq!(rebuilt.node_count(), reachable);

        // The root got correctly translated
        assert_ne!(root.get(), prev_root);
        assert_eq!(root.get(), expected_root.unchecked_into());

        assert_eq!(expected, rebuilt);
    }

    #[test]
    fn rebuild_shrink_32_to_16() {
        use super::GarbageCollector;
        let mut table = NodeTable32::new();
        // Make sure that the mark does not interfere with rebuilding.
        table.current_mark = table.current_mark.flipped();

        let reachable = NodeTable32::SHRINK_THRESHOLD_16_BIT;

        let (expected, expected_root, root, mut roots, max_var) =
            init_rebuild::<_, NodeTable16, _, _, _, _>(&mut table, reachable);

        let prev_root = root.get();
        let rebuilt = match table.rebuild_shrink(&mut roots, reachable, max_var) {
            NodeTable::Size16(table) => table,
            _ => panic!("expected 16-bit table"),
        };

        assert_eq!(rebuilt.node_count(), reachable);

        // The root got correctly translated
        assert_ne!(root.get(), prev_root);
        assert_eq!(root.get(), expected_root.unchecked_into());

        assert_eq!(expected, rebuilt);
    }

    #[test]
    fn collect_garbage_delete() {
        let mut table = NodeTable32::new();

        // Make a small tree that will be unreachable
        let unreachable = 10;
        let mut ids_unreachable = vec![];
        for _ in 0..unreachable {
            let id = table
                .ensure_node(
                    VarIdPacked32::new(100),
                    NodeId32::one(),
                    *ids_unreachable.last().unwrap_or(&NodeId32::zero()),
                )
                .unwrap();
            ids_unreachable.push(id);
        }

        let unreachable_root = ids_unreachable[ids_unreachable.len() - 1];

        // And a few more unreachable nodes
        for i in 0..10 {
            let id = table
                .ensure_node(VarIdPacked32::new(i), NodeId32::zero(), NodeId32::one())
                .unwrap();
            ids_unreachable.push(id);
        }

        // Make a big reachable tree
        let reachable = 1000;
        let mut ids_reachable = vec![NodeId32::zero(), NodeId32::one()];
        for i in 2..reachable {
            let id = table
                .ensure_node(
                    VarIdPacked32::new(1000),
                    ids_reachable[i - 2],
                    ids_reachable[i - 1],
                )
                .unwrap();
            ids_reachable.push(id);
        }

        let root = ids_reachable[ids_reachable.len() - 1];

        let (_strongs, mut roots) = make_roots(&[root], &[unreachable_root]);

        fn validate_table_after_gc(
            new_table: &mut NodeTable32,
            roots: &[Weak<Cell<NodeId>>],
            reachable: usize,
            ids_unreachable: &[NodeId32],
            ids_reachable: &[NodeId32],
        ) {
            assert_eq!(roots.len(), 1);
            assert_eq!(new_table.node_count(), reachable);

            // Check unreachable nodes
            for &id in ids_unreachable.iter() {
                assert!(new_table.get_entry(id).is_none());
                assert!(new_table.entries[id.as_usize()].is_deleted());
            }

            // Check reachable nodes
            for &id in ids_reachable.iter() {
                assert!(new_table.get_entry(id).is_some());
                assert!(!new_table.entries[id.as_usize()].is_deleted());
            }

            // No translations happened and all the old nodes are the same
            for i in 2..reachable {
                let id = new_table
                    .ensure_node(
                        VarIdPacked32::new(1000),
                        ids_reachable[i - 2],
                        ids_reachable[i - 1],
                    )
                    .unwrap();
                assert_eq!(id, ids_reachable[i]);
            }
        }

        let mut new_table = match table.collect_garbage(&mut roots) {
            NodeTable::Size32(table) => table,
            _ => panic!("expected 32-bit table"),
        };

        validate_table_after_gc(
            &mut new_table,
            &roots,
            reachable,
            &ids_unreachable,
            &ids_reachable,
        );

        // Check that collecting garbage again does not change anything.
        let mut new_table = match new_table.collect_garbage(&mut roots) {
            NodeTable::Size32(table) => table,
            _ => panic!("expected 32-bit table"),
        };

        validate_table_after_gc(
            &mut new_table,
            &roots,
            reachable,
            &ids_unreachable,
            &ids_reachable,
        );
    }

    #[test]
    fn collect_garbage_rebuild() {
        let mut table = NodeTable32::new();

        // Make a big tree that will be unreachable
        let unreachable = 1000;
        let mut ids_unreachable = vec![];
        for _ in 0..unreachable {
            let id = table
                .ensure_node(
                    VarIdPacked32::new(1000),
                    NodeId32::one(),
                    *ids_unreachable.last().unwrap_or(&NodeId32::zero()),
                )
                .unwrap();
            ids_unreachable.push(id);
        }

        let unreachable_root = ids_unreachable[ids_unreachable.len() - 1];

        // Make a small reachable tree and also construct the expected result.
        let reachable = 10;
        let mut ids_reachable = vec![NodeId32::zero(), NodeId32::one()];
        let mut ids_expected = vec![NodeId16::zero(), NodeId16::one()];
        let mut expected = NodeTable16::default();
        for i in 2..reachable {
            let id = table
                .ensure_node(
                    VarIdPacked32::new(10),
                    ids_reachable[i - 2],
                    ids_reachable[i - 1],
                )
                .unwrap();
            ids_reachable.push(id);

            let id_expected = expected
                .ensure_node(
                    VarIdPacked16::new(10),
                    ids_expected[i - 2],
                    ids_expected[i - 1],
                )
                .unwrap();
            ids_expected.push(id_expected);
        }

        let root = ids_reachable[ids_reachable.len() - 1];

        let (_strongs, mut roots) = make_roots(&[root], &[unreachable_root]);

        let new_table = match table.collect_garbage(&mut roots) {
            NodeTable::Size16(table) => table,
            _ => panic!("expected 16-bit table"),
        };

        assert_eq!(roots.len(), 1);
        assert_eq!(new_table.node_count(), reachable);
        assert_eq!(new_table, expected);

        // Check that collecting garbage again does not change anything.

        let mut new_table = match new_table.collect_garbage(&mut roots) {
            NodeTable::Size16(table) => table,
            _ => panic!("expected 16-bit table"),
        };

        assert_eq!(roots.len(), 1);
        assert_eq!(new_table.node_count(), reachable);
        // The tables should still be equal except for the mark.
        // This might break if we also start checking that the packed information
        // in the variables is the same.
        assert_ne!(new_table.current_mark, expected.current_mark);
        new_table.current_mark = new_table.current_mark.flipped();
        assert_eq!(new_table, expected);
    }

    #[test]
    fn into_reachable_bdd() {
        // Manually create a very broken bdd (not in post-order, with nodes
        // not in the bdd inbetween nodes in the bdd) inside the node table.
        // bdd: v1 or !v2 or v3 or v4
        // the nodes are organized as: 0, 1, v4, vi1, vi2, v1, vi3, v2, v3
        let mut table = NodeTable32::new();

        let v1 = VarIdPacked32::new(1);
        let v2 = VarIdPacked32::new(2);
        let v3 = VarIdPacked32::new(3);
        let v4 = VarIdPacked32::new(4);

        let vi1 = VarIdPacked32::new(10);
        let vi2 = VarIdPacked32::new(11);
        let vi3 = VarIdPacked32::new(12);

        let n: Vec<NodeId32> = (0..9).map(NodeId32::new).collect();

        table.entries.push(BddNode32::new(v4, n[0], n[1]).into());
        table.entries.push(BddNode32::new(vi1, n[0], n[4]).into());
        table.entries.push(BddNode32::new(vi2, n[6], n[0]).into());
        table.entries.push(BddNode32::new(v1, n[7], n[1]).into());
        table.entries.push(BddNode32::new(vi3, n[1], n[0]).into());
        table.entries.push(BddNode32::new(v2, n[1], n[8]).into());
        table.entries.push(BddNode32::new(v3, n[2], n[1]).into());

        let b1 = Bdd32::new_literal(v1, true);
        let b2 = Bdd32::new_literal(v2, false);
        let b3 = Bdd32::new_literal(v3, true);
        let b4 = Bdd32::new_literal(v4, true);

        let expected = b1.or(&b2).unwrap().or(&b3).unwrap().or(&b4).unwrap();

        let result: Bdd32 = unsafe { table.into_reachable_bdd(n[5]) };

        assert_eq!(result.node_count(), expected.node_count());
        assert!(result.iff(&expected).unwrap().is_true());
        assert!(expected.structural_eq(&result));
    }
}
