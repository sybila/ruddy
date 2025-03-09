//! Defines the representation of node tables (used to represent BDDs). Includes: [`NodeTableAny`],
//! [`NodeTable`], [`NodeTable16`], [`NodeTable32`], and [`NodeTable64`].
//!
use std::cmp::{max, min};
use std::fmt;

use crate::conversion::UncheckedInto;
use crate::{
    bdd::BddAny,
    bdd_node::{BddNode16, BddNode32, BddNode64, BddNodeAny},
    node_id::{NodeId16, NodeId32, NodeId64, NodeIdAny},
    variable_id::{VarIdPacked16, VarIdPacked32, VarIdPacked64, VarIdPackedAny},
};

/// The `NodeTableAny` is a data structure that enforces uniqueness of BDD nodes created
/// during the BDD construction process.
pub trait NodeTableAny: Default {
    type Id: NodeIdAny;
    type VarId: VarIdPackedAny;
    type Node: BddNodeAny<Id = Self::Id, VarId = Self::VarId>;

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

    /// Deletes the entry with the given `id` from the node table.
    ///
    /// This method must not be used to delete terminal nodes.
    fn delete(&mut self, id: Self::Id);
}

/// An error that is returned when [`NodeTableAny`] is full and a new node cannot be added.
///
/// It carries the bit-width of the current node ID type for which the error was raised.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct NodeTableFullError {
    width: usize,
}

impl fmt::Display for NodeTableFullError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ensuring node on a full {}-bit node table", self.width)
    }
}

impl std::error::Error for NodeTableFullError {}

/// An element of the [`NodeTable`]. Consists of a [`BddNodeAny`] node, and three node pointers,
/// referencing the `parent` tree that is rooted in this entry, plus two `next_parent` pointers
/// that define the parent tree which contains the entry itself.
#[derive(Debug, Clone, PartialEq, Eq)]
struct NodeEntry<
    TNodeId: NodeIdAny,
    TVarId: VarIdPackedAny,
    TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
> {
    node: TNode,
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

/// A mutable view into a deleted entry in a [`NodeTable`].
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
/// The `NodeTable` also supports deletion of nodes. When a node is deleted,
/// it is not actually removed from the table, but only marked as deleted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeTable<
    TNodeId: NodeIdAny,
    TVarId: VarIdPackedAny,
    TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
> {
    entries: Vec<NodeEntry<TNodeId, TVarId, TNode>>,
    first_free: TNodeId,
    deleted: usize,
}

impl<TNodeId, TVarId, TNode> NodeTable<TNodeId, TVarId, TNode>
where
    TNodeId: NodeIdAny,
    TVarId: VarIdPackedAny,
    TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
{
    /// Make a new [`NodeTable`] containing nodes `0` and `1`.
    pub fn new() -> Self {
        Self {
            entries: vec![NodeEntry::zero(), NodeEntry::one()],
            first_free: TNodeId::undefined(),
            deleted: 0,
        }
    }

    /// Make a new [`NodeTable`] containing nodes `0` and `1`, but make sure the underlying
    /// vector has at least the specified amount of `capacity`.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut entries = Vec::with_capacity(capacity);
        entries.push(NodeEntry::zero());
        entries.push(NodeEntry::one());
        Self {
            entries,
            first_free: TNodeId::undefined(),
            deleted: 0,
        }
    }

    /// Returns the number of entries in the node table,
    /// including the entries for the terminal nodes.
    pub fn node_count(&self) -> usize {
        self.entries.len() - self.deleted
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
        self.get_entry_unchecked_mut(low)
            .node
            .increment_parent_counter();

        self.get_entry_unchecked_mut(high)
            .node
            .increment_parent_counter();

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
    fn get_entry(&self, id: TNodeId) -> Option<&NodeEntry<TNodeId, TVarId, TNode>> {
        self.entries
            .get(id.as_usize())
            .filter(|entry| !entry.is_deleted())
    }

    /// Get a checked mutable reference to the entry with the given `id`, or `None` if the entry does not exist.
    fn get_entry_mut(&mut self, id: TNodeId) -> Option<&mut NodeEntry<TNodeId, TVarId, TNode>> {
        self.entries
            .get_mut(id.as_usize())
            .filter(|entry| !entry.is_deleted())
    }

    /// An unchecked variant of [`NodeTable::get_entry`].
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

    /// An unchecked variant of [`NodeTable::get_entry_mut`].
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
}

impl<
        TNodeId: NodeIdAny,
        TVarId: VarIdPackedAny,
        TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
    > Default for NodeTable<TNodeId, TVarId, TNode>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
        TNodeId: NodeIdAny,
        TVarId: VarIdPackedAny,
        TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
    > NodeTableAny for NodeTable<TNodeId, TVarId, TNode>
{
    type Id = TNodeId;
    type VarId = TVarId;
    type Node = TNode;

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
    /// the node table's search trees (such as in [`NodeTable::ensure_node`]). This is
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
}

pub type NodeTable16 = NodeTable<NodeId16, VarIdPacked16, BddNode16>;
pub type NodeTable32 = NodeTable<NodeId32, VarIdPacked32, BddNode32>;
pub type NodeTable64 = NodeTable<NodeId64, VarIdPacked64, BddNode64>;

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
                }
            }
        }
    };
}

impl_from_node_table!(NodeTable16, NodeTable32);
impl_from_node_table!(NodeTable16, NodeTable64);
impl_from_node_table!(NodeTable32, NodeTable64);

#[cfg(test)]
mod tests {
    use crate::conversion::UncheckedInto;
    use crate::node_id::{NodeId16, NodeId32, NodeIdAny};
    use crate::node_table::{hash_node_data, NodeTable16, NodeTable32, NodeTable64, NodeTableAny};
    use crate::variable_id::{VarIdPacked16, VarIdPacked32};

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
            let j = NodeId16::new(i - 1);
            let k = NodeId16::new(i - 2);
            assert!(table.ensure_node(v, j, k).is_ok());
        }

        let err = table
            .ensure_node(v, NodeId16::zero(), NodeId16::one())
            .unwrap_err();
        println!("{}", err);
        assert_eq!(err.width, 16);
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
                table.get_entry(id).unwrap().node,
                table_delete.get_entry(id).unwrap().node
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
}
