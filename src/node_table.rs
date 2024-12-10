//! Defines the representation of node tables (used to represent BDDs). Includes: [`NodeTable`],
//! [`NodeEntry32`] and [`NodeTable32`].
//!
use std::cmp::{max, min};

use crate::{
    bdd::{Bdd, Bdd32},
    bdd_node::{BddNode, BddNode32},
    node_id::{BddNodeId, NodeId32},
    variable_id::{VarIdPacked32, VariableId},
};

/// The `NodeTable` is a data structure that enforces uniqueness of BDD nodes created
/// during the BDD construction process.
pub trait NodeTable {
    type Id: BddNodeId;
    type VarId: VariableId;

    /// Searches the `NodeTable` for a node matching node `(var, (low, high))`, and returns its
    /// identifier (i.e. the node's variable is `variable`, and the node's low and high children
    /// are `low` and `high`, respectively). If such a node is not found, a new node is created
    /// and added to the `NodeTable`.
    ///
    /// This method should not be used to "create" terminal nodes, i.e. it must hold that
    /// `variable != VarId::undefined`.
    fn ensure_node(&mut self, variable: Self::VarId, low: Self::Id, high: Self::Id) -> Self::Id;
}

/// An element of the [`NodeTable32`]. Consists of a [`BddNode32`] node, and three node pointers,
/// referencing the `parent` tree that is rooted in this entry, plus two `next_parent` pointers
/// that define the parent tree which contains the entry itself.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeEntry32 {
    node: BddNode32,
    parent: NodeId32,
    next_parent_zero: NodeId32,
    next_parent_one: NodeId32,
}

impl From<BddNode32> for NodeEntry32 {
    fn from(node: BddNode32) -> Self {
        NodeEntry32 {
            node,
            parent: NodeId32::undefined(),
            next_parent_zero: NodeId32::undefined(),
            next_parent_one: NodeId32::undefined(),
        }
    }
}

impl NodeEntry32 {
    fn zero() -> Self {
        Self::from(BddNode32::zero())
    }

    fn one() -> Self {
        Self::from(BddNode32::one())
    }
}

/// An implementation of [`NodeTable`] backed by [`BddNode32`] (or rather [`NodeEntry32`]).
///
/// Instead of "normal" hashing, it uses a "tree of parents" scheme, where each node is stored
/// in its maximal child. This is effective because most nodes in a BDD only have one or two
/// parents, so we know the tree will stay shallow and will be searched in `O(1)` time. Only a
/// few nodes have many parents, resulting in average `O(log)` search time (this part does
/// in fact depend on hash collisions).
///
/// (Note that we could obtain a tight `O(log)` bound by using a search tree instead of
/// a hash-prefix tree, but this would require balancing, which adds a lot of overhead)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeTable32 {
    entries: Vec<NodeEntry32>,
}

impl NodeTable32 {
    /// Make a new [`NodeTable32`] containing nodes `0` and `1`.
    pub fn new() -> NodeTable32 {
        NodeTable32 {
            entries: vec![NodeEntry32::zero(), NodeEntry32::one()],
        }
    }

    /// Make a new [`NodeTable32`] containing nodes `0` and `1`, but make sure the underlying
    /// vector has at least the specified amount of `capacity`.
    pub fn with_capacity(capacity: usize) -> NodeTable32 {
        let mut entries = Vec::with_capacity(capacity);
        entries.push(NodeEntry32::zero());
        entries.push(NodeEntry32::one());
        NodeTable32 { entries }
    }

    /// Returns the number of entries in the node table, including the entries for the terminal nodes.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the node table has no entries other than entries for the
    /// terminal nodes and `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.len() <= 2
    }

    /// Create a new [`BddNode32`] in this table (without checking for uniqueness), and increment
    /// the parent counters of its child nodes.
    ///
    /// ## Safety
    ///
    /// The function requires that `low` and `high` point to existing entries in this node table.
    /// This requirement is not checked and if broken results in undefined behavior.
    unsafe fn push_node(&mut self, variable: VarIdPacked32, low: NodeId32, high: NodeId32) {
        let low_entry = self.entries.get_unchecked_mut(low.as_usize());
        low_entry.node.increment_parents();

        let high_entry = self.entries.get_unchecked_mut(high.as_usize());
        high_entry.node.increment_parents();

        self.entries
            .push(BddNode32::new(variable.reset(), low, high).into());
    }
}

impl Default for NodeTable32 {
    fn default() -> Self {
        NodeTable32::new()
    }
}

impl Bdd32 {
    /// Create a new [`Bdd32`] from a [`NodeTable32`] rooted in `root`. The conversion preserves
    /// *all* nodes that are present in the given `table`, not just the ones reachable from the
    /// root node.
    ///
    /// ## Safety
    ///
    /// Similar to [Bdd32::new_unchecked], this function is unsafe, because it can be used to
    /// create an invariant-breaking BDD. While [`NodeTable32`] cannot be used (under normal
    /// conditions) to create BDDs with cycles, it can definitely be used to create BDDs with
    /// broken variable ordering.
    pub(crate) unsafe fn from_table(root: NodeId32, table: NodeTable32) -> Bdd32 {
        // Zero and one have special cases to always ensure that they are structurally equivalent
        // to the result of [Bdd32::new_false]/[Bdd32::new_true], regardless of what's in the
        // provided node table.
        if root.is_zero() {
            return Bdd32::new_false();
        }
        if root.is_one() {
            return Bdd32::new_true();
        }
        let nodes = table.entries.into_iter().map(|entry| entry.node).collect();
        unsafe { Bdd32::new_unchecked(root, nodes) }
    }
}

impl NodeTable for NodeTable32 {
    type Id = NodeId32;
    type VarId = VarIdPacked32;

    fn ensure_node(&mut self, variable: VarIdPacked32, low: NodeId32, high: NodeId32) -> NodeId32 {
        if low == high {
            // We don't need to create a new node in this case.
            return low;
        }

        // We just save this for later use (because later we might hold
        // a mutable reference to self.entries).
        // Also check that the node table is not full. This will probably have a
        // different behavior once pointer compression is implemented.
        let new_node = NodeId32::new(
            self.entries
                .len()
                .try_into()
                .expect("32-bit node table does not exceed 2**32 nodes (32-bit overflow)"),
        );

        debug_assert!((low.as_usize()) < self.entries.len());
        debug_assert!((high.as_usize()) < self.entries.len());

        // Max child is used to select the tree of nodes into which we are inserting.
        // The reason why `max_child` is used instead of `min_child` is that it avoids terminal
        // nodes which typically have many parents.
        let max_child = max(low, high);

        // A reasonably random hash of the remaining node data. Note that we want to first
        // use the high bits of this hash, since these are the "best" bits of a multiplicative
        // hash. Furthermore, since we are actually extending the width from 32 to 64 bits, this
        // should be a "perfect" hash in the sense that each input hashes to a unique output.
        let min_child = min(low, high);
        let (mut hash, _) = ((min_child.as_u64() << 32) | u64::from(variable.unpack()))
            .overflowing_mul(14695981039346656039);

        let root_node = unsafe { self.entries.get_unchecked_mut(max_child.as_usize()) };
        let mut current_node = root_node.parent;

        if current_node.is_undefined() {
            // This is the first parent for the root node.
            root_node.parent = new_node;
            unsafe {
                self.push_node(variable, low, high);
            }
            return new_node;
        }

        // There is a larger tree that we need to explore.
        loop {
            debug_assert!((current_node.as_usize()) < self.entries.len());

            let current_entry = unsafe { self.entries.get_unchecked_mut(current_node.as_usize()) };

            if current_entry.node.variable() == variable
                && current_entry.node.low() == low
                && current_entry.node.high() == high
            {
                // The node exists and is stored inside `current_node`.
                return current_node;
            }

            if hash & (1 << (u64::BITS - 1)) != 0 {
                // Top bit is "one"...
                if current_entry.next_parent_one.is_undefined() {
                    // Next "one" slot is empty. We can save the node there.
                    current_entry.next_parent_one = new_node;
                    unsafe {
                        self.push_node(variable, low, high);
                    }
                    return new_node;
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
                    return new_node;
                } else {
                    current_node = current_entry.next_parent_zero;
                }
            }

            // Rotate hash so that the next iteration will see the next top-most bit.
            hash = hash.rotate_left(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::node_id::{BddNodeId, NodeId32};
    use crate::node_table::{NodeTable, NodeTable32};
    use crate::variable_id::VarIdPacked32;

    #[test]
    pub fn node_table_32_basic() {
        let mut table = NodeTable32::new();
        assert!(table.is_empty());
        assert_eq!(2, table.len());
        assert_eq!(table, NodeTable32::with_capacity(1024));

        // Create some nodes
        let id1 = table.ensure_node(VarIdPacked32::new(4), NodeId32::zero(), NodeId32::one());
        let id1_p = table.ensure_node(VarIdPacked32::new(4), NodeId32::zero(), NodeId32::one());
        assert_eq!(id1, id1_p);
        let id2 = table.ensure_node(VarIdPacked32::new(3), id1, NodeId32::one());
        let id2_p = table.ensure_node(VarIdPacked32::new(3), id1, NodeId32::one());
        assert_eq!(id2, id2_p);

        assert_eq!(id1, table.ensure_node(VarIdPacked32::new(3), id1, id1));

        // Make a bunch of nodes that should all "collide" in the parent tree of the `1` node.
        let ids = (0..1000u32)
            .map(|i| table.ensure_node(VarIdPacked32::new(i), NodeId32::zero(), NodeId32::one()))
            .collect::<Vec<_>>();

        for (i, id) in ids.iter().enumerate() {
            let id_p = table.ensure_node(
                VarIdPacked32::new(i as u32),
                NodeId32::zero(),
                NodeId32::one(),
            );
            assert_eq!(*id, id_p);
        }

        assert_eq!(1003, table.len());
    }
}
