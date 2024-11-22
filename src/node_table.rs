use std::cmp::{max, min};

use crate::{
    bdd_node::{BddNode, BddNode32},
    node_id::{BddNodeId, NodeId32},
    variable_id::{VarIdPacked32, VariableId},
};

/// The `NodeTable` is a data structure that enforces uniqueness of BDD nodes created
/// during the BDD construction process.
pub trait NodeTable {
    type Id: BddNodeId;
    type VarId: VariableId;

    /// Searches the `NodeTable` for a node with the id `x`, such that the node's variable is `variable`,
    /// and the node's low and high children are `low` and `high`, respectively. If such a node is not found,
    /// a new node is created and added to the `NodeTable`.
    fn ensure_node(&mut self, variable: Self::VarId, low: Self::Id, high: Self::Id) -> Self::Id;
}

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
        NodeEntry32 {
            node: BddNode32::zero(),
            parent: NodeId32::undefined(),
            next_parent_zero: NodeId32::undefined(),
            next_parent_one: NodeId32::undefined(),
        }
    }

    fn one() -> Self {
        NodeEntry32 {
            node: BddNode32::one(),
            parent: NodeId32::undefined(),
            next_parent_zero: NodeId32::undefined(),
            next_parent_one: NodeId32::undefined(),
        }
    }
}

pub struct NodeTable32 {
    entries: Vec<NodeEntry32>,
    // Remember if the underlying bdd is not false. This is here, so we can always
    // initialize the node table with both terminal nodes and return a false bdd
    // if it turns out during the computation that the bdd is false.
    bdd_is_false: bool,
}

impl NodeTable32 {
    pub fn new(capacity: usize) -> NodeTable32 {
        let mut entries = Vec::with_capacity(capacity);
        entries.push(NodeEntry32::zero());
        entries.push(NodeEntry32::one());
        NodeTable32 {
            entries,
            bdd_is_false: true,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    unsafe fn push_node(&mut self, variable: VarIdPacked32, low: NodeId32, high: NodeId32) {
        let low_entry = self.entries.get_unchecked_mut(low.as_usize());
        low_entry.node.variable().increment_parents();

        let high_entry = self.entries.get_unchecked_mut(high.as_usize());
        high_entry.node.variable().increment_parents();

        self.entries
            .push(BddNode32::new(variable, low, high).into());
    }
}

impl NodeTable for NodeTable32 {
    type Id = NodeId32;
    type VarId = VarIdPacked32;

    fn ensure_node(&mut self, variable: VarIdPacked32, low: NodeId32, high: NodeId32) -> NodeId32 {
        //
        self.bdd_is_false = !(low.is_one() || high.is_one());

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
        let max_child = max(low, high);

        // A reasonably random hash of the remaining node data. Note that we want to first
        // use the high bits of this hash, since these are the "best" bits of a multiplicative
        // hash. Furthermore, since we are actually extending the width from 32 to 64 bits, this
        // should be a "perfect" hash in the sense that each input hashes to a unique output.
        let min_child = min(low, high);
        let mut hash = ((min_child.as_u64() << 32) | u64::from(variable.unpack()))
            .overflowing_mul(14695981039346656039)
            .0;

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
