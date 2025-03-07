//! Defines the representation of node tables (used to represent BDDs). Includes: [`NodeTableAny`],
//! [`NodeTable`], [`NodeTable16`], [`NodeTable32`], and [`NodeTable64`].
//!
use std::cmp::{max, min};
use std::fmt;

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeTable<
    TNodeId: NodeIdAny,
    TVarId: VarIdPackedAny,
    TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
> {
    entries: Vec<NodeEntry<TNodeId, TVarId, TNode>>,
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
        }
    }

    /// Make a new [`NodeTable`] containing nodes `0` and `1`, but make sure the underlying
    /// vector has at least the specified amount of `capacity`.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut entries = Vec::with_capacity(capacity);
        entries.push(NodeEntry::zero());
        entries.push(NodeEntry::one());
        Self { entries }
    }

    /// Returns the number of entries in the node table,
    /// including the entries for the terminal nodes.
    pub fn node_count(&self) -> usize {
        self.entries.len()
    }

    /// Create a new [`NodeEntry`] in this table (without checking for uniqueness), and increment
    /// the parent counters of its child nodes.
    ///
    /// ## Safety
    ///
    /// The function requires that `low` and `high` IDs point to existing entries in this node
    /// table. This requirement is not checked in release mode and if broken results
    /// in undefined behavior.
    unsafe fn push_node(&mut self, variable: TVarId, low: TNodeId, high: TNodeId) {
        debug_assert!(low.as_usize() < self.node_count());
        debug_assert!(high.as_usize() < self.node_count());

        let low_entry = self.entries.get_unchecked_mut(low.as_usize());
        low_entry.node.increment_parent_counter();

        let high_entry = self.entries.get_unchecked_mut(high.as_usize());
        high_entry.node.increment_parent_counter();

        self.entries.push(TNode::new(variable, low, high).into());
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
        if low == high {
            // We don't need to create a new node in this case.
            return Ok(low);
        }

        // Check that the node table is not full. We also save the ID of the new
        // node for later use.
        let new_node = TNodeId::try_from(self.entries.len()).map_err(|_| NodeTableFullError {
            width: std::mem::size_of::<TNodeId>() * 8,
        })?;

        debug_assert!((low.as_usize()) < self.entries.len());
        debug_assert!((high.as_usize()) < self.entries.len());

        // Max child is used to select the tree of nodes into which we are inserting.
        // The reason why `max_child` is used instead of `min_child` is that it avoids terminal
        // nodes which typically have many parents.
        let max_child = max(low, high);

        // A reasonably random hash of the remaining node data.
        // https://stackoverflow.com/a/27952689
        // TODO: is this a good hash?
        let min_child = min(low, high);
        let min_child_u64: u64 = min_child.unchecked_into();
        let hash_child: u64 = min_child_u64.wrapping_mul(14695981039346656039);
        let hash_variable: u64 = variable.unpack_u64().wrapping_mul(14695981039346656039);
        let mut hash = hash_child.wrapping_mul(3).wrapping_add(hash_variable);

        let root_node = unsafe { self.entries.get_unchecked_mut(max_child.as_usize()) };
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
            debug_assert!((current_node.as_usize()) < self.entries.len());

            let current_entry = unsafe { self.entries.get_unchecked_mut(current_node.as_usize()) };

            if current_entry.node.variable() == variable
                && current_entry.node.low() == low
                && current_entry.node.high() == high
            {
                // The node exists and is stored inside `current_node`.
                return Ok(current_node);
            }

            if hash & (1 << (u64::BITS - 1)) != 0 {
                // Top bit is "one"...
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
                Self { entries }
            }
        }
    };
}

impl_from_node_table!(NodeTable16, NodeTable32);
impl_from_node_table!(NodeTable16, NodeTable64);
impl_from_node_table!(NodeTable32, NodeTable64);

#[cfg(test)]
mod tests {
    use crate::node_id::{NodeId16, NodeId32, NodeIdAny};
    use crate::node_table::{NodeTable16, NodeTable32, NodeTable64, NodeTableAny};
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
}
