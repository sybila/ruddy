//! Defines the representation of (standalone) binary decision diagrams. Includes: [`Bdd`]
//! and [`Bdd32`].

use crate::variable_id::{VarIdPacked32, VariableId};
use crate::{
    bdd_node::{BddNode, BddNode32},
    node_id::{BddNodeId, NodeId32},
};

/// A trait implemented by types that can serve as *standalone* BDDs.
///
/// Internally, a standalone BDD is a collection of [`BddNode`] instances, such that one node is
/// designated as the *root* of the BDD. Aside from the root, a non-trivial BDD must also contain
/// a single [BddNode::zero] node identified by [BddNodeId::zero] and a single [BddNode::one] node
/// identified by [BddNodeId::one]. Finally, the graph induced by the [BddNode::low] and
/// [BddNode::high] edges must not contain any cycles (except for the self-loops on the
/// two terminal nodes).
///
/// Note that it is not required for the nodes to (1) be sorted in any specific way; (2) be
/// reachable from the root node; (3) adhere to any specific variable ordering or structural
/// properties other than those outlined above. In other words, an arbitrary instance of [`Bdd`]
/// is not required to be an OBDD (ordered BDD) or ROBDD (reduced and ordered BDD). However,
/// these properties are typically enforced in practice by the implementations of this trait.
///
/// Finally, note that standalone BDDs are typically assumed to be immutable. There are certain
/// situations where mutability makes sense (for example to allow chaining of multiple smaller
/// structural changes without unnecessary copying), but this is mostly an exception to the rule.
/// In particular, no method in the actual trait allows mutability.
pub trait Bdd: Clone {
    /// The type of node ID used by [Bdd::Node].
    type Id: BddNodeId;
    /// The type of variable ID used by [Bdd::Node].
    type VarId: VariableId;
    /// The type of node used in this [`Bdd`].
    type Node: BddNode<Id = Self::Id, VarId = Self::VarId>;

    /// Create a new BDD representing the constant boolean function `true`.
    fn new_true() -> Self;
    /// Create a new BDD representing the constant boolean function `false`.
    fn new_false() -> Self;
    /// Create a new BDD representing the boolean function `var=value`.
    fn new_literal(var: Self::VarId, value: bool) -> Self;

    /// ID of the BDD root node.
    fn root(&self) -> Self::Id;
    /// Get a (checked) reference to a node, or `None` if such node does not exist.
    fn get(&self, id: Self::Id) -> Option<&Self::Node>;
}

/// An implementation of [`Bdd`] using [`BddNode32`]. In addition to the requirements of the
/// [`Bdd`] trait, this struct also expects the BDD to be ordered and reduced.
#[derive(Clone)]
pub struct Bdd32 {
    root: NodeId32,
    nodes: Vec<BddNode32>,
}

impl Bdd32 {
    /// Create a new instance of [`Bdd32`] using a raw list of [`BddNode32`] items and a single
    /// [`NodeId32`] root.
    ///
    /// ## Safety
    ///
    /// This function is unsafe because it can be used to create a BDD object that does not
    /// respect the prescribed invariants of [`Bdd32`]. For example, it can be used to create
    /// a BDD that is not acyclic or a BDD that is not reduced.
    pub unsafe fn new_unchecked(root: NodeId32, nodes: Vec<BddNode32>) -> Self {
        Bdd32 { root, nodes }
    }

    /// Returns the number of nodes in the BDD, including the terminal nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the BDD has no nodes other than the terminal nodes and
    /// `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.len() <= 2
    }

    /// Convert the BDD into a raw list of nodes.
    pub fn into_nodes(self) -> Vec<BddNode32> {
        self.nodes
    }

    /// An unchecked variant of [Bdd::get].
    ///
    /// # Safety
    ///
    /// Calling this method with an `id` that is not in the bdd is undefined behavior.
    pub(crate) unsafe fn get_node_unchecked(&self, id: NodeId32) -> &BddNode32 {
        unsafe { self.nodes.get_unchecked(id.as_usize()) }
    }

    /// Compares two [`Bdd32`] instances structurally, i.e. by comparing their roots and the
    /// underlying lists of nodes.
    ///
    /// Note that this does not guarantee that the two BDDs represent the same boolean function,
    /// unless their nodes are also ordered the same way (which they are, assuming the BDDs were
    /// created using the same `apply` algorithm).
    pub fn structural_eq(&self, other: &Self) -> bool {
        self.root == other.root && self.nodes == other.nodes
    }
}

impl Bdd for Bdd32 {
    type Id = NodeId32;
    type VarId = VarIdPacked32;
    type Node = BddNode32;

    fn new_true() -> Self {
        Bdd32 {
            root: NodeId32::one(),
            nodes: vec![BddNode32::zero(), BddNode32::one()],
        }
    }

    fn new_false() -> Self {
        Bdd32 {
            root: NodeId32::zero(),
            nodes: vec![BddNode32::zero()],
        }
    }

    fn new_literal(var: Self::VarId, value: bool) -> Self {
        let decision_node = if value {
            BddNode32::new(var, NodeId32::zero(), NodeId32::one())
        } else {
            BddNode32::new(var, NodeId32::one(), NodeId32::zero())
        };
        Bdd32 {
            root: NodeId32::new(2),
            nodes: vec![BddNode32::zero(), BddNode32::one(), decision_node],
        }
    }

    fn root(&self) -> Self::Id {
        self.root
    }
    fn get(&self, id: Self::Id) -> Option<&Self::Node> {
        self.nodes.get(id.as_usize())
    }
}

#[cfg(test)]
mod tests {
    use crate::bdd::{Bdd, Bdd32};
    use crate::bdd_node::BddNode;
    use crate::node_id::{BddNodeId, NodeId32};
    use crate::variable_id::VarIdPacked32;

    #[test]
    pub fn bdd_32_invariants() {
        assert!(Bdd32::new_true().root().is_one());
        assert!(Bdd32::new_true().is_empty());
        assert!(Bdd32::new_false().root().is_zero());
        assert!(Bdd32::new_false().is_empty());

        let v = VarIdPacked32::new(1);
        let x = Bdd32::new_literal(v, true);
        assert!(!x.is_empty());
        assert!(!x.root().is_terminal());
        assert!(x.get(NodeId32::new(3)).is_none());
        assert_eq!(v, x.get(x.root()).unwrap().variable());
        assert_eq!(x.len(), 3);
        unsafe {
            assert_eq!(v, x.get_node_unchecked(x.root()).variable());
            let x_p = Bdd32::new_unchecked(x.root(), x.clone().into_nodes());
            assert!(x.structural_eq(&x_p));
        }
    }
}
