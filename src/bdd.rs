use crate::{
    bdd_node::{BddNode, BddNode32},
    node_id::{BddNodeId, NodeId32},
};

/// A trait implemented by types that can serve as BDDs.
pub trait Bdd {
    type Node: BddNode;

    /// Create a new BDD representing the constant boolean function `true`.
    fn new_true() -> Self;
    /// Create a new BDD representing the constant boolean function `false`.
    fn new_false() -> Self;

    fn push(&mut self, node: Self::Node);
}

pub struct Bdd32 {
    root: NodeId32,
    nodes: Vec<BddNode32>,
}

impl Bdd32 {
    pub(crate) fn new(root: NodeId32, nodes: Vec<BddNode32>) -> Self {
        Bdd32 { root, nodes }
    }

    pub fn root(&self) -> NodeId32 {
        self.root
    }

    /// # Safety
    ///
    /// Calling this method with an `id` that is not in the bdd is undefined behavior.
    pub(crate) unsafe fn get_node_unchecked(&self, id: NodeId32) -> &BddNode32 {
        unsafe { self.nodes.get_unchecked(id.as_usize()) }
    }
}

impl Bdd for Bdd32 {
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

    fn push(&mut self, node: BddNode32) {
        self.nodes.push(node);
    }
}
