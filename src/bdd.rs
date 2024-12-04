use std::ops::{Index, IndexMut};

use rand::{seq::SliceRandom, Rng};

use crate::{
    bdd_node::{BddNode, BddNode32},
    node_id::{BddNodeId, NodeId32},
    usize_is_at_least_32_bits,
};

/// A trait implemented by types that can serve as BDDs.
pub trait Bdd {
    type Node: BddNode;

    /// Create a new BDD representing the constant boolean function `true`.
    fn new_true() -> Self;
    /// Create a new BDD representing the constant boolean function `false`.
    fn new_false() -> Self;

    fn is_true(&self) -> bool;
    fn is_false(&self) -> bool;
}

#[derive(Clone)]
pub struct Bdd32 {
    root: NodeId32,
    nodes: Vec<BddNode32>,
}

impl Bdd32 {
    // TODO: Make this method pub(crate) once we have serialization. It is
    // used in ruddy-benchmarks, which currently handles serialization.
    pub fn new(root: NodeId32, nodes: Vec<BddNode32>) -> Self {
        Bdd32 { root, nodes }
    }

    pub fn root(&self) -> NodeId32 {
        self.root
    }

    /// Returns the number of nodes in the BDD, including the terminal nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the BDD has no nodes other than the terminal nodes and
    /// `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.len() < 2
    }

    pub fn into_nodes(self) -> Vec<BddNode32> {
        self.nodes
    }

    /// # Safety
    ///
    /// Calling this method with an `id` that is not in the bdd is undefined behavior.
    pub(crate) unsafe fn get_node_unchecked(&self, id: NodeId32) -> &BddNode32 {
        unsafe { self.nodes.get_unchecked(id.as_usize()) }
    }

    pub fn len_u32(&self) -> u32 {
        self.nodes
            .len()
            .try_into()
            .expect("the BDD contains less than 2**32 nodes")
    }

    /// Reorder the BDD based on the given "shuffle" vector.
    ///
    /// The shuffle vector cannot relocate the `0` and `1` nodes and it must be a bijection
    /// (there are no collisions).
    fn permutation(&self, old_id_to_new_map: &[NodeId32]) -> Self {
        if self.is_empty() {
            return self.clone();
        }

        debug_assert_eq!(old_id_to_new_map[0], NodeId32::zero());
        debug_assert_eq!(old_id_to_new_map[1], NodeId32::one());

        let mut new_nodes = vec![BddNode32::zero(); self.nodes.len()];
        new_nodes[1] = BddNode32::one();

        for id in (2..self.len_u32()).map(NodeId32::new) {
            let current_node = &self[id];
            let new_node = current_node.permute(old_id_to_new_map);
            // Check that we are not rewriting an existing new node.
            debug_assert!(new_nodes[old_id_to_new_map[id.as_usize()].as_usize()].is_zero());
            new_nodes[old_id_to_new_map[id.as_usize()].as_usize()] = new_node;
        }
        // Since we migrated all old nodes and we haven't overwritten anything, the permutation
        // must have been a proper bijection.

        Bdd32 {
            root: old_id_to_new_map[self.root().as_usize()],
            nodes: new_nodes,
        }
    }

    /// Sort the nodes in this BDD with respect to DFS pre-order.
    ///
    /// This assumes all nodes in the BDD are used.
    ///
    /// The DFS always explores "low" nodes first, then "high" nodes.
    ///
    /// Note that both preorder and postorder satisfy that for node with a specific `id`,
    /// `low < id` and `high < id`. However, it is not strictly required to maintain this
    /// invariant in order "node sorts".
    pub fn sort_preorder(&self) -> Self {
        if self.is_empty() {
            return self.clone();
        }

        let mut new_ids = vec![NodeId32::undefined(); self.nodes.len()];
        new_ids[0] = NodeId32::zero();
        new_ids[1] = NodeId32::one();

        let mut stack = vec![self.root()];

        // First, compute the "translation map" by traversing the BDD.
        let mut next_id = self.len_u32() - 1;
        while let Some(top) = stack.pop() {
            let node_id = &mut new_ids[top.as_usize()];
            if node_id.is_undefined() {
                *node_id = NodeId32::new(next_id);
                next_id -= 1;
                let node = &self[top];
                stack.push(node.high());
                stack.push(node.low());
            }
        }

        debug_assert!(NodeId32::new(next_id).is_one());
        self.permutation(&new_ids)
    }

    /// Same as `sort_preorder` but with postorder.
    pub fn sort_postorder(&self) -> Self {
        if self.is_empty() {
            return self.clone();
        }

        let mut new_ids = vec![NodeId32::undefined(); self.nodes.len()];
        new_ids[0] = NodeId32::zero();
        new_ids[1] = NodeId32::one();

        let mut stack = vec![self.root()];

        // First, compute the "translation map" by traversing the BDD.
        let mut next_id = 2u32;
        while let Some(&top) = stack.last() {
            if new_ids[top.as_usize()].is_undefined() {
                let node: &BddNode32 = &self[top];
                let new_high = new_ids[node.high().as_usize()];
                let new_low = new_ids[node.low().as_usize()];

                let high_is_undefined = new_high.is_undefined();
                let low_is_undefined = new_low.is_undefined();

                if !high_is_undefined && !low_is_undefined {
                    new_ids[top.as_usize()] = NodeId32::new(next_id);
                    next_id += 1;
                    stack.pop();
                }
                if high_is_undefined {
                    stack.push(node.high());
                }
                if low_is_undefined {
                    stack.push(node.low());
                }
            } else {
                stack.pop();
            }
        }

        debug_assert_eq!(usize_is_at_least_32_bits(next_id), self.nodes.len());
        self.permutation(&new_ids)
    }

    pub fn shuffle<R: Rng>(&self, rng: &mut R) -> Self {
        let mut indices = (2..(self.len_u32())).map(NodeId32::new).collect::<Vec<_>>();
        indices.shuffle(rng);
        indices.push(NodeId32::one());
        indices.push(NodeId32::zero());
        indices.reverse();
        self.permutation(&indices)
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

    fn is_true(&self) -> bool {
        self.root == NodeId32::one()
    }

    fn is_false(&self) -> bool {
        self.root == NodeId32::zero()
    }
}

impl Index<NodeId32> for Bdd32 {
    type Output = BddNode32;

    fn index(&self, index: NodeId32) -> &Self::Output {
        &self.nodes[index.as_usize()]
    }
}

impl IndexMut<NodeId32> for Bdd32 {
    fn index_mut(&mut self, index: NodeId32) -> &mut Self::Output {
        &mut self.nodes[index.as_usize()]
    }
}
