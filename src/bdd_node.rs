use crate::{
    node_id::{BddNodeId, NodeId32},
    variable_id::{VarIdPacked32, VariableId},
};

/// An internal trait implemented by types that can serve as BDD nodes.
pub trait BddNode {
    type Id: BddNodeId;
    type VarId: VariableId;

    /// Return a terminal `0` node.
    fn zero() -> Self;
    /// Return a terminal `1` node.
    fn one() -> Self;

    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn is_terminal(&self) -> bool;

    fn low(&self) -> Self::Id;
    fn high(&self) -> Self::Id;
    fn variable(&self) -> Self::VarId;
}

#[derive(Clone, Hash)]
pub struct BddNode32 {
    variable: VarIdPacked32,
    low: NodeId32,
    high: NodeId32,
}

impl BddNode for BddNode32 {
    type Id = NodeId32;
    type VarId = VarIdPacked32;

    fn zero() -> Self {
        BddNode32 {
            variable: VarIdPacked32::undefined(),
            low: NodeId32::zero(),
            high: NodeId32::zero(),
        }
    }

    fn one() -> Self {
        BddNode32 {
            variable: VarIdPacked32::undefined(),
            low: NodeId32::one(),
            high: NodeId32::one(),
        }
    }

    fn is_zero(&self) -> bool {
        self.low.is_zero() && self.high.is_zero()
    }

    fn is_one(&self) -> bool {
        self.low.is_one() && self.high.is_one()
    }

    fn is_terminal(&self) -> bool {
        self.low.is_terminal() && self.high.is_terminal()
    }

    fn low(&self) -> NodeId32 {
        self.low
    }

    fn high(&self) -> NodeId32 {
        self.high
    }

    fn variable(&self) -> VarIdPacked32 {
        self.variable
    }
}

impl BddNode32 {
    pub fn new(variable: VarIdPacked32, low: NodeId32, high: NodeId32) -> Self {
        BddNode32 {
            variable,
            low,
            high,
        }
    }

    pub fn increment_parents(&mut self) {
        self.variable.increment_parents();
    }

    pub fn has_many_parents(&self) -> bool {
        self.variable.has_many_parents()
    }

    pub(crate) fn permute(&self, permutation: &[NodeId32]) -> Self {
        BddNode32 {
            variable: self.variable,
            low: permutation[self.low.as_usize()],
            high: permutation[self.high.as_usize()],
        }
    }
}
