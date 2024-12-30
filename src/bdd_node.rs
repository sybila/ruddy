//! Defines the representation of BDD nodes. Includes: [`BddNode`] and [`BddNode32`].

use crate::{
    node_id::{BddNodeId, NodeId32},
    variable_id::{VarIdPacked32, VariableId},
};

/// An internal trait implemented by types that can serve as BDD nodes. Each BDD node is either
/// a *terminal node* (`0` or `1`) or a *decision node*. A decision node consists of the decision
/// variable (of type [`VariableId`]) and two child references, *low* and *high*
/// (of type [`BddNodeId`]).
pub trait BddNode: Clone + Eq {
    /// Variable ID type used by this [`BddNode`].
    type Id: BddNodeId;
    /// Node ID type used by this [`BddNode`].
    type VarId: VariableId;

    /// Return an instance of the terminal `0` node.
    fn zero() -> Self;
    /// Return an instance of the terminal `1` node.
    fn one() -> Self;

    /// Checks if this node is [`BddNode::zero`].
    fn is_zero(&self) -> bool;
    /// Checks if this node is [`BddNode::one`].
    fn is_one(&self) -> bool;
    /// Checks if this node is [`BddNode::zero`] or [`BddNode::one`].
    fn is_terminal(&self) -> bool;

    /// Return the low-child reference, assuming this is a decision node. For terminal nodes,
    /// return a reference to `self`.
    fn low(&self) -> Self::Id;
    /// Return the high-child reference, assuming this is a decision node. For terminal nodes,
    /// return a reference to `self`.
    fn high(&self) -> Self::Id;
    /// Return the decision variable, assuming this is a decision node. For terminal nodes,
    /// return [`VariableId::undefined`].
    fn variable(&self) -> Self::VarId;
}

/// An implementation of [`BddNode`] backed by [`NodeId32`] and [`VarIdPacked32`].
///
/// Note that [`VarIdPacked32`] also uses three of its bits to provide a `{0,1,many}` "parent
/// counter" and a "use cache" boolean flag.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
        self.is_terminal() && self.low.is_zero()
    }

    fn is_one(&self) -> bool {
        self.is_terminal() && self.low.is_one()
    }

    fn is_terminal(&self) -> bool {
        // Only the terminal nodes are allowed to have undefined as their decision variable.
        self.variable.is_undefined()
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
    /// Create a new *decision node* instance of [`BddNode32`].
    ///
    /// ## Undefined behavior
    ///
    /// When creating a new [`BddNode32`], no argument is allowed to be "undefined". If this happens,
    /// the implementation will panic in debug mode, but these checks do not run in release mode
    /// for performance reasons. Hence, using an undefined value can result in undefined behavior.
    pub fn new(variable: VarIdPacked32, low: NodeId32, high: NodeId32) -> Self {
        debug_assert!(!variable.is_undefined());
        debug_assert!(!low.is_undefined());
        debug_assert!(!high.is_undefined());
        BddNode32 {
            variable,
            low,
            high,
        }
    }

    /// Increment the "parend counter" of the underlying [`VarIdPacked32`] variable ID.
    pub fn increment_parents(&mut self) {
        self.variable.increment_parents();
    }

    /// Check if the "parend counter" of the underlying [`VarIdPacked32`] variable ID is
    /// in the `many` state.
    pub fn has_many_parents(&self) -> bool {
        self.variable.has_many_parents()
    }
}

#[cfg(test)]
mod tests {
    use crate::bdd_node::{BddNode, BddNode32};
    use crate::node_id::{BddNodeId, NodeId32};
    use crate::variable_id::{VarIdPacked32, VariableId};

    #[test]
    pub fn test_bdd_node_32_invariants() {
        assert!(BddNode32::zero().is_zero());
        assert!(BddNode32::zero().is_terminal());
        assert!(BddNode32::zero().variable().is_undefined());
        assert_eq!(BddNode32::zero().low(), BddNodeId::zero());
        assert_eq!(BddNode32::zero().high(), BddNodeId::zero());

        assert!(BddNode32::one().is_one());
        assert!(BddNode32::one().is_terminal());
        assert!(BddNode32::one().variable().is_undefined());
        assert_eq!(BddNode32::one().low(), BddNodeId::one());
        assert_eq!(BddNode32::one().high(), BddNodeId::one());

        let v = VarIdPacked32::new(1);
        // This is a normal decision node that is not terminal.
        assert!(!BddNode32::new(v, NodeId32::one(), NodeId32::zero()).is_terminal());
        assert!(!BddNode32::new(v, NodeId32::one(), NodeId32::zero()).is_one());
        assert!(!BddNode32::new(v, NodeId32::one(), NodeId32::zero()).is_zero());
        // These are "useless" decision nodes that should not appear in a canonical BDD,
        // but are nevertheless not terminal.
        assert!(!BddNode32::new(v, NodeId32::zero(), NodeId32::zero()).is_terminal());
        assert!(!BddNode32::new(v, NodeId32::zero(), NodeId32::zero()).is_one());
        assert!(!BddNode32::new(v, NodeId32::zero(), NodeId32::zero()).is_zero());
        assert!(!BddNode32::new(v, NodeId32::one(), NodeId32::one()).is_terminal());
        assert!(!BddNode32::new(v, NodeId32::one(), NodeId32::one()).is_one());
        assert!(!BddNode32::new(v, NodeId32::one(), NodeId32::one()).is_zero());

        let n = BddNode32::new(v, NodeId32::new(15), NodeId32::zero());
        assert_eq!(n.variable(), v);
        assert_eq!(n.low(), NodeId32::new(15));
        assert_eq!(n.high(), NodeId32::zero());
    }

    #[test]
    pub fn test_bdd_node_32_delegation() {
        let mut n = BddNode32::new(VarIdPacked32::new(1), NodeId32::zero(), NodeId32::one());
        assert!(!n.has_many_parents());
        n.increment_parents();
        assert!(!n.has_many_parents());
        n.increment_parents();
        assert!(n.has_many_parents());
        n.increment_parents();
        assert!(n.has_many_parents());
    }

    #[test]
    #[should_panic]
    pub fn test_bdd_node_32_invalid_1() {
        BddNode32::new(
            VarIdPacked32::new(1),
            NodeId32::one(),
            NodeId32::undefined(),
        );
    }

    #[test]
    #[should_panic]
    pub fn test_bdd_node_32_invalid_2() {
        BddNode32::new(
            VarIdPacked32::new(1),
            NodeId32::undefined(),
            NodeId32::zero(),
        );
    }

    #[test]
    #[should_panic]
    pub fn test_bdd_node_32_invalid_3() {
        BddNode32::new(VarIdPacked32::undefined(), NodeId32::one(), NodeId32::one());
    }
}
