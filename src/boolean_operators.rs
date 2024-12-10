//! Defines three-valued logic operators based on [`TriBool`]. These are used to implement
//! Boolean operators in the `apply` algorithms.
//!
use std::cmp::{max, min};

use crate::node_id::BddNodeId;

/// Applies a three-value logic operator to a pair of [`BddNodeId`] arguments.
pub fn lift_operator<NodeId, TriBoolOperator>(
    left: NodeId,
    right: NodeId,
    operator: TriBoolOperator,
) -> NodeId
where
    NodeId: BddNodeId,
    TriBoolOperator: Fn(TriBool, TriBool) -> TriBool,
{
    let left = left.to_three_valued();
    let right = right.to_three_valued();
    BddNodeId::from_three_valued(operator(left, right))
}

/// Thee-valued conjunction of two [`BddNodeId`] identifiers.
pub fn and<NodeId>(left: NodeId, right: NodeId) -> NodeId
where
    NodeId: BddNodeId,
{
    lift_operator(left, right, TriBool::and)
}

/// Thee-valued disjunction of two [`BddNodeId`] identifiers.
pub fn or<NodeId>(left: NodeId, right: NodeId) -> NodeId
where
    NodeId: BddNodeId,
{
    lift_operator(left, right, TriBool::or)
}

/// Thee-valued exclusive or (non-equivalence) of two [`BddNodeId`] identifiers.
pub fn xor<NodeId>(left: NodeId, right: NodeId) -> NodeId
where
    NodeId: BddNodeId,
{
    lift_operator(left, right, TriBool::xor)
}

/// Thee-valued implication of two [`BddNodeId`] identifiers.
pub fn implies<NodeId>(left: NodeId, right: NodeId) -> NodeId
where
    NodeId: BddNodeId,
{
    lift_operator(left, right, TriBool::implies)
}

/// Thee-valued equivalence of two [`BddNodeId`] identifiers.
pub fn iff<NodeId>(left: NodeId, right: NodeId) -> NodeId
where
    NodeId: BddNodeId,
{
    lift_operator(left, right, TriBool::iff)
}

/// A type representing a three-valued logic value.
#[derive(PartialOrd, Ord, PartialEq, Eq, Debug)]
#[repr(i8)]
pub enum TriBool {
    True = 1,
    Indeterminate = 0,
    False = -1,
}

impl TriBool {
    /// Logical disjunction.
    pub(crate) fn or(self, other: Self) -> Self {
        max(self, other)
    }

    /// Logical conjunction.
    pub(crate) fn and(self, other: Self) -> Self {
        min(self, other)
    }

    /// Exclusive or (non-equivalence).
    pub(crate) fn xor(self, other: Self) -> Self {
        // min(max(a,b), neg(min(a,b)))
        let [smaller, greater] = if self < other {
            [self, other]
        } else {
            [other, self]
        };
        min(greater, !smaller)
    }

    /// Implication.
    pub(crate) fn implies(self, other: Self) -> Self {
        (!self).or(other)
    }

    /// Equivalence.
    pub(crate) fn iff(self, other: Self) -> Self {
        !self.xor(other)
    }
}

impl std::ops::Not for TriBool {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Self::True => Self::False,
            Self::Indeterminate => Self::Indeterminate,
            Self::False => Self::True,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_id::NodeId32;

    #[test]
    pub fn three_valued_logic_invariants() {
        // Just a representative subset of input-output pairs for each operator.

        assert!(and(NodeId32::one(), NodeId32::one()).is_one());
        assert!(and(NodeId32::zero(), NodeId32::one()).is_zero());
        assert!(and(NodeId32::zero(), NodeId32::undefined()).is_zero());
        assert!(and(NodeId32::one(), NodeId32::undefined()).is_undefined());

        assert!(or(NodeId32::zero(), NodeId32::zero()).is_zero());
        assert!(or(NodeId32::one(), NodeId32::zero()).is_one());
        assert!(or(NodeId32::one(), NodeId32::undefined()).is_one());
        assert!(or(NodeId32::zero(), NodeId32::undefined()).is_undefined());

        assert!(implies(NodeId32::one(), NodeId32::zero()).is_zero());
        assert!(implies(NodeId32::zero(), NodeId32::zero()).is_one());
        assert!(implies(NodeId32::zero(), NodeId32::undefined()).is_one());
        assert!(implies(NodeId32::one(), NodeId32::undefined()).is_undefined());

        assert!(xor(NodeId32::one(), NodeId32::one()).is_zero());
        assert!(xor(NodeId32::zero(), NodeId32::one()).is_one());
        assert!(xor(NodeId32::zero(), NodeId32::undefined()).is_undefined());
        assert!(xor(NodeId32::one(), NodeId32::undefined()).is_undefined());

        assert!(iff(NodeId32::one(), NodeId32::one()).is_one());
        assert!(iff(NodeId32::zero(), NodeId32::one()).is_zero());
        assert!(iff(NodeId32::zero(), NodeId32::undefined()).is_undefined());
        assert!(iff(NodeId32::one(), NodeId32::undefined()).is_undefined());
    }
}
