//! Defines three-valued logic operators based on [`TriBool`]. These are used to implement
//! Boolean operators in the `apply` algorithms.
//!
use std::cmp::{max, min};

use crate::node_id::NodeIdAny;

/// Lifts a three-valued logic operator to operate on [`NodeIdAny`] identifiers.
pub fn lift_operator<
    TId1: NodeIdAny,
    TId2: NodeIdAny,
    TResultId: NodeIdAny,
    TTriBoolOperator: Fn(TriBool, TriBool) -> TriBool,
>(
    operator: TTriBoolOperator,
) -> impl Fn(TId1, TId2) -> TResultId {
    move |left, right| {
        let left = left.to_three_valued();
        let right = right.to_three_valued();
        TResultId::from_three_valued(operator(left, right))
    }
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

pub trait BooleanOperator: Clone + Copy {
    /// Return a function implementing this boolean operation on [`NodeIdAny`] identifiers,
    /// suitable for split BDDs.
    fn for_split<TId1: NodeIdAny, TId2: NodeIdAny, TResultId: NodeIdAny>(
        self,
    ) -> impl Fn(TId1, TId2) -> TResultId;
    /// Return a function implementing this boolean operation on [`NodeIdAny`] identifiers,
    /// suitable for shared BDDs.
    fn for_shared<TId1: NodeIdAny, TId2: NodeIdAny, TResultId: NodeIdAny>(
        self,
    ) -> impl Fn(TId1, TId2) -> TResultId;
}

/// A type representing a logical conjunction.
#[derive(Clone, Copy)]
pub struct And;

impl BooleanOperator for And {
    fn for_split<TId1: NodeIdAny, TId2: NodeIdAny, TResultId: NodeIdAny>(
        self,
    ) -> impl Fn(TId1, TId2) -> TResultId {
        lift_operator(TriBool::and)
    }

    fn for_shared<TId1: NodeIdAny, TId2: NodeIdAny, TResultId: NodeIdAny>(
        self,
    ) -> impl Fn(TId1, TId2) -> TResultId {
        // TODO: For now just use the same operator as split.
        self.for_split()
    }
}

/// A type representing a logical disjunction.
#[derive(Clone, Copy)]
pub struct Or;

impl BooleanOperator for Or {
    fn for_split<TId1: NodeIdAny, TId2: NodeIdAny, TResultId: NodeIdAny>(
        self,
    ) -> impl Fn(TId1, TId2) -> TResultId {
        lift_operator(TriBool::or)
    }

    fn for_shared<TId1: NodeIdAny, TId2: NodeIdAny, TResultId: NodeIdAny>(
        self,
    ) -> impl Fn(TId1, TId2) -> TResultId {
        // TODO: For now just use the same operator as split.
        self.for_split()
    }
}

/// A type representing a logical equivalence.
#[derive(Clone, Copy)]
pub struct Iff;

impl BooleanOperator for Iff {
    fn for_split<TId1: NodeIdAny, TId2: NodeIdAny, TResultId: NodeIdAny>(
        self,
    ) -> impl Fn(TId1, TId2) -> TResultId {
        lift_operator(TriBool::iff)
    }

    fn for_shared<TId1: NodeIdAny, TId2: NodeIdAny, TResultId: NodeIdAny>(
        self,
    ) -> impl Fn(TId1, TId2) -> TResultId {
        // TODO: For now just use the same operator as split.
        self.for_split()
    }
}

/// A type representing a logical implication.
#[derive(Clone, Copy)]
pub struct Implies;

impl BooleanOperator for Implies {
    fn for_split<TId1: NodeIdAny, TId2: NodeIdAny, TResultId: NodeIdAny>(
        self,
    ) -> impl Fn(TId1, TId2) -> TResultId {
        lift_operator(TriBool::implies)
    }

    fn for_shared<TId1: NodeIdAny, TId2: NodeIdAny, TResultId: NodeIdAny>(
        self,
    ) -> impl Fn(TId1, TId2) -> TResultId {
        // TODO: For now just use the same operator as split.
        self.for_split()
    }
}

/// A type representing a logical xor.
#[derive(Clone, Copy)]
pub struct Xor;

impl BooleanOperator for Xor {
    fn for_split<TId1: NodeIdAny, TId2: NodeIdAny, TResultId: NodeIdAny>(
        self,
    ) -> impl Fn(TId1, TId2) -> TResultId {
        lift_operator(TriBool::xor)
    }

    fn for_shared<TId1: NodeIdAny, TId2: NodeIdAny, TResultId: NodeIdAny>(
        self,
    ) -> impl Fn(TId1, TId2) -> TResultId {
        // TODO: For now just use the same operator as split.
        self.for_split()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_id::NodeId32;

    #[test]
    pub fn operators_for_split() {
        // Just a representative subset of input-output pairs for each operator.

        let and = And.for_split::<NodeId32, NodeId32, NodeId32>();
        let or = Or.for_split::<NodeId32, NodeId32, NodeId32>();
        let implies = Implies.for_split::<NodeId32, NodeId32, NodeId32>();
        let xor = Xor.for_split::<NodeId32, NodeId32, NodeId32>();
        let iff = Iff.for_split::<NodeId32, NodeId32, NodeId32>();

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
