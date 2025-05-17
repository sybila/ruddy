use std::cmp::{max, min};

use crate::node_id::NodeIdAny;

/// Convert an ID into a [`TriBool`], where the terminal node 0 is mapped to `False`,
/// the terminal node 1 is mapped to `True`, and all other IDs are mapped to `Indeterminate`.
fn to_three_valued<T: NodeIdAny>(id: T) -> TriBool {
    // Decompiles to branch-less, nice code
    // 0 -> -1, 1 -> 1, _ -> 0
    match -i8::from(id.is_zero()) + i8::from(id.is_one()) {
        1 => TriBool::True,
        0 => TriBool::Indeterminate,
        -1 => TriBool::False,
        _ => unreachable!(),
    }
}

/// Convert a [`TriBool`] to a node ID. The value
/// `True` is mapped to the terminal node 1, `False` is mapped to the terminal node 0, and
/// `Indeterminate` is mapped to the ID with the undefined value.
fn from_three_valued<T: NodeIdAny>(value: TriBool) -> T {
    match value {
        TriBool::True => T::one(),
        TriBool::False => T::zero(),
        TriBool::Indeterminate => T::undefined(),
    }
}

/// Lifts a three-valued logic operator to operate on [`NodeIdAny`] identifiers.
fn lift_operator<
    TId1: NodeIdAny,
    TId2: NodeIdAny,
    TResultId: NodeIdAny,
    TTriBoolOperator: Fn(TriBool, TriBool) -> TriBool,
>(
    operator: TTriBoolOperator,
) -> impl Fn(TId1, TId2) -> TResultId {
    move |left, right| {
        let left = to_three_valued(left);
        let right = to_three_valued(right);
        from_three_valued(operator(left, right))
    }
}

/// A type representing a three-valued logic value.
#[derive(PartialOrd, Ord, PartialEq, Eq, Debug)]
#[repr(i8)]
enum TriBool {
    True = 1,
    Indeterminate = 0,
    False = -1,
}

impl TriBool {
    /// Logical disjunction.
    fn or(self, other: Self) -> Self {
        max(self, other)
    }

    /// Logical conjunction.
    fn and(self, other: Self) -> Self {
        min(self, other)
    }

    /// Exclusive or (non-equivalence).
    fn xor(self, other: Self) -> Self {
        // min(max(a,b), neg(min(a,b)))
        let [smaller, greater] = if self < other {
            [self, other]
        } else {
            [other, self]
        };
        min(greater, !smaller)
    }

    /// Implication.
    fn implies(self, other: Self) -> Self {
        (!self).or(other)
    }

    /// Equivalence.
    fn iff(self, other: Self) -> Self {
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
    fn for_shared<TId: NodeIdAny>(self) -> impl Fn(TId, TId) -> TId;
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

    fn for_shared<TId: NodeIdAny>(self) -> impl Fn(TId, TId) -> TId {
        |left, right| {
            if left.is_zero() || right.is_zero() {
                TId::zero()
            } else if left.is_one() {
                right
            } else if right.is_one() {
                left
            } else if left == right {
                right
            } else {
                TId::undefined()
            }
        }
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

    fn for_shared<TId: NodeIdAny>(self) -> impl Fn(TId, TId) -> TId {
        |left, right| {
            if left.is_one() || right.is_one() {
                TId::one()
            } else if left.is_zero() {
                right
            } else if right.is_zero() {
                left
            } else if left == right {
                right
            } else {
                TId::undefined()
            }
        }
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

    fn for_shared<TId: NodeIdAny>(self) -> impl Fn(TId, TId) -> TId {
        |left, right| {
            if left.is_one() {
                right
            } else if right.is_one() {
                left
            } else if left == right {
                TId::one()
            } else {
                TId::undefined()
            }
        }
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

    fn for_shared<TId: NodeIdAny>(self) -> impl Fn(TId, TId) -> TId {
        |left, right| {
            if left.is_zero() {
                TId::one()
            } else if left.is_one() {
                right
            } else if right.is_one() || left == right {
                TId::one()
            } else {
                TId::undefined()
            }
        }
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

    fn for_shared<TId: NodeIdAny>(self) -> impl Fn(TId, TId) -> TId {
        |left, right| {
            if left.is_zero() {
                right
            } else if right.is_zero() {
                left
            } else if left == right {
                TId::zero()
            } else {
                TId::undefined()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_id::NodeId32;

    #[test]
    fn id_tribool_conversion() {
        assert!(from_three_valued::<NodeId32>(TriBool::True).is_one());
        assert!(from_three_valued::<NodeId32>(TriBool::False).is_zero());
        assert!(from_three_valued::<NodeId32>(TriBool::Indeterminate).is_undefined());

        assert_eq!(
            to_three_valued(NodeId32::undefined()),
            TriBool::Indeterminate
        );
        assert_eq!(to_three_valued(NodeId32::one()), TriBool::True);
        assert_eq!(to_three_valued(NodeId32::zero()), TriBool::False);
    }

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

    #[test]
    fn and_shared() {
        let and = And.for_shared::<NodeId32>();
        let zero = NodeId32::zero();
        let one = NodeId32::one();
        let n22 = NodeId32::new(22);
        let n11 = NodeId32::new(11);

        // zero
        assert!(and(one, zero).is_zero());
        assert!(and(zero, one).is_zero());
        assert!(and(zero, n22).is_zero());
        assert!(and(n11, zero).is_zero());

        // one
        assert!(and(one, one).is_one());
        assert_eq!(n11, and(one, n11));
        assert_eq!(n22, and(n22, one));

        // equal
        assert_eq!(n11, and(n11, n11));
        assert_eq!(n22, and(n22, n22));

        // not known
        assert!(and(n11, n22).is_undefined());
        assert!(and(n22, n11).is_undefined());
    }

    #[test]
    fn or_shared() {
        let or = Or.for_shared::<NodeId32>();
        let zero = NodeId32::zero();
        let one = NodeId32::one();
        let n22 = NodeId32::new(22);
        let n11 = NodeId32::new(11);

        // one
        assert!(or(one, zero).is_one());
        assert!(or(zero, one).is_one());
        assert!(or(one, n22).is_one());
        assert!(or(n11, one).is_one());

        // zero
        assert!(or(zero, zero).is_zero());
        assert_eq!(n11, or(zero, n11));
        assert_eq!(n22, or(n22, zero));

        // equal
        assert_eq!(n11, or(n11, n11));
        assert_eq!(n22, or(n22, n22));

        // not known
        assert!(or(n11, n22).is_undefined());
        assert!(or(n22, n11).is_undefined());
    }

    #[test]
    fn implies_shared() {
        let implies = Implies.for_shared::<NodeId32>();
        let zero = NodeId32::zero();
        let one = NodeId32::one();
        let n22 = NodeId32::new(22);
        let n11 = NodeId32::new(11);

        // false (only when true implies false)
        assert!(implies(one, zero).is_zero());

        // true when antecedent is false (regardless of consequent)
        assert!(implies(zero, zero).is_one());
        assert!(implies(zero, one).is_one());
        assert!(implies(zero, n11).is_one());

        // true when consequent is true (regardless of antecedent)
        assert!(implies(one, one).is_one());
        assert!(implies(n22, one).is_one());

        // equal implies equal is true
        assert!(implies(n11, n11).is_one());
        assert!(implies(n22, n22).is_one());

        // not known
        assert!(implies(n11, n22).is_undefined());
        assert!(implies(n22, n11).is_undefined());
    }

    #[test]
    fn xor_shared() {
        let xor = Xor.for_shared::<NodeId32>();
        let zero = NodeId32::zero();
        let one = NodeId32::one();
        let n22 = NodeId32::new(22);
        let n11 = NodeId32::new(11);

        // XOR with zero returns the other operand
        assert_eq!(one, xor(zero, one));
        assert_eq!(one, xor(one, zero));
        assert_eq!(n11, xor(zero, n11));
        assert_eq!(n22, xor(n22, zero));

        // XOR of same values is zero
        assert!(xor(one, one).is_zero());
        assert!(xor(zero, zero).is_zero());
        assert_eq!(zero, xor(n11, n11));
        assert_eq!(zero, xor(n22, n22));

        // not known
        assert!(xor(n11, n22).is_undefined());
        assert!(xor(n22, n11).is_undefined());
        assert!(xor(one, n11).is_undefined());
        assert!(xor(n22, one).is_undefined());
    }

    #[test]
    fn iff_shared() {
        let iff = Iff.for_shared::<NodeId32>();
        let zero = NodeId32::zero();
        let one = NodeId32::one();
        let n22 = NodeId32::new(22);
        let n11 = NodeId32::new(11);

        // true when both inputs are the same constant
        assert!(iff(zero, zero).is_one());
        assert!(iff(one, one).is_one());

        // false when constants are different
        assert!(iff(zero, one).is_zero());
        assert!(iff(one, zero).is_zero());

        // same node is equivalent to itself
        assert!(iff(n11, n11).is_one());
        assert!(iff(n22, n22).is_one());

        // returns other operand when one
        assert_eq!(n11, iff(one, n11));
        assert_eq!(n22, iff(n22, one));

        // not known
        assert!(iff(n11, zero).is_undefined());
        assert!(iff(zero, n22).is_undefined());
        assert!(iff(n11, n22).is_undefined());
        assert!(iff(n22, n11).is_undefined());
    }
}
