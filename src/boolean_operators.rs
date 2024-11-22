use std::cmp::{max, min};

use crate::node_id::BddNodeId;

pub fn lift_operator<NodeId, TriBoolOperator>(
    left: NodeId,
    right: NodeId,
    operator: TriBoolOperator,
) -> Option<NodeId>
where
    NodeId: BddNodeId,
    TriBoolOperator: Fn(TriBool, TriBool) -> TriBool,
{
    let left = left.to_three_valued();
    let right = right.to_three_valued();
    BddNodeId::from_three_valued(operator(left, right))
}

pub fn and<NodeId>(left: NodeId, right: NodeId) -> Option<NodeId>
where
    NodeId: BddNodeId,
{
    lift_operator(left, right, TriBool::and)
}

pub fn or<NodeId>(left: NodeId, right: NodeId) -> Option<NodeId>
where
    NodeId: BddNodeId,
{
    lift_operator(left, right, TriBool::or)
}

pub fn xor<NodeId>(left: NodeId, right: NodeId) -> Option<NodeId>
where
    NodeId: BddNodeId,
{
    lift_operator(left, right, TriBool::xor)
}

pub fn implies<NodeId>(left: NodeId, right: NodeId) -> Option<NodeId>
where
    NodeId: BddNodeId,
{
    lift_operator(left, right, TriBool::implies)
}

pub fn iff<NodeId>(left: NodeId, right: NodeId) -> Option<NodeId>
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
    pub(crate) fn or(self, other: Self) -> Self {
        max(self, other)
    }

    pub(crate) fn and(self, other: Self) -> Self {
        min(self, other)
    }

    pub(crate) fn xor(self, other: Self) -> Self {
        // min(max(a,b), neg(min(a,b)))
        let [smaller, greater] = if self < other {
            [self, other]
        } else {
            [other, self]
        };
        min(greater, !smaller)
    }

    pub(crate) fn implies(self, other: Self) -> Self {
        !self.or(!other)
    }

    pub(crate) fn iff(self, other: Self) -> Self {
        !(self.xor(other))
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
