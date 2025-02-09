//! Defines the representation of BDD nodes. Includes: [`BddNodeAny`], [`BddNode16`],
//! [`BddNode32`], and [`BddNode64`].

use std::fmt::Debug;
use std::{convert::TryFrom, fmt};

use crate::{
    node_id::{NodeId16, NodeId32, NodeId64, NodeIdAny, TryFromNodeIdError},
    variable_id::{
        TryFromVarIdPackedError, VarIdPacked16, VarIdPacked32, VarIdPacked64, VarIdPackedAny,
    },
};

/// An internal trait implemented by types that can serve as BDD nodes. Each BDD node is either
/// a *terminal node* (`0` or `1`) or a *decision node*. A decision node consists of the decision
/// variable (of type [`VarIdPackedAny`]) and two child references, *low* and *high*
/// (of type [`NodeIdAny`]).
pub trait BddNodeAny: Clone + Eq + Debug {
    /// Node ID type used by this [`BddNodeAny`].
    type Id: NodeIdAny;
    /// Variable ID type used by this [`BddNodeAny`].
    type VarId: VarIdPackedAny;

    /// Create a new *decision node* instance of [`BddNodeAny`].
    ///
    /// The "parent counter" and "use cache" flags in `variable` are reset.
    /// ## Undefined behavior
    ///
    /// When creating a new [`BddNodeAny`], no argument is allowed to be "undefined". If this happens,
    /// the implementation will panic in debug mode, but these checks do not run in release mode
    /// for performance reasons. Hence, using an undefined value can result in undefined behavior.
    fn new(variable: Self::VarId, low: Self::Id, high: Self::Id) -> Self;

    /// Return an instance of the terminal `0` node.
    fn zero() -> Self;
    /// Return an instance of the terminal `1` node.
    fn one() -> Self;

    /// Checks if this node is [`BddNodeAny::zero`].
    fn is_zero(&self) -> bool;
    /// Checks if this node is [`BddNodeAny::one`].
    fn is_one(&self) -> bool;
    /// Checks if this node is [`BddNodeAny::zero`] or [`BddNodeAny::one`].
    fn is_terminal(&self) -> bool;

    /// Return the low-child reference, assuming this is a decision node. For terminal nodes,
    /// return a reference to `self`.
    fn low(&self) -> Self::Id;
    /// Return the high-child reference, assuming this is a decision node. For terminal nodes,
    /// return a reference to `self`.
    fn high(&self) -> Self::Id;
    /// Return the decision variable, assuming this is a decision node. For terminal nodes,
    /// return [`VarIdPackedAny::undefined`].
    fn variable(&self) -> Self::VarId;

    /// Increment the parent counter of the node.
    fn increment_parent_counter(&mut self);

    /// Check if the node has more than one parent.
    fn has_many_parents(&self) -> bool {
        self.variable().has_many_parents()
    }
}

macro_rules! impl_bdd_node {
    ($name:ident, $NodeId:ident, $VarId:ident) => {
        impl BddNodeAny for $name {
            type Id = $NodeId;
            type VarId = $VarId;

            fn new(variable: Self::VarId, low: Self::Id, high: Self::Id) -> Self {
                debug_assert!(!variable.is_undefined());
                debug_assert!(!low.is_undefined());
                debug_assert!(!high.is_undefined());
                Self {
                    variable: variable.reset(),
                    low,
                    high,
                }
            }

            fn zero() -> Self {
                Self {
                    variable: $VarId::undefined(),
                    low: $NodeId::zero(),
                    high: $NodeId::zero(),
                }
            }

            fn one() -> Self {
                Self {
                    variable: $VarId::undefined(),
                    low: $NodeId::one(),
                    high: $NodeId::one(),
                }
            }

            fn is_zero(&self) -> bool {
                self.is_terminal() && self.low.is_zero()
            }

            fn is_one(&self) -> bool {
                self.is_terminal() && self.low.is_one()
            }

            fn is_terminal(&self) -> bool {
                self.variable.is_undefined()
            }

            fn low(&self) -> Self::Id {
                self.low
            }

            fn high(&self) -> Self::Id {
                self.high
            }

            fn variable(&self) -> Self::VarId {
                self.variable
            }

            fn increment_parent_counter(&mut self) {
                self.variable.increment_parents();
            }
        }
    };
}

/// An implementation of [`BddNodeAny`] backed by [`NodeId16`] and [`VarIdPacked16`].
///
/// Note that [`VarIdPacked16`] also uses three of its bits to provide a `{0,1,many}` "parent
/// counter" and a "use cache" boolean flag.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BddNode16 {
    pub variable: VarIdPacked16,
    pub low: NodeId16,
    pub high: NodeId16,
}

impl_bdd_node!(BddNode16, NodeId16, VarIdPacked16);

/// An implementation of [`BddNodeAny`] backed by [`NodeId32`] and [`VarIdPacked32`].
///
/// Note that [`VarIdPacked32`] also uses three of its bits to provide a `{0,1,many}` "parent
/// counter" and a "use cache" boolean flag.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BddNode32 {
    pub variable: VarIdPacked32,
    pub low: NodeId32,
    pub high: NodeId32,
}

impl_bdd_node!(BddNode32, NodeId32, VarIdPacked32);

/// An implementation of [`BddNodeAny`] backed by [`NodeId64`] and [`VarIdPacked64`].
///
/// Note that [`VarIdPacked64`] also uses three of its bits to provide a `{0,1,many}` "parent
/// counter" and a "use cache" boolean flag.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BddNode64 {
    pub variable: VarIdPacked64,
    pub low: NodeId64,
    pub high: NodeId64,
}

impl_bdd_node!(BddNode64, NodeId64, VarIdPacked64);

macro_rules! impl_from {
    ($Small:ident => $Large:ident) => {
        impl From<$Small> for $Large {
            fn from(node: $Small) -> Self {
                Self {
                    variable: node.variable().into(),
                    low: node.low().into(),
                    high: node.high().into(),
                }
            }
        }
    };
}

impl_from!(BddNode16 => BddNode32);
impl_from!(BddNode16 => BddNode64);
impl_from!(BddNode32 => BddNode64);

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum TryFromBddNodeError {
    Variable(TryFromVarIdPackedError),
    Low(TryFromNodeIdError),
    High(TryFromNodeIdError),
}

impl fmt::Display for TryFromBddNodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bdd node cannot be converted: ")?;
        match self {
            Self::Variable(e) => write!(f, "{}", e),
            Self::Low(e) => write!(f, "low child {}", e),
            Self::High(e) => write!(f, "high child {}", e),
        }
    }
}

impl std::error::Error for TryFromBddNodeError {}

macro_rules! impl_try_from {
    ($Large:ident => $Small:ident) => {
        impl TryFrom<$Large> for $Small {
            type Error = TryFromBddNodeError;

            fn try_from(node: $Large) -> Result<Self, Self::Error> {
                Ok(Self {
                    variable: node
                        .variable()
                        .try_into()
                        .map_err(TryFromBddNodeError::Variable)?,
                    low: node.low().try_into().map_err(TryFromBddNodeError::Low)?,
                    high: node.high().try_into().map_err(TryFromBddNodeError::High)?,
                })
            }
        }
    };
}

impl_try_from!(BddNode32 => BddNode16);
impl_try_from!(BddNode64 => BddNode16);
impl_try_from!(BddNode64 => BddNode32);

#[cfg(test)]
mod tests {
    use crate::bdd_node::{BddNode16, BddNode32, BddNode64, BddNodeAny};
    use crate::node_id::{NodeId16, NodeId32, NodeId64, NodeIdAny};
    use crate::variable_id::{VarIdPacked16, VarIdPacked32, VarIdPacked64, VarIdPackedAny};

    macro_rules! test_bdd_node_invariants {
        ($func:ident, $BddNode:ident, $VarId:ident, $NodeId:ident) => {
            #[test]
            pub fn $func() {
                assert!($BddNode::zero().is_zero());
                assert!($BddNode::zero().is_terminal());
                assert!($BddNode::zero().variable().is_undefined());
                assert_eq!($BddNode::zero().low(), $NodeId::zero());
                assert_eq!($BddNode::zero().high(), $NodeId::zero());

                assert!($BddNode::one().is_one());
                assert!($BddNode::one().is_terminal());
                assert!($BddNode::one().variable().is_undefined());
                assert_eq!($BddNode::one().low(), $NodeId::one());
                assert_eq!($BddNode::one().high(), $NodeId::one());

                let v = $VarId::new(1);
                // This is a normal decision node that is not terminal.
                assert!(!$BddNode::new(v, $NodeId::one(), $NodeId::zero()).is_terminal());
                assert!(!$BddNode::new(v, $NodeId::one(), $NodeId::zero()).is_one());
                assert!(!$BddNode::new(v, $NodeId::one(), $NodeId::zero()).is_zero());
                // These are "useless" decision nodes that should not appear in a canonical BDD,
                // but are nevertheless not terminal.
                assert!(!$BddNode::new(v, $NodeId::zero(), $NodeId::zero()).is_terminal());
                assert!(!$BddNode::new(v, $NodeId::zero(), $NodeId::zero()).is_one());
                assert!(!$BddNode::new(v, $NodeId::zero(), $NodeId::zero()).is_zero());
                assert!(!$BddNode::new(v, $NodeId::one(), $NodeId::one()).is_terminal());
                assert!(!$BddNode::new(v, $NodeId::one(), $NodeId::one()).is_one());
                assert!(!$BddNode::new(v, $NodeId::one(), $NodeId::one()).is_zero());

                let n = $BddNode::new(v, $NodeId::new(15), $NodeId::zero());
                assert_eq!(n.variable(), v);
                assert_eq!(n.low(), $NodeId::new(15));
                assert_eq!(n.high(), $NodeId::zero());
            }
        };
    }

    test_bdd_node_invariants!(bdd_node_16_invariants, BddNode16, VarIdPacked16, NodeId16);
    test_bdd_node_invariants!(bdd_node_32_invariants, BddNode32, VarIdPacked32, NodeId32);
    test_bdd_node_invariants!(bdd_node_64_invariants, BddNode64, VarIdPacked64, NodeId64);

    macro_rules! test_bdd_node_delegation {
        ($func:ident, $BddNode:ident, $VarId:ident, $NodeId:ident) => {
            #[test]
            pub fn $func() {
                let mut n = $BddNode::new($VarId::new(1), $NodeId::zero(), $NodeId::one());
                assert!(!n.has_many_parents());
                n.increment_parent_counter();
                assert!(!n.has_many_parents());
                n.increment_parent_counter();
                assert!(n.has_many_parents());
                n.increment_parent_counter();
                assert!(n.has_many_parents());
            }
        };
    }

    test_bdd_node_delegation!(bdd_node_16_delegation, BddNode16, VarIdPacked16, NodeId16);
    test_bdd_node_delegation!(bdd_node_32_delegation, BddNode32, VarIdPacked32, NodeId32);
    test_bdd_node_delegation!(bdd_node_64_delegation, BddNode64, VarIdPacked64, NodeId64);

    macro_rules! test_bdd_node_invalid_1 {
        ($func:ident, $BddNode:ident, $VarId:ident, $NodeId:ident) => {
            #[test]
            #[should_panic]
            pub fn $func() {
                $BddNode::new($VarId::new(1), $NodeId::one(), $NodeId::undefined());
            }
        };
    }

    test_bdd_node_invalid_1!(bdd_node_16_invalid_1, BddNode16, VarIdPacked16, NodeId16);
    test_bdd_node_invalid_1!(bdd_node_32_invalid_1, BddNode32, VarIdPacked32, NodeId32);
    test_bdd_node_invalid_1!(bdd_node_64_invalid_1, BddNode64, VarIdPacked64, NodeId64);

    macro_rules! test_bdd_node_invalid_2 {
        ($func:ident, $BddNode:ident, $VarId:ident, $NodeId:ident) => {
            #[test]
            #[should_panic]
            pub fn $func() {
                $BddNode::new($VarId::new(1), $NodeId::undefined(), $NodeId::zero());
            }
        };
    }

    test_bdd_node_invalid_2!(bdd_node_16_invalid_2, BddNode16, VarIdPacked16, NodeId16);
    test_bdd_node_invalid_2!(bdd_node_32_invalid_2, BddNode32, VarIdPacked32, NodeId32);
    test_bdd_node_invalid_2!(bdd_node_64_invalid_2, BddNode64, VarIdPacked64, NodeId64);

    macro_rules! test_bdd_node_invalid_3 {
        ($func:ident, $BddNode:ident, $VarId:ident, $NodeId:ident) => {
            #[test]
            #[should_panic]
            pub fn $func() {
                $BddNode::new($VarId::undefined(), $NodeId::one(), $NodeId::one());
            }
        };
    }

    test_bdd_node_invalid_3!(bdd_node_16_invalid_3, BddNode16, VarIdPacked16, NodeId16);
    test_bdd_node_invalid_3!(bdd_node_32_invalid_3, BddNode32, VarIdPacked32, NodeId32);
    test_bdd_node_invalid_3!(bdd_node_64_invalid_3, BddNode64, VarIdPacked64, NodeId64);
}
