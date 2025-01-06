//! Defines the representation of node identifiers. Includes: [`NodeIdAny`], [`NodeId16`],
//! [`NodeId32`] and [`NodeId64`].

use crate::{boolean_operators::TriBool, usize_is_at_least_32_bits, usize_is_at_least_64_bits};
use std::fmt::Debug;
use std::hash::Hash;
use std::num::TryFromIntError;

/// An internal trait implemented by types that can serve as BDD node identifiers. The core
/// property of this trait is that a node ID must have one designated "undefined" value
/// (similar to `Option::None` and equivalent to the maximal representable value), and two
/// designated "terminal" values equivalent to `0` and `1`. Every other node id must
/// fall in the `1 < id < undefined` interval.
pub trait NodeIdAny:
    Eq + Ord + Copy + Hash + Debug + Into<u64> + Into<u128> + TryFrom<usize>
{
    /// Return an instance of the "undefined" node ID.
    fn undefined() -> Self;
    /// Return an instance of the zero node ID.
    fn zero() -> Self;
    /// Return an instance of the one node ID.
    fn one() -> Self;

    /// Checks if this ID is [`NodeIdAny::undefined`].
    fn is_undefined(self) -> bool;
    /// Checks if this ID is [`NodeIdAny::zero`].
    fn is_zero(self) -> bool;
    /// Checks if this ID is [`NodeIdAny::one`].
    fn is_one(self) -> bool;
    /// Checks if this ID is [`NodeIdAny::zero`] or [`NodeIdAny::one`].
    fn is_terminal(self) -> bool;

    /// Convert the ID into a [`TriBool`], where the terminal node 0 is mapped to `False`,
    /// the terminal node 1 is mapped to `True`, and all other nodes are mapped to `Indeterminate`.
    fn to_three_valued(self) -> TriBool {
        // Decompiles to branch-less, nice code
        // 0 -> -1, 1 -> 1, _ -> 0
        match -i8::from(self.is_zero()) + i8::from(self.is_one()) {
            1 => TriBool::True,
            0 => TriBool::Indeterminate,
            -1 => TriBool::False,
            _ => unreachable!(),
        }
    }

    /// Convert a [`TriBool`] to a node ID, if possible. The value
    /// `True` is mapped to the terminal node 1, `False` is mapped to the terminal node 0, and
    /// `Indeterminate` is mapped to the ID with the undefined value.
    fn from_three_valued(value: TriBool) -> Self {
        match value {
            TriBool::True => Self::one(),
            TriBool::False => Self::zero(),
            TriBool::Indeterminate => Self::undefined(),
        }
    }

    /// Convert the ID safely into a `u16` value. Truncates the value if it is larger than [`u16::MAX`].
    fn as_u16(self) -> u16;
    /// Convert the ID safely into a `u32` value. Truncates the value if it is larger than [`u32::MAX`].
    fn as_u32(self) -> u32;

    /// Convert the ID safely into a value that can be used for indexing.
    ///
    /// ## Undefined behavior
    ///
    /// The result is not defined for [`NodeIdAny::undefined`]. In debug mode, the method will panic.
    /// In release mode, the result is undefined behavior.
    fn as_usize(self) -> usize;
}

/// Implementation of [`NodeIdAny`] backed by `u16`. The maximal ID is `2**16 - 1`.
#[repr(transparent)]
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct NodeId16(u16);

/// Implementation of [`NodeIdAny`] backed by `u32`. The maximal ID is `2**32 - 1`.
#[repr(transparent)]
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct NodeId32(u32);

/// Implementation of [`NodeIdAny`] backed by `u64`. The maximal ID is `2**64 - 1`.
#[repr(transparent)]
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct NodeId64(u64);

impl NodeId16 {
    /// The largest ID representable by [`NodeId16`].
    pub const MAX_ID: u16 = u16::MAX - 1;

    /// Create a new valid [`NodeId16`] from the underlying type `u16`.
    ///
    /// ## Undefined behavior
    ///
    /// This method should not be used to create instances of [`NodeId16::undefined`]. In debug mode,
    /// such operation will panic. In release mode, this is not checked but can cause undefined
    /// behavior.
    pub fn new(id: u16) -> Self {
        debug_assert!(id != u16::MAX, "cannot create 16-bit undefined id");
        Self(id)
    }
}

impl NodeIdAny for NodeId16 {
    fn undefined() -> Self {
        Self(u16::MAX)
    }

    fn zero() -> Self {
        Self(0)
    }

    fn one() -> Self {
        Self(1)
    }

    fn is_undefined(self) -> bool {
        self.0 == u16::MAX
    }

    fn is_zero(self) -> bool {
        self.0 == 0
    }

    fn is_one(self) -> bool {
        self.0 == 1
    }

    fn is_terminal(self) -> bool {
        self.0 <= 1
    }

    fn as_usize(self) -> usize {
        debug_assert!(
            self.0 != u16::MAX,
            "cannot use 16-bit undefined id as index"
        );
        usize::from(self.0)
    }

    fn as_u16(self) -> u16 {
        self.0
    }

    fn as_u32(self) -> u32 {
        u32::from(self.0)
    }
}

impl NodeId32 {
    /// The largest ID representable by [`NodeId32`].
    pub const MAX_ID: u32 = u32::MAX - 1;

    /// Create a new valid [`NodeId32`] from the underlying type `u32`.
    ///
    /// ## Undefined behavior
    ///
    /// This method should not be used to create instances of [`NodeId32::undefined`]. In debug mode,
    /// such operation will panic. In release mode, this is not checked but can cause undefined
    /// behavior.
    pub fn new(id: u32) -> Self {
        debug_assert!(id != u32::MAX, "cannot create 32-bit undefined id");
        Self(id)
    }

    /// Convert the underlying ID to a 4-byte array (for serialization).
    pub fn to_le_bytes(self) -> [u8; 4] {
        self.0.to_le_bytes()
    }

    /// Create a new [`NodeId32`] from a 4-byte array (for serialization).
    pub fn from_le_bytes(bytes: [u8; 4]) -> Self {
        Self(u32::from_le_bytes(bytes))
    }
}

impl NodeIdAny for NodeId32 {
    fn undefined() -> Self {
        Self(u32::MAX)
    }

    fn zero() -> Self {
        Self(0)
    }

    fn one() -> Self {
        Self(1)
    }

    fn is_undefined(self) -> bool {
        self.0 == u32::MAX
    }

    fn is_zero(self) -> bool {
        self.0 == 0
    }

    fn is_one(self) -> bool {
        self.0 == 1
    }

    fn is_terminal(self) -> bool {
        self.0 <= 1
    }

    fn as_usize(self) -> usize {
        debug_assert!(
            self.0 != u32::MAX,
            "cannot use 32-bit undefined id as index"
        );
        usize_is_at_least_32_bits(self.0)
    }

    #[allow(clippy::as_conversions)]
    fn as_u16(self) -> u16 {
        self.0 as u16
    }

    /// Convert the ID safely into a `u32` value.
    fn as_u32(self) -> u32 {
        self.0
    }
}

impl NodeId64 {
    /// The largest ID representable by [`NodeId64`].
    pub const MAX_ID: u64 = u64::MAX - 1;

    /// Create a new valid [`NodeId64`] from the underlying type `u64`.
    ///
    /// ## Undefined behavior
    ///
    /// This method should not be used to create instances of [`NodeId64::undefined`]. In debug mode,
    /// such operation will panic. In release mode, this is not checked but can cause undefined
    /// behavior.
    pub fn new(id: u64) -> Self {
        debug_assert!(id != u64::MAX, "cannot create 64-bit undefined id");
        Self(id)
    }
}

impl NodeIdAny for NodeId64 {
    fn undefined() -> Self {
        Self(u64::MAX)
    }

    fn zero() -> Self {
        Self(0)
    }

    fn one() -> Self {
        Self(1)
    }

    fn is_undefined(self) -> bool {
        self.0 == u64::MAX
    }

    fn is_zero(self) -> bool {
        self.0 == 0
    }

    fn is_one(self) -> bool {
        self.0 == 1
    }

    fn is_terminal(self) -> bool {
        self.0 <= 1
    }

    fn as_usize(self) -> usize {
        debug_assert!(
            self.0 != u64::MAX,
            "cannot use 64-bit undefined id as index"
        );
        usize_is_at_least_64_bits(self.0)
    }

    #[allow(clippy::as_conversions)]
    fn as_u16(self) -> u16 {
        self.0 as u16
    }

    #[allow(clippy::as_conversions)]
    fn as_u32(self) -> u32 {
        self.0 as u32
    }
}

macro_rules! impl_from_node_to_int {
    ($NodeId:ident => $Int:ident) => {
        impl From<$NodeId> for $Int {
            fn from(id: $NodeId) -> Self {
                $Int::from(id.0)
            }
        }
    };
    ($NodeId:ident => $($Int:ident),+) => {
        $(impl_from_node_to_int!($NodeId => $Int);)+
    };
}

impl_from_node_to_int!(NodeId16 => u16, u32, u64, u128);
impl_from_node_to_int!(NodeId32 => u32, u64, u128);
impl_from_node_to_int!(NodeId64 => u64, u128);

macro_rules! impl_from {
    ($Small:ident => $Large:ident) => {
        impl From<$Small> for $Large {
            fn from(id: $Small) -> Self {
                Self::new(id.0.into())
            }
        }
    };
}

impl_from!(NodeId16 => NodeId32);
impl_from!(NodeId16 => NodeId64);
impl_from!(NodeId32 => NodeId64);

macro_rules! impl_try_from {
    ($Large:ident => $Small:ident) => {
        impl TryFrom<$Large> for $Small {
            type Error = TryFromIntError;

            fn try_from(id: $Large) -> Result<Self, Self::Error> {
                id.0.try_into().map(Self::new)
            }
        }
    };
}

impl_try_from!(NodeId64 => NodeId16);
impl_try_from!(NodeId64 => NodeId32);
impl_try_from!(NodeId32 => NodeId16);

macro_rules! impl_try_from_usize {
    ($NodeId:ident) => {
        impl TryFrom<usize> for $NodeId {
            type Error = TryFromIntError;

            fn try_from(value: usize) -> Result<Self, Self::Error> {
                value.try_into().map(Self::new)
            }
        }
    };
}

impl_try_from_usize!(NodeId16);
impl_try_from_usize!(NodeId32);
impl_try_from_usize!(NodeId64);

/// A trait for the ability to upcast to the node ID specified by the generic type.
pub trait AsNodeId<TNodeId: NodeIdAny>: NodeIdAny + Into<TNodeId> {}

impl AsNodeId<NodeId16> for NodeId16 {}
impl AsNodeId<NodeId32> for NodeId16 {}
impl AsNodeId<NodeId64> for NodeId16 {}
impl AsNodeId<NodeId32> for NodeId32 {}
impl AsNodeId<NodeId64> for NodeId32 {}
impl AsNodeId<NodeId64> for NodeId64 {}

#[cfg(test)]
mod tests {
    use crate::boolean_operators::TriBool;
    use crate::node_id::{NodeId16, NodeId32, NodeId64, NodeIdAny};

    fn test_node_invariants<Node: NodeIdAny>() {
        assert!(Node::undefined().is_undefined());
        assert!(Node::zero().is_zero());
        assert!(Node::one().is_one());

        assert!(!Node::undefined().is_zero());
        assert!(!Node::one().is_zero());

        assert!(!Node::undefined().is_one());
        assert!(!Node::zero().is_one());

        assert!(!Node::zero().is_undefined());
        assert!(!Node::one().is_undefined());

        assert!(!Node::undefined().is_terminal());
        assert!(Node::one().is_terminal());
        assert!(Node::zero().is_terminal());

        assert_ne!(Node::undefined(), Node::one());
        assert_ne!(Node::undefined(), Node::zero());

        assert!(Node::from_three_valued(TriBool::True).is_one());
        assert!(Node::from_three_valued(TriBool::False).is_zero());
        assert!(Node::from_three_valued(TriBool::Indeterminate).is_undefined());

        assert_eq!(Node::undefined().to_three_valued(), TriBool::Indeterminate);
        assert_eq!(Node::one().to_three_valued(), TriBool::True);
        assert_eq!(Node::zero().to_three_valued(), TriBool::False);
    }

    fn test_node_invalid_as_usize<Node: NodeIdAny>() {
        Node::undefined().as_usize();
    }

    #[test]
    pub fn test_node_id_16_invariants() {
        test_node_invariants::<NodeId16>();
    }

    #[test]
    #[should_panic]
    pub fn test_node_id_16_as_usize() {
        test_node_invalid_as_usize::<NodeId16>()
    }

    #[test]
    pub fn test_node_id_32_invariants() {
        test_node_invariants::<NodeId32>();
    }

    #[test]
    #[should_panic]
    pub fn test_node_id_32_as_usize() {
        test_node_invalid_as_usize::<NodeId32>()
    }

    #[test]
    pub fn test_node_id_64_invariants() {
        test_node_invariants::<NodeId64>();
    }

    #[test]
    #[should_panic]
    pub fn test_node_id_64_as_usize() {
        test_node_invalid_as_usize::<NodeId64>()
    }

    #[test]
    #[should_panic]
    pub fn test_node_32_invalid_new() {
        NodeId32::new(NodeId32::MAX_ID + 1);
    }

    #[test]
    pub fn test_node_32_bytes() {
        let x = NodeId32::new(10);
        assert_eq!(NodeId32::from_le_bytes(x.to_le_bytes()), x);
    }
}
