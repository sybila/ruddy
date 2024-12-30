//! Defines the representation of node identifiers. Includes: [`BddNodeId`], [`NodeId16`],
//! [`NodeId32`] and [`NodeId64`].

use crate::{boolean_operators::TriBool, usize_is_at_least_32_bits, usize_is_at_least_64_bits};
use std::fmt::Debug;
use std::hash::Hash;

/// An internal trait implemented by types that can serve as BDD node identifiers. The core
/// property of this trait is that a node ID must have one designated "undefined" value
/// (similar to `Option::None` and equivalent to the maximal representable value), and two
/// designated "terminal" values equivalent to `0` and `1`. Every other node id must
/// fall in the `1 < id < undefined` interval.
pub trait BddNodeId: Eq + Ord + Copy + Hash + Debug {
    /// Return an instance of the "undefined" node ID.
    fn undefined() -> Self;
    /// Return an instance of the zero node ID.
    fn zero() -> Self;
    /// Return an instance of the one node ID.
    fn one() -> Self;

    /// Checks if this ID is [`BddNodeId::undefined`].
    fn is_undefined(self) -> bool;
    /// Checks if this ID is [`BddNodeId::zero`].
    fn is_zero(self) -> bool;
    /// Checks if this ID is [`BddNodeId::one`].
    fn is_one(self) -> bool;
    /// Checks if this ID is [`BddNodeId::zero`] or [`BddNodeId::one`].
    fn is_terminal(self) -> bool;

    /// Convert the ID safely into a value that can be used for indexing.
    ///
    /// ## Undefined behavior
    ///
    /// The result is not defined for [`BddNodeId::undefined`]. In debug mode, the method will panic.
    /// In release mode, the result is undefined behavior.
    fn as_usize(&self) -> usize;

    /// Convert the ID into a [`TriBool`], where the terminal node 0 is mapped to `False`,
    /// the terminal node 1 is mapped to `True`, and all other nodes are mapped to `Indeterminate`.
    fn to_three_valued(&self) -> TriBool {
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
}

/// Implementation of [`BddNodeId`] backed by `u16`. The maximal ID is `2**16 - 1`.
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct NodeId16(u16);

/// Implementation of [`BddNodeId`] backed by `u32`. The maximal ID is `2**32 - 1`.
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct NodeId32(u32);

/// Implementation of [`BddNodeId`] backed by `u64`. The maximal ID is `2**64 - 1`.
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct NodeId64(u64);

impl BddNodeId for NodeId16 {
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

    fn as_usize(&self) -> usize {
        debug_assert!(self.0 != u16::MAX, "Cannot use undefined as index");
        usize::from(self.0)
    }
}

impl NodeId32 {
    /// The largest ID representable by [`NodeId32`].
    pub const MAX_ID: u32 = u32::MAX - 1;

    /// Create a new valid [`NodeId32`] from an integer.
    ///
    /// ## Undefined behavior
    ///
    /// This method should not be used to create instances of [`NodeId32::undefined`]. In debug mode,
    /// such operation will panic. In release mode, this is not checked but can cause undefined
    /// behavior.
    pub fn new(id: u32) -> Self {
        debug_assert!(id != u32::MAX, "Cannot create undefined");
        Self(id)
    }

    /// Convert the underlying ID to `u64` (infallible).
    pub fn as_u64(self) -> u64 {
        u64::from(self.0)
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

impl BddNodeId for NodeId32 {
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

    fn as_usize(&self) -> usize {
        debug_assert!(self.0 != u32::MAX, "Cannot use undefined as index");
        usize_is_at_least_32_bits(self.0)
    }
}

impl BddNodeId for NodeId64 {
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

    fn as_usize(&self) -> usize {
        debug_assert!(self.0 != u64::MAX, "Cannot use undefined as index");
        usize_is_at_least_64_bits(self.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::boolean_operators::TriBool;
    use crate::node_id::{BddNodeId, NodeId16, NodeId32, NodeId64};

    fn test_node_invariants<Node: BddNodeId>() {
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

    fn test_node_invalid_unpack<Node: BddNodeId>() {
        Node::undefined().as_usize();
    }

    #[test]
    pub fn test_node_16_invariants() {
        test_node_invariants::<NodeId16>();
    }

    #[test]
    #[should_panic]
    pub fn test_node_16_unpack() {
        test_node_invalid_unpack::<NodeId16>()
    }

    #[test]
    pub fn test_node_32_invariants() {
        test_node_invariants::<NodeId32>();
    }

    #[test]
    #[should_panic]
    pub fn test_node_32_unpack() {
        test_node_invalid_unpack::<NodeId32>()
    }

    #[test]
    pub fn test_node_64_invariants() {
        test_node_invariants::<NodeId64>();
    }

    #[test]
    #[should_panic]
    pub fn test_node_64_unpack() {
        test_node_invalid_unpack::<NodeId64>()
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
