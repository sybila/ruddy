use crate::{boolean_operators::TriBool, usize_is_at_least_32_bits, usize_is_at_least_64_bits};
use std::hash::Hash;

/// An internal trait implemented by types that can serve as BDD node identifiers. The core feature
/// of this trait is that a node ID must have one designated "undefined" value (similar to
/// `Option::None`), and two designated "terminal" values equivalent to `0` and `1`.
///
/// TODO: We might need more methods for converting between IDs and integers?
pub trait BddNodeId: Eq + Ord + Copy + Hash {
    /// Return an instance of the "undefined" node ID.
    fn undefined() -> Self;
    /// Return an instance of the zero node ID.
    fn zero() -> Self;
    /// Return an instance of the one node ID.
    fn one() -> Self;

    fn is_undefined(self) -> bool;
    fn is_zero(self) -> bool;
    fn is_one(self) -> bool;
    fn is_terminal(self) -> bool;

    /// Convert the ID safely into a value that can be used for indexing.
    ///
    /// This method should panic if the ID is undefined, but only in debug mode (similar to
    /// how overflow checks work in Rust).
    fn as_usize(&self) -> usize;

    /// Convert the ID into a [TriBool], where the terminal node 0 is mapped to `False`,
    /// the terminal node 1 is mapped to `True`, and all other nodes are mapped to `Indeterminate`.
    fn to_three_valued(&self) -> TriBool {
        // Decompiles to branchless, nice code
        // 0 -> -1, 1 -> 1, _ -> 0
        match -i8::from(self.is_zero()) + i8::from(self.is_one()) {
            1 => TriBool::True,
            0 => TriBool::Indeterminate,
            -1 => TriBool::False,
            _ => unreachable!(),
        }
    }

    /// Convert a [TriBool] to a node ID, if possible. The value
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

#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct NodeId16(u16);

#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct NodeId32(u32);

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
        usize::from(self.0)
    }
}

impl NodeId32 {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn as_u64(self) -> u64 {
        u64::from(self.0)
    }

    pub fn to_le_bytes(self) -> [u8; 4] {
        self.0.to_le_bytes()
    }

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
        usize_is_at_least_64_bits(self.0)
    }
}
