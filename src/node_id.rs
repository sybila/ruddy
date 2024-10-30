use crate::{usize_is_at_least_32_bits, usize_is_at_least_64_bits};
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
