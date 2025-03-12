//! Defines the representation of node identifiers. Includes: [`NodeIdAny`], [`NodeId16`],
//! [`NodeId32`] and [`NodeId64`].

use crate::boolean_operators::TriBool;
use crate::conversion::{UncheckedFrom, UncheckedInto};
use std::fmt::{self, Debug};
use std::hash::Hash;

/// An internal trait implemented by types that can serve as BDD node identifiers. The core
/// property of this trait is that a node ID must have one designated "undefined" value
/// (similar to `Option::None` and equivalent to the maximal representable value), and two
/// designated "terminal" values equivalent to `0` and `1`. Every other node id must
/// fall in the `1 < id < undefined` interval.
pub trait NodeIdAny:
    Eq
    + Ord
    + Copy
    + Hash
    + Debug
    + TryFrom<usize>
    + UncheckedFrom<usize>
    + UncheckedInto<u16>
    + UncheckedInto<u32>
    + UncheckedInto<u64>
    + UncheckedInto<u128>
    + UncheckedInto<usize>
    + UncheckedInto<NodeId>
    + UncheckedFrom<NodeId>
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

    /// Return the flipped terminal ID if the ID is terminal (`0` -> `1`, `1` -> `0`).
    fn flipped_if_terminal(self) -> Self {
        if self.is_zero() {
            Self::one()
        } else if self.is_one() {
            Self::zero()
        } else {
            self
        }
    }

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

    /// Convert the ID into a value that can be used for indexing. This can truncate IDs wider
    /// than the index type, but this should never happen on 64-bit systems.
    ///
    /// ## Undefined behavior
    ///
    /// The result is not defined for [`NodeIdAny::undefined`]. In debug mode, the method will
    /// panic. In release mode, the result is undefined behavior.
    fn as_usize(self) -> usize {
        self.unchecked_into()
    }
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
    /// A special value representing an undefined node ID.
    const UNDEFINED: u16 = u16::MAX;

    /// Create a new valid [`NodeId16`] from the underlying type `u16`.
    ///
    /// ## Undefined behavior
    ///
    /// This method should not be used to create instances of [`NodeId16::undefined`]. In debug
    /// mode, such operation will panic. In release mode, this is not checked but can cause
    /// undefined behavior.
    pub fn new(id: u16) -> Self {
        debug_assert!(id != Self::UNDEFINED, "cannot create 16-bit undefined id");
        Self(id)
    }
}

impl NodeIdAny for NodeId16 {
    fn undefined() -> Self {
        Self(Self::UNDEFINED)
    }

    fn zero() -> Self {
        Self(0)
    }

    fn one() -> Self {
        Self(1)
    }

    fn is_undefined(self) -> bool {
        self.0 == Self::UNDEFINED
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
}

impl NodeId32 {
    /// The largest ID representable by [`NodeId32`].
    pub const MAX_ID: u32 = u32::MAX - 1;
    /// A special value representing an undefined node ID.
    const UNDEFINED: u32 = u32::MAX;

    /// Create a new valid [`NodeId32`] from the underlying type `u32`.
    ///
    /// ## Undefined behavior
    ///
    /// This method should not be used to create instances of [`NodeId32::undefined`]. In debug
    /// mode, such operation will panic. In release mode, this is not checked but can cause
    /// undefined behavior.
    pub fn new(id: u32) -> Self {
        debug_assert!(id != Self::UNDEFINED, "cannot create 32-bit undefined id");
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
        Self(Self::UNDEFINED)
    }

    fn zero() -> Self {
        Self(0)
    }

    fn one() -> Self {
        Self(1)
    }

    fn is_undefined(self) -> bool {
        self.0 == Self::UNDEFINED
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
}

impl NodeId64 {
    /// The largest ID representable by [`NodeId64`].
    pub const MAX_ID: u64 = u64::MAX - 1;
    /// A special value representing an undefined node ID.
    const UNDEFINED: u64 = u64::MAX;

    /// Create a new valid [`NodeId64`] from the underlying type `u64`.
    ///
    /// ## Undefined behavior
    ///
    /// This method should not be used to create instances of [`NodeId64::undefined`]. In debug
    /// mode, such operation will panic. In release mode, this is not checked but can cause
    /// undefined behavior.
    pub fn new(id: u64) -> Self {
        debug_assert!(id != Self::UNDEFINED, "cannot create 64-bit undefined id");
        Self(id)
    }
}

impl NodeIdAny for NodeId64 {
    fn undefined() -> Self {
        Self(Self::UNDEFINED)
    }

    fn zero() -> Self {
        Self(0)
    }

    fn one() -> Self {
        Self(1)
    }

    fn is_undefined(self) -> bool {
        self.0 == Self::UNDEFINED
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
}

macro_rules! impl_from_node_to_int {
    ($NodeId:ident => $Int:ident) => {
        impl UncheckedFrom<$NodeId> for $Int {
            fn unchecked_from(id: $NodeId) -> Self {
                debug_assert!(!id.is_undefined());
                id.0.unchecked_into()
            }
        }
    };
    ($NodeId:ident => $($Int:ident),+) => {
        $(impl_from_node_to_int!($NodeId => $Int);)+
    };
}

impl_from_node_to_int!(NodeId16 => u16, u32, u64, u128, usize);
impl_from_node_to_int!(NodeId32 => u16, u32, u64, u128, usize);
impl_from_node_to_int!(NodeId64 => u16, u32, u64, u128, usize);

macro_rules! impl_from {
    ($Small:ident => $Large:ident) => {
        impl From<$Small> for $Large {
            fn from(id: $Small) -> Self {
                match id.0 {
                    $Small::UNDEFINED => Self::undefined(),
                    _ => Self(id.0.into()),
                }
            }
        }
    };
}

impl_from!(NodeId16 => NodeId32);
impl_from!(NodeId16 => NodeId64);
impl_from!(NodeId32 => NodeId64);

impl UncheckedFrom<NodeId32> for NodeId16 {
    #[allow(clippy::cast_possible_truncation)]
    fn unchecked_from(id: NodeId32) -> Self {
        Self(match id.0 {
            NodeId32::UNDEFINED => NodeId16::UNDEFINED,
            id => {
                debug_assert!(
                    id <= NodeId16::MAX_ID.into(),
                    "32-bit node ID {id} does not fit into 16-bit node ID"
                );
                id as u16
            }
        })
    }
}

impl UncheckedFrom<NodeId64> for NodeId16 {
    #[allow(clippy::cast_possible_truncation)]
    fn unchecked_from(id: NodeId64) -> Self {
        Self(match id.0 {
            NodeId64::UNDEFINED => NodeId16::UNDEFINED,
            id => {
                debug_assert!(
                    id <= NodeId16::MAX_ID.into(),
                    "64-bit node ID {id} does not fit into 16-bit node ID"
                );
                id as u16
            }
        })
    }
}

impl UncheckedFrom<NodeId64> for NodeId32 {
    #[allow(clippy::cast_possible_truncation)]
    fn unchecked_from(id: NodeId64) -> Self {
        Self(match id.0 {
            NodeId64::UNDEFINED => NodeId32::UNDEFINED,
            id => {
                debug_assert!(
                    id <= NodeId32::MAX_ID.into(),
                    "64-bit node ID {id} does not fit into 32-bit node ID"
                );
                id as u32
            }
        })
    }
}

/// An implementation of [`std::error::Error`] that is reported when conversion
/// between instances of [`NodeIdAny`] is not possible.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct TryFromNodeIdError {
    id: u64,
    from_width: usize,
    to_width: usize,
}

impl fmt::Display for TryFromNodeIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}-bit node ID {} cannot be converted to {}-bit",
            self.from_width, self.id, self.to_width
        )
    }
}

impl std::error::Error for TryFromNodeIdError {}

macro_rules! impl_try_from {
    ($Large:ident => $Small:ident) => {
        impl TryFrom<$Large> for $Small {
            type Error = TryFromNodeIdError;

            fn try_from(id: $Large) -> Result<Self, Self::Error> {
                if id.is_undefined() {
                    return Ok(Self::undefined());
                }

                let id_num: u64 = id.unchecked_into();
                match id.0.try_into() {
                    Ok(value) => Ok(Self(value)),
                    Err(_) => Err(TryFromNodeIdError {
                        id: id_num,
                        from_width: std::mem::size_of::<$Large>() * 8,
                        to_width: std::mem::size_of::<$Small>() * 8,
                    }),
                }
            }
        }
    };
}

impl_try_from!(NodeId64 => NodeId16);
impl_try_from!(NodeId64 => NodeId32);
impl_try_from!(NodeId32 => NodeId16);

/// An implementation of [`std::error::Error`] that is reported when an instance of
/// [`NodeIdAny`] cannot be created from a value of type `usize`.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct TryFromUsizeError {
    id: usize,
    to_width: usize,
}

impl fmt::Display for TryFromUsizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot convert usize {} to {}-bit node ID",
            self.id, self.to_width
        )
    }
}

macro_rules! impl_try_from_usize {
    ($NodeId:ident) => {
        impl TryFrom<usize> for $NodeId {
            type Error = TryFromUsizeError;

            fn try_from(value: usize) -> Result<Self, Self::Error> {
                if let Ok(value) = value.try_into() {
                    if value <= Self::MAX_ID {
                        return Ok(Self(value));
                    }
                }

                // At this point, the value is either undefined, or it does not fit
                // into the requested node ID type.
                Err(TryFromUsizeError {
                    id: value,
                    to_width: std::mem::size_of::<$NodeId>() * 8,
                })
            }
        }
    };
}

impl_try_from_usize!(NodeId16);
impl_try_from_usize!(NodeId32);
impl_try_from_usize!(NodeId64);

macro_rules! impl_unchecked_from_usize {
    ($NodeId:ident, $width:ident) => {
        #[allow(clippy::cast_possible_truncation)]
        impl UncheckedFrom<usize> for $NodeId {
            fn unchecked_from(value: usize) -> Self {
                debug_assert!($NodeId::try_from(value).is_ok());
                $NodeId::new(value as $width)
            }
        }
    };
}

impl_unchecked_from_usize!(NodeId16, u16);
impl_unchecked_from_usize!(NodeId32, u32);
impl_unchecked_from_usize!(NodeId64, u64);

/// A trait that ensures that the type implements both [`NodeIdAny`] and `Into<TNodeId>`.
///
/// This mainly allows us to write `AsNodeId<T>` instead of needing to write
/// `NodeIdAny + Into<T>` everywhere.
pub trait AsNodeId<TNodeId: NodeIdAny>: NodeIdAny + Into<TNodeId> {}
impl<A: NodeIdAny, B: NodeIdAny + Into<A>> AsNodeId<A> for B {}

impl fmt::Display for NodeId16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for NodeId32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for NodeId64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A type for identifying nodes in BDDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeId(u64);

impl NodeId {
    /// Create a new `NodeId` representing the terminal node `0`.
    pub(crate) fn zero() -> Self {
        Self(0)
    }

    /// Create a new `NodeId` representing the terminal node `1`.
    pub(crate) fn one() -> Self {
        Self(1)
    }

    /// Create a new undefined `NodeId` (similar to `Option::None`).
    ///
    /// We don't expect the undefined id to be converted from or into [`NodeId16`],
    /// [`NodeId32`] or [`NodeId64`]. This function is therefore the only way to create
    /// an undefined `NodeId`.
    pub(crate) fn undefined() -> Self {
        Self(u64::MAX)
    }

    /// Check if the ID is [`NodeId::undefined`].
    pub(crate) fn is_undefined(self) -> bool {
        self.0 == u64::MAX
    }

    /// Returns `true` if the ID is representable by a [`NodeId16`].
    pub(crate) fn fits_in_node_id16(self) -> bool {
        self.0 <= NodeId16::MAX_ID.into()
    }

    /// Returns `true` if the ID is representable by a [`NodeId32`].
    pub(crate) fn fits_in_node_id32(self) -> bool {
        self.0 <= NodeId32::MAX_ID.into()
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl UncheckedFrom<NodeId> for NodeId16 {
    #[allow(clippy::cast_possible_truncation)]
    fn unchecked_from(value: NodeId) -> NodeId16 {
        debug_assert!(!value.is_undefined());
        debug_assert!(
            value.fits_in_node_id16(),
            "node ID {value} does not fit into 16-bit node ID"
        );
        NodeId16(value.0 as u16)
    }
}

impl UncheckedFrom<NodeId> for NodeId32 {
    #[allow(clippy::cast_possible_truncation)]
    fn unchecked_from(value: NodeId) -> NodeId32 {
        debug_assert!(!value.is_undefined());
        debug_assert!(
            value.fits_in_node_id32(),
            "node ID {value} does not fit into 32-bit node ID"
        );
        NodeId32(value.0 as u32)
    }
}

impl UncheckedFrom<NodeId> for NodeId64 {
    fn unchecked_from(value: NodeId) -> NodeId64 {
        debug_assert!(!value.is_undefined());
        NodeId64(value.0)
    }
}

impl UncheckedFrom<NodeId16> for NodeId {
    fn unchecked_from(value: NodeId16) -> NodeId {
        debug_assert!(!value.is_undefined());
        NodeId(u64::from(value.0))
    }
}

impl UncheckedFrom<NodeId32> for NodeId {
    fn unchecked_from(value: NodeId32) -> NodeId {
        debug_assert!(!value.is_undefined());
        NodeId(u64::from(value.0))
    }
}

impl UncheckedFrom<NodeId64> for NodeId {
    fn unchecked_from(value: NodeId64) -> NodeId {
        debug_assert!(!value.is_undefined());
        NodeId(value.0)
    }
}

#[cfg(test)]
mod tests {

    use crate::boolean_operators::TriBool;
    use crate::conversion::{UncheckedFrom, UncheckedInto};
    use crate::node_id::{NodeId16, NodeId32, NodeId64, NodeIdAny};
    use crate::{usize_is_at_least_32_bits, usize_is_at_least_64_bits};

    use super::NodeId;

    fn node_invariants<Node: NodeIdAny>() {
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

    fn node_invalid_as_usize<Node: NodeIdAny>() {
        Node::undefined().as_usize();
    }

    #[test]
    pub fn node_id_16_invariants() {
        node_invariants::<NodeId16>();
    }

    #[test]
    #[should_panic]
    pub fn node_id_16_as_usize() {
        node_invalid_as_usize::<NodeId16>()
    }

    #[test]
    pub fn node_id_32_invariants() {
        node_invariants::<NodeId32>();
    }

    #[test]
    #[should_panic]
    pub fn node_id_32_as_usize() {
        node_invalid_as_usize::<NodeId32>()
    }

    #[test]
    pub fn node_id_64_invariants() {
        node_invariants::<NodeId64>();
    }

    #[test]
    #[should_panic]
    pub fn node_id_64_as_usize() {
        node_invalid_as_usize::<NodeId64>()
    }

    #[test]
    #[should_panic]
    pub fn node_32_invalid_new() {
        NodeId32::new(NodeId32::MAX_ID + 1);
    }

    #[test]
    pub fn node_32_bytes() {
        let x = NodeId32::new(10);
        assert_eq!(NodeId32::from_le_bytes(x.to_le_bytes()), x);
    }

    macro_rules! test_node_id_from {
        ($Small:ident => $Large:ident, $func:ident) => {
            #[test]
            fn $func() {
                assert_eq!($Large::one(), $Large::from($Small::one()));
                assert_eq!($Large::zero(), $Large::from($Small::zero()));
                assert_eq!($Large::new(256), $Large::from($Small::new(256)));
            }
        };
    }

    test_node_id_from!(NodeId16 => NodeId32, node_id_32_from_16);
    test_node_id_from!(NodeId16 => NodeId64, node_id_64_from_16);
    test_node_id_from!(NodeId32 => NodeId64, node_id_64_from_32);

    macro_rules! test_node_id_from_undefined {
        ($Small:ident => $Large:ident, $func:ident) => {
            #[test]
            fn $func() {
                assert_eq!($Large::undefined(), $Large::from($Small::undefined()));
            }
        };
    }

    test_node_id_from_undefined!(NodeId16 => NodeId32, node_id_32_from_16_undefined);
    test_node_id_from_undefined!(NodeId16 => NodeId64, node_id_64_from_16_undefined);
    test_node_id_from_undefined!(NodeId32 => NodeId64, node_id_64_from_32_undefined);

    macro_rules! test_node_id_unchecked_from_undefined {
        ($Large:ident => $Small:ident, $func:ident) => {
            #[test]
            fn $func() {
                let large_undefined = $Large::undefined();
                let small_undefined = $Small::undefined();

                assert_eq!(small_undefined, $Small::unchecked_from(large_undefined));
            }
        };
    }

    test_node_id_unchecked_from_undefined!(NodeId32 => NodeId16, node_id_16_unchecked_from_32_undefined);
    test_node_id_unchecked_from_undefined!(NodeId64 => NodeId16, node_id_16_unchecked_from_64_undefined);
    test_node_id_unchecked_from_undefined!(NodeId64 => NodeId32, node_id_32_unchecked_from_16_undefined);

    macro_rules! test_node_id_unchecked_from {
        ($Large:ident => $Small:ident, $func:ident) => {
            #[test]
            fn $func() {
                assert_eq!($Small::zero(), $Small::unchecked_from($Large::zero()));
                assert_eq!($Small::one(), $Small::unchecked_from($Large::one()));
                assert_eq!($Small::new(256), $Small::unchecked_from($Large::new(256)));
            }
        };
    }

    test_node_id_unchecked_from!(NodeId64 => NodeId16, node_id_16_unchecked_from_64);
    test_node_id_unchecked_from!(NodeId64 => NodeId32, node_id_32_unchecked_from_64);
    test_node_id_unchecked_from!(NodeId32 => NodeId16, node_id_16_unchecked_from_32);

    macro_rules! test_node_id_unchecked_from_invalid {
        ($Large:ident => $Small:ident, $LargeWidth:ident, $func:ident) => {
            #[test]
            #[should_panic]
            fn $func() {
                let large = $Large::new($LargeWidth::from($Small::MAX_ID) + 2);
                let _ = $Small::unchecked_from(large);
            }
        };
    }

    test_node_id_unchecked_from_invalid!(NodeId64 => NodeId16, u64, node_id_16_unchecked_from_64_invalid);
    test_node_id_unchecked_from_invalid!(NodeId64 => NodeId32, u64, node_id_32_unchecked_from_64_invalid);
    test_node_id_unchecked_from_invalid!(NodeId32 => NodeId16, u32, node_id_16_unchecked_from_32_invalid);

    macro_rules! test_node_id_try_from_undefined {
        ($Large:ident => $Small:ident, $func:ident) => {
            #[test]
            fn $func() {
                let large_undefined = $Large::undefined();
                let small_undefined = $Small::undefined();

                assert_eq!(small_undefined, $Small::try_from(large_undefined).unwrap());
            }
        };
    }

    test_node_id_try_from_undefined!(NodeId32 => NodeId16, node_id_16_try_from_32_undefined);
    test_node_id_try_from_undefined!(NodeId64 => NodeId16, node_id_16_try_from_64_undefined);
    test_node_id_try_from_undefined!(NodeId64 => NodeId32, node_id_32_try_from_64_undefined);

    macro_rules! test_node_id_try_from {
        ($Large:ident => $Small:ident, $func:ident) => {
            #[test]
            fn $func() {
                assert_eq!($Small::zero(), $Small::try_from($Large::zero()).unwrap());
                assert_eq!($Small::one(), $Small::try_from($Large::one()).unwrap());
                assert_eq!(
                    $Small::new(256),
                    $Small::try_from($Large::new(256)).unwrap()
                );
            }
        };
    }

    test_node_id_try_from!(NodeId64 => NodeId16, node_id_16_try_from_64);
    test_node_id_try_from!(NodeId64 => NodeId32, node_id_32_try_from_64);
    test_node_id_try_from!(NodeId32 => NodeId16, node_id_16_try_from_32);

    macro_rules! test_node_id_try_from_invalid {
        ($Large:ident => $Small:ident, $LargeWidth:ident, $func:ident) => {
            #[test]
            #[should_panic]
            fn $func() {
                let large = $Large::new($LargeWidth::from($Small::MAX_ID) + 2);
                let _ = $Small::try_from(large).unwrap();
            }
        };
    }

    test_node_id_try_from_invalid!(NodeId64 => NodeId16, u64, node_id_16_try_from_64_invalid);
    test_node_id_try_from_invalid!(NodeId64 => NodeId32, u64, node_id_32_try_from_64_invalid);
    test_node_id_try_from_invalid!(NodeId32 => NodeId16, u32, node_id_16_try_from_32_invalid);

    #[test]
    #[should_panic]
    fn node_id_16_try_from_usize_invalid() {
        let m = usize::from(NodeId16::MAX_ID) + 1;
        let _ = NodeId16::try_from(m).unwrap();
    }

    #[test]
    #[should_panic]
    fn node_id_32_try_from_usize_invalid() {
        let m = usize_is_at_least_32_bits(NodeId32::MAX_ID) + 1;
        let _ = NodeId32::try_from(m).unwrap();
    }

    #[test]
    #[should_panic]
    fn node_id_64_try_from_usize_invalid() {
        let m = usize_is_at_least_64_bits(NodeId64::MAX_ID) + 1;
        let _ = NodeId64::try_from(m).unwrap();
    }

    #[test]
    #[should_panic]
    fn node_id_16_unchecked_from_usize_invalid() {
        let m = usize::from(NodeId16::MAX_ID) + 1;
        let _ = NodeId16::unchecked_from(m);
    }

    #[test]
    #[should_panic]
    fn node_id_32_unchecked_from_usize_invalid() {
        let m = usize_is_at_least_32_bits(NodeId32::MAX_ID) + 1;
        let _ = NodeId32::unchecked_from(m);
    }

    #[test]
    #[should_panic]
    fn node_id_64_unchecked_from_usize_invalid() {
        let m = usize_is_at_least_64_bits(NodeId64::MAX_ID) + 1;
        let _ = NodeId64::unchecked_from(m);
    }

    #[test]
    fn node_id_display() {
        let x16 = NodeId16::new(1234);
        let x32 = NodeId32::new(1234);
        let x64 = NodeId64::new(1234);

        assert_eq!(x16.to_string(), x32.to_string());
        assert_eq!(x32.to_string(), x64.to_string());
        assert_eq!(x64.to_string(), x16.to_string());
    }

    #[test]
    fn node_id_try_from_conversion() {
        let id = NodeId32::new((u16::MAX as u32) + 1);
        let err = NodeId16::try_from(id).unwrap_err();
        println!("{}", err);
        assert_eq!(err.from_width, 32);
        assert_eq!(err.to_width, 16);

        let size_id = (u16::MAX as usize) + 1;
        let err = NodeId16::try_from(size_id).unwrap_err();
        println!("{}", err);
        assert_eq!(err.id, size_id);
        assert_eq!(err.to_width, 16);
    }

    #[test]
    fn node_id_flip() {
        let zero = NodeId16::zero();
        let one = NodeId16::one();
        let two = NodeId16::new(2);
        assert_eq!(zero.flipped_if_terminal(), one);
        assert_eq!(one.flipped_if_terminal(), zero);
        assert_eq!(two.flipped_if_terminal(), two);
    }

    #[test]
    fn node_id_terminals() {
        let zero_16 = NodeId16::zero();
        let one_16 = NodeId16::one();
        let zero_32 = NodeId32::zero();
        let one_32 = NodeId32::one();
        let zero_64 = NodeId64::zero();
        let one_64 = NodeId64::one();

        let zero = NodeId::zero();
        let one = NodeId::one();

        assert_eq!(NodeId::unchecked_from(zero_16), zero);
        assert_eq!(NodeId16::unchecked_from(zero), zero_16);
        assert_eq!(NodeId::unchecked_from(one_16), one);
        assert_eq!(NodeId16::unchecked_from(one), one_16);

        assert_eq!(NodeId::unchecked_from(zero_32), zero);
        assert_eq!(NodeId32::unchecked_from(zero), zero_32);
        assert_eq!(NodeId::unchecked_from(one_32), one);
        assert_eq!(NodeId32::unchecked_from(one), one_32);

        assert_eq!(NodeId::unchecked_from(zero_64), zero);
        assert_eq!(NodeId64::unchecked_from(zero), zero_64);
        assert_eq!(NodeId::unchecked_from(one_64), one);
        assert_eq!(NodeId64::unchecked_from(one), one_64);
    }

    fn test_node_id_from_undefined<TNodeId: NodeIdAny>() {
        let undefined = TNodeId::undefined();
        let _: NodeId = undefined.unchecked_into();
    }

    #[test]
    #[should_panic]
    fn node_id_from_undefined_16() {
        test_node_id_from_undefined::<NodeId16>();
    }

    #[test]
    #[should_panic]
    fn node_id_from_undefined_32() {
        test_node_id_from_undefined::<NodeId32>();
    }

    #[test]
    #[should_panic]
    fn node_id_from_undefined_64() {
        test_node_id_from_undefined::<NodeId64>();
    }

    macro_rules! test_node_id_from_defined_into_undefined {
        ($FromUndefinedId:ident => $ToUndefinedId:ident, $func:ident) => {
            #[test]
            #[should_panic]
            fn $func() {
                let x = $FromUndefinedId::new($ToUndefinedId::UNDEFINED.into());
                let id: NodeId = x.unchecked_into();
                let _: $ToUndefinedId = id.unchecked_into();
            }
        };
    }

    test_node_id_from_defined_into_undefined!(NodeId32 => NodeId16, node_id_from_defined_32_into_undefined_16);
    test_node_id_from_defined_into_undefined!(NodeId64 => NodeId16, node_id_from_defined_64_into_undefined_16);
    test_node_id_from_defined_into_undefined!(NodeId64 => NodeId32, node_id_from_defined_64_into_undefined_32);

    #[test]
    fn node_id_bounds() {
        let id16 = NodeId16::new(NodeId16::MAX_ID);
        let id: NodeId = id16.unchecked_into();

        assert!(id.fits_in_node_id16());
        assert!(id.fits_in_node_id32());

        let id32 = NodeId32::new(NodeId32::MAX_ID);
        let id: NodeId = id32.unchecked_into();

        assert!(!id.fits_in_node_id16());
        assert!(id.fits_in_node_id32());

        let id64 = NodeId64::new(NodeId64::MAX_ID);
        let id: NodeId = id64.unchecked_into();

        assert!(!id.fits_in_node_id16());
        assert!(!id.fits_in_node_id32());
    }

    #[test]
    fn node_id_conversion_from_64_to_16() {
        let id64 = NodeId64::new(NodeId16::MAX_ID as u64);
        let id: NodeId = id64.unchecked_into();
        let id16: NodeId16 = id.unchecked_into();
        assert_eq!(id16, NodeId16::new(NodeId16::MAX_ID));
    }

    #[test]
    fn node_id_conversion_from_64_to_32() {
        let id64 = NodeId64::new(NodeId32::MAX_ID as u64);
        let id: NodeId = id64.unchecked_into();
        let id32: NodeId32 = id.unchecked_into();
        assert_eq!(id32, NodeId32::new(NodeId32::MAX_ID));
    }

    #[test]
    fn node_id_conversion_from_32_to_16() {
        let id32 = NodeId32::new(NodeId16::MAX_ID as u32);
        let id: NodeId = id32.unchecked_into();
        let id16: NodeId16 = id.unchecked_into();
        assert_eq!(id16, NodeId16::new(NodeId16::MAX_ID));
    }

    #[test]
    #[should_panic]
    fn node_id_unsuccessful_conversion_from_64_to_16() {
        let id64 = NodeId64::new(NodeId64::MAX_ID);
        let id: NodeId = id64.unchecked_into();
        let _: NodeId16 = id.unchecked_into();
    }

    #[test]
    #[should_panic]
    fn node_id_unsuccessful_conversion_from_64_to_32() {
        let id64 = NodeId64::new(NodeId64::MAX_ID);
        let id: NodeId = id64.unchecked_into();
        let _: NodeId32 = id.unchecked_into();
    }

    #[test]
    #[should_panic]
    fn node_id_unsuccessful_conversion_from_32_to_16() {
        let id32 = NodeId32::new(NodeId32::MAX_ID);
        let id: NodeId = id32.unchecked_into();
        let _: NodeId16 = id.unchecked_into();
    }
}
