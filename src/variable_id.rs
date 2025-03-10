//! Defines the representation of variable identifiers. Includes: [`VarIdPackedAny`],
//! [`VarIdPacked16`], [`VarIdPacked32`], and [`VarIdPacked64`].

use std::{
    convert::TryFrom,
    fmt::{self, Debug},
    hash::Hash,
};

use crate::conversion::{UncheckedFrom, UncheckedInto};

/// An internal trait implemented by types that can serve as BDD variable identifiers within
/// BDD nodes.
///
/// The trait has several features that differentiate it from a simple numeric ID:
///
///  - The maximum representable value of each ID type serves as an "undefined" value
///    (similar to `Option::None`).
///  - Each ID type has a "should use cache" flag (implemented by setting a particular
///    bit of the identifier), which can be used to control whether the associated BDD
///    node should be entered into the task cache during the `apply` algorithm.
///  - Each ID type has a two-bit (`0`, `1`, `many`) counter tracking the number of parent
///    nodes of the associated BDD node (if any). This is also used to guide heuristics
///    within the `apply` algorithm.
///
/// Note that the last two features are primarily used to speed up the `apply` algorithm and
/// have no meaning outside of it.
pub trait VarIdPackedAny: Copy + Clone + PartialEq + Eq + PartialOrd + Ord + Hash + Debug {
    /// Return an instance of the "undefined" variable ID.
    fn undefined() -> Self;
    /// Checks if this ID is [`VarIdPackedAny::undefined`].
    fn is_undefined(self) -> bool;

    /// Unpack the packed variable ID safely into a `u64`, giving the "true" variable ID without
    /// the additional information.
    ///
    /// ## Undefined behavior
    ///
    /// For [`VarIdPackedAny::undefined`], the result is not defined. *In debug mode, the method
    /// will panic.* In release mode, unpacking an undefined value results in undefined behavior.
    fn unpack_u64(self) -> u64;

    /// Returns true if the internal parent counter is set to `many` (i.e. not `0` or `1`).
    ///
    /// For [`VarIdPackedAny::undefined`], the behavior is undefined, but unchecked.
    fn has_many_parents(self) -> bool;

    /// Check if the "use cache" flag is set on this variable ID.
    ///
    /// For [`VarIdPackedAny::undefined`], the behavior is undefined, but unchecked.
    fn use_cache(self) -> bool;

    /// Update the "use cache" flag of this variable ID.
    ///
    /// For [`VarIdPackedAny::undefined`], the behavior is undefined, but unchecked.
    fn set_use_cache(&mut self, value: bool);

    /// Increment the parent counter, assuming it is not already set to `many` (in that case,
    /// the counter stays the same).
    ///
    /// For [`VarIdPackedAny::undefined`], the behavior is undefined, but unchecked.
    fn increment_parents(&mut self);
}

/// A 16-bit implementation of the [`VarIdPackedAny`] trait that packs additional
/// information about the node containing the variable into the variable ID
/// to make the apply algorithm more efficient.
///
/// This means that [`VarIdPacked16`] can only represent `2**13 - 1` unique variables (see also
/// [`VarIdPacked16::MAX_ID`]).
///
/// The packed information is as follows:
///  - Two least-significant bits are used as a `{0, 1, many}` counter that keeps track of how
///    many parents the node containing the variable has.
///  - Third least-significant bit is used to indicate if the node containing the variable should
///    use the task cache in the apply algorithm.
///
/// Note that the "packed metadata" is ignored when comparing variable IDs using `Eq`,
/// `Ord` or `Hash`.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct VarIdPacked16(u16);

impl VarIdPacked16 {
    /// The largest variable ID that can be safely represented by [`VarIdPacked16`].
    pub const MAX_ID: u16 = (u16::MAX >> 3) - 1;
    /// An internal "mask bit" that is used when manipulating the "use cache"
    /// flag in packed variable IDs.
    const USE_CACHE_MASK: u16 = 0b100;
    /// An internal mask that can be used to reset the "packed information".
    const RESET_MASK: u16 = !0b111;
    /// A special value that represents an "undefined" variable ID.
    const UNDEFINED: u16 = u16::MAX;

    /// Create a new instance of [`VarIdPacked16`] with the specified variable ID. It must hold
    /// that `0 <= id <= MAX_ID`.
    ///
    /// The variable ID is shifted left by 3 bits to make room for the additional
    /// "packed" information.
    ///
    /// ## Undefined behavior
    ///
    /// For performance reasons, range checks on variable IDs are only performed in debug mode.
    /// In release mode, undefined behavior can occur if the ID is invalid.
    pub fn new(id: u16) -> Self {
        // `<<` should fail when overflowing in debug mode, but this is a bit easier to trace.
        debug_assert!(
            id <= Self::MAX_ID,
            "16-bit variable ID {id} not within representable range"
        );
        Self(id << 3)
    }

    /// Returns the same variable ID, but with the "parent counter" and "use cache" flag reset to
    /// their default state.
    pub(crate) fn reset(self) -> Self {
        Self(self.0 & Self::RESET_MASK)
    }

    /// Unpack the packed variable ID, giving the "true" variable ID without the
    /// additional information.
    ///
    /// ## Undefined behavior
    ///
    /// For [`VarIdPacked16::undefined`], the result is not defined. In debug mode, the method
    /// will fail. In release mode, unpacking an undefined value results in undefined behavior.
    pub fn unpack(self) -> u16 {
        debug_assert!(
            !self.is_undefined(),
            "cannot unpack undefined 16-bit variable ID"
        );
        self.0 >> 3
    }
}

/// A 32-bit implementation of the [`VarIdPackedAny`] trait that packs additional
/// information about the node containing the variable into the variable ID
/// to make the apply algorithm more efficient.
///
/// This means that [`VarIdPacked32`] can only represent `2**29 - 1` unique variables (see also
/// [`VarIdPacked32::MAX_ID`]).
///
/// The packed information is as follows:
///  - Two least-significant bits are used as a `{0, 1, many}` counter that keeps track of how
///    many parents the node containing the variable has.
///  - Third least-significant bit is used to indicate if the node containing the variable should
///    use the task cache in the apply algorithm.
///
/// Note that the "packed metadata" is ignored when comparing variable IDs using `Eq`,
/// `Ord` or `Hash`.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct VarIdPacked32(u32);

impl VarIdPacked32 {
    /// The largest variable ID that can be safely represented by [`VarIdPacked32`].
    pub const MAX_ID: u32 = (u32::MAX >> 3) - 1;
    /// An internal "mask bit" that is used when manipulating the "use cache"
    /// flag in packed variable IDs.
    const USE_CACHE_MASK: u32 = 0b100;
    /// An internal mask that can be used to reset the "packed information".
    const RESET_MASK: u32 = !0b111;
    /// A special value that represents an "undefined" variable ID.
    const UNDEFINED: u32 = u32::MAX;

    /// Create a new instance of [`VarIdPacked32`] with the specified variable ID. It must hold
    /// that `0 <= id <= MAX_ID`.
    ///
    /// The variable ID is shifted left by 3 bits to make room for the additional
    /// "packed" information.
    ///
    /// ## Undefined behavior
    ///
    /// For performance reasons, range checks on variable IDs are only performed in debug mode.
    /// In release mode, undefined behavior can occur if the ID is invalid.
    pub fn new(id: u32) -> Self {
        // `<<` should fail when overflowing in debug mode, but this is a bit easier to trace.
        debug_assert!(
            id <= Self::MAX_ID,
            "32-bit variable ID {id} not within representable range"
        );
        Self(id << 3)
    }

    /// Returns the same variable ID, but with the "parent counter" and "use cache" flag reset to
    /// their default state.
    pub(crate) fn reset(self) -> Self {
        Self(self.0 & Self::RESET_MASK)
    }

    /// Unpack the packed variable ID, giving the "true" variable ID without the
    /// additional information.
    ///
    /// ## Undefined behavior
    ///
    /// For [`VarIdPacked32::undefined`], the result is not defined. In debug mode, the method
    /// will fail. In release mode, unpacking an undefined value results in undefined behavior.
    pub fn unpack(self) -> u32 {
        debug_assert!(
            !self.is_undefined(),
            "cannot unpack undefined 32-bit variable ID"
        );
        self.0 >> 3
    }

    /// Check if the variable ID is representable by `VarIdPacked16`.
    pub(crate) fn fits_in_packed16(self) -> bool {
        self.is_undefined() || self.unpack() <= u32::from(VarIdPacked16::MAX_ID)
    }
}

/// A 64-bit implementation of the [`VarIdPackedAny`] trait that packs additional
/// information about the node containing the variable into the variable ID
/// to make the apply algorithm more efficient.
///
/// This means that [`VarIdPacked64`] can only represent `2**61 - 1` unique variables (see also
/// [`VarIdPacked64::MAX_ID`]).
///
/// The packed information is as follows:
///  - Two least-significant bits are used as a `{0, 1, many}` counter that keeps track of how
///    many parents the node containing the variable has.
///  - Third least-significant bit is used to indicate if the node containing the variable should
///    use the task cache in the apply algorithm.
///
/// Note that the "packed metadata" is ignored when comparing variable IDs using `Eq`,
/// `Ord` or `Hash`.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct VarIdPacked64(u64);

impl VarIdPacked64 {
    /// The largest variable ID that can be safely represented by [`VarIdPacked64`].
    pub const MAX_ID: u64 = (u64::MAX >> 3) - 1;
    /// An internal "mask bit" that is used when manipulating the "use cache"
    /// flag in packed variable IDs.
    const USE_CACHE_MASK: u64 = 0b100;
    /// An internal mask that can be used to reset the "packed information".
    const RESET_MASK: u64 = !0b111;
    /// A special value that represents an "undefined" variable ID.
    const UNDEFINED: u64 = u64::MAX;

    /// Create a new instance of [`VarIdPacked64`] with the specified variable ID. It must hold
    /// that `0 <= id <= MAX_ID`.
    ///
    /// The variable ID is shifted left by 3 bits to make room for the additional
    /// "packed" information.
    ///
    /// ## Undefined behavior
    ///
    /// For performance reasons, range checks on variable IDs are only performed in debug mode.
    /// In release mode, undefined behavior can occur if the ID is invalid.
    pub fn new(id: u64) -> Self {
        // `<<` should fail when overflowing in debug mode, but this is a bit easier to trace.
        debug_assert!(
            id <= Self::MAX_ID,
            "64-bit variable ID {id} not within representable range"
        );
        Self(id << 3)
    }

    /// Returns the same variable ID, but with the "parent counter" and "use cache" flag reset to
    /// their default state.
    pub(crate) fn reset(self) -> Self {
        Self(self.0 & Self::RESET_MASK)
    }

    /// Unpack the packed variable ID, giving the "true" variable ID without the
    /// additional information.
    ///
    /// ## Undefined behavior
    ///
    /// For [`VarIdPacked64::undefined`], the result is not defined. In debug mode, the method
    /// will fail. In release mode, unpacking an undefined value results in undefined behavior.
    pub fn unpack(self) -> u64 {
        debug_assert!(
            !self.is_undefined(),
            "cannot unpack undefined 64-bit variable ID"
        );
        self.0 >> 3
    }

    /// Check if the variable ID is representable by `VarIdPacked16`.
    pub(crate) fn fits_in_packed16(self) -> bool {
        self.is_undefined() || self.unpack() <= u64::from(VarIdPacked16::MAX_ID)
    }

    /// Check if the variable ID is representable by `VarIdPacked32`.
    pub(crate) fn fits_in_packed32(self) -> bool {
        self.is_undefined() || self.unpack() <= u64::from(VarIdPacked32::MAX_ID)
    }
}

macro_rules! impl_var_id_packed {
    ($name:ident, $width:ident) => {
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                // After xor, only different bits will be set. Since the "packed" data
                // is in the three least significant bits, any actual difference must result
                // in a number that requires more than three bits to represent.
                (self.0 ^ other.0) <= 0b111
            }
        }

        impl Eq for $name {}

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(Ord::cmp(&self, &other))
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.reset().0.cmp(&other.reset().0)
            }
        }

        impl Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.reset().0.hash(state)
            }
        }

        impl VarIdPackedAny for $name {
            fn undefined() -> Self {
                $name(Self::UNDEFINED)
            }

            fn is_undefined(self) -> bool {
                self.0 == Self::UNDEFINED
            }

            fn unpack_u64(self) -> u64 {
                u64::from(self.unpack())
            }

            fn has_many_parents(self) -> bool {
                self.0 & 0b10 != 0
            }

            fn use_cache(self) -> bool {
                self.0 & Self::USE_CACHE_MASK != 0
            }

            fn set_use_cache(&mut self, value: bool) {
                self.0 = (self.0 & !Self::USE_CACHE_MASK) | ($width::from(value) << 2);
            }

            fn increment_parents(&mut self) {
                // 00 -> 01 -> 00 | 01 -> 01
                // 01 -> 10 -> 01 | 10 -> 11
                // 10 -> 11 -> 10 | 11 -> 11
                // 11 -> 00 -> 11 | 00 -> 11
                let counter = self.0.wrapping_add(1) & 0b11;
                self.0 |= counter;
            }
        }
    };
}

impl_var_id_packed!(VarIdPacked16, u16);
impl_var_id_packed!(VarIdPacked32, u32);
impl_var_id_packed!(VarIdPacked64, u64);

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

impl_from!(VarIdPacked16 => VarIdPacked32);
impl_from!(VarIdPacked16 => VarIdPacked64);
impl_from!(VarIdPacked32 => VarIdPacked64);

impl UncheckedFrom<VarIdPacked64> for VarIdPacked16 {
    #[allow(clippy::cast_possible_truncation)]
    fn unchecked_from(id: VarIdPacked64) -> Self {
        debug_assert!(
            id.fits_in_packed16(),
            "64-bit variable ID {id} does not fit into 16-bit variable ID"
        );
        // Assuming the number fits into u16 (which is checked by the debug assert), it's safe
        // to just "forget" the top 48 bits (undefined value will stay undefined, rest maps
        // to valid 16-bit values). Note that we can't use `UncheckedInto` here, as that would
        // panic in debug mode for undefined values.
        VarIdPacked16(id.0 as u16)
    }
}

impl UncheckedFrom<VarIdPacked64> for VarIdPacked32 {
    #[allow(clippy::cast_possible_truncation)]
    fn unchecked_from(id: VarIdPacked64) -> Self {
        debug_assert!(
            id.fits_in_packed32(),
            "64-bit variable ID {id} does not fit into 32-bit variable ID"
        );
        // See also VarIdPacked16::unchecked_from::<VarIdPacked64>.
        VarIdPacked32(id.0 as u32)
    }
}

impl UncheckedFrom<VarIdPacked32> for VarIdPacked16 {
    #[allow(clippy::cast_possible_truncation)]
    fn unchecked_from(id: VarIdPacked32) -> Self {
        debug_assert!(
            id.fits_in_packed16(),
            "32-bit variable ID {id} does not fit into 16-bit variable ID"
        );
        // See also VarIdPacked16::unchecked_from::<VarIdPacked64>.
        VarIdPacked16(id.0 as u16)
    }
}

/// An implementation of [`std::error::Error`] that is reported when checked conversion
/// between instances of [`VarIdPackedAny`] is not possible.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct TryFromVarIdPackedError {
    id: u64,
    from_width: usize,
    to_width: usize,
}

impl fmt::Display for TryFromVarIdPackedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}-bit variable ID {} cannot be converted to {}-bit",
            self.from_width, self.id, self.to_width
        )
    }
}

impl std::error::Error for TryFromVarIdPackedError {}

macro_rules! impl_try_from {
    ($Large:ident => $Small:ident) => {
        impl TryFrom<$Large> for $Small {
            type Error = TryFromVarIdPackedError;

            fn try_from(id: $Large) -> Result<Self, Self::Error> {
                if id.is_undefined() {
                    return Ok(Self::undefined());
                }

                if let Ok(id) = id.0.try_into() {
                    if id <= Self::MAX_ID {
                        return Ok(Self(id));
                    }
                }

                // At this point, the value is greater than the largest representable ID
                // in the smaller type (but not undefined!).
                Err(TryFromVarIdPackedError {
                    id: id.unpack_u64(),
                    from_width: std::mem::size_of::<$Large>() * 8,
                    to_width: std::mem::size_of::<$Small>() * 8,
                })
            }
        }
    };
}

impl_try_from!(VarIdPacked32 => VarIdPacked16);
impl_try_from!(VarIdPacked64 => VarIdPacked16);
impl_try_from!(VarIdPacked64 => VarIdPacked32);

/// A trait that ensures that the type implements both [`VarIdPackedAny`] and `Into<TVarId>`.
///
/// This mainly allows us to write `AsVarId<T>` instead of needing to write
/// `VarIdPackedAny + Into<T>` everywhere.
pub trait AsVarId<TVarId: VarIdPackedAny>: VarIdPackedAny + Into<TVarId> {}
impl<A: VarIdPackedAny, B: VarIdPackedAny + Into<A>> AsVarId<A> for B {}

/// A type for identifying variables in BDDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct VariableId(u64);

impl VariableId {
    pub const MAX_16_BIT_ID: u64 = VarIdPacked16::MAX_ID as u64;
    pub const MAX_32_BIT_ID: u64 = VarIdPacked32::MAX_ID as u64;
    pub const MAX_64_BIT_ID: u64 = VarIdPacked64::MAX_ID;

    /// Create a new variable ID.
    ///
    /// This operation must always succeed; we require that the underlying algorithms all
    /// support at least 2^32 distinct variable IDs.
    pub fn new(id: u32) -> Self {
        Self(u64::from(id))
    }

    /// Create a new `VariableId` from a larger, `u64` value.
    ///
    /// This operation can fail, since the underlying algorithms are not required to support
    /// all 2^64 distinct IDs. You can assume that most IDs are still supported
    /// (at least on 64-bit systems). However, for the purpose of ensuring backwards compatibility
    /// in the future, we do not provide a guaranteed maximum ID that is always supported and
    /// is larger than 2^32.
    pub fn new_long(id: u64) -> Option<Self> {
        if id <= Self::MAX_64_BIT_ID {
            Some(Self(id))
        } else {
            None
        }
    }

    /// Check that the variable ID fits into a 16-bit packed variable ID.
    pub(crate) fn fits_in_packed16(self) -> bool {
        self.0 <= Self::MAX_16_BIT_ID
    }

    /// Check that the variable ID fits into a 32-bit packed variable ID.
    pub(crate) fn fits_in_packed32(self) -> bool {
        self.0 <= Self::MAX_32_BIT_ID
    }

    /// Check that the variable ID fits into a 64-bit packed variable ID.
    pub(crate) fn fits_in_packed64(self) -> bool {
        self.0 <= Self::MAX_64_BIT_ID
    }
}

impl UncheckedFrom<VariableId> for VarIdPacked64 {
    fn unchecked_from(value: VariableId) -> Self {
        debug_assert!(value.fits_in_packed64());
        VarIdPacked64::new(value.0)
    }
}

impl UncheckedFrom<VariableId> for VarIdPacked32 {
    fn unchecked_from(value: VariableId) -> Self {
        debug_assert!(value.fits_in_packed32());
        VarIdPacked32::new(value.0.unchecked_into())
    }
}

impl UncheckedFrom<VariableId> for VarIdPacked16 {
    fn unchecked_from(value: VariableId) -> Self {
        debug_assert!(value.fits_in_packed32());
        VarIdPacked16::new(value.0.unchecked_into())
    }
}

impl From<u16> for VariableId {
    fn from(value: u16) -> Self {
        VariableId::new(u32::from(value))
    }
}

impl From<u32> for VariableId {
    fn from(value: u32) -> Self {
        VariableId::new(value)
    }
}

impl fmt::Display for VarIdPacked16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.unpack())
    }
}

impl fmt::Display for VarIdPacked32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.unpack())
    }
}

impl fmt::Display for VarIdPacked64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.unpack())
    }
}

#[cfg(test)]
mod tests {
    use crate::conversion::{UncheckedFrom, UncheckedInto};
    use crate::variable_id::{
        VarIdPacked16, VarIdPacked32, VarIdPacked64, VarIdPackedAny, VariableId,
    };

    macro_rules! test_var_packed_undefined {
        ($func:ident, $VarId:ident) => {
            #[test]
            fn $func() {
                assert!($VarId::undefined().is_undefined());
                assert!(!$VarId::new(0).is_undefined());
            }
        };
    }

    test_var_packed_undefined!(var_packed_16_undefined, VarIdPacked16);
    test_var_packed_undefined!(var_packed_32_undefined, VarIdPacked32);
    test_var_packed_undefined!(var_packed_64_undefined, VarIdPacked64);

    macro_rules! test_var_packed_parent_counter {
        ($func:ident, $VarId:ident) => {
            #[test]
            fn $func() {
                let mut x = $VarId::new(0);
                assert!(!x.has_many_parents());
                x.increment_parents();
                assert!(!x.has_many_parents());
                x.increment_parents();
                assert!(x.has_many_parents());
                x.increment_parents();
                assert!(x.has_many_parents());
            }
        };
    }

    test_var_packed_parent_counter!(var_packed_16_parent_counter, VarIdPacked16);
    test_var_packed_parent_counter!(var_packed_32_parent_counter, VarIdPacked32);
    test_var_packed_parent_counter!(var_packed_64_parent_counter, VarIdPacked64);

    macro_rules! test_var_packed_use_cache {
        ($func:ident, $VarId:ident) => {
            #[test]
            fn $func() {
                let mut x = $VarId::new(0);
                assert!(!x.use_cache());
                x.set_use_cache(true);
                assert!(x.use_cache());
                x.set_use_cache(false);
                assert!(!x.use_cache());
            }
        };
    }

    test_var_packed_use_cache!(var_packed_16_use_cache, VarIdPacked16);
    test_var_packed_use_cache!(var_packed_32_use_cache, VarIdPacked32);
    test_var_packed_use_cache!(var_packed_64_use_cache, VarIdPacked64);

    macro_rules! test_var_packed_sort {
        ($func:ident, $VarId:ident) => {
            #[test]
            fn $func() {
                // Check that packed bits don't interfere with `Ord` and `Eq` implementations.
                let mut one = $VarId::new(1);
                let two = $VarId::new(2);
                assert_eq!(one.unpack(), 1);
                assert_eq!(two.unpack(), 2);
                assert!(one < two);
                one.set_use_cache(true);
                assert!(one < two);
                one.increment_parents();
                assert!(one < two);
                one.increment_parents();
                assert!(one < two);
                one.increment_parents();
                assert!(one < two);
                one.set_use_cache(false);
                assert!(one < two);
                assert_ne!(one, two);
            }
        };
    }

    test_var_packed_sort!(var_packed_16_sort, VarIdPacked16);
    test_var_packed_sort!(var_packed_32_sort, VarIdPacked32);
    test_var_packed_sort!(var_packed_64_sort, VarIdPacked64);

    macro_rules! test_var_packed_invalid {
        ($func:ident, $VarId:ident) => {
            #[test]
            #[should_panic]
            fn $func() {
                $VarId::new($VarId::MAX_ID + 1);
            }
        };
    }

    test_var_packed_invalid!(var_packed_16_invalid, VarIdPacked16);
    test_var_packed_invalid!(var_packed_32_invalid, VarIdPacked32);
    test_var_packed_invalid!(var_packed_64_invalid, VarIdPacked64);

    macro_rules! test_var_packed_unpack_invalid {
        ($func:ident, $VarId:ident) => {
            #[test]
            #[should_panic]
            fn $func() {
                $VarId::undefined().unpack();
            }
        };
    }

    test_var_packed_unpack_invalid!(var_packed_16_unpack_invalid, VarIdPacked16);
    test_var_packed_unpack_invalid!(var_packed_32_unpack_invalid, VarIdPacked32);
    test_var_packed_unpack_invalid!(var_packed_64_unpack_invalid, VarIdPacked64);

    macro_rules! test_var_packed_from_undefined {
        ($Small:ident => $Large:ident, $func:ident) => {
            #[test]
            fn $func() {
                assert_eq!($Large::undefined(), $Large::from($Small::undefined()));
            }
        };
    }

    test_var_packed_from_undefined!(VarIdPacked16 => VarIdPacked32, var_packed_16_from_32_undefined);
    test_var_packed_from_undefined!(VarIdPacked16 => VarIdPacked64, var_packed_16_from_64_undefined);
    test_var_packed_from_undefined!(VarIdPacked32 => VarIdPacked64, var_packed_32_from_64_undefined);

    macro_rules! test_var_packed_from {
        ($Small:ident => $Large:ident, $func:ident) => {
            #[test]
            fn $func() {
                let mut small = $Small::new(256);
                small.increment_parents();
                small.increment_parents();

                let large = $Large::from(small);

                assert_eq!(large.unpack(), 256);
                assert!(!large.use_cache());
                assert!(large.has_many_parents());

                let mut small = $Small::new(0);
                small.set_use_cache(true);

                let large = $Large::from(small);

                assert_eq!(large.unpack(), 0);
                assert!(large.use_cache());
                assert!(!large.has_many_parents());

                let mut small = $Small::new(1);
                small.increment_parents();

                let large = $Large::from(small);

                assert_eq!(large.unpack(), 1);
                assert!(!large.use_cache());
                assert!(!large.has_many_parents());
            }
        };
    }

    test_var_packed_from!(VarIdPacked16 => VarIdPacked32, var_packed_16_from_32);
    test_var_packed_from!(VarIdPacked16 => VarIdPacked64, var_packed_16_from_64);
    test_var_packed_from!(VarIdPacked32 => VarIdPacked64, var_packed_32_from_64);

    macro_rules! test_var_packed_unchecked_from_undefined {
        ($Large:ident => $Small:ident, $func:ident) => {
            #[test]
            fn $func() {
                assert_eq!(
                    $Small::undefined(),
                    $Small::unchecked_from($Large::undefined())
                );
            }
        };
    }

    test_var_packed_unchecked_from_undefined!(VarIdPacked32 => VarIdPacked16, var_packed_16_unchecked_from_32_undefined);
    test_var_packed_unchecked_from_undefined!(VarIdPacked64 => VarIdPacked16, var_packed_16_unchecked_from_64_undefined);
    test_var_packed_unchecked_from_undefined!(VarIdPacked64 => VarIdPacked32, var_packed_32_unchecked_from_64_undefined);

    macro_rules! test_var_packed_unchecked_from_invalid {
        ($Large:ident => $Small:ident, $func:ident) => {
            #[test]
            #[should_panic]
            fn $func() {
                let large = $Large::new(($Small::MAX_ID + 1).into());
                let _ = $Small::try_from(large).unwrap();
            }
        };
    }

    test_var_packed_unchecked_from_invalid!(VarIdPacked32 => VarIdPacked16, var_packed_16_unchecked_from_32_invalid);
    test_var_packed_unchecked_from_invalid!(VarIdPacked64 => VarIdPacked16, var_packed_16_unchecked_from_64_invalid);
    test_var_packed_unchecked_from_invalid!(VarIdPacked64 => VarIdPacked32, var_packed_32_unchecked_from_64_invalid);

    macro_rules! test_var_packed_unchecked_from {
        ($Large:ident => $Small:ident, $func:ident) => {
            #[test]
            fn $func() {
                let mut large = $Large::new(256);
                large.increment_parents();
                large.increment_parents();

                let small = $Small::unchecked_from(large);

                assert_eq!(small.unpack(), 256);

                assert!(!small.use_cache());
                assert!(small.has_many_parents());

                let mut large = $Large::new(0);
                large.set_use_cache(true);

                let small = $Small::unchecked_from(large);

                assert_eq!(small.unpack(), 0);
                assert!(small.use_cache());
                assert!(!small.has_many_parents());

                let mut large = $Large::new(1);
                large.increment_parents();

                let small = $Small::unchecked_from(large);

                assert_eq!(small.unpack(), 1);
                assert!(!small.use_cache());
                assert!(!small.has_many_parents());
            }
        };
    }

    test_var_packed_unchecked_from!(VarIdPacked32 => VarIdPacked16, var_packed_16_unchecked_from_32);
    test_var_packed_unchecked_from!(VarIdPacked64 => VarIdPacked16, var_packed_16_unchecked_from_64);
    test_var_packed_unchecked_from!(VarIdPacked64 => VarIdPacked32, var_packed_32_unchecked_from_64);

    macro_rules! test_var_packed_try_from_undefined {
        ($Large:ident => $Small:ident, $func:ident) => {
            #[test]
            fn $func() {
                assert_eq!(
                    $Small::undefined(),
                    $Small::try_from($Large::undefined()).unwrap()
                );
            }
        };
    }

    test_var_packed_try_from_undefined!(VarIdPacked32 => VarIdPacked16, var_packed_16_try_from_32_undefined);
    test_var_packed_try_from_undefined!(VarIdPacked64 => VarIdPacked16, var_packed_16_try_from_64_undefined);
    test_var_packed_try_from_undefined!(VarIdPacked64 => VarIdPacked32, var_packed_32_try_from_64_undefined);

    macro_rules! test_var_packed_try_from_invalid {
        ($Large:ident => $Small:ident, $func:ident) => {
            #[test]
            #[should_panic]
            fn $func() {
                let large = $Large::new(($Small::MAX_ID + 1).into());
                let _ = $Small::try_from(large).unwrap();
            }
        };
    }

    test_var_packed_try_from_invalid!(VarIdPacked32 => VarIdPacked16, var_packed_16_try_from_32_invalid);
    test_var_packed_try_from_invalid!(VarIdPacked64 => VarIdPacked16, var_packed_16_try_from_64_invalid);
    test_var_packed_try_from_invalid!(VarIdPacked64 => VarIdPacked32, var_packed_32_try_from_64_invalid);

    macro_rules! test_var_packed_try_from {
        ($Large:ident => $Small:ident, $func:ident) => {
            #[test]
            fn $func() {
                let mut large = $Large::new(256);
                large.increment_parents();
                large.increment_parents();

                let small = $Small::try_from(large).unwrap();

                assert_eq!(small.unpack(), 256);

                assert!(!small.use_cache());
                assert!(small.has_many_parents());

                let mut large = $Large::new(0);
                large.set_use_cache(true);

                let small = $Small::try_from(large).unwrap();

                assert_eq!(small.unpack(), 0);
                assert!(small.use_cache());
                assert!(!small.has_many_parents());

                let mut large = $Large::new(1);
                large.increment_parents();

                let small = $Small::try_from(large).unwrap();

                assert_eq!(small.unpack(), 1);
                assert!(!small.use_cache());
                assert!(!small.has_many_parents());
            }
        };
    }

    test_var_packed_try_from!(VarIdPacked32 => VarIdPacked16, var_packed_16_try_from_32);
    test_var_packed_try_from!(VarIdPacked64 => VarIdPacked16, var_packed_16_try_from_64);
    test_var_packed_try_from!(VarIdPacked64 => VarIdPacked32, var_packed_32_try_from_64);

    #[test]
    fn invalid_downsizing_conversions() {
        let almost_u16a = VarIdPacked32::new(u32::from(VarIdPacked16::MAX_ID + 1));
        let almost_u16b = VarIdPacked64::new(u64::from(VarIdPacked16::MAX_ID + 1));
        let almost_u16c = VariableId::new_long(u64::from(VarIdPacked16::MAX_ID + 1)).unwrap();
        let almost_u32a = VarIdPacked64::new(u64::from(VarIdPacked32::MAX_ID + 1));
        let almost_u32b = VariableId::new_long(u64::from(VarIdPacked32::MAX_ID + 1)).unwrap();

        assert!(!almost_u16a.fits_in_packed16());
        assert!(!almost_u16b.fits_in_packed16());
        assert!(!almost_u32a.fits_in_packed32());
        assert!(almost_u16b.fits_in_packed32());

        assert!(!almost_u16c.fits_in_packed16());
        assert!(almost_u16c.fits_in_packed32());
        assert!(almost_u16c.fits_in_packed64());

        assert!(!almost_u32b.fits_in_packed16());
        assert!(!almost_u32b.fits_in_packed32());
        assert!(almost_u32b.fits_in_packed64());
    }

    #[test]
    fn variable_id_display() {
        let x16 = VarIdPacked16::new(1234);
        let x32 = VarIdPacked32::new(1234);
        let x64 = VarIdPacked64::new(1234);
        assert_eq!(x16.to_string(), x32.to_string());
        assert_eq!(x32.to_string(), x64.to_string());
        assert_eq!(x64.to_string(), x16.to_string());
    }

    #[test]
    fn variable_id_too_large() {
        assert!(VariableId::new_long((u32::MAX as u64) + 1).is_some());
        assert!(VariableId::new_long(u64::MAX).is_none());
    }

    #[test]
    fn packed_id_unsuccessful_conversion() {
        let id = VarIdPacked32::new((u16::MAX as u32) + 1);
        let err = VarIdPacked16::try_from(id).unwrap_err();
        println!("{}", err);
        assert_eq!(err.from_width, 32);
        assert_eq!(err.to_width, 16);
    }

    #[test]
    #[should_panic]
    fn packed_id_unsuccessful_conversion_32_16_panic() {
        let id = VarIdPacked32::new((u16::MAX as u32) + 1);
        let _id: VarIdPacked16 = id.unchecked_into();
    }

    #[test]
    #[should_panic]
    fn packed_id_unsuccessful_conversion_64_16_panic() {
        let id = VarIdPacked64::new((u16::MAX as u64) + 1);
        let _id: VarIdPacked16 = id.unchecked_into();
    }

    #[test]
    #[should_panic]
    fn packed_id_unsuccessful_conversion_64_32_panic() {
        let id = VarIdPacked64::new((u32::MAX as u64) + 1);
        let _id: VarIdPacked32 = id.unchecked_into();
    }
}
