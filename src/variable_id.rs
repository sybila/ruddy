//! Defines the representation of variable identifiers. Includes: [VariableId] and [VarIdPacked32].

use std::hash::Hash;

/// An internal trait implemented by types that can serve as BDD variable identifiers.
/// The core feature of this trait is that a variable ID must have one designated
/// "undefined" value (similar to `Option::None`). Furthermore, it must hold that
/// `id < undefined` for every other `id` value of the same type.
pub trait VariableId: PartialEq + Eq + PartialOrd + Ord + Hash {
    /// Return an instance of the "undefined" variable ID.
    fn undefined() -> Self;
    /// Checks if this ID is [VariableId::undefined].
    fn is_undefined(&self) -> bool;
}

/// A 32-bit implementation of the [VariableId] trait that packs additional
/// information about the node containing the variable into the variable ID
/// to make the apply algorithm more efficient.
///
/// This means that [VarIdPacked32] can only represent `2**29 - 1` unique variables (see also
/// [VarIdPacked32::MAX_ID]).
///
/// The packed information is as follows:
///  - Two least-significant bits are used as a `{0, 1, many}` counter that keeps track of how
///    many parents the node containing the variable has.
///  - Third least-significant bit is used to indicate if the node containing the variable should
///    use the task cache in the apply algorithm.
///
/// Note that the "packed metadata" is not ignored when comparing variable IDs using `Eq`,
/// `Ord` or `Hash`.
#[derive(Clone, Copy, Debug)]
pub struct VarIdPacked32(u32);

/// An internal "mask bit" that is used when manipulating the "use cache"
/// flag in packed variable IDs.
const USE_CACHE_MASK: u32 = 0b100;
/// An internal mask that can be used to reset both the "parent counter" and the "use cache" flag.
const RESET_MASK: u32 = (u32::MAX >> 3) << 3;

impl VarIdPacked32 {
    /// The largest variable ID that can be safely represented by [VarIdPacked32].
    pub const MAX_ID: u32 = (u32::MAX >> 3) - 1;

    /// Create a new instance of [VarIdPacked32] with the specified variable ID. It must hold
    /// that `0 <= id <= MAX_ID`.
    ///
    /// The variable ID is shifted left by 3 bits to make room for the additional
    /// "packed" information.
    ///
    /// ## Undefined behavior
    ///
    /// For performance reasons, range checks on variable IDs are only performed in debug mode.
    /// In release mode, undefined behavior can occur if the ID is invalid.
    pub fn new(id: u32) -> VarIdPacked32 {
        // `<<` should fail when overflowing in debug mode, but this is a bit easier to trace.
        debug_assert!(id <= Self::MAX_ID, "Variable ID too large: {id}");
        VarIdPacked32(id << 3)
    }

    /// Unpack the packed variable ID, giving the "true" variable ID without the
    /// additional information.
    ///
    /// ## Undefined behavior
    ///
    /// For [VarIdPacked32::undefined], the result is not defined. In debug mode, the method
    /// will fail. In release mode, unpacking an undefined value results in undefined behavior.
    pub fn unpack(&self) -> u32 {
        debug_assert!(self.0 != u32::MAX, "Cannot unpack undefined");
        self.0 >> 3
    }

    /// Returns true if the internal parent counter is set to `many` (i.e. not `0` or `1`).
    pub(crate) fn has_many_parents(&self) -> bool {
        self.0 & 0b10 != 0
    }

    /// Check if the "use cache" flag is set on this variable ID.
    pub(crate) fn use_cache(&self) -> bool {
        self.0 & USE_CACHE_MASK != 0
    }

    /// Update the "use cache" flag of this variable ID.
    pub(crate) fn set_use_cache(&mut self, value: bool) {
        self.0 = (self.0 & !USE_CACHE_MASK) | (u32::from(value) << 2);
    }

    /// Increment the parent counter, assuming it is not already set to `many` (in that case,
    /// the counter stays the same).
    pub(crate) fn increment_parents(&mut self) {
        // 00 -> 01 -> 00 | 01 -> 01
        // 01 -> 10 -> 01 | 10 -> 11
        // 10 -> 11 -> 10 | 11 -> 11
        // 11 -> 00 -> 11 | 00 -> 11
        let counter = (self.0.overflowing_add(1).0) & 0b11;
        self.0 |= counter;
    }

    /// Returns the same variable ID, but with the "parent counter" and "use cache" flag reset to
    /// their default state.
    pub(crate) fn reset(&self) -> Self {
        VarIdPacked32(self.0 & RESET_MASK)
    }
}

impl PartialEq for VarIdPacked32 {
    fn eq(&self, other: &Self) -> bool {
        self.reset().0 == other.reset().0
    }
}

impl Eq for VarIdPacked32 {}

impl PartialOrd for VarIdPacked32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(Ord::cmp(&self, &other))
    }
}

impl Ord for VarIdPacked32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.reset().0.cmp(&other.reset().0)
    }
}

impl Hash for VarIdPacked32 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.reset().0.hash(state)
    }
}

impl VariableId for VarIdPacked32 {
    fn undefined() -> Self {
        VarIdPacked32(u32::MAX)
    }

    fn is_undefined(&self) -> bool {
        self.0 == u32::MAX
    }
}

#[cfg(test)]
mod tests {
    use crate::variable_id::{VarIdPacked32, VariableId};

    #[test]
    pub fn var_packed_32_undefined() {
        assert!(VarIdPacked32::undefined().is_undefined());
        assert!(!VarIdPacked32::new(0).is_undefined());
    }

    #[test]
    pub fn var_packed_32_parent_counter() {
        let mut x = VarIdPacked32::new(0);
        assert!(!x.has_many_parents());
        x.increment_parents();
        assert!(!x.has_many_parents());
        x.increment_parents();
        assert!(x.has_many_parents());
        x.increment_parents();
        assert!(x.has_many_parents());
    }

    #[test]
    pub fn var_packed_32_use_cache() {
        let mut x = VarIdPacked32::new(0);
        assert!(!x.use_cache());
        x.set_use_cache(true);
        assert!(x.use_cache());
        x.set_use_cache(false);
        assert!(!x.use_cache());
    }

    #[test]
    pub fn var_packed_32_sort() {
        // Check that packed bits don't interfere with `Ord` and `Eq` implementations.
        let mut one = VarIdPacked32::new(1);
        let two = VarIdPacked32::new(2);
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

    #[test]
    #[should_panic]
    pub fn var_packed_32_invalid() {
        VarIdPacked32::new(VarIdPacked32::MAX_ID + 1);
    }

    #[test]
    #[should_panic]
    pub fn var_packed_32_invalid_unpack() {
        VarIdPacked32::undefined().unpack();
    }
}
