/// An internal trait implemented by types that can serve as BDD variable identifiers.
/// The core feature of this trait is that a variable ID must have one designated
/// "undefined" value (similar to `Option::None`).
pub trait VariableId {
    fn undefined() -> Self;
    fn is_undefined(&self) -> bool;
}

/// A 32-bit implementation of the [VariableId] trait that packs additional
/// information about the node containing the variable into the variable ID
/// to make the apply algorithm more efficient.
///
/// This means that [VarIdPacked32] can only represent 2**29 - 1 unique variables.
///
/// The packed information is as follows:
/// Two low bits are used as a {0, 1, many} counter that keeps track of how many
/// parents the node containing the variable has.
/// Third low bit is used to indicate if the node containing the variable should
/// use the task cache in the apply algorithm.
#[derive(Clone, Copy)]
pub struct VarIdPacked32(u32);

impl VarIdPacked32 {
    /// Create a new instance of [VarIdPacked32] with the specified variable ID.
    /// The variable ID is shifted left by 3 bits to make room for the additional
    /// information.
    pub fn new(id: u32) -> VarIdPacked32 {
        VarIdPacked32(id << 3)
    }

    /// Unpack the packed variable ID, giving the "true" variable ID without the
    /// additional information.
    pub fn unpack(&self) -> u32 {
        self.0 >> 3
    }

    pub fn has_many_parents(&self) -> bool {
        self.0 & 0b10 != 0
    }

    pub fn use_cache(&self) -> bool {
        self.0 & 0b100 != 0
    }

    pub fn set_use_cache(&mut self, value: bool) {
        self.0 = (self.0 & !(1 << 2)) | (u32::from(value) << 2);
    }

    pub fn increment_parents(&mut self) {
        // 00 -> 01 -> 00 | 01 -> 01
        // 01 -> 10 -> 01 | 10 -> 11
        // 10 -> 11 -> 10 | 11 -> 11
        // 11 -> 00 -> 11 | 00 -> 11
        let counter = (self.0.overflowing_add(1).0) & 0b11;
        self.0 |= counter;
    }
}

impl PartialEq for VarIdPacked32 {
    fn eq(&self, other: &Self) -> bool {
        self.unpack() == other.unpack()
    }
}

impl Eq for VarIdPacked32 {}

impl PartialOrd for VarIdPacked32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.unpack().cmp(&other.unpack()))
    }
}

impl Ord for VarIdPacked32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.unpack().cmp(&other.unpack())
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
