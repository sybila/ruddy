use std::fmt::Debug;

use crate::{
    node_id::{AsNodeId, NodeId16, NodeId32, NodeId64, NodeIdAny},
    usize_is_at_least_32_bits, usize_is_at_least_64_bits,
};

/// Task cache is a "leaky" hash table that maps a pair of [`NodeIdAny`] instances (representing
/// a "task") to another instance of [`NodeIdAny`], representing the result of said task.
///
/// The table can "lose" any value stored in it, especially in case of a collision with
/// another value. The cache is responsible for growing itself when too many collisions occur.
///
/// The `get`/`set` methods are generic to allow combinations of different bit-widths in one
/// computation. For example, we can create an instance of [`TaskCacheAny`] where the keys are
/// 16-bit identifiers, but the result is a 32-bit identifier. Importantly, the hash function
/// only depends on the bit-width of the keys, meaning that increasing the bit-width of
/// [`Self::ResultId`] can be done without recomputing the hashes.
pub(crate) trait TaskCacheAny: Default {
    type ResultId: NodeIdAny;

    /// Retrieve the result value stored for the given `task`, or [`NodeIdAny::undefined`]
    /// when the `task` has not been encountered before.
    ///
    /// The `task` consists of two [`NodeIdAny`] instances. These can be any implementations of
    /// said trait, as long as they can be safely converted to [`Self::ResultId`].
    fn get<TKeyId1: AsNodeId<Self::ResultId>, TKeyId2: AsNodeId<Self::ResultId>>(
        &self,
        task: (TKeyId1, TKeyId2),
    ) -> Self::ResultId;

    /// Update the `result` value for the given `task`, possibly overwriting any previously
    /// stored data.
    ///
    /// Just as [`TaskCacheAny::get`], the `task` can be represented using any combination of
    /// [`NodeIdAny`] implementations as long as conversion into [`Self::ResultId`] is possible.
    fn set<TKeyId1: AsNodeId<Self::ResultId>, TKeyId2: AsNodeId<Self::ResultId>>(
        &mut self,
        task: (TKeyId1, TKeyId2),
        result: Self::ResultId,
    );
}

/// A trait for types that can be used as a hash (based on knuth multiplicative hashing)
/// in the `TaskCache`. The hash must be able to be computed from a pair of [`NodeIdAny`] instances.
pub(crate) trait KnuthHash: Clone + Copy + PartialEq + Eq + Debug {
    const PRIME: Self;
    const PRIME_INVERSE: Self;

    /// Return the zero value for the hash type.
    fn zero() -> Self;

    /// Compute the hash of a pair of [`NodeIdAny`] instances using Knuth multiplicative hashing.
    fn knuth_hash<TKeyId1: NodeIdAny, TKeyId2: NodeIdAny>(val1: TKeyId1, val2: TKeyId2) -> Self;

    /// Compute the table index where a particular `hash` should be stored.
    /// The `log_size` is the base-2 logarithm of the table size.
    ///
    /// Traditionally, in the multiplicative hashing, the index is determined by the
    /// most significant bits of the hash, as these are the most "random".
    fn to_slot(self, log_size: u32) -> usize;
}

impl KnuthHash for u32 {
    const PRIME: Self = 2654435761;
    const PRIME_INVERSE: Self = 244002641;

    fn zero() -> Self {
        0
    }

    /// Compute the hash of a pair of [`NodeIdAny`] instances using Knuth multiplicative hashing.
    ///
    /// This function assumes that both input keys can undergo lossless conversion to
    /// 16-bit integers. Under this assumption, the hash function is a bijection.
    fn knuth_hash<TKeyId1: NodeIdAny, TKeyId2: NodeIdAny>(val1: TKeyId1, val2: TKeyId2) -> Self {
        let val1: u16 = val1.unchecked_into();
        let val2: u16 = val2.unchecked_into();
        let combined = u32::from(val1) << 16 | u32::from(val2);
        combined.wrapping_mul(Self::PRIME)
    }

    fn to_slot(self, log_size: u32) -> usize {
        usize_is_at_least_32_bits(self >> (u32::BITS - log_size))
    }
}

impl KnuthHash for u64 {
    const PRIME: Self = 11400714819323198549;
    const PRIME_INVERSE: Self = 6236490470931210493;

    fn zero() -> Self {
        0
    }

    /// Compute the hash of a pair of [`NodeIdAny`] instances using Knuth multiplicative hashing.
    ///
    /// This function assumes that both input keys can undergo lossless conversion to
    /// 32-bit integers. Under this assumption, the hash function is a bijection.
    fn knuth_hash<TKeyId1: NodeIdAny, TKeyId2: NodeIdAny>(val1: TKeyId1, val2: TKeyId2) -> Self {
        let val1: u32 = val1.unchecked_into();
        let val2: u32 = val2.unchecked_into();
        let combined = u64::from(val1) << 32 | u64::from(val2);
        combined.wrapping_mul(Self::PRIME)
    }

    fn to_slot(self, log_size: u32) -> usize {
        usize_is_at_least_64_bits(self >> (u64::BITS - log_size))
    }
}

impl KnuthHash for u128 {
    const PRIME: Self = 210306068529402873165736369884012333113;
    const PRIME_INVERSE: Self = 191516043691840152436889831717894333961;

    fn zero() -> Self {
        0
    }

    /// Compute the hash of a pair of [`NodeIdAny`] instances using Knuth multiplicative hashing.
    ///
    /// This function assumes that both input keys can undergo lossless conversion to
    /// 64-bit integers. Under this assumption, the hash function is a bijection.
    fn knuth_hash<TKeyId1: NodeIdAny, TKeyId2: NodeIdAny>(val1: TKeyId1, val2: TKeyId2) -> Self {
        let val1: u128 = val1.unchecked_into();
        let val2: u128 = val2.unchecked_into();
        let combined: u128 = val1 << 64 | val2;
        combined.wrapping_mul(Self::PRIME)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn to_slot(self, log_size: u32) -> usize {
        // Here, we can assume the conversion is safe, because the size of the task cache
        // should not exceed 2^64, meaning the integer should have at most 64 bits after the shift.
        usize_is_at_least_64_bits((self >> (u128::BITS - log_size)) as u64)
    }
}

#[allow(clippy::cast_possible_truncation)]
fn low_u16(val: u32) -> u16 {
    val as u16
}

fn high_u16(val: u32) -> u16 {
    (val >> 16) as u16
}

#[allow(clippy::cast_possible_truncation)]
fn low_u32(val: u64) -> u32 {
    val as u32
}

fn high_u32(val: u64) -> u32 {
    (val >> 32) as u32
}

/// Used to convert the hash to a wider type.
///
/// Ensures that for two IDs `x, y`, `Self::knuth_hash(x,y).extend() == Self::Wider::knuth_hash(x,y)`.
trait ExtendKnuthHash: KnuthHash {
    type Wider: KnuthHash;

    /// Extend the hash to a wider type, such that for two IDs `x, y`,
    /// `Self::knuth_hash(x,y).extend() == Self::Wider::knuth_hash(x,y)`.
    fn extend(self) -> Self::Wider;
}

impl ExtendKnuthHash for u32 {
    type Wider = u64;

    fn extend(self) -> Self::Wider {
        let task_combined = self.wrapping_mul(Self::PRIME_INVERSE);
        let val1 = high_u16(task_combined);
        let val2 = low_u16(task_combined);
        let task_combined = u64::from(val1) << 32 | u64::from(val2);
        task_combined.wrapping_mul(u64::PRIME)
    }
}
impl ExtendKnuthHash for u64 {
    type Wider = u128;

    fn extend(self) -> Self::Wider {
        let task_combined = self.wrapping_mul(Self::PRIME_INVERSE);
        let val1 = high_u32(task_combined);
        let val2 = low_u32(task_combined);
        let task_combined = u128::from(val1) << 64 | u128::from(val2);
        task_combined.wrapping_mul(u128::PRIME)
    }
}

/// Implementation of [`TaskCacheAny`] based on [`NodeIdAny`] and using Knuth multiplicative
/// hashing (via [`KnuthHash`]).
///
/// This implementation uses a hash table with its size always being a power of two. The table
/// is expanded (doubles its size) when the number of collisions exceeds half of the current
/// table size.
///
/// To further optimize resource consumption (mainly queue capacity for load/store instructions),
/// each key (a pair of [`NodeIdAny`] instances) is stored as a single integer representing
/// the key's hash. Since the hash function is a bijection (as long as the input and output have
/// the same number of bits), we are not losing any information by this transformation.
///
/// Finally, note that we are using `0` as the hash of an "empty" key-value pair. This is valid,
/// since the `(0, 0)` key should never be stored in this table because it is a "trivial" task
/// that does not need to be resolved via task cache. We could also use the hash of
/// `(undefined, undefined)` for the same purpose, but this is slightly less efficient because
/// initializing memory to zero often has special system-level optimizations.
#[derive(Debug)]
pub(crate) struct TaskCache<THashSize, TResultId> {
    table: Vec<(THashSize, TResultId)>,
    log_size: u32,
    collisions: usize,
}

impl<THashSize: KnuthHash, TResultId: NodeIdAny> TaskCache<THashSize, TResultId> {
    pub fn new() -> Self {
        Self::with_log_size(1)
    }

    /// Create a new instance of [`TaskCache`] with `2**log_size` entries.
    /// The `log_size` must be at least `1` (otherwise it will be set to `1`).
    pub fn with_log_size(log_size: u32) -> Self {
        let log_size = log_size.max(1);
        Self {
            table: vec![(THashSize::zero(), TResultId::undefined()); 1 << log_size],
            log_size,
            collisions: 0,
        }
    }

    /// Create a new instance of [`TaskCache`] with the reserved capacity of at
    /// least `2**max(log_capacity, log_size)`, but only initialize the first `2**log_size`
    /// entries. The `log_size` must be at least `1` (otherwise it will be set to `1`).
    ///
    /// **Right now, the method is unused, but it could be used in the future in situations
    /// where we want a pre-grown but empty table.**
    #[allow(dead_code)]
    pub fn with_log_size_and_log_capacity(log_size: u32, log_capacity: u32) -> Self {
        let log_size = log_size.max(1);
        let capacity = 1 << log_capacity.max(log_size);
        let size = 1 << log_size;
        let mut table = Vec::with_capacity(capacity);
        // The point of this unsafe code is to avoid unnecessary reallocation and bounds checks
        // during initialization, since we can't otherwise create a vector with known items
        // but greater capacity.
        #[allow(clippy::uninit_vec)]
        unsafe {
            table.set_len(size);
        }
        for i in 0..size {
            unsafe { *table.get_unchecked_mut(i) = (THashSize::zero(), TResultId::undefined()) };
        }
        Self {
            table,
            log_size,
            collisions: 0,
        }
    }

    /// The maximum number of key-value pairs that can be stored in the table at the moment.
    pub fn size(&self) -> usize {
        1 << self.log_size
    }

    /// Clears all stored values in the cache.
    /// This method keeps the allocated memory for reuse.
    ///
    /// **Right now, the method is unused, but will likely become relevant once we want to
    /// start reusing allocated cache memory between operations.**
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.log_size = 1;
        self.collisions = 0;
        unsafe {
            self.table.set_len(self.size());
            *self.table.get_unchecked_mut(0) = (THashSize::zero(), TResultId::undefined());
            *self.table.get_unchecked_mut(1) = (THashSize::zero(), TResultId::undefined());
        }
    }

    /// Compute a complete hash of the given `task` based on Knuth multiplicative hashing.
    fn hash<TKeyId1: AsNodeId<TResultId>, TKeyId2: AsNodeId<TResultId>>(
        &self,
        task: (TKeyId1, TKeyId2),
    ) -> THashSize {
        THashSize::knuth_hash(task.0, task.1)
    }

    /// Compute the table index where a particular `hash` should be stored.
    fn find_slot(&self, hash: THashSize) -> usize {
        hash.to_slot(self.log_size)
    }

    /// Grow the underlying table to two times the current size.
    fn expand(&mut self) {
        let previous_size = self.size();

        self.collisions = 0;
        self.log_size += 1;

        // Double the table size if the current reserved capacity is not enough.
        self.table.reserve(previous_size);

        // Artificially "enable" newly reserved slots without initializing them. The subsequent
        // for-loop is responsible for doing that.
        #[allow(clippy::uninit_vec)]
        unsafe {
            self.table.set_len(self.size());
        }

        // We know that a value at position X will be stored either at position 2*X or 2*X+1,
        // (depending on its hash). As such, assuming we copy values starting at the back
        // of the table, we will never overwrite anything relevant, and we will initialize the
        // whole table exactly once.
        for i in (0..previous_size).rev() {
            let (hash, result) = unsafe { *self.table.get_unchecked(i) };

            let new_slot = (i << 1) | (self.find_slot(hash) & 1);
            unsafe {
                *self.table.get_unchecked_mut(new_slot) = (hash, result);
                *self.table.get_unchecked_mut(new_slot ^ 1) =
                    (THashSize::zero(), TResultId::undefined());
            }
        }
    }
}

impl<THashSize: KnuthHash, TResultId: NodeIdAny> Default for TaskCache<THashSize, TResultId> {
    fn default() -> Self {
        Self::new()
    }
}

impl<THashSize: KnuthHash, TResultId: NodeIdAny> TaskCacheAny for TaskCache<THashSize, TResultId> {
    type ResultId = TResultId;

    fn get<TKeyId1: AsNodeId<Self::ResultId>, TKeyId2: AsNodeId<Self::ResultId>>(
        &self,
        task: (TKeyId1, TKeyId2),
    ) -> Self::ResultId {
        let hash = self.hash(task);
        let slot_index = self.find_slot(hash);
        let slot = unsafe { self.table.get_unchecked(slot_index) };

        if slot.0 == hash {
            return slot.1;
        }

        TResultId::undefined()
    }

    fn set<TKeyId1: AsNodeId<Self::ResultId>, TKeyId2: AsNodeId<Self::ResultId>>(
        &mut self,
        task: (TKeyId1, TKeyId2),
        result: Self::ResultId,
    ) {
        let hash = self.hash(task);
        let slot_index = self.find_slot(hash);
        let slot = unsafe { self.table.get_unchecked_mut(slot_index) };

        // First check if the slot is empty
        if slot.0 == THashSize::zero() {
            *slot = (hash, result);
            return;
        }
        // Otherwise, check if the hash matches and we know the result
        if slot.0 == hash {
            return;
        }

        *slot = (hash, result);

        self.collisions += 1;
        if self.collisions > self.table.len() / 2 {
            self.expand();
        }
    }
}

pub(crate) type TaskCache16<TResultId = NodeId16> = TaskCache<u32, TResultId>;
pub(crate) type TaskCache32<TResultId = NodeId32> = TaskCache<u64, TResultId>;
pub(crate) type TaskCache64<TResultId = NodeId64> = TaskCache<u128, TResultId>;

macro_rules! impl_from_task_cache_no_extension {
    ($from_id:ident, $to_id:ident) => {
        impl<THashSize> From<TaskCache<THashSize, $from_id>> for TaskCache<THashSize, $to_id>
        where
            THashSize: KnuthHash,
        {
            fn from(cache: TaskCache<THashSize, $from_id>) -> Self {
                let mut new_table = Vec::with_capacity(cache.size());
                #[allow(clippy::uninit_vec)]
                unsafe {
                    new_table.set_len(cache.size());
                }
                for (i, (hash, result)) in cache.table.into_iter().enumerate() {
                    unsafe {
                        *new_table.get_unchecked_mut(i) = (hash, result.into());
                    }
                }
                Self {
                    table: new_table,
                    log_size: cache.log_size,
                    collisions: cache.collisions,
                }
            }
        }
    };
}

impl_from_task_cache_no_extension!(NodeId16, NodeId32);
impl_from_task_cache_no_extension!(NodeId32, NodeId64);

macro_rules! impl_from_task_cache_with_extension {
    ($from_id:ident, $to_id:ident, $from_hash:ident, $to_hash:ident) => {
        impl From<TaskCache<$from_hash, $from_id>> for TaskCache<$to_hash, $to_id> {
            fn from(cache: TaskCache<$from_hash, $from_id>) -> Self {
                let mut table = vec![(0, $to_id::undefined()); cache.size()];
                let log_size = cache.log_size;
                let mut collisions = 0;

                for (hash, result) in cache.table.into_iter() {
                    let new_hash = hash.extend();
                    let new_slot_index = new_hash.to_slot(log_size);
                    let new_slot = unsafe { table.get_unchecked_mut(new_slot_index) };

                    collisions += (new_slot.0 != 0) as usize;
                    *new_slot = (new_hash, result.into());
                }
                Self {
                    table,
                    log_size,
                    collisions,
                }
            }
        }
    };
}

impl_from_task_cache_with_extension!(NodeId16, NodeId32, u32, u64);
impl_from_task_cache_with_extension!(NodeId32, NodeId64, u64, u128);

#[cfg(test)]
mod tests {
    use crate::node_id::{NodeId16, NodeId32, NodeId64, NodeIdAny};
    use crate::task_cache::{ExtendKnuthHash, KnuthHash};
    use crate::task_cache::{TaskCache16, TaskCache32, TaskCache64, TaskCacheAny};

    #[test]
    pub fn task_cache_basic() {
        let mut cache: TaskCache32<NodeId32> = TaskCache32::with_log_size(4);
        let cache2: TaskCache32<NodeId32> = TaskCache32::with_log_size_and_log_capacity(4, 20);
        assert_eq!(cache.size(), cache2.size());

        let p0 = NodeId32::zero();
        let p1 = NodeId32::one();
        let p2 = NodeId32::new(2);

        assert!(cache.get((p0, p2)).is_undefined());
        assert!(cache.get((p2, p0)).is_undefined());
        cache.set((p0, p2), p1); // write new
        assert!(cache.get((p0, p2)).is_one());
        assert!(cache.get((p2, p0)).is_undefined());
        cache.set((p0, p2), p1); // overwrite existing
        assert!(cache.get((p0, p2)).is_one());
        assert!(cache.get((p2, p0)).is_undefined());
        cache.expand(); // expand (should not modify)
        assert_ne!(cache.size(), cache2.size());
        assert!(cache.get((p0, p2)).is_one());
        assert!(cache.get((p2, p0)).is_undefined());
        cache.clear(); // remove everything
        assert!(cache.get((p0, p2)).is_undefined());
        assert!(cache.get((p2, p0)).is_undefined());
    }

    #[test]
    pub fn task_cache_collisions() {
        let p0 = NodeId32::zero();
        let p1 = NodeId32::one();
        let p3 = NodeId32::new(3);

        let mut cache = TaskCache32::with_log_size(1);
        assert_eq!(cache.size(), 2);
        // Here, we have computed that (p0,p1) and (p0,p3) will cause a collision as long as
        // the cache only has two slots.
        cache.set((p0, p1), p1);
        assert!(cache.get((p0, p1)).is_one());
        assert!(cache.get((p0, p3)).is_undefined());
        cache.set((p0, p3), p1);
        println!("{:?}", cache.table);
        assert!(cache.get((p0, p1)).is_undefined());
        assert!(cache.get((p0, p3)).is_one());
        cache.set((p0, p1), p1);
        cache.set((p0, p3), p1);
        assert!(cache.size() > 2);
    }

    #[test]
    fn hash_extension_32_to_64() {
        let p17 = NodeId16::new(17);
        let p32 = NodeId16::new(32);

        let p17_32 = NodeId32::from(p17);
        let p32_32 = NodeId32::from(p32);

        let hash = u32::knuth_hash(p17, p32);
        let extended = hash.extend();

        let hash_32 = u64::knuth_hash(p17_32, p32);
        assert_eq!(hash_32, extended);

        let hash_32 = u64::knuth_hash(p17_32, p32_32);
        assert_eq!(hash_32, extended);

        let hash_32 = u64::knuth_hash(p17, p32_32);
        assert_eq!(hash_32, extended);
    }

    #[test]
    fn hash_extension_64_to_128() {
        let p255 = NodeId32::new(255);
        let p7987 = NodeId32::new(7987);

        let p255_64 = NodeId64::from(p255);
        let p7987_64 = NodeId64::from(p7987);

        let hash = u64::knuth_hash(p255, p7987);
        let extended = hash.extend();

        let hash_64 = u128::knuth_hash(p255_64, p7987);
        assert_eq!(hash_64, extended);

        let hash_64 = u128::knuth_hash(p255_64, p7987_64);
        assert_eq!(hash_64, extended);

        let hash_64 = u128::knuth_hash(p255, p7987_64);
        assert_eq!(hash_64, extended);
    }

    #[test]
    pub fn task_cache_upcast() {
        let mut cache = TaskCache16::<NodeId16>::with_log_size(4);

        let p0 = NodeId16::zero();
        let p1 = NodeId16::one();
        let p2 = NodeId16::new(2);

        cache.set((p0, p1), p2);

        let mut cache = TaskCache16::<NodeId32>::from(cache);
        assert_eq!(cache.get((p0, p1)), NodeId32::from(p2));

        let p3 = NodeId32::new(3);
        cache.set((p1, p0), p3);
        assert_eq!(cache.get((p0, p1)), NodeId32::from(p2));
        assert_eq!(cache.get((p1, p0)), p3);

        let mut cache = TaskCache16::<NodeId64>::from(cache);

        let p4 = NodeId64::new(4);
        cache.set((p1, p1), p4);
        assert_eq!(cache.get((p0, p1)), NodeId64::from(p2));
        assert_eq!(cache.get((p1, p0)), NodeId64::from(p3));
        assert_eq!(cache.get((p1, p1)), p4);
    }

    #[test]
    fn task_cache_upcast_with_extension() {
        let mut cache = TaskCache16::<NodeId16>::with_log_size(3);

        let p0 = NodeId16::zero();
        let p1 = NodeId16::one();
        let p2 = NodeId16::new(2);

        cache.set((p0, p1), p2);

        let mut cache = TaskCache32::<NodeId32>::from(cache);
        assert_eq!(cache.get((p0, p1)), NodeId32::from(p2));

        let p3 = NodeId32::new(3);
        cache.set((p1, p0), p3);
        println!("{:?}", cache.table);
        assert_eq!(cache.get((p0, p1)), NodeId32::from(p2));
        assert_eq!(cache.get((p1, p0)), p3);

        let mut cache = TaskCache64::<NodeId64>::from(cache);
        println!("{:?}", cache.table);
        assert_eq!(cache.get((p0, p1)), NodeId64::from(p2));
        assert_eq!(cache.get((p1, p0)), NodeId64::from(p3));
        let p4 = NodeId64::new(4);
        cache.set((p1, p3), p4);
        println!("{:?}", cache.table);
        assert_eq!(cache.get((p0, p1)), NodeId64::from(p2));
        assert_eq!(cache.get((p1, p0)), NodeId64::from(p3));
        assert_eq!(cache.get((p1, p3)), p4);
    }
}
