//! Defines the representation of task caches (used in `apply` algorithms). Includes: [`TaskCacheAny`]
//! [`TaskCache`], [`TaskCache16`], [`TaskCache32`], and [`TaskCache64`].

use std::fmt::Debug;

use crate::{
    node_id::{AsNodeId, NodeId16, NodeId32, NodeId64, NodeIdAny},
    usize_is_at_least_32_bits, usize_is_at_least_64_bits,
};

/// Task cache is a "leaky" hash table that maps a pair of [`NodeIdAny`] instances (representing
/// a "task") to another instance of [`NodeIdAny`], representing the result of said task.
///
/// The table can "lose" any value that is stored in it, especially in case of a collision with
/// another value. The cache is responsible for growing itself when too many collisions occur.
pub trait TaskCacheAny: Default {
    type ResultId: NodeIdAny;

    /// Retrieve the result value that is stored for the given `task`, or [`NodeIdAny::undefined`]
    /// when the `task` has not been encountered before.
    fn get<TKeyId1: AsNodeId<Self::ResultId>, TKeyId2: AsNodeId<Self::ResultId>>(
        &self,
        task: (TKeyId1, TKeyId2),
    ) -> Self::ResultId;

    /// Update the `result` value for the given `task`, possibly overwriting any previously
    /// stored data.
    fn set<TKeyId1: AsNodeId<Self::ResultId>, TKeyId2: AsNodeId<Self::ResultId>>(
        &mut self,
        task: (TKeyId1, TKeyId2),
        result: Self::ResultId,
    );
}

/// A trait for types that can be used as a hash (based on knuth multiplicative hashing)
/// in the `TaskCache`. The hash must be able to be computed from a pair of [`NodeIdAny`] instances.
pub trait KnuthHash: Clone + Copy + PartialEq + Eq + Debug {
    /// Return the zero value for the hash type.
    fn zero() -> Self;

    /// Compute the hash of a pair of [`NodeIdAny`] instances using Knuth multiplicative hashing.
    fn knuth_hash<TKeyId1: NodeIdAny, TKeyId2: NodeIdAny>(val1: TKeyId1, val2: TKeyId2) -> Self;

    /// Compute the table index where a particular `hash` should be stored.
    /// The `log_size` is the base-2 logarithm of the table size.
    fn to_slot(self, log_size: u32) -> usize;
}

impl KnuthHash for u32 {
    fn zero() -> Self {
        0
    }

    /// Compute the hash of a pair of [`NodeIdAny`] instances using Knuth multiplicative hashing.
    ///
    /// This function assumes that both input keys can be losslessly represented as 16-bit integers.
    /// Under this assumption, the hash function is a bijection.
    fn knuth_hash<TKeyId1: NodeIdAny, TKeyId2: NodeIdAny>(val1: TKeyId1, val2: TKeyId2) -> Self {
        let val1: u16 = val1.unchecked_into();
        let val2: u16 = val2.unchecked_into();

        let combined = u32::from(val1) << 16 | u32::from(val2);
        combined.wrapping_mul(2654435769)
    }

    fn to_slot(self, log_size: u32) -> usize {
        usize_is_at_least_32_bits(self >> (u32::BITS - log_size))
    }
}

impl KnuthHash for u64 {
    fn zero() -> Self {
        0
    }

    /// Compute the hash of a pair of [`NodeIdAny`] instances using Knuth multiplicative hashing.
    ///
    /// This function assumes that both input keys can be losslessly represented as 32-bit integers.
    /// Under this assumption, the hash function is a bijection.
    fn knuth_hash<TKeyId1: NodeIdAny, TKeyId2: NodeIdAny>(val1: TKeyId1, val2: TKeyId2) -> Self {
        let val1: u32 = val1.unchecked_into();
        let val2: u32 = val2.unchecked_into();

        let combined = u64::from(val1) << 32 | u64::from(val2);
        combined.wrapping_mul(14695981039346656039)
    }

    fn to_slot(self, log_size: u32) -> usize {
        usize_is_at_least_64_bits(self >> (u64::BITS - log_size))
    }
}

impl KnuthHash for u128 {
    fn zero() -> Self {
        0
    }

    /// Compute the hash of a pair of [`NodeIdAny`] instances using Knuth multiplicative hashing.
    fn knuth_hash<TKeyId1: NodeIdAny, TKeyId2: NodeIdAny>(val1: TKeyId1, val2: TKeyId2) -> Self {
        let combined: u128 = Into::<u128>::into(val1) << 64 | Into::<u128>::into(val2);
        combined.wrapping_mul(210306068529402873165736369884012333108)
    }

    // TODO: how to handle conversions to usize here?
    #[allow(clippy::as_conversions)]
    fn to_slot(self, log_size: u32) -> usize {
        (self >> (u128::BITS - log_size)) as usize
    }
}

/// Implementation of [`TaskCacheAny`] based on [`NodeIdAny`] and using Knuth multiplicative hashing.
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
/// that does not need to be resolved via task cache. However, we could also use the hash of
/// `(undefined, undefined)` for the same purpose.
#[derive(Debug)]
pub struct TaskCache<THashSize, TResultId> {
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

        // Double the table size if current reserved capacity is not sufficient.
        self.table.reserve(previous_size);

        // Artificially "enable" newly reserved slots without initializing them. The subsequent
        // for-loop is responsible for doing that.
        #[allow(clippy::uninit_vec)]
        unsafe {
            self.table.set_len(self.size());
        }

        // We know that a value at position X will be stored either at position 2*X, or 2*X+1,
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

pub type TaskCache16<TResultId> = TaskCache<u32, TResultId>;
pub type TaskCache32<TResultId> = TaskCache<u64, TResultId>;
pub type TaskCache64<TResultId> = TaskCache<u128, TResultId>;

macro_rules! impl_from_task_cache {
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

impl_from_task_cache!(NodeId16, NodeId32);
impl_from_task_cache!(NodeId32, NodeId64);
impl_from_task_cache!(NodeId16, NodeId64);

#[cfg(test)]
mod tests {
    use crate::node_id::{NodeId32, NodeIdAny};
    use crate::task_cache::{TaskCache32, TaskCacheAny};

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
        let p2 = NodeId32::new(2);

        let mut cache = TaskCache32::with_log_size(1);
        assert_eq!(cache.size(), 2);
        // Here, we have computed that (p0,p1) and (p0,p2) will cause a collision as long as
        // the cache only has two slots.
        cache.set((p0, p1), p1);
        assert!(cache.get((p0, p1)).is_one());
        assert!(cache.get((p0, p2)).is_undefined());
        cache.set((p0, p2), p1);
        println!("{:?}", cache.table);
        assert!(cache.get((p0, p1)).is_undefined());
        assert!(cache.get((p0, p2)).is_one());
        cache.set((p0, p1), p1);
        cache.set((p0, p2), p1);
        assert!(cache.size() > 2);
    }
}
