//! Defines the representation of task caches (used in `apply` algorithms). Includes: [`TaskCache`]
//! and [`TaskCache32`].

use crate::{
    node_id::{BddNodeId, NodeId32},
    usize_is_at_least_64_bits,
};

/// Task cache is a "leaky" hash table that maps a pair of [`BddNodeId`] instances (representing
/// a "task") to another instance of [`BddNodeId`], representing the result of said task.
///
/// The table can "lose" any value that is stored in it, especially in case of a collision with
/// another value. The cache is responsible for growing itself when too many collisions occur.
pub trait TaskCache {
    type Id: BddNodeId;

    /// Retrieve the result value that is stored for the given `task`, or [BddNodeId::undefined]
    /// when the `task` has not been encountered before.
    fn get(&self, task: (Self::Id, Self::Id)) -> Self::Id;

    /// Update the `result` value for the given `task`, possibly overwriting any previously
    /// stored data.
    fn set(&mut self, task: (Self::Id, Self::Id), result: Self::Id);
}

/// Implementation of [`TaskCache`] based on [`NodeId32`] and using Knuth multiplicative hashing.
///
/// This implementation uses a hash table with its size always being a power of two. The table
/// is expanded (doubles its size) when the number of collisions exceeds half of the current
/// table size.
///
/// To further optimize resource consumption (mainly queue capacity for load/store instructions),
/// each key (a pair of [`NodeId32`] instances) is stored as a single 64-bit integer representing
/// the key's hash. Since the hash function is a bijection (as long as the input and output have
/// the same number of bits), we are not losing any information by this transformation.
///
/// Finally, note that we are using `0` as the hash of an "empty" key-value pair. This is valid,
/// since the `(0, 0)` key should never be stored in this table because it is a "trivial" task
/// that does not need to be resolved via task cache. However, we could also use the hash of
/// `(undefined, undefined)` for the same purpose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskCache32 {
    table: Vec<(u64, NodeId32)>,
    log_size: u32,
    collisions: usize,
}

impl TaskCache32 {
    /// Create a new instance of [`TaskCache32`] with `2**log_size` entries.
    /// The `log_size` must be at least `1` (otherwise it will be set to `1`).
    pub fn with_log_size(log_size: u32) -> TaskCache32 {
        let log_size = log_size.max(1);
        TaskCache32 {
            table: vec![(0, NodeId32::undefined()); 1 << log_size],
            log_size,
            collisions: 0,
        }
    }

    /// Create a new instance of [`TaskCache32`] with the reserved capacity of at
    /// least `2**max(log_capacity, log_size)`, but only initialize the first `2**log_size`
    /// entries. The `log_size` must be at least `1` (otherwise it will be set to `1`).
    pub fn with_log_size_and_log_capacity(log_size: u32, log_capacity: u32) -> TaskCache32 {
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
            unsafe { *table.get_unchecked_mut(i) = (0, NodeId32::undefined()) };
        }
        TaskCache32 {
            table,
            log_size,
            collisions: 0,
        }
    }

    /// Clears all stored values in the cache.
    /// This method keeps the allocated memory for reuse.
    pub fn clear(&mut self) {
        self.log_size = 1;
        self.collisions = 0;
        unsafe {
            self.table.set_len(self.size());
            *self.table.get_unchecked_mut(0) = (0, NodeId32::undefined());
            *self.table.get_unchecked_mut(1) = (0, NodeId32::undefined());
        }
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
                *self.table.get_unchecked_mut(new_slot ^ 1) = (0, NodeId32::undefined());
            }
        }
    }

    /// Compute a complete hash of the given `task` based on Knuth multiplicative hashing.
    fn hash(&self, task: (NodeId32, NodeId32)) -> u64 {
        // Combine the two nodes into a single value
        let combined = (task.0.as_u64()) << 32 | task.1.as_u64();
        // It seems that this hash is indeed a "complete" hash:
        // https://memotut.com/en/aeefa085417b7134f793/
        combined.overflowing_mul(14695981039346656039).0
    }

    /// Compute the table index where a particular `hash` should be stored.
    ///
    /// Note that we do not have to use `%` here because the table size is always a power of two.
    fn find_slot(&self, hash: u64) -> usize {
        usize_is_at_least_64_bits(hash >> (u64::BITS - self.log_size))
    }

    /// The maximum number of key-value pairs that can be stored in the table at the moment.
    pub fn size(&self) -> usize {
        1 << self.log_size
    }
}

impl TaskCache for TaskCache32 {
    type Id = NodeId32;

    fn get(&self, task: (Self::Id, Self::Id)) -> Self::Id {
        let hash = self.hash(task);
        let slot_index = self.find_slot(hash);
        let slot = unsafe { self.table.get_unchecked(slot_index) };
        if slot.0 == hash {
            slot.1
        } else {
            NodeId32::undefined()
        }
    }

    fn set(&mut self, task: (Self::Id, Self::Id), result: Self::Id) {
        let hash = self.hash(task);
        let slot_index = self.find_slot(hash);
        let slot = unsafe { self.table.get_unchecked_mut(slot_index) };

        // First check if the slot is empty
        if slot.0 == 0 {
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

#[cfg(test)]
mod tests {
    use crate::node_id::{BddNodeId, NodeId32};
    use crate::task_cache::{TaskCache, TaskCache32};

    #[test]
    pub fn test_task_cache_basic() {
        let mut cache = TaskCache32::with_log_size(4);
        let cache2 = TaskCache32::with_log_size_and_log_capacity(4, 20);
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
    pub fn test_task_cache_collisions() {
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
