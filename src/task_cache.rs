use crate::{
    node_id::{BddNodeId, NodeId32},
    usize_is_at_least_64_bits,
};

/// Task cache is a "leaky" hash table that maps a pair of [BddNodeId] instances (representing
/// a "task") to another instance of [BddNodeId], representing the result of said task.
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

/// A 32-bit implementation of [TaskCache]. This implementation uses a hash table with
/// its size always being a power of two.
///
/// The table is expanded (doubles its size) when the number of collisions exceeds half of the table size.
pub struct TaskCache32 {
    table: Vec<(u64, NodeId32)>,
    log_size: u32,
    collisions: usize,
}

impl TaskCache32 {
    /// Create a new instance of [TaskCache32] with 2**log_size entries.
    /// The log_size must be at least 1 (otherwise it will be set to 1).
    pub fn with_log_size(log_size: u32) -> TaskCache32 {
        let log_size = log_size.max(1);
        TaskCache32 {
            table: vec![(0, NodeId32::undefined()); 1 << log_size],
            log_size,
            collisions: 0,
        }
    }

    /// Create a new instance of [TaskCache32] with the reserved capacity of at
    /// least 2**max(log_capacity, log_size), but only initialize the first 2**log_size entries.
    /// The log_size must be at least 1 (otherwise it will be set to 1).
    pub fn with_log_size_and_log_capacity(log_size: u32, log_capacity: u32) -> TaskCache32 {
        let log_size = log_size.max(1);
        let capacity = 1 << log_capacity.max(log_size);
        let size = 1 << log_size;
        let mut table = Vec::with_capacity(capacity);
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
        }
        for i in 0..self.size() {
            unsafe { *self.table.get_unchecked_mut(i) = (0, NodeId32::undefined()) };
        }
    }

    fn expand(&mut self) {
        let previous_size = self.size();

        self.collisions = 0;
        self.log_size += 1;

        // Double the table size if needed
        self.table.reserve(previous_size);
        #[allow(clippy::uninit_vec)]
        unsafe {
            self.table.set_len(self.size());
        }

        for i in (0..previous_size).rev() {
            let (hash, result) = unsafe { *self.table.get_unchecked(i) };

            let new_slot = (i << 1) | (self.find_slot(hash) & 1);
            unsafe {
                *self.table.get_unchecked_mut(new_slot) = (hash, result);
                *self.table.get_unchecked_mut(new_slot ^ 1) = (0, NodeId32::undefined());
            }
        }
    }

    fn hash(&self, task: (NodeId32, NodeId32)) -> u64 {
        // Combine the two nodes into a single value
        let combined = (task.0.as_u64()) << 32 | task.1.as_u64();
        // It seems that this hash is indeed a "complete" hash:
        // https://memotut.com/en/aeefa085417b7134f793/
        combined.overflowing_mul(14695981039346656039).0
    }

    fn find_slot(&self, hash: u64) -> usize {
        usize_is_at_least_64_bits(hash >> (u64::BITS - self.log_size))
    }

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
