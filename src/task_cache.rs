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
/// its size always being a power of two (2**log_size).
///
/// The table is expanded (doubles its size) when the number of collisions exceeds half of the table size.
pub struct TaskCache32 {
    table: Vec<(u64, NodeId32)>,
    log_size: u32,
    collisions: usize,
}

impl TaskCache32 {
    /// Create a new instance of [TaskCache32] with 2**log_size entries.
    pub fn with_log_size(log_size: u32) -> TaskCache32 {
        TaskCache32 {
            table: vec![(0, NodeId32::undefined()); 1 << log_size],
            log_size,
            collisions: 0,
        }
    }

    /// Create a new instance of [TaskCache32] with the reserved capacity of at
    /// least 2**max(log_capacity, log_size), but only initialize the first 2**log_size entries.
    pub fn with_log_size_and_log_capacity(log_size: u32, log_capacity: u32) -> TaskCache32 {
        let capacity = 1 << log_capacity.max(log_size);
        let table = Vec::with_capacity(capacity);
        let mut table = TaskCache32 {
            table,
            log_size,
            collisions: 0,
        };
        table.reset();
        table
    }

    /// Reset the table to its initial state, clearing all stored values.
    /// This method does not change the size of the table.
    pub fn reset(&mut self) {
        for i in 0..self.size() {
            let slot = unsafe { self.table.get_unchecked_mut(i) };
            *slot = (0, NodeId32::undefined());
        }
    }

    fn expand(&mut self) {
        let previous_size = self.size();

        self.collisions = 0;
        self.log_size += 1;

        // Double the table size if needed
        self.table.reserve(previous_size);

        for i in (0..previous_size).rev() {
            let (hash, result) = unsafe { *self.table.get_unchecked(i) };

            let new_slot = (i << 1) | (self.find_slot(hash) & 1);
            unsafe {
                let x = self.table.get_unchecked_mut(new_slot);
                *x = (hash, result);
                let x = self.table.get_unchecked_mut(new_slot ^ 1);
                *x = (0, NodeId32::undefined());
            }
        }

        // Everything is initialized, so we can set the new length
        unsafe {
            self.table.set_len(self.size());
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
        // Otherwise, check if the hash matches and we know the result
        } else if slot.0 == hash {
            return;
        }

        *slot = (hash, result);

        self.collisions += 1;
        if self.collisions > self.table.len() / 2 {
            self.expand();
        }
    }
}
