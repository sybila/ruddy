// A lot of these things would eventually go into some kind of "internal" module, since
// we don't actually want to expose them in the public API?
mod apply;
pub mod bdd;
pub mod bdd_node;
pub mod boolean_operators;
mod conversion;
pub mod node_id;
pub mod node_table;
pub mod task_cache;
pub mod variable_id;

// TODO:
//     These functions are only valid on 32-bit and/or 64-bit systems. We can hopefully assume that
//     this library will only be used on 64-bit computers, but we should probably add some safety
//     check that will fail if that is not the case. To deal with this, I have added these two
//     functions that can be used to convert u32/u64 numbers. Once we figure out how to deal with
//     this limitation, we will have a single central place where all unsafe conversions happen,
//     so that we don't have to check the rest of the source code.

/// A conversion function asserting that we are running on (at least) a 32-bit platform.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
pub fn usize_is_at_least_32_bits(x: u32) -> usize {
    x as usize
}

/// A conversion function asserting that we are running on (at least) a 64-bit platform.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
pub fn usize_is_at_least_64_bits(x: u64) -> usize {
    x as usize
}
