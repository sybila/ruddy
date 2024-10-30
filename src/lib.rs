// A lot of these things would eventually go into some kind of "internal" module, since
// we don't actually want to expose them in the public API?
pub mod node_id;
pub mod task_cache;

pub trait Bdd {}

// TODO:
//     These functions are only valid on 32-bit and/or 64-bit systems. We can hopefully assume that
//     this library will only be used on 64-bit computers, but we should probably add some safety
//     check that will fail if that is not the case. To deal with this, I have added these two
//     functions that can be used to convert u32/u64 numbers. Once we figure out how to deal with
//     this limitation, we will have a single central place where all unsafe conversions happen,
//     so that we don't have to check the rest of the source code.

#[inline(always)]
#[allow(clippy::as_conversions)]
pub fn usize_is_at_least_32_bits(x: u32) -> usize {
    x as usize
}

#[inline(always)]
#[allow(clippy::as_conversions)]
pub fn usize_is_at_least_64_bits(x: u64) -> usize {
    x as usize
}
