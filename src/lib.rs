/*!
Ruddy is a minimalistic, high-performance Rust library for [(reduced and ordered) binary decision diagrams](https://en.wikipedia.org/wiki/Binary_decision_diagram) (BDDs), which are a compact representation of Boolean functions.

> The name `Ruddy` is a blend of `BuDDy` (one of the first widely used BDD libraries) and `Rust`. However,
> `Ruddy` is not affiliated with the developers of `BuDDy` in any formal way.

## Features

Most popular BDD implementations use a *shared* representation, where all BDD nodes are stored in a single pool managed by a manager. This allows subgraphs shared between BDDs to be stored only once in memory. An alternative is the *split* representation, where each BDD owns its nodes (e.g., as an array).

When BDDs do not meaningfully share nodes, split representations can be faster and more memory-efficient. Ruddy implements both representations, letting users choose what works best for their use case.

Currently, Ruddy supports the following features:
- Binary logical operators (conjunction, xor, ...) and negation.
- Existential and universal quantification, also combined with a binary operator.
- Satisfying valuations and paths iterators along with methods to count them.
- Export to DOT.
- Binary and text (de)serialization of split BDDs.

## Usage

An example of using the Ruddy split implementation:
```rust
use ruddy::split::Bdd;
use ruddy::VariableId;

// Create two variables v0 and v1.
let v0 = VariableId::new(0);
let v1 = VariableId::new(1);

// Make the BDDs representing `v0=true` and `v1=true`.
let a = Bdd::new_literal(v0, true);
let b = Bdd::new_literal(v1, true);

// Calculate BDD `a => b`.
let imp = a.implies(&b);

// Calculate BDD `!a ∨ b`.
let equiv_imp = a.not().or(&b);

// Check that they are equivalent.
assert!(imp.iff(&equiv_imp).is_true());
```
or, with the shared implementation:
```rust
use ruddy::shared::BddManager;
use ruddy::VariableId;

// Create the manager.
let mut manager = BddManager::new();

// Create two variables v0 and v1.
let v0 = VariableId::new(0);
let v1 = VariableId::new(1);

// Make the BDDs representing `v0=true` and `v1=true`.
let a = manager.new_bdd_literal(v0, true);
let b = manager.new_bdd_literal(v1, true);

// Calculate BDD `a => b`.
let imp = manager.implies(&a, &b);

// Calculate BDD `!a ∨ b`.
let not_a = manager.not(&a);
let equiv_imp = manager.or(&not_a, &b);

// Check that they are equivalent.
assert!(imp == equiv_imp);
```
More complex examples can be found in the `examples` folder.
*/
mod bdd_node;
pub mod boolean_operators;
mod conversion;
mod iterators;
mod node_id;
mod node_table;
pub mod shared;
pub mod split;
mod task_cache;
mod variable_id;

pub use node_id::DeserializeIdError;
pub use node_id::NodeId;
pub use node_id::NodeIdAny;
pub use variable_id::VariableId;

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
pub(crate) const fn usize_is_at_least_32_bits(x: u32) -> usize {
    x as usize
}

/// A conversion function asserting that we are running on (at least) a 64-bit platform.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn usize_is_at_least_64_bits(x: u64) -> usize {
    x as usize
}
