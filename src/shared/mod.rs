//! Provides the structures for working with shared binary decision diagrams, mainly
//! [`BddManager`] and [`Bdd`].

mod apply;
pub(crate) mod manager;
mod nested_apply;

pub use crate::iterators::shared::SatisfyingPaths;
pub use crate::iterators::shared::SatisfyingValuations;
pub use manager::Bdd;
pub use manager::BddManager;
pub use manager::GarbageCollection;
