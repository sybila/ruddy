//! Provides the structures for working with split binary decision diagrams, mainly
//! [`Bdd`].

pub(crate) mod apply;
pub(crate) mod bdd;
pub(crate) mod bdd_iterator;
pub(crate) mod nested_apply;
mod serialization;

pub use crate::iterators::split::SatisfyingPaths;
pub use crate::iterators::split::SatisfyingValuations;
pub use bdd::Bdd;
pub use bdd_iterator::ValuationsIterator;
pub use serialization::BddDeserializationError;
