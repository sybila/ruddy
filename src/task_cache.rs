use crate::node_id::BddNodeId;

/// Task cache is a "leaky" hash table that maps a pair of [BddNodeId] instances (representing
/// a "task") to another instance of [BddNodeId], representing the result of said task.
///
/// The table can "lose" any value that is stored in it, especially in case of a collision with
/// another value. The cache is responsible for growing itself when too many collisions occur.
pub trait TaskCache {
    type Id: BddNodeId;

    /// Retrieve the result value that is stored for the given `task`, or [BddNodeId::undefined]
    /// when the `task` has not been encountered before.
    fn get(task: (Self::Id, Self::Id)) -> Self::Id;

    /// Update the `result` value for the given `task`, possibly overwriting any previously
    /// stored data.
    fn set(task: (Self::Id, Self::Id), result: Self::Id);
}
