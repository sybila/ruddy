//! Defines binary logic operators on [Bdd] objects using the `apply` algorithm.
//!
use crate::bdd::Bdd;
use crate::{
    bdd::Bdd32,
    bdd_node::BddNode,
    boolean_operators,
    node_id::{BddNodeId, NodeId32},
    node_table::{NodeTable, NodeTable32},
    task_cache::{TaskCache, TaskCache32},
    variable_id::{VarIdPacked32, VariableId},
};

impl Bdd32 {
    /// Calculate a [Bdd32] representing the boolean formula `self && other` (conjunction).
    pub fn and(&self, other: &Bdd32) -> Bdd32 {
        self.apply_default(other, boolean_operators::and)
    }

    /// Calculate a [Bdd32] representing the boolean formula `self || other` (disjunction).
    pub fn or(&self, other: &Bdd32) -> Bdd32 {
        self.apply_default(other, boolean_operators::or)
    }

    /// Calculate a [Bdd32] representing the boolean formula `self ^ other` (xor; non-equivalence).
    pub fn xor(&self, other: &Bdd32) -> Bdd32 {
        self.apply_default(other, boolean_operators::xor)
    }

    /// Calculate a [Bdd32] representing the boolean formula `self => other` (implication).
    pub fn implies(&self, other: &Bdd32) -> Bdd32 {
        self.apply_default(other, boolean_operators::implies)
    }

    /// Calculate a [Bdd32] representing the boolean formula `self <=> other` (equivalence).
    pub fn iff(&self, other: &Bdd32) -> Bdd32 {
        self.apply_default(other, boolean_operators::iff)
    }

    /// Like [Bdd32::apply], but constructs the `task_cache` and `node_table` itself,
    /// and returns the resulting `Bdd32`.
    fn apply_default(&self, other: &Bdd32, operator: fn(NodeId32, NodeId32) -> NodeId32) -> Bdd32 {
        let mut node_table = NodeTable32::new();
        let mut task_cache = TaskCache32::with_log_size(1);

        let root = self.apply(other, operator, &mut task_cache, &mut node_table);
        unsafe { Bdd32::from_table(root, node_table) }
    }

    /// A universal function used for implementing logical operators.
    ///
    /// The `operator` function is the logical operator to be applied to the BDDs.
    /// It is expected to be defined mainly for terminal [NodeId32] arguments. However,
    /// since some logical operators can return the result even if only one of the arguments
    /// is a terminal node, it has to work for non-terminal nodes as well. If the result is not
    /// yet known, the function should return [NodeId32::undefined]. For example, the logical
    /// operator implementing disjunction would be defined as:
    /// ```text
    /// or(NodeId32(1), NodeId32(_)) -> NodeId32(1)
    /// or(NodeId32(_), NodeId32(1)) -> NodeId32(1)
    /// or(NodeId32(0), NodeId32(0)) -> NodeId32(0)
    /// or(NodeId32(_), NodeId32(_)) -> NodeId32::undefined(),
    /// ```
    /// The function does not return a `Bdd32` directly, as it expects the
    /// resulting BDD to be stored inside the `node_table`.
    ///
    /// The method returns the ID of the root node, stored in the `node_table`.
    pub fn apply<BooleanOperator, Cache, Table>(
        &self,
        other: &Bdd32,
        operator: BooleanOperator,
        task_cache: &mut Cache,
        node_table: &mut Table,
    ) -> NodeId32
    where
        BooleanOperator: Fn(NodeId32, NodeId32) -> NodeId32,
        Cache: TaskCache<Id = NodeId32>,
        Table: NodeTable<Id = NodeId32, VarId = VarIdPacked32>,
    {
        let mut stack = Vec::new();
        let mut results = Vec::new();

        stack.push((self.root(), other.root(), VarIdPacked32::undefined()));

        while let Some((left_id, right_id, variable)) = stack.pop() {
            // Check if the result is known because the operation short-circuited
            // the computation.
            let result = operator(left_id, right_id);
            if !result.is_undefined() {
                results.push(result);
                continue;
            }

            if variable.is_undefined() {
                // The task has not been expanded yet

                let left_node = unsafe { self.get_node_unchecked(left_id) };
                let right_node = unsafe { other.get_node_unchecked(right_id) };

                let use_cache = left_node.has_many_parents() || right_node.has_many_parents();

                if use_cache {
                    let result = task_cache.get((left_id, right_id));
                    if !result.is_undefined() {
                        results.push(result);
                        continue;
                    }
                }

                let left_variable = left_node.variable();
                let right_variable = right_node.variable();

                let mut variable = left_variable.min(right_variable);

                let (left_low, left_high) = if variable == left_variable {
                    (left_node.low(), left_node.high())
                } else {
                    (left_id, left_id)
                };

                let (right_low, right_high) = if variable == right_variable {
                    (right_node.low(), right_node.high())
                } else {
                    (right_id, right_id)
                };

                variable.set_use_cache(use_cache);
                stack.push((left_id, right_id, variable));
                stack.push((left_high, right_high, VarIdPacked32::undefined()));
                stack.push((left_low, right_low, VarIdPacked32::undefined()));

                continue;
            }
            let high_result = results.pop().expect("high result present in result stack");
            let low_result = results.pop().expect("low result present in result stack");

            let node_id = node_table.ensure_node(variable, low_result, high_result);
            if variable.use_cache() {
                task_cache.set((left_id, right_id), node_id);
            }
            results.push(node_id);
        }

        let root = results.pop().expect("root result present in result stack");
        debug_assert!(results.is_empty());
        root
    }
}

#[cfg(test)]
mod tests {
    use crate::bdd::{Bdd, Bdd32};
    use crate::variable_id::VarIdPacked32;

    #[test]
    pub fn basic_apply_invariants() {
        // These are obviously not all invariants/equalities, but at least something to
        // check that we have the major corner cases covered.

        let a = Bdd32::new_literal(VarIdPacked32::new(1), true);
        let b = Bdd32::new_literal(VarIdPacked32::new(2), true);
        let a_n = Bdd32::new_literal(VarIdPacked32::new(1), false);
        let b_n = Bdd32::new_literal(VarIdPacked32::new(2), false);
        let tt = Bdd32::new_true();
        let ff = Bdd32::new_false();

        assert!(Bdd32::structural_eq(&a.and(&a), &a));
        assert!(Bdd32::structural_eq(&a.and(&tt), &a));
        assert!(Bdd32::structural_eq(&a.and(&ff), &ff));
        assert!(Bdd32::structural_eq(&a.and(&b), &b.and(&a)));

        assert!(Bdd32::structural_eq(&a.or(&a), &a));
        assert!(Bdd32::structural_eq(&a.or(&tt), &tt));
        assert!(Bdd32::structural_eq(&a.or(&ff), &a));
        assert!(Bdd32::structural_eq(&a.or(&b), &b.or(&a)));

        assert!(Bdd32::structural_eq(&a.implies(&a), &tt));
        assert!(Bdd32::structural_eq(&a.implies(&tt), &tt));
        assert!(Bdd32::structural_eq(&a.implies(&ff), &a_n));
        assert!(Bdd32::structural_eq(&a.implies(&b), &a_n.or(&b)));

        assert!(Bdd32::structural_eq(&a.xor(&a), &ff));
        assert!(Bdd32::structural_eq(&a.xor(&tt), &a_n));
        assert!(Bdd32::structural_eq(&a.xor(&ff), &a));
        assert!(Bdd32::structural_eq(
            &a.xor(&b),
            &a.and(&b_n).or(&a_n.and(&b))
        ));

        assert!(Bdd32::structural_eq(&a.iff(&a), &tt));
        assert!(Bdd32::structural_eq(&a.iff(&tt), &a));
        assert!(Bdd32::structural_eq(&a.iff(&ff), &a_n));
        assert!(Bdd32::structural_eq(
            &a.iff(&b),
            &a.and(&b).or(&a_n.and(&b_n))
        ));
    }
}
