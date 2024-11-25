use crate::{
    bdd::Bdd32,
    bdd_node::BddNode,
    node_id::{BddNodeId, NodeId32},
    node_table::NodeTable,
    task_cache::TaskCache,
    variable_id::{VarIdPacked32, VariableId},
};

impl Bdd32 {
    pub fn apply<BooleanOperator>(
        &self,
        other: &Bdd32,
        operator: BooleanOperator,
        task_cache: &mut dyn TaskCache<Id = NodeId32>,
        node_table: &mut dyn NodeTable<Id = NodeId32, VarId = VarIdPacked32>,
    ) where
        BooleanOperator: Fn(NodeId32, NodeId32) -> Option<NodeId32>,
    {
        let mut stack = Vec::with_capacity(1 << 11);
        let mut results = Vec::with_capacity(1 << 11);

        stack.push((self.root(), other.root(), VarIdPacked32::undefined()));

        while let Some((left_id, right_id, variable)) = stack.pop() {
            // Check if the result is known because the operation short-circuited
            // the computation.
            if let Some(result) = operator(left_id, right_id) {
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
    }
}
