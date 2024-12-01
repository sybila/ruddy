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
    /// Calculate a `Bdd32` representing the formula $\phi \land \psi$, where
    /// $\phi$ and $\psi$ are represented by the BDD `self` and `other`, respectively.
    pub fn and(&self, other: &Bdd32) -> Bdd32 {
        self.apply_default(other, boolean_operators::and)
    }

    /// Calculate a `Bdd32` representing the formula $\phi \lor \psi$, where
    /// $\phi$ and $\psi$ are represented by the BDD `self` and `other`, respectively.
    pub fn or(&self, other: &Bdd32) -> Bdd32 {
        self.apply_default(other, boolean_operators::or)
    }

    /// Calculate a `Bdd32`representing the formula $\phi \oplus \psi$, where
    /// $\phi$ and $\psi$ are represented by the BDD `self` and `other`, respectively.
    pub fn xor(&self, other: &Bdd32) -> Bdd32 {
        self.apply_default(other, boolean_operators::xor)
    }

    /// Calculate a `Bdd32` representing the formula $\phi \Rightarrow \psi$, where
    /// $\phi$ and $\psi$ are represented by the BDD `self` and `other`, respectively.
    pub fn implies(&self, other: &Bdd32) -> Bdd32 {
        self.apply_default(other, boolean_operators::implies)
    }

    /// Calculate a `Bdd32` representing the formula $\phi \Leftrightarrow \psi$, where
    /// $\phi$ and $\psi$ are represented by the BDD `self` and `other`, respectively.
    pub fn iff(&self, other: &Bdd32) -> Bdd32 {
        self.apply_default(other, boolean_operators::iff)
    }

    /// Like [Bdd32::apply], but constructs the `task_cache` and `node_table` itself,
    /// and returns the resulting `Bdd32`.
    fn apply_default(&self, other: &Bdd32, operator: fn(NodeId32, NodeId32) -> NodeId32) -> Bdd32 {
        let mut node_table = NodeTable32::new();
        let mut task_cache = TaskCache32::with_log_size(1);

        self.apply(other, operator, &mut task_cache, &mut node_table);

        node_table.into()
    }

    /// A universal function used for implementing logical operators.
    ///
    /// The `operator` function is the logical operator to be applied to the BDDs.
    /// It is expected to be defined mainly for terminal [NodeId32] arguments. However,
    /// since some logical operators can return the result even if only one of the arguments
    /// is a terminal node, it has to work for non-terminal nodes as well. If the result is not
    /// yet known, the function should return the undefined node id. For example, the logical
    /// operator implementing logical or would be defined as:
    /// ```text
    /// or(NodeId32(1), NodeId32(_)) -> NodeId32(1)
    /// or(NodeId32(_), NodeId32(1)) -> NodeId32(1)
    /// or(NodeId32(0), NodeId32(0)) -> NodeId32(0)
    /// or(NodeId32(_), NodeId32(_)) -> NodeId32::undefined(),
    /// ```
    /// The function does not return a `Bdd32` directly, as it expects the
    /// resulting BDD to be stored inside the `node_table`.
    pub fn apply<BooleanOperator>(
        &self,
        other: &Bdd32,
        operator: BooleanOperator,
        task_cache: &mut dyn TaskCache<Id = NodeId32>,
        node_table: &mut dyn NodeTable<Id = NodeId32, VarId = VarIdPacked32>,
    ) where
        BooleanOperator: Fn(NodeId32, NodeId32) -> NodeId32,
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
    }
}
