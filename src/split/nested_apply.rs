//! Defines binary logic operators on [`Bdd`] objects using the `nested_apply` algorithm.
//!

#![allow(clippy::type_complexity)]

use rustc_hash::FxHashSet;

use crate::{
    bdd_node::BddNodeAny,
    boolean_operators::{self, BooleanOperator},
    conversion::UncheckedInto,
    node_id::{NodeId16, NodeId32, NodeId64, NodeIdAny},
    node_table::{NodeTable16, NodeTable32, NodeTable64, NodeTableAny},
    split::bdd::{AsBdd, Bdd, Bdd16, Bdd32, Bdd64, BddAny},
    task_cache::{TaskCache16, TaskCache32, TaskCache64, TaskCacheAny},
    variable_id::{VarIdPacked16, VarIdPacked32, VarIdPacked64, VarIdPackedAny, VariableId},
};

use super::bdd::BddInner;

impl Bdd {
    /// Calculates a `Bdd` with the given `variables` eliminated using existential quantification.
    pub fn exists(&self, variables: &[VariableId]) -> Bdd {
        self.binary_op_with_exists(self, boolean_operators::And, variables)
    }

    /// Calculates a `Bdd` with the given `variables` eliminated using universal quantification.
    pub fn for_all(&self, variables: &[VariableId]) -> Bdd {
        self.binary_op_with_for_all(self, boolean_operators::And, variables)
    }

    /// Applies the `operator` to the `Bdd`s and eliminates the given `variables` using existential
    /// quantification from the result.
    pub fn binary_op_with_exists<TBooleanOp: BooleanOperator>(
        &self,
        other: &Bdd,
        operator: TBooleanOp,
        variables: &[VariableId],
    ) -> Bdd {
        self.nested_apply(other, operator, boolean_operators::Or, variables)
    }

    /// Applies the `operator` to the `Bdd`s and eliminates the given `variables` using universal
    /// quantification from the result.
    pub fn binary_op_with_for_all<TBooleanOp: BooleanOperator>(
        &self,
        other: &Bdd,
        operator: TBooleanOp,
        variables: &[VariableId],
    ) -> Bdd {
        self.nested_apply(other, operator, boolean_operators::And, variables)
    }

    /// Applies the `outer_op` to the `Bdd`s. On each node of the resulting `Bdd`,
    /// if its variable is in `variables`, the node is replaced with the result
    /// of applying `inner_op` to its low and high children.
    ///
    /// This function is useful for implementing combinations of applying binary
    /// operators and quantification. For example, if `outer_op` is [`boolean_operators::And`]
    /// and `inner_op` is [`boolean_operators::Or`], this combination corresponds to the
    /// "relational product" operation.
    pub fn nested_apply<TOuterOp: BooleanOperator, TInnerOp: BooleanOperator>(
        &self,
        other: &Bdd,
        outer_op: TOuterOp,
        inner_op: TInnerOp,
        variables: &[VariableId],
    ) -> Bdd {
        match (&self.0, &other.0) {
            (BddInner::Size16(left), BddInner::Size16(right)) => {
                nested_apply_16_bit_input(left, right, outer_op, inner_op, variables)
            }
            (BddInner::Size16(left), BddInner::Size32(right)) => {
                nested_apply_32_bit_input(left, right, outer_op, inner_op, variables)
            }
            (BddInner::Size16(left), BddInner::Size64(right)) => {
                nested_apply_64_bit_input(left, right, outer_op, inner_op, variables)
            }
            (BddInner::Size32(left), BddInner::Size16(right)) => {
                nested_apply_32_bit_input(left, right, outer_op, inner_op, variables)
            }
            (BddInner::Size32(left), BddInner::Size32(right)) => {
                nested_apply_32_bit_input(left, right, outer_op, inner_op, variables)
            }
            (BddInner::Size32(left), BddInner::Size64(right)) => {
                nested_apply_64_bit_input(left, right, outer_op, inner_op, variables)
            }
            (BddInner::Size64(left), BddInner::Size16(right)) => {
                nested_apply_64_bit_input(left, right, outer_op, inner_op, variables)
            }
            (BddInner::Size64(left), BddInner::Size32(right)) => {
                nested_apply_64_bit_input(left, right, outer_op, inner_op, variables)
            }
            (BddInner::Size64(left), BddInner::Size64(right)) => {
                nested_apply_64_bit_input(left, right, outer_op, inner_op, variables)
            }
        }
        .shrink()
    }
}

/// Data used to store the state of the inner `apply` algorithm.
#[derive(Debug, Clone)]
pub(crate) struct InnerApplyState<TNodeTable: NodeTableAny> {
    pub(crate) stack: Vec<(TNodeTable::Id, TNodeTable::Id, TNodeTable::VarId)>,
    pub(crate) results: Vec<TNodeTable::Id>,
}

macro_rules! impl_inner_apply_state_conversion {
    ($from_result:ident, $to_result:ident) => {
        impl From<InnerApplyState<$from_result>> for InnerApplyState<$to_result> {
            fn from(state: InnerApplyState<$from_result>) -> Self {
                InnerApplyState {
                    stack: state
                        .stack
                        .into_iter()
                        .map(|(left, right, var)| (left.into(), right.into(), var.into()))
                        .collect(),
                    results: state.results.into_iter().map(|id| id.into()).collect(),
                }
            }
        }
    };
}

impl_inner_apply_state_conversion!(NodeTable16, NodeTable32);
impl_inner_apply_state_conversion!(NodeTable32, NodeTable64);

impl<TNodeTable: NodeTableAny> Default for InnerApplyState<TNodeTable> {
    fn default() -> Self {
        Self {
            stack: Vec::new(),
            results: Vec::new(),
        }
    }
}

impl<TNodeTable: NodeTableAny> InnerApplyState<TNodeTable> {
    /// Returns `true` if the state is empty.
    pub(crate) fn is_empty(&self) -> bool {
        debug_assert!(!self.stack.is_empty() || self.results.is_empty());
        self.stack.is_empty()
    }
}

/// Similar to [`apply_any`], but operates on the nodes inside the `node_table`.
pub(crate) fn inner_apply_any<
    TNodeTable: NodeTableAny,
    TTaskCache: TaskCacheAny<ResultId = TNodeTable::Id>,
    TBooleanOp: Fn(TNodeTable::Id, TNodeTable::Id) -> TNodeTable::Id,
>(
    operator: TBooleanOp,
    state: InnerApplyState<TNodeTable>,
    task_cache: &mut TTaskCache,
    node_table: &mut TNodeTable,
) -> Result<TNodeTable::Id, InnerApplyState<TNodeTable>> {
    debug_assert!(!state.is_empty());

    let InnerApplyState {
        mut stack,
        mut results,
    } = state;

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

            let left_node = unsafe { node_table.get_node_unchecked(left_id) };
            let right_node = unsafe { node_table.get_node_unchecked(right_id) };

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
            stack.push((left_high, right_high, TNodeTable::VarId::undefined()));
            stack.push((left_low, right_low, TNodeTable::VarId::undefined()));

            continue;
        }
        let high_result = results.pop().expect("high result present in result stack");
        let low_result = results.pop().expect("low result present in result stack");

        let node_id = match node_table.ensure_node(variable, low_result, high_result) {
            Ok(id) => id,
            Err(_) => {
                results.push(low_result);
                results.push(high_result);
                stack.push((left_id, right_id, variable));

                return Err(InnerApplyState { stack, results });
            }
        };

        if variable.use_cache() {
            task_cache.set((left_id, right_id), node_id);
        }
        results.push(node_id);
    }

    let root = results.pop().expect("root present in result stack");
    debug_assert!(results.is_empty());

    Ok(root)
}

type NodeId<B> = <B as BddAny>::Id;

/// Data used to store the state of the nested `apply` algorithm.
#[derive(Debug)]
struct NestedApplyState<
    TResultBdd: BddAny,
    TNodeId1,
    TNodeId2,
    TOuterCache,
    TInnerCache,
    TNodeTable: NodeTableAny,
> {
    stack: Vec<(TNodeId1, TNodeId2, TResultBdd::VarId)>,
    results: Vec<TResultBdd::Id>,
    outer_task_cache: TOuterCache,
    inner_task_cache: TInnerCache,
    node_table: TNodeTable,
    inner_state: InnerApplyState<TNodeTable>,
}

/// Generates a variant of [`NestedApplyState`] with bit-width defined by `out_bdd` type and
/// associated `task_cache` and `node_table` types. Also generates a default "constructor" for
/// this type alias. The variants are special in the sense that they specify the exact output
/// type of the BDD and associated auxiliary structures. What is left as parametric are the
/// types of the input BDDs and the outer tash cache (which can change during computation).
macro_rules! impl_nested_apply_state_variant {
    ($variant_name:ident, $constructor:ident, $out_bdd:ident, $task_cache:ident, $node_table:ident) => {
        type $variant_name<Id1, Id2, TOuterCache = $task_cache> =
            NestedApplyState<$out_bdd, Id1, Id2, TOuterCache, $task_cache, $node_table>;

        fn $constructor<TBdd1: AsBdd<$out_bdd>, TBdd2: AsBdd<$out_bdd>>(
            left: &TBdd1,
            right: &TBdd2,
        ) -> $variant_name<TBdd1::Id, TBdd2::Id> {
            let undefined_var = <$out_bdd as BddAny>::VarId::undefined();
            NestedApplyState {
                stack: vec![(left.root(), right.root(), undefined_var)],
                results: Vec::new(),
                outer_task_cache: $task_cache::default(),
                inner_task_cache: $task_cache::default(),
                node_table: $node_table::default(),
                inner_state: InnerApplyState::default(),
            }
        }
    };
}

impl_nested_apply_state_variant!(
    NestedApplyState16,
    default_state_16,
    Bdd16,
    TaskCache16,
    NodeTable16
);

impl_nested_apply_state_variant!(
    NestedApplyState32,
    default_state_32,
    Bdd32,
    TaskCache32,
    NodeTable32
);

impl_nested_apply_state_variant!(
    NestedApplyState64,
    default_state_64,
    Bdd64,
    TaskCache64,
    NodeTable64
);

/// Generates a conversion between two variants of [`NestedApplyState`] (see also
/// [`crate::impl_nested_apply_state_variant!()`]). The conversion uses
/// default `From` implementations, and is generic in the outer task cache argument,
/// meaning any conversion between variants should be supported as long as the
/// underlying tables are compatible.
macro_rules! impl_nested_apply_state_conversion {
    ($from_variant:ident, $to_variant:ident) => {
        impl<TNodeId1, TNodeId2, TCacheIn, TCacheOut: From<TCacheIn>>
            From<$from_variant<TNodeId1, TNodeId2, TCacheIn>>
            for $to_variant<TNodeId1, TNodeId2, TCacheOut>
        {
            fn from(state: $from_variant<TNodeId1, TNodeId2, TCacheIn>) -> Self {
                Self {
                    stack: state
                        .stack
                        .into_iter()
                        .map(|(left, right, var)| (left, right, var.into()))
                        .collect(),
                    results: state.results.into_iter().map(|id| id.into()).collect(),
                    outer_task_cache: state.outer_task_cache.into(),
                    inner_task_cache: state.inner_task_cache.into(),
                    node_table: state.node_table.into(),
                    inner_state: state.inner_state.into(),
                }
            }
        }
    };
}

impl_nested_apply_state_conversion!(NestedApplyState16, NestedApplyState32);
impl_nested_apply_state_conversion!(NestedApplyState32, NestedApplyState64);

/// A nested `apply` algorithm that interleaves two passes of the `apply` algorithm.
/// First, `outer_op` is applied to `left` and `right`. If the resulting node's
/// variable triggers `trigger`, `inner_op` is applied to the node's children.
fn nested_apply_any<
    TResultBdd: BddAny,
    TBdd1: AsBdd<TResultBdd>,
    TBdd2: AsBdd<TResultBdd>,
    TOuterOp: BooleanOperator,
    TInnerOp: BooleanOperator,
    TTrigger: Fn(TResultBdd::VarId) -> bool,
    TOuterCache: TaskCacheAny<ResultId = TResultBdd::Id>,
    TInnerCache: TaskCacheAny<ResultId = TResultBdd::Id>,
    TNodeTable: NodeTableAny<Id = TResultBdd::Id, VarId = TResultBdd::VarId, Node = TResultBdd::Node>,
>(
    left: &TBdd1,
    right: &TBdd2,
    outer_op: TOuterOp,
    inner_op: TInnerOp,
    trigger: TTrigger,
    state: NestedApplyState<TResultBdd, TBdd1::Id, TBdd2::Id, TOuterCache, TInnerCache, TNodeTable>,
) -> Result<
    TResultBdd,
    NestedApplyState<TResultBdd, TBdd1::Id, TBdd2::Id, TOuterCache, TInnerCache, TNodeTable>,
> {
    let outer_op = outer_op.for_split::<TBdd1::Id, TBdd2::Id, TResultBdd::Id>();
    let inner_op = inner_op.for_split::<TResultBdd::Id, TResultBdd::Id, TResultBdd::Id>();

    let NestedApplyState {
        mut stack,
        mut results,
        mut outer_task_cache,
        mut inner_task_cache,
        mut node_table,
        mut inner_state,
    } = state;

    while let Some((left_id, right_id, variable)) = stack.pop() {
        // Check if the result is known because the operation short-circuited
        // the computation.
        let result = outer_op(left_id, right_id);
        if !result.is_undefined() {
            results.push(result);
            continue;
        }

        if variable.is_undefined() {
            // The task has not been expanded yet

            let left_node = unsafe { left.get_node_unchecked(left_id) };
            let right_node = unsafe { right.get_node_unchecked(right_id) };

            let use_cache = left_node.has_many_parents() || right_node.has_many_parents();

            if use_cache {
                let result = outer_task_cache.get((left_id, right_id));
                if !result.is_undefined() {
                    results.push(result);
                    continue;
                }
            }

            let left_variable = left_node.variable().into();
            let right_variable = right_node.variable().into();

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
            stack.push((left_high, right_high, TResultBdd::VarId::undefined()));
            stack.push((left_low, right_low, TResultBdd::VarId::undefined()));

            continue;
        }
        let high_result = results.pop().expect("high result present in result stack");
        let low_result = results.pop().expect("low result present in result stack");

        let node_id = if trigger(variable) {
            if inner_state.is_empty() {
                // Construct the starting state for inner `apply`
                inner_state = InnerApplyState {
                    stack: vec![(low_result, high_result, TResultBdd::VarId::undefined())],
                    results: Vec::new(),
                };
            }

            match inner_apply_any(
                &inner_op,
                // Make sure that in the next iteration, the inner state is
                // empty, so that it is correctly initialized.
                std::mem::take(&mut inner_state),
                &mut inner_task_cache,
                &mut node_table,
            ) {
                Ok(id) => id,
                Err(inner_state) => {
                    results.push(low_result);
                    results.push(high_result);
                    stack.push((left_id, right_id, variable));

                    return Err(NestedApplyState {
                        stack,
                        results,
                        outer_task_cache,
                        inner_task_cache,
                        node_table,
                        inner_state,
                    });
                }
            }
        } else {
            match node_table.ensure_node(variable, low_result, high_result) {
                Ok(id) => id,
                Err(_) => {
                    results.push(low_result);
                    results.push(high_result);
                    stack.push((left_id, right_id, variable));
                    return Err(NestedApplyState {
                        stack,
                        results,
                        outer_task_cache,
                        inner_task_cache,
                        node_table,
                        inner_state: InnerApplyState::default(),
                    });
                }
            }
        };
        if variable.use_cache() {
            outer_task_cache.set((left_id, right_id), node_id);
        }
        results.push(node_id);
    }

    let root = results.pop().expect("root result present in result stack");
    debug_assert!(results.is_empty());

    Ok(unsafe { node_table.export_reachable_bdd(root) })
}

/// Like [`nested_apply_any_default_state`], but specifically for 16-bit BDDs.
///
/// The function automatically grows the BDD to 32 or 64 bits if the result does not fit.
fn nested_apply_16_bit_input<TOuterOp: BooleanOperator, TInnerOp: BooleanOperator>(
    left: &Bdd16,
    right: &Bdd16,
    outer_op: TOuterOp,
    inner_op: TInnerOp,
    variables: &[VariableId],
) -> Bdd {
    let variable_set: FxHashSet<VarIdPacked16> =
        FxHashSet::from_iter(variables.iter().map(|&v| v.unchecked_into()));

    let trigger = |var: VarIdPacked16| variable_set.contains(&var);
    let state = default_state_16(left, right);
    let result_16 = nested_apply_any(left, right, outer_op, inner_op, trigger, state);
    let state = match result_16 {
        Ok(bdd) => return bdd.into(),
        Err(state) => state,
    };

    let trigger = |var: VarIdPacked32| variable_set.contains(&var.unchecked_into());
    let state: NestedApplyState32<NodeId16, NodeId16, TaskCache16<NodeId32>> = state.into();
    let result_32 = nested_apply_any(left, right, outer_op, inner_op, trigger, state);
    let state = match result_32 {
        Ok(bdd) => return bdd.into(),
        Err(state) => state,
    };

    let trigger = |var: VarIdPacked64| variable_set.contains(&var.unchecked_into());
    let state: NestedApplyState64<NodeId16, NodeId16, TaskCache16<NodeId64>> = state.into();
    let result_64 = nested_apply_any(left, right, outer_op, inner_op, trigger, state);
    match result_64 {
        Ok(bdd) => bdd.into(),
        Err(_state) => unreachable!("BDD does not fit into 64-bit bounds."),
    }
}

/// Like [`nested_apply_any_default_state`], but specifically for at most 32-bit wide BDDs.
///
/// The function automatically grows the BDD to 64 bits if the result does not fit.
fn nested_apply_32_bit_input<
    TBdd1: AsBdd<Bdd32> + AsBdd<Bdd64>,
    TBdd2: AsBdd<Bdd32> + AsBdd<Bdd64>,
    TOuterOp: BooleanOperator,
    TInnerOp: BooleanOperator,
>(
    left: &TBdd1,
    right: &TBdd2,
    outer_op: TOuterOp,
    inner_op: TInnerOp,
    variables: &[VariableId],
) -> Bdd {
    let variable_set: FxHashSet<VarIdPacked32> =
        FxHashSet::from_iter(variables.iter().map(|&v| v.unchecked_into()));

    let trigger = |var: VarIdPacked32| variable_set.contains(&var);
    let state = default_state_32(left, right);
    let result_32 = nested_apply_any(left, right, outer_op, inner_op, trigger, state);
    let state = match result_32 {
        Ok(bdd) => return bdd.into(),
        Err(state) => state,
    };

    let trigger = |var: VarIdPacked64| variable_set.contains(&var.unchecked_into());
    let state: NestedApplyState64<NodeId<TBdd1>, NodeId<TBdd2>, TaskCache32<NodeId64>> =
        state.into();
    let result_64 = nested_apply_any(left, right, outer_op, inner_op, trigger, state);
    match result_64 {
        Ok(bdd) => bdd.into(),
        Err(_state) => unreachable!("BDD does not fit into 64-bit bounds."),
    }
}

/// Like [`nested_apply_any_default_state`], but specifically for at most 64-bit wide BDDs.
fn nested_apply_64_bit_input<
    TBdd1: AsBdd<Bdd64>,
    TBdd2: AsBdd<Bdd64>,
    TOuterOp: BooleanOperator,
    TInnerOp: BooleanOperator,
>(
    left: &TBdd1,
    right: &TBdd2,
    outer_op: TOuterOp,
    inner_op: TInnerOp,
    variables: &[VariableId],
) -> Bdd {
    let variable_set: FxHashSet<VarIdPacked64> =
        FxHashSet::from_iter(variables.iter().map(|&v| v.unchecked_into()));

    let trigger = |var: VarIdPacked64| variable_set.contains(&var);
    let state = default_state_64(left, right);
    let result_64 = nested_apply_any(left, right, outer_op, inner_op, trigger, state);
    match result_64 {
        Ok(bdd) => bdd.into(),
        Err(_state) => unreachable!("BDD does not fit into 64-bit bounds."),
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        bdd_node::BddNodeAny,
        boolean_operators::{self, BooleanOperator},
        node_id::NodeId32,
        node_table::{NodeTable32, NodeTableAny},
        split::{
            bdd::{Bdd, Bdd32, BddAny},
            nested_apply::{InnerApplyState, inner_apply_any},
        },
        task_cache::TaskCache32,
        variable_id::{VarIdPacked32, VarIdPackedAny, VariableId},
    };

    #[test]
    fn basic_nested_apply_invariants() {
        let v_a = VariableId::from(1u32);
        let v_b = VariableId::from(2u32);

        let a = Bdd::new_literal(v_a, true);
        let b = Bdd::new_literal(v_b, true);
        let tt = Bdd::new_true();
        let ff = Bdd::new_false();

        // True / false constants
        assert!(Bdd::structural_eq(&tt.exists(&[v_a]), &tt));
        assert!(Bdd::structural_eq(&ff.exists(&[v_a]), &ff));
        assert!(Bdd::structural_eq(&tt.for_all(&[v_a]), &tt));
        assert!(Bdd::structural_eq(&ff.for_all(&[v_a]), &ff));

        // Quantifying over the same variable
        assert!(Bdd::structural_eq(&a.exists(&[v_a]), &tt));
        assert!(Bdd::structural_eq(&a.for_all(&[v_a]), &ff));

        // Quantifying over independent variables
        assert!(Bdd::structural_eq(&a.exists(&[v_b]), &a));
        assert!(Bdd::structural_eq(&a.for_all(&[v_b]), &a));

        // Quantifying over an empty set of variables
        let a_and_b = a.and(&b);
        assert!(Bdd::structural_eq(&a_and_b, &a_and_b.exists(&[])));
        assert!(Bdd::structural_eq(&a_and_b, &a_and_b.for_all(&[])));
    }

    #[test]
    fn nested_apply_quantifier_distributivity() {
        let v_x = VariableId::from(1u32);
        let v_y = VariableId::from(2u32);
        let v_z = VariableId::from(3u32);
        let v_w = VariableId::from(4u32);

        let x = Bdd::new_literal(v_x, true);
        let y = Bdd::new_literal(v_y, true);
        let z = Bdd::new_literal(v_z, true);
        let w = Bdd::new_literal(v_w, true);

        // Create f(x) = x ∧ y (depends on x)
        let f_x = x.and(&y);

        // Create g = z <=> w (independent of x)
        let g = z.iff(&w);

        // Calculate left side: ∃x.(f(x) ∧ g)
        let left_side = f_x.binary_op_with_exists(&g, boolean_operators::And, &[v_x]);

        // Calculate right side: (∃x.f(x)) ∧ g
        let right_side = f_x.exists(&[v_x]).and(&g);

        assert!(Bdd::structural_eq(&left_side, &right_side));

        // Also test for universal quantification: ∀x.(f(x) ∨ g) = (∀x.f(x)) ∨ g
        let left_side_forall = f_x.binary_op_with_for_all(&g, boolean_operators::Or, &[v_x]);
        let right_side_forall = f_x.for_all(&[v_x]).or(&g);

        assert!(Bdd::structural_eq(&left_side_forall, &right_side_forall));
    }

    #[test]
    fn nested_apply_quantifier_commutativity() {
        let v_x = VariableId::from(1u32);
        let v_y = VariableId::from(2u32);
        let v_z = VariableId::from(3u32);
        let v_w = VariableId::from(4u32);

        let x = Bdd::new_literal(v_x, true);
        let y = Bdd::new_literal(v_y, true);
        let z = Bdd::new_literal(v_z, true);
        let w = Bdd::new_literal(v_w, true);

        // Create a formula: ((x ∧ z ∧ w) ∨ w) ∧ (y xor z)
        let formula = x.and(&z).and(&w).or(&x.and(&z).or(&w)).and(&y.xor(&z));

        // Test that the order of quantification doesn't matter for the same quantifier
        let exists_x_then_y = formula.exists(&[v_x]).exists(&[v_y]);
        let exists_y_then_x = formula.exists(&[v_y]).exists(&[v_x]);
        let exists_xy = formula.exists(&[v_x, v_y]);

        dbg!(exists_x_then_y.node_count());
        dbg!(exists_y_then_x.node_count());
        dbg!(exists_xy.node_count());

        assert!(Bdd::structural_eq(&exists_x_then_y, &exists_y_then_x));
        assert!(Bdd::structural_eq(&exists_x_then_y, &exists_xy));

        // Create a formula: (x => z) ∧ (y => w)
        let formula = x.implies(&z).and(&y.implies(&w));

        let forall_x_then_y = formula.for_all(&[v_x]).for_all(&[v_y]);
        let forall_y_then_x = formula.for_all(&[v_y]).for_all(&[v_x]);
        let forall_xy = formula.for_all(&[v_x, v_y]);

        assert!(Bdd::structural_eq(&forall_x_then_y, &forall_y_then_x));
        assert!(Bdd::structural_eq(&forall_x_then_y, &forall_xy));
    }

    #[test]
    fn inner_apply() {
        let v1 = VarIdPacked32::new(1u32);
        let v2 = VarIdPacked32::new(2u32);
        let v3 = VarIdPacked32::new(3u32);

        let x1 = Bdd32::new_literal(v1, true);
        let x2 = Bdd32::new_literal(v2, true);
        let x3 = Bdd32::new_literal(v3, true);

        // Make bdd for x1 ∧ (x2 <=> x3)
        let iff = x2.iff(&x3).unwrap();
        let bdd = x1.and(&iff).unwrap();

        // Populate the node table as `inner_apply` expects it.
        let mut table = NodeTable32::default();
        let root = bdd.root();
        let low = bdd.get(root).unwrap().low();
        let high = bdd.get(root).unwrap().high();

        for node in bdd.nodes.iter().skip(2) {
            table
                .ensure_node(node.variable(), node.low(), node.high())
                .unwrap();
        }

        let state = InnerApplyState {
            stack: vec![(low, high, VarIdPacked32::undefined())],
            results: vec![],
        };
        let mut cache: TaskCache32<NodeId32> = TaskCache32::default();

        let result = inner_apply_any(
            boolean_operators::Or.for_split(),
            state,
            &mut cache,
            &mut table,
        )
        .unwrap();

        let result: Bdd32 = unsafe { table.export_reachable_bdd(result) };

        assert!(Bdd32::structural_eq(&iff, &result));
    }

    #[test]
    fn nested_apply_without_quantification_is_equivalent_to_apply() {
        let v1 = VariableId::from(1u32);
        let v2 = VariableId::from(2u32);
        let v3 = VariableId::from(3u32);
        let v4 = VariableId::from(4u32);
        let v5 = VariableId::from(5u32);

        // High variable IDs to force different BDD sizes
        let v_high16 = VariableId::from(1u16 << 8);
        let v_high32 = VariableId::from(1u32 << 24);

        // Create several complex BDDs
        let x1 = Bdd::new_literal(v1, true);
        let x2 = Bdd::new_literal(v2, true);
        let x3 = Bdd::new_literal(v3, true);
        let x4 = Bdd::new_literal(v4, true);
        let x5 = Bdd::new_literal(v5, true);

        let high16 = Bdd::new_literal(v_high16, true);
        let high32 = Bdd::new_literal(v_high32, true);

        // Create more complex BDDs
        let e1 = x1.implies(&x2).and(&x3.xor(&x4.iff(&x5)));
        let e2 = x1.and(&x2).or(&x3).xor(&(x4.and(&x5))).and(&high16);
        let e3 = x3.implies(&x4).or(&x5).iff(&(x1.or(&x2))).and(&high32);

        let nested_and =
            |b1: &Bdd, b2: &Bdd| b1.binary_op_with_exists(b2, boolean_operators::And, &[]);

        let nested_iff =
            |b1: &Bdd, b2: &Bdd| b1.binary_op_with_for_all(b2, boolean_operators::Iff, &[]);

        let result1 = nested_and(&e1, &e2);
        let result2 = e1.and(&e2);

        let result3 = nested_iff(&e1, &e3);
        let result4 = e1.iff(&e3);

        assert!(Bdd::structural_eq(&result1, &result2));
        assert!(Bdd::structural_eq(&result3, &result4));
    }

    #[test]
    fn nested_apply_growth_test() {
        // This is the same test as implemented for shared BDDs, but with standalone ones.

        fn ripple_carry_adder(num_vars: u32) -> Bdd {
            let mut result = Bdd::new_false();
            for x in 0..(num_vars / 2) {
                let x1 = Bdd::new_literal(VariableId::new(x), true);
                let x2 = Bdd::new_literal(VariableId::new(x + num_vars / 2), true);
                // To make it a bit more interesting, we always erase some previously used variable.
                // This means we are not computing ripple carry adder after all, but at least
                // it tests the nested `apply` operator.
                let and = Bdd::binary_op_with_exists(&x1, &x2, boolean_operators::And, &[]);
                result = Bdd::binary_op_with_exists(
                    &result,
                    &and,
                    boolean_operators::Or,
                    &[VariableId::new(x / 4)],
                );
            }
            result
        }

        let result = ripple_carry_adder(4);
        assert_eq!(result.node_count(), 6);

        let result = ripple_carry_adder(8);
        assert_eq!(result.node_count(), 24);

        let result = ripple_carry_adder(16);
        assert_eq!(result.node_count(), 256);

        let result = ripple_carry_adder(24);
        assert_eq!(result.node_count(), 2560);

        let result = ripple_carry_adder(32);
        assert_eq!(result.node_count(), 24576);

        let result = ripple_carry_adder(40);
        assert_eq!(result.node_count(), 229376);

        let result = ripple_carry_adder(42);
        assert_eq!(result.node_count(), 262144);
    }

    #[test]
    fn nested_apply_input_sizes() {
        for var_a in [1 << 12, 1 << 28, 1u64 << 60] {
            let var_a = VariableId::new_long(var_a).unwrap();
            for var_b in [1 << 12, 1 << 28, 1u64 << 60] {
                let var_b = VariableId::new_long(var_b).unwrap();

                let bdd_a = Bdd::new_literal(var_a, true);
                let bdd_b = Bdd::new_literal(var_b, true);
                // Exists a: a | b is a tautology.
                // Forall a,b: a | b is a contradiction.
                let exists =
                    Bdd::binary_op_with_exists(&bdd_a, &bdd_b, boolean_operators::Or, &[var_a]);
                assert!(exists.is_true());
                let forall = Bdd::binary_op_with_for_all(
                    &bdd_a,
                    &bdd_b,
                    boolean_operators::Or,
                    &[var_a, var_b],
                );
                assert!(forall.is_false());
            }
        }
    }
}
