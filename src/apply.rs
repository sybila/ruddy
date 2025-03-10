//! Defines binary logic operators on [`Bdd`] objects using the `apply` algorithm.
//!

#![allow(clippy::type_complexity)]

use std::fmt::Display;

use crate::{
    bdd::{AsBdd, Bdd, Bdd16, Bdd32, Bdd64, BddAny},
    bdd_node::BddNodeAny,
    boolean_operators,
    boolean_operators::{lift_operator, TriBool},
    node_id::{NodeId32, NodeId64, NodeIdAny},
    node_table::{NodeTable16, NodeTable32, NodeTable64, NodeTableAny},
    task_cache::{TaskCache16, TaskCache32, TaskCache64, TaskCacheAny},
    variable_id::VarIdPackedAny,
};

/// Like [`apply_any_default_state`], but specifically for 16-bit BDDs.
///
/// The function automatically grows the BDD to 32, or 64 bits if the result does not fit.
pub(crate) fn apply_16_bit_input<TTriBoolOp: Fn(TriBool, TriBool) -> TriBool>(
    left: &Bdd16,
    right: &Bdd16,
    operator: TTriBoolOp,
) -> Bdd {
    let data = match apply_any_default_state::<Bdd16, _, _, _, _, _>(
        left,
        right,
        lift_operator(&operator),
    ) {
        Ok(bdd) => return Bdd::Size16(bdd),
        Err(data) => data,
    };

    let data =
        match apply_any::<Bdd32, _, _, _, _, _>(left, right, lift_operator(&operator), data.into())
        {
            Ok(bdd) => return Bdd::Size32(bdd),
            Err(data) => data,
        };

    Bdd::Size64(
        apply_any::<Bdd64, _, _, _, _, _>(left, right, lift_operator(operator), data.into())
            .expect("TODO: 64 bit apply failed"),
    )
}

/// Like [`apply_any_default_state`], but specifically at most 32-bit wide BDDs.
///
/// The function automatically grows the BDD to 64 bits if the result does not fit.
pub(crate) fn apply_32_bit_input<
    TBdd1: AsBdd<Bdd32> + AsBdd<Bdd64>,
    TBdd2: AsBdd<Bdd32> + AsBdd<Bdd64>,
    TTriBoolOp: Fn(TriBool, TriBool) -> TriBool,
>(
    left: &TBdd1,
    right: &TBdd2,
    operator: TTriBoolOp,
) -> Bdd {
    let data = match apply_any_default_state::<Bdd32, _, _, _, TaskCache32<NodeId32>, _>(
        left,
        right,
        lift_operator(&operator),
    ) {
        Ok(bdd) => return Bdd::Size32(bdd),
        Err(data) => data,
    };

    Bdd::Size64(
        apply_any::<Bdd64, _, _, _, _, _>(left, right, lift_operator(operator), data.into())
            .expect("TODO: 64 bit apply failed"),
    )
}

/// Like [`apply_any_default_state`], but specifically for at most 64-bit wide BDDs.
pub(crate) fn apply_64_bit_input<
    TBdd1: AsBdd<Bdd64>,
    TBdd2: AsBdd<Bdd64>,
    TTriBoolOp: Fn(TriBool, TriBool) -> TriBool,
>(
    left: &TBdd1,
    right: &TBdd2,
    operator: TTriBoolOp,
) -> Bdd {
    Bdd::Size64(
        apply_any_default_state::<Bdd64, _, _, _, TaskCache64<NodeId64>, NodeTable64>(
            left,
            right,
            lift_operator(operator),
        )
        .expect("TODO: 64-bit apply failed"),
    )
}

/// Data used to store the state of the apply algorithm.
#[derive(Debug)]
pub(crate) struct ApplyState<TResultBdd: BddAny, TNodeId1, TNodeId2, TTaskCache, TNodeTable> {
    stack: Vec<(TNodeId1, TNodeId2, TResultBdd::VarId)>,
    results: Vec<TResultBdd::Id>,
    task_cache: TTaskCache,
    node_table: TNodeTable,
}

type NodeId<B> = <B as BddAny>::Id;

macro_rules! impl_apply_state_conversion {
    ($from_result:ident, $to_result:ident, $cache:ident, $from_table:ident, $to_table:ident) => {
        impl<TNodeId1: NodeIdAny, TNodeId2: NodeIdAny>
            From<
                ApplyState<
                    $from_result,
                    TNodeId1,
                    TNodeId2,
                    $cache<NodeId<$from_result>>,
                    $from_table,
                >,
            >
            for ApplyState<$to_result, TNodeId1, TNodeId2, $cache<NodeId<$to_result>>, $to_table>
        {
            fn from(
                data: ApplyState<
                    $from_result,
                    TNodeId1,
                    TNodeId2,
                    $cache<NodeId<$from_result>>,
                    $from_table,
                >,
            ) -> Self {
                Self {
                    stack: data
                        .stack
                        .into_iter()
                        .map(|(left, right, var)| (left, right, var.into()))
                        .collect(),
                    results: data.results.into_iter().map(|id| id.into()).collect(),
                    task_cache: data.task_cache.into(),
                    node_table: data.node_table.into(),
                }
            }
        }
    };
}

impl_apply_state_conversion!(Bdd16, Bdd32, TaskCache16, NodeTable16, NodeTable32);
impl_apply_state_conversion!(Bdd32, Bdd64, TaskCache16, NodeTable32, NodeTable64);
impl_apply_state_conversion!(Bdd32, Bdd64, TaskCache32, NodeTable32, NodeTable64);

/// Like [`apply_any`], but constructs the initial state necessary to start the computation.
pub(crate) fn apply_any_default_state<
    TResultBdd: BddAny,
    TBdd1: AsBdd<TResultBdd>,
    TBdd2: AsBdd<TResultBdd>,
    TBooleanOp: Fn(TBdd1::Id, TBdd2::Id) -> TResultBdd::Id,
    TTaskCache: TaskCacheAny<ResultId = TResultBdd::Id>,
    TNodeTable: NodeTableAny<Id = TResultBdd::Id, VarId = TResultBdd::VarId, Node = TResultBdd::Node>,
>(
    left: &TBdd1,
    right: &TBdd2,
    operator: TBooleanOp,
) -> Result<TResultBdd, ApplyState<TResultBdd, TBdd1::Id, TBdd2::Id, TTaskCache, TNodeTable>> {
    let stack = vec![(left.root(), right.root(), <TResultBdd::VarId>::undefined())];
    let results: Vec<TResultBdd::Id> = Vec::new();
    let task_cache = TTaskCache::default();
    let node_table = TNodeTable::default();

    let data = ApplyState {
        stack,
        results,
        task_cache,
        node_table,
    };

    apply_any(left, right, operator, data)
}

/// A generic universal function used for implementing logical operators. The function works
/// for any (reasonable) combination of BDD widths.
///
/// The `operator` function is the logical operator to be applied to the BDDs.
/// It is expected to be defined mainly for terminal [`NodeIdAny`] arguments. However,
/// since some logical operators can return the result even if only one of the arguments
/// is a terminal node, it has to work for non-terminal nodes as well. If the result is not
/// yet known, the function should return [`NodeIdAny::undefined`]. For example, the logical
/// operator implementing disjunction would be defined as:
/// ```text
/// or(NodeIdAny(1), NodeIdAny(_)) -> NodeIdAny(1)
/// or(NodeIdAny(_), NodeIdAny(1)) -> NodeIdAny(1)
/// or(NodeIdAny(0), NodeIdAny(0)) -> NodeIdAny(0)
/// or(NodeIdAny(_), NodeIdAny(_)) -> NodeIdAny::undefined(),
/// ```
/// The function returns [`Result::Ok`] with the computed BDD or [`Result::Err`]
/// with the state of the computation if the BDD could not fit into the target width.
fn apply_any<
    TResultBdd: BddAny,
    TBdd1: AsBdd<TResultBdd>,
    TBdd2: AsBdd<TResultBdd>,
    TBooleanOp: Fn(TBdd1::Id, TBdd2::Id) -> TResultBdd::Id,
    TTaskCache: TaskCacheAny<ResultId = TResultBdd::Id>,
    TNodeTable: NodeTableAny<Id = TResultBdd::Id, VarId = TResultBdd::VarId, Node = TResultBdd::Node>,
>(
    left: &TBdd1,
    right: &TBdd2,
    operator: TBooleanOp,
    state: ApplyState<TResultBdd, TBdd1::Id, TBdd2::Id, TTaskCache, TNodeTable>,
) -> Result<TResultBdd, ApplyState<TResultBdd, TBdd1::Id, TBdd2::Id, TTaskCache, TNodeTable>> {
    let ApplyState {
        mut stack,
        mut results,
        mut task_cache,
        mut node_table,
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
            // The task has not been expanded yet.

            let left_node = unsafe { left.get_node_unchecked(left_id) };
            let right_node = unsafe { right.get_node_unchecked(right_id) };

            let use_cache = left_node.has_many_parents() || right_node.has_many_parents();

            if use_cache {
                let result = task_cache.get((left_id, right_id));
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
            stack.push((left_high, right_high, <TResultBdd::VarId>::undefined()));
            stack.push((left_low, right_low, <TResultBdd::VarId>::undefined()));

            continue;
        }

        let high_result = results.pop().expect("high result present in result stack");
        let low_result = results.pop().expect("low result present in result stack");

        let node_id = match node_table.ensure_node(variable, low_result, high_result) {
            Ok(node_id) => node_id,
            Err(_) => {
                results.push(low_result);
                results.push(high_result);
                stack.push((left_id, right_id, variable));

                return Err(ApplyState {
                    stack,
                    results,
                    task_cache,
                    node_table,
                });
            }
        };

        if variable.use_cache() {
            task_cache.set((left_id, right_id), node_id);
        }
        results.push(node_id);
    }
    let root = results.pop().expect("root result present in result stack");
    debug_assert!(results.is_empty());
    Ok(unsafe { node_table.into_bdd(root) })
}

impl Bdd {
    /// Calculate a [`Bdd`] representing the boolean formula `self && other` (conjunction).
    pub fn and(&self, other: &Bdd) -> Bdd {
        self.apply(other, TriBool::and)
    }

    /// Calculate a [`Bdd`] representing the boolean formula `self || other` (disjunction).
    pub fn or(&self, other: &Bdd) -> Bdd {
        self.apply(other, TriBool::or)
    }

    /// Calculate a [`Bdd`] representing the boolean formula `self ^ other` (xor; non-equivalence).
    pub fn xor(&self, other: &Bdd) -> Bdd {
        self.apply(other, TriBool::xor)
    }

    /// Calculate a [`Bdd`] representing the boolean formula `self => other` (implication).
    pub fn implies(&self, other: &Bdd) -> Bdd {
        self.apply(other, TriBool::implies)
    }

    /// Calculate a [`Bdd`] representing the boolean formula `self <=> other` (equivalence).
    pub fn iff(&self, other: &Bdd) -> Bdd {
        self.apply(other, TriBool::iff)
    }

    pub(crate) fn apply<TTriBoolOp: Fn(TriBool, TriBool) -> TriBool>(
        &self,
        other: &Bdd,
        operator: TTriBoolOp,
    ) -> Bdd {
        match (self, other) {
            (Bdd::Size16(left), Bdd::Size16(right)) => apply_16_bit_input(left, right, operator),
            (Bdd::Size16(left), Bdd::Size32(right)) => apply_32_bit_input(left, right, operator),
            (Bdd::Size16(left), Bdd::Size64(right)) => apply_64_bit_input(left, right, operator),
            (Bdd::Size32(left), Bdd::Size16(right)) => apply_32_bit_input(left, right, operator),
            (Bdd::Size32(left), Bdd::Size32(right)) => apply_32_bit_input(left, right, operator),
            (Bdd::Size32(left), Bdd::Size64(right)) => apply_64_bit_input(left, right, operator),
            (Bdd::Size64(left), Bdd::Size16(right)) => apply_64_bit_input(left, right, operator),
            (Bdd::Size64(left), Bdd::Size32(right)) => apply_64_bit_input(left, right, operator),
            (Bdd::Size64(left), Bdd::Size64(right)) => apply_64_bit_input(left, right, operator),
        }
        .shrink()
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
/// Error returned when the BDD operation overflows the target width.
pub struct BddOverflowError {
    width: usize,
}

impl Display for BddOverflowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BDD operation overflowed the target width of {} bits",
            self.width
        )
    }
}

impl BddOverflowError {
    pub fn new<T: BddAny>() -> BddOverflowError {
        BddOverflowError {
            width: std::mem::size_of::<NodeId<T>>() * 8,
        }
    }
}

macro_rules! impl_bdd_operations {
    ($Bdd:ident, $Cache:ident, $Table:ident) => {
        impl $Bdd {
            /// Calculate a Bdd representing the boolean formula `self && other` (conjunction).
            pub fn and(&self, other: &$Bdd) -> Result<$Bdd, BddOverflowError> {
                Self::apply(self, other, boolean_operators::and)
            }

            /// Calculate a Bdd representing the boolean formula `self || other` (disjunction).
            pub fn or(&self, other: &$Bdd) -> Result<$Bdd, BddOverflowError> {
                Self::apply(self, other, boolean_operators::or)
            }

            /// Calculate a Bdd representing the boolean formula `self ^ other` (xor; non-equivalence).
            pub fn xor(&self, other: &$Bdd) -> Result<$Bdd, BddOverflowError> {
                Self::apply(self, other, boolean_operators::xor)
            }

            /// Calculate a Bdd representing the boolean formula `self => other` (implication).
            pub fn implies(&self, other: &$Bdd) -> Result<$Bdd, BddOverflowError> {
                Self::apply(self, other, boolean_operators::implies)
            }

            /// Calculate a Bdd representing the boolean formula `self <=> other` (equivalence).
            pub fn iff(&self, other: &$Bdd) -> Result<$Bdd, BddOverflowError> {
                Self::apply(self, other, boolean_operators::iff)
            }

            pub fn apply<TBooleanOp: Fn(NodeId<$Bdd>, NodeId<$Bdd>) -> NodeId<$Bdd>>(
                left: &$Bdd,
                right: &$Bdd,
                operator: TBooleanOp,
            ) -> Result<$Bdd, BddOverflowError> {
                apply_any_default_state::<$Bdd, _, _, _, $Cache<NodeId<$Bdd>>, $Table>(
                    left, right, operator,
                )
                .map_err(|_| BddOverflowError::new::<$Bdd>())
            }
        }
    };
}

impl_bdd_operations!(Bdd16, TaskCache16, NodeTable16);
impl_bdd_operations!(Bdd32, TaskCache32, NodeTable32);
impl_bdd_operations!(Bdd64, TaskCache64, NodeTable64);

#[cfg(test)]
mod tests {
    use crate::apply::BddOverflowError;
    use crate::bdd::{Bdd16, Bdd32, Bdd64, BddAny};
    use crate::variable_id::{VarIdPacked16, VarIdPacked32, VarIdPacked64};
    use crate::{bdd::Bdd, variable_id::VariableId};

    #[test]
    pub fn basic_apply_invariants() {
        // These are obviously not all invariants/equalities, but at least something to
        // check that we have the major corner cases covered.

        let a = Bdd::new_literal(VariableId::from(1u32), true);
        let b = Bdd::new_literal(VariableId::from(2u32), true);
        let a_n = Bdd::new_literal(VariableId::from(1u32), false);
        let b_n = Bdd::new_literal(VariableId::from(2u32), false);
        let tt = Bdd::new_true();
        let ff = Bdd::new_false();

        let res = &a.and(&a);

        assert!(Bdd::structural_eq(res, &a));
        assert!(Bdd::structural_eq(&a.and(&tt), &a));
        assert!(Bdd::structural_eq(&a.and(&ff), &ff));
        assert!(Bdd::structural_eq(&a.and(&b), &b.and(&a)));

        assert!(Bdd::structural_eq(&a.or(&a), &a));
        assert!(Bdd::structural_eq(&a.or(&tt), &tt));
        assert!(Bdd::structural_eq(&a.or(&ff), &a));
        assert!(Bdd::structural_eq(&a.or(&b), &b.or(&a)));

        assert!(Bdd::structural_eq(&a.implies(&a), &tt));
        assert!(Bdd::structural_eq(&a.implies(&tt), &tt));
        assert!(Bdd::structural_eq(&a.implies(&ff), &a_n));
        assert!(Bdd::structural_eq(&a.implies(&b), &a_n.or(&b)));

        assert!(Bdd::structural_eq(&a.xor(&a), &ff));
        assert!(Bdd::structural_eq(&a.xor(&tt), &a_n));
        assert!(Bdd::structural_eq(&a.xor(&ff), &a));
        assert!(Bdd::structural_eq(
            &a.xor(&b),
            &a.and(&b_n).or(&a_n.and(&b))
        ));

        assert!(Bdd::structural_eq(&a.iff(&a), &tt));
        assert!(Bdd::structural_eq(&a.iff(&tt), &a));
        assert!(Bdd::structural_eq(&a.iff(&ff), &a_n));
        assert!(Bdd::structural_eq(
            &a.iff(&b),
            &a.and(&b).or(&a_n.and(&b_n))
        ));
    }

    #[test]
    fn bdd_size_combinations_apply() {
        // Make BDDs of different sizes and combine them to ensure every size combination
        // is tested using at least basic operations.

        let data = vec![
            Bdd::new_literal(VariableId::new(1u32 << 8), true),
            Bdd::new_literal(VariableId::new(1u32 << 24), true),
            Bdd::new_literal(VariableId::new_long(1u64 << 48).unwrap(), true),
        ];

        for a in &data {
            for b in &data {
                let iff = a.xor(b);
                let other_iff = (a.and(&b.not())).or(&a.not().and(b));
                assert!(iff.structural_eq(&other_iff));
            }
        }
    }

    #[test]
    fn bdd_test_checked_logical_operators() {
        // This only covers very basic "positive" outcomes (i.e. no size overflow).
        // Making the BDDs overflow is much harder because we actually need the result to have
        // at least 2^16 (resp. 2^32) nodes.

        let bdd16 = Bdd16::new_literal(VarIdPacked16::new(1u16 << 8), true);
        let bdd32 = Bdd32::new_literal(VarIdPacked32::new(1u32 << 24), true);
        let bdd64 = Bdd64::new_literal(VarIdPacked64::new(1u64 << 48), true);

        assert!(bdd16.structural_eq(&bdd16.and(&bdd16).unwrap()));
        assert!(bdd32.structural_eq(&bdd32.and(&bdd32).unwrap()));
        assert!(bdd64.structural_eq(&bdd64.and(&bdd64).unwrap()));

        assert!(bdd16.structural_eq(&bdd16.or(&bdd16).unwrap()));
        assert!(bdd32.structural_eq(&bdd32.or(&bdd32).unwrap()));
        assert!(bdd64.structural_eq(&bdd64.or(&bdd64).unwrap()));

        assert!(bdd16.xor(&bdd16).unwrap().is_false());
        assert!(bdd32.xor(&bdd32).unwrap().is_false());
        assert!(bdd64.xor(&bdd64).unwrap().is_false());

        assert!(bdd16.implies(&bdd16).unwrap().is_true());
        assert!(bdd32.implies(&bdd32).unwrap().is_true());
        assert!(bdd64.implies(&bdd64).unwrap().is_true());

        assert!(bdd16.iff(&bdd16).unwrap().is_true());
        assert!(bdd32.iff(&bdd32).unwrap().is_true());
        assert!(bdd64.iff(&bdd64).unwrap().is_true());
    }

    #[test]
    fn bdd_size_overflow_test() {
        fn ripple_carry_adder(num_vars: u16) -> Result<Bdd16, BddOverflowError> {
            let mut result = Bdd16::new_false();
            for x in 0..(num_vars / 2) {
                let x1 = Bdd16::new_literal(VarIdPacked16::new(x), true);
                let x2 = Bdd16::new_literal(VarIdPacked16::new(x + num_vars / 2), true);
                result = result.or(&x1.and(&x2)?)?;
            }
            Ok(result)
        }

        let result = ripple_carry_adder(4).unwrap();
        assert_eq!(result.node_count(), 8);

        let result = ripple_carry_adder(8).unwrap();
        assert_eq!(result.node_count(), 32);

        let result = ripple_carry_adder(16).unwrap();
        assert_eq!(result.node_count(), 512);

        let result = ripple_carry_adder(24).unwrap();
        assert_eq!(result.node_count(), 8192);

        let err = ripple_carry_adder(32).unwrap_err();
        println!("{}", err);
        assert_eq!(err.width, 16);
    }
}
