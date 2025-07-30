//! Defines binary logic operators on [`Bdd`] objects using the `apply` algorithm.
//!

#![allow(clippy::type_complexity)]

use std::vec;

use crate::{
    bdd_node::BddNodeAny,
    boolean_operators::{self, BooleanOperator},
    node_id::{NodeId16, NodeId32, NodeId64, NodeIdAny},
    node_table::{NodeTable16, NodeTable32, NodeTable64, NodeTableAny},
    split::bdd::{AsBdd, Bdd, Bdd16, Bdd32, Bdd64, BddAny},
    task_cache::{TaskCache16, TaskCache32, TaskCache64, TaskCacheAny},
    variable_id::VarIdPackedAny,
};

use super::bdd::BddInner;

/// Like [`apply_any`], but specifically for 16-bit BDDs.
///
/// The function automatically grows the BDD to 32, or 64 bits if the result does not fit.
pub(crate) fn apply_16_bit_input<TBooleanOp: BooleanOperator>(
    left: &Bdd16,
    right: &Bdd16,
    operator: TBooleanOp,
) -> Bdd {
    let state: ApplyState16<NodeId16, NodeId16, TaskCache16> = default_state_16(left, right);

    let state: ApplyState32<NodeId16, NodeId16, TaskCache16<NodeId32>> =
        match apply_any(left, right, operator, state) {
            Ok(bdd) => return bdd.into(),
            Err(state) => state.into(),
        };

    let state: ApplyState64<NodeId16, NodeId16, TaskCache16<NodeId64>> =
        match apply_any(left, right, operator, state) {
            Ok(bdd) => return bdd.into(),
            Err(state) => state.into(),
        };

    apply_any(left, right, operator, state)
        .expect("64 bit operation failed")
        .into()
}

/// Like [`apply_any`], but specifically for at most 32-bit wide BDDs.
///
/// The function automatically grows the BDD to 64 bits if the result does not fit.
pub(crate) fn apply_32_bit_input<
    TBdd1: AsBdd<Bdd32> + AsBdd<Bdd64>,
    TBdd2: AsBdd<Bdd32> + AsBdd<Bdd64>,
    TBooleanOp: BooleanOperator,
>(
    left: &TBdd1,
    right: &TBdd2,
    operator: TBooleanOp,
) -> Bdd {
    let state: ApplyState32<NodeId<TBdd1>, NodeId<TBdd2>, TaskCache32> =
        default_state_32(left, right);

    let state: ApplyState64<NodeId<TBdd1>, NodeId<TBdd2>, TaskCache32<NodeId64>> =
        match apply_any(left, right, operator, state) {
            Ok(bdd) => return bdd.into(),
            Err(state) => state.into(),
        };

    apply_any(left, right, operator, state)
        .expect("64 bit operation failed")
        .into()
}

/// Like [`apply_any`], but specifically for at most 64-bit wide BDDs.
pub(crate) fn apply_64_bit_input<
    TBdd1: AsBdd<Bdd64>,
    TBdd2: AsBdd<Bdd64>,
    TBooleanOp: BooleanOperator,
>(
    left: &TBdd1,
    right: &TBdd2,
    operator: TBooleanOp,
) -> Bdd {
    let state: ApplyState64<NodeId<TBdd1>, NodeId<TBdd2>, TaskCache64> =
        default_state_64(left, right);
    apply_any(left, right, operator, state)
        .expect("64-bit operation failed")
        .into()
}

/// Data used to store the state of the `apply` algorithm.
#[derive(Debug)]
pub(crate) struct ApplyState<TResultBdd: BddAny, TNodeId1, TNodeId2, TTaskCache, TNodeTable> {
    stack: Vec<(TNodeId1, TNodeId2, TResultBdd::VarId)>,
    results: Vec<TResultBdd::Id>,
    task_cache: TTaskCache,
    node_table: TNodeTable,
}

type NodeId<B> = <B as BddAny>::Id;

macro_rules! impl_apply_state_variant {
    ($variant_name:ident, $constructor:ident, $out_bdd:ident, $task_cache:ident, $node_table:ident) => {
        type $variant_name<TId1, TId2, TCache = $task_cache> =
            ApplyState<$out_bdd, TId1, TId2, TCache, $node_table>;

        fn $constructor<TBdd1: AsBdd<$out_bdd>, TBdd2: AsBdd<$out_bdd>>(
            left: &TBdd1,
            right: &TBdd2,
        ) -> $variant_name<TBdd1::Id, TBdd2::Id> {
            let undefined_var = <$out_bdd as BddAny>::VarId::undefined();
            ApplyState {
                stack: vec![(left.root(), right.root(), undefined_var)],
                results: Vec::new(),
                task_cache: $task_cache::default(),
                node_table: $node_table::default(),
            }
        }
    };
}

impl_apply_state_variant!(
    ApplyState16,
    default_state_16,
    Bdd16,
    TaskCache16,
    NodeTable16
);

impl_apply_state_variant!(
    ApplyState32,
    default_state_32,
    Bdd32,
    TaskCache32,
    NodeTable32
);

impl_apply_state_variant!(
    ApplyState64,
    default_state_64,
    Bdd64,
    TaskCache64,
    NodeTable64
);

macro_rules! impl_apply_state_conversion {
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
                    task_cache: state.task_cache.into(),
                    node_table: state.node_table.into(),
                }
            }
        }
    };
}

impl_apply_state_conversion!(ApplyState16, ApplyState32);
impl_apply_state_conversion!(ApplyState32, ApplyState64);

/// A generic universal function used for implementing logical operators. The function works
/// for any (reasonable) combination of BDD widths.
///
/// The function returns [`Ok`] with the computed BDD or [`Err`]
/// with the state of the computation if the BDD could not fit into the target width.
fn apply_any<
    TResultBdd: BddAny,
    TBdd1: AsBdd<TResultBdd>,
    TBdd2: AsBdd<TResultBdd>,
    TBooleanOp: BooleanOperator,
    TTaskCache: TaskCacheAny<ResultId = TResultBdd::Id>,
    TNodeTable: NodeTableAny<Id = TResultBdd::Id, VarId = TResultBdd::VarId, Node = TResultBdd::Node>,
>(
    left: &TBdd1,
    right: &TBdd2,
    operator: TBooleanOp,
    state: ApplyState<TResultBdd, TBdd1::Id, TBdd2::Id, TTaskCache, TNodeTable>,
) -> Result<TResultBdd, ApplyState<TResultBdd, TBdd1::Id, TBdd2::Id, TTaskCache, TNodeTable>> {
    let operator = operator.for_split::<TBdd1::Id, TBdd2::Id, TResultBdd::Id>();

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
    /// Calculates a `Bdd` representing the boolean formula `self && other` (conjunction).
    pub fn and(&self, other: &Bdd) -> Bdd {
        self.apply(other, boolean_operators::And)
    }

    /// Calculates a `Bdd` representing the boolean formula `self || other` (disjunction).
    pub fn or(&self, other: &Bdd) -> Bdd {
        self.apply(other, boolean_operators::Or)
    }

    /// Calculates a `Bdd` representing the boolean formula `self ^ other` (xor; non-equivalence).
    pub fn xor(&self, other: &Bdd) -> Bdd {
        self.apply(other, boolean_operators::Xor)
    }

    /// Calculates a `Bdd` representing the boolean formula `self => other` (implication).
    pub fn implies(&self, other: &Bdd) -> Bdd {
        self.apply(other, boolean_operators::Implies)
    }

    /// Calculates a `Bdd` representing the boolean formula `self <=> other` (equivalence).
    pub fn iff(&self, other: &Bdd) -> Bdd {
        self.apply(other, boolean_operators::Iff)
    }

    /// Calculates a `Bdd` representing the boolean formula `self 'operator' other`.
    pub fn apply<TBooleanOp: BooleanOperator>(&self, other: &Bdd, operator: TBooleanOp) -> Bdd {
        match (&self.0, &other.0) {
            (BddInner::Size16(left), BddInner::Size16(right)) => {
                apply_16_bit_input(left, right, operator)
            }
            (BddInner::Size16(left), BddInner::Size32(right)) => {
                apply_32_bit_input(left, right, operator)
            }
            (BddInner::Size16(left), BddInner::Size64(right)) => {
                apply_64_bit_input(left, right, operator)
            }
            (BddInner::Size32(left), BddInner::Size16(right)) => {
                apply_32_bit_input(left, right, operator)
            }
            (BddInner::Size32(left), BddInner::Size32(right)) => {
                apply_32_bit_input(left, right, operator)
            }
            (BddInner::Size32(left), BddInner::Size64(right)) => {
                apply_64_bit_input(left, right, operator)
            }
            (BddInner::Size64(left), BddInner::Size16(right)) => {
                apply_64_bit_input(left, right, operator)
            }
            (BddInner::Size64(left), BddInner::Size32(right)) => {
                apply_64_bit_input(left, right, operator)
            }
            (BddInner::Size64(left), BddInner::Size64(right)) => {
                apply_64_bit_input(left, right, operator)
            }
        }
        .shrink()
    }
}

#[cfg(test)]
pub mod tests {
    use std::fmt::Display;

    use crate::boolean_operators::{self, BooleanOperator};
    use crate::split::bdd::{Bdd16, Bdd32, Bdd64, BddAny};
    use crate::variable_id::{VarIdPacked16, VarIdPacked32, VarIdPacked64};
    use crate::{split::bdd::Bdd, variable_id::VariableId};

    use super::{apply_any, default_state_16, default_state_32, default_state_64};

    type NodeId<B> = <B as BddAny>::Id;

    #[derive(PartialEq, Eq, Clone, Debug)]
    /// Error returned when the BDD operation overflows the target width.
    pub(crate) struct BddOverflowError {
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
                width: size_of::<NodeId<T>>() * 8,
            }
        }
    }

    // We use a macro here because it is hard to give a concrete `Cache` type here
    // (as that would require specifying the `THashSize` generic parameter).
    // That is doable by, for example, giving a `Width` associated type to `NodeIdAny`
    // but that seems like overkill for now.
    macro_rules! impl_bdd_operations {
        ($Bdd:ident, $Cache:ident, $Table:ident, $state_constructor:ident) => {
            impl $Bdd {
                /// Calculate a Bdd representing the boolean formula `self && other` (conjunction).
                pub fn and(&self, other: &$Bdd) -> Result<$Bdd, BddOverflowError> {
                    Self::apply(self, other, boolean_operators::And)
                }

                /// Calculate a Bdd representing the boolean formula `self || other` (disjunction).
                pub fn or(&self, other: &$Bdd) -> Result<$Bdd, BddOverflowError> {
                    Self::apply(self, other, boolean_operators::Or)
                }

                /// Calculate a Bdd representing the boolean formula `self ^ other` (xor; non-equivalence).
                pub fn xor(&self, other: &$Bdd) -> Result<$Bdd, BddOverflowError> {
                    Self::apply(self, other, boolean_operators::Xor)
                }

                /// Calculate a Bdd representing the boolean formula `self => other` (implication).
                pub fn implies(&self, other: &$Bdd) -> Result<$Bdd, BddOverflowError> {
                    Self::apply(self, other, boolean_operators::Implies)
                }

                /// Calculate a Bdd representing the boolean formula `self <=> other` (equivalence).
                pub fn iff(&self, other: &$Bdd) -> Result<$Bdd, BddOverflowError> {
                    Self::apply(self, other, boolean_operators::Iff)
                }

                pub fn apply<TBooleanOp: BooleanOperator>(
                    left: &$Bdd,
                    right: &$Bdd,
                    operator: TBooleanOp,
                ) -> Result<$Bdd, BddOverflowError> {
                    let state = $state_constructor(left, right);
                    apply_any(left, right, operator, state)
                        .map_err(|_| BddOverflowError::new::<$Bdd>())
                }
            }
        };
    }

    impl_bdd_operations!(Bdd16, TaskCache16, NodeTable16, default_state_16);
    impl_bdd_operations!(Bdd32, TaskCache32, NodeTable32, default_state_32);
    impl_bdd_operations!(Bdd64, TaskCache64, NodeTable64, default_state_64);

    #[test]
    pub fn basic_apply_invariants() {
        // These are not all invariants/equalities, but at least something to
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
                let other_iff = a.and(&b.not()).or(&a.not().and(b));
                assert!(iff.structural_eq(&other_iff));
            }
        }
    }

    #[test]
    fn bdd_test_checked_logical_operators() {
        // This only covers very basic "positive" outcomes (i.e., no size overflow).
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

    pub fn ripple_carry_adder(num_vars: u16) -> Result<Bdd16, BddOverflowError> {
        let mut result = Bdd16::new_false();
        for x in 0..(num_vars / 2) {
            let x1 = Bdd16::new_literal(VarIdPacked16::new(x), true);
            let x2 = Bdd16::new_literal(VarIdPacked16::new(x + num_vars / 2), true);
            result = result.or(&x1.and(&x2)?)?;
        }
        Ok(result)
    }

    #[test]
    fn bdd_size_overflow_test() {
        let result = ripple_carry_adder(4).unwrap();
        assert_eq!(result.node_count(), 8);

        let result = ripple_carry_adder(8).unwrap();
        assert_eq!(result.node_count(), 32);

        let result = ripple_carry_adder(16).unwrap();
        assert_eq!(result.node_count(), 512);

        let result = ripple_carry_adder(24).unwrap();
        assert_eq!(result.node_count(), 8192);

        let err = ripple_carry_adder(32).unwrap_err();
        println!("{err}");
        assert_eq!(err.width, 16);
    }
}
