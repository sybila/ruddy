//! Defines the representation of (standalone) binary decision diagrams. Includes: [`BddAny`]
//! and [`Bdd32`].

use crate::bdd_node::{BddNode16, BddNode32, BddNode64, BddNodeAny, BddNodeImpl};
use crate::conversion::{UncheckedFrom, UncheckedInto};
use crate::node_id::{AsNodeId, NodeId, NodeId16, NodeId64};
use crate::node_id::{NodeId32, NodeIdAny};
use crate::variable_id::{
    AsVarId, VarIdPacked16, VarIdPacked32, VarIdPacked64, VarIdPackedAny, VariableId,
};
use std::fmt::Debug;

/// A trait implemented by types that can serve as *standalone* BDDs.
///
/// Internally, a standalone BDD is a collection of [`BddNodeAny`] instances, such that one node is
/// designated as the *root* of the BDD. Aside from the root, a non-trivial BDD must also contain
/// a single [`BddNodeAny::zero`] node identified by [`NodeIdAny::zero`] and a single [`BddNodeAny::one`] node
/// identified by [`NodeIdAny::one`]. Finally, the graph induced by the [`BddNodeAny::low`] and
/// [`BddNodeAny::high`] edges must not contain any cycles (except for the self-loops on the
/// two terminal nodes).
///
/// Note that it is not required for the nodes to (1) be sorted in any specific way; (2) be
/// reachable from the root node; (3) adhere to any specific variable ordering or structural
/// properties other than those outlined above. In other words, an arbitrary instance of [`BddAny`]
/// is not required to be an OBDD (ordered BDD) or ROBDD (reduced and ordered BDD). However,
/// these properties are typically enforced in practice by the implementations of this trait.
///
/// Finally, note that standalone BDDs are typically assumed to be immutable. There are certain
/// situations where mutability makes sense (for example to allow chaining of multiple smaller
/// structural changes without unnecessary copying), but this is mostly an exception to the rule.
/// In particular, no method in the actual trait allows mutability.
pub trait BddAny: Debug + Clone {
    /// The type of node ID used by [`BddAny::Node`].
    type Id: NodeIdAny;
    /// The type of variable ID used by [`BddAny::Node`].
    type VarId: VarIdPackedAny;
    /// The type of node used in this [`BddAny`].
    type Node: BddNodeAny<Id = Self::Id, VarId = Self::VarId>;

    /// Create a new BDD representing the constant boolean function `true`.
    fn new_true() -> Self;
    /// Create a new BDD representing the constant boolean function `false`.
    fn new_false() -> Self;
    /// Create a new BDD representing the boolean function `var=value`.
    fn new_literal(var: Self::VarId, value: bool) -> Self;

    /// Returns `true` if the BDD represents the constant boolean function `true`.
    fn is_true(&self) -> bool;

    /// Returns `true` if the BDD represents the constant boolean function `false`.
    fn is_false(&self) -> bool;

    /// Create a new instance of [`BddAny`] using a raw list of [`BddNodeAny`] items and a single
    /// [`NodeIdAny`] root.
    ///
    /// ## Safety
    ///
    /// This function is unsafe because it can be used to create a BDD object that does not
    /// respect the prescribed invariants of [`BddAny`]. For example, it can be used to create
    /// a BDD that is not acyclic or a BDD that is not reduced.
    unsafe fn new_unchecked(root: Self::Id, nodes: Vec<Self::Node>) -> Self;

    /// ID of the BDD root node.
    fn root(&self) -> Self::Id;
    /// Get a (checked) reference to a node, or `None` if such node does not exist.
    fn get(&self, id: Self::Id) -> Option<&Self::Node>;

    /// An unchecked variant of [`BddAny::get`].
    ///
    /// # Safety
    ///
    /// Calling this method with an `id` that is not in the bdd is undefined behavior.
    unsafe fn get_node_unchecked(&self, id: Self::Id) -> &Self::Node;
}

/// A generic implementation of [`BddAny`] using [`BddNodeImpl`]. In addition to
/// the requirements of the [`BddAny`] trait, this struct also expects the BDD to
/// be ordered and reduced.
#[derive(Debug, Clone)]
pub struct BddImpl<TNodeId: NodeIdAny, TVarId: VarIdPackedAny> {
    root: TNodeId,
    nodes: Vec<BddNodeImpl<TNodeId, TVarId>>,
}

impl<TNodeId: NodeIdAny, TVarId: VarIdPackedAny> BddImpl<TNodeId, TVarId> {
    /// Returns the number of nodes in the BDD, including the terminal nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Calculate a BDD representing the boolean formula `!self` (negation).
    fn not(&self) -> Self {
        if self.is_true() {
            return Self::new_false();
        }
        if self.is_false() {
            return Self::new_true();
        }

        let mut nodes = self.nodes.clone();
        for node in nodes.iter_mut().skip(2) {
            node.low = node.low.flipped_if_terminal();
            node.high = node.high.flipped_if_terminal();
        }

        Self {
            root: self.root,
            nodes,
        }
    }

    /// Compares the two BDDs structurally, i.e. by comparing their roots and the
    /// underlying lists of nodes.
    ///
    /// Note that this does not guarantee that the two BDDs represent the same boolean function,
    /// unless their nodes are also ordered the same way (which they are, assuming the BDDs were
    /// created using the same `apply` algorithm).
    pub fn structural_eq(&self, other: &Self) -> bool {
        self.root == other.root && self.nodes == other.nodes
    }

    /// Convert the BDD into a raw list of nodes.
    pub fn into_nodes(self) -> Vec<BddNodeImpl<TNodeId, TVarId>> {
        self.nodes
    }
}

impl<TNodeId: NodeIdAny, TVarId: VarIdPackedAny> BddAny for BddImpl<TNodeId, TVarId> {
    type Id = TNodeId;
    type VarId = TVarId;
    type Node = BddNodeImpl<TNodeId, TVarId>;

    fn new_true() -> Self {
        Self {
            root: TNodeId::one(),
            nodes: vec![BddNodeImpl::zero(), BddNodeImpl::one()],
        }
    }

    fn new_false() -> Self {
        Self {
            root: TNodeId::zero(),
            nodes: vec![BddNodeImpl::zero()],
        }
    }

    fn new_literal(var: Self::VarId, value: bool) -> Self {
        let node = if value {
            Self::Node::new(var, TNodeId::zero(), TNodeId::one())
        } else {
            Self::Node::new(var, TNodeId::one(), TNodeId::zero())
        };
        Self {
            root: 2usize.unchecked_into(),
            nodes: vec![Self::Node::zero(), Self::Node::one(), node],
        }
    }

    fn is_true(&self) -> bool {
        debug_assert!(self.root.is_one() == (self.nodes.len() == 2));
        self.root.is_one()
    }

    fn is_false(&self) -> bool {
        debug_assert!(self.root.is_zero() == (self.nodes.len() == 1));
        self.root.is_zero()
    }

    unsafe fn new_unchecked(root: Self::Id, nodes: Vec<Self::Node>) -> Self {
        Self { root, nodes }
    }

    fn root(&self) -> Self::Id {
        self.root
    }

    fn get(&self, id: Self::Id) -> Option<&Self::Node> {
        self.nodes.get(id.as_usize())
    }

    unsafe fn get_node_unchecked(&self, id: Self::Id) -> &Self::Node {
        unsafe { self.nodes.get_unchecked(id.as_usize()) }
    }
}

/// An implementation of [`BddAny`] using [`BddNode16`]. In addition to the requirements of the
/// [`BddAny`] trait, this type also expects the BDD to be ordered and reduced.
pub type Bdd16 = BddImpl<NodeId16, VarIdPacked16>;

/// An implementation of [`BddAny`] using [`BddNode32`]. In addition to the requirements of the
/// [`BddAny`] trait, this type also expects the BDD to be ordered and reduced.
pub type Bdd32 = BddImpl<NodeId32, VarIdPacked32>;

/// An implementation of [`BddAny`] using [`BddNode64`]. In addition to the requirements of the
/// [`BddAny`] trait, this type also expects the BDD to be ordered and reduced.
pub type Bdd64 = BddImpl<NodeId64, VarIdPacked64>;

/// A trait indicating that the node and variable IDs of the BDD can be upcast
/// to the node and variable IDs of the BDD specified by the generic type.
pub trait AsBdd<TBdd: BddAny>: BddAny<Id: AsNodeId<TBdd::Id>, VarId: AsVarId<TBdd::VarId>> {}

impl AsBdd<Bdd16> for Bdd16 {}
impl AsBdd<Bdd32> for Bdd16 {}
impl AsBdd<Bdd64> for Bdd16 {}
impl AsBdd<Bdd32> for Bdd32 {}
impl AsBdd<Bdd64> for Bdd32 {}
impl AsBdd<Bdd64> for Bdd64 {}

macro_rules! impl_unchecked_from {
    ($Large:ident => $Small:ident) => {
        impl UncheckedFrom<$Large> for $Small {
            fn unchecked_from(large: $Large) -> Self {
                Self {
                    root: large.root.unchecked_into(),
                    nodes: large
                        .nodes
                        .into_iter()
                        .map(UncheckedInto::unchecked_into)
                        .collect(),
                }
            }
        }
    };
}

impl_unchecked_from!(Bdd32 => Bdd16);
impl_unchecked_from!(Bdd64 => Bdd16);
impl_unchecked_from!(Bdd64 => Bdd32);

/// A public facade of the existing [`BddAny`] types.
///
/// TODO: Write documentation for this type.
#[derive(Clone, Debug)]
pub enum Bdd {
    Size16(Bdd16),
    Size32(Bdd32),
    Size64(Bdd64),
}

impl Bdd {
    /// Create a new BDD representing the constant boolean function `true`.
    pub fn new_true() -> Self {
        Self::Size16(Bdd16::new_true())
    }

    /// Create a new BDD representing the constant boolean function `false`.
    pub fn new_false() -> Self {
        Self::Size16(Bdd16::new_false())
    }

    /// Create a new BDD representing the boolean function `var=value`.
    ///
    /// # Memory
    /// In order to optimize memory usage and performance, please use the
    /// smallest possible variable identifiers. This will allow the BDD to
    /// shrink to a smaller width if possible. See [`VariableId::MAX_16_BIT_ID`],
    /// [`VariableId::MAX_32_BIT_ID`] and [`VariableId::MAX_64_BIT_ID`] for the
    /// maximum values that can be used for each width.
    pub fn new_literal(var: VariableId, value: bool) -> Self {
        if var.fits_in_packed16() {
            Self::Size16(Bdd16::new_literal(var.unchecked_into(), value))
        } else if var.fits_in_packed32() {
            Self::Size32(Bdd32::new_literal(var.unchecked_into(), value))
        } else if var.fits_in_packed64() {
            Self::Size64(Bdd64::new_literal(var.unchecked_into(), value))
        } else {
            unreachable!("Maximum representable variable identifier exceeded.");
        }
    }

    /// Returns the number of nodes in the BDD, including the terminal nodes.
    pub fn node_count(&self) -> usize {
        match self {
            Bdd::Size16(bdd) => bdd.node_count(),
            Bdd::Size32(bdd) => bdd.node_count(),
            Bdd::Size64(bdd) => bdd.node_count(),
        }
    }

    /// Calculate a [`Bdd`] representing the boolean formula `!self` (negation).
    pub fn not(&self) -> Self {
        match self {
            Bdd::Size16(bdd) => Bdd::Size16(bdd.not()),
            Bdd::Size32(bdd) => Bdd::Size32(bdd.not()),
            Bdd::Size64(bdd) => Bdd::Size64(bdd.not()),
        }
    }

    /// Returns `true` if the BDD represents the constant boolean function `true`.
    pub fn is_true(&self) -> bool {
        match self {
            Bdd::Size16(bdd) => bdd.is_true(),
            Bdd::Size32(bdd) => bdd.is_true(),
            Bdd::Size64(bdd) => bdd.is_true(),
        }
    }

    /// Returns `true` if the BDD represents the constant boolean function `false`.
    pub fn is_false(&self) -> bool {
        match self {
            Bdd::Size16(bdd) => bdd.is_false(),
            Bdd::Size32(bdd) => bdd.is_false(),
            Bdd::Size64(bdd) => bdd.is_false(),
        }
    }

    /// Shrink the BDD to the smallest possible bit-width.
    ///
    /// - If the BDD has less than `2**16` nodes and all variables fit in 16 bits, it will be
    ///   shrunk to a 16-bit BDD.
    /// - If the BDD has less than `2**32` nodes and all variables fit in 32 bits, it will be
    ///   shrunk to a 32-bit BDD.
    /// - Otherwise, the BDD will remain the same bit-width.
    pub(crate) fn shrink(self) -> Self {
        match self {
            Bdd::Size64(bdd) if bdd.node_count() < 1 << 32 => {
                let (mut vars_fit_in_16, mut vars_fit_in_32) = (true, true);
                for node in bdd.nodes.iter() {
                    vars_fit_in_16 &= node.variable().fits_in_packed16();
                    vars_fit_in_32 &= node.variable().fits_in_packed32();

                    if !vars_fit_in_16 && !vars_fit_in_32 {
                        break;
                    }
                }

                if (bdd.node_count() < 1 << 16) && vars_fit_in_16 {
                    return Bdd::Size16(bdd.unchecked_into());
                }

                if vars_fit_in_32 {
                    return Bdd::Size32(bdd.unchecked_into());
                }
                Bdd::Size64(bdd)
            }
            Bdd::Size32(bdd)
                if (bdd.node_count() < 1 << 16)
                    && bdd
                        .nodes
                        .iter()
                        .all(|node| node.variable().fits_in_packed16()) =>
            {
                Bdd::Size16(bdd.unchecked_into())
            }
            _ => self,
        }
    }

    /// Compares the two BDDs structurally, i.e. by comparing their roots and the
    /// underlying lists of nodes.
    ///
    /// Note that this does not guarantee that the two BDDs represent the same boolean function,
    /// unless their nodes are also ordered the same way (which they are, assuming the BDDs were
    /// created using the same `apply` algorithm).
    pub fn structural_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Bdd::Size16(a), Bdd::Size16(b)) => a.structural_eq(b),
            (Bdd::Size32(a), Bdd::Size32(b)) => a.structural_eq(b),
            (Bdd::Size64(a), Bdd::Size64(b)) => a.structural_eq(b),
            _ => false,
        }
    }

    pub fn root(&self) -> NodeId {
        match self {
            Bdd::Size16(bdd) => bdd.root().unchecked_into(),
            Bdd::Size32(bdd) => bdd.root().unchecked_into(),
            Bdd::Size64(bdd) => bdd.root().unchecked_into(),
        }
    }

    pub fn get_variable(&self, node: NodeId) -> VariableId {
        let index: usize = node.unchecked_into();
        match self {
            Bdd::Size16(bdd) => bdd.nodes[index].variable.unpack().into(),
            Bdd::Size32(bdd) => bdd.nodes[index].variable.unpack().into(),
            Bdd::Size64(bdd) => VariableId::new_long(bdd.nodes[index].variable.unpack())
                .unwrap_or_else(|| {
                    unreachable!("Variable stored in BDD table does not fit into standard range.")
                }),
        }
    }

    pub fn get_links(&self, node: NodeId) -> (NodeId, NodeId) {
        // The unchecked casts are necessary to ensure we are not using any undefined values.
        let index: usize = node.unchecked_into();
        match self {
            Bdd::Size16(bdd) => {
                let BddNode16 { low, high, .. } = bdd.nodes[index];
                (low.unchecked_into(), high.unchecked_into())
            }
            Bdd::Size32(bdd) => {
                let BddNode32 { low, high, .. } = bdd.nodes[index];
                (low.unchecked_into(), high.unchecked_into())
            }
            Bdd::Size64(bdd) => {
                let BddNode64 { low, high, .. } = bdd.nodes[index];
                (low.unchecked_into(), high.unchecked_into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::bdd::{Bdd, Bdd16, Bdd32, Bdd64, BddAny};
    use crate::bdd_node::BddNodeAny;
    use crate::conversion::UncheckedInto;
    use crate::node_id::{NodeId, NodeId16, NodeId32, NodeId64, NodeIdAny};
    use crate::variable_id::{VarIdPacked16, VarIdPacked32, VarIdPacked64, VariableId};

    macro_rules! test_bdd_not_invariants {
        ($func:ident, $Bdd:ident, $VarId:ident) => {
            #[test]
            pub fn $func() {
                assert!($Bdd::new_true().not().structural_eq(&$Bdd::new_false()));
                assert!($Bdd::new_false().not().structural_eq(&$Bdd::new_true()));

                let v = $VarId::new(1);
                let bdd = $Bdd::new_literal(v, true);
                assert!(bdd.not().not().structural_eq(&bdd));
            }
        };
    }

    test_bdd_not_invariants!(bdd16_not_invariants, Bdd16, VarIdPacked16);
    test_bdd_not_invariants!(bdd32_not_invariants, Bdd32, VarIdPacked32);
    test_bdd_not_invariants!(bdd64_not_invariants, Bdd64, VarIdPacked64);

    macro_rules! test_bdd_invariants {
        ($func:ident, $Bdd:ident, $VarId:ident, $NodeId:ident) => {
            #[test]
            pub fn $func() {
                assert!($Bdd::new_true().root().is_one());
                assert!($Bdd::new_true().is_true());
                assert!($Bdd::new_false().root().is_zero());
                assert!($Bdd::new_false().is_false());

                let v = $VarId::new(1);
                let x = $Bdd::new_literal(v, true);
                assert!(!x.root().is_terminal());
                assert!(x.get($NodeId::new(3)).is_none());
                assert_eq!(v, x.get(x.root()).unwrap().variable());
                assert_eq!(x.node_count(), 3);
                unsafe {
                    assert_eq!(v, x.get_node_unchecked(x.root()).variable());
                    let x_p = $Bdd::new_unchecked(x.root(), x.clone().into_nodes());
                    assert!(x.structural_eq(&x_p));
                }
            }
        };
    }

    test_bdd_invariants!(bdd16_invariants, Bdd16, VarIdPacked16, NodeId16);
    test_bdd_invariants!(bdd32_invariants, Bdd32, VarIdPacked32, NodeId32);
    test_bdd_invariants!(bdd64_invariants, Bdd64, VarIdPacked64, NodeId64);

    #[test]
    fn bdd_expands_to_32_and_shrinks_to_16() {
        let n: u16 = 16;
        // Create a BDD with 2n-2 variables v_1, ..., v_{2n-2} for the function
        // f(v_1, ..., v_{2n-2}) = v_1 * v_2 + v_3 * v_4 + ... + v_{2n-3} * v_{2n-2}.
        // with the variable ordering v_1 < v_3 < ... < v_{2n-3} < v_2 < v_4 < ... < v_{2n-2}.
        // The BDD will have 2^n nodes, hence it should grow to a 32-bit BDD.
        let low_vars: Vec<_> = (1..n).map(VariableId::from).collect();
        let high_vars: Vec<_> = (n + 1..2 * n).map(VariableId::from).collect();

        let mut bdd = Bdd::new_false();
        let mut bdd32 = Bdd32::new_false();

        for i in 0..high_vars.len() {
            let prod =
                Bdd::new_literal(low_vars[i], true).and(&Bdd::new_literal(high_vars[i], true));

            let prod32 = Bdd32::new_literal(low_vars[i].unchecked_into(), true)
                .and(&Bdd32::new_literal(high_vars[i].unchecked_into(), true))
                .unwrap();

            bdd = bdd.or(&prod);
            bdd32 = bdd32.or(&prod32).unwrap();
        }

        assert_eq!(bdd.node_count(), 1 << n);
        assert_eq!(bdd32.node_count(), 1 << n);

        // Check that the BDD grew correctly.
        match &bdd {
            Bdd::Size32(bdd_inner) => {
                // both checks should not be necessary, but they are here as a sanity check
                assert!(bdd_inner.iff(&bdd32).unwrap().is_true());
                assert!(bdd_inner.structural_eq(&bdd32));
            }
            _ => panic!("expected 32-bit BDD"),
        }

        // Now, transform the BDD into the function
        // f(v_1, ..., v_{2n-2}) = v_1 * v_2 * v_3 * v_4 * ... * v_{2n-3} * v_{2n-2}.
        // The BDD will have 32 nodes and hence should shrink to a 16-bit BDD.

        let mut bdd16_ands = Bdd16::new_true();

        for i in 0..high_vars.len() {
            let prod =
                Bdd::new_literal(low_vars[i], true).and(&Bdd::new_literal(high_vars[i], true));

            let prod16 = Bdd16::new_literal(low_vars[i].unchecked_into(), true)
                .and(&Bdd16::new_literal(high_vars[i].unchecked_into(), true))
                .unwrap();

            bdd = bdd.and(&prod);
            bdd16_ands = bdd16_ands.and(&prod16).unwrap();
        }

        assert_eq!(bdd.node_count(), usize::from(2 * n));

        // Check that the BDD shrank correctly.
        match &bdd {
            Bdd::Size16(bdd_inner) => {
                assert!(bdd_inner.iff(&bdd16_ands).unwrap().is_true());
                assert!(bdd_inner.structural_eq(&bdd16_ands));
            }
            _ => panic!("expected 16-bit BDD"),
        }
    }

    /// Since creating a BDD with more than 2^32 nodes is impractical, we
    /// have to create a 64-bit BDD manually and then test if it shrinks correctly.
    ///
    /// See [`bdd_expands_to_32_and_shrinks_to_16`] for details on the BDD structure.
    #[test]
    fn bdd_64_shrink_to_32() {
        let n = 17;
        let low_vars: Vec<_> = (1..n).map(VarIdPacked64::new).collect();
        let high_vars: Vec<_> = (n + 1..2 * n).map(VarIdPacked64::new).collect();

        let mut bdd64 = Bdd64::new_false();
        let mut bdd32 = Bdd32::new_false();

        for i in 0..high_vars.len() {
            let prod = Bdd64::new_literal(low_vars[i], true)
                .and(&Bdd64::new_literal(high_vars[i], true))
                .unwrap();

            let prod32 = Bdd32::new_literal(low_vars[i].try_into().unwrap(), true)
                .and(&Bdd32::new_literal(high_vars[i].try_into().unwrap(), true))
                .unwrap();

            bdd64 = bdd64.or(&prod).unwrap();
            bdd32 = bdd32.or(&prod32).unwrap();
        }

        assert_eq!(bdd64.node_count(), 1 << n);

        let bdd = Bdd::Size64(bdd64).shrink();

        assert_eq!(bdd.node_count(), 1 << n);

        match bdd {
            Bdd::Size32(bdd_inner) => {
                assert!(bdd_inner.iff(&bdd32).unwrap().is_true());
                assert!(bdd_inner.structural_eq(&bdd32));
            }
            _ => panic!("expected 32-bit BDD"),
        }
    }

    /// Similar to [`bdd_64_shrink_to_32`], but shrinks to a 16-bit BDD.
    ///
    /// See [`bdd_expands_to_32_and_shrinks_to_16`] for details on the BDD structure.
    #[test]
    fn bdd_64_shrink_to_16() {
        let n = 4;
        let low_vars: Vec<_> = (1..n).map(VarIdPacked64::new).collect();
        let high_vars: Vec<_> = (n + 1..2 * n).map(VarIdPacked64::new).collect();

        let mut bdd16 = Bdd16::new_false();
        let mut bdd64 = Bdd64::new_false();

        for i in 0..high_vars.len() {
            let prod = Bdd64::new_literal(low_vars[i], true)
                .and(&Bdd64::new_literal(high_vars[i], true))
                .unwrap();

            let prod16 = Bdd16::new_literal(low_vars[i].try_into().unwrap(), true)
                .and(&Bdd16::new_literal(high_vars[i].try_into().unwrap(), true))
                .unwrap();

            bdd64 = bdd64.or(&prod).unwrap();
            bdd16 = bdd16.or(&prod16).unwrap();
        }

        assert_eq!(bdd64.node_count(), 1 << n);

        let bdd = Bdd::Size64(bdd64).shrink();

        assert_eq!(bdd.node_count(), 1 << n);

        match bdd {
            Bdd::Size16(bdd_inner) => {
                assert!(bdd_inner.iff(&bdd16).unwrap().is_true());
                assert!(bdd_inner.structural_eq(&bdd16));
            }
            _ => panic!("expected 16-bit BDD"),
        }
    }

    #[test]
    fn new_bdd_literal_16() {
        let var = VariableId::from(1u32);
        let bdd = Bdd::new_literal(var, true);
        let bdd16 = Bdd16::new_literal(var.unchecked_into(), true);

        match bdd {
            Bdd::Size16(bdd_inner) => {
                assert!(bdd_inner.iff(&bdd16).unwrap().is_true());
                assert!(bdd_inner.structural_eq(&bdd16));
            }
            _ => panic!("expected 16-bit BDD"),
        }
    }

    #[test]
    fn new_bdd_literal_32() {
        let max_id: u32 = VariableId::MAX_16_BIT_ID.try_into().unwrap();
        let var = VariableId::from(max_id + 1);
        let bdd = Bdd::new_literal(var, true);
        let bdd32 = Bdd32::new_literal(var.unchecked_into(), true);

        match bdd {
            Bdd::Size32(bdd_inner) => {
                assert!(bdd_inner.iff(&bdd32).unwrap().is_true());
                assert!(bdd_inner.structural_eq(&bdd32));
            }
            _ => panic!("expected 32-bit BDD"),
        }
    }

    #[test]
    fn new_bdd_literal_64() {
        let var = VariableId::new_long(VariableId::MAX_32_BIT_ID + 1).unwrap();
        let bdd = Bdd::new_literal(var, true);
        let bdd64 = Bdd64::new_literal(var.unchecked_into(), true);

        match bdd {
            Bdd::Size64(bdd_inner) => {
                assert!(bdd_inner.iff(&bdd64).unwrap().is_true());
                assert!(bdd_inner.structural_eq(&bdd64));
            }
            _ => panic!("expected 64-bit BDD"),
        }
    }

    #[test]
    fn new_bdd_literal_64_but_should_be_16() {
        let var = VariableId::from(1u32);
        let bdd = Bdd::new_literal(var, true);
        let bdd16 = Bdd16::new_literal(var.unchecked_into(), true);

        match bdd {
            Bdd::Size16(bdd_inner) => {
                assert!(bdd_inner.iff(&bdd16).unwrap().is_true());
                assert!(bdd_inner.structural_eq(&bdd16));
            }
            _ => panic!("expected 16-bit BDD"),
        }
    }

    #[test]
    fn new_bdd_literal_32_but_should_be_16() {
        let var = VariableId::from(1u32);
        let bdd = Bdd::new_literal(var, true);
        let bdd16 = Bdd16::new_literal(var.unchecked_into(), true);

        match bdd {
            Bdd::Size16(bdd_inner) => {
                assert!(bdd_inner.iff(&bdd16).unwrap().is_true());
                assert!(bdd_inner.structural_eq(&bdd16));
            }
            _ => panic!("expected 16-bit BDD"),
        }
    }

    #[test]
    fn new_bdd_literal_64_but_should_be_32() {
        let var = VariableId::new_long(VariableId::MAX_16_BIT_ID + 1).unwrap();
        let bdd = Bdd::new_literal(var, true);
        let bdd32 = Bdd32::new_literal(var.unchecked_into(), true);

        match bdd {
            Bdd::Size32(bdd_inner) => {
                assert!(bdd_inner.iff(&bdd32).unwrap().is_true());
                assert!(bdd_inner.structural_eq(&bdd32));
            }
            _ => panic!("expected 32-bit BDD"),
        }
    }

    #[test]
    fn bdd_node_count() {
        let bdd16 = Bdd::new_literal(VariableId::new(1u32 << 8), true);
        let bdd32 = Bdd::new_literal(VariableId::new(1u32 << 24), true);
        let bdd64 = Bdd::new_literal(VariableId::new_long(1u64 << 48).unwrap(), true);

        assert_eq!(bdd16.node_count(), 3);
        assert_eq!(bdd32.node_count(), 3);
        assert_eq!(bdd64.node_count(), 3);
    }

    #[test]
    fn bdd_simple_not() {
        let bdd16_1 = Bdd::new_literal(VariableId::new(1u32 << 8), true);
        let bdd32_1 = Bdd::new_literal(VariableId::new(1u32 << 24), true);
        let bdd64_1 = Bdd::new_literal(VariableId::new_long(1u64 << 48).unwrap(), true);

        let bdd16_0 = Bdd::new_literal(VariableId::new(1u32 << 8), false);
        let bdd32_0 = Bdd::new_literal(VariableId::new(1u32 << 24), false);
        let bdd64_0 = Bdd::new_literal(VariableId::new_long(1u64 << 48).unwrap(), false);

        assert!(bdd16_1.not().structural_eq(&bdd16_0));
        assert!(bdd32_1.not().structural_eq(&bdd32_0));
        assert!(bdd64_1.not().structural_eq(&bdd64_0));
    }

    #[test]
    fn bdd_structural_eq() {
        // Test that BDDs of different sizes are not equal even if they have "the same" nodes.
        let bdd16 = Bdd::Size16(Bdd16::new_literal(VarIdPacked16::new(1234), true));
        let bdd32 = Bdd::Size32(Bdd32::new_literal(VarIdPacked32::new(1234), true));
        assert!(!bdd16.structural_eq(&bdd32));
    }

    #[test]
    fn bdd_cannot_shrink() {
        let bdd64 = Bdd::new_literal(VariableId::new_long(1u64 << 48).unwrap(), true);
        assert!(bdd64.clone().shrink().structural_eq(&bdd64));
    }

    #[test]
    fn bdd_constants() {
        let bdd_true = Bdd::new_true();
        let bdd_false = Bdd::new_false();
        assert!(bdd_true.is_true() && !bdd_true.is_false());
        assert!(bdd_false.is_false() && !bdd_false.is_true());

        // There is no "normal" way to build 32-bit and 64-bit constant BDD,
        // but in theory they are valid BDDs, they are just not "reduced" properly.

        let true_32 = Bdd::Size32(Bdd32::new_true());
        let false_32 = Bdd::Size32(Bdd32::new_false());
        assert!(true_32.is_true() && !true_32.is_false());
        assert!(false_32.is_false() && !false_32.is_true());

        let true_64 = Bdd::Size64(Bdd64::new_true());
        let false_64 = Bdd::Size64(Bdd64::new_false());
        assert!(true_64.is_true() && !true_64.is_false());
        assert!(false_64.is_false() && !false_64.is_true());
    }

    #[test]
    fn bdd_getters() {
        let var16 = VariableId::new(1u32 << 8);
        let var32 = VariableId::new(1u32 << 24);
        let var64 = VariableId::new_long(1u64 << 48).unwrap();
        let bdd16 = Bdd::new_literal(var16, true);
        let bdd32 = Bdd::new_literal(var32, true);
        let bdd64 = Bdd::new_literal(var64, true);
        let zero = NodeId::zero();
        let one = NodeId::one();

        assert_eq!(bdd16.get_variable(bdd16.root()), var16);
        assert_eq!(bdd32.get_variable(bdd32.root()), var32);
        assert_eq!(bdd64.get_variable(bdd64.root()), var64);

        assert_eq!(bdd16.get_links(bdd16.root()), (zero, one));
        assert_eq!(bdd32.get_links(bdd32.root()), (zero, one));
        assert_eq!(bdd64.get_links(bdd64.root()), (zero, one));
    }
}
