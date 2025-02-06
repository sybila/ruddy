//! Defines the representation of (standalone) binary decision diagrams. Includes: [`BddAny`]
//! and [`Bdd32`].

use std::fmt::Debug;

use crate::bdd_node::{BddNode16, BddNode32, BddNode64, BddNodeAny};
use crate::node_id::{AsNodeId, NodeId16, NodeId64};
use crate::node_id::{NodeId32, NodeIdAny};
use crate::variable_id::{AsVarId, VarIdPacked16, VarIdPacked32, VarIdPacked64, VarIdPackedAny};

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

macro_rules! impl_bdd {
    ($name:ident, $NodeId:ident, $VarId:ident, $Node:ident) => {
        impl $name {
            /// Returns the number of nodes in the BDD, including the terminal nodes.
            pub fn len(&self) -> usize {
                self.nodes.len()
            }

            /// Returns `true` if the BDD has no nodes other than the terminal nodes and
            /// `false` otherwise.
            pub fn is_empty(&self) -> bool {
                self.len() <= 2
            }

            /// Convert the BDD into a raw list of nodes.
            pub fn into_nodes(self) -> Vec<$Node> {
                self.nodes
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
        }

        impl BddAny for $name {
            type Id = $NodeId;
            type VarId = $VarId;
            type Node = $Node;

            fn new_true() -> Self {
                Self {
                    root: $NodeId::one(),
                    nodes: vec![$Node::zero(), $Node::one()],
                }
            }

            fn new_false() -> Self {
                Self {
                    root: $NodeId::zero(),
                    nodes: vec![$Node::zero()],
                }
            }

            fn new_literal(var: Self::VarId, value: bool) -> Self {
                let root = $NodeId::new(2);
                let low = if value {
                    $NodeId::one()
                } else {
                    $NodeId::zero()
                };
                let high = if value {
                    $NodeId::zero()
                } else {
                    $NodeId::one()
                };
                let node = $Node::new(var, low, high);
                Self {
                    root,
                    nodes: vec![$Node::zero(), $Node::one(), node],
                }
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
    };
}

/// An implementation of [`BddAny`] using [`BddNode16`]. In addition to the requirements of the
/// [`BddAny`] trait, this struct also expects the BDD to be ordered and reduced.
#[derive(Clone, Debug)]
pub struct Bdd16 {
    root: NodeId16,
    nodes: Vec<BddNode16>,
}

impl_bdd!(Bdd16, NodeId16, VarIdPacked16, BddNode16);

/// An implementation of [`BddAny`] using [`BddNode32`]. In addition to the requirements of the
/// [`BddAny`] trait, this struct also expects the BDD to be ordered and reduced.
#[derive(Clone, Debug)]
pub struct Bdd32 {
    root: NodeId32,
    nodes: Vec<BddNode32>,
}

impl_bdd!(Bdd32, NodeId32, VarIdPacked32, BddNode32);

/// An implementation of [`BddAny`] using [`BddNode64`]. In addition to the requirements of the
/// [`BddAny`] trait, this struct also expects the BDD to be ordered and reduced.
#[derive(Clone, Debug)]
pub struct Bdd64 {
    root: NodeId64,
    nodes: Vec<BddNode64>,
}

impl_bdd!(Bdd64, NodeId64, VarIdPacked64, BddNode64);

/// A trait indicating that the node and variable IDs of the BDD can be upcast
/// to the node and variable IDs of the BDD specified by the generic type.
pub trait AsBdd<TBdd: BddAny>: BddAny<Id: AsNodeId<TBdd::Id>, VarId: AsVarId<TBdd::VarId>> {}

impl AsBdd<Bdd16> for Bdd16 {}
impl AsBdd<Bdd32> for Bdd16 {}
impl AsBdd<Bdd64> for Bdd16 {}
impl AsBdd<Bdd32> for Bdd32 {}
impl AsBdd<Bdd64> for Bdd32 {}
impl AsBdd<Bdd64> for Bdd64 {}

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
    /// TODO: just a 16-bit variable ID for now, will need to add a better
    /// public interface later
    pub fn new_literal(var: VarIdPacked16, value: bool) -> Self {
        Self::Size16(Bdd16::new_literal(var, value))
    }

    /// Shrink the BDD to a [`Bdd16`], if it has less than `2**8` nodes or a [`Bdd32`] if it has
    /// less than `2**16` nodes. This function is a no-op otherwise.
    pub(crate) fn shrink(self) -> Self {
        // TODO: correctly handle variables
        // Currently, it is possible that the variable IDs will be
        // out of bounds when shrinking into a smaller width.
        // Not sure how to handle this now.

        match self {
            Bdd::Size64(bdd) => {
                if bdd.len() < 1 << 8 {
                    Bdd::Size16(Bdd16 {
                        root: bdd
                            .root
                            .try_into()
                            .expect("root of 64-bit BDD with less than 2**8 nodes fits into 16 bits"),
                        nodes: bdd
                            .nodes
                            .into_iter()
                            .map(|node| {
                                node.try_into().expect(
                                    "id and variable id of node of 64-bit BDD with less than 2**8 nodes fits into 16 bits",
                                )
                            })
                            .collect(),
                    })
                } else if bdd.len() < 1 << 16 {
                    Bdd::Size32(Bdd32 {
                        root: bdd.root.try_into().expect(
                            "root of 64-bit BDD with less than 2**16 nodes fits into 32 bits",
                        ),
                        nodes: bdd
                            .nodes
                            .into_iter()
                            .map(|node| node.try_into().expect("id and variable id of node of 64-bit BDD with less than 2**16 nodes fits into 32 bits"))
                            .collect(),
                    })
                } else {
                    Bdd::Size64(bdd)
                }
            }
            Bdd::Size32(bdd) => {
                if bdd.len() < 1 << 8 {
                    Bdd::Size16(Bdd16 {
                        root: bdd.root.try_into().expect(
                            "root of 32-bit BDD with less than 2**8 nodes fits into 16 bits",
                        ),
                        nodes: bdd
                            .nodes
                            .into_iter()
                            .map(|node| node.try_into().expect("id and variable id of node of 32-bit BDD with less than 2**8 nodes fits into 16 bits"))
                            .collect(),
                    })
                } else {
                    Bdd::Size32(bdd)
                }
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
}

#[cfg(test)]
mod tests {
    use crate::bdd::{Bdd16, Bdd32, Bdd64, BddAny};
    use crate::bdd_node::BddNodeAny;
    use crate::node_id::{NodeId16, NodeId32, NodeId64, NodeIdAny};
    use crate::variable_id::{VarIdPacked16, VarIdPacked32, VarIdPacked64};

    macro_rules! test_bdd_invariants {
        ($func:ident, $Bdd:ident, $VarId:ident, $NodeId:ident) => {
            #[test]
            pub fn $func() {
                assert!($Bdd::new_true().root().is_one());
                assert!($Bdd::new_true().is_empty());
                assert!($Bdd::new_false().root().is_zero());
                assert!($Bdd::new_false().is_empty());

                let v = $VarId::new(1);
                let x = $Bdd::new_literal(v, true);
                assert!(!x.is_empty());
                assert!(!x.root().is_terminal());
                assert!(x.get($NodeId::new(3)).is_none());
                assert_eq!(v, x.get(x.root()).unwrap().variable());
                assert_eq!(x.len(), 3);
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
}
