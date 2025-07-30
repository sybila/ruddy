use crate::bdd_node::{BddNode16, BddNode32, BddNode64, BddNodeAny, BddNodeImpl};
use crate::conversion::{UncheckedFrom, UncheckedInto};
use crate::node_id::{AsNodeId, NodeId, NodeId16, NodeId64};
use crate::node_id::{NodeId32, NodeIdAny};
use crate::variable_id::{
    variables_between, AsVarId, VarIdPacked16, VarIdPacked32, VarIdPacked64, VarIdPackedAny,
    VariableId,
};
use std::fmt::Debug;
use std::io::{self, Write};

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
pub(crate) trait BddAny: Debug + Clone {
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
    #[allow(dead_code)]
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
pub(crate) struct BddImpl<TNodeId: NodeIdAny, TVarId: VarIdPackedAny> {
    pub(crate) root: TNodeId,
    pub(crate) nodes: Vec<BddNodeImpl<TNodeId, TVarId>>,
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

    /// Get the largest [`VariableId`] in the BDD, assuming it does not represent
    /// a constant function.
    pub(crate) fn get_largest_variable(&self) -> VariableId {
        self.nodes
            .iter()
            .map(|node| node.variable())
            .reduce(TVarId::max_defined)
            .expect("BDD is not constant")
            .unchecked_into()
    }

    /// Approximately counts the number of satisfying paths in the BDD.
    fn count_satisfying_paths(&self) -> f64 {
        if self.is_false() {
            return 0.0;
        }

        // Use a negative value to indicate that the count is not yet computed.
        let mut counts = vec![-1.0f64; self.node_count()];
        counts[0] = 0.0;
        counts[1] = 1.0;

        let root = self.root;

        let mut stack = vec![root];

        while let Some(id) = stack.pop() {
            if counts[id.as_usize()] >= 0.0 {
                continue;
            }

            let node = unsafe { self.get_node_unchecked(id) };
            let low = node.low;
            let high = node.high;
            let low_count = counts[low.as_usize()];
            let high_count = counts[high.as_usize()];
            let low_is_done = low_count >= 0.0;
            let high_is_done = high_count >= 0.0;

            if low_is_done && high_is_done {
                counts[id.as_usize()] = low_count + high_count;
                continue;
            }

            stack.push(id);

            if !low_is_done {
                stack.push(low);
            }

            if !high_is_done {
                stack.push(high);
            }
        }
        let result: f64 = counts[root.as_usize()];
        debug_assert!(result >= 0.0);
        debug_assert!(!result.is_nan());
        result
    }

    /// Approximately counts the number of satisfying valuations in the BDD. If
    /// `largest_variable` is [`Option::Some`], then it is assumed to be the largest
    /// variable. Otherwise, the largest variable in the BDD is used.
    ///
    /// Assumes that the given variable is greater than or equal to any
    /// variable in the BDD. Otherwise, the function may give unexpected results
    /// in release mode or panic in debug mode.
    fn count_satisfying_valuations(&self, largest_variable: Option<VariableId>) -> f64 {
        if self.is_false() {
            return 0.0;
        }

        if self.is_true() {
            if let Some(largest_variable) = largest_variable {
                let exponent = (Into::<u64>::into(largest_variable) + 1)
                    .try_into()
                    .unwrap_or(f64::MAX_EXP);
                return 2.0f64.powi(exponent);
            }
            return 1.0f64;
        }

        let largest_variable = largest_variable.unwrap_or_else(|| self.get_largest_variable());

        // Use a negative value to indicate that the count is not yet computed.
        let mut counts = vec![-1.0f64; self.node_count()];
        counts[0] = 0.0;
        counts[1] = 1.0;

        let root = self.root;

        let mut stack = vec![root];

        while let Some(id) = stack.pop() {
            if counts[id.as_usize()] >= 0.0 {
                continue;
            }

            let node = unsafe { self.get_node_unchecked(id) };
            let low = node.low();
            let high = node.high();
            let low_count = counts[low.as_usize()];
            let high_count = counts[high.as_usize()];
            let low_is_done = low_count >= 0.0;
            let high_is_done = high_count >= 0.0;

            let low_node = unsafe { self.get_node_unchecked(low) };
            let high_node = unsafe { self.get_node_unchecked(high) };

            let variable = node.variable();
            let low_variable = low_node.variable();
            let high_variable = high_node.variable();

            if low_is_done && high_is_done {
                let skipped = variables_between(low_variable, variable, largest_variable)
                    .try_into()
                    .unwrap_or(f64::MAX_EXP);
                let low_count = low_count * 2.0f64.powi(skipped);

                let skipped = variables_between(high_variable, variable, largest_variable)
                    .try_into()
                    .unwrap_or(f64::MAX_EXP);
                let high_count = high_count * 2.0f64.powi(skipped);

                counts[id.as_usize()] = low_count + high_count;

                continue;
            }

            stack.push(id);

            if !low_is_done {
                stack.push(low);
            }

            if !high_is_done {
                stack.push(high);
            }
        }

        debug_assert!(counts[root.as_usize()] >= 0.0);

        let root_variable = unsafe { self.get_node_unchecked(root) }.variable();
        let result = counts[root.as_usize()]
            * 2.0f64.powi(
                root_variable
                    .unpack_u64()
                    .try_into()
                    .unwrap_or(f64::MAX_EXP),
            );

        if result.is_nan() {
            f64::INFINITY
        } else {
            result
        }
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
pub(crate) type Bdd16 = BddImpl<NodeId16, VarIdPacked16>;

/// An implementation of [`BddAny`] using [`BddNode32`]. In addition to the requirements of the
/// [`BddAny`] trait, this type also expects the BDD to be ordered and reduced.
pub(crate) type Bdd32 = BddImpl<NodeId32, VarIdPacked32>;

/// An implementation of [`BddAny`] using [`BddNode64`]. In addition to the requirements of the
/// [`BddAny`] trait, this type also expects the BDD to be ordered and reduced.
pub(crate) type Bdd64 = BddImpl<NodeId64, VarIdPacked64>;

/// A trait indicating that the node and variable IDs of the BDD can be upcast
/// to the node and variable IDs of the BDD specified by the generic type.
pub(crate) trait AsBdd<TBdd: BddAny>:
    BddAny<Id: AsNodeId<TBdd::Id>, VarId: AsVarId<TBdd::VarId>>
{
}

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

impl<TNodeId: NodeIdAny, TVarId: VarIdPackedAny> BddImpl<TNodeId, TVarId> {
    /// Write this BDD as a DOT graph to the given `output` stream.
    fn write_as_dot(&self, output: &mut dyn Write) -> io::Result<()> {
        writeln!(output, "digraph BDD {{")?;
        writeln!(
            output,
            "  __ruddy_root [label=\"\", style=invis, height=0, width=0];"
        )?;

        writeln!(output, "  __ruddy_root -> {};", self.root)?;
        writeln!(output)?;
        writeln!(output, "  edge [dir=none];")?;
        writeln!(output)?;

        writeln!(
            output,
            "  0 [label=\"0\", shape=box, width=0.3, height=0.3];"
        )?;
        writeln!(
            output,
            "  1 [label=\"1\", shape=box, width=0.3, height=0.3];"
        )?;

        for (id, node) in self.nodes.iter().enumerate().skip(2) {
            let low = node.low();
            let high = node.high();
            writeln!(
                output,
                "  {} [label=\"{}\", shape=box, width=0.3, height=0.3];",
                id,
                node.variable()
            )?;
            writeln!(output, "  {id} -> {low} [style=dashed];")?;
            writeln!(output, "  {id} -> {high};")?;
        }

        writeln!(output, "}}")?;

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) enum BddInner {
    Size16(Bdd16),
    Size32(Bdd32),
    Size64(Bdd64),
}

/// A type representing a binary decision diagram, that is split, i.e., owns all
/// of its nodes in a vector and is immutable.
///
/// Consequently, each operation produces a new BDD, which is independent of its
/// operands.
///
/// This allows the BDD to be easily copied and even shared between threads. Moreover,
/// there is no need to perform explicit garbage collection --- once the `Bdd` is dropped,
/// so are all of its nodes.
///
/// However, being self-contained, if multiple BDDs share the same subgraph,
/// each needs to store a copy of it. Furthermore, supporting data structures like the
/// unique node table and computed cache are rebuilt for each operation, not shared or reused.
#[derive(Clone, Debug)]
pub struct Bdd(pub(crate) BddInner);

impl From<Bdd16> for Bdd {
    fn from(bdd: Bdd16) -> Self {
        Self(BddInner::Size16(bdd))
    }
}

impl From<Bdd32> for Bdd {
    fn from(bdd: Bdd32) -> Self {
        Self(BddInner::Size32(bdd))
    }
}

impl From<Bdd64> for Bdd {
    fn from(bdd: Bdd64) -> Self {
        Self(BddInner::Size64(bdd))
    }
}

impl Bdd {
    /// Creates a new BDD representing the constant boolean function `true`.
    pub fn new_true() -> Self {
        Bdd16::new_true().into()
    }

    /// Creates a new BDD representing the constant boolean function `false`.
    pub fn new_false() -> Self {
        Bdd16::new_false().into()
    }

    /// Creates a new BDD representing the boolean function `var=value`.
    pub fn new_literal(var: VariableId, value: bool) -> Self {
        if var.fits_in_packed16() {
            Bdd16::new_literal(var.unchecked_into(), value).into()
        } else if var.fits_in_packed32() {
            Bdd32::new_literal(var.unchecked_into(), value).into()
        } else if var.fits_in_packed64() {
            Bdd64::new_literal(var.unchecked_into(), value).into()
        } else {
            unreachable!("Maximum representable variable identifier exceeded.");
        }
    }

    /// Returns the number of nodes in the `Bdd`, including the terminal nodes.
    pub fn node_count(&self) -> usize {
        match &self.0 {
            BddInner::Size16(bdd) => bdd.node_count(),
            BddInner::Size32(bdd) => bdd.node_count(),
            BddInner::Size64(bdd) => bdd.node_count(),
        }
    }

    /// Calculate a `Bdd` representing the boolean formula `!self` (negation).
    pub fn not(&self) -> Self {
        match &self.0 {
            BddInner::Size16(bdd) => bdd.not().into(),
            BddInner::Size32(bdd) => bdd.not().into(),
            BddInner::Size64(bdd) => bdd.not().into(),
        }
    }

    /// Returns `true` if the BDD represents the constant boolean function `true`.
    pub fn is_true(&self) -> bool {
        match &self.0 {
            BddInner::Size16(bdd) => bdd.is_true(),
            BddInner::Size32(bdd) => bdd.is_true(),
            BddInner::Size64(bdd) => bdd.is_true(),
        }
    }

    /// Returns `true` if the BDD represents the constant boolean function `false`.
    pub fn is_false(&self) -> bool {
        match &self.0 {
            BddInner::Size16(bdd) => bdd.is_false(),
            BddInner::Size32(bdd) => bdd.is_false(),
            BddInner::Size64(bdd) => bdd.is_false(),
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
        match self.0 {
            BddInner::Size64(bdd) if bdd.node_count() < 1 << 32 => {
                let (mut vars_fit_in_16, mut vars_fit_in_32) = (true, true);
                for node in bdd.nodes.iter() {
                    vars_fit_in_16 &= node.variable().fits_in_packed16();
                    vars_fit_in_32 &= node.variable().fits_in_packed32();

                    if !vars_fit_in_16 && !vars_fit_in_32 {
                        break;
                    }
                }

                if (bdd.node_count() < 1 << 16) && vars_fit_in_16 {
                    let bdd16: Bdd16 = bdd.unchecked_into();
                    return bdd16.into();
                }

                if vars_fit_in_32 {
                    let bdd32: Bdd32 = bdd.unchecked_into();
                    return bdd32.into();
                }
                bdd.into()
            }
            BddInner::Size32(bdd)
                if (bdd.node_count() < 1 << 16)
                    && bdd
                        .nodes
                        .iter()
                        .all(|node| node.variable().fits_in_packed16()) =>
            {
                let bdd16: Bdd16 = bdd.unchecked_into();
                bdd16.into()
            }
            _ => self,
        }
    }

    /// Compares the two `Bdd`s structurally, i.e. by comparing their roots and the
    /// underlying lists of nodes.
    ///
    /// Note that this does not guarantee that the two BDDs represent the same boolean function,
    /// unless their nodes are also ordered the same way (which they are, assuming the BDDs were
    /// created using the same `apply` algorithm).
    pub fn structural_eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (BddInner::Size16(a), BddInner::Size16(b)) => a.structural_eq(b),
            (BddInner::Size32(a), BddInner::Size32(b)) => a.structural_eq(b),
            (BddInner::Size64(a), BddInner::Size64(b)) => a.structural_eq(b),
            _ => false,
        }
    }

    /// Returns the identifier of the root node of the `Bdd`.
    pub fn root(&self) -> NodeId {
        match &self.0 {
            BddInner::Size16(bdd) => bdd.root().unchecked_into(),
            BddInner::Size32(bdd) => bdd.root().unchecked_into(),
            BddInner::Size64(bdd) => bdd.root().unchecked_into(),
        }
    }

    /// Returns the variable identifier of the given `node`.
    pub fn get_variable(&self, node: NodeId) -> VariableId {
        let index: usize = node.unchecked_into();
        match &self.0 {
            BddInner::Size16(bdd) => bdd.nodes[index].variable.unpack().into(),
            BddInner::Size32(bdd) => bdd.nodes[index].variable.unpack().into(),
            BddInner::Size64(bdd) => VariableId::new_long(bdd.nodes[index].variable.unpack())
                .unwrap_or_else(|| {
                    unreachable!("Variable stored in BDD table does not fit into standard range.")
                }),
        }
    }

    /// Returns the `low` and `high` children of the given `node`.
    pub fn get_links(&self, node: NodeId) -> (NodeId, NodeId) {
        // The unchecked casts are necessary to ensure we are not using any undefined values.
        let index: usize = node.unchecked_into();
        match &self.0 {
            BddInner::Size16(bdd) => {
                let BddNode16 { low, high, .. } = bdd.nodes[index];
                (low.unchecked_into(), high.unchecked_into())
            }
            BddInner::Size32(bdd) => {
                let BddNode32 { low, high, .. } = bdd.nodes[index];
                (low.unchecked_into(), high.unchecked_into())
            }
            BddInner::Size64(bdd) => {
                let BddNode64 { low, high, .. } = bdd.nodes[index];
                (low.unchecked_into(), high.unchecked_into())
            }
        }
    }

    /// Approximately counts the number of satisfying valuations of the `Bdd`. If
    /// `largest_variable` is [`Option::Some`], then it is assumed to be the largest
    /// variable. Otherwise, the largest variable in the BDD is used.
    ///
    /// # Panics
    ///
    /// Assumes that the given variable is greater than or equal to than any
    /// variable in the BDD. Otherwise, the function may give unexpected results
    /// in release mode or panic in debug mode.
    pub fn count_satisfying_valuations(&self, largest_variable: Option<VariableId>) -> f64 {
        match &self.0 {
            BddInner::Size16(bdd) => bdd.count_satisfying_valuations(largest_variable),
            BddInner::Size32(bdd) => bdd.count_satisfying_valuations(largest_variable),
            BddInner::Size64(bdd) => bdd.count_satisfying_valuations(largest_variable),
        }
    }

    /// Approximately counts the number of satisfying paths in the `Bdd`.
    pub fn count_satisfying_paths(&self) -> f64 {
        match &self.0 {
            BddInner::Size16(bdd) => bdd.count_satisfying_paths(),
            BddInner::Size32(bdd) => bdd.count_satisfying_paths(),
            BddInner::Size64(bdd) => bdd.count_satisfying_paths(),
        }
    }

    /// Writes this `Bdd` as a DOT graph to the given `output` stream.
    pub fn write_bdd_as_dot(&self, output: &mut dyn Write) -> io::Result<()> {
        match &self.0 {
            BddInner::Size16(bdd) => bdd.write_as_dot(output),
            BddInner::Size32(bdd) => bdd.write_as_dot(output),
            BddInner::Size64(bdd) => bdd.write_as_dot(output),
        }
    }

    /// Converts this `Bdd` into a DOT graph string.
    pub fn to_dot_string(&self) -> String {
        let mut output = Vec::new();
        self.write_bdd_as_dot(&mut output).unwrap();
        String::from_utf8(output).unwrap()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::bdd_node::BddNodeAny;
    use crate::conversion::UncheckedInto;
    use crate::node_id::{NodeId, NodeId16, NodeId32, NodeId64, NodeIdAny};
    use crate::split::bdd::{Bdd, Bdd16, Bdd32, Bdd64, BddAny};
    use crate::variable_id::{VarIdPacked16, VarIdPacked32, VarIdPacked64, VariableId};

    use super::BddInner;

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
                    let x_p = $Bdd::new_unchecked(x.root(), x.nodes.clone());
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
        match &bdd.0 {
            BddInner::Size32(b) => {
                // both checks should not be necessary, but they are here as a sanity check
                assert!(b.iff(&bdd32).unwrap().is_true());
                assert!(b.structural_eq(&bdd32));
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
        match &bdd.0 {
            BddInner::Size16(b) => {
                assert!(b.iff(&bdd16_ands).unwrap().is_true());
                assert!(b.structural_eq(&bdd16_ands));
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

        let bdd = Into::<Bdd>::into(bdd64).shrink();

        assert_eq!(bdd.node_count(), 1 << n);

        match &bdd.0 {
            BddInner::Size32(b) => {
                assert!(b.iff(&bdd32).unwrap().is_true());
                assert!(b.structural_eq(&bdd32));
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

        let bdd = Into::<Bdd>::into(bdd64).shrink();

        assert_eq!(bdd.node_count(), 1 << n);

        match &bdd.0 {
            BddInner::Size16(b) => {
                assert!(b.iff(&bdd16).unwrap().is_true());
                assert!(b.structural_eq(&bdd16));
            }
            _ => panic!("expected 16-bit BDD"),
        }
    }

    #[test]
    fn new_bdd_literal_16() {
        let var = VariableId::from(1u32);
        let bdd = Bdd::new_literal(var, true);
        let bdd16 = Bdd16::new_literal(var.unchecked_into(), true);

        match &bdd.0 {
            BddInner::Size16(b) => {
                assert!(b.iff(&bdd16).unwrap().is_true());
                assert!(b.structural_eq(&bdd16));
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

        match &bdd.0 {
            BddInner::Size32(b) => {
                assert!(b.iff(&bdd32).unwrap().is_true());
                assert!(b.structural_eq(&bdd32));
            }
            _ => panic!("expected 32-bit BDD"),
        }
    }

    #[test]
    fn new_bdd_literal_64() {
        let var = VariableId::new_long(VariableId::MAX_32_BIT_ID + 1).unwrap();
        let bdd = Bdd::new_literal(var, true);
        let bdd64 = Bdd64::new_literal(var.unchecked_into(), true);

        match &bdd.0 {
            BddInner::Size64(b) => {
                assert!(b.iff(&bdd64).unwrap().is_true());
                assert!(b.structural_eq(&bdd64));
            }
            _ => panic!("expected 64-bit BDD"),
        }
    }

    #[test]
    fn new_bdd_literal_64_but_should_be_16() {
        let var = VariableId::from(1u32);
        let bdd = Bdd::new_literal(var, true);
        let bdd16 = Bdd16::new_literal(var.unchecked_into(), true);

        match &bdd.0 {
            BddInner::Size16(b) => {
                assert!(b.iff(&bdd16).unwrap().is_true());
                assert!(b.structural_eq(&bdd16));
            }
            _ => panic!("expected 16-bit BDD"),
        }
    }

    #[test]
    fn new_bdd_literal_32_but_should_be_16() {
        let var = VariableId::from(1u32);
        let bdd = Bdd::new_literal(var, true);
        let bdd16 = Bdd16::new_literal(var.unchecked_into(), true);

        match &bdd.0 {
            BddInner::Size16(b) => {
                assert!(b.iff(&bdd16).unwrap().is_true());
                assert!(b.structural_eq(&bdd16));
            }
            _ => panic!("expected 16-bit BDD"),
        }
    }

    #[test]
    fn new_bdd_literal_64_but_should_be_32() {
        let var = VariableId::new_long(VariableId::MAX_16_BIT_ID + 1).unwrap();
        let bdd = Bdd::new_literal(var, true);
        let bdd32 = Bdd32::new_literal(var.unchecked_into(), true);

        match &bdd.0 {
            BddInner::Size32(b) => {
                assert!(b.iff(&bdd32).unwrap().is_true());
                assert!(b.structural_eq(&bdd32));
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
        let bdd16: Bdd = Bdd16::new_literal(VarIdPacked16::new(1234), true).into();
        let bdd32: Bdd = Bdd32::new_literal(VarIdPacked32::new(1234), true).into();
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

        let true_32: Bdd = Bdd32::new_true().into();
        let false_32: Bdd = Bdd32::new_false().into();
        assert!(true_32.is_true() && !true_32.is_false());
        assert!(false_32.is_false() && !false_32.is_true());

        let true_64: Bdd = Bdd64::new_true().into();
        let false_64: Bdd = Bdd64::new_false().into();
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

    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn queens(n: usize) -> Bdd {
        fn mk_negative_literals(n: usize) -> Vec<Bdd> {
            let mut bdd_literals = Vec::with_capacity(n * n);
            for i in 0..n {
                for j in 0..n {
                    let literal = Bdd::new_literal(((i * n + j) as u32).into(), false);
                    bdd_literals.push(literal);
                }
            }
            bdd_literals
        }

        fn one_queen(n: usize, i: usize, j: usize, negative: &[Bdd]) -> Bdd {
            let mut s = Bdd::new_literal(((i * n + j) as u32).into(), true);

            // no queens in the same row
            for k in 0..n {
                if k != j {
                    s = s.and(&negative[i * n + k]);
                }
            }

            // no queens in the same column
            for k in 0..n {
                if k != i {
                    s = s.and(&negative[k * n + j]);
                }
            }

            // no queens in the main diagonal (top-left to bot-right)
            // r - c = i - j  =>  c = (r + j) - i
            for row in 0..n {
                if let Some(col) = (row + j).checked_sub(i) {
                    if col < n && row != i {
                        s = s.and(&negative[row * n + col]);
                    }
                }
            }

            // no queens in the anti diagonal (top-right to bot-left)
            // r + c = i + j  =>  c = (i + j) - r
            for row in 0..n {
                if let Some(col) = (i + j).checked_sub(row) {
                    if col < n && row != i {
                        s = s.and(&negative[row * n + col]);
                    }
                }
            }

            s
        }

        fn queen_in_row(n: usize, row: usize, negative: &[Bdd]) -> Bdd {
            let mut r = Bdd::new_false();
            for col in 0..n {
                let one_queen = one_queen(n, row, col, negative);
                r = r.or(&one_queen);
            }
            r
        }

        let negative = mk_negative_literals(n);
        let mut result = Bdd::new_true();
        for row in 0..n {
            let in_row = queen_in_row(n, row, &negative);
            result = result.and(&in_row);
        }
        result
    }

    #[test]
    fn count_sat_valuations() {
        assert_eq!(Bdd::new_false().count_satisfying_valuations(None), 0.0,);

        assert_eq!(Bdd::new_true().count_satisfying_valuations(None), 1.0,);

        assert_eq!(
            Bdd::new_true().count_satisfying_valuations(Some(VariableId::new(0))),
            2.0,
        );

        assert_eq!(
            Bdd::new_literal(VariableId::new(0u32), true)
                .count_satisfying_valuations(Some(VariableId::new(0u32))),
            1.0,
        );

        assert_eq!(
            Bdd::new_literal(VariableId::new(0u32), true).count_satisfying_valuations(None),
            1.0,
        );

        assert_eq!(
            Bdd::new_literal(VariableId::new(0u32), false).count_satisfying_valuations(None),
            1.0,
        );

        assert_eq!(
            Bdd::new_literal(VariableId::new(0u32), false)
                .count_satisfying_valuations(Some(VariableId::new(2u32))),
            4.0,
        );

        let bdd6 = queens(6);
        let bdd8 = queens(9);

        assert_eq!(bdd6.count_satisfying_valuations(None), 4.0);
        assert_eq!(bdd8.count_satisfying_valuations(None), 352.0);

        let or3 = Bdd::new_literal(VariableId::new(0u32), true)
            .or(&Bdd::new_literal(VariableId::new(1u32), true))
            .or(&Bdd::new_literal(VariableId::new(3u32), true));

        let and3 = Bdd::new_literal(VariableId::new(0u32), true)
            .and(&Bdd::new_literal(VariableId::new(1u32), true))
            .and(&Bdd::new_literal(VariableId::new(3u32), true));

        assert_eq!(or3.count_satisfying_valuations(None), 14.0);
        assert_eq!(and3.count_satisfying_valuations(None), 2.0);
    }

    #[test]
    fn count_sat_paths() {
        assert_eq!(Bdd::new_false().count_satisfying_paths(), 0.0,);
        assert_eq!(Bdd::new_true().count_satisfying_paths(), 1.0,);

        assert_eq!(
            Bdd::new_literal(VariableId::new(0u32), true).count_satisfying_paths(),
            1.0,
        );

        assert_eq!(
            Bdd::new_literal(VariableId::new(0u32), false).count_satisfying_paths(),
            1.0,
        );

        let bdd4 = queens(6);
        let bdd8 = queens(9);

        assert_eq!(bdd4.count_satisfying_paths(), 4.0);
        assert_eq!(bdd8.count_satisfying_paths(), 352.0);

        let or3 = Bdd::new_literal(VariableId::new(0u32), true)
            .or(&Bdd::new_literal(VariableId::new(1u32), true))
            .or(&Bdd::new_literal(VariableId::new(3u32), true));

        let and3 = Bdd::new_literal(VariableId::new(0u32), true)
            .and(&Bdd::new_literal(VariableId::new(1u32), true))
            .and(&Bdd::new_literal(VariableId::new(3u32), true));

        assert_eq!(or3.count_satisfying_paths(), 3.0);
        assert_eq!(and3.count_satisfying_paths(), 1.0);
    }

    #[test]
    fn bdd_to_dot() {
        let v_1 = VariableId::new(1);
        let v_4 = VariableId::new(4);

        let bdd = Bdd::new_literal(v_1, true).xor(&Bdd::new_literal(v_4, true));

        let result = bdd.to_dot_string();

        let expected = r#"digraph BDD {
  __ruddy_root [label="", style=invis, height=0, width=0];
  __ruddy_root -> 4;

  edge [dir=none];

  0 [label="0", shape=box, width=0.3, height=0.3];
  1 [label="1", shape=box, width=0.3, height=0.3];
  2 [label="4", shape=box, width=0.3, height=0.3];
  2 -> 0 [style=dashed];
  2 -> 1;
  3 [label="4", shape=box, width=0.3, height=0.3];
  3 -> 1 [style=dashed];
  3 -> 0;
  4 [label="1", shape=box, width=0.3, height=0.3];
  4 -> 2 [style=dashed];
  4 -> 3;
}
"#;

        assert_eq!(result, expected);
    }
}
