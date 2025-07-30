use crate::{bdd_node::BddNodeAny, node_id::NodeIdAny, variable_id::VarIdPackedAny};

pub(crate) trait NodeAccess {
    type Id: NodeIdAny;
    type Node: BddNodeAny<Id = Self::Id>;

    unsafe fn get_node_unchecked(&self, id: Self::Id) -> &Self::Node;
}

struct SatisfyingPathsImpl<'a, TNodeSource: NodeAccess> {
    bdd: &'a TNodeSource,
    stack: Vec<TNodeSource::Id>,
    partial_valuation: Vec<Option<bool>>,
}

impl<'a, TNodeSource: NodeAccess> Iterator for SatisfyingPathsImpl<'a, TNodeSource> {
    type Item = Vec<Option<bool>>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(id) = self.stack.pop() {
            if id.is_zero() {
                continue;
            }

            if id.is_one() {
                return Some(self.partial_valuation.clone());
            }

            let node = unsafe { self.bdd.get_node_unchecked(id) };

            let assignment_to_variable = &mut self.partial_valuation[node.variable().as_usize()];

            match assignment_to_variable {
                None => {
                    *assignment_to_variable = Some(false);
                    self.stack.push(id);
                    self.stack.push(node.low());
                }
                Some(false) => {
                    *assignment_to_variable = Some(true);
                    self.stack.push(id);
                    self.stack.push(node.high());
                }
                Some(true) => {
                    *assignment_to_variable = None;
                }
            }
        }

        None
    }
}

struct SatisfyingValuationsImpl<'a, TNodeSource: NodeAccess> {
    sat_paths: SatisfyingPathsImpl<'a, TNodeSource>,
    valuation: Vec<bool>,
}

impl<'a, TNodeSource: NodeAccess> Iterator for SatisfyingValuationsImpl<'a, TNodeSource> {
    type Item = Vec<bool>;

    fn next(&mut self) -> Option<Self::Item> {
        for (assignment, _) in self
            .valuation
            .iter_mut()
            .zip(self.sat_paths.partial_valuation.iter())
            .filter(|(_, partial_assignment)| partial_assignment.is_none())
            .rev()
        {
            match assignment {
                true => {
                    *assignment = false;
                }
                false => {
                    *assignment = true;
                    return Some(self.valuation.clone());
                }
            }
        }
        // The final valuation had all of the variables assigned to `true`, so the
        // loop exited without returning a value. Move to next path.
        let new_path = self.sat_paths.next()?;

        // Start with all of the variables not on the path assigned to `false`.
        self.valuation = new_path
            .iter()
            .map(|assignment| assignment.unwrap_or(false))
            .collect();

        Some(self.valuation.clone())
    }
}

enum SatisfyingPathsInner<
    TSatPaths16: Iterator<Item = Vec<Option<bool>>>,
    TSatPaths32: Iterator<Item = Vec<Option<bool>>>,
    TSatPaths64: Iterator<Item = Vec<Option<bool>>>,
> {
    Size16(TSatPaths16),
    Size32(TSatPaths32),
    Size64(TSatPaths64),
}

enum SatisfyingValuationsInner<
    TSatValuations16: Iterator<Item = Vec<bool>>,
    TSatValuations32: Iterator<Item = Vec<bool>>,
    TSatValuations64: Iterator<Item = Vec<bool>>,
> {
    Size16(TSatValuations16),
    Size32(TSatValuations32),
    Size64(TSatValuations64),
}

pub(crate) mod split {
    use crate::{
        node_id::NodeIdAny,
        split::{
            Bdd,
            bdd::{Bdd16, Bdd32, Bdd64, BddAny, BddImpl, BddInner},
        },
        variable_id::{VarIdPackedAny, VariableId},
    };

    use super::{
        NodeAccess, SatisfyingPathsImpl, SatisfyingPathsInner, SatisfyingValuationsImpl,
        SatisfyingValuationsInner,
    };

    impl<TNodeId: NodeIdAny, TVarId: VarIdPackedAny> NodeAccess for BddImpl<TNodeId, TVarId> {
        type Id = TNodeId;
        type Node = <Self as BddAny>::Node;

        unsafe fn get_node_unchecked(&self, id: Self::Id) -> &Self::Node {
            BddAny::get_node_unchecked(self, id)
        }
    }

    /// An iterator over the satisfying paths in a split [`Bdd`].
    pub struct SatisfyingPaths<'a>(
        SatisfyingPathsInner<
            SatisfyingPathsImpl<'a, Bdd16>,
            SatisfyingPathsImpl<'a, Bdd32>,
            SatisfyingPathsImpl<'a, Bdd64>,
        >,
    );

    impl<'a> Iterator for SatisfyingPaths<'a> {
        type Item = Vec<Option<bool>>;

        fn next(&mut self) -> Option<Self::Item> {
            match &mut self.0 {
                SatisfyingPathsInner::Size16(iterator) => iterator.next(),
                SatisfyingPathsInner::Size32(iterator) => iterator.next(),
                SatisfyingPathsInner::Size64(iterator) => iterator.next(),
            }
        }
    }

    /// An iterator over the satisfying valuations of a split [`Bdd`].
    pub struct SatisfyingValuations<'a>(
        SatisfyingValuationsInner<
            SatisfyingValuationsImpl<'a, Bdd16>,
            SatisfyingValuationsImpl<'a, Bdd32>,
            SatisfyingValuationsImpl<'a, Bdd64>,
        >,
    );

    impl<'a> Iterator for SatisfyingValuations<'a> {
        type Item = Vec<bool>;

        fn next(&mut self) -> Option<Self::Item> {
            match &mut self.0 {
                SatisfyingValuationsInner::Size16(iterator) => iterator.next(),
                SatisfyingValuationsInner::Size32(iterator) => iterator.next(),
                SatisfyingValuationsInner::Size64(iterator) => iterator.next(),
            }
        }
    }

    impl<TNodeId: NodeIdAny, TVarId: VarIdPackedAny> BddImpl<TNodeId, TVarId> {
        /// Gets an iterator over the satisfying paths in this BDD.
        fn satisfying_paths(
            &self,
            largest_variable: Option<VariableId>,
        ) -> SatisfyingPathsImpl<'_, Self> {
            SatisfyingPathsImpl {
                bdd: self,
                stack: vec![self.root()],
                partial_valuation: if self.is_false() {
                    vec![]
                } else if self.is_true() {
                    if let Some(largest_variable) = largest_variable {
                        vec![None; largest_variable.as_usize() + 1]
                    } else {
                        vec![]
                    }
                } else {
                    vec![
                        None;
                        largest_variable
                            .unwrap_or_else(|| self.get_largest_variable())
                            .as_usize()
                            + 1
                    ]
                },
            }
        }

        /// Gets an iterator over the satisfying valuations of this BDD.
        fn satisfying_valuations(
            &self,
            largest_variable: Option<VariableId>,
        ) -> SatisfyingValuationsImpl<'_, Self> {
            SatisfyingValuationsImpl {
                sat_paths: self.satisfying_paths(largest_variable),
                valuation: vec![],
            }
        }
    }

    impl Bdd {
        /// Gets an iterator over the satisfying paths in this `Bdd`. If `largest_variable`
        /// is [`Option::Some`], then it is assumed to be the largest variable.
        /// Otherwise, the largest variable in the BDD is used.
        ///
        /// # Panics
        ///
        /// Panics if the given variable is smaller than any variable in the BDD.
        pub fn satisfying_paths(&self, largest_variable: Option<VariableId>) -> SatisfyingPaths {
            match &self.0 {
                BddInner::Size16(bdd) => SatisfyingPaths(SatisfyingPathsInner::Size16(
                    bdd.satisfying_paths(largest_variable),
                )),
                BddInner::Size32(bdd) => SatisfyingPaths(SatisfyingPathsInner::Size32(
                    bdd.satisfying_paths(largest_variable),
                )),
                BddInner::Size64(bdd) => SatisfyingPaths(SatisfyingPathsInner::Size64(
                    bdd.satisfying_paths(largest_variable),
                )),
            }
        }

        /// Gets an iterator over the satisfying valuations of this `Bdd`. If `largest_variable`
        /// is [`Option::Some`], then it is assumed to be the largest variable.
        /// Otherwise, the largest variable in the BDD is used.
        ///
        /// # Panics
        ///
        /// Panics if the given variable is smaller than any variable in the BDD.
        pub fn satisfying_valuation(
            &self,
            largest_variable: Option<VariableId>,
        ) -> SatisfyingValuations {
            match &self.0 {
                BddInner::Size16(bdd) => SatisfyingValuations(SatisfyingValuationsInner::Size16(
                    bdd.satisfying_valuations(largest_variable),
                )),
                BddInner::Size32(bdd) => SatisfyingValuations(SatisfyingValuationsInner::Size32(
                    bdd.satisfying_valuations(largest_variable),
                )),
                BddInner::Size64(bdd) => SatisfyingValuations(SatisfyingValuationsInner::Size64(
                    bdd.satisfying_valuations(largest_variable),
                )),
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::{
            split::{Bdd, bdd::tests::queens},
            variable_id::VariableId,
        };

        #[test]
        fn iter_sat_paths() {
            assert!(Bdd::new_false().satisfying_paths(None).next().is_none());
            assert_eq!(Bdd::new_true().satisfying_paths(None).next(), Some(vec![]));
            assert_eq!(
                Bdd::new_true()
                    .satisfying_paths(Some(VariableId::new(1)))
                    .next(),
                Some(vec![None, None])
            );

            let v0 = VariableId::new(0);
            let v1 = VariableId::new(1);
            let v2 = VariableId::new(2);
            let v3 = VariableId::new(3);

            let bdd = Bdd::new_literal(v0, true)
                .or(&Bdd::new_literal(v1, true))
                .or(&Bdd::new_literal(v2, false));

            let iter = bdd.satisfying_paths(Some(v3));

            let valuations_in_order = [
                vec![Some(false), Some(false), Some(false), None],
                vec![Some(false), Some(true), None, None],
                vec![Some(true), None, None, None],
            ];

            for (i, partial_valuation) in iter.enumerate() {
                assert_eq!(partial_valuation, valuations_in_order[i]);
            }

            let bdd9 = queens(9);
            assert_eq!(bdd9.satisfying_paths(None).count(), 352);
        }

        #[test]
        fn iter_sat_valuations() {
            let f = Bdd::new_false();

            assert!(f.satisfying_valuation(None).next().is_none());
            assert!(
                f.satisfying_valuation(Some(VariableId::new(2)))
                    .next()
                    .is_none()
            );

            let t = Bdd::new_true();

            assert_eq!(
                Bdd::new_true().satisfying_valuation(None).next(),
                Some(vec![])
            );

            let iter = t.satisfying_valuation(Some(VariableId::new(1)));
            let valuations_in_order = [
                vec![false, false],
                vec![false, true],
                vec![true, false],
                vec![true, true],
            ];

            for (i, valuation) in iter.enumerate() {
                assert_eq!(valuation, valuations_in_order[i]);
            }

            let bdd = Bdd::new_literal(VariableId::new(0), true)
                .or(&Bdd::new_literal(VariableId::new(1), true))
                .or(&Bdd::new_literal(VariableId::new(2), false));

            let iter = bdd.satisfying_valuation(Some(VariableId::new(3)));

            let valuations_in_order = [
                vec![false, false, false, false],
                vec![false, false, false, true],
                vec![false, true, false, false],
                vec![false, true, false, true],
                vec![false, true, true, false],
                vec![false, true, true, true],
                vec![true, false, false, false],
                vec![true, false, false, true],
                vec![true, false, true, false],
                vec![true, false, true, true],
                vec![true, true, false, false],
                vec![true, true, false, true],
                vec![true, true, true, false],
                vec![true, true, true, true],
            ];

            for (i, valuation) in iter.enumerate() {
                assert_eq!(valuation, valuations_in_order[i]);
            }

            let bdd9 = queens(9);
            assert_eq!(bdd9.satisfying_valuation(None).count(), 352);
        }
    }
}

pub(crate) mod shared {
    use crate::{
        bdd_node::BddNodeAny,
        conversion::UncheckedInto,
        node_id::NodeIdAny,
        node_table::{
            NodeTable, NodeTable16, NodeTable32, NodeTable64, NodeTableAny, NodeTableImpl,
        },
        shared::{Bdd, BddManager},
        variable_id::{VarIdPackedAny, VariableId},
    };

    use super::{
        NodeAccess, SatisfyingPathsImpl, SatisfyingPathsInner, SatisfyingValuationsImpl,
        SatisfyingValuationsInner,
    };

    impl<
        TNodeId: NodeIdAny,
        TVarId: VarIdPackedAny,
        TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
    > NodeAccess for NodeTableImpl<TNodeId, TVarId, TNode>
    {
        type Id = TNodeId;
        type Node = TNode;

        unsafe fn get_node_unchecked(&self, id: Self::Id) -> &Self::Node {
            NodeTableAny::get_node_unchecked(self, id)
        }
    }

    /// An iterator over the satisfying paths in a shared [`Bdd`].
    pub struct SatisfyingPaths<'a>(
        SatisfyingPathsInner<
            SatisfyingPathsImpl<'a, NodeTable16>,
            SatisfyingPathsImpl<'a, NodeTable32>,
            SatisfyingPathsImpl<'a, NodeTable64>,
        >,
    );

    impl<'a> Iterator for SatisfyingPaths<'a> {
        type Item = Vec<Option<bool>>;

        fn next(&mut self) -> Option<Self::Item> {
            match &mut self.0 {
                SatisfyingPathsInner::Size16(iterator) => iterator.next(),
                SatisfyingPathsInner::Size32(iterator) => iterator.next(),
                SatisfyingPathsInner::Size64(iterator) => iterator.next(),
            }
        }
    }

    /// An iterator over the satisfying valuations of a shared [`Bdd`].
    pub struct SatisfyingValuations<'a>(
        SatisfyingValuationsInner<
            SatisfyingValuationsImpl<'a, NodeTable16>,
            SatisfyingValuationsImpl<'a, NodeTable32>,
            SatisfyingValuationsImpl<'a, NodeTable64>,
        >,
    );

    impl<'a> Iterator for SatisfyingValuations<'a> {
        type Item = Vec<bool>;

        fn next(&mut self) -> Option<Self::Item> {
            match &mut self.0 {
                SatisfyingValuationsInner::Size16(iterator) => iterator.next(),
                SatisfyingValuationsInner::Size32(iterator) => iterator.next(),
                SatisfyingValuationsInner::Size64(iterator) => iterator.next(),
            }
        }
    }

    impl<
        TNodeId: NodeIdAny,
        TVarId: VarIdPackedAny,
        TNode: BddNodeAny<Id = TNodeId, VarId = TVarId>,
    > NodeTableImpl<TNodeId, TVarId, TNode>
    {
        /// Gets an iterator over the satisfying paths in the BDD rooted in `root`.
        fn satisfying_paths(
            &self,
            root: TNodeId,
            largest_variable: Option<VariableId>,
        ) -> SatisfyingPathsImpl<'_, Self> {
            let largest_variable = largest_variable.or_else(|| self.get_largest_variable());
            SatisfyingPathsImpl {
                bdd: self,
                stack: vec![root],
                partial_valuation: if root.is_zero()
                    || (root.is_one() && largest_variable.is_none())
                {
                    vec![]
                } else {
                    vec![
                        None;
                        largest_variable
                            .expect("node table contains non-terminal nodes")
                            .as_usize()
                            + 1
                    ]
                },
            }
        }

        /// Gets an iterator over the satisfying valuations of the BDD rooted in `root`.
        fn satisfying_valuations(
            &self,
            root: TNodeId,
            largest_variable: Option<VariableId>,
        ) -> SatisfyingValuationsImpl<'_, Self> {
            SatisfyingValuationsImpl {
                sat_paths: self.satisfying_paths(root, largest_variable),
                valuation: vec![],
            }
        }
    }

    impl BddManager {
        /// Gets an iterator over the satisfying paths in the `bdd`. If `largest_variable`
        /// is [`Option::Some`], then it is assumed to be the largest variable.
        /// Otherwise, the largest variable residing in the manager is used.
        ///
        /// # Panics
        ///
        /// Panics if the given variable is smaller than any variable in the BDD.
        pub fn satisfying_paths(
            &self,
            bdd: &Bdd,
            largest_variable: Option<VariableId>,
        ) -> SatisfyingPaths {
            match &self.unique_table {
                NodeTable::Size16(table) => SatisfyingPaths(SatisfyingPathsInner::Size16(
                    table.satisfying_paths(bdd.root.get().unchecked_into(), largest_variable),
                )),
                NodeTable::Size32(table) => SatisfyingPaths(SatisfyingPathsInner::Size32(
                    table.satisfying_paths(bdd.root.get().unchecked_into(), largest_variable),
                )),
                NodeTable::Size64(table) => SatisfyingPaths(SatisfyingPathsInner::Size64(
                    table.satisfying_paths(bdd.root.get().unchecked_into(), largest_variable),
                )),
            }
        }

        /// Gets an iterator over the satisfying valuations in the `bdd`. If `largest_variable`
        /// is [`Option::Some`], then it is assumed to be the largest variable.
        /// Otherwise, the largest variable residing in the manager is used.
        ///
        /// # Panics
        ///
        /// Panics if the given variable is smaller than any variable in the BDD.
        pub fn satisfying_valuations(
            &self,
            bdd: &Bdd,
            largest_variable: Option<VariableId>,
        ) -> SatisfyingValuations {
            match &self.unique_table {
                NodeTable::Size16(table) => {
                    SatisfyingValuations(SatisfyingValuationsInner::Size16(
                        table.satisfying_valuations(
                            bdd.root.get().unchecked_into(),
                            largest_variable,
                        ),
                    ))
                }
                NodeTable::Size32(table) => {
                    SatisfyingValuations(SatisfyingValuationsInner::Size32(
                        table.satisfying_valuations(
                            bdd.root.get().unchecked_into(),
                            largest_variable,
                        ),
                    ))
                }
                NodeTable::Size64(table) => {
                    SatisfyingValuations(SatisfyingValuationsInner::Size64(
                        table.satisfying_valuations(
                            bdd.root.get().unchecked_into(),
                            largest_variable,
                        ),
                    ))
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::{
            shared::{BddManager, manager::tests::queens},
            variable_id::VariableId,
        };

        #[test]
        fn iter_sat_paths() {
            let mut m = BddManager::no_gc();

            assert!(
                m.satisfying_paths(&m.new_bdd_false(), None)
                    .next()
                    .is_none()
            );
            assert_eq!(
                m.satisfying_paths(&m.new_bdd_true(), None).next(),
                Some(vec![])
            );

            let v0 = VariableId::new(0);
            let v1 = VariableId::new(1);
            let v2 = VariableId::new(2);
            let v3 = VariableId::new(3);

            let bdd0 = m.new_bdd_literal(v0, true);
            let bdd1 = m.new_bdd_literal(v1, true);

            assert_eq!(
                m.satisfying_paths(&m.new_bdd_true(), None).next(),
                Some(vec![None, None])
            );

            let bdd2 = m.new_bdd_literal(v2, false);
            let _bdd3 = m.new_bdd_literal(v3, true);
            let mut bdd = m.or(&bdd0, &bdd1);
            bdd = m.or(&bdd, &bdd2);

            let valuations_in_order = [
                vec![Some(false), Some(false), Some(false), None],
                vec![Some(false), Some(true), None, None],
                vec![Some(true), None, None, None],
            ];
            let iter = m.satisfying_paths(&bdd, None);
            for (i, partial_valuation) in iter.enumerate() {
                assert_eq!(partial_valuation, valuations_in_order[i]);
            }

            let (m, bdd9) = queens(9);
            assert_eq!(m.satisfying_paths(&bdd9, None).count(), 352);
        }

        #[test]
        fn iter_sat_valuations() {
            let mut m = BddManager::no_gc();

            assert!(
                m.satisfying_paths(&m.new_bdd_false(), None)
                    .next()
                    .is_none()
            );
            assert_eq!(
                m.satisfying_paths(&m.new_bdd_true(), None).next(),
                Some(vec![])
            );

            let v0 = VariableId::new(0);
            let v1 = VariableId::new(1);
            let v2 = VariableId::new(2);
            let v3 = VariableId::new(3);

            let bdd0 = m.new_bdd_literal(v0, true);
            let bdd1 = m.new_bdd_literal(v1, true);

            let valuations_in_order = [
                vec![false, false],
                vec![false, true],
                vec![true, false],
                vec![true, true],
            ];
            let iter = m.satisfying_valuations(&m.new_bdd_true(), None);
            for (i, partial_valuation) in iter.enumerate() {
                assert_eq!(partial_valuation, valuations_in_order[i]);
            }

            let bdd2 = m.new_bdd_literal(v2, false);
            let _bdd3 = m.new_bdd_literal(v3, true);

            let mut bdd = m.or(&bdd0, &bdd1);
            bdd = m.or(&bdd, &bdd2);

            let valuations_in_order = [
                vec![false, false, false, false],
                vec![false, false, false, true],
                vec![false, true, false, false],
                vec![false, true, false, true],
                vec![false, true, true, false],
                vec![false, true, true, true],
                vec![true, false, false, false],
                vec![true, false, false, true],
                vec![true, false, true, false],
                vec![true, false, true, true],
                vec![true, true, false, false],
                vec![true, true, false, true],
                vec![true, true, true, false],
                vec![true, true, true, true],
            ];

            let iter = m.satisfying_valuations(&bdd, None);
            for (i, partial_valuation) in iter.enumerate() {
                assert_eq!(partial_valuation, valuations_in_order[i]);
            }

            let (m, bdd9) = queens(9);
            assert_eq!(m.satisfying_valuations(&bdd9, None).count(), 352);
        }
    }
}
