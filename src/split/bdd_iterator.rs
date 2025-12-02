use crate::conversion::UncheckedInto;
use crate::node_id::NodeId64;
use crate::split::bdd::{Bdd64, BddAny};
use crate::{NodeIdAny, VariableId};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
enum PathChoice {
    // Decisions based on the given decision node, eventually leading into the given child node.
    PositiveNode(VariableId, #[allow(dead_code)] NodeId64, NodeId64),
    NegativeNode(VariableId, NodeId64, NodeId64),
    // Decisions based on free variables, eventually leading to the given node.
    PositiveFree(VariableId, NodeId64),
    NegativeFree(VariableId, NodeId64),
}

impl PathChoice {
    fn variable_id(&self) -> VariableId {
        match self {
            PathChoice::PositiveNode(var, _, _) => *var,
            PathChoice::NegativeNode(var, _, _) => *var,
            PathChoice::PositiveFree(var, _) => *var,
            PathChoice::NegativeFree(var, _) => *var,
        }
    }

    fn next_node(&self) -> NodeId64 {
        match self {
            PathChoice::PositiveNode(_, _, x) => *x,
            PathChoice::NegativeNode(_, _, x) => *x,
            PathChoice::PositiveFree(_, x) => *x,
            PathChoice::NegativeFree(_, x) => *x,
        }
    }

    fn to_value_pair(&self) -> (VariableId, bool) {
        match self {
            PathChoice::PositiveNode(var, _, _) => (*var, true),
            PathChoice::NegativeNode(var, _, _) => (*var, false),
            PathChoice::PositiveFree(var, _) => (*var, true),
            PathChoice::NegativeFree(var, _) => (*var, false),
        }
    }
}

pub struct ValuationsIterator {
    bdd: Bdd64,
    min_variable: VariableId,
    max_variable: VariableId,
    next_path: Option<Vec<PathChoice>>,
}

impl Iterator for ValuationsIterator {
    type Item = BTreeMap<VariableId, bool>;

    fn next(&mut self) -> Option<Self::Item> {
        let next_path = self.next_path.as_ref()?;
        let next_result = next_path
            .iter()
            .map(|it| it.to_value_pair())
            .collect::<BTreeMap<VariableId, bool>>();

        self.next_path = self.find_next_path();
        Some(next_result)
    }
}

impl ValuationsIterator {
    pub(crate) fn new(bdd: Bdd64, min_variable: VariableId, max_variable: VariableId) -> Self {
        assert!(min_variable <= max_variable);
        // TODO: Maybe we should test that the max variable is "in range" for this BDD.

        let mut it = ValuationsIterator {
            bdd,
            min_variable,
            max_variable,
            next_path: None,
        };
        it.next_path = it.find_first_path();
        it
    }

    /// Find the valuation that follows after the one currently stored in this
    /// [`ValuationsIterator`]. If there is no additional valuation stored, this method also
    /// returns `None`.
    fn find_next_path(&mut self) -> Option<Vec<PathChoice>> {
        let mut path = self.next_path.clone()?;

        // Pop from the path until we find a place where we can advance to a new path.
        while let Some(top) = path.pop() {
            match top {
                PathChoice::PositiveFree(_, _) => {
                    // Positive free choice cannot advance anywhere.
                    continue;
                }
                PathChoice::PositiveNode(_, _, _) => {
                    // Positive dependent choice also cannot advance anywhere.
                    continue;
                }
                PathChoice::NegativeFree(var, next_node) => {
                    // If the last choice is a negative free variable, we can advance that
                    // free choice to a positive one.
                    path.push(PathChoice::PositiveFree(var, next_node));
                    break;
                }
                PathChoice::NegativeNode(var, node, _) => {
                    // If the last choice is a negative dependent variable, we can advance it
                    // if and only if the positive link does not point directly to zero.
                    let node_data = self
                        .bdd
                        .get(node)
                        .expect("Correctness violation: node does not exist.");
                    if !node_data.high.is_zero() {
                        path.push(PathChoice::PositiveNode(var, node, node_data.high));
                        break;
                    }
                }
            }
        }

        if path.is_empty() {
            // If the path is now empty, we can't advance anywhere and this was the last path.
            return None;
        }

        self.extend_path(&mut path);

        Some(path)
    }

    /// Find the first (most zero) valuation in this BDD, assuming that some exists.
    fn find_first_path(&self) -> Option<Vec<PathChoice>> {
        if self.bdd.is_false() {
            return None;
        }

        let mut path = Vec::new();

        // Initialize the path with the first node, which is either root decision,
        // or a free choice.

        let root = self.bdd.root;
        let root_data = self
            .bdd
            .get(root)
            .expect("Correctness violation: root does not exist.");
        let root_var: VariableId = if root.is_one() {
            self.max_variable.next()
        } else {
            root_data.variable.unchecked_into()
        };
        assert!(self.min_variable <= root_var);

        if root_var == self.min_variable {
            // If the root has the minimal decision variable, we can condition on it directly.
            if root_data.low.is_zero() {
                // If the negative path ends in zero, skip it right away.
                path.push(PathChoice::PositiveNode(root_var, root, root_data.high));
            } else {
                // If it does not, pick it in order to get the most negative valuation.
                path.push(PathChoice::NegativeNode(root_var, root, root_data.low));
            }
        } else {
            // If the root node has a different (higher) variable, we have a free choice to make
            // that should eventually lead us to the root.
            path.push(PathChoice::NegativeFree(self.min_variable, root));
        }

        // Continue the path:
        self.extend_path(&mut path);
        Some(path)
    }

    /// Extend the given path to the first (most negative) complete valuation. The path must not
    /// be empty, as the next variable/node is read from the last element in the path.
    fn extend_path(&self, path: &mut Vec<PathChoice>) {
        let last = path
            .last()
            .cloned()
            .expect("Correctness violation: path cannot be empty.");

        let mut next_var = last.variable_id().next();
        let mut next_node = last.next_node();

        while next_var <= self.max_variable {
            let next_node_data = self
                .bdd
                .get(next_node)
                .expect("Correctness violation: node does not exist.");

            let next_node_var: VariableId = if next_node.is_one() {
                self.max_variable.next()
            } else {
                next_node_data.variable.unchecked_into()
            };

            assert!(next_node_var >= next_var);
            assert!(!next_node.is_zero());

            if next_node_var == next_var {
                // We have to make a decision on this node.
                if next_node_data.low.is_zero() {
                    // If low is `0`, we have to go high.
                    path.push(PathChoice::PositiveNode(
                        next_var,
                        next_node,
                        next_node_data.high,
                    ));
                    next_node = next_node_data.high;
                } else {
                    // If low is not `0`, we can follow it, and eventually we will find some `1`.
                    path.push(PathChoice::NegativeNode(
                        next_var,
                        next_node,
                        next_node_data.low,
                    ));
                    next_node = next_node_data.low;
                }
            } else {
                // This is a free decision, just pick `0`.
                path.push(PathChoice::NegativeFree(next_var, next_node));
            }

            // Move on to next variable.
            next_var = next_var.next();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ValuationsIterator;
    use crate::conversion::UncheckedInto;
    use crate::split::bdd::{Bdd, Bdd64, BddAny};
    use crate::variable_id::VariableId;
    use std::collections::BTreeMap;

    fn collect_valuations(iter: ValuationsIterator) -> Vec<BTreeMap<VariableId, bool>> {
        iter.collect()
    }

    #[test]
    fn test_false_bdd() {
        let bdd = Bdd64::new_false();
        let min_var = VariableId::zero();
        let max_var = VariableId::zero();
        let iter = ValuationsIterator::new(bdd, min_var, max_var);
        let valuations: Vec<_> = collect_valuations(iter);
        assert_eq!(valuations.len(), 0);
    }

    #[test]
    fn test_single_literal_positive() {
        let var = VariableId::zero();
        let bdd = Bdd64::new_literal(var.unchecked_into(), true);
        let min_var = VariableId::zero();
        let max_var = VariableId::zero();
        let iter = ValuationsIterator::new(bdd, min_var, max_var);
        let valuations: Vec<_> = collect_valuations(iter);

        assert_eq!(valuations.len(), 1);
        assert_eq!(valuations[0].get(&var), Some(&true));
    }

    #[test]
    fn test_single_literal_negative() {
        let var = VariableId::zero();
        let bdd = Bdd64::new_literal(var.unchecked_into(), false);
        let min_var = VariableId::zero();
        let max_var = VariableId::zero();
        let iter = ValuationsIterator::new(bdd, min_var, max_var);
        let valuations: Vec<_> = collect_valuations(iter);

        assert_eq!(valuations.len(), 1);
        assert_eq!(valuations[0].get(&var), Some(&false));
    }

    #[test]
    fn test_and_two_variables() {
        let v0 = VariableId::zero();
        let v1 = VariableId::from(1u32);
        let bdd = Bdd64::new_literal(v0.unchecked_into(), true)
            .and(&Bdd64::new_literal(v1.unchecked_into(), true))
            .unwrap();
        let min_var = VariableId::zero();
        let max_var = VariableId::from(1u32);
        let iter = ValuationsIterator::new(bdd, min_var, max_var);
        let valuations: Vec<_> = collect_valuations(iter);

        assert_eq!(valuations.len(), 1);
        assert_eq!(valuations[0].get(&v0), Some(&true));
        assert_eq!(valuations[0].get(&v1), Some(&true));
    }

    #[test]
    fn test_gap_in_variables() {
        let v0 = VariableId::zero();
        let v2 = VariableId::from(2u32);

        // Create BDD: v0 && v2
        let bdd = Bdd64::new_literal(v0.unchecked_into(), true)
            .and(&Bdd64::new_literal(v2.unchecked_into(), true))
            .unwrap();
        let min_var = VariableId::zero();
        let max_var = VariableId::from(2u32);
        let iter = ValuationsIterator::new(bdd, min_var, max_var);
        let valuations: Vec<_> = collect_valuations(iter);

        // Should get 2 valuations: (true, false, true) and (true, true, true)
        // v1 is free, so it can be either false or true
        assert_eq!(valuations.len(), 2);

        for valuation in &valuations {
            assert_eq!(valuation.get(&v0), Some(&true));
            assert_eq!(valuation.get(&v2), Some(&true));
        }

        let v1 = VariableId::from(1u32);
        let v1_values: Vec<_> = valuations.iter().map(|v| v.get(&v1).copied()).collect();

        // v1 should be both false and true
        assert!(v1_values.contains(&Some(false)));
        assert!(v1_values.contains(&Some(true)));
    }
    #[test]
    fn test_true_bdd_single_variable() {
        let bdd = Bdd64::new_true();
        let min_var = VariableId::zero();
        let max_var = VariableId::zero();
        let iter = ValuationsIterator::new(bdd, min_var, max_var);
        let valuations: Vec<_> = collect_valuations(iter);

        assert_eq!(valuations.len(), 2);
        let val0 = &valuations[0];
        let val1 = &valuations[1];
        assert_eq!(val0.get(&min_var), Some(&false));
        assert_eq!(val1.get(&min_var), Some(&true));
    }

    #[test]
    fn test_true_bdd_multiple_variables() {
        let bdd = Bdd64::new_true();
        let min_var = VariableId::zero();
        let max_var = VariableId::from(2u32);
        let iter = ValuationsIterator::new(bdd, min_var, max_var);
        let valuations: Vec<_> = collect_valuations(iter);

        assert_eq!(valuations.len(), 8);

        // Check lexicographic ordering
        let expected = vec![
            vec![false, false, false],
            vec![false, false, true],
            vec![false, true, false],
            vec![false, true, true],
            vec![true, false, false],
            vec![true, false, true],
            vec![true, true, false],
            vec![true, true, true],
        ];

        for (i, valuation) in valuations.iter().enumerate() {
            let mut vec_val = Vec::new();
            let mut var = min_var;
            while var <= max_var {
                vec_val.push(*valuation.get(&var).unwrap_or(&false));
                var = var.next();
            }
            assert_eq!(
                vec_val, expected[i],
                "Valuation {} should match expected",
                i
            );
        }
    }

    #[test]
    fn test_single_literal_with_free_variables() {
        // Issue: When a literal BDD has free variables in the range that lead to terminal nodes,
        // the iterator fails to handle the terminal node when extending the path.
        // Expected: Should return 4 valuations (v0 and v2 are free, v1 must be true)
        // Current: Panics when trying to unpack variable from terminal node
        let var = VariableId::from(1u32);
        let bdd = Bdd64::new_literal(var.unchecked_into(), true);
        let min_var = VariableId::zero();
        let max_var = VariableId::from(2u32);
        let iter = ValuationsIterator::new(bdd, min_var, max_var);
        let valuations: Vec<_> = collect_valuations(iter);

        assert_eq!(valuations.len(), 4);

        for valuation in &valuations {
            assert_eq!(valuation.get(&var), Some(&true));
        }

        let v0_values: Vec<_> = valuations
            .iter()
            .map(|v| v.get(&min_var).copied())
            .collect();
        let v2_values: Vec<_> = valuations
            .iter()
            .map(|v| v.get(&VariableId::from(2u32)).copied())
            .collect();

        assert_eq!(
            v0_values,
            vec![Some(false), Some(false), Some(true), Some(true)]
        );
        assert_eq!(
            v2_values,
            vec![Some(false), Some(true), Some(false), Some(true)]
        );
    }

    #[test]
    fn test_or_two_variables() {
        // Expected: Should return 3 valuations: (true, false), (false, true), (true, true)
        let v0 = VariableId::zero();
        let v1 = VariableId::from(1u32);
        let bdd = Bdd64::new_literal(v0.unchecked_into(), true)
            .or(&Bdd64::new_literal(v1.unchecked_into(), true))
            .unwrap();
        let min_var = VariableId::zero();
        let max_var = VariableId::from(1u32);
        let iter = ValuationsIterator::new(bdd, min_var, max_var);
        let valuations: Vec<_> = collect_valuations(iter);

        assert_eq!(valuations.len(), 3);

        let vec_vals: Vec<_> = valuations
            .iter()
            .map(|v| {
                let mut vec = Vec::new();
                let mut var = min_var;
                while var <= max_var {
                    vec.push(*v.get(&var).unwrap_or(&false));
                    var = var.next();
                }
                vec
            })
            .collect();

        assert!(vec_vals.contains(&vec![false, true]));
        assert!(vec_vals.contains(&vec![true, false]));
        assert!(vec_vals.contains(&vec![true, true]));
        assert!(!vec_vals.contains(&vec![false, false]));
    }

    #[test]
    fn test_xor_two_variables() {
        // Issue: XOR creates paths that lead to terminal nodes, causing failures.
        // Expected: Should return 2 valuations: (true, false), (false, true)
        // Current: Returns 3 valuations (includes an extra valuation due to iterator behavior)
        let v0 = VariableId::zero();
        let v1 = VariableId::from(1u32);
        let bdd = Bdd64::new_literal(v0.unchecked_into(), true)
            .xor(&Bdd64::new_literal(v1.unchecked_into(), true))
            .unwrap();
        let min_var = VariableId::zero();
        let max_var = VariableId::from(1u32);
        let iter = ValuationsIterator::new(bdd, min_var, max_var);
        let valuations: Vec<_> = collect_valuations(iter);

        assert_eq!(valuations.len(), 2);

        let vec_vals: Vec<_> = valuations
            .iter()
            .map(|v| {
                let mut vec = Vec::new();
                let mut var = min_var;
                while var <= max_var {
                    vec.push(*v.get(&var).unwrap_or(&false));
                    var = var.next();
                }
                vec
            })
            .collect();

        assert!(vec_vals.contains(&vec![false, true]));
        assert!(vec_vals.contains(&vec![true, false]));
    }

    #[test]
    fn test_lexicographic_ordering_with_or() {
        // Expected: Should return 7 valuations in lexicographic order (all except (false, false, false))
        let v0 = VariableId::zero();
        let v1 = VariableId::from(1u32);
        let v2 = VariableId::from(2u32);

        let bdd = Bdd64::new_literal(v0.unchecked_into(), true)
            .or(&Bdd64::new_literal(v1.unchecked_into(), true))
            .unwrap()
            .or(&Bdd64::new_literal(v2.unchecked_into(), true))
            .unwrap();

        let min_var = VariableId::zero();
        let max_var = VariableId::from(2u32);
        let iter = ValuationsIterator::new(bdd, min_var, max_var);
        let valuations: Vec<_> = collect_valuations(iter);

        assert_eq!(valuations.len(), 7);

        let vec_vals: Vec<_> = valuations
            .iter()
            .map(|v| {
                let mut vec = Vec::new();
                let mut var = min_var;
                while var <= max_var {
                    vec.push(*v.get(&var).unwrap_or(&false));
                    var = var.next();
                }
                vec
            })
            .collect();

        // Check lexicographic ordering
        for i in 0..vec_vals.len() - 1 {
            assert!(
                vec_vals[i] < vec_vals[i + 1],
                "Valuations should be in lexicographic order: {:?} < {:?}",
                vec_vals[i],
                vec_vals[i + 1]
            );
        }

        assert_eq!(vec_vals[0], vec![false, false, true]);
    }

    #[test]
    fn test_large_range_with_free_variables() {
        // Expected: Should return 2^10 = 1024 valuations (all combinations of free variables)
        let v5 = VariableId::from(5u32);
        let bdd = Bdd64::new_literal(v5.unchecked_into(), true);
        let min_var = VariableId::zero();
        let max_var = VariableId::from(10u32);
        let iter = ValuationsIterator::new(bdd, min_var, max_var);
        let valuations: Vec<_> = collect_valuations(iter);

        assert_eq!(valuations.len(), 1024);

        for valuation in &valuations {
            assert_eq!(valuation.get(&v5), Some(&true));
        }
    }

    #[test]
    fn test_comparison_with_standard_iterator() {
        // Expected: Both iterators should produce the same set of valuations
        let v0 = VariableId::zero();
        let v1 = VariableId::from(1u32);
        let v2 = VariableId::from(2u32);

        let bdd_split = Bdd::new_literal(v0, true)
            .and(&Bdd::new_literal(v1, true))
            .or(&Bdd::new_literal(v1, true).and(&Bdd::new_literal(v2, true)));

        let min_var = VariableId::zero();
        let max_var = VariableId::from(2u32);

        let iter_valuations: Vec<_> = bdd_split
            .iter_satisfying_valuations(min_var, max_var)
            .map(|v| {
                let mut vec = vec![false; 3];
                for (var, val) in v {
                    let idx = var.as_usize();
                    if idx < 3 {
                        vec[idx] = val;
                    }
                }
                vec
            })
            .collect();

        let standard_valuations: Vec<_> = bdd_split.satisfying_valuation(Some(max_var)).collect();

        let standard_vec: Vec<_> = standard_valuations
            .iter()
            .map(|v| {
                let mut vec = vec![false; 3];
                for (idx, val) in v.iter().enumerate() {
                    if idx < 3 {
                        vec[idx] = *val;
                    }
                }
                vec
            })
            .collect();

        assert_eq!(iter_valuations.len(), standard_vec.len());
        assert!(iter_valuations.len() > 0);

        for val in &iter_valuations {
            assert!(
                standard_vec.contains(val),
                "Iterator valuation {:?} should be in standard set",
                val
            );
        }

        for val in &standard_vec {
            assert!(
                iter_valuations.contains(val),
                "Standard valuation {:?} should be in iterator set",
                val
            );
        }
    }

    #[test]
    fn test_complex_formula_with_multiple_operations() {
        // Test a complex boolean formula: (v0 && v1) || (v2 && v3) || (v1 && v4)
        // This creates multiple paths and tests the iterator's ability to handle
        // complex BDD structures with various variable orderings.
        let v0 = VariableId::zero();
        let v1 = VariableId::from(1u32);
        let v2 = VariableId::from(2u32);
        let v3 = VariableId::from(3u32);
        let v4 = VariableId::from(4u32);

        let term1 = Bdd64::new_literal(v0.unchecked_into(), true)
            .and(&Bdd64::new_literal(v1.unchecked_into(), true))
            .unwrap();
        let term2 = Bdd64::new_literal(v2.unchecked_into(), true)
            .and(&Bdd64::new_literal(v3.unchecked_into(), true))
            .unwrap();
        let term3 = Bdd64::new_literal(v1.unchecked_into(), true)
            .and(&Bdd64::new_literal(v4.unchecked_into(), true))
            .unwrap();

        let bdd = term1.or(&term2).unwrap().or(&term3).unwrap();

        let min_var = VariableId::zero();
        let max_var = VariableId::from(4u32);
        let iter = ValuationsIterator::new(bdd, min_var, max_var);
        let valuations: Vec<_> = collect_valuations(iter);

        // Verify we get the expected number of satisfying valuations
        // The formula is satisfied when:
        // - (v0 && v1) is true, OR
        // - (v2 && v3) is true, OR
        // - (v1 && v4) is true
        assert_eq!(valuations.len(), 17);

        // Verify lexicographic ordering
        let vec_vals: Vec<_> = valuations
            .iter()
            .map(|v| {
                let mut vec = Vec::new();
                let mut var = min_var;
                while var <= max_var {
                    vec.push(*v.get(&var).unwrap_or(&false));
                    var = var.next();
                }
                vec
            })
            .collect();

        for i in 0..vec_vals.len() - 1 {
            assert!(
                vec_vals[i] < vec_vals[i + 1],
                "Valuations should be in lexicographic order: {:?} < {:?}",
                vec_vals[i],
                vec_vals[i + 1]
            );
        }

        // Verify that all valuations actually satisfy the formula
        for valuation in &valuations {
            let v0_val = *valuation.get(&v0).unwrap_or(&false);
            let v1_val = *valuation.get(&v1).unwrap_or(&false);
            let v2_val = *valuation.get(&v2).unwrap_or(&false);
            let v3_val = *valuation.get(&v3).unwrap_or(&false);
            let v4_val = *valuation.get(&v4).unwrap_or(&false);

            let term1_sat = v0_val && v1_val;
            let term2_sat = v2_val && v3_val;
            let term3_sat = v1_val && v4_val;

            assert!(
                term1_sat || term2_sat || term3_sat,
                "Valuation {:?} should satisfy the formula (v0&&v1) || (v2&&v3) || (v1&&v4)",
                valuation
            );
        }

        // Verify specific expected valuations are present
        let vec_vals_set: std::collections::HashSet<_> = vec_vals.iter().collect();

        // (v0=true, v1=true, ...) should be present (satisfies term1)
        assert!(
            vec_vals_set.contains(&vec![true, true, false, false, false]),
            "Should contain valuation where v0=true and v1=true"
        );

        // (..., v2=true, v3=true, ...) should be present (satisfies term2)
        assert!(
            vec_vals_set.contains(&vec![false, false, true, true, false]),
            "Should contain valuation where v2=true and v3=true"
        );

        // (..., v1=true, ..., v4=true) should be present (satisfies term3)
        assert!(
            vec_vals_set.contains(&vec![false, true, false, false, true]),
            "Should contain valuation where v1=true and v4=true"
        );
    }

    #[test]
    fn test_complex_lexicographic_ordering_with_many_valuations() {
        // Test a formula that creates many satisfying valuations and verify
        // strict lexicographic ordering across all of them.
        // Formula: (v0 || v1) && (v2 || v3) && (v4 || v5)
        // This creates 3^3 = 27 satisfying valuations (each pair can be 00, 01, or 10)
        let v0 = VariableId::zero();
        let v1 = VariableId::from(1u32);
        let v2 = VariableId::from(2u32);
        let v3 = VariableId::from(3u32);
        let v4 = VariableId::from(4u32);
        let v5 = VariableId::from(5u32);

        let clause1 = Bdd64::new_literal(v0.unchecked_into(), true)
            .or(&Bdd64::new_literal(v1.unchecked_into(), true))
            .unwrap();
        let clause2 = Bdd64::new_literal(v2.unchecked_into(), true)
            .or(&Bdd64::new_literal(v3.unchecked_into(), true))
            .unwrap();
        let clause3 = Bdd64::new_literal(v4.unchecked_into(), true)
            .or(&Bdd64::new_literal(v5.unchecked_into(), true))
            .unwrap();

        let bdd = clause1.and(&clause2).unwrap().and(&clause3).unwrap();

        let min_var = VariableId::zero();
        let max_var = VariableId::from(5u32);
        let iter = ValuationsIterator::new(bdd, min_var, max_var);
        let valuations: Vec<_> = collect_valuations(iter);

        // Should have 3^3 = 27 satisfying valuations
        // (each of the 3 clauses can be satisfied in 3 ways: v0, v1, or both)
        assert_eq!(
            valuations.len(),
            27,
            "Should have exactly 27 satisfying valuations"
        );

        // Convert to vectors for easier comparison
        let vec_vals: Vec<_> = valuations
            .iter()
            .map(|v| {
                let mut vec = Vec::new();
                let mut var = min_var;
                while var <= max_var {
                    vec.push(*v.get(&var).unwrap_or(&false));
                    var = var.next();
                }
                vec
            })
            .collect();

        // Verify strict lexicographic ordering
        for i in 0..vec_vals.len() - 1 {
            assert!(
                vec_vals[i] < vec_vals[i + 1],
                "Valuations should be in strict lexicographic order:\n  {:?} < {:?}\n  (at position {})",
                vec_vals[i],
                vec_vals[i + 1],
                i
            );
        }

        // Verify first valuation is lexicographically smallest
        // For (v0||v1) && (v2||v3) && (v4||v5), the smallest is [false,true, false,true, false,true]
        // (using the second variable in each clause to minimize the valuation)
        assert_eq!(
            vec_vals[0],
            vec![false, true, false, true, false, true],
            "First valuation should be lexicographically smallest"
        );

        // Verify last valuation is lexicographically largest (all true)
        assert_eq!(
            vec_vals[26],
            vec![true, true, true, true, true, true],
            "Last valuation should be lexicographically largest"
        );

        // Verify that all valuations satisfy the formula
        for (idx, valuation) in valuations.iter().enumerate() {
            let v0_val = *valuation.get(&v0).unwrap_or(&false);
            let v1_val = *valuation.get(&v1).unwrap_or(&false);
            let v2_val = *valuation.get(&v2).unwrap_or(&false);
            let v3_val = *valuation.get(&v3).unwrap_or(&false);
            let v4_val = *valuation.get(&v4).unwrap_or(&false);
            let v5_val = *valuation.get(&v5).unwrap_or(&false);

            let clause1_sat = v0_val || v1_val;
            let clause2_sat = v2_val || v3_val;
            let clause3_sat = v4_val || v5_val;

            assert!(
                clause1_sat && clause2_sat && clause3_sat,
                "Valuation {} ({:?}) should satisfy all clauses",
                idx,
                vec_vals[idx]
            );
        }

        // Verify no duplicates
        let unique_vals: std::collections::HashSet<_> = vec_vals.iter().collect();
        assert_eq!(
            unique_vals.len(),
            vec_vals.len(),
            "All valuations should be unique"
        );

        // Verify specific expected valuations are present
        let vec_vals_set: std::collections::HashSet<_> = vec_vals.iter().collect();

        // Check a few specific valuations
        assert!(
            vec_vals_set.contains(&vec![true, false, true, false, true, false]),
            "Should contain (v0, v2, v4)"
        );
        assert!(
            vec_vals_set.contains(&vec![false, true, false, true, false, true]),
            "Should contain (v1, v3, v5)"
        );
        assert!(
            vec_vals_set.contains(&vec![true, true, true, true, true, true]),
            "Should contain all variables true"
        );
    }
}
