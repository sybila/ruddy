use std::{fs::File, io::BufWriter};

use ruddy::{shared::BddManager, VariableId};

fn main() {
    let mut manager = BddManager::new();

    // Define variables v0, v1, v2.
    let v0 = VariableId::new(0);
    let v1 = VariableId::new(1);
    let v2 = VariableId::new(2);

    // Create BDDs representing the variables.
    let bdd_v0_true = manager.new_bdd_literal(v0, true);
    let bdd_v0_false = manager.new_bdd_literal(v0, false); // Represents !v0
    let bdd_v1_true = manager.new_bdd_literal(v1, true);
    let bdd_v2_true = manager.new_bdd_literal(v2, true);

    // Construct Term 1: `(v0 AND v1)``
    let term1 = manager.and(&bdd_v0_true, &bdd_v1_true);

    // Construct Term 2: `(!v0 AND v2)`
    let term2 = manager.and(&bdd_v0_false, &bdd_v2_true);

    // Combine terms to form `(v0 AND v1) OR (!v0 AND v2)`.
    let bdd = manager.or(&term1, &term2);

    // Export `bdd` to a .dot file for visualization
    let file = File::create("example.dot").unwrap();
    let mut buffer = BufWriter::new(file);
    manager.write_bdd_as_dot(&bdd, &mut buffer).unwrap();

    // Perform existential quantification: `exists v0 . bdd``
    // This should yield a BDD equivalent to `(v1 OR v2)``
    let bdd_after_exists = manager.exists(&bdd, &[v0]);

    // Print out the satisfying paths in the resulting BDD.
    // These valuations should only involve v1 and v2.
    for sat_path in manager.satisfying_paths(&bdd_after_exists, None) {
        println!("Satisfying path: {:?}", sat_path);
    }

    // For verification, let's directly construct the BDD for 'v1 OR v2'.
    let bdd_v1_or_v2_direct = manager.or(&bdd_v1_true, &bdd_v2_true);

    assert!(bdd_after_exists == bdd_v1_or_v2_direct);
}
