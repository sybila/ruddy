use ruddy::{split::Bdd, VariableId};

/// Make a `Vec<Vec<Bdd>>` of `Bdd`s representing whether there is a queen
/// placed on a particular square in a chessboard of size `n` by `n`.
///
/// More precisely, the `Bdd` at `(i, j)` is true if and only if there is a queen
/// placed on the square at row `i` and column `j`.
#[allow(clippy::cast_possible_truncation)]
fn bdds_squares(n: usize) -> Vec<Vec<Bdd>> {
    let mut squares = Vec::new();
    for i in 0..n {
        squares.push(Vec::new());
        for j in 0..n {
            let var = VariableId::new((i * n + j) as u32);
            squares[i].push(Bdd::new_literal(var, true));
        }
    }
    squares
}

/// Construct the `Bdd` representing a safe placement of a queen on the `i`th row
/// and `j`th column of the chessboard.
///
/// The resulting `Bdd` is true if and only if there is a queen placed on the square
/// at `(i, j)` and there are no other queens in the same row, column, or
/// diagonal.
#[allow(clippy::needless_range_loop)]
fn safe_queen_at(n: usize, i: usize, j: usize, squares: &[Vec<Bdd>]) -> Bdd {
    let mut safe_queen = squares[i][j].clone();

    // no queens in the same row
    for row in 0..n {
        if row != j {
            safe_queen = safe_queen.and(&squares[i][row].not());
        }
    }

    // no queens in the same column
    for col in 0..n {
        if col != i {
            safe_queen = safe_queen.and(&squares[col][j].not());
        }
    }

    // no queens in the main diagonal (top-left to bot-right)
    // r - c = i - j  =>  c = (r + j) - i
    for row in 0..n {
        if let Some(col) = (row + j).checked_sub(i) {
            if col < n && row != i {
                safe_queen = safe_queen.and(&squares[row][col].not());
            }
        }
    }

    // no queens in the anti diagonal (top-right to bot-left)
    // r + c = i + j  =>  c = (i + j) - r
    for row in 0..n {
        if let Some(col) = (i + j).checked_sub(row) {
            if col < n && row != i {
                safe_queen = safe_queen.and(&squares[row][col].not());
            }
        }
    }

    safe_queen
}

/// Construct the `Bdd` asserting that there is a (safe) queen in the `row`-th
/// row of the chessboard.
fn queen_in_row(n: usize, row: usize, squares: &[Vec<Bdd>]) -> Bdd {
    let mut queen_in_row = Bdd::new_false();
    for col in 0..n {
        let one_queen = safe_queen_at(n, row, col, squares);
        queen_in_row = queen_in_row.or(&one_queen);
    }
    queen_in_row
}

/// Construct the `Bdd` for the whole n-queens problem.
fn queens(n: usize) -> Bdd {
    let squares = bdds_squares(n);
    let mut result = Bdd::new_true();
    // iterate over all rows
    for row in 0..n {
        // assert that there is a (safe) queen in the row
        let in_row = queen_in_row(n, row, &squares);
        result = result.and(&in_row);
    }
    result
}

fn main() {
    let n = 8;
    let bdd = queens(n);

    assert_eq!(bdd.count_satisfying_valuations(None), 92.0);
}
