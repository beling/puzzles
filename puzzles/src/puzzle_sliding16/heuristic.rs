use crate::puzzle_sliding16::utils::MAX_BOARD_SIZE;
use crate::puzzle_sliding16::neighbors::cell_nr;
use crate::puzzle_sliding16::state::State;

/// Distances between all pair of cells.
pub type CellMetric = [[u8; MAX_BOARD_SIZE]; MAX_BOARD_SIZE];

/// Returns Manhattan metric for the board of given size.
pub fn manhattan_metric(cols: u8, rows: u8) -> CellMetric {
    let mut cell_distances = [[0u8; MAX_BOARD_SIZE]; MAX_BOARD_SIZE];
    for first_r in 0..rows {
        for first_c in 0..cols {
            let first_cell = cell_nr(cols, first_c, first_r) as usize;
            for second_r in 0..rows {
                let row_dist = if second_r >= first_r { second_r - first_r } else { first_r - second_r };
                //let col_dist = second_c - first_c;
                for second_c in first_c..cols {
                    let second_cell = cell_nr(cols, second_c, second_r) as usize;
                    let distance = row_dist + second_c - first_c;
                    cell_distances[first_cell][second_cell] = distance;
                    cell_distances[second_cell][first_cell] = distance;
                }
            }
        }
    }
    cell_distances
}

pub fn calc_manhattan_heuristic(metric: &CellMetric, state: State) -> u8 {
    state.iter().enumerate().map(|(i, t)| if t==0 { 0 } else { metric[i][t as usize] }).sum()
}
//#[inline(always)] fn update_manhattan_dist(to_update: &mut u8,  metric: &manhattan_metric) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manhattan_32() {
        // 0, 1, 2
        // 3, 4, 5
        let metric = manhattan_metric(3, 2);
        assert_eq!(metric[0][0], 0);
        assert_eq!(metric[0][1], 1);    assert_eq!(metric[1][0], 1);
        assert_eq!(metric[0][2], 2);    assert_eq!(metric[2][0], 2);
        assert_eq!(metric[0][3], 1);    assert_eq!(metric[3][0], 1);
        assert_eq!(metric[0][4], 2);    assert_eq!(metric[4][0], 2);
        assert_eq!(metric[0][5], 3);    assert_eq!(metric[5][0], 3);
        assert_eq!(metric[1][1], 0);
        assert_eq!(metric[1][2], 1);    assert_eq!(metric[2][1], 1);
        assert_eq!(metric[1][3], 2);    assert_eq!(metric[3][1], 2);
        assert_eq!(metric[1][4], 1);    assert_eq!(metric[4][1], 1);
        assert_eq!(metric[1][5], 2);    assert_eq!(metric[5][1], 2);
        assert_eq!(metric[2][2], 0);
        assert_eq!(metric[2][3], 3);    assert_eq!(metric[3][2], 3);
        assert_eq!(metric[2][4], 2);    assert_eq!(metric[4][2], 2);
        assert_eq!(metric[2][5], 1);    assert_eq!(metric[5][2], 1);
        assert_eq!(metric[3][3], 0);
        assert_eq!(metric[3][4], 1);    assert_eq!(metric[4][3], 1);
        assert_eq!(metric[3][5], 2);    assert_eq!(metric[5][3], 2);
        assert_eq!(metric[4][4], 0);
        assert_eq!(metric[4][5], 1);    assert_eq!(metric[5][4], 1);
        assert_eq!(metric[5][5], 0);

        let state: State = [2, 0, 5,  1, 4, 3].iter().cloned().collect();
        assert_eq!(calc_manhattan_heuristic(&metric, state), 7); // 2 /+ 0/ + 1 + 2 + 0 + 2
    }
}