use crate::puzzle_sliding16::utils::{MAX_BOARD_SIZE, DENIED};
use arrayvec::ArrayVec;

pub const LEFT: usize = 0;
pub const UP: usize  = 1;
pub const RIGHT: usize  = 2;
pub const DOWN: usize  = 3;

/// Stores indices of neighbors (or DENIED in the case of no neighbor) and is indexed by (in order): index of the cell and the direction.
pub type Neighbors = [[u8; 4]; MAX_BOARD_SIZE];

/// Returns tile number = index of cell with given (c, r) coordinates in the board with given number of cols.
#[inline(always)] pub fn cell_nr(cols: u8, c: u8, r: u8) -> u8 { r * cols + c }

/// Returns iterator over tile numbers = indices of cells with given (c, r) coordinates in the board with given number of cols.
pub fn cell_nrs(cols: u8, crs: impl IntoIterator<Item=(u8, u8)>) -> impl Iterator<Item=u8> {
    crs.into_iter().map(move |(c,r)| cell_nr(cols, c, r))
}

/// Constructs neighbors matrix for the board of the size `cols` x `rows`.
pub fn construct_neighbors(cols: u8, rows: u8) -> Neighbors {
    let mut neighbors = [[DENIED; 4]; MAX_BOARD_SIZE];
    for r in 0..rows {
        for c in 0..cols {
            let cell = &mut neighbors[cell_nr(cols, c, r) as usize];
            if c != 0 { cell[LEFT] = cell_nr(cols, c-1, r); }
            if r != 0 { cell[UP] = cell_nr(cols, c, r-1); }
            if c+1 != cols { cell[RIGHT] = cell_nr(cols, c+1, r); }
            if r+1 != rows { cell[DOWN] = cell_nr(cols, c, r+1); }
        }
    }
    neighbors
}

/// Returns neighbors (cell numbers) of the given `cell`.
pub fn neighbors_of(neighbors: &Neighbors, cell: u8) -> ArrayVec::<u8, 4> {
    let mut result = ArrayVec::<u8, 4>::new();
    for dir in 0..4 {
        let neighbor_pos = neighbors[cell as usize][dir];
        if neighbor_pos != DENIED {
            result.push(neighbor_pos);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::iter::FromIterator;

    #[test]
    fn test_cell_nrs() {
        assert_eq!(cell_nr(2, 0, 0), 0);
        assert_eq!(cell_nr(2, 1, 0), 1);
        assert_eq!(cell_nr(2, 0, 1), 2);
        assert_eq!(cell_nr(2, 1, 1), 3);
        assert_eq!(cell_nr(2, 0, 2), 4);
        assert_eq!(cell_nr(2, 1, 2), 5);
    }

    #[test]
    fn test_neighbors_3x2() {
        let neighbors = construct_neighbors(3, 2);
        assert_eq!(neighbors[cell_nr(3, 0, 0) as usize][LEFT], DENIED);
        assert_eq!(neighbors[cell_nr(3, 0, 0) as usize][UP], DENIED);
        assert_eq!(neighbors[cell_nr(3, 0, 0) as usize][RIGHT], cell_nr(3, 1, 0));
        assert_eq!(neighbors[cell_nr(3, 0, 0) as usize][DOWN], cell_nr(3, 0, 1));
        let neighbors_of_00 = neighbors_of(&neighbors, cell_nr(3, 0, 0));
        assert_eq!(neighbors_of_00.len(), 2);
        assert_eq!(HashSet::<u8>::from_iter(cell_nrs(3, [(1, 0), (0, 1)].iter().cloned())),
                    HashSet::from_iter(neighbors_of_00));

        assert_eq!(neighbors[cell_nr(3, 1, 1) as usize][LEFT], cell_nr(3, 0, 1));
        assert_eq!(neighbors[cell_nr(3, 1, 1) as usize][UP], cell_nr(3, 1, 0));
        assert_eq!(neighbors[cell_nr(3, 1, 1) as usize][RIGHT], cell_nr(3, 2, 1));
        assert_eq!(neighbors[cell_nr(3, 1, 1) as usize][DOWN], DENIED);
        assert_eq!(neighbors_of(&neighbors, cell_nr(3, 1, 1)).len(), 3);

        assert_eq!(neighbors[cell_nr(3, 2, 1) as usize][LEFT], cell_nr(3, 1, 1));
        assert_eq!(neighbors[cell_nr(3, 2, 1) as usize][UP], cell_nr(3, 2, 0));
        assert_eq!(neighbors[cell_nr(3, 2, 1) as usize][RIGHT], DENIED);
        assert_eq!(neighbors[cell_nr(3, 2, 1) as usize][DOWN], DENIED);
        assert_eq!(neighbors_of(&neighbors, cell_nr(3, 2, 1)).len(), 2);
    }
}