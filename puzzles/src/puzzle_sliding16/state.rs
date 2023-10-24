use crate::puzzle_sliding16::utils::{BITS_PER_CELL, BITS_PER_CELL_MASK64};
use std::iter::{FromIterator, FusedIterator};

/// Board state.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct State {
    /// Indexed by board indices, gives tiles numbers that occupy given board cell.
    pub board: u64,
    //board: [u8; MAX_BOARD_SIZE],

    // Index of blank (to speed up operations).
    //blank_position: u8
}

impl FromIterator<u8> for State {
    fn from_iter<T: IntoIterator<Item=u8>>(tiles: T) -> Self {
        let mut board = 0u64;
        let mut index = 0;
        for t in tiles {
            board |= (t as u64) << index;
            index += BITS_PER_CELL;
        }
        Self { board }
    }
}

/// Translate `tile` number from bottom-right format that assumes blank to be in the bottom-right corner of the goal state.
#[inline] pub fn from_bottom_right_format(tile: u8, board_size: u8) -> u8 {
    if tile == 0 { 0 } else {board_size-tile}
}

impl State {

    /// Constructs goal state for the board with board_size cells.
    pub fn goal(board_size: u8) -> Self {
        (0..board_size).collect()
    }

    /// Swap blank with the tile that occupy new_blank_position.
    /// Returns number of this tile.
    pub fn move_blank(&mut self, current_blank_position: u8, new_blank_position: u8) -> u8 {
        let new_blank_index = new_blank_position * BITS_PER_CELL;
        let result = (self.board >> new_blank_index) & BITS_PER_CELL_MASK64;
        self.board &= !(BITS_PER_CELL_MASK64 << new_blank_index);       // clear cell which should be blank
        self.board |= result << (current_blank_position * BITS_PER_CELL);    // set old blank cell to result
        result as u8
    }
    /*fn move_blank(&mut self, new_blank_position: u8) -> u8 {
        let new_blank_index = new_blank_position * BITS_PER_CELL;
        let result = (self.board >> new_blank_index) & BITS_PER_CELL_MASK64;
        self.board &= !(BITS_PER_CELL_MASK64 << new_blank_index);       // clear cell which should be blank
        self.board |= result << (self.blank_position*BITS_PER_CELL);    // set old blank cell to result
        self.blank_position = new_blank_position;
        result as u8
    }*/

    /// Constructs `State` from data given in format that assumes blank to be in the bottom-right corner of the goal state.
    pub fn from_bottom_right_format(tiles: &[u8]) -> Self {
        let s = tiles.len() as u8;
        tiles.iter().rev().map(|t| from_bottom_right_format(*t, s)).collect()
    }

    // Tile at position.
    pub fn tile_at(&self, position: u8) -> u8 {
        ((self.board >> (position * BITS_PER_CELL)) & BITS_PER_CELL_MASK64) as u8
    }

    /// Returns number of cells in board.
    pub fn board_size(&self) -> u8 {
        let mut c = self.board;
        let mut seen_blank = false;
        let mut result = 0;
        while c != 0 {
            result += 1;
            if (c & BITS_PER_CELL_MASK64) == 0 { seen_blank = true; }
            c >>= BITS_PER_CELL;
        }
        if !seen_blank { result += 1; }
        result
    }

    pub fn print(&self, cols: u8) {
        let mut to_print = self.board;
        let mut c = 0;
        loop {
            let cell = to_print & BITS_PER_CELL_MASK64;
            if cell==0 { print!(" "); } else { print!("{}", cell) };
            to_print >>= BITS_PER_CELL;
            if to_print == 0 {
                println!();
                break;
            }
            c += 1;
            if c == cols {
                println!();
                c = 0;
            } else { print!(" "); }
        }
    }

    #[inline] pub fn iter(&self) -> TilesIterator {
        TilesIterator::new(self.board)
    }
}

#[derive(Copy, Clone)]
pub struct TilesIterator {
    rest: u64,
    seen_blank: bool
}

impl TilesIterator {
    #[inline] pub fn new(board: u64) -> Self {
        Self { rest: board, seen_blank: false }
    }
}

impl Iterator for TilesIterator {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.rest == 0 {
            if self.seen_blank {
                None
            } else {
                self.seen_blank = true;
                Some(0)
            }
        } else {
            let result = (self.rest & BITS_PER_CELL_MASK64) as u8;
            if result == 0 { self.seen_blank = true; }
            self.rest >>= BITS_PER_CELL;
            Some(result)
        }
    }
}

impl FusedIterator for TilesIterator {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_and_move_blank_32() {
        let tiles = [2, 0, 5,  1, 4, 3];
        let mut state: State = tiles.iter().cloned().collect();
        assert_eq!(state.board_size(), 6);
        assert_eq!(state.tile_at(0), 2);
        assert_eq!(state.tile_at(1), 0);
        assert_eq!(state.tile_at(2), 5);
        assert_eq!(state.tile_at(3), 1);
        assert_eq!(state.tile_at(4), 4);
        assert_eq!(state.tile_at(5), 3);
        assert_eq!(state.iter().collect::<Vec<_>>(), &tiles);
        state.move_blank(1, 4);
        assert_eq!(state.board_size(), 6);
        assert_eq!(state.tile_at(0), 2);
        assert_eq!(state.tile_at(1), 4);    //<
        assert_eq!(state.tile_at(2), 5);
        assert_eq!(state.tile_at(3), 1);
        assert_eq!(state.tile_at(4), 0);    //<
        assert_eq!(state.tile_at(5), 3);
        assert_eq!(state.iter().collect::<Vec<_>>(), &[2, 4, 5, 1, 0, 3]);
        state.move_blank(4, 5);
        assert_eq!(state.board_size(), 6);
        assert_eq!(state.tile_at(0), 2);
        assert_eq!(state.tile_at(1), 4);
        assert_eq!(state.tile_at(2), 5);
        assert_eq!(state.tile_at(3), 1);
        assert_eq!(state.tile_at(4), 3);    //<
        assert_eq!(state.tile_at(5), 0);    //<
        assert_eq!(state.iter().collect::<Vec<_>>(), &[2, 4, 5, 1, 3, 0]);
    }

    #[test]
    fn test_from_and_move_blank_33() {
        let mut state: State = [4, 7, 0,   2, 3, 6,   8, 1, 5].iter().cloned().collect();
        assert_eq!(state.board_size(), 9);
        assert_eq!(state.tile_at(0), 4);
        assert_eq!(state.tile_at(1), 7);
        assert_eq!(state.tile_at(2), 0);
        assert_eq!(state.tile_at(3), 2);
        assert_eq!(state.tile_at(4), 3);
        assert_eq!(state.tile_at(5), 6);
        assert_eq!(state.tile_at(6), 8);
        assert_eq!(state.tile_at(7), 1);
        assert_eq!(state.tile_at(8), 5);
        state.move_blank(2, 5);
        assert_eq!(state.board_size(), 9);
        assert_eq!(state.tile_at(0), 4);
        assert_eq!(state.tile_at(1), 7);
        assert_eq!(state.tile_at(2), 6);    //<
        assert_eq!(state.tile_at(3), 2);
        assert_eq!(state.tile_at(4), 3);
        assert_eq!(state.tile_at(5), 0);    //<
        assert_eq!(state.tile_at(6), 8);
        assert_eq!(state.tile_at(7), 1);
        assert_eq!(state.tile_at(8), 5);
        state.move_blank(5, 4);
        assert_eq!(state.board_size(), 9);
        assert_eq!(state.tile_at(0), 4);
        assert_eq!(state.tile_at(1), 7);
        assert_eq!(state.tile_at(2), 6);
        assert_eq!(state.tile_at(3), 2);
        assert_eq!(state.tile_at(4), 0);    //<
        assert_eq!(state.tile_at(5), 3);    //<
        assert_eq!(state.tile_at(6), 8);
        assert_eq!(state.tile_at(7), 1);
        assert_eq!(state.tile_at(8), 5);
    }

    #[test]
    fn test_goal_33() {
        let state = State::goal(3*3);
        assert_eq!(state.board_size(), 9);
        assert_eq!(state.tile_at(0), 0);
        assert_eq!(state.tile_at(1), 1);
        assert_eq!(state.tile_at(2), 2);
        assert_eq!(state.tile_at(3), 3);
        assert_eq!(state.tile_at(4), 4);
        assert_eq!(state.tile_at(5), 5);
        assert_eq!(state.tile_at(6), 6);
        assert_eq!(state.tile_at(7), 7);
        assert_eq!(state.tile_at(8), 8);
    }

    #[test]
    fn test_goal_43() {
        let state = State::goal(4*3);
        assert_eq!(state.board_size(), 12);
        assert_eq!(state.tile_at(0), 0);
        assert_eq!(state.tile_at(1), 1);
        assert_eq!(state.tile_at(2), 2);
        assert_eq!(state.tile_at(3), 3);
        assert_eq!(state.tile_at(4), 4);
        assert_eq!(state.tile_at(5), 5);
        assert_eq!(state.tile_at(6), 6);
        assert_eq!(state.tile_at(7), 7);
        assert_eq!(state.tile_at(8), 8);
        assert_eq!(state.tile_at(9), 9);
        assert_eq!(state.tile_at(10), 10);
        assert_eq!(state.tile_at(11), 11);
    }

    #[test]
    fn test_from_bottom_right_format() {
        assert_eq!(from_bottom_right_format(0, 4), 0);
        assert_eq!(from_bottom_right_format(1, 4), 3);
    }
}