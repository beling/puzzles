use crate::puzzle_sliding16::utils::{DENIED, MAX_BOARD_SIZE, BITS_PER_CELL, BITS_PER_CELL_MASK32, BITS_PER_CELL_MASK64};
use crate::puzzle_sliding16::neighbors::Neighbors;
use arrayvec::ArrayVec;
use crate::puzzle_sliding16::state::State;
use std::iter::FusedIterator;

/// Manipulate patterns.
///
/// Pattern is a vector of positions of tiles important for pattern (which usually includes blank).
/// The tiles important for pattern are numbered from 0 (see nr_of_tile_in_pattern) and these numbers
/// are indices of the pattern vector.
/// Pattern is encoded in u32 and uses BITS_PER_CELL bits per important tile to store its position.
///
/// Note: set_blank_position, blank_position, and move_blank, moved_blank, neighbors suppose that 0-th important tile is blank.
#[derive(Clone, Copy)]
pub struct PatternManipulator {
    /// Convert: tile number (index) -> 4 * number of important tile or DENIED if the tile is not important
    index_of_tile_in_pattern: [u8; 16],

    /// Number of important tiles in pattern.
    pattern_len: u8
}

impl PatternManipulator {
    /// Returns pattern manipulator and the goal pattern for given list of numbers of important tiles.
    pub fn new(important_tiles: impl IntoIterator<Item=u8>) -> (Self, u32) {
        let mut goal_pattern = 0u32;
        let mut index_of_tile_in_pattern = [DENIED; MAX_BOARD_SIZE];
        let mut index_of_important = 0u8;
        for tile_nr in important_tiles {
            goal_pattern |= (tile_nr as u32) << index_of_important;
            index_of_tile_in_pattern[tile_nr as usize] = index_of_important;
            index_of_important += BITS_PER_CELL;
        }
        (Self { index_of_tile_in_pattern, pattern_len: index_of_important/BITS_PER_CELL }, goal_pattern)
    }

    /// Returns the position (field number) of tile with given number `tile_nr` in `pattern`.
    pub fn position_of(&self, pattern: u32, tile_nr: u8) -> u8 {
        let index = self.index_of_tile_in_pattern[tile_nr as usize];
        if index == DENIED {
            DENIED
        } else {
            ((pattern >> index) & BITS_PER_CELL_MASK32) as u8
        }
    }

    /// Returns the pattern which the given `state` matches to.
    pub fn pattern_for(&self, mut state: State) -> u32 {
        let mut pattern = 0;
        let mut position = 0;
        let mut seen_blank = false;
        while state.board != 0 {
            let tile_nr = (state.board & BITS_PER_CELL_MASK64) as u8;
            if tile_nr == 0 { seen_blank = true; }
            self.init_position(&mut pattern, tile_nr, position);
            state.board >>= BITS_PER_CELL;
            position += 1;
        }
        if !seen_blank { Self::init_blank_position(&mut pattern, position); }
        pattern
    }

    /// Modifies `pattern` by setting position (which must be `0` before this call) of tile with given number `tile_nr` to `new_position`.
    /// Does nothing if tile_nr is not important for pattern.
    pub fn init_position(&self, pattern: &mut u32, tile_nr: u8, new_position: u8) {
        let index = self.index_of_tile_in_pattern[tile_nr as usize];
        if index != DENIED {
            *pattern |= (new_position as u32) << index;
        }
    }

    /// Modifies `pattern` by setting position of tile with given number `tile_nr` to `new_position`.
    /// Does nothing if tile_nr is not important for pattern.
    pub fn set_position(&self, pattern: &mut u32, tile_nr: u8, new_position: u8) {
        let index = self.index_of_tile_in_pattern[tile_nr as usize];
        if index != DENIED {
            *pattern &= !(BITS_PER_CELL_MASK32 << index);
            *pattern |= (new_position as u32) << index;
        }
    }

    /// Same as `init_position(pattern, 0, new_blank_position)`, but faster.
    /// See also: `move_blank`
    #[inline(always)]
    pub fn init_blank_position(pattern: &mut u32, new_blank_position: u8) {
        *pattern |= new_blank_position as u32;
    }

    /// Same as `set_important_tile_position(pattern, 0, new_blank_position)`, but faster.
    /// See also: `move_blank`
    #[inline(always)]
    pub fn set_blank_position(pattern: &mut u32, new_blank_position: u8) {
        *pattern &= !BITS_PER_CELL_MASK32;
        *pattern |= new_blank_position as u32;
    }

    /// Returns position of blank in given `pattern`.
    #[inline(always)]
    pub fn blank_position(pattern: u32) -> u8 {
        (pattern & BITS_PER_CELL_MASK32) as u8
    }

    /// Sets position of important tile with given number `important_tile_nr` in `pattern` to `new_position`.
    #[inline(always)]
    pub fn set_important_tile_position(&self, pattern: &mut u32, important_tile_nr: u8, new_position: u8) {
        let index = important_tile_nr * BITS_PER_CELL;
        *pattern &= !(BITS_PER_CELL_MASK32 << index);
        *pattern |= (new_position as u32) << index;
    }

    /// Manipulates `pattern` by swapping blank with the tile that occupies `new_blank_position`.
    pub fn move_blank(&self, pattern: &mut u32, new_blank_position: u8) {
        let old_blank_pos = Self::blank_position(*pattern);
        Self::set_blank_position(pattern, new_blank_position);
        let mut to_process = *pattern;
        for tile_nr in 1..self.pattern_len {
            to_process >>= BITS_PER_CELL;
            if (to_process & BITS_PER_CELL_MASK32) as u8 == new_blank_position {
                // tile_nr has position new_blank_position and is just swapped with blank
                self.set_important_tile_position(pattern, tile_nr, old_blank_pos);
                break;
            }
        }
    }

    /// Returns a modified copy of `pattern` with swapped blank with the tile that occupies `new_blank_position`.
    #[inline(always)]
    pub fn moved_blank(&self, mut pattern: u32, new_blank_position: u8) -> u32 {
        self.move_blank(&mut pattern, new_blank_position); pattern
    }

    /// Returns a copy of `pattern` with swapped blank with `tile_at_new_blank_pos` tile that occupies `new_blank_position`.
    /// This method is faster than `moved_blank`, but requires `tile_at_new_blank_pos`.
    #[inline(always)]
    pub fn moved_blank_fast(&self, mut pattern: u32, tile_at_new_blank_pos: u8, new_blank_position: u8) -> u32 {
        let blank_pos = Self::blank_position(pattern);
        self.set_position(&mut pattern, tile_at_new_blank_pos, blank_pos);
        Self::set_blank_position(&mut pattern, new_blank_position);
        pattern
    }

    /// Returns all patterns that can be obtained from `pattern` by swapping blank with a neighbor tile.
    pub fn neighbors(&self, pattern: u32, neighbors: &Neighbors) -> ArrayVec::<u32, 4> {
        let mut result = ArrayVec::<u32, 4>::new();
        let blank = Self::blank_position(pattern) as usize;
        for dir in 0..4 {
            let neighbor_pos = neighbors[blank][dir];
            if neighbor_pos != DENIED {
                result.push(self.moved_blank(pattern, neighbor_pos));
            }
        }
        result
    }
}

/*pub trait Canonicalizer {
    fn canonical_form_of(pattern: u32) -> u32;
    #[inline(always)] fn is_always_identical() -> bool { false }
}

impl Canonicalizer for () {
    #[inline(always)] fn canonical_form_of(pattern: u32) -> u32 { pattern }
    #[inline(always)] fn is_always_identical() -> bool { true }
}

pub struct Fringle4x4Canonicalizer;

impl Canonicalizer for Fringle4x4Canonicalizer {
    fn canonical_form_of(pattern: u32) -> u32 {
        const MASK3: u32 = BITS_PER_CELL_MASK32 | (BITS_PER_CELL_MASK32<<BITS_PER_CELL) | (BITS_PER_CELL_MASK32<<(2*BITS_PER_CELL));
        const BITS_IN_MASK3: u8 = 3*BITS_PER_CELL;
        const RIGHT_BOUND: u32 = MASK3 << BITS_PER_CELL; // (without bottom-right corner)
        const BOTTOM_BOUND: u32 = MASK3 << (4*BITS_PER_CELL); // (without bottom-right corner)
        const REST: u32 = !(RIGHT_BOUND |BOTTOM_BOUND); // top-left and bottom right corners
        let r = !pattern;   // positions recalculate to mirrored ones - to jest źle, trzeba przeliczyć tablicą, np. pole 1->4, 2->8, albo przenumerować pola żeby działał np. jakiś xor, przekątna musi być odwzorowana na samą siebie
        pattern.min(
            ((r & RIGHT_BOUND) << BITS_IN_MASK3) |   // left bound moved to bottom bound
                ((r & BOTTOM_BOUND) >> BITS_IN_MASK3) |     // bottom bound moved to left bound
                (pattern & REST)  // corners does not change the position
        )
    }
}*/



/// Returns pattern database for given `max_pattern_distance` (which can be clamped to maximum distance exist) and `important_tiles`.
pub fn build_pattern_db(max_pattern_distance: &mut u8, important_tiles: impl IntoIterator<Item=u8>, neighbors: &Neighbors)
                        -> (Vec::<Vec::<u32>>, PatternManipulator)
{
    let (pattern_manipulator, goal) = PatternManipulator::new(important_tiles);
    let mut pattern_db = Vec::<Vec::<u32>>::new();  // i-th vector contains all patterns with distance to goal equal i
    pattern_db.push([goal].into());
    if *max_pattern_distance > 0 {
        let mut current = pattern_manipulator.neighbors(goal, &neighbors).to_vec();
        current.sort();
        while pattern_db.len() < *max_pattern_distance as usize {
            let prev = pattern_db.last().unwrap();
            let mut next = Vec::new();
            for pattern in current.iter() {
                for n in pattern_manipulator.neighbors(*pattern, &neighbors) {
                    if prev.binary_search(&n).is_err() && current.binary_search(&n).is_err() {
                        next.push(n);
                    }
                }
            }
            if next.is_empty() { *max_pattern_distance = pattern_db.len() as u8; break; }
            pattern_db.push(current);
            current = next;
            current.sort();
            current.dedup();
            current.shrink_to_fit();
        }
        pattern_db.push(current);
    }
    (pattern_db, pattern_manipulator)
}

/// Generator of a pattern database. It uses BFS starting form the goal state.
/// Generator can be convert to iterator over (pattern, distance to goal) pairs.
#[derive(Clone)]
pub struct PatternDBGenerator {
    /// States generated in the previous iteration.
    prev: Vec<u32>,
    /// Current states, generated in the last iteration.
    pub current: Vec<u32>,
    pattern_manipulator: PatternManipulator,
    neighbors: Neighbors
}

impl PatternDBGenerator {
    /// Constructs `PatternDBGenerator` that has the goal pattern in the `current` vector.
    pub fn new(important_tiles: impl IntoIterator<Item=u8>, neighbors: Neighbors) -> Self {
        let (pattern_manipulator, goal) = PatternManipulator::new(important_tiles);
        Self {
            prev: Vec::new(),
            current: [goal].into(),
            pattern_manipulator,
            neighbors
        }
    }

    /// Generates a set of patterns one move further away from the goal pattern than the patterns stored in `current`.
    /// Moves the content of `current` to `prev` and replaces it with newly generated patterns.
    ///
    /// The `low_mem` argument indicates whether the memory should be aggressively saved during the construction, at the cost of running time
    /// (if `low_mem` is `true`, the algorithm uses about 2x more time and upto 2x less memory).
    /// Usually it is better to pass `false`.
    pub fn advance(&mut self, low_mem: bool) {
        let mut next = if low_mem {
            let mut size = 0;
            for pattern in self.current.iter() {
                for n in self.pattern_manipulator.neighbors(*pattern, &self.neighbors) {
                    if self.prev.binary_search(&n).is_err() && self.current.binary_search(&n).is_err() {
                        size += 1;
                    }
                }
            }
            Vec::with_capacity(size)
        } else {
            Vec::new()
        };
        for pattern in self.current.iter() {
            for n in self.pattern_manipulator.neighbors(*pattern, &self.neighbors) {
                if self.prev.binary_search(&n).is_err() && self.current.binary_search(&n).is_err() {
                    next.push(n);
                }
            }
        }
        self.prev = std::mem::replace(&mut self.current, next); // delete old prev
        self.current.sort();
        self.current.dedup();
        self.current.shrink_to_fit();   // this can use extra memory
    }

    /// Returns `self` converted into iterator of (pattern, distance to goal) pairs.
    /// The iterator uses BFS starting form the goal state
    /// and finishing at patterns whose distance to the goal is `max_distance_to_goal`.
    ///
    /// The `low_mem` argument indicates whether the memory should be aggressively saved during the construction, at the cost of running time
    /// (if `low_mem` is `true`, the algorithm uses about 2x more time and upto 2x less memory).
    /// Usually it is better to pass `false`.
    pub fn into_iter_upto(self, max_distance_to_goal: u8, low_mem: bool) -> PatternDBIter {
        PatternDBIter {
            pattern_db: self,
            index_of_current: 0,
            distance_to_goal: 0,
            max_distance_to_goal,
            low_mem
        }
    }
}

impl IntoIterator for PatternDBGenerator {
    type Item = (u32, u8);
    type IntoIter = PatternDBIter;

    /// Returns `into_iter_upto(u8::MAX, false)`.
    fn into_iter(self) -> Self::IntoIter {
        self.into_iter_upto(u8::MAX, false)
    }
}

/// Iterator that generates a pattern database.
/// It uses BFS starting form the goal state and finishing at patterns whose distance to the goal is `max_distance_to_goal`.
pub struct PatternDBIter {
    pattern_db: PatternDBGenerator,
    index_of_current: usize,
    distance_to_goal: u8,
    max_distance_to_goal: u8,
    low_mem: bool
}

impl Iterator for PatternDBIter {
    type Item = (u32, u8);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(pattern) = self.pattern_db.current.get(self.index_of_current) {
                self.index_of_current += 1;
                return Some((*pattern, self.distance_to_goal));
            } else {
                if self.index_of_current == 0 || self.distance_to_goal == self.max_distance_to_goal {    // pattern_db.current must be empty
                    return None;
                }
                self.pattern_db.advance(self.low_mem);
                self.index_of_current = 0;
                self.distance_to_goal += 1;
            }
        }
    }
}

impl FusedIterator for PatternDBIter {}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle_sliding16::neighbors::construct_neighbors;
    use std::collections::HashSet;
    use std::iter::FromIterator;

    #[test]
    fn pattern_manipulator_3x2() {
        let (pm, goal) = PatternManipulator::new([0, 2,   3, 5].iter().cloned());
        assert_eq!(pm.pattern_len, 4);
        assert_eq!(pm.index_of_tile_in_pattern[0..6], [0, DENIED, 1*4, 2*4, DENIED, 3*4]);
        assert_eq!(PatternManipulator::blank_position(goal), 0);
        assert_eq!(pm.position_of(goal, 0), 0);
        assert_eq!(pm.position_of(goal, 1), DENIED);
        assert_eq!(pm.position_of(goal, 2), 2);
        assert_eq!(pm.position_of(goal, 3), 3);
        assert_eq!(pm.position_of(goal, 4), DENIED);
        assert_eq!(pm.position_of(goal, 5), 5);
        assert_eq!(pm.pattern_for(State::goal(3*2)), goal);
        let mut s = goal;
        pm.set_position(&mut s, 5, 4);
        assert_eq!(pm.position_of(s, 0), 0);
        assert_eq!(pm.position_of(s, 1), DENIED);
        assert_eq!(pm.position_of(s, 2), 2);
        assert_eq!(pm.position_of(s, 3), 3);
        assert_eq!(pm.position_of(s, 4), DENIED);
        assert_eq!(pm.position_of(s, 5), 4);     // <
        PatternManipulator::set_blank_position(&mut s, 1);
        assert_eq!(pm.position_of(s, 0), 1);    // <
        assert_eq!(pm.position_of(s, 1), DENIED);
        assert_eq!(pm.position_of(s, 2), 2);
        assert_eq!(pm.position_of(s, 3), 3);
        assert_eq!(pm.position_of(s, 4), DENIED);
        assert_eq!(pm.position_of(s, 5), 4);
        pm.move_blank(&mut s, 2);
        assert_eq!(pm.position_of(s, 0), 2);    // <
        assert_eq!(pm.position_of(s, 1), DENIED);
        assert_eq!(pm.position_of(s, 2), 1);    // <
        assert_eq!(pm.position_of(s, 3), 3);
        assert_eq!(pm.position_of(s, 4), DENIED);
        assert_eq!(pm.position_of(s, 5), 4);
        let sl = pm.moved_blank_fast(s, 2, 1);
        assert_eq!(pm.position_of(sl, 0), 1);    // <
        assert_eq!(pm.position_of(sl, 1), DENIED);
        assert_eq!(pm.position_of(sl, 2), 2);    // <
        assert_eq!(pm.position_of(sl, 3), 3);
        assert_eq!(pm.position_of(sl, 4), DENIED);
        assert_eq!(pm.position_of(sl, 5), 4);
        let sd = pm.moved_blank_fast(s, 4, 5);
        assert_eq!(pm.position_of(sd, 0), 5);    // <
        assert_eq!(pm.position_of(sd, 1), DENIED);
        assert_eq!(pm.position_of(sd, 2), 1);
        assert_eq!(pm.position_of(sd, 3), 3);
        assert_eq!(pm.position_of(sd, 4), DENIED);
        assert_eq!(pm.position_of(sd, 5), 4);
        let sd = pm.moved_blank(s, 5);
        assert_eq!(pm.position_of(sd, 0), 5);    // <
        assert_eq!(pm.position_of(sd, 1), DENIED);
        assert_eq!(pm.position_of(sd, 2), 1);
        assert_eq!(pm.position_of(sd, 3), 3);
        assert_eq!(pm.position_of(sd, 4), DENIED);
        assert_eq!(pm.position_of(sd, 5), 4);
        let n = pm.neighbors(s, &construct_neighbors(3, 2));
        assert_eq!(n.len(), 2);
        assert_eq!(HashSet::<u32>::from_iter(n.iter().cloned()), HashSet::from_iter([sl, sd].iter().cloned()))
    }

    #[test]
    fn build_pattern_db_3x2_only_blank_full_lazy() {
        let neighbors = construct_neighbors(3, 2);
        let mut gen = PatternDBGenerator::new(0..=0, neighbors);
        assert_eq!(gen.current, &[0]);
        gen.advance(false);
        assert_eq!(gen.current, [1, 3]);
        gen.advance(true);
        assert_eq!(gen.current, [2, 4]);
        gen.advance(false);
        assert_eq!(gen.current, [5]);
        gen.advance(true);
        assert_eq!(gen.current, []);
    }

    #[test]
    fn build_pattern_db_3x2_only_blank_full_iterator() {
        let neighbors = construct_neighbors(3, 2);
        let mut gen = PatternDBGenerator::new(0..=0, neighbors);
        assert_eq!(gen.into_iter().collect::<Vec<_>>(),
                   [(0, 0), (1, 1), (3, 1), (2, 2), (4, 2), (5, 3)]);
    }

    #[test]
    fn build_pattern_db_3x2_only_blank_full_iterator_upto1() {
        let neighbors = construct_neighbors(3, 2);
        let mut gen = PatternDBGenerator::new(0..=0, neighbors);
        assert_eq!(gen.into_iter_upto(1, false).collect::<Vec<_>>(), [(0, 0), (1, 1), (3, 1)]);
    }

    #[test]
    fn build_pattern_db_3x2_only_blank_full() {
        let neighbors = &construct_neighbors(3, 2);
        let mut max_dist = u8::MAX;
        let (db, _) = build_pattern_db(&mut max_dist, 0..=0, neighbors);
        assert_eq!(max_dist, 3);
        assert_eq!(db.len(), 4);
        assert_eq!(db[0], [0]);
        assert_eq!(db[1], [1, 3]);
        assert_eq!(db[2], [2, 4]);
        assert_eq!(db[3], [5]);
    }

    #[test]
    fn build_pattern_db_3x2_only_blank_max1() {
        let neighbors = &construct_neighbors(3, 2);
        let mut max_dist = 1;
        let (db, _) = build_pattern_db(&mut max_dist, 0..=0, neighbors);
        assert_eq!(max_dist, 1);
        assert_eq!(db.len(), 2);
        assert_eq!(db[0], [0]);
        assert_eq!(db[1], [1, 3]);
    }

}