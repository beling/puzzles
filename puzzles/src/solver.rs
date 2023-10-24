use crate::puzzle_sliding16::utils::MAX_BOARD_SIZE;
use crate::puzzle_sliding16::neighbors::{Neighbors, construct_neighbors, neighbors_of, cell_nrs};
use crate::puzzle_sliding16::pattern::{PatternManipulator, build_pattern_db};
use crate::puzzle_sliding16::state::State;
use arrayvec::ArrayVec;
use crate::pattern_db::{PatternDBManager};
use crate::stats::SearchStatsCollector;
use crate::puzzle_sliding16::heuristic::{CellMetric, manhattan_metric, calc_manhattan_heuristic};

pub trait ExtraHeuristic {
    const IS_FAKE: bool = false;
    type Value: Copy+Clone;
    fn new(cols: u8, rows: u8) -> Self;
    fn value_for_state(&self, state: State) -> Self::Value;
    fn value_for_state_u8(&self, state: State) -> u8 { Self::to_u8(self.value_for_state(state)) }
    fn update_value(&self, old_value: Self::Value, tile: u8, from: u8, to: u8) -> Self::Value;
    fn to_u8(extra: Self::Value) -> u8;
    fn max(from_db: u8, extra: Self::Value) -> u8 { from_db.max(Self::to_u8(extra)) }
}

impl ExtraHeuristic for () {
    const IS_FAKE: bool = true;
    type Value = ();
    #[inline(always)] fn new(_cols: u8, _rows: u8) -> Self { () }
    #[inline(always)] fn value_for_state(&self, _state: State) -> Self::Value { () }
    #[inline(always)] fn update_value(&self, _old_value: Self::Value, _tile: u8, _from: u8, _to: u8) -> Self::Value { () }
    #[inline(always)] fn to_u8(_extra: Self::Value) -> u8 { 0 }
    #[inline(always)] fn max(from_db: u8, _extra: Self::Value) -> u8 { from_db }
}

impl ExtraHeuristic for CellMetric {
    type Value = u8;

    #[inline(always)] fn new(cols: u8, rows: u8) -> Self { manhattan_metric(cols, rows) }

    #[inline(always)] fn value_for_state(&self, state: State) -> Self::Value {
        calc_manhattan_heuristic(self, state)
    }

    #[inline(always)] fn update_value(&self, old_value: Self::Value, tile: u8, from: u8, to: u8) -> Self::Value {
        old_value
            - self[tile as usize][from as usize]
            + self[tile as usize][to as usize]
    }

    #[inline(always)] fn to_u8(extra: Self::Value) -> u8 { extra }

    #[inline(always)] fn max(from_db: u8, extra: Self::Value) -> u8 {
        from_db.max(extra)
    }
}

#[derive(Copy, Clone)]
struct SearchState<ExtraHeuristicValue: Copy+Clone> {
    state: State,
    pattern: u32,
    extra_heuristic: ExtraHeuristicValue
}

impl<ExtraHeuristicValue: Copy+Clone> From<SearchStateWithEval<ExtraHeuristicValue>> for SearchState<ExtraHeuristicValue> {
    #[inline(always)] fn from(src: SearchStateWithEval<ExtraHeuristicValue>) -> Self {
        src.search_state
    }
}

#[derive(Copy, Clone)]
struct SearchStateWithEval<ExtraHeuristicValue: Copy+Clone> {
    search_state: SearchState<ExtraHeuristicValue>,
    heuristic_value: u8
}

pub struct SlidingPuzzleSolver<PDBM: PatternDBManager, EH: ExtraHeuristic> {
    /// Stores indices of neighbors (or DENIED in the case of no neighbor) and is indexed by (in order): index of the cell and the direction.
    pub neighbors: Neighbors,

    pub pattern_manipulator: PatternManipulator,

    pub goal_state: State,

    //board_size: u8

    pub extra_heuristic: EH,

    pub pattern_db: PDBM::PatternDB,

    pub max_pattern_distance_plus_one: u8
}

impl<PDBM: PatternDBManager, EH: ExtraHeuristic> SlidingPuzzleSolver<PDBM, EH> {

    pub fn new_indices(cols: u8, rows: u8, important_tiles: impl IntoIterator<Item=u8>, mut max_pattern_distance: u8, pdbm: PDBM) -> Self {
        let board_size = cols as usize * rows as usize;
        assert!(board_size <= MAX_BOARD_SIZE);
        let neighbors = construct_neighbors(cols, rows);
        let (pattern_db, pattern_manipulator) = build_pattern_db(&mut max_pattern_distance, important_tiles, &neighbors);
        Self { neighbors, pattern_manipulator,
            goal_state: State::goal(board_size as u8),
            extra_heuristic: EH::new(cols, rows),
            pattern_db: pdbm.construct(pattern_db), max_pattern_distance_plus_one: max_pattern_distance + 1 }
    }

    #[inline(always)] pub fn new_coords(cols: u8, rows: u8, important_tiles: impl IntoIterator<Item=(u8, u8)>, max_pattern_distance: u8, pdbm: PDBM) -> Self {
        Self::new_indices(cols, rows, cell_nrs(cols, important_tiles), max_pattern_distance, pdbm)
    }

    /// Returns neighbors (cell numbers) of the given `cell`.
    #[inline(always)] pub fn neighbors_of_cell(&self, cell: u8) -> ArrayVec<u8, 4> {
        neighbors_of(&self.neighbors, cell)
    }

    /// Returns subset of this neighbors of `search_state` for which `skip` returns `false`, sorted by heuristic values.
    fn neighbors_of_state<SkipF/*, SkipManhattanF*/>(&self, search_state: &SearchState<EH::Value>, skip: SkipF/*, mut skip_manhattan: SkipManhattanF*/) -> ArrayVec<SearchStateWithEval<EH::Value>, 4>
    where SkipF: Fn(&State) -> bool//, SkipManhattanF: FnMut(u8) -> bool,
    {
        let mut result = ArrayVec::<SearchStateWithEval<EH::Value>, 4>::new();
        let blank_pos = PatternManipulator::blank_position(search_state.pattern);
        for neighbor_cell in self.neighbors_of_cell(blank_pos) { // neighbor is a cell number
            let mut neighbor_state = search_state.state;
            let neighbor_tile = neighbor_state.move_blank(blank_pos, neighbor_cell);
            if skip(&neighbor_state) { continue; }
            // neighbor_tile was at neighbor cell and now is at blank_pos cell
            let manhattan_heuristic = self.extra_heuristic.update_value(search_state.extra_heuristic, neighbor_tile, neighbor_cell, blank_pos);
            //if skip_manhattan(manhattan_heuristic) { continue; }
            let neighbor_pattern = self.pattern_manipulator.moved_blank_fast(search_state.pattern, neighbor_tile, neighbor_cell);
            result.push(SearchStateWithEval{
                search_state: SearchState {
                    state: neighbor_state,
                    pattern: neighbor_pattern,
                    extra_heuristic: manhattan_heuristic
                },
                heuristic_value: EH::max(self.heuristic_value_from_db(
                    //self.pattern_manipulator.pattern_for(neighbor_state)
                    neighbor_pattern
                ), manhattan_heuristic)
            });
        }
        result.sort_by_key(|neighbor| neighbor.heuristic_value);
        result
    }

    /// Implementation of DFS with limited depth that is a part of IDA*.
    ///
    /// # Arguments
    /// * `state` - position to evaluate,
    /// * `path` - path from the initial position (included) to the `state` (excluded), should be empty if `state` is the initial state
    /// * `depth_limit` - limit to the number of moves from initial to the goal state (length of path)
    /// * `stats` - collects search statistics
    ///
    /// # Return
    /// Let `r` be the returned value, and `m` be the number of moves needed to achieve the goal state from the initial state. Then:
    /// * if `r == depth_limit` then `m == r`;
    /// * if `r` is in range (`depth_limit`, `u16::MAX`) then `m >= r`;
    /// * if `r==u16::MAX` then the goal state cannot be achieved.
    /// Note that the algorithm assumes that `m >= depth_limit`.
    fn search_rec(&self, state_with_eval: SearchStateWithEval<EH::Value>, path: &mut Vec/*HashSet*/<State>, depth_limit: u16, stats: &mut impl SearchStatsCollector) -> u16 {
        let depth = path.len() as u16;
        if state_with_eval.search_state.state == self.goal_state {
            stats.leaf();
            assert_eq!(depth, depth_limit);
            return depth;
        }
        // Note about "max(1)": we know that state_with_eval is not the goal state, so we can correct potential heuristic_value=0
        let mut min_depth = depth + state_with_eval.heuristic_value.max(1) as u16;
        if min_depth > depth_limit {
            if !stats.leaf() { return u16::MAX; };
            return min_depth;
        }
        stats.internal();
        min_depth = u16::MAX;
        let neighbors = self.neighbors_of_state(
            &state_with_eval.into(),
            |n| path.contains(n)
        );
        path.push(state_with_eval.search_state.state);
        //path.insert(state.state);
        for neighbor in neighbors {
            let v = self.search_rec(neighbor, path, depth_limit, stats);
            if v == depth_limit { return depth_limit; } // the goal is found!
            if v < min_depth { min_depth = v; }
        }
        path.pop();
        //path.remove(&state.state);
        min_depth
    }

    #[inline(always)] pub fn heuristic_value_from_db(&self, pattern: u32) -> u8 {
        <PDBM as PatternDBManager>::heuristic_value(&self.pattern_db, pattern, self.max_pattern_distance_plus_one)
    }

    fn search_iter(&self, first_search_state: &SearchState<EH::Value>, depth_limit: u16, stats: &mut impl SearchStatsCollector) -> u16 {
        let mut min_depth = u16::MAX;
        let mut states = Vec::with_capacity(256);
        let mut levels = Vec::with_capacity(128);
        states.push(*first_search_state);
        levels.push((0usize, 1usize));
        while let Some((ref mut current, end)) = levels.last_mut() {
            if current == end {
                levels.pop();
                if let Some((_, prev_end)) = levels.last() {
                    states.resize_with(*prev_end, || unreachable!());
                    continue;
                } else {
                    break;
                }
            }
            let search_state = states[*current];
            *current += 1;
            stats.internal();
            let level_begin = states.len();
            let depth = levels.len(); // depth of neighbors
            let neighbors = self.neighbors_of_state(
                &search_state,
                |n| states.iter().any(|stacked| stacked.state == *n)
            );
            for neighbor in neighbors {
                if neighbor.search_state.state == self.goal_state {
                    stats.leaf();   // solution found at given depth
                    assert_eq!(depth as u16, depth_limit);
                    return depth as u16;
                }
                // Note about "max(1)": we know that neighbor is not the goal state, so we can correct potential heuristic_value=0
                let neighbor_min_depth = depth as u16 + neighbor.heuristic_value.max(1) as u16;
                if neighbor_min_depth > depth_limit {
                    if !stats.leaf() { return u16::MAX; }
                    if neighbor_min_depth < min_depth { min_depth = neighbor_min_depth};
                } else {
                    states.push(neighbor.into());
                }
            }
            levels.push((level_begin, states.len()));
        }
        min_depth
    }

    /*fn search(&self, first_search_state: &SearchStateWithEval, depth_limit: u16, stats: &mut impl SearchStatsCollector) -> u16 {
        let mut min_depth = u16::MAX;
        let mut stack = Vec::with_capacity(128);
        stack.push((*first_search_state, 0u8));
        while let Some((state, mut depth)) = stack.pop() {
            if !stats.visit() { return u16::MAX }
            if state.state == self.goal_state {
                assert_eq!(depth as u16, depth_limit);
                return depth_limit;
            }
            // Note about "max(1)": we know that neighbor is not the goal state, so we can correct potential heuristic_value=0 from lossy database.
            let mut state_min_depth = depth as u16 + state.heuristic_value.max(1) as u16;
            if state_min_depth > depth_limit {
                if state_min_depth < min_depth {min_depth = state_min_depth};
                continue;
            }
            depth += 1; // depth of neighbors
            let mut neighbors = self.neighbors_of_state(&state.into());
            neighbors.sort_by_key(|neighbor| neighbor.heuristic_value);
            //history_range = stack.len().saturating_sub(16)..stack.len();
            for neighbor in neighbors {
                if stack.iter().any(|(hs, _)| hs.state == neighbor.state) {
                    stats.visit();  // TODO czy to powinno być?
                    continue;
                }
                // TODO sprawdzić czy neighbor wraca, TT?
                /*if stack[stack.len().saturating_sub(16)..].iter().any(|(hs, _)| hs.state == neighbor.state) {
                    stats.visit();  // TODO czy to powinno być?
                    continue;
                }*/
                stack.push((neighbor.into(), depth));
            }
        }
        min_depth
    }*/

    /// Returns number of moves to solve the state or u16::MAX if it is not solvable.
    /// Collect statistics during search.
    pub fn moves_to_solve_stats(&self, state: State, stats: &mut impl SearchStatsCollector) -> u16 {
        if state == self.goal_state { return 0; }
        let search_state = {
            let pattern = self.pattern_manipulator.pattern_for(state);
            let manhattan_heuristic = self.extra_heuristic.value_for_state(state);
            SearchStateWithEval {
                search_state: SearchState {
                    state, pattern,
                    extra_heuristic: manhattan_heuristic
                },
                heuristic_value: EH::max(self.heuristic_value_from_db(pattern), manhattan_heuristic)
            }
        };
        let mut depth_limit = search_state.heuristic_value as u16;
        loop {
            //println!("{}", depth_limit);
            let v = self.search_rec(search_state, &mut Vec/*HashSet*/::with_capacity(128), depth_limit, stats);
            //let v = self.search_iter(&search_state.into(), depth_limit, stats);
            if v == depth_limit || v == u16::MAX {
                /*for s in path {
                    s.state.print(3);
                    println!();
                }*/
                return v;
            }
            depth_limit = v;
        }
    }

    /// Returns number of moves to solve the state or u32::MAX if it is not solvable.
    #[inline] pub fn moves_to_solve(&self, state: State) -> u16 {
        self.moves_to_solve_stats(state, &mut ())
    }
}

impl<PDBM: PatternDBManager> SlidingPuzzleSolver<PDBM, CellMetric> {
    fn search_rec_manhattan_first(&self, search_state: SearchState<u8>, path: &mut Vec/*HashSet*/<State>, depth_limit: u16, stats: &mut impl SearchStatsCollector) -> u16 {
        stats.internal();
        let mut min_depth = u16::MAX;
        let neighbor_depth = path.len() as u16 + 1;
        let mut neighbors = ArrayVec::<SearchStateWithEval<u8>, 4>::new();
        {
            let blank_pos = PatternManipulator::blank_position(search_state.pattern);
            for neighbor_cell in self.neighbors_of_cell(blank_pos) { // neighbor is a cell number
                let mut neighbor_state = search_state.state;
                let neighbor_tile = neighbor_state.move_blank(blank_pos, neighbor_cell);
                if path.contains(&neighbor_state) { continue; }
                if neighbor_state == self.goal_state {
                    stats.leaf();
                    assert_eq!(neighbor_depth, depth_limit);
                    return neighbor_depth;
                }
                // neighbor_tile was at neighbor cell and now is at blank_pos cell
                let manhattan_heuristic = self.extra_heuristic.update_value(search_state.extra_heuristic, neighbor_tile, neighbor_cell, blank_pos);
                let lower_bound = neighbor_depth + manhattan_heuristic as u16;
                if lower_bound > depth_limit {
                    if !stats.leaf() { return u16::MAX; };
                    if lower_bound < min_depth { min_depth = lower_bound; }
                    continue;
                }
                let neighbor_pattern = self.pattern_manipulator.moved_blank_fast(search_state.pattern, neighbor_tile, neighbor_cell);
                let neighbor_db_heuristic = self.heuristic_value_from_db(neighbor_pattern);
                let heuristic_value = if neighbor_db_heuristic > manhattan_heuristic {
                    let lower_bound = neighbor_depth + neighbor_db_heuristic as u16;
                    if lower_bound > depth_limit {
                        if !stats.leaf() { return u16::MAX; };
                        if lower_bound < min_depth { min_depth = lower_bound; }
                        continue;
                    }
                    neighbor_db_heuristic
                } else {
                    manhattan_heuristic
                };
                neighbors.push(SearchStateWithEval{
                    search_state: SearchState {
                        state: neighbor_state,
                        pattern: neighbor_pattern,
                        extra_heuristic: manhattan_heuristic
                    },
                    heuristic_value
                });
            }
            neighbors.sort_by_key(|neighbor| neighbor.heuristic_value);
        }
        path.push(search_state.state);
        //path.insert(state.state);
        for neighbor in neighbors {
            let v = self.search_rec(neighbor, path, depth_limit, stats);
            if v == depth_limit { return depth_limit; } // the goal is found!
            if v < min_depth { min_depth = v; }
        }
        path.pop();
        //path.remove(&state.state);
        min_depth
    }

    /// Returns number of moves to solve the state or u16::MAX if it is not solvable.
    /// Collect statistics during search.
    pub fn moves_to_solve_manhattan_first_stats(&self, state: State, stats: &mut impl SearchStatsCollector) -> u16 {
        if state == self.goal_state { return 0; }
        let search_state = {
            let pattern = self.pattern_manipulator.pattern_for(state);
            let manhattan_heuristic = calc_manhattan_heuristic(&self.extra_heuristic, state);
            SearchStateWithEval {
                search_state: SearchState {
                    state, pattern,
                    extra_heuristic: manhattan_heuristic
                },
                heuristic_value: self.heuristic_value_from_db(pattern).max(manhattan_heuristic)
            }
        };
        let mut depth_limit = search_state.heuristic_value as u16;
        loop {
            let v = self.search_rec_manhattan_first(search_state.search_state, &mut Vec/*HashSet*/::with_capacity(128), depth_limit, stats);
            if v == depth_limit || v == u16::MAX { return v; }
            depth_limit = v;
        }
    }

    /// Returns number of moves to solve the state or u32::MAX if it is not solvable.
    #[inline] pub fn moves_to_solve_manhattan_first(&self, state: State) -> u16 {
        self.moves_to_solve_manhattan_first_stats(state, &mut ())
    }
}


#[cfg(test)]
mod tests {
    //use super::*;


}