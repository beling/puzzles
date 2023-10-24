#![doc = include_str!("../README.md")]

use puzzles::solver::{SlidingPuzzleSolver, ExtraHeuristic};

use puzzles::puzzle_sliding16::state::State;
use puzzles::puzzle_sliding16::neighbors::{Neighbors, neighbors_of, construct_neighbors};
use cpu_time::ProcessTime;
use csf::{fp, bits_to_store};
use puzzles::pattern_db::{UseHashMap, UseLSCMap, UseLSMap, PatternDBManager, UseFPMap, UseFPCMap, UseHashMin, UseHashMinGrouped};
use puzzles::puzzle_sliding16::pattern::{PatternManipulator, build_pattern_db};
use fsum::FSum;
use csf::fp::collision_solver::AcceptLimitedAverageDifference;
use csf::fp::level_size_chooser::{OptimalGroupedLevelSize, ResizedLevel};
use rand_chacha::ChaCha8Rng;
use std::fs::File;
use std::io::{Write, BufReader, BufRead};
use std::{env, io};
use std::ops::RangeInclusive;
use std::collections::HashMap;
use puzzles::stats::SearchAllStats;
//use pdbs::puzzle_sliding16::heuristic::{manhattan_metric, calc_manhattan_heuristic};
use puzzles::hash_min_db::HashMinGroupedPatternDBConf;
use rand::{SeedableRng, seq::SliceRandom};

struct TestStateSolution {
    who_solved: String,
    move_to_solve: u16
}

/// State to be tested.
struct TestState {
    state: State,
    solution: Option<TestStateSolution>
}

impl TestState {
    pub fn new(state: State) -> Self { Self{ state, solution: None } }
}

struct Test<EH: ExtraHeuristic + Clone> {
    neighbors: Neighbors,
    //pattern_manipulator: PatternManipulator,
    goal_state: State,
    important_tiles: Vec<u8>,
    pattern_db: Vec<Vec<u32>>,
    cols: u8,
    rows: u8,
    test_states: Vec<TestState>,
    pattern_manipulator: PatternManipulator,
    extra_heuristic: EH,
    store_details: bool
}

// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
fn sdev(sum: u64, sqrsum: u64, n: u64) -> f64 {
    ((sqrsum as f64 - (sum * sum) as f64 / n as f64) / (n-1) as f64).sqrt()
}

impl<EH: ExtraHeuristic+Clone> Test<EH> {
    fn new(cols: u8, rows: u8, important_tiles: Vec<u8>, extra_heuristic: EH) -> Self {
        let neighbors = construct_neighbors(cols, rows);
        let (pattern_db, pattern_manipulator) = build_pattern_db(&mut {u8::MAX}, important_tiles.iter().cloned(), &neighbors);
        Self {
            neighbors,
            //pattern_manipulator,
            goal_state: State::goal(cols * rows),
            important_tiles,
            pattern_db,
            cols, rows,
            test_states: Vec::new(),
            pattern_manipulator,
            extra_heuristic,
            store_details: false
        }
    }

    fn print_pattern_db_stats(&self) {
        let total_len = self.pattern_db_len();
        let mut distance_sum = 0;
        let mut len_sum = 0;
        for (distance_to_goal, len) in self.pattern_db.iter().map(|v|v.len()).enumerate() {
            len_sum += len;
            distance_sum += distance_to_goal * len;
            let total_len = total_len as f64;
            println!("{}\t{} {:.2}%\t{} {:.2}%", distance_to_goal, len, len as f64 *100.0 / total_len, len_sum, len_sum as f64*100.0 / total_len);
        }
        let entropy = self.pattern_db_value_entropy();
        println!("pattern database: {} abstract states with average distance to goal {:.2}, value entropy {:.2}, bits to store value {}.",
                 total_len, distance_sum as f64 / total_len as f64, entropy, self.pattern_db_bits_per_value());
    }

    /// Returns number of abstract states in the whole pattern database.
    fn pattern_db_len(&self) -> usize {
        self.pattern_db.iter().map(|v|v.len()).sum()
    }

    /// Returns entropy of values assigned to abstract states in pattern database.
    fn pattern_db_value_entropy(&self) -> f64 {
        let total_len = self.pattern_db_len() as f64;
        - FSum::with_all(self.pattern_db.iter().map(|v| {
            let p = v.len() as f64 / total_len as f64;
            p * p.log2()
        })).value()
    }

    /// Returns number of bits needed to represent values (number of moves to goal).
    fn pattern_db_bits_per_value(&self) -> u8 {
        bits_to_store!(self.pattern_db.len()-1)
    }

    /// Returns random puzzle state.
    fn rand_state(&self, rng: &mut ChaCha8Rng) -> State {
        let mut state = self.goal_state;
        let mut blank_pos = 0;
        let mut prev_blank_pos = u8::MAX; // to: (1) do not undo moves; (2) randomized number of moves made
        for _ in 0..1000 {
            let new_blank_pos = *neighbors_of(&self.neighbors, blank_pos).choose(rng).unwrap();
            if new_blank_pos == prev_blank_pos { continue; }
            prev_blank_pos = blank_pos;
            state.move_blank(blank_pos, new_blank_pos);
            blank_pos = new_blank_pos;
        }
        state
    }

    fn add_test_state(&mut self, state: State) {
        self.test_states.push(TestState::new(state));
    }

    fn add_random_test_states(&mut self, rng: &mut ChaCha8Rng, how_many: usize) {
        for _ in 0..how_many { self.add_test_state(self.rand_state(rng)); }
    }

    fn add_test_states_from_file(&mut self, file_name: &str) -> io::Result<()> {
        let f = File::open(file_name)?;
        let f = BufReader::new(f);
        for line in f.lines() {
            let state: State = line?.split_whitespace().skip(1).map(|s| s.parse().unwrap()).collect();
            if state.board != 0 { self.add_test_state(state); }
        }
        Ok(())
    }

    /*fn pattern_db(&self, max_pattern_distance: &mut u8) -> Vec::<Vec::<u32>> {
        build_pattern_db(max_pattern_distance, self.important_tiles.iter().cloned(), &self.neighbors).0
    }*/

    fn solver<PDBM: PatternDBManager>(&self, pdbm: PDBM, max_pattern_distance: u8) -> SlidingPuzzleSolver<PDBM, EH> {
        SlidingPuzzleSolver::<PDBM, EH> {
            neighbors: self.neighbors,
            pattern_manipulator: self.pattern_manipulator,
            goal_state: self.goal_state,
            extra_heuristic: self.extra_heuristic.clone(),
            pattern_db: pdbm.construct(self.pattern_db[0..=max_pattern_distance as usize].into()),
            max_pattern_distance_plus_one: max_pattern_distance+1
        }
    }

    fn create_file(&self, name: &str, tested_param: &RangeInclusive<usize>, step: usize) -> (File, Option<File>) {
        let file_name = format!("{}x{}_tiles_{}__{}__{}to{}s{}.csv", self.cols, self.rows,
                                self.important_tiles.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("_"),
                                name.replace(" ", "_"),
                                tested_param.start(), tested_param.end(), step);
        println!("{}", file_name);
        let mut file = File::create(&file_name).unwrap();
        write!(file, "method,parameter,\
                      size,size_per_entry,\
                      distance_db,distance_db_relative,").unwrap();
        if !EH::IS_FAKE {
            write!(file, "distance_states_db,distance_states_db_sdev,\
                          distance_states_manhattan,distance_states_manhattan_sdev,").unwrap();
        }
        writeln!(file, "distance_states,distance_states_sdev,\
                        nodes,states,nodes_per_state,nodes_per_state_sdev,\
                        internal,leaves,time_per_state").unwrap();
        if self.store_details {
            let mut file_details = File::create(format!("details_{}", file_name)).unwrap();
            writeln!(file_details, "parameter,state_index,internal,leaves,time").unwrap();
            (file, Some(file_details))
        } else {
            (file, None)
        }
    }

    fn test_solver<PDBM: PatternDBManager>(&mut self, (file, file_details): (&mut File, &mut Option<File>), name: &str, parameter: usize, solver: SlidingPuzzleSolver<PDBM, EH>, max_pattern_distance: u8, pdbm_is_lossless: bool) {
        let total_len = self.pattern_db_len();
        let db_size_bytes = PDBM::size_bytes(&solver.pattern_db);
        let db_size_bits = 8 * db_size_bytes;
        println!("  {:.3} bits/entry used by pattern db ({} bytes),", db_size_bits as f64 / total_len as f64, db_size_bytes);
        write!(file, "{},{},{},{}", name, parameter, db_size_bytes, db_size_bits as f64 / total_len as f64).unwrap();

        let mut total_dist_to_goal_patterns = 0u64;
        let mut total_real_dist_to_goal_patterns = 0u64;
        for (real_dist_to_goal, patterns) in self.pattern_db.iter().enumerate() {
            for pattern in patterns {
                let dist_to_goal_from_db = solver.heuristic_value_from_db(*pattern);
                if pdbm_is_lossless && real_dist_to_goal <= max_pattern_distance as usize {
                    if dist_to_goal_from_db as usize != real_dist_to_goal {
                        eprintln!("{}: heuristic value for pattern {} is {} != {} (real)",
                                  name, pattern, dist_to_goal_from_db, real_dist_to_goal);
                    }
                } else if dist_to_goal_from_db as usize > real_dist_to_goal {
                    eprintln!("{}: heuristic value for pattern {} is {} > {} (real)",
                              name, pattern, dist_to_goal_from_db, real_dist_to_goal);
                }
                total_dist_to_goal_patterns += dist_to_goal_from_db as u64;
            }
            total_real_dist_to_goal_patterns += real_dist_to_goal as u64 * patterns.len() as u64;
        }
        write!(file, ",{},{}", total_dist_to_goal_patterns as f64 / total_len as f64, total_dist_to_goal_patterns as f64 / total_real_dist_to_goal_patterns as f64).unwrap();
        let mut total_dist_to_goal_states_db = 0u64;
        let mut total_sqr_dist_to_goal_states_db = 0u64;
        let mut total_dist_to_goal_states_manhattan = 0u64;
        let mut total_sqr_dist_to_goal_states_manhattan = 0u64;
        let mut total_dist_to_goal_states = 0u64;
        let mut total_sqr_dist_to_goal_states = 0u64;
        {
            let mut rng = ChaCha8Rng::seed_from_u64(314);
            const SAMPLE_SIZE: u64 = 1000000;
            let mut max_total_dist_to_goal = 0;
            for _ in 0..SAMPLE_SIZE {
                let state = self.rand_state(&mut rng);
                let db_dist = solver.heuristic_value_from_db(
                    self.pattern_manipulator.pattern_for(state)
                ) as u64;
                total_dist_to_goal_states_db += db_dist;
                total_sqr_dist_to_goal_states_db += db_dist*db_dist;
                if !EH::IS_FAKE {
                    let manhattan_dist = solver.extra_heuristic.value_for_state_u8(state) as u64;
                    total_dist_to_goal_states_manhattan += manhattan_dist;
                    total_sqr_dist_to_goal_states_manhattan += manhattan_dist * manhattan_dist;
                    let max_dist = db_dist.max(manhattan_dist);
                    total_dist_to_goal_states += max_dist;
                    total_sqr_dist_to_goal_states += max_dist*max_dist;
                    if max_dist > max_total_dist_to_goal { max_total_dist_to_goal = max_dist; }
                } else {
                    if db_dist > max_total_dist_to_goal { max_total_dist_to_goal = db_dist; }
                }
            }
            println!("  {:.2} (max {}) and {:.2} distance to goal for random states and patterns in database respectively.",
                     total_dist_to_goal_states_db as f64 / SAMPLE_SIZE as f64, max_total_dist_to_goal,
                     total_dist_to_goal_patterns as f64 / total_len as f64);
            write!(file, ",{},{}",
                   total_dist_to_goal_states_db as f64 / SAMPLE_SIZE as f64, sdev(total_dist_to_goal_states_db, total_sqr_dist_to_goal_states_db, SAMPLE_SIZE)
            ).unwrap();
            if !EH::IS_FAKE {
                write!(file, ",{},{},{},{}",
                       total_dist_to_goal_states_manhattan as f64 / SAMPLE_SIZE as f64, sdev(total_dist_to_goal_states_manhattan, total_sqr_dist_to_goal_states_manhattan, SAMPLE_SIZE),
                       total_dist_to_goal_states as f64 / SAMPLE_SIZE as f64, sdev(total_dist_to_goal_states, total_sqr_dist_to_goal_states, SAMPLE_SIZE)
                ).unwrap();
            }
        }

        let mut total_visits = SearchAllStats::default();
        let mut total_visits_sqr = 0;
        let mut total_seconds = 0f64;
        for (state_idx, test) in self.test_states.iter_mut().enumerate() {
            let mut visits = SearchAllStats::default();
            let start_moment = ProcessTime::try_now().expect("Getting process time failed");
            let ans = solver.moves_to_solve_stats(test.state, &mut visits);
            let seconds = start_moment.try_elapsed().expect("Getting process time failed").as_secs_f64();
            total_visits += visits;
            total_visits_sqr += { let v = visits.visits(); v*v };
            total_seconds += seconds;
            if let Some(ref solution) = test.solution {
                if ans != solution.move_to_solve  {
                    eprintln!("{}: wrong answer given for state {}: {} (got) != {} (by {})", name, test.state.board, ans, solution.move_to_solve, solution.who_solved);
                }
            } else {
                test.solution = Some(TestStateSolution{who_solved: name.to_owned(), move_to_solve: ans});
            }
            //print!("."); io::stdout().flush().unwrap();
            //println!("{}: answer {}, visits {} time {}", index, ans, visits, seconds)
            //method,parameter,state_index,nodes,internal,leaves,time
            if let Some(file_details) = file_details {
                writeln!(file_details, "{},{},{},{},{}", parameter, state_idx, visits.internal, visits.leaves, seconds).unwrap();
            }
        }
        let v = total_visits.visits();
        println!("  {:.0} nodes/case expanded, {} sec/case.",
                 v as f64 / self.test_states.len() as f64,
                 total_seconds / self.test_states.len() as f64);
        writeln!(file, ",{},{},{},{},{},{},{}",
                 v, self.test_states.len(), v as f64 / self.test_states.len() as f64,
                 sdev(total_visits.visits(), total_visits_sqr, self.test_states.len() as _),
                 total_visits.internal, total_visits.leaves,
                 total_seconds / self.test_states.len() as f64).unwrap();
    }

    fn max_pattern_distances<PDBM: Clone + PatternDBManager>(&mut self, name: &str, pdbm: PDBM, max_pattern_distances: RangeInclusive<u8>, step: u8) {
        let max_pattern_distances = *max_pattern_distances.start()..=(*max_pattern_distances.end()).min((self.pattern_db.len()-1) as u8);
        let (mut file, mut file_details) = self.create_file(name, &(*max_pattern_distances.start() as usize..=*max_pattern_distances.end() as usize), step.into());
        let pdbm_is_lossless = pdbm.is_lossless();
        for max_pattern_distance in max_pattern_distances.step_by(step as usize) {
            println!("{} m={}", name, max_pattern_distance);
            self.test_solver((&mut file, &mut file_details), name, max_pattern_distance as _, self.solver(pdbm.clone(), max_pattern_distance), max_pattern_distance, pdbm_is_lossless);
        }
    }

    fn mindb_sizes(&mut self, /*max_pattern_distance: u8,*/ relative_sizes_percent: RangeInclusive<usize>, step: usize) {
        let name = "hashmin";
        let max_pattern_distance = (self.pattern_db.len()-1) as u8;
        let db_len = self.pattern_db_len();
        let (mut file, mut file_details) = self.create_file(name, &relative_sizes_percent, step);
        for size in relative_sizes_percent.step_by(step) {
            let pdbm = UseHashMin::new(self.pattern_db_bits_per_value(), size * db_len / 100);
            let pdbm_is_lossless = pdbm.is_lossless();
            println!("{} s={}%", name, size);
            self.test_solver(
                (&mut file, &mut file_details), name,
                size, self.solver(pdbm, max_pattern_distance),
                max_pattern_distance, pdbm_is_lossless);
        }
    }

    fn mingroupeddb_sizes(&mut self, bits_per_seed: u8, cells_per_group: u8, /*max_pattern_distance: u8,*/ relative_sizes_percent: RangeInclusive<usize>, step: usize) {
        let name = format!("hashgroupedmin bps={} g={}", bits_per_seed, cells_per_group);
        let max_pattern_distance = (self.pattern_db.len()-1) as u8;
        let db_len_bits = self.pattern_db_len() * self.pattern_db_bits_per_value() as usize;
        let (mut file, mut file_details) = self.create_file(&name, &relative_sizes_percent, step);
        for size in relative_sizes_percent.step_by(step) {
            let pdbm = UseHashMinGrouped::from(
                HashMinGroupedPatternDBConf::with_total_size(
                    db_len_bits * size / 100,
                    cells_per_group,
                    self.pattern_db_bits_per_value(),
                    bits_per_seed)
            );
            let pdbm_is_lossless = pdbm.is_lossless();
            println!("{} s={}%", name, size);
            self.test_solver(
                (&mut file, &mut file_details), &name,
                size, self.solver(pdbm, max_pattern_distance),
                max_pattern_distance, pdbm_is_lossless);
        }
    }

    fn fp_thresholds(&mut self, max_pattern_distance: u8, thresholds: RangeInclusive<u8>, resize_percent: u16) {
        let name = if resize_percent == 100 {
            format!("fp m={}", max_pattern_distance)
        } else {
            format!("fp m={} s={}", max_pattern_distance, resize_percent)
        };
        let (mut file, mut file_details) = self.create_file(&name, &(*thresholds.start() as usize..=*thresholds.end() as usize), 1);
        for threshold in thresholds {
            println!("{} t={}", name, threshold);
            if threshold == 0 {
                if resize_percent == 100 {
                    let pdbm = UseFPMap::default();
                    let pdbm_is_lossless = pdbm.is_lossless();
                    self.test_solver((&mut file, &mut file_details), &name, threshold as _, self.solver(pdbm, max_pattern_distance), max_pattern_distance, pdbm_is_lossless);
                } else {
                    let pdbm = UseFPMap::from(fp::MapConf::lsize(ResizedLevel::new(resize_percent, fp::OptimalLevelSize::default())));
                    let pdbm_is_lossless = pdbm.is_lossless();
                    self.test_solver((&mut file, &mut file_details), &name, threshold as _, self.solver(pdbm, max_pattern_distance), max_pattern_distance, pdbm_is_lossless);
                }
            } else {
                if resize_percent == 100 {
                    let pdbm = UseFPMap::from(fp::MapConf::lsize_cs(
                        OptimalGroupedLevelSize::with_divider(threshold + 1),
                        AcceptLimitedAverageDifference::new(threshold)));
                    let pdbm_is_lossless = pdbm.is_lossless();
                    self.test_solver((&mut file, &mut file_details), &name, threshold as _, self.solver(pdbm, max_pattern_distance), max_pattern_distance, pdbm_is_lossless);
                } else {
                    let pdbm = UseFPMap::from(fp::MapConf::lsize_cs(
                        ResizedLevel::new(resize_percent, OptimalGroupedLevelSize::with_divider(threshold + 1)),
                        AcceptLimitedAverageDifference::new(threshold)));
                    let pdbm_is_lossless = pdbm.is_lossless();
                    self.test_solver((&mut file, &mut file_details), &name, threshold as _, self.solver(pdbm, max_pattern_distance), max_pattern_distance, pdbm_is_lossless);
                }
            }
        }
    }
}

enum Args {
    Run(HashMap<String, bool>),
    Help(Vec<String>)
}

impl Args {
    fn new() -> Self {
        let args: HashMap<String, bool> = env::args().skip(1).map(|s| (s, false)).collect();
        if args.is_empty() { Self::Help(Vec::new()) } else { Self::Run(args) }
    }

    fn case(&mut self, s: &str) -> bool {
        match self {
            &mut Self::Run(ref mut set) => {
                if let Some(used) = set.get_mut(s) {
                    *used = true;
                    println!("---=== run {} ===---", s);
                    true
                } else { false }
            }
            &mut Self::Help(ref mut v) => { v.push(s.to_string()); false }
        }
    }
}

impl Drop for Args {
    fn drop(&mut self) {
        match self {
            Self::Run(ref set) => {
                for (k, used) in set {
                    if !used { eprintln!("Unrecognized argument: {}", k); }
                }
            }
            Self::Help(ref v) => {
                println!("Acceptable arguments:");
                for a in v { println!(" {}", a); }
            }
        }
    }
}

fn main() {
    let mut args = Args::new();

    //let mut test = Test::new(4, 4, [0u8, 3, 7, 11, 12, 13, 14, 15].to_vec(), manhattan_metric(4, 4));
    //let mut test = Test::new(4, 4, [0u8, 3, 7, 11, 12, 13, 14, 15].to_vec(), ());
    //test.add_test_states_from_file("korf100.txt").unwrap();

    // 6 -> 4 ??
    let mut test = Test::new(4, 3, [0u8, 3, 6, 7, 8, 9, 10, 11].to_vec(), ());
    test.add_random_test_states(&mut ChaCha8Rng::seed_from_u64(123), 10000); // 200000

    test.store_details = test.rows == 4;

    test.print_pattern_db_stats();

    let max_m = (test.pattern_db.len()-1) as u8;

    if args.case("HashMap") {
        //let max = (test.pattern_db.len()-1) as u8;
        test.max_pattern_distances("HashMap", UseHashMap, if test.rows == 3 {24..=30} else {32..=38}, 1);
    }
    if args.case("HashMapMax") {
        //let max = (test.pattern_db.len()-1) as u8;
        test.max_pattern_distances("HashMapMax", UseHashMap, max_m..=max_m, 1);
    }

    if args.case("hashmin") {
        test.mindb_sizes(5..=120, 5);
    }

    for bits_per_seed in 1..=12 {
        for cells_per_group in 2..=8 {
            let name = format!("ghashmin bps={} g={}", bits_per_seed, cells_per_group);
            if args.case(&name) ||
                args.case("ghashmin") ||
                args.case(&format!("ghashmin bps={}", bits_per_seed)) ||
                args.case(&format!("ghashmin g={}", cells_per_group))
            {
                test.mingroupeddb_sizes(bits_per_seed, cells_per_group, 5..=100, 5);
            }
        }
    }


    if args.case("fpc") {
        test.max_pattern_distances("fpc", UseFPCMap::default(), if test.rows == 3 {26..=37} else {34..=43}, 1);
    }
    for s in (110..=250).step_by(10) {
        let name = format!("fpc s={}", s);
        if args.case(&name) {
            let pdbs = UseFPCMap::from(fp::CMapConf::lsize(ResizedLevel::new(s,fp::OptimalLevelSize::default())));
            test.max_pattern_distances(&name, pdbs, if test.rows == 3 {26..=36} else {34..=43}, 1);
        }

    }

    if args.case("ls") {
        test.max_pattern_distances("ls", UseLSMap::randomly(12321, 0), if test.rows == 3 {31..=36} else {39..=46}, 1);
    }
    for extra_bits in 1..=3 {
        let name = format!("ls+{}", extra_bits);
        if args.case(&name) {
            test.max_pattern_distances(&name, UseLSMap::randomly(12321, extra_bits), if test.rows == 3 {24..=33} else {35..=44}, 1);
        }
    }

    for b in 1..=3 {
        let name = format!("lsc b={}", b);
        if args.case(&name) {
            test.max_pattern_distances(&name, UseLSCMap::bpf(b), if test.rows == 3 {/*26*/28..=44} else {35..=45}, 1);
            //test.max_pattern_distances(&name, UseLSCMap::bpf(b), 29..=40, 1);
        }
        for extra_bits in 1..=3 {
            let name = format!("lsc+{} b={}", extra_bits, b);
            if args.case(&name) {
                test.max_pattern_distances(&name, UseLSCMap::randomly_bits(12321, b, extra_bits), if test.rows == 3 {25..=40} else {36..=40}, 1);
                //test.max_pattern_distances(&name, UseLSCMap::randomly_bits(12321, b, extra_bits), 28..=40, 1);
            }
        }
    }

    /*for t in 1..=9 {
        let name = format!("fp t={}", t);
        if args.case(&name) {
            test.max_pattern_distances(&name, UseFPMap::from(FPCMapConf::lsize_cs(
                OptimalGroupedLevelSize::with_divider(t + 1),
                AcceptLimitedAverageDifference::new(t))),
                                       27..=35, 1);
        }
    }*/

    if args.case("fp m=31") {
        test.fp_thresholds(31, 0..=8, 100);
    }
    if args.case("fp m=34") {
        test.fp_thresholds(34, 0..=10, 100);
    }
    if args.case("fp m=36") {
        test.fp_thresholds(36, 0..=10, 100);
    }
    if args.case("fp m=39") {
        test.fp_thresholds(39, 0..=12, 100);
    }
    if args.case("fp m=44") {
        test.fp_thresholds(44, 0..=16, 100);
    }
    if test.rows == 4 && args.case("fp m=63") {
        test.fp_thresholds(63, 0..=30, 100);
    }

    if args.case(&format!("fp m={}", max_m)) {
        test.fp_thresholds(max_m, 0..=18, 100);
    }

    for s in (150..=400).step_by(50) {
        let name = format!("fp m={} s={}", max_m, s);
        if args.case(&name) {
            //test.fp_thresholds(max_m, 1..=26, s);
            test.fp_thresholds(max_m, 0..=49, s);
        }
        let name = format!("fp m=63 s={}", s);
        if test.rows == 4 && args.case(&name) {
            //test.fp_thresholds(max_m, 1..=26, s);
            test.fp_thresholds(63, 0..=49, s);
        }
    }
}
