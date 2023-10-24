use std::collections::HashMap;
use csf::{ls, fp, GetSize, bits_to_store};
use ph::{BuildSeededHasher, BuildDefaultSeededHasher};
use std::hash::{BuildHasherDefault, BuildHasher};
use std::collections::hash_map::DefaultHasher;
use csf::ls::{MapConf, FillRandomly, ValuesPreFiller};
use csf::fp::{CollisionSolverBuilder, IsLossless, LoMemAcceptEquals};
use crate::hash_min_db::{HashMinPatternDB, HashMinGroupedPatternDB, HashMinGroupedPatternDBConf};
use csf::coding::{BuildMinimumRedundancy, minimum_redundancy};

/// Representation of pattern -> distance to goal map, which can be (lossy) compressed.
pub trait PatternDBManager {

    type PatternDB;

    /// Get from `db` and returns heuristic value for given `pattern` or `max_pattern_distance_plus_one` if `pattern` is not in `db`.
    /// Heuristic value is equal to or lower than `pattern`'s distance to goal.
    fn heuristic_value(db: &Self::PatternDB, pattern: u32, max_pattern_distance_plus_one: u8) -> u8;

    /// Constructs (lossy compressed) representation of database with given distances to goal (`data`).
    /// i-th vector in `data` contains all patterns with distance to goal equal i
    fn construct(self, data: Vec::<Vec::<u32>>) -> Self::PatternDB;

    /// Returns the size of `db` in bytes.
    fn size_bytes(db: &Self::PatternDB) -> usize;

    /// Returns `true` if `construct` produces the database which are lossless for positions given as argument.
    fn is_lossless(&self) -> bool { true }
}

/// Fake `PatternDBManager` implementation that discards all data and always returns `0` as heuristic value.
impl PatternDBManager for () {
    type PatternDB = ();
    #[inline(always)] fn heuristic_value(_db: &Self::PatternDB, _pattern: u32, _max_pattern_distance_plus_one: u8) -> u8 { 0 }
    #[inline(always)] fn construct(self, _data: Vec<Vec<u32>>) -> Self::PatternDB { () }
    #[inline(always)] fn size_bytes(_db: &Self::PatternDB) -> usize { 0 }
}

#[derive(Copy, Clone)]
pub struct UseHashMap;

impl PatternDBManager for UseHashMap {
    type PatternDB = HashMap<u32, u8>;

    #[inline(always)] fn heuristic_value(db: &Self::PatternDB, pattern: u32, max_pattern_distance_plus_one: u8) -> u8 {
        *db.get(&pattern).unwrap_or(&max_pattern_distance_plus_one)
    }

    fn construct(self, data: Vec<Vec<u32>>) -> Self::PatternDB {
        let mut result = HashMap::with_capacity(data.iter().map(|v|v.len()).sum());
        for (distance, positions) in data.into_iter().enumerate() {
            result.extend(positions.into_iter().map(|p| (p, distance as u8)));
        }
        result
    }

    fn size_bytes(db: &Self::PatternDB) -> usize { 5*db.len() }
}

#[derive(Default, Copy, Clone)]
pub struct UseLSMap<BM = (), S = BuildDefaultSeededHasher> {
    pub conf: MapConf<BM, S>,
    pub extra_bits_per_value: u8
}

impl UseLSMap {
    pub fn new(extra_bits_per_value: u8) -> Self {
        Self {
            conf: Default::default(),
            extra_bits_per_value
        }
    }
}

impl UseLSMap<FillRandomly> {
    pub fn randomly(seed: u64, extra_bits_per_value: u8) -> Self {
        Self {
            conf: MapConf::randomly(seed),
            extra_bits_per_value
        }
    }
}

impl<BM: ValuesPreFiller, S: BuildSeededHasher> PatternDBManager for UseLSMap<BM, S> {
    type PatternDB = ls::Map<S>;

    #[inline(always)] fn heuristic_value(db: &Self::PatternDB, pattern: u32, max_pattern_distance_plus_one: u8) -> u8 {
        db.get(&pattern).min(max_pattern_distance_plus_one as u64) as u8
    }

    fn construct(self, data: Vec<Vec<u32>>) -> Self::PatternDB {
        let bpv = bits_to_store!(data.len()-1) + self.extra_bits_per_value;
        Self::PatternDB::try_from_hashmap_bpv(UseHashMap.construct(data), bpv, self.conf).unwrap()
    }

    fn size_bytes(db: &Self::PatternDB) -> usize { db.size_bytes() }
}

#[derive(Default, Copy, Clone)]
pub struct UseLSCMap<BM = (), S = BuildDefaultSeededHasher> {
    pub conf: MapConf<BM, S>,
    pub bits_per_fragment: u8,
    pub bdz_extra_bits_per_fragment: u8
}

impl UseLSCMap {
    pub fn new() -> Self { Default::default() }

    pub fn bpf(bits_per_fragment: u8) -> Self {
        Self { conf: Default::default(), bits_per_fragment, bdz_extra_bits_per_fragment: Default::default() }
    }
}

impl UseLSCMap<FillRandomly> {
    pub fn randomly(seed: u64) -> Self {
        Self { conf: MapConf::randomly(seed), bits_per_fragment: Default::default(), bdz_extra_bits_per_fragment: Default::default() }
    }

    pub fn randomly_bits(seed: u64, bits_per_fragment: u8, bdz_extra_bits_per_fragment: u8) -> Self {
        Self { conf: MapConf::randomly(seed), bits_per_fragment, bdz_extra_bits_per_fragment }
    }
}

impl<BM: ValuesPreFiller, S: BuildSeededHasher> PatternDBManager for UseLSCMap<BM, S> {
    type PatternDB = ls::CMap::<minimum_redundancy::Coding<u8>, S>;

    #[inline(always)] fn heuristic_value(db: &Self::PatternDB, pattern: u32, max_pattern_distance_plus_one: u8) -> u8 {
        (*db.get(&pattern).unwrap_or(&max_pattern_distance_plus_one)).min(max_pattern_distance_plus_one)
    }

    fn construct(self, data: Vec<Vec<u32>>) -> Self::PatternDB {
        Self::PatternDB::try_from_map_with_conf(&UseHashMap.construct(data), self.bits_per_fragment, self.conf, self.bdz_extra_bits_per_fragment).unwrap()
    }

    fn size_bytes(db: &Self::PatternDB) -> usize { db.size_bytes() }
}

//#[derive(Default)]
#[derive(Clone)]
pub struct UseFPCMap<
    LSC = fp::OptimalLevelSize,
    CSB: CollisionSolverBuilder = LoMemAcceptEquals,
    S: BuildSeededHasher = BuildDefaultSeededHasher
> {
    pub conf: fp::CMapConf<BuildMinimumRedundancy, LSC, CSB, S>
}

impl Default for UseFPCMap<fp::OptimalLevelSize, LoMemAcceptEquals, BuildDefaultSeededHasher> {
    fn default() -> Self {
        Self {conf: Default::default()}
    }
}

impl<LSC, CSB, S> From<fp::CMapConf<BuildMinimumRedundancy, LSC, CSB, S>> for UseFPCMap<LSC, CSB, S>
where CSB: CollisionSolverBuilder, S: BuildSeededHasher
{
    fn from(conf: fp::CMapConf<BuildMinimumRedundancy, LSC, CSB, S>) -> Self {
        Self { conf }
    }
}

impl<LSC: fp::LevelSizeChooser, CS: CollisionSolverBuilder+IsLossless, S: BuildSeededHasher> PatternDBManager for UseFPCMap<LSC, CS, S> {
    type PatternDB = fp::CMap<minimum_redundancy::Coding<u8>, S>;

    #[inline(always)] fn heuristic_value(db: &Self::PatternDB, pattern: u32, max_pattern_distance_plus_one: u8) -> u8 {
        *db.get(&pattern).unwrap_or(&max_pattern_distance_plus_one)
    }

    fn construct(self, data: Vec<Vec<u32>>) -> Self::PatternDB {
        Self::PatternDB::from_map_with_conf(&UseHashMap.construct(data), self.conf, &mut ())
    }

    fn size_bytes(db: &Self::PatternDB) -> usize { db.size_bytes() }
}

#[derive(Clone)]
pub struct UseFPMap<
    LSC = fp::OptimalLevelSize,
    CSB: CollisionSolverBuilder = LoMemAcceptEquals,
    S: BuildSeededHasher = BuildDefaultSeededHasher
> {
    pub conf: fp::MapConf<LSC, CSB, S>
}

impl Default for UseFPMap<fp::OptimalLevelSize, LoMemAcceptEquals, BuildDefaultSeededHasher> {
    fn default() -> Self {
        Self {conf: Default::default()}
    }
}

impl<LSC, CSB, S> From<fp::MapConf<LSC, CSB, S>> for UseFPMap<LSC, CSB, S>
    where CSB: CollisionSolverBuilder, S: BuildSeededHasher
{
    fn from(conf: fp::MapConf<LSC, CSB, S>) -> Self {
        Self { conf }
    }
}

impl<LSC: fp::SimpleLevelSizeChooser, CS: CollisionSolverBuilder, S: BuildSeededHasher> PatternDBManager for UseFPMap<LSC, CS, S> {
    type PatternDB = fp::Map<S>;

    #[inline(always)] fn heuristic_value(db: &Self::PatternDB, pattern: u32, max_pattern_distance_plus_one: u8) -> u8 {
        db.get(&pattern).unwrap_or(max_pattern_distance_plus_one as _) as u8
    }

    fn construct(self, data: Vec<Vec<u32>>) -> Self::PatternDB {
        Self::PatternDB::with_map_conf(&UseHashMap.construct(data), self.conf, &mut /*()*/ ph::stats::BuildStatsPrinter::stdout())
    }

    fn size_bytes(db: &Self::PatternDB) -> usize { db.size_bytes() }

    fn is_lossless(&self) -> bool { self.conf.collision_solver.is_lossless() }
}

pub struct UseHashMin<S: BuildHasher = BuildHasherDefault<DefaultHasher>> {
    bit_per_cell: u8,
    num_of_cells: usize,
    hasher: S
}

impl UseHashMin {
    pub fn new(bit_per_cell: u8, num_of_cells: usize) -> Self {
        Self { bit_per_cell, num_of_cells, hasher: Default::default() }
    }
}

impl<S: BuildHasher> PatternDBManager for UseHashMin<S> {
    type PatternDB = HashMinPatternDB<S>;

    fn heuristic_value(db: &Self::PatternDB, pattern: u32, _max_pattern_distance_plus_one: u8) -> u8 {
        db.get(&pattern) as u8
    }

    fn construct(self, data: Vec<Vec<u32>>) -> Self::PatternDB {
        let mut result = HashMinPatternDB::with_hasher(self.num_of_cells, self.bit_per_cell, self.hasher);
        for (nimber, patterns) in data.iter().enumerate() {
            for pattern in patterns {
                result.set(pattern, nimber as u64);
            }
        }
        result
    }

    fn size_bytes(db: &Self::PatternDB) -> usize { db.size_bytes() }

    fn is_lossless(&self) -> bool { false }
}


pub struct UseHashMinGrouped<S: BuildHasher = BuildHasherDefault<DefaultHasher>> {
    conf: HashMinGroupedPatternDBConf<S>
}

impl<S: BuildHasher> From<HashMinGroupedPatternDBConf<S>> for UseHashMinGrouped<S> {
    fn from(conf: HashMinGroupedPatternDBConf<S>) -> Self {
        Self { conf }
    }
}

impl<S: BuildHasher> PatternDBManager for UseHashMinGrouped<S> {
    type PatternDB = HashMinGroupedPatternDB<S>;

    fn heuristic_value(db: &Self::PatternDB, pattern: u32, _max_pattern_distance_plus_one: u8) -> u8 {
        db.get(&pattern) as u8
    }

    fn construct(self, data: Vec<Vec<u32>>) -> Self::PatternDB {
        HashMinGroupedPatternDB::from_kv_conf::<u32, _,_,_,_>(
            || data.iter().enumerate().flat_map(
                |(nimber, patterns)| {
                    patterns.iter().map(move |pattern| (pattern, nimber as u8))
                }),
            self.conf
        )
    }

    fn size_bytes(db: &Self::PatternDB) -> usize { db.size_bytes() }

    fn is_lossless(&self) -> bool { false }
}