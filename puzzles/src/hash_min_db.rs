use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use bitm::{BitAccess, BitVec};
use std::borrow::Borrow;
use std::convert::TryInto;

/// PDBS that maps abstract positions to array cells with hasher
/// and stores minimum distance of abstract positions assigned to each cell.
pub struct HashMinPatternDB<S: BuildHasher = BuildHasherDefault<DefaultHasher>> {
    cells: Box<[u64]>,
    bit_per_cell: u8,
    num_of_cells: usize,
    hasher: S,
}

impl<S: BuildHasher> HashMinPatternDB<S> {
    #[inline] fn index<K: Hash>(&self, k: &K) -> usize {
        let mut hasher = self.hasher.build_hasher();
        //hasher.write_u32(seed);
        k.hash(&mut hasher);
        return (hasher.finish() as usize) % self.num_of_cells
    }

    #[inline] pub fn get<K: Hash>(&self, k: &K) -> u64 {
        self.cells.get_fragment(self.index(k), self.bit_per_cell)
    }

    #[inline] pub fn set<K: Hash>(&mut self, k: &K, value: u64) {
        self.cells.conditionally_change_fragment(
            |old| (value < old).then(|| value),
            self.index(k), self.bit_per_cell);
    }

    pub fn with_hasher(num_of_cells: usize, bit_per_cell: u8, hasher: S) -> Self {
        Self {
            cells: Box::<[u64]>::with_filled_bits(num_of_cells * bit_per_cell as usize),
            bit_per_cell, num_of_cells, hasher
        }
    }

    pub fn with_maxvalue_hasher(num_of_cells: usize, bit_per_cell: u8, maxvalue: u64, hasher: S) -> Self {
        assert!(maxvalue < (1<<bit_per_cell));
        let mut result = Self {
            cells: Box::<[u64]>::with_zeroed_bits(num_of_cells * bit_per_cell as usize),
            bit_per_cell, num_of_cells, hasher
        };
        for i in 0..num_of_cells {
            result.cells.init_fragment(i, maxvalue, bit_per_cell);
        }
        result
    }

    /// Calculate approximate number of bytes occupied by dynamic part of `self`.
    /// Same as `self.size_bytes() - std::mem::size_of_val(self)`.
    pub fn size_bytes_dyn(&self) -> usize {
        self.cells.len() * std::mem::size_of::<u64>()
    }

    /// Calculate approximate, total (including heap memory) number of bytes occupied by `self`.
    #[inline] pub fn size_bytes(&self) -> usize {
        std::mem::size_of_val(self) + self.size_bytes_dyn()
    }
}

impl HashMinPatternDB {
    fn new(num_of_cells: usize, bit_per_cell: u8) -> Self {
        Self::with_hasher(num_of_cells, bit_per_cell, Default::default())
    }

    fn with_maxvalue(num_of_cells: usize, bit_per_cell: u8, maxvalue: u64) -> Self {
        Self::with_maxvalue_hasher(num_of_cells, bit_per_cell, maxvalue, Default::default())
    }
}

/// PDBS that maps abstract positions to array cells with hasher
/// and stores minimum distance of abstract positions assigned to each cell.
/// Cells are grouped and each group has assigned additional, optimal hash seed.
pub struct HashMinGroupedPatternDBConf<S: BuildHasher = BuildHasherDefault<DefaultHasher>> {
    number_of_groups: usize,
    cells_per_group: u8,
    bits_per_cell: u8,
    bits_per_seed: u8,
    hasher: S,
}

impl HashMinGroupedPatternDBConf {
    pub fn new(number_of_groups: usize,
               cells_per_group: u8,
               bits_per_cell: u8,
               bits_per_seed: u8) -> Self
    {
        Self {
            number_of_groups,
            cells_per_group,
            bits_per_cell,
            bits_per_seed,
            hasher: Default::default()
        }
    }

    pub fn with_total_size(total_size_bits: usize, cells_per_group: u8, bits_per_cell: u8, bits_per_seed: u8) -> Self {
        Self {
            number_of_groups: (total_size_bits / (cells_per_group as usize*bits_per_cell as usize+bits_per_seed as usize)).max(1),
            cells_per_group,
            bits_per_cell,
            bits_per_seed,
            hasher: Default::default()
        }
    }
}

impl<S: BuildHasher> HashMinGroupedPatternDBConf<S> {

    /// Returns total error (sum of value difference) in each group for all `kv` items, when each group is indexed with the same, given `seed`.
    pub fn calc_errors<K, V, KBorrow, KVIntoIterator, KVGetter>(&self, kv: &KVGetter, seed: u64) -> Box<[u32]>
        where
            K: Hash,
            V: Into<u64>,
            KBorrow: Borrow<K>,
            KVIntoIterator: IntoIterator<Item=(KBorrow, V)>,    // Iterator over keys and values
            KVGetter: Fn() -> KVIntoIterator,      // Returns iterator over keys
    {
        let num_of_cells = self.number_of_groups * self.cells_per_group as usize;
        let index = |key: KBorrow| {
            let mut hasher = self.hasher.build_hasher();
            key.borrow().hash(&mut hasher);
            let group_nr = (hasher.finish() as usize) % self.number_of_groups;
            hasher.write_u64(seed);
            (group_nr, group_nr * self.cells_per_group as usize + hasher.finish() as usize % self.cells_per_group as usize)
        };
        let mut cells = Box::<[u64]>::with_filled_bits(num_of_cells * self.bits_per_cell as usize);
        for (k, v) in kv() {
            let new = v.into() as u64;
            cells.conditionally_change_fragment(|old| (new < old).then(|| new), index(k).1, self.bits_per_cell);
        }
        let mut errors = vec![0u32; self.number_of_groups].into_boxed_slice();
        for (k, v) in kv() {
            let (g, i) = index(k);
            errors[g] = errors[g].saturating_add((v.into() - cells.get_fragment(i, self.bits_per_cell)).try_into().unwrap());
        }
        errors
    }
}

pub struct HashMinGroupedPatternDB<S: BuildHasher = BuildHasherDefault<DefaultHasher>> {
    cells: Box<[u64]>,
    bits_per_cell: u8,
    bits_per_seed: u8,
    bits_per_group: u16, // = bits_per_seed + bits_per_cell * cells_per_group
    number_of_groups: usize,
    cells_per_group: u8,
    hasher: S,
}

impl<S: BuildHasher> HashMinGroupedPatternDB<S> {

    // Returns index of the group that can contain the given `key`.
    /*#[inline] fn group_index<K: Hash>(&self, key: &K) -> usize {
        let mut hasher = self.hasher.build_hasher();
        //hasher.write_u32(seed);
        key.hash(&mut hasher);
        return (hasher.finish() as usize) % self.number_of_groups
    }*/

    /// Returns `shift` (in `cells`) and the `seed` of the group with given `group_index`.
    #[inline] fn group_shift_seed(&self, group_index: usize) -> (usize, u64) {
        let shift = group_index * self.bits_per_group as usize;
        (shift, self.cells.get_bits(shift, self.bits_per_seed))
    }

    /*#[inline] fn in_group_index<K: Hash>(&self, seed: u8, k: &K) -> usize {
        let mut hasher = self.hasher.build_hasher();
        k.hash(&mut hasher);
        hasher.write_u8(seed);
        return (hasher.finish() as usize) & self.cells_per_group_mask
    }*/

    pub fn begin_bit_index<K: Hash>(&self, key: &K) -> usize {
        let mut hasher = self.hasher.build_hasher();
        key.hash(&mut hasher);
        let group = (hasher.finish() as usize) % self.number_of_groups;
        let (shift, seed) = self.group_shift_seed(group);
        hasher.write_u64(seed);
        let in_group_index = (hasher.finish() as usize) % self.cells_per_group as usize;
        shift + self.bits_per_seed as usize + in_group_index * self.bits_per_cell as usize
    }

    #[inline] pub fn get<K: Hash>(&self, key: &K) -> u64 {
        self.cells.get_bits(self.begin_bit_index(key), self.bits_per_cell)
    }

    #[inline] pub fn set<K: Hash>(&mut self, key: &K, value: u64) {
        self.cells.conditionally_change_bits(
            |old| if value < old { Some(value) } else { None },
            self.begin_bit_index(key), self.bits_per_cell);
    }

    pub fn from_kv_conf<K, V, KBorrow, KVIntoIterator, KVGetter>(kv: KVGetter, conf: HashMinGroupedPatternDBConf<S>) -> Self
    where
        K: Hash,
        V: Into<u64>,
        KBorrow: Borrow<K>,
        KVIntoIterator: IntoIterator<Item=(KBorrow, V)>,    // Iterator over keys and values
        KVGetter: Fn() -> KVIntoIterator,      // Returns iterator over keys
    {
        let mut best_errors = conf.calc_errors(&kv, 0);
        let bits_per_group = conf.bits_per_seed as usize + conf.bits_per_cell as usize * conf.cells_per_group as usize;
        let mut cells = Box::<[u64]>::with_filled_bits(conf.number_of_groups * bits_per_group as usize);
        for group in 0..conf.number_of_groups {
            cells.set_bits(group * bits_per_group, 0, conf.bits_per_seed);
        }
        for seed in 1u64..(1u64<<conf.bits_per_seed) {
            for (group_index, (best, new)) in best_errors.iter_mut().zip(conf.calc_errors(&kv, seed).iter()).enumerate() {
                if new < best {
                    *best = *new;
                    cells.set_bits(group_index * bits_per_group, seed, conf.bits_per_seed);
                }
            }
        }
        let mut result = Self {
            cells,
            bits_per_cell: conf.bits_per_cell,
            bits_per_seed: conf.bits_per_seed,
            bits_per_group: bits_per_group as u16,
            number_of_groups: conf.number_of_groups,
            cells_per_group: conf.cells_per_group,
            hasher: conf.hasher
        };
        for (k, v) in kv() {
            result.set(k.borrow(), v.into());
        }
        result
    }

    /// Calculate approximate number of bytes occupied by dynamic part of `self`.
    /// Same as `self.size_bytes() - std::mem::size_of_val(self)`.
    pub fn size_bytes_dyn(&self) -> usize {
        self.cells.len() * std::mem::size_of::<u64>()
    }

    /// Calculate approximate, total (including heap memory) number of bytes occupied by `self`.
    #[inline] pub fn size_bytes(&self) -> usize {
        std::mem::size_of_val(self) + self.size_bytes_dyn()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_min_db() {
        let mut db = HashMinPatternDB::new(100, 7);
        for i in 0..250 {
            db.set(&i, i/3 + 11);
        }
        for i in 0..250 {
            let from_db = db.get(&i);
            assert!(from_db >= 11);
            assert!(from_db <= i/3 + 11);
        }
        for i in 250..500 {
            let from_db = db.get(&i);
            assert!(11 <= from_db && from_db <= (250/3+11) || from_db == (1<<7)-1);
        }
    }

    #[test]
    fn hash_min_grouped_db() {
        let conf = HashMinGroupedPatternDBConf::new(
            10,
            8,
            7,
            2
        );
        let db = HashMinGroupedPatternDB::from_kv_conf(
            || (0u64..250).map(|i| (i, i/3+11)),
            conf
        );
        for i in 0..250 {
            let from_db = db.get(&i);
            assert!(from_db >= 11);
            assert!(from_db <= i/3 + 11);
        }
        for i in 250..500 {
            let from_db = db.get(&i);
            assert!(11 <= from_db && from_db <= (250/3+11) || from_db == (1<<7)-1);
        }
    }
}