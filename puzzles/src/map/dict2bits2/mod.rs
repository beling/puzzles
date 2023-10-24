use std::hash::{BuildHasherDefault, BuildHasher, Hash};
use std::collections::hash_map::DefaultHasher;
use crate::stats;
use bitm::{BitAccess, BitVec, ceiling_div};
use crate::fp::CollisionSolver;
use crate::read_array;
use crate::fp::collision_solver::CountPositiveCollisions;

mod level_size_chooser;
mod conf;

use level_size_chooser::LevelSizeChooser;
use dyn_size_of::GetSize;
use std::io;
use binout::{read_int, write_int};

// TODO special version of GOFPMap with bits_per_group_seed=4 and values_per_group=30
/// Maps keys to values in range 0-2 (i.e. modulo 3).
pub struct Dict2Bits2<S = BuildHasherDefault<DefaultHasher>> {
    values: Box<[u64]>,
    //group_seeds: Box<[u64]>,   //  Box<[u8]>,
    level_size: Box<[u32]>, // number of groups
    hash_builder: S,
    /// always even
    bits_per_group_seed: u8,
    /// number of 2-bit values in each group
    values_per_group: u8,
    /// total size of the group in bits, equals to bits_per_group_seed + 2*values_per_group
    total_bits_per_group: u8    // u16?
}

impl<S> GetSize for Dict2Bits2<S> {
    fn size_bytes_dyn(&self) -> usize {
        self.values.size_bytes_dyn() + self.level_size.size_bytes_dyn()
    }
    const USES_DYN_MEM: bool = true;
}

impl<S: BuildHasher> Dict2Bits2<S> {

    /// Calculates bit index assigned to the key hashed by the `hasher`.
    #[inline] fn value(&self, hasher: S::Hasher, group: u32) -> u8 {
        let gr_begin = group as usize * self.total_bits_per_group as usize;
        let group_seed = self.values.get_bits(gr_begin, self.bits_per_group_seed) as u8;
        let in_gr_index = in_group_index_with_size(hasher, group_seed, self.values_per_group) as usize;
        self.values.get_bits(gr_begin + self.bits_per_group_seed as usize + 2*in_gr_index, 2) as u8
    }

    /// Gets the value associated with the given key `key` and reports statistics to `access_stats`.
    pub fn get_stats<K: Hash, A: stats::AccessStatsCollector>(&self, key: &K, access_stats: &mut A) -> Option<u8> {
        let mut groups_before = 0u32;
        let mut level_nr = 0u32;
        loop {
            let level_size_groups = *self.level_size.get(level_nr as usize)?;
            let mut hasher = self.hash_builder.build_hasher();
            let group = groups_before + group_nr_at_level(&mut hasher, key, level_nr, level_size_groups);
            let v = self.value(hasher, group);
            if v != 3 {
                access_stats.found_on_level(level_nr);
                return Some(v);
            }
            groups_before += level_size_groups;
            level_nr += 1;
        }
    }

    /// Gets the value associated with the given key `k`.
    #[inline(always)] pub fn get<K: Hash>(&self, k: &K) -> Option<u8> {
        self.get_stats(k, &mut ())
    }

    /// Returns number of group and index inside the group for the `key`.
    fn in_level_index<K, LSC, GetGroupSeed>(key: &K, conf: &conf::Conf<LSC, S>, level_size_groups: u32, level_nr: u32, group_seed: &GetGroupSeed) -> (u32, u8)
        where K: Hash, GetGroupSeed: Fn(u32) -> u8
    {
        let mut hasher = conf.hash_builder.build_hasher();
        let group = group_nr_at_level(&mut hasher, &key, level_nr, level_size_groups);
        let in_gr_index = in_group_index_with_size(hasher, group_seed(group), conf.values_per_group) as u8;
        (group, in_gr_index)
    }

    fn consider_all<K, LSC, GetGroupSeed, CS>(conf: &conf::Conf<LSC, S>,
                                              keys: &[K], values: &[u64],
                                              level_size_groups: u32, level_nr: u32,
                                              group_seed: GetGroupSeed, collision_solver: &mut CS)
        where K: Hash, GetGroupSeed: Fn(u32) -> u8, CS: CollisionSolver  // returns group seed for group with given index
    {
        for i in 0..keys.len() {
            let (group, in_gr_index) = Self::in_level_index(&keys[i], conf, level_size_groups, level_nr, &group_seed);
            let index = (group as usize * conf.values_per_group as usize)+in_gr_index as usize;
            if collision_solver.is_under_collision(index) { continue }
            collision_solver.process_fragment(index,values.get_fragment(i, 2) as u8, 2);
        }
    }

    fn count_collisions_in_groups<K, LSC>(conf: &conf::Conf<LSC, S>,
                                          keys: &[K], values: &[u64],
                                          level_size_groups: u32, level_nr: u32, group_seed: u8) -> Box<[u8]>
        where K: Hash
    {
        let mut collision_solver = CountPositiveCollisions::new(level_size_groups as usize * conf.values_per_group as usize);
        Self::consider_all(conf, keys, values, level_size_groups, level_nr, |_| group_seed, &mut collision_solver);
        collision_solver.positive_collisions_of_groups(conf.values_per_group, 2)
    }

    /// Build `Dict2Bits2` for given `keys` -> `values` map, where:
    /// - `keys` are given directly
    /// - `values` are encoded as bit vector (2-bits/value)
    /// The arrays must be of the same length.
    pub fn with_slice_bitvec_conf_stats<K, LSC, BS>(
        keys: &mut [K], values: &mut [u64],
        conf: conf::Conf<LSC, S>, stats: &mut BS) -> Self
        where K: Hash,
              LSC: LevelSizeChooser,
              BS: stats::BuildStatsCollector
    {
        let mut level_size = Vec::<u32>::new();
        let mut out_values = Vec::<u64>::new();
        let mut total_size_bits = 0usize;
        let mut input_size = keys.len();
        let mut level_nr = 0u32;
        let total_bits_per_group = conf::total_bits_per_group(conf.bits_per_group_seed, conf.values_per_group);
        let max_seed = ((1u16 << conf.bits_per_group_seed) - 1) as u8;
        while input_size != 0 {
            let in_keys = &keys[0..input_size];
            let level_size_groups = conf.level_size_chooser.size_groups(values, input_size, conf.values_per_group);
            let level_size_bits = level_size_groups * total_bits_per_group as usize;
            //let level_size = level_size_segments as usize * 64;
            stats.level(input_size, level_size_bits);
            out_values.resize(ceiling_div(total_size_bits + level_size_bits, 64), u64::MAX);
            let mut best_counts = Self::count_collisions_in_groups(&conf, in_keys, values,
                                                                   level_size_groups as u32,
                                                                   level_nr, max_seed);
            for group_seed in 0..max_seed {
                let with_new_seed = Self::count_collisions_in_groups(&conf, in_keys, values,
                                                                     level_size_groups as u32,
                                                                     level_nr, group_seed);
                for group in 0..level_size_groups {
                    let new = with_new_seed[group];
                    let best = &mut best_counts[group];
                    if new > *best {
                        *best = new;
                        out_values.set_bits(total_size_bits + group as usize * total_bits_per_group as usize,
                                            group_seed as _, conf.bits_per_group_seed);
                    }
                }
            }

            let mut collided = Box::<[u64]>::with_zeroed_bits(level_size_groups * conf.values_per_group as usize);
            for i in 0..in_keys.len() {
                let mut hasher = conf.hash_builder.build_hasher();
                let group = group_nr_at_level(&mut hasher, &in_keys[i], level_nr, level_size_groups as u32);
                let gr_begin = total_size_bits + group as usize * total_bits_per_group as usize;
                let in_gr_index = in_group_index_with_size(hasher,
                                                           out_values.get_bits(gr_begin, conf.bits_per_group_seed) as u8,
                                                           conf.values_per_group) as u8;
                let collided_index = group as usize * conf.values_per_group as usize + in_gr_index as usize;
                if !collided.get_bit(collided_index) {
                    let new = values.get_fragment(i, 2);
                    out_values.conditionally_change_bits(
                        |old| {
                            if old == 3 { Some(new) } else {
                                if old == new { None } else {
                                    collided.set_bit(collided_index);
                                    Some(3)
                                }
                            }
                        },
                        gr_begin + conf.bits_per_group_seed as usize + 2 * in_gr_index as usize, 2);
                }
            }
            drop(collided);

            let mut i = 0usize;
            while i < input_size {
                let mut hasher = conf.hash_builder.build_hasher();
                let group = group_nr_at_level(&mut hasher, &keys[i], level_nr, level_size_groups as u32);
                let gr_begin = total_size_bits + group as usize * total_bits_per_group as usize;
                let in_gr_index = in_group_index_with_size(hasher,
                                                           out_values.get_bits(gr_begin, conf.bits_per_group_seed) as u8,
                                                           conf.values_per_group);
                if out_values.get_bits(gr_begin + conf.bits_per_group_seed as usize + 2 * in_gr_index as usize, 2) == 3 {   // collision
                    i += 1;
                } else {    // no collision
                    input_size -= 1;
                    keys.swap(i, input_size);
                    values.swap_fragments(i, input_size, 2);
                    // stats.value_on_level(level_nr); // TODO do we need this? we can get average levels from lookups
                }
            }
            level_size.push(level_size_groups as u32);
            level_nr += 1;
            total_size_bits += level_size_bits;
        }
        stats.end();
        Self {
            values: out_values.into_boxed_slice(),
            level_size: level_size.into_boxed_slice(),
            hash_builder: conf.hash_builder,
            bits_per_group_seed: conf.bits_per_group_seed,
            values_per_group: conf.values_per_group,
            total_bits_per_group
        }
    }

    pub fn with_slice_bitvec_conf<K, LSC>(keys: &mut [K], values: &mut [u64], conf: conf::Conf<LSC, S>) -> Self
            where K: Hash, LSC: LevelSizeChooser //LSC: LevelSizeChooser,
    {
        Self::with_slice_bitvec_conf_stats(keys, values, conf, &mut ())
    }

    /// Returns number of bytes which `write` will write.
    pub fn write_bytes(&self) -> usize {
        2*std::mem::size_of::<u8>()
            + std::mem::size_of::<u32>()
            + self.level_size.size_bytes_dyn()
            + self.values.size_bytes_dyn()
    }

    /// Writes `self` to the `output`.
    pub fn write(&self, output: &mut dyn io::Write) -> io::Result<()>
    {
        write_int!(output, self.bits_per_group_seed)?;
        write_int!(output, self.values_per_group)?;
        write_int!(output, self.level_size.len() as u32)?;
        self.level_size.iter().try_for_each(|l| { write_int!(output, l) })?;
        self.values.iter().try_for_each(|l| { write_int!(output, l) })
    }

    /// Reads `Self` from the `input`.
    /// Hasher must be the same as the one used to write.
    pub fn read_with_hasher(input: &mut dyn io::Read, hash_builder: S) -> io::Result<Self>
    {
        let bits_per_group_seed = read_int!(input, u8)?;
        let values_per_group = read_int!(input, u8)?;
        let level_size = read_array!([u32; read u32] from input).into_boxed_slice();
        let number_of_groups = level_size.iter().map(|v|*v as usize).sum::<usize>();
        let total_bits_per_group = conf::total_bits_per_group(bits_per_group_seed, values_per_group);
        let values = read_array!(total_bits_per_group as usize * number_of_groups as usize; bits from input).into_boxed_slice();

        Ok(Self {
            values,
            level_size,
            hash_builder,
            bits_per_group_seed,
            values_per_group,
            total_bits_per_group
        })
    }
}

impl Dict2Bits2 {
    pub fn with_slice_bitvec<K: Hash>(keys: &mut [K], values: &mut [u64]) -> Self {
        Self::with_slice_bitvec_conf_stats(keys, values, conf::Conf::default(), &mut ())
    }

    /// Reads `Self` from the `input`.
    /// Only `Dict2Bits2`s that use default hasher can be read by this method.
    pub fn read(input: &mut dyn io::Read) -> io::Result<Self> {
        Self::read_with_hasher(input, Default::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_read_write(d: &Dict2Bits2) {
        let mut buff = Vec::new();
        d.write(&mut buff).unwrap();
        assert_eq!(buff.len(), d.write_bytes());
        let read = Dict2Bits2::read(&mut &buff[..]).unwrap();
        assert_eq!(d.level_size, read.level_size);
        assert_eq!(d.values, read.values);
        assert_eq!(d.bits_per_group_seed, read.bits_per_group_seed);
        assert_eq!(d.values_per_group, read.values_per_group);
        assert_eq!(d.total_bits_per_group, read.total_bits_per_group);
    }

    fn test_dict2bits2_invariants(d: &Dict2Bits2) {
        let number_of_groups = d.level_size.iter().map(|v| *v as usize).sum::<usize>();
        assert_eq!(ceiling_div(number_of_groups * d.total_bits_per_group as usize, 64), d.values.len());
    }

    #[test]
    fn with_slice_bitvec() {
        let mut keys: Box<[u32]> = (11..111).collect();
        let mut values = Box::<[u64]>::with_zeroed_bits(keys.len()*2);
        for i in 0..keys.len() { values.set_fragment(i, (i%3) as u64, 2); }
        let d = Dict2Bits2::with_slice_bitvec(&mut keys, &mut values);
        for i in 0..keys.len() {
            assert_eq!(d.get(&keys[i]), Some(values.get_fragment(i, 2) as u8));
        }
        test_read_write(&d);
        test_dict2bits2_invariants(&d);
    }
}