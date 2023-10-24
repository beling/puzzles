use std::hash::{BuildHasherDefault, BuildHasher, Hasher, Hash};
use std::collections::hash_map::DefaultHasher;
use bitm::ceiling_div;
use ph::stats::{AccessStatsCollector, BuildStatsCollector};
use std::borrow::Borrow;

pub struct Conf<S = BuildHasherDefault<DefaultHasher>> {
    hash_builder: S,
}

pub struct Level {
    values: Box<[u64]>
}

impl Level {

    fn with_segments(len: usize) -> Self {
        Self { values: vec![u64::MAX; len].into_boxed_slice() }
    }

    /// Returns number of values.
    #[inline(always)] fn len(&self) -> usize {
        self.values.len()*32
    }

    /// Calculates approximate number of bytes occupied by dynamic part of `self`.
    fn size_bytes_dyn(&self) -> usize {
        self.values.len() * std::mem::size_of::<u64>()
    }

    fn index(&self, hash_builder: &impl BuildHasher, key: &impl Hash, level_nr: u32) -> usize {
        let mut hasher = hash_builder.build_hasher();
        hasher.write_u32(level_nr);
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.len()
    }

    #[inline(always)] fn shift(index: usize) -> usize { 2 * (index % 32) }

    fn get(&self, index: usize) -> u8 {
        ((self.values[index / 32] >> Self::shift(index)) & 3) as u8
    }

    /*fn set(&mut self, index: usize, value: u8) {
        let cell = &mut self.values[index / 32];
        let shift = Self::shift(index);
        *cell &= !(3 << shift);
        *cell |= (value as u64) << shift;
    }*/

    /// Try to init `index`-th 2-bit cell to `value`.
    /// The cell can be inited (then `true` is returned) only if it equals `3` or `value`.
    /// Otherwise `false` is returned and the cell is set to `3`.
    fn try_init(&mut self, index: usize, value: u8) -> bool {
        let cell = &mut self.values[index / 32];
        let shift = Self::shift(index);
        let old_value = ((*cell >> shift) & 3) as u8;
        if old_value == value {
            true
        } else if old_value == 3 {
            *cell &= !(3 << shift);
            *cell |= (value as u64) << shift;
            true
        } else {    // collision
            *cell |= 3 << shift;  // mark as uninit again
            false
        }
    }

    fn get_key(&self, hash_builder: &impl BuildHasher, key: &impl Hash, level_nr: u32) -> u8 {
        self.get(self.index(hash_builder, key, level_nr))
    }
}

pub struct FPDict2bits<S = BuildHasherDefault<DefaultHasher>> {
    levels: Box<[Level]>,
    hash_builder: S,
}

impl<S: BuildHasher> FPDict2bits<S> {
    /// Calculates approximate number of bytes occupied by dynamic part of `self`.
    /// Same as `self.size_bytes() - std::mem::size_of_val(self)`.
    pub fn size_bytes_dyn(&self) -> usize {
        self.levels.iter().map(|l| l.size_bytes_dyn()).sum()
    }

    /// Calculates approximate, total (including heap memory) number of bytes occupied by `self`.
    pub fn size_bytes(&self) -> usize {
        std::mem::size_of_val(self) + self.size_bytes_dyn()
    }

    /// Gets the value associated with the given `key` and reports statistics to `access_stats`.
    pub fn get_stats<K: Hash, A: AccessStatsCollector>(&self, key: &K, access_stats: &mut A) -> Option<u8> {
        let mut level_nr = 0u32;
        loop {
            let v = self.levels.get(level_nr as usize)?.get_key(&self.hash_builder, key, level_nr);
            if v != 3 {
                access_stats.found_on_level(level_nr);
                return Some(v);
            }
            level_nr += 1;
        }
    }

    /// Gets the value associated with the given `key`.
    pub fn get<K: Hash>(&self, key: &K) -> Option<u8> {
        self.get_stats(key, &mut ())
    }

    fn process_kv(conf: &Conf<S>, collisions: &mut [u64], level: &mut Level, level_nr: u32, key: &impl Hash, value: u8) {
        let index = level.index(&conf.hash_builder, key, level_nr);
        let collision_index = index / 64;
        let collision_bit = 1u64 << (index % 64);
        if collisions[collision_index] & collision_bit == 0 {    // was no collision
            if !level.try_init(index, value) {
                collisions[collision_index] |= collision_bit;
            }
        }
    }

    /// Builds `BBDict2bits` for given `keys` and `values`.
    pub fn from_slices_conf_stats<K, BS>(
        keys: &mut [K],
        values: &mut [u8],  // TODO change to vector of 2 bits
        conf: Conf<S>,
        stats: &mut BS
    ) -> Self
        where K: Hash, BS: BuildStatsCollector
    {
        let mut levels = Vec::<Level>::new();
        let mut input_size = keys.len();
        let mut level_nr = 0u32;
        while input_size != 0 {
            let mut collisions = vec![0u64; ceiling_div(input_size, 64)].into_boxed_slice();
            let mut level = Level::with_segments(collisions.len()*2);
            let level_size = collisions.len() * 64;
            stats.level(input_size, level_size);
            let in_keys = &mut keys[0..input_size];
            let in_values = &mut values[0..input_size];
            for (key, value) in in_keys.iter().zip(in_values.iter()) {
                Self::process_kv(&conf, &mut collisions, &mut level, level_nr, key, *value);
            }
            let mut i = 0usize;
            while i < input_size {
                let index = level.index(&conf.hash_builder, &in_keys[i], level_nr);
                if (collisions[index / 64] & (1u64 << (index % 64))) == 0 { // no collision
                    input_size -= 1;
                    in_keys.swap(i, input_size);
                    in_values.swap(i, input_size);
                    // stats.value_on_level(level_nr); // TODO do we need this? we can get average levels from lookups
                } else {
                    i += 1;
                }
            }
            levels.push(level);
            level_nr += 1;
        }
        stats.end();
        Self {
            levels: levels.into_boxed_slice(),
            hash_builder: conf.hash_builder
        }
    }

    pub fn from_iter_conf_stats<K, V, KBorrow, KVIntoIterator, KVGetter, BS>(kv: KVGetter, mut input_size: usize, conf: Conf<S>, stats: &mut BS) -> Self
        where
            K: Hash,
            V: Into<u8>,
            KBorrow: Borrow<K>,
            KVIntoIterator: IntoIterator<Item=(KBorrow, V)>,    // Iterator over keys and values
            KVGetter: Fn() -> KVIntoIterator,      // Returns iterator over keys and values
            BS: BuildStatsCollector
    {
        let mut levels = Vec::<Level>::new();
        let mut level_nr = 0u32;
        while input_size != 0 {
            let mut collisions = vec![0u64; ceiling_div(input_size, 64)].into_boxed_slice();
            let mut level = Level::with_segments(collisions.len()*2);
            let level_size = collisions.len() * 64;
            stats.level(input_size, level_size);
            let accept_only_new = |(k, _v): &(KBorrow, V)| levels.iter()
                .enumerate()
                .all(|(lnr, l)| l.get_key(&conf.hash_builder, k.borrow(), lnr as u32) == 3);
            for (key, value) in kv().into_iter().filter(accept_only_new) {
                Self::process_kv(&conf, &mut collisions, &mut level, level_nr, key.borrow(), value.into());
            }
            for (key, _) in kv().into_iter().filter(accept_only_new) {
                let index = level.index(&conf.hash_builder, key.borrow(), level_nr);
                if (collisions[index / 64] & (1u64 << (index % 64))) == 0 { // no collision
                    // stats.value_on_level(level_nr); // TODO do we need this? we can get average levels from lookups
                    input_size -= 1;
                }
            }
            levels.push(level);
            level_nr += 1;
        }
        stats.end();
        Self {
            levels: levels.into_boxed_slice(),
            hash_builder: conf.hash_builder
        }
    }

}