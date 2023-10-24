use std::hash::{BuildHasherDefault, BuildHasher};
use std::collections::hash_map::DefaultHasher;
use super::level_size_chooser::{LevelSizeChooser, OptimalLevelSize};

#[derive(Copy, Clone)]
pub struct Conf<LSC = OptimalLevelSize, S = BuildHasherDefault<DefaultHasher>> {
    pub level_size_chooser: LSC,
    pub hash_builder: S,
    pub bits_per_group_seed: u8,
    pub values_per_group: u8
}

impl Default for Conf<OptimalLevelSize, BuildHasherDefault<DefaultHasher>> {
    fn default() -> Self { Self::lsize_hash(Default::default(), Default::default()) }
}

impl Conf<OptimalLevelSize, BuildHasherDefault<DefaultHasher>> {
    pub fn bps(bits_per_group_seed: u8) -> Self {
        Self { bits_per_group_seed, ..Default::default() }
    }
    pub fn vpg(values_per_group: u8) -> Self {
        Self { values_per_group, ..Default::default() }
    }
    pub fn bps_vpg(bits_per_group_seed: u8, values_per_group: u8) -> Self {
        Self { bits_per_group_seed, values_per_group, ..Default::default() }
    }
}

impl<LSC: LevelSizeChooser> Conf<LSC, BuildHasherDefault<DefaultHasher>> {
    pub fn lsize(level_size_chooser: LSC) -> Self {
        Self::lsize_hash(level_size_chooser, Default::default())
    }
    pub fn lsize_bps(level_size_chooser: LSC, bits_per_group_seed: u8) -> Self {
        Self { bits_per_group_seed, ..Self::lsize(level_size_chooser) }
    }
    pub fn lsize_vpg(level_size_chooser: LSC, values_per_group: u8) -> Self {
        Self { values_per_group, ..Self::lsize(level_size_chooser) }
    }
    pub fn lsize_bps_vpg(level_size_chooser: LSC, bits_per_group_seed: u8, values_per_group: u8) -> Self {
        Self { bits_per_group_seed, values_per_group, ..Self::lsize(level_size_chooser) }
    }
}

impl<S: BuildHasher> Conf<OptimalLevelSize, S> {
    pub fn hash(hash_builder: S) -> Self {
        Self::lsize_hash(Default::default(), hash_builder)
    }
    pub fn hash_bps(hash_builder: S, bits_per_group_seed: u8) -> Self {
        Self { bits_per_group_seed, ..Self::hash(hash_builder) }
    }
    pub fn hash_vpg(hash_builder: S, values_per_group: u8) -> Self {
        Self { values_per_group, ..Self::hash(hash_builder) }
    }
    pub fn hash_bps_vpg(hash_builder: S, bits_per_group_seed: u8, values_per_group: u8) -> Self {
        Self { bits_per_group_seed, values_per_group, ..Self::hash(hash_builder) }
    }
}

impl<LSC: LevelSizeChooser, S: BuildHasher> Conf<LSC, S> {
    pub fn lsize_hash(level_size_chooser: LSC, hash_builder: S) -> Self { Self {
        level_size_chooser,
        hash_builder,
        bits_per_group_seed: 4,
        values_per_group: 30
    } }
    pub fn lsize_hash_bps(level_size_chooser: LSC, hash_builder: S, bits_per_group_seed: u8) -> Self {
        Self { bits_per_group_seed, ..Self::lsize_hash(level_size_chooser, hash_builder) }
    }
    pub fn lsize_hash_vpg(level_size_chooser: LSC, hash_builder: S, values_per_group: u8) -> Self {
        Self { values_per_group, ..Self::lsize_hash(level_size_chooser, hash_builder) }
    }
    pub fn lsize_hash_bps_vpg(level_size_chooser: LSC, hash_builder: S, bits_per_group_seed: u8, values_per_group: u8) -> Self {
        Self { bits_per_group_seed, values_per_group, ..Self::lsize_hash(level_size_chooser, hash_builder) }
    }
}

/// Returns number of bits needed to store the group,
/// i.e. its seed (using `bits_per_group_seed` bits) and values (using `2*values_per_group` bits).
#[inline] pub fn total_bits_per_group(bits_per_group_seed: u8, values_per_group: u8) -> u8 {
    bits_per_group_seed + 2*values_per_group
}