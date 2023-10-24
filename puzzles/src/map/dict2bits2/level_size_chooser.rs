use bitm::{ceiling_div, BitAccess};
use fsum::FSum;
use crate::fp::level_size_chooser::positive_collisions_prob;

/// Chooses the size of level for the given level input.
pub trait LevelSizeChooser {

    /// Returns number of groups (each includes `values_per_group` 2-bit values)
    /// to use for given level input (2-bit values bit vector of size `input_size`).
    fn size_groups(&self, _values: &[u64], input_size: usize, values_per_group: u8) -> usize {
        self.max_size_groups(input_size, values_per_group)
    }

    /// Returns maximal number of groups (each includes `values_per_group` 2-bit values)
    /// that can be returned by `size_groups` for given input (`input_size` and `values_per_group`).
    fn max_size_groups(&self, input_size: usize, values_per_group: u8) -> usize;
}

/// Choose level size as a percent of the input size.
#[derive(Copy, Clone)]
pub struct ProportionalLevelSize {
    percent: u8
}

impl ProportionalLevelSize {
    pub fn with_percent(percent: u8) -> Self { Self{percent} }
}

impl Default for ProportionalLevelSize {
    fn default() -> Self { Self::with_percent(90) } // TODO what is the best default?
}

impl LevelSizeChooser for ProportionalLevelSize {
    fn max_size_groups(&self, input_size: usize, values_per_group: u8) -> usize {
        ceiling_div(input_size * self.percent as usize, values_per_group as usize * 100)
    }
}


/// Chooses optimal level size considering distribution of incidence of values.
#[derive(Default, Copy, Clone)]
pub struct OptimalLevelSize;

impl OptimalLevelSize {
    fn size_groups_for_dist(counts: &mut [u32], input_size: usize, values_per_group: u8) -> usize {
        let mut result = ceiling_div(input_size, values_per_group as _);
        if result == 1 { return 1; }
        let positive_collisions_p = positive_collisions_prob(counts, input_size);
        let mut result_eval = f64::MAX;
        while result >= 1 {
            let result_bits = (result * values_per_group as usize) as f64;
            let numerator =  2f64 + 64f64 / result_bits; // (values bits + metadata) / result_bits

            let mut denominator = FSum::new();
            let lambda = input_size as f64 / result_bits;
            let mut lambda_to_power_k = lambda;
            let mut k_factorial = 1u64;
            for i in 0usize..16 {
                let k = i as u32 + 1;
                k_factorial *= k as u64;
                let pk = positive_collisions_p[i] * lambda_to_power_k * (-lambda).exp() / k_factorial as f64;
                lambda_to_power_k *= lambda;
                denominator += pk * k as f64;
            }
            let new_result_eval = numerator / denominator.value();
            if new_result_eval >= result_eval {  // impossible in the first iteration
                return result + 1;
            }
            result_eval = new_result_eval;
            result -= 1;
        }
        1
    }
}

impl LevelSizeChooser for OptimalLevelSize {
    fn size_groups(&self, values: &[u64], input_size: usize, values_per_group: u8) -> usize {
        let mut counts = [0u32; 3];
        for i in 0..input_size { counts[values.get_fragment(i, 2) as usize] += 1; }
        Self::size_groups_for_dist(&mut counts, input_size, values_per_group)
    }

    fn max_size_groups(&self, input_size: usize, values_per_group: u8) -> usize {
        ceiling_div(input_size, values_per_group as _)
    }
}