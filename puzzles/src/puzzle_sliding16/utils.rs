pub const DENIED: u8 = u8::MAX;

/// Number of bits needed to store either tile number or its position (index of the board cell).
pub const BITS_PER_CELL: u8 = 4;

/// 0..01..1 mask with BITS_PER_CELL bits set.
pub const BITS_PER_CELL_MASK32: u32 = (1u32<<BITS_PER_CELL)-1;
pub const BITS_PER_CELL_MASK64: u64 = BITS_PER_CELL_MASK32 as u64;

pub const MAX_BOARD_SIZE: usize = 1<<BITS_PER_CELL;

