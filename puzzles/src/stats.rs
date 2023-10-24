use std::ops::AddAssign;

/// Search statistic collector.
/// It collects data during IDA* search.
pub trait SearchStatsCollector {
    /// Called for each state visited, can return false to cancel search process.
    #[inline(always)] fn leaf(&mut self) -> bool { true }
    #[inline(always)] fn internal(&mut self) { }
}

/// Search statistic collector that ignore all events.
impl SearchStatsCollector for () {}

impl SearchStatsCollector for u64 {
    #[inline(always)] fn leaf(&mut self) -> bool { *self += 1; true }
    #[inline(always)] fn internal(&mut self) { *self += 1; }
}

#[derive(Default, Copy, Clone)]
pub struct SearchAllStats {
    pub internal: u64,
    pub leaves: u64
}

impl SearchAllStats {
    pub fn visits(&self) -> u64 { self.internal + self.leaves }
}

impl AddAssign for SearchAllStats {
    fn add_assign(&mut self, rhs: Self) {
        self.internal += rhs.internal;
        self.leaves += rhs.leaves;
    }
}

impl SearchStatsCollector for SearchAllStats {
    #[inline(always)] fn leaf(&mut self) -> bool { self.internal += 1; true }
    #[inline(always)] fn internal(&mut self) { self.leaves += 1; }
}

pub struct Limited {
    pub internal: u64,
    pub leaves: u64,
    pub limit: u64
}

impl Limited {
    pub fn with_limit(limit: u64) -> Self { Self{internal: 0, leaves: 0, limit} }

    pub fn reset_visits(&mut self) { self.internal = 0; self.leaves = 0; }

    pub fn reset_limit(&mut self, limit: u64) { self.reset_visits(); self.limit = limit; }

    pub fn visits(&self) -> u64 { self.internal + self.leaves }
}

impl SearchStatsCollector for Limited {
    #[inline(always)] fn leaf(&mut self) -> bool {
        if self.visits() >= self.limit { return false; }
        self.leaves += 1;
        true
    }

    #[inline(always)] fn internal(&mut self) { self.internal += 1; }
}