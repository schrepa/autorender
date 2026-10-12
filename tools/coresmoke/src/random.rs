//! Seeded pseudo-random numbers (SplitMix64): the same seed gives the same
//! edits on every machine.

pub struct Random(u64);

impl Random {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A number in `0..count` (`count` above 0).
    pub fn below(&mut self, count: usize) -> usize {
        (self.next() % count as u64) as usize
    }

    /// Whether a chance of `percent` in 100 came up.
    pub fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }

    /// A number in `low..high`.
    pub fn range(&mut self, low: f32, high: f32) -> f32 {
        let unit = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        low + (high - low) * unit
    }
}
