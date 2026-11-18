fn positive(w: f32) -> f32 {
    if w > 0.0 { w } else { 0.0 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut rng = Rng {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        };
        rng.next_u64();
        rng
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0)
    }

    pub fn range(&mut self, span: [f32; 2]) -> f32 {
        span[0] + (span[1] - span[0]) * self.unit()
    }

    pub fn chance(&mut self, share: f32) -> bool {
        self.unit() < share
    }

    pub fn below(&mut self, count: usize) -> usize {
        if count == 0 {
            return 0;
        }
        (self.next_u64() % count as u64) as usize
    }

    pub fn pick_weighted(&mut self, weights: &[f32]) -> usize {
        let total: f32 = weights.iter().map(|&w| positive(w)).sum();
        if total <= 0.0 {
            return 0;
        }
        let mut roll = self.unit() * total;
        for (index, weight) in weights.iter().enumerate() {
            let weight = positive(*weight);
            if roll < weight {
                return index;
            }
            roll -= weight;
        }
        weights.len() - 1
    }
}
