use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct Stepper {
    straight: f32,
    diagonal: f32,
    scale: f32,
}

impl Stepper {
    pub(super) fn new(spacing: f32, max_slope: f32) -> Self {
        Self {
            straight: max_slope * spacing,
            diagonal: max_slope * (spacing * std::f32::consts::SQRT_2),
            scale: STRAIGHT as f32 / spacing,
        }
    }

    pub(super) fn cost(&self, from: f32, to: f32, diagonal: bool) -> Option<u32> {
        let rise = (to - from).abs();
        let (most, run) = if diagonal {
            (self.diagonal, DIAGONAL)
        } else {
            (self.straight, STRAIGHT)
        };
        (rise <= most).then(|| run + (rise * self.scale).round() as u32)
    }
}

pub(super) fn octile(a: Sample, b: Sample) -> u64 {
    let (x, z) = ((a[0] - b[0]).unsigned_abs(), (a[1] - b[1]).unsigned_abs());
    let (long, short) = if x > z { (x, z) } else { (z, x) };
    u64::from(STRAIGHT) * (long - short) + u64::from(DIAGONAL) * short
}

const WINDOW: usize = 1 << 13;

pub(super) struct Buckets {
    lists: Vec<Vec<u32>>,
    at: u64,
    len: usize,
}

impl Default for Buckets {
    fn default() -> Self {
        Self {
            lists: vec![Vec::new(); WINDOW],
            at: 0,
            len: 0,
        }
    }
}

impl Buckets {
    pub(super) fn restart(&mut self, first: u64) {
        for list in &mut self.lists {
            list.clear();
        }
        self.len = 0;
        self.at = first;
    }

    pub(super) fn push(&mut self, key: u64, item: u32) {
        debug_assert!(
            key >= self.at && key - self.at < WINDOW as u64,
            "a key outside the queue's window"
        );
        self.lists[(key % WINDOW as u64) as usize].push(item);
        self.len += 1;
    }

    pub(super) fn pop(&mut self) -> Option<(u64, u32)> {
        if self.len == 0 {
            return None;
        }
        loop {
            if let Some(item) = self.lists[(self.at % WINDOW as u64) as usize].pop() {
                self.len -= 1;
                return Some((self.at, item));
            }
            self.at += 1;
        }
    }
}

#[derive(Default)]
pub(super) struct Flood {
    pub(super) cost: Vec<u32>,
    queue: Buckets,
    targets: Vec<bool>,
    left: usize,
}

impl Flood {
    pub(super) fn start(&mut self, start: usize, targets: &[usize]) {
        self.cost.clear();
        self.cost.resize(SIDE * SIDE, UNREACHED);
        self.targets.clear();
        self.targets.resize(SIDE * SIDE, false);
        self.left = 0;
        for &target in targets {
            if !self.targets[target] {
                self.targets[target] = true;
                self.left += 1;
            }
        }
        self.queue.restart(0);
        self.cost[start] = 0;
        self.queue.push(0, start as u32);
    }

    pub(super) fn step(
        &mut self,
        ground: &WalkGround,
        stands: &[bool],
        stepper: &Stepper,
        (budget, searched): (&mut u64, &mut usize),
    ) -> bool {
        while *budget > 0 {
            if self.left == 0 {
                return true;
            }
            let Some((cost, sample)) = self.queue.pop() else {
                return true;
            };
            let sample = sample as usize;
            if u64::from(self.cost[sample]) != cost {
                continue;
            }
            if std::mem::replace(&mut self.targets[sample], false) {
                self.left -= 1;
            }
            *budget = budget.saturating_sub(EXPAND_COST);
            *searched += 1;
            let (x, z) = ((sample % SIDE) as i64, (sample / SIDE) as i64);
            let inside = |x: i64, z: i64| {
                (x >= 0 && z >= 0 && x < SIDE as i64 && z < SIDE as i64)
                    .then(|| z as usize * SIDE + x as usize)
            };
            for (index, &(dx, dz)) in NEIGHBORS.iter().enumerate() {
                let Some(next) = inside(x + dx, z + dz).filter(|&next| stands[next]) else {
                    continue;
                };
                let diagonal = index >= 4;
                if diagonal
                    && !(inside(x + dx, z).is_some_and(|side| stands[side])
                        && inside(x, z + dz).is_some_and(|side| stands[side]))
                {
                    continue;
                }
                let Some(step) =
                    stepper.cost(ground.heights[sample], ground.heights[next], diagonal)
                else {
                    continue;
                };
                let reached = self.cost[sample].saturating_add(step);
                if reached < self.cost[next] {
                    self.cost[next] = reached;
                    self.queue.push(u64::from(reached), next as u32);
                }
            }
        }
        false
    }
}
