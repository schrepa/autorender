use engine_math::{max, sin_cos};

pub const SUN_STEPS: usize = 10;
pub const VIEW_STEPS: usize = 10;
pub const TURN_STEPS: usize = 5;
pub const WORDS: usize = SUN_STEPS * VIEW_STEPS * TURN_STEPS;
pub const UNIT: u32 = 4096;
pub const MOST: u32 = 0xffff;

const HEADINGS: usize = 8;
const SUMMED: usize = 1 << 16;
const DIAGONAL: f32 = std::f32::consts::FRAC_1_SQRT_2;
const TOWARD: [[f32; 2]; HEADINGS] = [
    [1.0, 0.0],
    [DIAGONAL, DIAGONAL],
    [0.0, 1.0],
    [-DIAGONAL, DIAGONAL],
    [-1.0, 0.0],
    [-DIAGONAL, -DIAGONAL],
    [0.0, -1.0],
    [DIAGONAL, -DIAGONAL],
];
const STEPS: [[i32; 2]; HEADINGS / 2] = [[1, 0], [1, 1], [0, 1], [-1, 1]];
const NEAREST_SEEN: f32 = 1.0e-4;

pub struct Ground {
    pub width: usize,
    pub height: usize,
    pub heights: Vec<u8>,
    pub tile: f32,
    pub depth: f32,
    pub roughness: f32,
}

#[derive(Clone, Copy, Default)]
struct Summed {
    upright: f32,
    along: [f32; HEADINGS / 2],
    horizons: [f32; HEADINGS],
}

#[derive(Clone, Copy, Default)]
struct Node {
    seen_light: f64,
    drawn_light: f64,
    seen_shine: f64,
    drawn_shine: f64,
}

#[derive(Clone, Copy)]
struct Height {
    sine: f32,
    cosine: f32,
    tangent: f32,
}

pub struct Roughening {
    ground: Ground,
    stride: usize,
    columns: usize,
    summed: Vec<Summed>,
    nodes: Vec<Node>,
    swept: usize,
    lit: usize,
}

fn heights_of(steps: usize) -> Vec<Height> {
    (0..steps)
        .map(|step| {
            let root = (step + 1) as f32 / steps as f32;
            let (sine, cosine) = if step + 1 == steps {
                (1.0, 0.0)
            } else {
                sin_cos(root * root * std::f32::consts::FRAC_PI_2)
            };
            let tangent = if cosine > 0.0 {
                sine / cosine
            } else {
                f32::INFINITY
            };
            Height {
                sine,
                cosine,
                tangent,
            }
        })
        .collect()
}

fn mixed(cell: usize) -> usize {
    let mut hash = (cell as u32).wrapping_mul(0x9e37_79b9);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x85eb_ca6b);
    hash ^= hash >> 13;
    hash as usize
}

fn divisor(a: usize, b: usize) -> usize {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl Roughening {
    pub fn new(ground: Ground) -> Self {
        let texels = ground.width * ground.height;
        let mut stride = 1;
        while texels / (stride * stride) > SUMMED {
            stride += 1;
        }
        let columns = ground.width / stride;
        let rows = ground.height / stride;
        Self {
            ground,
            stride,
            columns,
            summed: vec![Summed::default(); columns * rows],
            nodes: vec![Node::default(); WORDS],
            swept: 0,
            lit: 0,
        }
    }

    fn flat(&self) -> bool {
        let first = self.ground.heights.first().copied().unwrap_or(0);
        self.ground.depth <= 0.0
            || self.ground.tile <= 0.0
            || self.summed.is_empty()
            || self.ground.heights.iter().all(|&height| height == first)
    }

    pub fn cost(&self) -> u64 {
        (self.ground.width * self.ground.height) as u64
    }

    pub fn advance(&mut self) -> bool {
        if self.swept == 0 && self.flat() {
            return true;
        }
        if self.swept < HEADINGS / 2 {
            if self.swept == 0 {
                self.face();
            }
            self.sweep(self.swept);
            self.swept += 1;
            return false;
        }
        if self.lit < HEADINGS {
            self.light(self.lit);
            self.lit += 1;
        }
        self.lit == HEADINGS
    }

    fn spacing(&self) -> [f32; 2] {
        [
            self.ground.tile / self.ground.width as f32,
            self.ground.tile / self.ground.height as f32,
        ]
    }

    fn lift(&self, x: usize, y: usize) -> f32 {
        f32::from(self.ground.heights[y * self.ground.width + x]) * (self.ground.depth / 255.0)
    }

    fn summed_at(&self, x: usize, y: usize) -> Option<usize> {
        let (column, row) = (x / self.stride, y / self.stride);
        if column >= self.columns || row * self.columns >= self.summed.len() {
            return None;
        }
        let at = row * self.columns + column;
        let picked = mixed(at) % (self.stride * self.stride);
        (x % self.stride == picked % self.stride && y % self.stride == picked / self.stride)
            .then_some(at)
    }

    fn face(&mut self) {
        let (width, height) = (self.ground.width, self.ground.height);
        let spacing = self.spacing();
        for y in 0..height {
            for x in 0..width {
                let Some(at) = self.summed_at(x, y) else {
                    continue;
                };
                let rise = [
                    (self.lift((x + 1) % width, y) - self.lift((x + width - 1) % width, y))
                        / (2.0 * spacing[0]),
                    (self.lift(x, (y + 1) % height) - self.lift(x, (y + height - 1) % height))
                        / (2.0 * spacing[1]),
                ];
                let upright = 1.0 / (1.0 + rise[0] * rise[0] + rise[1] * rise[1]).sqrt();
                let summed = &mut self.summed[at];
                summed.upright = upright;
                for (along, toward) in summed.along.iter_mut().zip(TOWARD) {
                    *along = -(rise[0] * toward[0] + rise[1] * toward[1]) * upright;
                }
            }
        }
    }

    fn sweep(&mut self, heading: usize) {
        let (width, height) = (self.ground.width, self.ground.height);
        let [dx, dy] = STEPS[heading];
        let spacing = self.spacing();
        let (lines, length) = match (dx, dy) {
            (_, 0) => (height, width),
            (0, _) => (width, height),
            _ => {
                let shared = divisor(width, height);
                (shared, width / shared * height)
            }
        };
        let apart = ((dx as f32 * spacing[0]) * (dx as f32 * spacing[0])
            + (dy as f32 * spacing[1]) * (dy as f32 * spacing[1]))
            .sqrt();
        let mut cells = Vec::with_capacity(length);
        let mut lifts = Vec::with_capacity(length);
        let mut hull = Vec::with_capacity(length);
        for line in 0..lines {
            cells.clear();
            lifts.clear();
            let (mut x, mut y) = if dy == 0 { (0, line) } else { (line, 0) };
            for _ in 0..length {
                cells.push(self.summed_at(x, y));
                lifts.push(self.lift(x, y));
                x = (x as i32 + dx).rem_euclid(width as i32) as usize;
                y = (y as i32 + dy).rem_euclid(height as i32) as usize;
            }
            for back in [false, true] {
                let toward = if back { heading + HEADINGS / 2 } else { heading };
                let lift = |place: usize| {
                    let at = place % length;
                    lifts[if back { length - 1 - at } else { at }]
                };
                let slope = |from: usize, to: usize| {
                    (lift(to) - lift(from)) / ((to - from) as f32 * apart)
                };
                hull.clear();
                for place in (0..2 * length).rev() {
                    while hull.len() >= 2
                        && slope(place, hull[hull.len() - 1]) <= slope(place, hull[hull.len() - 2])
                    {
                        hull.pop();
                    }
                    if place < length {
                        let cell = cells[if back { length - 1 - place } else { place }];
                        if let (Some(cell), Some(&top)) = (cell, hull.last()) {
                            self.summed[cell].horizons[toward] = slope(place, top);
                        }
                    }
                    hull.push(place);
                }
            }
        }
    }

    fn light(&mut self, sun: usize) {
        let alpha = self.ground.roughness * self.ground.roughness;
        let suns = heights_of(SUN_STEPS);
        let views = heights_of(VIEW_STEPS);
        let count = self.summed.len() as f64;
        let along = |summed: &Summed, heading: usize| {
            if heading < HEADINGS / 2 {
                summed.along[heading]
            } else {
                -summed.along[heading - HEADINGS / 2]
            }
        };
        for view in 0..HEADINGS {
            let apart = (sun + HEADINGS - view) % HEADINGS;
            let turn = apart.min(HEADINGS - apart);
            let between = TOWARD[sun][0] * TOWARD[view][0] + TOWARD[sun][1] * TOWARD[view][1];
            let mut halves = [0.0_f32; SUN_STEPS * VIEW_STEPS];
            for (v, seen) in views.iter().enumerate() {
                for (s, lit) in suns.iter().enumerate() {
                    let facing = lit.sine * seen.sine + lit.cosine * seen.cosine * between;
                    halves[v * SUN_STEPS + s] = 1.0 / max(2.0 + 2.0 * facing, 1.0e-12).sqrt();
                }
            }
            let mut sums = [[0.0_f64; 5]; SUN_STEPS * VIEW_STEPS];
            for summed in &self.summed {
                let lit_along = along(summed, sun);
                let seen_along = along(summed, view);
                for (v, seen) in views.iter().enumerate() {
                    let n_v = summed.upright * seen.sine + seen_along * seen.cosine;
                    let shown = n_v > 0.0 && seen.tangent > summed.horizons[view];
                    let weight = if shown { n_v / summed.upright } else { 0.0 };
                    let faced = max(n_v, NEAREST_SEEN);
                    for (s, lit) in suns.iter().enumerate() {
                        let sum = &mut sums[v * SUN_STEPS + s];
                        sum[0] += f64::from(weight);
                        let n_l = summed.upright * lit.sine + lit_along * lit.cosine;
                        if n_l <= 0.0 {
                            continue;
                        }
                        let n_h = max(n_l + n_v, 0.0) * halves[v * SUN_STEPS + s];
                        let shine = distribution(n_h, alpha) * visibility(faced, n_l, alpha) * n_l;
                        sum[1] += f64::from(n_l);
                        sum[2] += f64::from(shine);
                        if shown && lit.tangent > summed.horizons[sun] {
                            sum[3] += f64::from(weight * n_l);
                            sum[4] += f64::from(weight * shine);
                        }
                    }
                }
            }
            for (pair, sum) in sums.iter().enumerate() {
                let node = &mut self.nodes[turn * SUN_STEPS * VIEW_STEPS + pair];
                node.drawn_light += sum[1] / count;
                node.drawn_shine += sum[2] / count;
                if sum[0] > 0.0 {
                    node.seen_light += sum[3] / sum[0];
                    node.seen_shine += sum[4] / sum[0];
                }
            }
        }
    }

    pub fn shares(&self) -> Vec<[f32; 2]> {
        self.nodes
            .iter()
            .map(|node| {
                [
                    share(node.seen_light, node.drawn_light),
                    share(node.seen_shine, node.drawn_shine),
                ]
            })
            .collect()
    }

    pub fn table(&self) -> Vec<u32> {
        self.shares()
            .into_iter()
            .map(|[light, shine]| word(light) | word(shine) << 16)
            .collect()
    }
}

fn share(seen: f64, drawn: f64) -> f32 {
    if drawn > 0.0 {
        (seen / drawn) as f32
    } else {
        1.0
    }
}

fn word(share: f32) -> u32 {
    ((share * UNIT as f32 + 0.5) as u32).min(MOST)
}

pub fn distribution(n_h: f32, alpha: f32) -> f32 {
    let alpha2 = alpha * alpha;
    let d = n_h * n_h * (alpha2 - 1.0) + 1.0;
    alpha2 / (std::f32::consts::PI * d * d)
}

pub fn visibility(n_v: f32, n_l: f32, alpha: f32) -> f32 {
    let smooth = 2.0 * n_l * n_v;
    0.5 / (smooth + (n_l + n_v - smooth) * alpha)
}
