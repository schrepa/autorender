use std::f32::consts::TAU;

use engine::prelude::math::{clamp, exp, max, min, sin_cos};

use super::Rgb;

pub(super) fn texture<T: Send>(size: usize, texel: impl Fn(f32, f32) -> T + Sync) -> Vec<T> {
    let row = |y: usize| -> Vec<T> {
        (0..size)
            .map(|x| {
                texel(
                    (x as f32 + 0.5) / size as f32,
                    (y as f32 + 0.5) / size as f32,
                )
            })
            .collect()
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        if threads > 1 {
            let rows = size.div_ceil(threads);
            let row = &row;
            return std::thread::scope(|scope| {
                let parts: Vec<_> = (0..threads)
                    .map(|thread| {
                        scope.spawn(move || {
                            let first = usize::min(thread * rows, size);
                            let last = usize::min(first + rows, size);
                            (first..last).flat_map(row).collect::<Vec<T>>()
                        })
                    })
                    .collect();
                parts
                    .into_iter()
                    .flat_map(|part| {
                        part.join()
                            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                    })
                    .collect()
            });
        }
    }
    (0..size).flat_map(row).collect()
}

pub(super) fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub(super) fn mix(a: Rgb, b: Rgb, t: f32) -> Rgb {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

pub(super) fn scale(color: Rgb, by: f32) -> Rgb {
    color.map(|channel| channel * by)
}

pub(super) fn saturate(value: f32) -> f32 {
    clamp(value, 0.0, 1.0)
}

pub(super) fn smoothstep(from: f32, to: f32, value: f32) -> f32 {
    let t = saturate((value - from) / (to - from));
    t * t * (3.0 - 2.0 * t)
}

pub(super) fn hash(x: u32, y: u32, seed: u32) -> u32 {
    let mut h =
        x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

pub(super) fn unit(hash: u32) -> f32 {
    (hash >> 8) as f32 * (1.0 / 16_777_216.0)
}

pub(super) struct Rng(u64);

impl Rng {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn next(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32
    }

    pub(super) fn unit(&mut self) -> f32 {
        unit(self.next())
    }

    pub(super) fn range(&mut self, low: f32, high: f32) -> f32 {
        lerp(low, high, self.unit())
    }
}

struct Lattice {
    period: usize,
    gradients: Vec<[f32; 2]>,
}

impl Lattice {
    fn new(period: usize, seed: u32) -> Self {
        let gradients = (0..period * period)
            .map(|index| {
                let (sin, cos) = sin_cos(unit(hash(index as u32, period as u32, seed)) * TAU);
                [cos, sin]
            })
            .collect();
        Self { period, gradients }
    }

    fn sample(&self, u: f32, v: f32) -> f32 {
        let period = self.period;
        let x = u * period as f32;
        let y = v * period as f32;
        let (cell_x, cell_y) = (x.floor(), y.floor());
        let (tx, ty) = (x - cell_x, y - cell_y);
        let x0 = (cell_x as i64).rem_euclid(period as i64) as usize;
        let y0 = (cell_y as i64).rem_euclid(period as i64) as usize;
        let (x1, y1) = ((x0 + 1) % period, (y0 + 1) % period);
        let corner = |cx: usize, cy: usize, dx: f32, dy: f32| {
            let [gx, gy] = self.gradients[cy * period + cx];
            gx * dx + gy * dy
        };
        let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let (sx, sy) = (fade(tx), fade(ty));
        let top = lerp(corner(x0, y0, tx, ty), corner(x1, y0, tx - 1.0, ty), sx);
        let bottom = lerp(
            corner(x0, y1, tx, ty - 1.0),
            corner(x1, y1, tx - 1.0, ty - 1.0),
            sx,
        );
        lerp(top, bottom, sy) * 1.4
    }
}

pub(super) struct Noise {
    octaves: Vec<(Lattice, f32)>,
    total: f32,
}

impl Noise {
    pub(super) fn new(period: usize, octaves: usize, gain: f32, seed: u32) -> Self {
        let mut amplitude = 1.0;
        let mut total = 0.0;
        let octaves = (0..octaves)
            .map(|octave| {
                let lattice =
                    Lattice::new(period << octave, seed.wrapping_add(octave as u32 * 7919));
                let entry = (lattice, amplitude);
                total += amplitude;
                amplitude *= gain;
                entry
            })
            .collect();
        Self { octaves, total }
    }

    pub(super) fn fbm(&self, u: f32, v: f32) -> f32 {
        self.octaves
            .iter()
            .map(|(lattice, amplitude)| lattice.sample(u, v) * amplitude)
            .sum::<f32>()
            / self.total
    }

    pub(super) fn unit(&self, u: f32, v: f32) -> f32 {
        saturate(self.fbm(u, v) * 0.5 + 0.5)
    }

    pub(super) fn ridged(&self, u: f32, v: f32) -> f32 {
        self.octaves
            .iter()
            .map(|(lattice, amplitude)| {
                let crest = 1.0 - saturate(lattice.sample(u, v).abs());
                crest * crest * amplitude
            })
            .sum::<f32>()
            / self.total
    }
}

pub(super) struct Cells {
    count: usize,
    points: Vec<[f32; 2]>,
}

pub(super) struct Hit {
    pub(super) id: u32,
    pub(super) center: f32,
    pub(super) edge: f32,
    pub(super) offset: [f32; 2],
}

impl Cells {
    pub(super) fn new(count: usize, jitter: f32, seed: u32) -> Self {
        let points = (0..count * count)
            .map(|index| {
                let index = index as u32;
                [
                    0.5 + (unit(hash(index, 0, seed)) - 0.5) * jitter,
                    0.5 + (unit(hash(index, 1, seed)) - 0.5) * jitter,
                ]
            })
            .collect();
        Self { count, points }
    }

    pub(super) fn sample(&self, u: f32, v: f32) -> Hit {
        let count = self.count as i64;
        let x = u * self.count as f32;
        let y = v * self.count as f32;
        let (cell_x, cell_y) = (x.floor(), y.floor());
        let (fx, fy) = (x - cell_x, y - cell_y);
        let (cell_x, cell_y) = (cell_x as i64, cell_y as i64);
        let point = |dx: i64, dy: i64| {
            let index = ((cell_y + dy).rem_euclid(count) * count + (cell_x + dx).rem_euclid(count))
                as usize;
            let [px, py] = self.points[index];
            (index as u32, [dx as f32 + px - fx, dy as f32 + py - fy])
        };
        let mut nearest = (f32::MAX, 0, [0.0; 2], (0, 0));
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (id, offset) = point(dx, dy);
                let distance = offset[0] * offset[0] + offset[1] * offset[1];
                if distance < nearest.0 {
                    nearest = (distance, id, offset, (dx, dy));
                }
            }
        }
        let (distance, id, near, (nx, ny)) = nearest;
        let mut edge = f32::MAX;
        for dy in -2..=2 {
            for dx in -2..=2 {
                let (_, offset) = point(nx + dx, ny + dy);
                let between = [offset[0] - near[0], offset[1] - near[1]];
                let length = (between[0] * between[0] + between[1] * between[1]).sqrt();
                if length > 1.0e-5 {
                    let middle = [(offset[0] + near[0]) * 0.5, (offset[1] + near[1]) * 0.5];
                    edge = min(
                        edge,
                        (middle[0] * between[0] + middle[1] * between[1]) / length,
                    );
                }
            }
        }
        Hit {
            id,
            center: distance.sqrt(),
            edge,
            offset: near,
        }
    }

    pub(super) fn faceted(
        &self,
        u: f32,
        v: f32,
        depth: f32,
        sharpness: f32,
        seed: u32,
    ) -> (f32, u32) {
        let count = self.count as i64;
        let x = u * self.count as f32;
        let y = v * self.count as f32;
        let (cell_x, cell_y) = (x.floor(), y.floor());
        let (fx, fy) = (x - cell_x, y - cell_y);
        let (cell_x, cell_y) = (cell_x as i64, cell_y as i64);
        let (mut sum, mut total, mut strongest) = (0.0, 0.0, (0.0, 0));
        for dy in -1..=1 {
            for dx in -1..=1 {
                let index = ((cell_y + dy).rem_euclid(count) * count
                    + (cell_x + dx).rem_euclid(count)) as usize;
                let [px, py] = self.points[index];
                let offset = [dx as f32 + px - fx, dy as f32 + py - fy];
                let id = index as u32;
                let random = |salt: u32| unit(hash(id, salt, seed)) * 2.0 - 1.0;
                let weight = exp(-sharpness * (offset[0] * offset[0] + offset[1] * offset[1]));
                let plane = random(1) * depth - random(2) * offset[0] - random(3) * offset[1];
                sum += weight * plane;
                total += weight;
                if weight > strongest.0 {
                    strongest = (weight, id);
                }
            }
        }
        (sum / max(total, 1.0e-12), strongest.1)
    }
}

pub(super) fn blur(field: &[f32], size: usize, radius: usize) -> Vec<f32> {
    let width = (2 * radius + 1) as f32;
    let pass = |source: &[f32], stride: usize, step: usize| {
        let mut out = vec![0.0; source.len()];
        for line in 0..size {
            let at = |i: i64| source[line * stride + (i.rem_euclid(size as i64) as usize) * step];
            let r = radius as i64;
            let mut sum: f32 = (-r..=r).map(at).sum();
            for i in 0..size as i64 {
                out[line * stride + i as usize * step] = sum / width;
                sum += at(i + r + 1) - at(i - r);
            }
        }
        out
    };
    let rows = pass(field, size, 1);
    pass(&rows, 1, size)
}

pub(super) fn low_pass(field: &[f32], size: usize, radius: usize) -> Vec<f32> {
    let once = blur(field, size, radius);
    let twice = blur(&once, size, radius);
    blur(&twice, size, radius)
}

pub(super) fn occlusion(height: &[f32], size: usize, strength: f32) -> Vec<f32> {
    let mut cavity = vec![0.0; height.len()];
    for (radius, weight) in [(0.003, 0.45), (0.01, 0.35), (0.03, 0.2)] {
        let radius = usize::max(1, (radius * size as f32).round() as usize);
        let blurred = blur(height, size, radius);
        for ((cavity, height), blurred) in cavity.iter_mut().zip(height).zip(&blurred) {
            *cavity += weight * max(blurred - height, 0.0);
        }
    }
    cavity
        .iter()
        .map(|cavity| clamp(1.0 - strength * cavity, 0.15, 1.0))
        .collect()
}
