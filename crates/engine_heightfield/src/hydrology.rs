use std::cmp::Reverse;
use std::collections::BinaryHeap;

use engine_math::{max, min};
use serde::Serialize;

use std::sync::Arc;

use glam::Vec2;

use crate::layers::HeightImage;
use crate::noise::hash;
use crate::rivers::Rivers;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct HydrologyStats {
    pub samples: usize,
    pub channel_samples: usize,
    pub lake_samples: usize,
    pub max_area: f32,
    pub falls: usize,
    pub reaches: [usize; 5],
    pub steps: usize,
    pub knobs: usize,
    pub given: usize,
    pub uncut: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Fall {
    pub top: [f32; 3],
    pub bottom: [f32; 3],
    pub path: Vec<[f32; 3]>,
    pub times: Vec<f32>,
    pub aloft: Vec<bool>,
    pub widths: Vec<f32>,
    pub sky: Vec<f32>,
    pub width: f32,
    pub area: f32,
    pub speed: f32,
    pub discharge: f32,
}

pub const FLOW_RANGE: [f32; 2] = [0.0, 12.0];
pub const WETNESS_RANGE: [f32; 2] = [-5.0, 25.0];
pub const LAKE_RANGE: [f32; 2] = [0.0, 200.0];

#[derive(Clone, Debug, PartialEq)]
pub struct HydrologyBake {
    pub spacing: f32,
    pub flow: HeightImage,
    pub wetness: HeightImage,
    pub lakes: HeightImage,
    pub water: Arc<WaterGrid>,
    pub rivers: Arc<Rivers>,
    pub falls: Arc<[Fall]>,
    pub stats: HydrologyStats,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WaterGrid {
    pub width: usize,
    pub height: usize,
    pub spacing: f32,
    pub surface: Vec<f32>,
    pub depth: Vec<f32>,
    pub flow: Vec<[f32; 2]>,
    pub lake: Vec<u16>,
    pub levels: Vec<f32>,
    pub areas: Vec<f32>,
}

pub const NEIGHBORS: [(i32, i32, f32); 8] = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, std::f32::consts::SQRT_2),
    (-1, 1, std::f32::consts::SQRT_2),
    (1, -1, std::f32::consts::SQRT_2),
    (-1, -1, std::f32::consts::SQRT_2),
];

#[inline]
fn level_key(height: f32) -> u64 {
    let bits = height.to_bits();
    let flip = (((bits as i32) >> 31) as u32) | 0x8000_0000;
    u64::from(bits ^ flip)
}

pub fn decode(sample: f32, range: [f32; 2]) -> f32 {
    range[0] + (range[1] - range[0]) * sample * (1.0 / 65_535.0)
}

#[derive(Default)]
pub struct Flood {
    closed: Vec<bool>,
    parents: Vec<u32>,
    open: BinaryHeap<Reverse<u64>>,
    pits: BinaryHeap<Reverse<(u64, u32)>>,
}

impl Flood {
    fn fill(&mut self, heights: &mut [f32], width: usize, order: &mut Vec<u32>) {
        let count = heights.len();
        let height = count / width;
        let Self {
            closed,
            parents,
            open,
            pits,
        } = self;
        closed.clear();
        closed.resize(count, false);
        parents.clear();
        parents.resize(count, u32::MAX);
        open.clear();
        pits.clear();
        order.clear();
        order.reserve(count);
        let mut reached = 0_u32;
        let edges = (0..width)
            .flat_map(|x| [(x, 0), (x, height - 1)])
            .chain((0..height).flat_map(|z| [(0, z), (width - 1, z)]));
        for (x, z) in edges {
            let sample = z * width + x;
            if !closed[sample] {
                closed[sample] = true;
                open.push(Reverse((level_key(heights[sample]) << 32) | sample as u64));
            }
        }
        let (columns, rows) = (width as u32, height as u32);
        loop {
            let sample = match pits.pop() {
                Some(Reverse((_, sample))) => sample,
                None => match open.pop() {
                    Some(Reverse(key)) => key as u32,
                    None => break,
                },
            };
            order.push(sample);
            let (x, z) = (sample % columns, sample / columns);
            let spill = heights[sample as usize].next_up();
            for (dx, dz, _) in NEIGHBORS {
                let (nx, nz) = (x.wrapping_add_signed(dx), z.wrapping_add_signed(dz));
                if nx >= columns || nz >= rows {
                    continue;
                }
                let neighbor = (nz * columns + nx) as usize;
                if closed[neighbor] {
                    continue;
                }
                closed[neighbor] = true;
                let key = level_key(heights[neighbor]) << 32;
                if heights[neighbor] < spill {
                    pits.push(Reverse((key | u64::from(reached), neighbor as u32)));
                    reached += 1;
                    heights[neighbor] = spill;
                    parents[neighbor] = sample;
                } else {
                    open.push(Reverse(key | neighbor as u64));
                }
            }
        }
    }
}

#[derive(Default)]
pub struct Drainage {
    pub filled: Vec<f32>,
    pub order: Vec<u32>,
    pub receivers: Vec<u32>,
    pub slopes: Vec<f32>,
    pub area: Vec<f32>,
}

const RECEIVER_SEED: u32 = 0x2f6b_3c91;

pub fn neighbor_steps(width: usize, spacing: f32) -> [(isize, f32); 8] {
    NEIGHBORS.map(|(dx, dz, distance)| {
        (
            dz as isize * width as isize + dx as isize,
            1.0 / (distance * spacing),
        )
    })
}

pub fn drainage(heights: &[f32], width: usize, spacing: f32) -> Drainage {
    let mut drainage = Drainage {
        filled: heights.to_vec(),
        ..Drainage::default()
    };
    drainage.route(width, spacing, 0, &mut Flood::default());
    drainage
}

impl Drainage {
    pub fn route(&mut self, width: usize, spacing: f32, draw: u32, flood: &mut Flood) {
        let Self {
            filled,
            order,
            receivers,
            slopes,
            area,
        } = self;
        let count = filled.len();
        let height = count / width;
        flood.fill(filled, width, order);
        receivers.clear();
        receivers.resize(count, u32::MAX);
        slopes.clear();
        slopes.resize(count, 0.0);
        let seed = RECEIVER_SEED.wrapping_add(draw.wrapping_mul(0x9e37_79b9));
        let steps = neighbor_steps(width, spacing);
        for z in 1..height - 1 {
            for x in 1..width - 1 {
                let sample = z * width + x;
                if flood.parents[sample] != u32::MAX {
                    receivers[sample] = flood.parents[sample];
                    continue;
                }
                let ground = filled[sample];
                let mut down = [0.0_f32; 8];
                let (mut total, mut first) = (0.0_f32, None);
                for (index, &(offset, inverse)) in steps.iter().enumerate() {
                    let drop = ground - filled[sample.wrapping_add_signed(offset)];
                    if drop > 0.0 {
                        down[index] = drop * inverse;
                        total += down[index];
                        first = first.or(Some(index));
                    }
                }
                let Some(first) = first else {
                    continue;
                };
                let unit = (hash(x as i32, z as i32, seed) >> 8) as f32 * (1.0 / 16_777_216.0);
                let mut pick = unit * total;
                let mut chosen = first;
                for (index, &slope) in down.iter().enumerate().skip(first) {
                    if slope > 0.0 {
                        chosen = index;
                        if pick < slope {
                            break;
                        }
                        pick -= slope;
                    }
                }
                receivers[sample] = sample.wrapping_add_signed(steps[chosen].0) as u32;
                slopes[sample] = down[chosen];
            }
        }
        area.clear();
        area.resize(count, spacing * spacing);
        for &sample in order.iter().rev() {
            let receiver = receivers[sample as usize];
            if receiver != u32::MAX {
                area[receiver as usize] += area[sample as usize];
            }
        }
    }
}

pub const MAP_LEVELS: u32 = 3;

#[derive(Clone, Copy)]
pub struct Grid<'a> {
    pub heights: &'a [f32],
    pub width: usize,
    pub spacing: f32,
}

impl Grid<'_> {
    pub fn rows(&self) -> usize {
        self.heights.len() / self.width
    }

    pub fn height(&self, x: f32, z: f32) -> f32 {
        let rows = self.rows();
        let x = min(max(x / self.spacing, 0.0), (self.width - 1) as f32);
        let z = min(max(z / self.spacing, 0.0), (rows - 1) as f32);
        let (column, row) = (x.floor() as usize, z.floor() as usize);
        let (right, below) = ((column + 1).min(self.width - 1), (row + 1).min(rows - 1));
        let (fx, fz) = (x - column as f32, z - row as f32);
        let at = |row: usize, column: usize| self.heights[row * self.width + column];
        let top = at(row, column) + (at(row, right) - at(row, column)) * fx;
        let bottom = at(below, column) + (at(below, right) - at(below, column)) * fx;
        top + (bottom - top) * fz
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[derive(schemars::JsonSchema)]
pub struct HydrologySample {
    pub flow: f32,
    pub wetness: f32,
    pub lake: f32,
    pub river: f32,
    pub grain: f32,
    pub power: f32,
    pub flood: f32,
}

pub const RIVER_FAR: f32 = 1.0e9;

impl Default for HydrologySample {
    fn default() -> Self {
        Self {
            flow: 0.0,
            wetness: 0.0,
            lake: 0.0,
            river: RIVER_FAR,
            grain: 0.0,
            power: 0.0,
            flood: RIVER_FAR,
        }
    }
}

impl HydrologyBake {
    pub fn level_for(&self, spacing: f32) -> usize {
        image_level(self.spacing, self.flow.levels(), spacing)
    }

    pub fn at(&self, x: f32, z: f32, spacing: f32) -> HydrologySample {
        let river = self.rivers.edge(x, z, spacing);
        self.masks_at(x, z, spacing, river)
    }

    pub fn masks_at(&self, x: f32, z: f32, spacing: f32, river: f32) -> HydrologySample {
        let level = self.level_for(spacing);
        let (u, v) = (x / self.spacing, z / self.spacing);
        HydrologySample {
            flow: decode(self.flow.sample_smooth(u, v, level), FLOW_RANGE),
            wetness: decode(self.wetness.sample_smooth(u, v, level), WETNESS_RANGE),
            lake: decode(self.lakes.sample_smooth(u, v, level), LAKE_RANGE),
            river,
            ..HydrologySample::default()
        }
    }

    pub fn rivers(&self) -> &Arc<Rivers> {
        &self.rivers
    }

    pub fn water_at(&self, x: f32, z: f32) -> Option<(f32, f32)> {
        self.water.at(x, z).or_else(|| self.rivers.water_at(x, z))
    }

    pub fn water_reaches(&self, low: Vec2, high: Vec2) -> bool {
        let water = &self.water;
        let pad = Vec2::splat(water.spacing);
        let (first, last) = ((low - pad) / water.spacing, (high + pad) / water.spacing);
        let columns = max(first.x.floor(), 0.0) as usize
            ..=min(last.x.ceil(), water.width as f32 - 1.0) as usize;
        let lake = (max(first.y.floor(), 0.0) as usize
            ..=min(last.y.ceil(), water.height as f32 - 1.0) as usize)
            .any(|z| {
                columns.clone().any(|x| {
                    let sample = z * water.width + x;
                    water.lake[sample] != 0 && water.depth[sample] > 0.0
                })
            });
        lake || self.rivers.reaches(low - pad, high + pad)
    }
}

impl WaterGrid {
    pub fn at(&self, x: f32, z: f32) -> Option<(f32, f32)> {
        let (u, v) = (x / self.spacing, z / self.spacing);
        if !(u >= 0.0 && v >= 0.0) {
            return None;
        }
        let (x0, z0) = (u.floor() as usize, v.floor() as usize);
        if x0 + 1 >= self.width || z0 + 1 >= self.height {
            return None;
        }
        let (fx, fz) = (u - x0 as f32, v - z0 as f32);
        let at = |values: &[f32], dx: usize, dz: usize| values[(z0 + dz) * self.width + x0 + dx];
        let blend = |values: &[f32]| {
            let low = at(values, 0, 0) + (at(values, 1, 0) - at(values, 0, 0)) * fx;
            let high = at(values, 0, 1) + (at(values, 1, 1) - at(values, 0, 1)) * fx;
            low + (high - low) * fz
        };
        let depth = blend(&self.depth);
        (depth > 0.0).then(|| (blend(&self.surface), depth))
    }

    pub fn lake_beside(&self, x: f32, z: f32) -> Option<f32> {
        let center = self.nearest(x, z)?;
        let (cx, cz) = ((center % self.width) as i64, (center / self.width) as i64);
        let mut level: Option<f32> = None;
        for dz in -1..=1 {
            for dx in -1..=1 {
                let (sx, sz) = (cx + dx, cz + dz);
                if sx < 0 || sz < 0 || sx >= self.width as i64 || sz >= self.height as i64 {
                    continue;
                }
                let sample = sz as usize * self.width + sx as usize;
                if self.depth[sample] > 0.0 && self.lake[sample] != 0 {
                    let lake = self.levels[usize::from(self.lake[sample]) - 1];
                    level = Some(level.map_or(lake, |level| max(level, lake)));
                }
            }
        }
        level
    }

    pub fn lake_reach(&self, x: f32, z: f32) -> Option<(f32, f32)> {
        if self.levels.is_empty() {
            return None;
        }
        let (u, v) = (x / self.spacing, z / self.spacing);
        if !(u.is_finite() && v.is_finite()) {
            return None;
        }
        let (i, j) = (u.floor(), v.floor());
        let (fu, fv) = (u - i, v - j);
        let (last_x, last_z) = (self.width as i64 - 1, self.height as i64 - 1);
        let lake = |dx: i64, dz: i64| {
            let x = (i as i64 + dx).clamp(0, last_x) as usize;
            let z = (j as i64 + dz).clamp(0, last_z) as usize;
            self.lake[z * self.width + x]
        };
        let corners = [
            (lake(0, 0), (1.0 - fu) * (1.0 - fv)),
            (lake(1, 0), fu * (1.0 - fv)),
            (lake(0, 1), (1.0 - fu) * fv),
            (lake(1, 1), fu * fv),
        ];
        let held: f32 = corners
            .iter()
            .filter(|(lake, _)| *lake != 0)
            .map(|(_, weight)| weight)
            .sum();
        if held < 0.5 {
            return None;
        }
        let (lake, _) = corners
            .iter()
            .filter(|(lake, _)| *lake != 0)
            .max_by(|a, b| a.1.total_cmp(&b.1))?;
        Some((self.levels[usize::from(*lake) - 1], self.span(*lake)))
    }

    fn span(&self, number: u16) -> f32 {
        2.0 * (self.areas[usize::from(number) - 1] / std::f32::consts::PI).sqrt()
    }

    fn nearest(&self, x: f32, z: f32) -> Option<usize> {
        let (u, v) = ((x / self.spacing).round(), (z / self.spacing).round());
        if !(u >= 0.0 && v >= 0.0) || u as usize >= self.width || v as usize >= self.height {
            return None;
        }
        Some(v as usize * self.width + u as usize)
    }
}

pub fn image_level(image_spacing: f32, levels: usize, spacing: f32) -> usize {
    let mut level = 0;
    while level < levels && image_spacing * (2_u32 << level) as f32 <= spacing * 1.0001 {
        level += 1;
    }
    level
}

engine_codec::record!(HydrologyStats {
    samples,
    channel_samples,
    lake_samples,
    max_area,
    falls,
    reaches,
    steps,
    knobs,
    given,
    uncut,
});
engine_codec::record!(Fall {
    top,
    bottom,
    path,
    times,
    aloft,
    widths,
    sky,
    width,
    area,
    speed,
    discharge,
});
engine_codec::record!(HydrologyBake {
    spacing,
    flow,
    wetness,
    lakes,
    water,
    rivers,
    falls,
    stats,
});
engine_codec::record!(WaterGrid {
    width,
    height,
    spacing,
    surface,
    depth,
    flow,
    lake,
    levels,
    areas,
});
