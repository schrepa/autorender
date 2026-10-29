use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::MAX_LEVEL;
use engine_math::{max, min};
use engine_settings::settings;
use serde::{Deserialize, Serialize};

use std::sync::Arc;

use glam::Vec2;

use crate::bed::{HEAD_FLOOD, NARROWEST, bankfull};
use crate::given::{Drained, GivenLayers, courses};
use crate::layers::HeightImage;
use crate::noise::hash;
use crate::rivers::{Channels, RiverLine, Rivers, trace};

settings! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct HydrologyLayer {
        pub level: u32 = 0 => range(0, MAX_LEVEL),
        pub discharge: f32 = DISCHARGE => range(0.001, 1000.0),
    }
}

const DISCHARGE: f32 = 1.0;

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
pub struct Hydrology {
    pub spacing: f32,
    pub flow: HeightImage,
    pub wetness: HeightImage,
    pub lakes: HeightImage,
    pub(crate) water: Arc<WaterGrid>,
    pub(crate) rivers: Arc<Rivers>,
    pub falls: Arc<[Fall]>,
    pub stats: HydrologyStats,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WaterGrid {
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

const SHORE_SAMPLES: usize = 2;
const SHORE_DROP: f32 = 0.5;
const DRY_DEPTH: f32 = 20.0;

const NEIGHBORS: [(i32, i32, f32); 8] = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, std::f32::consts::SQRT_2),
    (-1, 1, std::f32::consts::SQRT_2),
    (1, -1, std::f32::consts::SQRT_2),
    (-1, -1, std::f32::consts::SQRT_2),
];

const LAKE_DEPTH: f32 = 2.0;
const WET: f32 = 1.0e-3;

fn lakes(heights: &[f32], filled: &[f32], width: usize) -> Vec<bool> {
    let count = heights.len();
    let height = count / width;
    let depth = |sample: usize| filled[sample] - heights[sample];
    let mut lake = vec![false; count];
    let mut seen = vec![false; count];
    let mut region = Vec::new();
    for start in 0..count {
        if seen[start] || depth(start) <= WET {
            continue;
        }
        region.clear();
        region.push(start);
        seen[start] = true;
        let mut deepest = 0.0_f32;
        let mut next = 0;
        while next < region.len() {
            let sample = region[next];
            next += 1;
            deepest = max(deepest, depth(sample));
            let (x, z) = ((sample % width) as i32, (sample / width) as i32);
            for (dx, dz, _) in NEIGHBORS {
                let (nx, nz) = (x + dx, z + dz);
                if nx < 0 || nz < 0 || nx >= width as i32 || nz >= height as i32 {
                    continue;
                }
                let neighbor = nz as usize * width + nx as usize;
                if !seen[neighbor] && depth(neighbor) > WET {
                    seen[neighbor] = true;
                    region.push(neighbor);
                }
            }
        }
        if deepest >= LAKE_DEPTH {
            for &sample in &region {
                lake[sample] = true;
            }
        }
    }
    lake
}

#[inline]
fn level_key(height: f32) -> u64 {
    let bits = height.to_bits();
    let flip = (((bits as i32) >> 31) as u32) | 0x8000_0000;
    u64::from(bits ^ flip)
}

fn encode(value: f32, range: [f32; 2]) -> u16 {
    let unit = (value - range[0]) / (range[1] - range[0]);
    (max(min(unit, 1.0), 0.0) * 65_535.0 + 0.5) as u16
}

pub fn decode(sample: f32, range: [f32; 2]) -> f32 {
    range[0] + (range[1] - range[0]) * sample * (1.0 / 65_535.0)
}

#[derive(Default)]
pub(crate) struct Flood {
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
pub(crate) struct Drainage {
    pub(crate) filled: Vec<f32>,
    pub(crate) order: Vec<u32>,
    pub(crate) receivers: Vec<u32>,
    pub(crate) slopes: Vec<f32>,
    pub(crate) area: Vec<f32>,
}

const RECEIVER_SEED: u32 = 0x2f6b_3c91;

fn neighbor_steps(width: usize, spacing: f32) -> [(isize, f32); 8] {
    NEIGHBORS.map(|(dx, dz, distance)| {
        (
            dz as isize * width as isize + dx as isize,
            1.0 / (distance * spacing),
        )
    })
}

pub(crate) fn drainage(heights: &[f32], width: usize, spacing: f32) -> Drainage {
    let mut drainage = Drainage {
        filled: heights.to_vec(),
        ..Drainage::default()
    };
    drainage.route(width, spacing, 0, &mut Flood::default());
    drainage
}

impl Drainage {
    pub(crate) fn route(&mut self, width: usize, spacing: f32, draw: u32, flood: &mut Flood) {
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

fn spread(heights: &[f32], drained: &Drainage, width: usize, spacing: f32) -> (Vec<f32>, Vec<f32>) {
    let count = heights.len();
    let steps = neighbor_steps(width, spacing);
    let mut area = vec![spacing * spacing; count];
    let mut steepest = vec![0.0_f32; count];
    for &sample in drained.order.iter().rev() {
        let sample = sample as usize;
        let receiver = drained.receivers[sample];
        if receiver == u32::MAX {
            continue;
        }
        let ground = drained.filled[sample];
        let mut down = [0.0_f32; 8];
        let mut most = 0.0_f32;
        if ground <= heights[sample] {
            for (index, &(offset, inverse)) in steps.iter().enumerate() {
                let drop = ground - drained.filled[sample.wrapping_add_signed(offset)];
                if drop > 0.0 {
                    down[index] = drop * inverse;
                    most = max(most, down[index]);
                }
            }
        }
        let water = area[sample];
        if most == 0.0 {
            area[receiver as usize] += water;
            continue;
        }
        steepest[sample] = most;
        let scale = 1.0 / most;
        let mut total = 0.0_f32;
        for slope in &mut down {
            let ratio = *slope * scale;
            let squared = ratio * ratio;
            *slope = squared * squared;
            total += *slope;
        }
        let share = water / total;
        for (&(offset, _), &weight) in steps.iter().zip(&down) {
            if weight > 0.0 {
                area[sample.wrapping_add_signed(offset)] += weight * share;
            }
        }
    }
    (area, steepest)
}

pub const MAP_LEVELS: u32 = 3;

#[derive(Clone, Copy)]
pub(crate) struct Grid<'a> {
    pub(crate) heights: &'a [f32],
    pub(crate) width: usize,
    pub(crate) spacing: f32,
}

impl Grid<'_> {
    pub(crate) fn rows(&self) -> usize {
        self.heights.len() / self.width
    }

    pub(crate) fn height(&self, x: f32, z: f32) -> f32 {
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

pub(crate) fn bake(
    (heights, width, spacing): (&[f32], usize, f32),
    (landscape, across, apart): (&[f32], usize, f32),
    (layer, given, finest): (&HydrologyLayer, &GivenLayers<'_>, f32),
) -> (Hydrology, Vec<RiverLine>) {
    let count = landscape.len();
    let rows = count / across;
    let routed = drainage(landscape, across, apart);
    let lake = lakes(landscape, &routed.filled, across);
    let (spread_area, steepest) = spread(landscape, &routed, across, apart);
    let filled = routed.filled;
    let found = Found {
        filled: &filled,
        lakes: &lake,
    };
    let water = Arc::new(water(landscape, &found, across, apart));
    let mut lake_samples = 0;
    let mut flow = Vec::with_capacity(count);
    let mut wetness = Vec::with_capacity(count);
    let mut lakes = Vec::with_capacity(count);
    for sample in 0..count {
        let standing = if lake[sample] {
            lake_samples += 1;
            filled[sample] - landscape[sample]
        } else {
            0.0
        };
        flow.push(encode(libm::log10f(spread_area[sample]), FLOW_RANGE));
        let catchment = spread_area[sample] / apart;
        let index = libm::logf(catchment / max(steepest[sample], 1.0e-3));
        wetness.push(encode(index, WETNESS_RANGE));
        lakes.push(encode(standing, LAKE_RANGE));
    }
    let image = |samples: Vec<u16>| {
        HeightImage::new(across as u32, rows as u32, samples).expect("grid_two_samples_each_way")
    };
    let Drainage {
        order,
        receivers,
        area,
        ..
    } = drainage(heights, width, spacing);
    let in_lake: Vec<bool> = (0..heights.len())
        .map(|sample| {
            let (x, z) = (
                (sample % width) as f32 * spacing,
                (sample / width) as f32 * spacing,
            );
            water
                .lake_reach(x, z)
                .is_some_and(|(level, _)| heights[sample] < level)
        })
        .collect();
    let (lines, channel_samples, max_area, given) = rivers_over(
        (heights, width, spacing),
        (water.as_ref(), &in_lake),
        (order, receivers, area),
        (layer, given, finest),
    );
    let hydrology = Hydrology {
        spacing: apart,
        flow: image(flow),
        wetness: image(wetness),
        lakes: image(lakes),
        water,
        rivers: Arc::new(Rivers::default()),
        stats: HydrologyStats {
            samples: heights.len(),
            channel_samples,
            lake_samples,
            max_area,
            given,
            ..HydrologyStats::default()
        },
        falls: Arc::from(Vec::new()),
    };
    (hydrology, lines)
}

fn rivers_over(
    (heights, width, spacing): (&[f32], usize, f32),
    (water, in_lake): (&WaterGrid, &[bool]),
    (order, receivers, area): (Vec<u32>, Vec<u32>, Vec<f32>),
    (layer, given, finest): (&HydrologyLayer, &GivenLayers<'_>, f32),
) -> (Vec<RiverLine>, usize, f32, usize) {
    let mut rank = vec![0_u32; heights.len()];
    for (position, &sample) in order.iter().enumerate() {
        rank[sample as usize] = position as u32;
    }
    let runoff = layer.discharge * 1.0e-6;
    let channel_area = HEAD_FLOOD / runoff;
    let given = courses(
        given,
        (
            &Grid {
                heights,
                width,
                spacing,
            },
            water,
        ),
        &Drained {
            order: &order,
            receivers: &receivers,
            area: &area,
            lakes: in_lake,
        },
        (runoff, NARROWEST * finest),
    );
    drop(order);
    let (area, within, continued, mut lines) = match given {
        Some(given) => (given.area, Some(given.within), given.continued, given.lines),
        None => (area, None, Vec::new(), Vec::new()),
    };
    let channel_samples = (0..area.len())
        .filter(|&sample| {
            (area[sample] >= channel_area && !in_lake[sample])
                || within.as_ref().is_some_and(|within| within[sample])
        })
        .count();
    let max_area = lines
        .iter()
        .flat_map(|line| &line.area)
        .chain(&area)
        .fold(0.0_f32, |most, &drained| max(most, drained));
    let given = lines.len();
    lines.extend(trace(
        &Channels {
            area: &area,
            receivers: &receivers,
            lakes: in_lake,
            rank: &rank,
            within: within.as_deref(),
            continued: &continued,
        },
        (width, spacing),
        channel_area,
        |drained| bankfull(drained * runoff).0,
    ));
    (lines, channel_samples, max_area, given)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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

impl Hydrology {
    pub fn level_for(&self, spacing: f32) -> usize {
        image_level(self.spacing, self.flow.levels(), spacing)
    }

    pub fn at(&self, x: f32, z: f32, spacing: f32) -> HydrologySample {
        let river = self.rivers.edge(x, z, spacing);
        self.masks_at(x, z, spacing, river)
    }

    pub(crate) fn masks_at(&self, x: f32, z: f32, spacing: f32, river: f32) -> HydrologySample {
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

    pub(crate) fn water_at(&self, x: f32, z: f32) -> Option<(f32, f32)> {
        self.water.at(x, z).or_else(|| self.rivers.water_at(x, z))
    }

    pub(crate) fn water_reaches(&self, low: Vec2, high: Vec2) -> bool {
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

struct Found<'a> {
    filled: &'a [f32],
    lakes: &'a [bool],
}

fn water(heights: &[f32], found: &Found<'_>, width: usize, spacing: f32) -> WaterGrid {
    let Found { filled, lakes } = *found;
    let count = heights.len();
    let height = count / width;
    let (last_x, last_z) = (width as i32 - 1, height as i32 - 1);
    let mut surface = vec![f32::NAN; count];
    let mut lake = vec![0_u16; count];
    let mut levels = Vec::new();
    let mut areas = Vec::new();
    let mut region = Vec::new();
    for start in 0..count {
        if !lakes[start] || lake[start] != 0 || levels.len() >= usize::from(u16::MAX) {
            continue;
        }
        levels.push(filled[start]);
        let number = levels.len() as u16;
        region.clear();
        region.push(start);
        lake[start] = number;
        let mut next = 0;
        while next < region.len() {
            let sample = region[next];
            next += 1;
            surface[sample] = filled[sample];
            let (x, z) = ((sample % width) as i32, (sample / width) as i32);
            for (dx, dz, _) in NEIGHBORS {
                let (nx, nz) = (x + dx, z + dz);
                if nx < 0 || nz < 0 || nx > last_x || nz > last_z {
                    continue;
                }
                let neighbor = nz as usize * width + nx as usize;
                if lakes[neighbor] && lake[neighbor] == 0 {
                    lake[neighbor] = number;
                    region.push(neighbor);
                }
            }
        }
        areas.push(region.len() as f32 * spacing * spacing);
    }
    let depth: Vec<f32> = (0..count)
        .map(|sample| {
            if surface[sample].is_nan() {
                0.0
            } else {
                max(surface[sample] - heights[sample], 0.0)
            }
        })
        .collect();
    for _ in 0..SHORE_SAMPLES {
        let (previous, numbers) = (surface.clone(), lake.clone());
        for sample in 0..count {
            if !previous[sample].is_nan() {
                continue;
            }
            let (x, z) = ((sample % width) as i32, (sample / width) as i32);
            for (dx, dz, _) in NEIGHBORS {
                let (nx, nz) = (x + dx, z + dz);
                if nx < 0 || nz < 0 || nx > last_x || nz > last_z {
                    continue;
                }
                let neighbor = nz as usize * width + nx as usize;
                let level = previous[neighbor];
                let ground = heights[sample];
                if !level.is_nan()
                    && ground >= level - SHORE_DROP
                    && (surface[sample].is_nan() || level > surface[sample])
                {
                    surface[sample] = level;
                    lake[sample] = numbers[neighbor];
                }
            }
        }
    }
    for (sample, level) in surface.iter_mut().enumerate() {
        if level.is_nan() {
            *level = heights[sample] - DRY_DEPTH;
        }
    }
    WaterGrid {
        width,
        height,
        spacing,
        surface,
        depth,
        flow: vec![[0.0; 2]; count],
        lake,
        levels,
        areas,
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

    pub(crate) fn lake_reach(&self, x: f32, z: f32) -> Option<(f32, f32)> {
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

pub(crate) fn image_level(image_spacing: f32, levels: usize, spacing: f32) -> usize {
    let mut level = 0;
    while level < levels && image_spacing * (2_u32 << level) as f32 <= spacing * 1.0001 {
        level += 1;
    }
    level
}

crate::codec::record!(HydrologyLayer { level, discharge });
crate::codec::record!(HydrologyStats {
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
crate::codec::record!(Fall {
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
crate::codec::record!(Hydrology {
    spacing,
    flow,
    wetness,
    lakes,
    water,
    rivers,
    falls,
    stats,
});
crate::codec::record!(WaterGrid {
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
