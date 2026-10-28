use std::ops::Range;

use glam::Vec2;

use crate::hydraulics::GRAVITY;
use crate::hydrology::WaterGrid;
use crate::noise::smooth;
use crate::rivers::{RiverRaster, RiverWater, Rivers};
use crate::shallow_water::DENSITY;
use crate::{
    Footprint, Layers, PATCH_QUADS, PATCHES_PER_TILE, Pyramid, Reach, TILE_QUADS, TILE_SAMPLES,
    TILE_STRIDE, Tile,
};

pub const WATER_FLOW_RANGE: f32 = 8.0;
pub const WATER_SPAN_LEAST: f32 = 0.25;
pub const WATER_SPAN_MOST: f32 = 4096.0;
const SPAN_STEPS: f32 = 126.0;
pub const WATER_SHEAR_RANGE: f32 = 8.0;
pub const TURBULENT_SPEED: f32 = 0.5;
pub const TURBULENT_SWIRL: f32 = 4.0;
pub const BREAK_ONSET: f32 = 0.6;
pub const BREAK_FULL: f32 = 1.2;
pub const BREAK_THINNEST: f32 = 0.02;
pub(crate) const NONE: u32 = u32::MAX;
pub const WATER_REACH: u32 = 4;
const NO_RING: u8 = u8::MAX;
const QUEUED: u8 = u8::MAX - 1;
const STOPPED: u8 = u8::MAX - 2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterPoint {
    pub surface: f32,
    pub depth: f32,
    pub flow: Vec2,
    pub span: f32,
    pub lake: bool,
    pub state: WaterState,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WaterState {
    pub turbulence: f32,
    pub breaking: f32,
    pub shear: f32,
    pub reach: Option<Reach>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TileWater {
    surface: Box<[f32]>,
    info: Box<[[u32; 2]]>,
    held: [u64; 4],
    solids: Box<[Footprint]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WaterMesh {
    holding: Box<[u64]>,
    levels: Box<[f32]>,
}

impl WaterMesh {
    pub fn holds(&self, at: usize) -> bool {
        self.holding[at / 64] & (1 << (at % 64)) != 0
    }

    pub fn level(&self, at: usize) -> f32 {
        self.levels[at]
    }

    pub(crate) fn levels(&self) -> &[f32] {
        &self.levels
    }
}

impl TileWater {
    pub(crate) fn new(
        mut surface: Box<[f32]>,
        mut info: Box<[[u32; 2]]>,
        (heights, spacing, origin): (&[f32], f32, Vec2),
        solids: &[Footprint],
    ) -> Option<Self> {
        let stride = TILE_STRIDE as usize;
        if surface.len() != stride * stride
            || info.len() != stride * stride
            || heights.len() != stride * stride
            || !surface.iter().zip(heights).any(|(s, h)| s > h)
        {
            return None;
        }
        extend(&mut surface, &mut info, heights);
        let mut held = [0_u64; 4];
        let (patch, patches) = (PATCH_QUADS as usize, PATCHES_PER_TILE as usize);
        for pz in 0..patches {
            for px in 0..patches {
                let holds = (pz * patch..=(pz + 1) * patch).any(|z| {
                    (px * patch..=(px + 1) * patch).any(|x| {
                        let at = (z + 1) * stride + x + 1;
                        surface[at].is_finite()
                    })
                });
                if holds {
                    let index = pz * patches + px;
                    held[index / 64] |= 1 << (index % 64);
                }
            }
        }
        let wet: Vec<bool> = surface
            .iter()
            .zip(heights)
            .map(|(&surface, &height)| surface > height)
            .collect();
        let solids = standing(&wet, (spacing, origin, solids));
        Some(Self {
            surface,
            info,
            held,
            solids,
        })
    }

    pub fn surfaces(&self) -> &[f32] {
        &self.surface
    }

    pub fn info(&self) -> &[[u32; 2]] {
        &self.info
    }

    pub fn solids(&self) -> &[Footprint] {
        &self.solids
    }

    pub fn patch_holds_water(&self, x: u32, z: u32) -> bool {
        let index = (z * PATCHES_PER_TILE + x) as usize;
        self.held[index / 64] & (1 << (index % 64)) != 0
    }

    pub(crate) fn mesh(&self, heights: &[f32]) -> WaterMesh {
        let stride = TILE_STRIDE as usize;
        let drawn = drawn_surfaces(&self.surface, heights);
        let holding: Vec<bool> = drawn.iter().map(|level| level.is_finite()).collect();
        let nearest = nearest(&holding);
        let (patch, quads) = (PATCH_QUADS as usize, TILE_QUADS as usize);
        let patches = |place: usize| {
            let sample = place.saturating_sub(1).min(quads);
            (sample.saturating_sub(1) / patch) as u32..=(sample.min(quads - 1) / patch) as u32
        };
        let unblocked = |at: usize| {
            let (x, z) = (at % stride, at / stride);
            patches(z).any(|pz| patches(x).any(|px| !self.patch_holds_water(px, pz)))
        };
        let levels = (0..drawn.len())
            .map(|at| {
                if holding[at] {
                    drawn[at]
                } else if nearest[at] == NONE || unblocked(at) {
                    heights[at]
                } else {
                    drawn[nearest[at] as usize]
                }
            })
            .collect();
        let mut bits = vec![0_u64; holding.len().div_ceil(64)];
        for (at, _) in holding.iter().enumerate().filter(|&(_, &holds)| holds) {
            bits[at / 64] |= 1 << (at % 64);
        }
        WaterMesh {
            holding: bits.into_boxed_slice(),
            levels,
        }
    }

    #[inline]
    pub(crate) fn surface(&self, x: i32, z: i32) -> f32 {
        let stride = TILE_STRIDE as i32;
        self.surface[((z + 1) * stride + x + 1) as usize]
    }

    #[inline]
    pub(crate) fn sample_info(&self, x: i32, z: i32) -> (Vec2, bool, f32) {
        let stride = TILE_STRIDE as i32;
        unpack(self.info[((z + 1) * stride + x + 1) as usize][0])
    }

    #[inline]
    pub(crate) fn sample_state(&self, x: i32, z: i32) -> WaterState {
        let stride = TILE_STRIDE as i32;
        unpack_state(self.info[((z + 1) * stride + x + 1) as usize][1])
    }

    pub(crate) fn bed_word(&self, x: i32, z: i32) -> u32 {
        let stride = TILE_STRIDE as i32;
        let [packed, state] = self.info[((z + 1) * stride + x + 1) as usize];
        state >> 24 | (packed >> 24) << 8
    }
}

impl crate::codec::Codec for TileWater {
    fn write(&self, out: &mut crate::codec::Writer) {
        crate::codec::put_grid(out, &self.surface, TILE_STRIDE as usize);
        out.put(&self.info).put(&self.held).put(&self.solids);
    }

    fn read(input: &mut crate::codec::Reader<'_>) -> Result<Self, crate::codec::CodecError> {
        let stride = TILE_STRIDE as usize;
        let surface = crate::codec::get_grid(input, stride)?;
        let info: Box<[[u32; 2]]> = input.get()?;
        let held: [u64; 4] = input.get()?;
        let solids: Box<[Footprint]> = input.get()?;
        if surface.len() != stride * stride
            || info.len() != stride * stride
            || !solids.iter().all(Footprint::valid)
        {
            return Err(crate::codec::CodecError("invalid tile water"));
        }
        Ok(Self {
            surface,
            info,
            held,
            solids,
        })
    }
}

fn extend(surface: &mut [f32], info: &mut [[u32; 2]], heights: &[f32]) {
    let stride = TILE_STRIDE as usize;
    let mut rings: Vec<u8> = surface
        .iter()
        .zip(heights)
        .map(|(level, ground)| if level > ground { 0 } else { NO_RING })
        .collect();
    for (at, &ring) in rings.iter().enumerate() {
        if ring == NO_RING {
            surface[at] = f32::NAN;
            info[at] = [0; 2];
        }
    }
    let mut inside: Vec<usize> = (0..rings.len()).filter(|&at| rings[at] == 0).collect();
    for ring in 1..=WATER_REACH as u8 {
        let mut queued = Vec::new();
        for &from in &inside {
            let (x, z) = (from % stride, from / stride);
            for nz in z.saturating_sub(1)..(z + 2).min(stride) {
                for nx in x.saturating_sub(1)..(x + 2).min(stride) {
                    let near = nz * stride + nx;
                    if rings[near] == NO_RING {
                        rings[near] = QUEUED;
                        queued.push(near);
                    }
                }
            }
        }
        let made: Vec<(f32, [u32; 2], bool)> = queued
            .iter()
            .map(|&at| continued(at, ring - 1, (&*surface, &*info, heights), &rings))
            .collect();
        inside.clear();
        for (&at, &(level, words, stops)) in queued.iter().zip(&made) {
            surface[at] = level;
            info[at] = words;
            rings[at] = if stops { STOPPED } else { ring };
            if !stops {
                inside.push(at);
            }
        }
    }
}

fn continued(
    at: usize,
    inner: u8,
    (surface, info, heights): (&[f32], &[[u32; 2]], &[f32]),
    rings: &[u8],
) -> (f32, [u32; 2], bool) {
    let stride = TILE_STRIDE as usize;
    let (x, z) = (at % stride, at / stride);
    let (mut level, mut apart, mut flow, mut lakes, mut span, mut count) =
        (0.0, 0.0, Vec2::ZERO, 0, 0.0, 0);
    let (mut state, mut reaches) = (WaterState::default(), [0_u32; 6]);
    for nz in z.saturating_sub(1)..(z + 2).min(stride) {
        for nx in x.saturating_sub(1)..(x + 2).min(stride) {
            let near = nz * stride + nx;
            if rings[near] != inner {
                continue;
            }
            let [packed, stirred] = info[near];
            let (near_flow, lake, near_span) = unpack(packed);
            let near_state = unpack_state(stirred);
            level += surface[near];
            apart += (surface[near] - heights[near]).abs();
            flow += near_flow;
            lakes += usize::from(lake);
            span += near_span;
            state.turbulence += near_state.turbulence;
            state.breaking += near_state.breaking;
            state.shear += near_state.shear;
            reaches[reach_code(near_state.reach) as usize] += 1;
            count += 1;
        }
    }
    let share = 1.0 / count as f32;
    let level = level * share;
    let stops = heights[at] < level;
    let standing = if stops {
        heights[at] - apart * share
    } else {
        level
    };
    let most = (0..reaches.len()).fold(0, |most, code| {
        if reaches[code] > reaches[most] {
            code
        } else {
            most
        }
    });
    let words = [
        pack(flow * share, false, lakes * 2 > count, span * share),
        pack_state(WaterState {
            turbulence: state.turbulence * share,
            breaking: state.breaking * share,
            shear: state.shear * share,
            reach: reach_of(most as u32),
        }),
    ];
    (standing, words, stops)
}

fn drawn_surfaces(surfaces: &[f32], heights: &[f32]) -> Vec<f32> {
    let stride = TILE_STRIDE as usize;
    let mut drawn = surfaces.to_vec();
    let mut rings: Vec<u8> = surfaces
        .iter()
        .zip(heights)
        .map(|(level, ground)| if level > ground { 0 } else { NO_RING })
        .collect();
    let mut inside: Vec<usize> = (0..rings.len()).filter(|&at| rings[at] == 0).collect();
    let around = move |at: usize| {
        let (x, z) = (at % stride, at / stride);
        (z.saturating_sub(1)..(z + 2).min(stride)).flat_map(move |nz| {
            (x.saturating_sub(1)..(x + 2).min(stride)).map(move |nx| nz * stride + nx)
        })
    };
    for ring in 1..=WATER_REACH as u8 {
        let inner = ring - 1;
        let mut queued = Vec::new();
        for &from in &inside {
            for near in around(from) {
                if rings[near] == NO_RING {
                    rings[near] = QUEUED;
                    queued.push(near);
                }
            }
        }
        let stops: Vec<bool> = queued
            .iter()
            .map(|&at| {
                let (mut level, mut count) = (0.0_f32, 0_u32);
                for near in around(at) {
                    if rings[near] == inner {
                        level += surfaces[near];
                        count += 1;
                    }
                }
                let level = level * (1.0 / count as f32);
                surfaces[at].is_nan() || surfaces[at] < level - crate::codec::HEIGHT_STEP
            })
            .collect();
        inside.clear();
        for (&at, &stopped) in queued.iter().zip(&stops) {
            if !stopped {
                rings[at] = ring;
                inside.push(at);
                continue;
            }
            rings[at] = STOPPED;
            drawn[at] = f32::NAN;
            if inner > 0 && surfaces[at].is_finite() {
                for near in around(at) {
                    if rings[near] == inner {
                        drawn[near] = f32::NAN;
                    }
                }
            }
        }
    }
    drawn
}

pub fn pack(flow: Vec2, sheeted: bool, lake: bool, span: f32) -> u32 {
    let byte =
        |value: f32| u32::from((engine_math::clamp(value, -1.0, 1.0) * 127.0).round() as i8 as u8);
    let sheet = if sheeted { -1.0 / 127.0 } else { 0.0 };
    let kind = if lake {
        -span_code(span).max(1)
    } else {
        span_code(span)
    };
    byte(flow.x / WATER_FLOW_RANGE)
        | byte(flow.y / WATER_FLOW_RANGE) << 8
        | byte(sheet) << 16
        | u32::from(kind as u8) << 24
}

pub fn unpack(info: u32) -> (Vec2, bool, f32) {
    let share = |shift: u32| engine_math::max(f32::from((info >> shift) as u8 as i8) / 127.0, -1.0);
    let kind = (info >> 24) as u8 as i8;
    (
        Vec2::new(share(0), share(8)) * WATER_FLOW_RANGE,
        kind < 0,
        span_of(kind),
    )
}

fn span_code(span: f32) -> i8 {
    if span.is_nan() || span <= 0.0 {
        return 0;
    }
    let octaves = libm::log2f(WATER_SPAN_MOST / WATER_SPAN_LEAST);
    let steps = libm::log2f(span / WATER_SPAN_LEAST) / octaves * SPAN_STEPS;
    engine_math::clamp(steps.round() + 1.0, 1.0, SPAN_STEPS + 1.0) as i8
}

fn span_of(code: i8) -> f32 {
    let steps = engine_math::min(f32::from(code).abs(), SPAN_STEPS + 1.0);
    if steps < 1.0 {
        return 0.0;
    }
    WATER_SPAN_LEAST
        * libm::powf(
            WATER_SPAN_MOST / WATER_SPAN_LEAST,
            (steps - 1.0) / SPAN_STEPS,
        )
}

pub(crate) fn pack_state(state: WaterState) -> u32 {
    unit(state.turbulence)
        | unit(state.breaking) << 8
        | unit(state.shear / WATER_SHEAR_RANGE) << 16
        | reach_code(state.reach) << 24
}

pub(crate) fn unpack_state(word: u32) -> WaterState {
    let share = |shift: u32| f32::from((word >> shift) as u8) / 255.0;
    WaterState {
        turbulence: share(0),
        breaking: share(8),
        shear: share(16) * WATER_SHEAR_RANGE,
        reach: reach_of(word >> 24),
    }
}

fn reach_code(reach: Option<Reach>) -> u32 {
    reach.map_or(0, |reach| u32::from(reach as u8) + 1)
}

fn reach_of(code: u32) -> Option<Reach> {
    match code {
        1 => Some(Reach::Pool),
        2 => Some(Reach::Plane),
        3 => Some(Reach::Steps),
        4 => Some(Reach::Cascade),
        5 => Some(Reach::Falls),
        _ => None,
    }
}

fn unit(value: f32) -> u32 {
    (engine_math::clamp(value, 0.0, 1.0) * 255.0).round() as u32
}

pub(crate) fn turbulence(rush: f32, speed: f32, swirl: f32) -> f32 {
    engine_math::clamp(
        rush / engine_math::max(speed, TURBULENT_SPEED) + swirl / TURBULENT_SWIRL,
        0.0,
        1.0,
    )
}

pub(crate) fn breaking(energy: f32, depth: f32, lip: f32) -> f32 {
    let depth = engine_math::max(depth, BREAK_THINNEST);
    let losing = energy / (DENSITY * depth);
    let eddies = libm::cbrtf(losing * depth);
    let breaks = eddies / (GRAVITY * depth).sqrt();
    let born = smooth((breaks - BREAK_ONSET) / (BREAK_FULL - BREAK_ONSET));
    engine_math::clamp(born + lip, 0.0, 1.0)
}

pub(crate) fn shears(flows: &[Vec2], wet: &dyn Fn(usize) -> bool, spacing: f32) -> Vec<f32> {
    let stride = TILE_STRIDE as usize;
    let mut shears = vec![0.0; flows.len()];
    for z in 0..stride {
        for x in 0..stride {
            let at = z * stride + x;
            if !wet(at) {
                continue;
            }
            let change = |before: Option<usize>, after: Option<usize>| {
                let before = before.filter(|&near| wet(near));
                let after = after.filter(|&near| wet(near));
                match (before, after) {
                    (Some(before), Some(after)) => (flows[after] - flows[before]) / (2.0 * spacing),
                    (Some(before), None) => (flows[at] - flows[before]) / spacing,
                    (None, Some(after)) => (flows[after] - flows[at]) / spacing,
                    (None, None) => Vec2::ZERO,
                }
            };
            let along_x = change((x > 0).then(|| at - 1), (x + 1 < stride).then(|| at + 1));
            let along_z = change(
                (z > 0).then(|| at - stride),
                (z + 1 < stride).then(|| at + stride),
            );
            shears[at] = (along_x.length_squared() + along_z.length_squared()).sqrt();
        }
    }
    shears
}

fn standing(
    wet: &[bool],
    (spacing, origin, solids): (f32, Vec2, &[Footprint]),
) -> Box<[Footprint]> {
    let stride = i64::from(TILE_STRIDE);
    let samples = |low: f32, high: f32, axis: usize| {
        let first = ((low - origin[axis]) / spacing).floor() as i64;
        let last = ((high - origin[axis]) / spacing).ceil() as i64 + 2;
        first.max(0)..=last.min(stride - 1)
    };
    solids
        .iter()
        .filter(|solid| {
            let (middle, reach) = (solid.middle(), solid.reach());
            let (low, high) = (middle - reach, middle + reach);
            samples(low.y, high.y, 1)
                .any(|z| samples(low.x, high.x, 0).any(|x| wet[(z * stride + x) as usize]))
        })
        .copied()
        .collect()
}

pub(crate) fn nearest(seeds: &[bool]) -> Vec<u32> {
    let stride = TILE_STRIDE as i32;
    let mut nearest: Vec<u32> = (0..seeds.len())
        .map(|at| if seeds[at] { at as u32 } else { NONE })
        .collect();
    let mut distance: Vec<i32> = seeds
        .iter()
        .map(|&seed| if seed { 0 } else { i32::MAX })
        .collect();
    let mut offer = |x: i32, z: i32, (dx, dz): (i32, i32)| {
        let (nx, nz) = (x + dx, z + dz);
        if nx < 0 || nz < 0 || nx >= stride || nz >= stride {
            return;
        }
        let seed = nearest[(nz * stride + nx) as usize];
        if seed == NONE {
            return;
        }
        let (sx, sz) = (seed as i32 % stride - x, seed as i32 / stride - z);
        let squared = sx * sx + sz * sz;
        let at = (z * stride + x) as usize;
        if squared < distance[at] {
            distance[at] = squared;
            nearest[at] = seed;
        }
    };
    for z in 0..stride {
        for x in 0..stride {
            for step in [(-1, 0), (0, -1), (-1, -1), (1, -1)] {
                offer(x, z, step);
            }
        }
        for x in (0..stride).rev() {
            offer(x, z, (1, 0));
        }
    }
    for z in (0..stride).rev() {
        for x in (0..stride).rev() {
            for step in [(1, 0), (0, 1), (1, 1), (-1, 1)] {
                offer(x, z, step);
            }
        }
        for x in 0..stride {
            offer(x, z, (-1, 0));
        }
    }
    nearest
}

pub(crate) fn water_rows(
    lakes: &WaterGrid,
    river: &dyn Fn(usize, usize) -> Option<RiverWater>,
    (rows, columns): (Range<usize>, &dyn Fn(usize) -> Range<usize>),
    (xs, zs): (&[f32], &[f32]),
    shown: &dyn Fn(f32, f32) -> f32,
    (surface, info): (&mut [f32], &mut [[u32; 2]]),
) {
    let stride = TILE_STRIDE as usize;
    let first = rows.start;
    for row in rows {
        let z = zs[row];
        for column in columns(row) {
            let x = xs[column];
            if shown(x, z) < 1.0 {
                continue;
            }
            let water = river(row, column);
            let at = (row - first) * stride + column;
            match lakes.lake_reach(x, z) {
                Some((level, span)) if !water.is_some_and(|water| water.surface > level) => {
                    surface[at] = level;
                    info[at] = [pack(Vec2::ZERO, false, true, span), 0];
                }
                _ => {
                    if let Some(water) = water {
                        surface[at] = water.surface;
                        info[at] = [pack(water.flow, water.sheeted, false, water.span), 0];
                    }
                }
            }
        }
    }
}

impl Tile {
    #[must_use]
    pub fn rewatered(
        &self,
        layers: &Layers,
        pyramid: &Pyramid,
        rivers: &Rivers,
        regions: &[(Vec2, Vec2)],
    ) -> Option<Self> {
        let (_, hydrology) = layers.hydrology_layer()?;
        let spacing = self.spacing();
        let [x, z] = pyramid.first_sample(self.key());
        let [dx, dz] = layers.offset_samples(spacing);
        let positions = |first: i64| -> Vec<f32> {
            (-1..=i64::from(TILE_SAMPLES))
                .map(|index| (first + index) as f32 * spacing)
                .collect()
        };
        let (xs, zs) = (positions(x + dx), positions(z + dz));
        let within = |values: &[f32], low: f32, high: f32| {
            values.partition_point(|&at| at < low)..values.partition_point(|&at| at <= high)
        };
        let stride = TILE_STRIDE as usize;
        let mut columns: Vec<Range<usize>> = vec![0..0; stride];
        for (low, high) in regions {
            let reached = within(&xs, low.x, high.x);
            if reached.is_empty() {
                continue;
            }
            for row in within(&zs, low.y, high.y) {
                let held = &mut columns[row];
                *held = if Range::is_empty(held) {
                    reached.clone()
                } else {
                    held.start.min(reached.start)..held.end.max(reached.end)
                };
            }
        }
        let (Some(first), Some(last)) = (
            columns.iter().position(|row| !row.is_empty()),
            columns.iter().rposition(|row| !row.is_empty()),
        ) else {
            return None;
        };
        let (mut surface, mut info) = self.water().map_or_else(
            || {
                (
                    vec![f32::NAN; stride * stride],
                    vec![[0; 2]; stride * stride],
                )
            },
            |water| (water.surfaces().to_vec(), water.info().to_vec()),
        );
        for (row, reached) in columns.iter().enumerate() {
            let at = row * stride + reached.start..row * stride + reached.end;
            surface[at.clone()].fill(f32::NAN);
            info[at].fill([0; 2]);
        }
        let raster = RiverRaster::whole(rivers, &xs, &zs[first..=last], spacing);
        let at = first * stride..(last + 1) * stride;
        water_rows(
            &hydrology.water,
            &|row, column| raster.as_ref()?.water(rivers, row - first, column),
            (first..last + 1, &|row| columns[row].clone()),
            (&xs, &zs),
            &|x, z| layers.shown(x, z),
            (&mut surface[at.clone()], &mut info[at]),
        );
        let solids = self.water().map_or(&[][..], TileWater::solids);
        Some(
            self.clone()
                .with_water(surface.into_boxed_slice(), info.into_boxed_slice(), solids),
        )
    }

    pub(crate) fn origin(&self) -> Vec2 {
        let key = self.key();
        Vec2::new(f32::from(key.x), f32::from(key.z)) * (TILE_QUADS as f32 * self.spacing())
    }
}
