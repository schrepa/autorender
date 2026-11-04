use std::ops::Range;
use std::sync::Arc;

use engine_codec::Error;
use engine_core::math::GRAVITY;
use engine_heightfield::noise::smooth;
use engine_heightfield::rivers::{RiverRaster, Rivers};
use engine_heightfield::tile::{
    Footprint, Generator, LAKE_REACH, PATCH_QUADS, PATCHES_PER_TILE, TILE_STRIDE, Tile,
    extend_water, hole_at, on_triangle, tile_cell, water_rows,
};
use engine_heightfield::{Layers, Pyramid, Reach, TILE_QUADS, TILE_SAMPLES, TileKey};
use engine_places::ground::TileFace;
use glam::Vec2;

use crate::gpu::WATER_SPAN_STEPS as SPAN_STEPS;
pub use crate::gpu::{
    BREAK_FULL, BREAK_ONSET, BREAK_THINNEST, TURBULENT_SPEED, TURBULENT_SWIRL, WATER_FLOW_RANGE,
    WATER_SHEAR_RANGE, WATER_SPAN_LEAST, WATER_SPAN_MOST,
};
use crate::shallow_water::DENSITY;

pub(crate) const NONE: u32 = u32::MAX;
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
    pub state: PointState,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PointState {
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
        extend_water(&mut surface, &mut info, heights, ([0; 2], &mixed));
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

    pub(crate) fn holding(&self) -> [u64; 4] {
        self.held
    }

    pub(crate) fn from_parts(
        surface: Box<[f32]>,
        info: Box<[[u32; 2]]>,
        held: [u64; 4],
        solids: Box<[Footprint]>,
    ) -> Self {
        Self {
            surface,
            info,
            held,
            solids,
        }
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
    pub(crate) fn sample_state(&self, x: i32, z: i32) -> PointState {
        let stride = TILE_STRIDE as i32;
        unpack_state(self.info[((z + 1) * stride + x + 1) as usize][1])
    }
}

impl crate::codec::Codec for TileWater {
    fn write(&self, out: &mut crate::codec::Writer) {
        engine_codec::put_grid(out, &self.surface, TILE_STRIDE as usize);
        out.put(&self.info).put(&self.held).put(&self.solids);
    }

    fn read(input: &mut crate::codec::Reader<'_>) -> Result<Self, crate::codec::Error> {
        let stride = TILE_STRIDE as usize;
        let surface: Box<[f32]> = crate::codec::get_grid(input, stride)?;
        let info: Box<[[u32; 2]]> = input.get()?;
        let held: [u64; 4] = input.get()?;
        let solids: Box<[Footprint]> = input.get()?;
        if surface.len() != stride * stride
            || info.len() != stride * stride
            || !solids.iter().all(Footprint::valid)
        {
            return Err(crate::codec::invalid("$.tile_water"));
        }
        Ok(Self {
            surface,
            info,
            held,
            solids,
        })
    }
}

fn mixed(near: &[usize], info: &[[u32; 2]]) -> [u32; 2] {
    let count = near.len();
    let (mut flow, mut lakes, mut span) = (Vec2::ZERO, 0, 0.0);
    let (mut state, mut reaches) = (PointState::default(), [0_u32; 6]);
    for &near in near {
        let [packed, stirred] = info[near];
        let (near_flow, lake, near_span) = unpack(packed);
        let near_state = unpack_state(stirred);
        flow += near_flow;
        lakes += usize::from(lake);
        span += near_span;
        state.turbulence += near_state.turbulence;
        state.breaking += near_state.breaking;
        state.shear += near_state.shear;
        reaches[reach_code(near_state.reach) as usize] += 1;
    }
    let share = 1.0 / count as f32;
    let most = (0..reaches.len()).fold(0, |most, code| {
        if reaches[code] > reaches[most] {
            code
        } else {
            most
        }
    });
    [
        pack(flow * share, false, lakes * 2 > count, span * share),
        pack_state(PointState {
            turbulence: state.turbulence * share,
            breaking: state.breaking * share,
            shear: state.shear * share,
            reach: reach_of(most as u32),
        }),
    ]
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
    for ring in 1..=LAKE_REACH {
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

pub(crate) fn pack_state(state: PointState) -> u32 {
    unit(state.turbulence)
        | unit(state.breaking) << 8
        | unit(state.shear / WATER_SHEAR_RANGE) << 16
        | reach_code(state.reach) << 24
}

pub(crate) fn unpack_state(word: u32) -> PointState {
    let share = |shift: u32| f32::from((word >> shift) as u8) / 255.0;
    PointState {
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

pub(crate) fn positions(layers: &Layers, pyramid: &Pyramid, key: TileKey) -> [Vec<f32>; 2] {
    let spacing = pyramid.spacing(u32::from(key.level));
    let [x, z] = pyramid.first_sample(key);
    let [dx, dz] = layers.offset_samples(spacing);
    let positions = |first: i64| -> Vec<f32> {
        (-1..=i64::from(TILE_SAMPLES))
            .map(|index| (first + index) as f32 * spacing)
            .collect()
    };
    [positions(x + dx), positions(z + dz)]
}

type Rows = (Vec<f32>, Vec<[u32; 2]>);

fn tile_rows(layers: &Layers, pyramid: &Pyramid, key: TileKey) -> Option<Rows> {
    let stride = TILE_STRIDE as usize;
    let (_, hydrology) = layers.hydrology_layer()?;
    let spacing = pyramid.spacing(u32::from(key.level));
    let [xs, zs] = positions(layers, pyramid, key);
    let raster = RiverRaster::whole(&hydrology.rivers, &xs, &zs, spacing);
    let count = stride * stride;
    let (mut surface, mut info) = (vec![f32::NAN; count], vec![[0_u32; 2]; count]);
    water_rows(
        &hydrology.water,
        &|row, column| raster.as_ref()?.water(&hydrology.rivers, row, column),
        (0..stride, &|_| 0..stride),
        (&xs, &zs),
        &|x, z| layers.shown(x, z),
        (&mut surface, &mut info),
        (&|span| [pack(Vec2::ZERO, false, true, span), 0], &|water| {
            [pack(water.flow, water.sheeted, false, water.span), 0]
        }),
    );
    Some((surface, info))
}

pub(crate) fn tile_origin(key: TileKey, spacing: f32) -> Vec2 {
    Vec2::new(f32::from(key.x), f32::from(key.z)) * (TILE_QUADS as f32 * spacing)
}

fn tile_water_of(rows: Option<Rows>, tile: &Tile) -> Option<TileWater> {
    let (surface, info) = rows?;
    let spacing = tile.spacing();
    TileWater::new(
        surface.into_boxed_slice(),
        info.into_boxed_slice(),
        (tile.heights(), spacing, tile_origin(tile.key(), spacing)),
        &[],
    )
}

fn kept(water: Option<TileWater>) -> Result<Option<TileWater>, Error> {
    engine_codec::decode(&engine_codec::encode(&water))
}

pub(crate) fn generated(
    generator: &mut Generator,
    budget: &mut u64,
    (layers, pyramid): (&Layers, &Pyramid),
) -> Result<Option<(Tile, Option<TileWater>)>, Error> {
    let Some(made) = generator.advance(budget) else {
        if generator.failed() {
            return Err(crate::jobs::invalid_heights());
        }
        return Ok(None);
    };
    let rows = tile_rows(layers, pyramid, generator.key());
    let own = kept(tile_water_of(rows, &made))?;
    let tile: Tile = engine_codec::decode(&engine_codec::encode(&made))?;
    Ok(Some((tile, own)))
}

pub(crate) struct BaseWater {
    layers: Arc<Layers>,
    pyramid: Pyramid,
    key: TileKey,
    rows: Option<Option<Rows>>,
    generator: Option<Generator>,
}

impl BaseWater {
    pub(crate) fn new(layers: Arc<Layers>, pyramid: Pyramid, key: TileKey) -> Self {
        Self {
            layers,
            pyramid,
            key,
            rows: None,
            generator: None,
        }
    }

    pub(crate) fn advance(&mut self, budget: &mut u64) -> Result<Option<Option<TileWater>>, Error> {
        if self.rows.is_none() {
            let stride = u64::from(TILE_STRIDE);
            *budget = budget.saturating_sub(stride * stride);
            self.rows = Some(tile_rows(&self.layers, &self.pyramid, self.key));
        }
        let wet = self.rows.as_ref().is_some_and(|rows| {
            rows.as_ref()
                .is_some_and(|(surface, _)| surface.iter().any(|level| level.is_finite()))
        });
        if !wet {
            return Ok(Some(None));
        }
        let (layers, pyramid, key) = (&self.layers, &self.pyramid, self.key);
        let generator = self
            .generator
            .get_or_insert_with(|| Generator::new(Arc::clone(layers), pyramid, key));
        let Some(made) = generator.advance(budget) else {
            if generator.failed() {
                return Err(crate::jobs::invalid_heights());
            }
            return Ok(None);
        };
        Ok(Some(tile_water_of(self.rows.take().flatten(), &made)))
    }
}

pub(crate) fn rewatered(
    (key, spacing, heights): (TileKey, f32, &[f32]),
    water: Option<&TileWater>,
    (layers, pyramid): (&Layers, &Pyramid),
    rivers: &Rivers,
    regions: &[(Vec2, Vec2)],
) -> Option<TileWater> {
    let (_, hydrology) = layers.hydrology_layer()?;
    let [xs, zs] = positions(layers, pyramid, key);
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
    let (mut surface, mut info) = water.map_or_else(
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
        (&|span| [pack(Vec2::ZERO, false, true, span), 0], &|water| {
            [pack(water.flow, water.sheeted, false, water.span), 0]
        }),
    );
    let solids = water.map_or(&[][..], TileWater::solids);
    TileWater::new(
        surface.into_boxed_slice(),
        info.into_boxed_slice(),
        (heights, spacing, tile_origin(key, spacing)),
        solids,
    )
}

fn stored(heights: &[f32], x: i32, z: i32) -> f32 {
    let stride = TILE_STRIDE as i32;
    heights[((z + 1) * stride + x + 1) as usize]
}

fn point_on(
    (heights, water): (&[f32], &TileWater),
    (i, j, fu, fv): (i32, i32, f32, f32),
) -> Option<WaterPoint> {
    let at = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)];
    let surfaces = at.map(|(x, z)| water.surface(x, z));
    let corners = if fu + fv <= 1.0 { [0, 1, 2] } else { [3, 2, 1] };
    if corners.iter().any(|&corner| !surfaces[corner].is_finite()) {
        return None;
    }
    let surface = on_triangle(surfaces, fu, fv);
    let depth = surface - on_triangle(at.map(|(x, z)| stored(heights, x, z)), fu, fv);
    if depth <= 0.0 {
        return None;
    }
    let info = at.map(|(x, z)| water.sample_info(x, z));
    let states = at.map(|(x, z)| water.sample_state(x, z));
    Some(WaterPoint {
        surface,
        depth,
        flow: Vec2::new(
            on_triangle(info.map(|(flow, _, _)| flow.x), fu, fv),
            on_triangle(info.map(|(flow, _, _)| flow.y), fu, fv),
        ),
        span: on_triangle(info.map(|(_, _, span)| span), fu, fv),
        lake: info[corners[0]].1,
        state: PointState {
            turbulence: on_triangle(states.map(|state| state.turbulence), fu, fv),
            breaking: on_triangle(states.map(|state| state.breaking), fu, fv),
            shear: on_triangle(states.map(|state| state.shear), fu, fv),
            reach: states[corners[0]].reach,
        },
    })
}

pub(crate) fn point(
    face: &TileFace,
    water: Option<&TileWater>,
    local: Vec2,
) -> Option<Option<WaterPoint>> {
    let (i, j, fu, fv) = tile_cell(face.spacing, local)?;
    let hole = face
        .holes
        .as_deref()
        .is_some_and(|holes| hole_at(holes, i as u32, j as u32));
    if hole {
        return Some(None);
    }
    Some(water.and_then(|water| point_on((&face.heights, water), (i, j, fu, fv))))
}

pub(crate) fn tile_point(
    tile: &Tile,
    water: Option<&TileWater>,
    local: Vec2,
) -> Option<WaterPoint> {
    let sample = tile.sample(local)?;
    if sample.hole {
        return None;
    }
    let cell = tile_cell(tile.spacing(), local)?;
    point_on((tile.heights(), water?), cell)
}
