use std::ops::Range;
use std::sync::{Arc, Mutex, PoisonError};

use glam::{Vec2, Vec3};

use engine_math::{clamp, max, min};

use crate::hydrology::RIVER_FAR;
use crate::layer::{Below, LayerTile, Row, TileFrame};
use crate::Layers;
use crate::rivers::RiverRaster;
use crate::surface::{
    BiomeRules, Control, RowInputs, RowScratch, SampleShape, erosion_row, smooth_gradient,
};
use crate::{
    ErrorPyramid, HydrologySample, PATCH_QUADS, PATCHES_PER_TILE, PathSample, Placed, Placing,
    Pyramid, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey,
};

mod beds;

pub(crate) use beds::RiverInputs;

const HOLE_CELLS: usize = TILE_SAMPLES as usize;
const HOLE_WORDS: usize = (HOLE_CELLS * HOLE_CELLS).div_ceil(64);

pub const RIVER_EDGE_STEP: f32 = 0.0625;
pub const RIVER_EDGE_ZERO: u16 = 32_768;
pub const RIVER_EDGE_FAR: u16 = u16::MAX;

pub(crate) fn river_step(meters: f32) -> u16 {
    if meters >= RIVER_FAR {
        return RIVER_EDGE_FAR;
    }
    let step = (meters / RIVER_EDGE_STEP).round() + f32::from(RIVER_EDGE_ZERO);
    clamp(step, 0.0, f32::from(RIVER_EDGE_FAR - 1)) as u16
}

pub(crate) fn river_meters(step: u16) -> f32 {
    if step == RIVER_EDGE_FAR {
        RIVER_FAR
    } else {
        (f32::from(step) - f32::from(RIVER_EDGE_ZERO)) * RIVER_EDGE_STEP
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tile {
    key: TileKey,
    spacing: f32,
    heights: Box<[f32]>,
    holes: Option<Box<[u64]>>,
    control: Option<Box<[u32]>>,
    river_edges: Option<Box<[u16]>>,
    river_beds: Option<Box<[u16]>>,
    placed: Option<Arc<[Placed]>>,
    placing: Option<Placing>,
    ground_errors: Arc<ErrorPyramid>,
    lake: Option<Arc<[f32]>>,
    bounds: [f32; 2],
    patch_bounds: Box<[[f32; 2]]>,
    patch_holes: [u64; 4],
    patch_culling: Box<[PatchCulling]>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PatchCulling {
    slope: [i16; 2],
    spread: u16,
    floors: [u8; 4],
}

crate::codec::record!(PatchCulling {
    slope,
    spread,
    floors,
});

const SLOPE_STEPS: f32 = 256.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileSample {
    pub height: f32,
    pub normal: Vec3,
    pub hole: bool,
}

impl Tile {
    pub fn new(
        key: TileKey,
        spacing: f32,
        heights: Box<[f32]>,
        holes: Option<Box<[u64]>>,
    ) -> Option<Self> {
        let mut tile = Self::shaped(key, spacing, heights, holes)?;
        tile.patch_culling = (0..PATCHES_PER_TILE * PATCHES_PER_TILE)
            .map(|index| tile.cull(index % PATCHES_PER_TILE, index / PATCHES_PER_TILE))
            .collect();
        Some(tile)
    }

    fn shaped(
        key: TileKey,
        spacing: f32,
        heights: Box<[f32]>,
        holes: Option<Box<[u64]>>,
    ) -> Option<Self> {
        let stride = TILE_STRIDE as usize;
        if heights.len() != stride * stride
            || holes.as_ref().is_some_and(|bits| bits.len() != HOLE_WORDS)
            || !heights.iter().all(|height| height.is_finite())
        {
            return None;
        }
        let holes = holes.filter(|bits| bits.iter().any(|&word| word != 0));
        let ground_errors = Arc::new(ErrorPyramid::ground(&heights));
        let mut tile = Self {
            key,
            spacing,
            heights,
            holes,
            control: None,
            river_edges: None,
            river_beds: None,
            placed: None,
            placing: None,
            ground_errors,
            lake: None,
            bounds: [0.0; 2],
            patch_bounds: vec![[0.0; 2]; (PATCHES_PER_TILE * PATCHES_PER_TILE) as usize]
                .into_boxed_slice(),
            patch_holes: [0; 4],
            patch_culling: Box::default(),
        };
        let mut low = f32::INFINITY;
        let mut high = f32::NEG_INFINITY;
        let patch = PATCH_QUADS as i32;
        for pz in 0..PATCHES_PER_TILE as i32 {
            for px in 0..PATCHES_PER_TILE as i32 {
                let mut patch_low = f32::INFINITY;
                let mut patch_high = f32::NEG_INFINITY;
                for z in pz * patch..=(pz + 1) * patch {
                    for x in px * patch..=(px + 1) * patch {
                        let height = tile.height(x, z);
                        patch_low = min(patch_low, height);
                        patch_high = max(patch_high, height);
                    }
                }
                let index = (pz * PATCHES_PER_TILE as i32 + px) as usize;
                tile.patch_bounds[index] = [patch_low, patch_high];
                low = min(low, patch_low);
                high = max(high, patch_high);
                let holed = (pz * patch..(pz + 1) * patch).any(|z| {
                    (px * patch..(px + 1) * patch).any(|x| tile.is_hole(x as u32, z as u32))
                });
                if holed {
                    tile.patch_holes[index / 64] |= 1 << (index % 64);
                }
            }
        }
        tile.bounds = [low, high];
        Some(tile)
    }

    fn cull(&self, x: u32, z: u32) -> PatchCulling {
        let patch = PATCH_QUADS as i32;
        let (first_x, first_z) = (x as i32 * patch, z as i32 * patch);
        let inverse = 1.0 / self.spacing;
        let mut low = [f32::INFINITY; 2];
        let mut high = [f32::NEG_INFINITY; 2];
        let mut slopes = Vec::with_capacity((2 * patch * patch) as usize);
        for cz in first_z..first_z + patch {
            for cx in first_x..first_x + patch {
                let a = self.height(cx, cz);
                let b = self.height(cx + 1, cz);
                let c = self.height(cx, cz + 1);
                let d = self.height(cx + 1, cz + 1);
                for slope in [
                    [(b - a) * inverse, (c - a) * inverse],
                    [(d - c) * inverse, (d - b) * inverse],
                ] {
                    low = [min(low[0], slope[0]), min(low[1], slope[1])];
                    high = [max(high[0], slope[0]), max(high[1], slope[1])];
                    slopes.push(slope);
                }
            }
        }
        let center = [0.5 * (low[0] + high[0]), 0.5 * (low[1] + high[1])];
        let step = |value: f32| clamp((value * SLOPE_STEPS).round(), -32767.0, 32767.0) as i16;
        let slope = center.map(step);
        let held = [
            f32::from(slope[0]) / SLOPE_STEPS,
            f32::from(slope[1]) / SLOPE_STEPS,
        ];
        let spread = slopes.iter().fold(0.0_f32, |far, &[sx, sz]| {
            let (dx, dz) = (sx - held[0], sz - held[1]);
            max(far, (dx * dx + dz * dz).sqrt())
        });
        let spread = (spread * SLOPE_STEPS + 1.0).ceil();
        let spread = if spread >= f32::from(u16::MAX) {
            u16::MAX
        } else {
            spread as u16
        };
        let [patch_low, patch_high] = self.patch_bounds[(z * PATCHES_PER_TILE + x) as usize];
        let range = patch_high - patch_low;
        let half = patch / 2;
        let mut floors = [0_u8; 4];
        for (quarter, floor) in floors.iter_mut().enumerate() {
            let (qx, qz) = (
                first_x + (quarter as i32 % 2) * half,
                first_z + (quarter as i32 / 2) * half,
            );
            let mut lowest = f32::INFINITY;
            for sz in qz..=qz + half {
                for sx in qx..=qx + half {
                    lowest = min(lowest, self.height(sx, sz));
                }
            }
            *floor = if range > 0.0 {
                (clamp(((lowest - patch_low) / range * 255.0).floor(), 0.0, 255.0) as u8)
                    .saturating_sub(1)
            } else {
                0
            };
        }
        PatchCulling {
            slope,
            spread,
            floors,
        }
    }

    pub fn with_control(mut self, control: Box<[u32]>) -> Option<Self> {
        let stride = TILE_STRIDE as usize;
        (control.len() == stride * stride).then(|| {
            self.control = Some(control);
            self
        })
    }

    pub fn river_edges(&self) -> Option<&[u16]> {
        self.river_edges.as_deref()
    }

    #[must_use]
    fn with_lake(mut self, mut surface: Vec<f32>) -> Self {
        if !surface.iter().zip(&self.heights[..]).any(|(s, h)| s > h) {
            return self;
        }
        extend_lake(&mut surface, &self.heights);
        self.lake = Some(Arc::from(surface));
        self
    }

    pub(crate) fn water_surfaces(&self) -> Option<&[f32]> {
        self.lake.as_deref()
    }

    pub(crate) fn lake_depth(&self, local: Vec2) -> f32 {
        let (Some(surface), Some((i, j, fu, fv))) = (self.lake.as_deref(), self.cell(local)) else {
            return 0.0;
        };
        let stride = TILE_STRIDE as i32;
        let at = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)];
        let surfaces = at.map(|(x, z)| surface[((z + 1) * stride + x + 1) as usize]);
        let corners = if fu + fv <= 1.0 { [0, 1, 2] } else { [3, 2, 1] };
        if corners.iter().any(|&corner| !surfaces[corner].is_finite()) {
            return 0.0;
        }
        let surface = on_triangle(surfaces, fu, fv);
        let depth = surface - on_triangle(at.map(|(x, z)| self.height(x, z)), fu, fv);
        if depth <= 0.0 { 0.0 } else { depth }
    }

    pub fn ground_errors(&self) -> &ErrorPyramid {
        &self.ground_errors
    }

    pub fn key(&self) -> TileKey {
        self.key
    }

    pub fn spacing(&self) -> f32 {
        self.spacing
    }

    pub fn control(&self) -> Option<&[u32]> {
        self.control.as_deref()
    }

    pub fn placed(&self) -> &[Placed] {
        self.placed.as_deref().unwrap_or(&[])
    }

    pub fn placing(&self) -> Option<Placing> {
        self.placing
    }

    #[must_use]
    pub fn with_placed(mut self, placed: Vec<Placed>, placing: Placing) -> Self {
        self.placed = (!placed.is_empty()).then(|| Arc::from(placed));
        self.placing = Some(placing);
        self
    }

    #[must_use]
    pub fn without_placed(mut self) -> Self {
        self.placed = None;
        self.placing = None;
        self
    }

    pub fn control_at(&self, x: i32, z: i32) -> Option<Control> {
        let stride = TILE_STRIDE as i32;
        let x = x.clamp(-1, stride - 2);
        let z = z.clamp(-1, stride - 2);
        self.control
            .as_ref()
            .map(|control| Control::unpack(control[((z + 1) * stride + x + 1) as usize]))
    }

    #[inline]
    pub fn height(&self, x: i32, z: i32) -> f32 {
        let stride = TILE_STRIDE as i32;
        self.heights[((z + 1) * stride + x + 1) as usize]
    }

    pub fn morph_error(&self, parent: &Tile) -> f32 {
        let offset = [
            i32::from(self.key.x % 2) * 128,
            i32::from(self.key.z % 2) * 128,
        ];
        let mut largest = 0.0_f32;
        for z in (0..=TILE_QUADS as i32).step_by(3) {
            for x in (0..=TILE_QUADS as i32).step_by(3) {
                let (px, pz) = (offset[0] * 2 + x, offset[1] * 2 + z);
                let (x0, z0) = (px.div_euclid(2), pz.div_euclid(2));
                let (fx, fz) = ((px % 2) as f32 * 0.5, (pz % 2) as f32 * 0.5);
                let a = parent.height(x0, z0);
                let b = parent.height(x0 + 1, z0);
                let c = parent.height(x0, z0 + 1);
                let d = parent.height(x0 + 1, z0 + 1);
                let coarse = (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fz;
                largest = max(largest, (self.height(x, z) - coarse).abs());
            }
        }
        largest
    }

    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    pub fn bounds(&self) -> [f32; 2] {
        self.bounds
    }

    pub fn patch_bounds(&self, x: u32, z: u32) -> [f32; 2] {
        self.patch_bounds[(z * PATCHES_PER_TILE + x) as usize]
    }

    pub fn patch_slopes(&self, x: u32, z: u32) -> [f32; 3] {
        let cull = self.patch_culling[(z * PATCHES_PER_TILE + x) as usize];
        [
            f32::from(cull.slope[0]) / SLOPE_STEPS,
            f32::from(cull.slope[1]) / SLOPE_STEPS,
            f32::from(cull.spread) / SLOPE_STEPS,
        ]
    }

    pub fn patch_floors(&self, x: u32, z: u32) -> [f32; 4] {
        let index = (z * PATCHES_PER_TILE + x) as usize;
        let [low, high] = self.patch_bounds[index];
        self.patch_culling[index]
            .floors
            .map(|floor| min(low + (high - low) * (f32::from(floor) / 255.0), high))
    }

    pub fn patch_has_holes(&self, x: u32, z: u32) -> bool {
        let index = (z * PATCHES_PER_TILE + x) as usize;
        self.patch_holes[index / 64] & (1 << (index % 64)) != 0
    }

    pub fn hole_bits(&self) -> Option<&[u64]> {
        self.holes.as_deref()
    }

    pub fn patch_holes(&self) -> &[u64; 4] {
        &self.patch_holes
    }

    pub fn is_hole(&self, x: u32, z: u32) -> bool {
        let Some(bits) = &self.holes else {
            return false;
        };
        if x as usize >= HOLE_CELLS || z as usize >= HOLE_CELLS {
            return false;
        }
        let index = z as usize * HOLE_CELLS + x as usize;
        bits[index / 64] & (1 << (index % 64)) != 0
    }

    fn cell(&self, local: Vec2) -> Option<(i32, i32, f32, f32)> {
        let size = TILE_QUADS as f32;
        let u = local.x / self.spacing;
        let v = local.y / self.spacing;
        if !(u.is_finite() && v.is_finite()) || u < 0.0 || v < 0.0 || u > size || v > size {
            return None;
        }
        let i = (u.floor() as i32).min(TILE_QUADS as i32 - 1);
        let j = (v.floor() as i32).min(TILE_QUADS as i32 - 1);
        Some((i, j, u - i as f32, v - j as f32))
    }

    pub fn sample(&self, local: Vec2) -> Option<TileSample> {
        let (i, j, fu, fv) = self.cell(local)?;
        let corners = [
            self.height(i, j),
            self.height(i + 1, j),
            self.height(i, j + 1),
            self.height(i + 1, j + 1),
        ];
        let around: [[f32; 4]; 4] =
            [-1, 0, 1, 2].map(|dz| [-1, 0, 1, 2].map(|dx| self.height(i + dx, j + dz)));
        let gradient = |x: usize, z: usize| {
            let [gx, gz] = smooth_gradient(
                [0, 1, 2].map(|row| [0, 1, 2].map(|column| around[z + row][x + column])),
                self.spacing,
            );
            Vec2::new(gx, gz)
        };
        let blend = |a: Vec2, b: Vec2, t: f32| a + (b - a) * t;
        let slope = blend(
            blend(gradient(0, 0), gradient(1, 0), fu),
            blend(gradient(0, 1), gradient(1, 1), fu),
            fv,
        );
        let normal = Vec3::new(-slope.x, 1.0, -slope.y).normalize();
        Some(TileSample {
            height: on_triangle(corners, fu, fv),
            normal,
            hole: self.is_hole(i as u32, j as u32),
        })
    }
}

const LAKE_REACH: u8 = 4;
const NO_RING: u8 = u8::MAX;
const QUEUED: u8 = u8::MAX - 1;
const STOPPED: u8 = u8::MAX - 2;

fn extend_lake(surface: &mut [f32], heights: &[f32]) {
    let stride = TILE_STRIDE as usize;
    let mut rings: Vec<u8> = surface
        .iter()
        .zip(heights)
        .map(|(level, ground)| if level > ground { 0 } else { NO_RING })
        .collect();
    for (at, &ring) in rings.iter().enumerate() {
        if ring == NO_RING {
            surface[at] = f32::NAN;
        }
    }
    let mut inside: Vec<usize> = (0..rings.len()).filter(|&at| rings[at] == 0).collect();
    for ring in 1..=LAKE_REACH {
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
        let made: Vec<(f32, bool)> = queued
            .iter()
            .map(|&at| continued_level(at, ring - 1, (&*surface, heights), &rings))
            .collect();
        inside.clear();
        for (&at, &(level, stops)) in queued.iter().zip(&made) {
            surface[at] = level;
            rings[at] = if stops { STOPPED } else { ring };
            if !stops {
                inside.push(at);
            }
        }
    }
}

fn continued_level(
    at: usize,
    inner: u8,
    (surface, heights): (&[f32], &[f32]),
    rings: &[u8],
) -> (f32, bool) {
    let stride = TILE_STRIDE as usize;
    let (x, z) = (at % stride, at / stride);
    let (mut level, mut apart, mut count) = (0.0, 0.0, 0);
    for nz in z.saturating_sub(1)..(z + 2).min(stride) {
        for nx in x.saturating_sub(1)..(x + 2).min(stride) {
            let near = nz * stride + nx;
            if rings[near] != inner {
                continue;
            }
            level += surface[near];
            apart += (surface[near] - heights[near]).abs();
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
    (standing, stops)
}

#[inline]
pub(crate) fn on_triangle([v00, v10, v01, v11]: [f32; 4], fu: f32, fv: f32) -> f32 {
    if fu + fv <= 1.0 {
        v00 + fu * (v10 - v00) + fv * (v01 - v00)
    } else {
        v11 + (1.0 - fu) * (v01 - v11) + (1.0 - fv) * (v10 - v11)
    }
}

struct Prepared(Mutex<Box<dyn LayerTile>>);

impl Prepared {
    fn tile(&mut self) -> &mut Box<dyn LayerTile> {
        self.0.get_mut().unwrap_or_else(PoisonError::into_inner)
    }
}

#[derive(Clone, Copy, Debug)]
enum Stage {
    Prepare { layer: usize },
    Rivers { layer: usize },
    Samples { row: i32 },
    Beds,
    Holes { row: u32 },
    Biomes { row: usize },
    Done,
    Place { layer: usize, row: usize },
    Failed,
}

pub struct Generator {
    layers: Arc<Layers>,
    key: TileKey,
    spacing: f32,
    first: [i64; 2],
    tiles: Vec<Option<Prepared>>,
    stage: Stage,
    heights: Box<[f32]>,
    holes: Option<Box<[u64]>>,
    xs: Box<[f32]>,
    row: Box<[f32]>,
    min: Vec2,
    max: Vec2,
    surface: Option<Arc<BiomeRules>>,
    control: Option<Box<[u32]>>,
    tile: Option<Tile>,
    placed: Vec<Placed>,
    rasters: Vec<Option<RiverRaster>>,
    river_edges: Option<Box<[u16]>>,
    river_beds: Option<Box<[u16]>>,
    zs: Box<[f32]>,
    inputs: SurfaceInputs,
    places: crate::placement::PlaceScratch,
    materials: bool,
}

#[derive(Default)]
pub(crate) struct SurfaceInputs {
    zs: Vec<f32>,
    height: Vec<f32>,
    gradient: Vec<[f32; 2]>,
    curvature: Vec<f32>,
    water: Vec<HydrologySample>,
    erosion: Vec<f32>,
    path: Vec<PathSample>,
    scratch: RowScratch,
    weights: Vec<(u8, f32)>,
}

impl SurfaceInputs {
    fn clear(&mut self, len: usize) {
        for row in [&mut self.height, &mut self.curvature] {
            row.clear();
            row.reserve(len);
        }
        self.gradient.clear();
        self.gradient.reserve(len);
        self.water.clear();
        self.path.clear();
        self.erosion.clear();
        self.erosion.resize(len, 0.0);
    }
}

pub(crate) fn control_row(
    (rules, layers): (&BiomeRules, &Layers),
    (heights, rivers): (&[f32], beds::RiverRows<'_>),
    (xs, z, spacing): (&[f32], f32, f32),
    (row, columns): (usize, Range<usize>),
    (out, inputs): (&mut [u32], &mut SurfaceInputs),
) {
    let stride = TILE_STRIDE as usize;
    let reads = rules.reads();
    let (up, down) = (row.saturating_sub(1), (row + 1).min(stride - 1));
    inputs.clear(columns.len());
    if reads.erosion {
        erosion_row(
            layers,
            &xs[columns.clone()],
            z,
            spacing,
            &mut inputs.erosion,
        );
    }
    for (index, column) in columns.clone().enumerate() {
        let held = RiverInputs::held(rivers, row * stride + column);
        let (left, right) = (column.saturating_sub(1), (column + 1).min(stride - 1));
        let at = |r: usize, c: usize| heights[r * stride + c];
        let height = at(row, column);
        let shape = SampleShape::of(
            [up, row, down].map(|r| [left, column, right].map(|c| at(r, c))),
            spacing,
        );
        let x = xs[column];
        let shown = layers.shown(x, z);
        let water = match layers.hydrology() {
            Some(water) if shown > 0.0 && reads.water => held.with_maps(water, [x, z], spacing),
            _ => HydrologySample::default(),
        };
        inputs.height.push(height);
        inputs.gradient.push(shape.gradient);
        inputs.curvature.push(shape.curvature);
        inputs.water.push(HydrologySample {
            flow: water.flow * shown,
            wetness: water.wetness * shown,
            lake: water.lake * shown,
            ..water
        });
        inputs.erosion[index] *= shown;
        inputs.path.push(if shown > 0.0 && reads.path {
            layers.path_at(x, z)
        } else {
            PathSample::FAR
        });
    }
    inputs.zs.clear();
    inputs.zs.resize(columns.len(), z);
    let row_inputs = RowInputs {
        xs: &xs[columns],
        zs: &inputs.zs,
        spacing,
        height: &inputs.height,
        gradient: &inputs.gradient,
        curvature: &inputs.curvature,
        water: &inputs.water,
        erosion: &inputs.erosion,
        path: &inputs.path,
    };
    rules.control_row(&row_inputs, out, (&mut inputs.scratch, &mut inputs.weights));
}

impl Generator {
    pub fn new(layers: Arc<Layers>, pyramid: &Pyramid, key: TileKey) -> Self {
        let level = u32::from(key.level);
        let spacing = pyramid.spacing(level);
        let [x, z] = pyramid.first_sample(key);
        let [dx, dz] = layers.offset_samples(spacing);
        let first = [x + dx, z + dz];
        let stride = TILE_STRIDE as usize;
        let xs: Box<[f32]> = (-1..=TILE_SAMPLES as i64)
            .map(|i| (first[0] + i) as f32 * spacing)
            .collect();
        let zs: Box<[f32]> = (-1..=TILE_SAMPLES as i64)
            .map(|i| (first[1] + i) as f32 * spacing)
            .collect();
        let min = Vec2::new(
            (first[0] - 1) as f32 * spacing,
            (first[1] - 1) as f32 * spacing,
        );
        let max = Vec2::new(
            (first[0] + i64::from(TILE_SAMPLES)) as f32 * spacing,
            (first[1] + i64::from(TILE_SAMPLES)) as f32 * spacing,
        );
        let holes = layers
            .holes_reach(min, max)
            .then(|| vec![0; HOLE_WORDS].into_boxed_slice());
        let frame = TileFrame {
            key,
            first,
            spacing,
            xs: &xs,
            zs: &zs,
            low: min,
            high: max,
        };
        let tiles = layers
            .layers()
            .iter()
            .map(|layer| layer.tile(&frame).map(|tile| Prepared(Mutex::new(tile))))
            .collect();
        Self {
            layers,
            key,
            spacing,
            first,
            tiles,
            stage: Stage::Prepare { layer: 0 },
            heights: vec![0.0; stride * stride].into_boxed_slice(),
            holes,
            xs,
            row: vec![0.0; stride].into_boxed_slice(),
            min,
            max,
            surface: None,
            control: None,
            tile: None,
            placed: Vec::new(),
            rasters: Vec::new(),
            river_edges: None,
            river_beds: None,
            zs,
            inputs: SurfaceInputs::default(),
            places: crate::placement::PlaceScratch::default(),
            materials: true,
        }
    }

    #[must_use]
    pub fn with_materials(mut self, materials: bool) -> Self {
        self.materials = materials;
        self
    }

    const HEIGHT_ROWS: (i32, i32) = (-1, TILE_SAMPLES as i32 + 1);

    #[must_use]
    pub fn with_surface(mut self, surface: Arc<BiomeRules>) -> Self {
        self.surface = Some(surface);
        self
    }

    pub fn key(&self) -> TileKey {
        self.key
    }

    pub fn failed(&self) -> bool {
        matches!(self.stage, Stage::Failed)
    }

    fn after_holes(&self) -> Stage {
        match &self.surface {
            Some(_) if self.materials => Stage::Biomes { row: 0 },
            _ => Stage::Done,
        }
    }

    fn surface_row(&mut self, row: usize) {
        let stride = TILE_STRIDE as usize;
        let Some(surface) = self.surface.clone() else {
            return;
        };
        let control = self
            .control
            .get_or_insert_with(|| vec![0; stride * stride].into_boxed_slice());
        let z = (self.first[1] + row as i64 - 1) as f32 * self.spacing;
        control_row(
            (&surface, &self.layers),
            (
                &self.heights,
                (self.river_edges.as_deref(), self.river_beds.as_deref()),
            ),
            (&self.xs, z, self.spacing),
            (row, 0..stride),
            (
                &mut control[row * stride..(row + 1) * stride],
                &mut self.inputs,
            ),
        );
    }

    pub fn advance(&mut self, budget: &mut u64) -> Option<Tile> {
        loop {
            if *budget == 0 {
                return None;
            }
            match self.stage {
                Stage::Prepare { layer } => {
                    let Some(prepared) = self.tiles.get_mut(layer) else {
                        self.stage = Stage::Rivers { layer: 0 };
                        continue;
                    };
                    if prepared
                        .as_mut()
                        .is_none_or(|prepared| prepared.tile().prepare(budget))
                    {
                        self.stage = Stage::Prepare { layer: layer + 1 };
                    }
                }
                Stage::Rivers { layer } => self.stage = self.rasterize(layer, budget),
                Stage::Samples { row } => {
                    self.sample_row(row);
                    let layers = self
                        .layers
                        .layers()
                        .iter()
                        .map(|layer| layer.row_cost())
                        .sum::<u64>();
                    *budget = budget.saturating_sub(u64::from(TILE_STRIDE) * layers.max(1));
                    self.stage = if row + 1 < Self::HEIGHT_ROWS.1 {
                        Stage::Samples { row: row + 1 }
                    } else {
                        Stage::Beds
                    };
                }
                Stage::Beds => self.stage = self.beds(budget),
                Stage::Holes { row } => {
                    let z = ((self.first[1] + i64::from(row)) as f32 + 0.5) * self.spacing;
                    let bits = self.holes.as_mut().expect("holes_stage_bits");
                    for column in 0..HOLE_CELLS as u32 {
                        let x = ((self.first[0] + i64::from(column)) as f32 + 0.5) * self.spacing;
                        if self.layers.hole_at(x, z) {
                            let index = row as usize * HOLE_CELLS + column as usize;
                            bits[index / 64] |= 1 << (index % 64);
                        }
                    }
                    *budget = budget.saturating_sub(HOLE_CELLS as u64);
                    self.stage = if row + 1 < HOLE_CELLS as u32 {
                        Stage::Holes { row: row + 1 }
                    } else {
                        self.after_holes()
                    };
                }
                Stage::Biomes { row } => {
                    self.surface_row(row);
                    let cost = self.surface.as_ref().map_or(1, |surface| surface.cost());
                    *budget = budget.saturating_sub(u64::from(TILE_STRIDE) * cost);
                    self.stage = if row + 1 < TILE_STRIDE as usize {
                        Stage::Biomes { row: row + 1 }
                    } else {
                        Stage::Done
                    };
                }
                Stage::Done => {
                    let lake = self.lake_rows(budget);
                    let heights = std::mem::take(&mut self.heights);
                    self.stage = Stage::Failed;
                    let tile = Tile::new(self.key, self.spacing, heights, self.holes.take())?;
                    let tile = match self.control.take() {
                        Some(control) => tile.with_control(control)?,
                        None => tile,
                    };
                    let tile = tile.with_rivers(self.river_edges.take(), self.river_beds.take())?;
                    let tile = match lake {
                        Some(surface) => tile.with_lake(surface),
                        None => tile,
                    };
                    self.stage = Stage::Done;
                    match &self.surface {
                        Some(surface) if surface.places(self.key.level, Placing::All) => {
                            self.tile = Some(tile);
                            self.stage = Stage::Place { layer: 0, row: 0 };
                        }
                        _ => return Some(tile),
                    }
                }
                Stage::Failed => return None,
                Stage::Place { layer, row } => {
                    let (Some(surface), Some(tile)) = (&self.surface, &self.tile) else {
                        self.stage = Stage::Failed;
                        return None;
                    };
                    let (cost, next) = surface.place_row(
                        tile,
                        (layer, row),
                        &mut self.placed,
                        (&self.layers, Placing::All, &mut self.places),
                    );
                    *budget = budget.saturating_sub(cost);
                    match next {
                        Some((layer, row)) => self.stage = Stage::Place { layer, row },
                        None => {
                            *budget = budget
                                .saturating_sub(surface.place_authored(tile, &mut self.placed));
                            let tile = self.tile.take()?;
                            let placed = std::mem::take(&mut self.placed);
                            return Some(tile.with_placed(placed, Placing::All));
                        }
                    }
                }
            }
        }
    }

    fn rasterize(&mut self, layer: usize, budget: &mut u64) -> Stage {
        if layer >= self.layers.layers().len() {
            for raster in &mut self.rasters {
                if raster.as_ref().is_some_and(|raster| !raster.reached()) {
                    *raster = None;
                }
            }
            self.river_edges = self
                .layers
                .hydrology_layer()
                .and_then(|(index, _)| self.rasters.get(index)?.as_ref()?.edges(river_step));
            return Stage::Samples {
                row: Self::HEIGHT_ROWS.0,
            };
        }
        let rivers = self.layers.rivers_of(layer);
        if self.rasters.len() <= layer {
            let raster = rivers
                .and_then(|rivers| RiverRaster::new(rivers, &self.xs, &self.zs, self.spacing));
            self.rasters.push(raster);
        }
        let done = match (&mut self.rasters[layer], rivers) {
            (Some(raster), Some(rivers)) => raster.stamp(rivers, budget),
            _ => true,
        };
        *budget = budget.saturating_sub(1);
        Stage::Rivers {
            layer: layer + usize::from(done),
        }
    }

    fn lake_rows(&mut self, budget: &mut u64) -> Option<Vec<f32>> {
        let stride = TILE_STRIDE as usize;
        let rasters = std::mem::take(&mut self.rasters);
        let (index, hydrology) = self.layers.hydrology_layer()?;
        let raster = rasters.get(index).and_then(Option::as_ref);
        let count = stride * stride;
        let mut surface = vec![f32::NAN; count];
        for row in 0..stride {
            let z = self.zs[row];
            for column in 0..stride {
                let x = self.xs[column];
                if self.layers.shown(x, z) < 1.0 {
                    continue;
                }
                let water = raster.and_then(|raster| raster.water(&hydrology.rivers, row, column));
                let at = row * stride + column;
                match hydrology.water.lake_reach(x, z) {
                    Some((level, _)) if !water.is_some_and(|water| water.surface > level) => {
                        surface[at] = level;
                    }
                    _ => {
                        if let Some(water) = water {
                            surface[at] = water.surface;
                        }
                    }
                }
            }
        }
        *budget = budget.saturating_sub(count as u64);
        Some(surface)
    }

    fn sample_row(&mut self, row: i32) {
        let z = (self.first[1] + i64::from(row)) as f32 * self.spacing;
        self.row.fill(0.0);
        let at = Row {
            xs: &self.xs,
            z,
            spacing: self.spacing,
            low: self.min,
            high: self.max,
        };
        for (index, layer) in self.layers.layers().iter().enumerate() {
            let below = Below::new(&self.layers, index);
            if layer.makes_water() {
                if let Some(raster) = self.rasters.get(index).and_then(Option::as_ref) {
                    let line = (row + 1) as usize;
                    self.layers
                        .cut_rivers(raster, line, &mut self.row, &self.xs, z);
                }
            } else if let Some(prepared) = self.tiles.get_mut(index).and_then(Option::as_mut) {
                prepared.tile().apply_row(below, row, &at, &mut self.row);
            } else {
                layer.apply_row(below, &at, &mut self.row);
            }
        }
        let stride = TILE_STRIDE as usize;
        let start = (row + 1) as usize * stride;
        self.heights[start..start + stride].copy_from_slice(&self.row);
    }
}

impl crate::codec::Codec for Tile {
    fn write(&self, out: &mut crate::codec::Writer) {
        let stride = TILE_STRIDE as usize;
        out.put(&self.key).put(&self.spacing);
        crate::codec::put_grid(out, &self.heights, stride);
        out.put(&self.holes)
            .put(&self.patch_culling)
            .put(&self.control);
        for planes in [&self.river_edges, &self.river_beds] {
            match planes {
                Some(planes) => {
                    out.put(&true);
                    crate::codec::put_wide_planes(out, planes, stride);
                }
                None => {
                    out.put(&false);
                }
            };
        }
        out.put(&self.placed).put(&self.placing);
        match &self.lake {
            Some(surface) => {
                out.put(&true);
                crate::codec::put_grid(out, surface, stride);
            }
            None => {
                out.put(&false);
            }
        }
    }

    fn read(input: &mut crate::codec::Reader<'_>) -> Result<Self, crate::codec::Error> {
        let invalid = || crate::codec::invalid("$.tile");
        let stride = TILE_STRIDE as usize;
        let key = input.get()?;
        let spacing = input.get()?;
        let heights = crate::codec::get_grid(input, stride)?;
        let holes = input.get()?;
        let culling: Box<[PatchCulling]> = input.get()?;
        let control: Option<Box<[u32]>> = input.get()?;
        let mut rivers = [None, None];
        for planes in &mut rivers {
            let held: bool = input.get()?;
            if held {
                *planes = Some(crate::codec::get_wide_planes(input, stride)?);
            }
        }
        let [river_edges, river_beds] = rivers;
        let placed: Option<Arc<[Placed]>> = input.get()?;
        let placing = input.get()?;
        let watered: bool = input.get()?;
        let lake = if watered {
            let surface = crate::codec::get_grid(input, stride)?;
            if surface.len() != stride * stride {
                return Err(crate::codec::invalid("$.tile_water"));
            }
            Some(Arc::from(surface))
        } else {
            None
        };
        let mut tile = Self::shaped(key, spacing, heights, holes).ok_or_else(invalid)?;
        if culling.len() != tile.patch_bounds.len() {
            return Err(invalid());
        }
        tile.patch_culling = culling;
        if let Some(control) = control {
            tile = tile.with_control(control).ok_or_else(invalid)?;
        }
        tile = tile
            .with_rivers(river_edges, river_beds)
            .ok_or_else(invalid)?;
        tile.placed = placed;
        tile.placing = placing;
        tile.lake = lake;
        Ok(tile)
    }
}
