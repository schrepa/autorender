use std::ops::Range;
use std::sync::Arc;

use glam::{Vec2, Vec3, Vec4};

use engine_math::{clamp, max, min};

use crate::hydrology::RIVER_FAR;
use crate::layers::{Layer, Layers, state_len};
use crate::rivers::RiverRaster;
use crate::surface::{
    BiomeRules, Control, RowInputs, RowScratch, SampleShape, erosion_row, smooth_gradient,
};
use crate::tile_water::{TileWater, WaterMesh, WaterPoint, WaterState};
use crate::{
    ErrorPyramid, HydrologySample, PATCH_QUADS, PATCHES_PER_TILE, PathSample, Placing, Pyramid,
    TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey, TileShading,
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
    placed: Option<Arc<[crate::Placed]>>,
    placing: Option<Placing>,
    water: Option<TileWater>,
    water_mesh: Option<Arc<WaterMesh>>,
    ground_errors: Arc<ErrorPyramid>,
    water_errors: Option<Arc<ErrorPyramid>>,
    shading: Option<Arc<TileShading>>,
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
            water: None,
            water_mesh: None,
            ground_errors,
            water_errors: None,
            shading: None,
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
    pub(crate) fn with_water(
        mut self,
        surface: Box<[f32]>,
        info: Box<[[u32; 2]]>,
        solids: &[crate::Footprint],
    ) -> Self {
        let tile = (&self.heights[..], self.spacing, self.origin());
        let water = TileWater::new(surface, info, tile, solids);
        self.hold(water);
        self
    }

    fn hold(&mut self, water: Option<TileWater>) {
        let mesh = water.as_ref().map(|water| water.mesh(&self.heights));
        self.water_errors = mesh
            .as_ref()
            .map(|mesh| Arc::new(ErrorPyramid::water(mesh)));
        self.water_mesh = mesh.map(Arc::new);
        self.water = water;
    }

    pub fn water(&self) -> Option<&TileWater> {
        self.water.as_ref()
    }

    pub fn ground_errors(&self) -> &ErrorPyramid {
        &self.ground_errors
    }

    pub fn water_mesh(&self) -> Option<&WaterMesh> {
        self.water_mesh.as_deref()
    }

    pub fn water_errors(&self) -> &ErrorPyramid {
        match &self.water_errors {
            Some(errors) => errors,
            None => ErrorPyramid::dry(),
        }
    }

    #[must_use]
    pub fn holding(&self, water: Option<TileWater>) -> Self {
        let mut tile = Self {
            key: self.key,
            spacing: self.spacing,
            heights: self.heights.clone(),
            holes: self.holes.clone(),
            control: self.control.clone(),
            river_edges: self.river_edges.clone(),
            river_beds: self.river_beds.clone(),
            placed: self.placed.clone(),
            placing: self.placing,
            water: None,
            water_mesh: None,
            ground_errors: Arc::clone(&self.ground_errors),
            water_errors: None,
            shading: self.shading.clone(),
            bounds: self.bounds,
            patch_bounds: self.patch_bounds.clone(),
            patch_holes: self.patch_holes,
            patch_culling: self.patch_culling.clone(),
        };
        tile.hold(water);
        tile
    }

    pub fn shading(&self) -> Option<&Arc<TileShading>> {
        self.shading.as_ref()
    }

    #[must_use]
    pub fn shaded(&self, shading: Option<Arc<TileShading>>) -> Self {
        Self {
            shading,
            ..self.clone()
        }
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

    pub fn placed(&self) -> &[crate::Placed] {
        self.placed.as_deref().unwrap_or(&[])
    }

    pub fn placing(&self) -> Option<Placing> {
        self.placing
    }

    #[must_use]
    pub fn with_placed(mut self, placed: Vec<crate::Placed>, placing: Placing) -> Self {
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

    pub fn water_at(&self, local: Vec2) -> Option<WaterPoint> {
        let water = self.water.as_ref()?;
        let (i, j, fu, fv) = self.cell(local)?;
        let at = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)];
        let surfaces = at.map(|(x, z)| water.surface(x, z));
        let corners = if fu + fv <= 1.0 { [0, 1, 2] } else { [3, 2, 1] };
        if corners.iter().any(|&corner| !surfaces[corner].is_finite()) {
            return None;
        }
        let surface = on_triangle(surfaces, fu, fv);
        let depth = surface - on_triangle(at.map(|(x, z)| self.height(x, z)), fu, fv);
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
            state: WaterState {
                turbulence: on_triangle(states.map(|state| state.turbulence), fu, fv),
                breaking: on_triangle(states.map(|state| state.breaking), fu, fv),
                shear: on_triangle(states.map(|state| state.shear), fu, fv),
                reach: states[corners[0]].reach,
            },
        })
    }

    pub fn bed_word(&self, local: Vec2) -> u32 {
        let (Some(water), Some((i, j, fu, fv))) = (self.water.as_ref(), self.cell(local)) else {
            return 0;
        };
        water.bed_word(i + i32::from(fu >= 0.5), j + i32::from(fv >= 0.5))
    }
}

#[inline]
pub(crate) fn on_triangle([v00, v10, v01, v11]: [f32; 4], fu: f32, fv: f32) -> f32 {
    if fu + fv <= 1.0 {
        v00 + fu * (v10 - v00) + fv * (v01 - v00)
    } else {
        v11 + (1.0 - fu) * (v01 - v11) + (1.0 - fv) * (v10 - v11)
    }
}

const GRID: usize = 69;
const GRID_FIRST: i64 = -2;
const GRID_STEP: i64 = 4;
const COARSE_COST: u64 = 24;
const NOISE_SAMPLE_COST: u64 = 4;

#[inline(always)]
fn interpolate(out: &mut [f32], weights: [f32; 4], taps: [&[f32]; 4]) {
    let [a, b, c, d] = taps;
    for ((((value, a), b), c), d) in out.iter_mut().zip(a).zip(b).zip(c).zip(d) {
        *value = weights[0] * a + weights[1] * b + weights[2] * c + weights[3] * d;
    }
}

const QUARTER_WEIGHTS: [[f32; 4]; 4] = [
    [0.0, 1.0, 0.0, 0.0],
    [-0.070_312_5, 0.867_187_5, 0.226_562_5, -0.023_437_5],
    [-0.0625, 0.5625, 0.5625, -0.0625],
    [-0.023_437_5, 0.226_562_5, 0.867_187_5, -0.070_312_5],
];

const LANE_WEIGHTS: [Vec4; 4] = {
    let w = QUARTER_WEIGHTS;
    [
        Vec4::new(w[0][0], w[1][0], w[2][0], w[3][0]),
        Vec4::new(w[0][1], w[1][1], w[2][1], w[3][1]),
        Vec4::new(w[0][2], w[1][2], w[2][2], w[3][2]),
        Vec4::new(w[0][3], w[1][3], w[2][3], w[3][3]),
    ]
};

#[inline(always)]
fn interpolate4(out: &mut [Vec4], taps: [&[f32]; 4]) {
    let [a, b, c, d] = taps;
    let [wa, wb, wc, wd] = LANE_WEIGHTS;
    for ((((value, &a), &b), &c), &d) in out.iter_mut().zip(a).zip(b).zip(c).zip(d) {
        *value = wa * a + wb * b + wc * c + wd * d;
    }
}

#[derive(Clone, Copy, Debug)]
enum Stage {
    Coarse { layer: usize, row: usize },
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
    bands: Vec<Option<(usize, Range<usize>, usize)>>,
    grids: Vec<Option<Box<[f32]>>>,
    stage: Stage,
    heights: Box<[f32]>,
    holes: Option<Box<[u64]>>,
    xs: Box<[f32]>,
    row: Box<[f32]>,
    values: Box<[f32]>,
    column: Box<[f32]>,
    state: Box<[f32]>,
    lanes: Box<[Vec4]>,
    min: Vec2,
    max: Vec2,
    surface: Option<Arc<BiomeRules>>,
    control: Option<Box<[u32]>>,
    tile: Option<Tile>,
    placed: Vec<crate::Placed>,
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
        let bands: Vec<_> = (0..layers.layers().len())
            .map(|index| layers.noise(index).map(|tables| tables.bands(spacing)))
            .collect();
        let lens: Vec<_> = bands
            .iter()
            .map(|bands| bands.as_ref().map(|(coarse, _, _)| state_len(*coarse)))
            .collect();
        let grids = lens
            .iter()
            .map(|len| len.map(|len| vec![0.0; GRID * GRID * len].into_boxed_slice()))
            .collect();
        let widest = lens.iter().flatten().copied().max().unwrap_or(0);
        let xs = (-1..=TILE_SAMPLES as i64)
            .map(|i| (first[0] + i) as f32 * spacing)
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
        Self {
            layers,
            key,
            spacing,
            first,
            bands,
            grids,
            stage: Stage::Coarse { layer: 0, row: 0 },
            heights: vec![0.0; stride * stride].into_boxed_slice(),
            holes,
            xs,
            row: vec![0.0; stride].into_boxed_slice(),
            values: vec![0.0; stride].into_boxed_slice(),
            column: vec![0.0; GRID * widest].into_boxed_slice(),
            state: vec![0.0; widest].into_boxed_slice(),
            lanes: vec![Vec4::ZERO; widest].into_boxed_slice(),
            min,
            max,
            surface: None,
            control: None,
            tile: None,
            placed: Vec::new(),
            rasters: Vec::new(),
            river_edges: None,
            river_beds: None,
            zs: (-1..=TILE_SAMPLES as i64)
                .map(|i| (first[1] + i) as f32 * spacing)
                .collect(),
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

    fn grid_rows() -> (usize, usize) {
        let (first, end) = Self::HEIGHT_ROWS;
        let low = i64::from(first).div_euclid(GRID_STEP) - GRID_FIRST - 1;
        let high = i64::from(end - 1).div_euclid(GRID_STEP) - GRID_FIRST + 3;
        (low as usize, std::cmp::min(high as usize, GRID))
    }

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

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub fn advance(&mut self, budget: &mut u64) -> Option<Tile> {
        loop {
            if *budget == 0 {
                return None;
            }
            match self.stage {
                Stage::Coarse { layer, row } => {
                    if layer >= self.grids.len() {
                        self.stage = Stage::Rivers { layer: 0 };
                        continue;
                    }
                    let Some(tables) = self.layers.noise(layer) else {
                        self.stage = Stage::Coarse {
                            layer: layer + 1,
                            row: Self::grid_rows().0,
                        };
                        continue;
                    };
                    let (coarse, _, gullies) = self.bands[layer].clone().unwrap_or_default();
                    let z = (self.first[1] + (GRID_FIRST + row as i64) * GRID_STEP) as f32
                        * self.spacing;
                    let grid = self.grids[layer].as_mut().expect("noise_layer_grid");
                    let len = state_len(coarse);
                    let (first, spacing) = (self.first[0], self.spacing);
                    let x = |column: usize| {
                        (first + (GRID_FIRST + column as i64) * GRID_STEP) as f32 * spacing
                    };
                    let mut column = 0;
                    while column < GRID {
                        let at = (row * GRID + column) * len;
                        if column + 4 <= GRID {
                            let xs = Vec4::from_array([0, 1, 2, 3].map(|lane| x(column + lane)));
                            tables.coarse4(xs, z, coarse, gullies, &mut grid[at..at + 4 * len]);
                            column += 4;
                        } else {
                            tables.coarse(x(column), z, coarse, gullies, &mut grid[at..at + len]);
                            column += 1;
                        }
                    }
                    *budget = budget.saturating_sub(GRID as u64 * COARSE_COST);
                    let (first, end) = Self::grid_rows();
                    self.stage = if row + 1 < end {
                        Stage::Coarse {
                            layer,
                            row: row + 1,
                        }
                    } else {
                        Stage::Coarse {
                            layer: layer + 1,
                            row: first,
                        }
                    };
                }
                Stage::Rivers { layer } => self.stage = self.rasterize(layer, budget),
                Stage::Samples { row } => {
                    self.sample_row(row);
                    let layers = self
                        .grids
                        .iter()
                        .map(|grid| if grid.is_some() { NOISE_SAMPLE_COST } else { 1 })
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
                    let water = self.water_rows(budget);
                    let heights = std::mem::take(&mut self.heights);
                    self.stage = Stage::Failed;
                    let tile = Tile::new(self.key, self.spacing, heights, self.holes.take())?;
                    let tile = match self.control.take() {
                        Some(control) => tile.with_control(control)?,
                        None => tile,
                    };
                    let tile = tile.with_rivers(self.river_edges.take(), self.river_beds.take())?;
                    let tile = match water {
                        Some((surface, info)) => tile.with_water(surface.into(), info.into(), &[]),
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
                .and_then(|(index, _)| self.rasters.get(index)?.as_ref()?.edges());
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

    fn water_rows(&mut self, budget: &mut u64) -> Option<(Vec<f32>, Vec<[u32; 2]>)> {
        let stride = TILE_STRIDE as usize;
        let rasters = std::mem::take(&mut self.rasters);
        let (index, hydrology) = self.layers.hydrology_layer()?;
        let raster = rasters.get(index).and_then(Option::as_ref);
        let count = stride * stride;
        let (mut surface, mut info) = (vec![f32::NAN; count], vec![[0_u32; 2]; count]);
        crate::tile_water::water_rows(
            &hydrology.water,
            &|row, column| raster?.water(&hydrology.rivers, row, column),
            (0..stride, &|_| 0..stride),
            (&self.xs, &self.zs),
            &|x, z| self.layers.shown(x, z),
            (&mut surface, &mut info),
        );
        *budget = budget.saturating_sub(count as u64);
        Some((surface, info))
    }

    fn sample_row(&mut self, row: i32) {
        let z = (self.first[1] + i64::from(row)) as f32 * self.spacing;
        self.row.fill(0.0);
        let coarse_row = i64::from(row).div_euclid(GRID_STEP);
        let vertical = QUARTER_WEIGHTS[i64::from(row).rem_euclid(GRID_STEP) as usize];
        for layer in 0..self.layers.layers().len() {
            let Some(tables) = self.layers.noise(layer) else {
                if matches!(self.layers.layers()[layer], Layer::Hydrology(_)) {
                    if let Some(raster) = self.rasters.get(layer).and_then(Option::as_ref) {
                        let at = (row + 1) as usize;
                        self.layers
                            .cut_rivers(raster, at, &mut self.row, &self.xs, z);
                    }
                } else {
                    self.layers.apply_local(
                        layer,
                        &mut self.row,
                        &self.xs,
                        z,
                        (self.min, self.max),
                        self.spacing,
                    );
                }
                continue;
            };
            let (octaves, fine, _) = self.bands[layer].clone().unwrap_or_default();
            let len = state_len(octaves);
            let grid = self.grids[layer].as_ref().expect("noise_layer_grid");
            let base = (coarse_row - 1 - GRID_FIRST) as usize;
            let tap = |row: usize, column: usize| {
                let at = ((base + row) * GRID + column) * len;
                &grid[at..at + len]
            };
            for (column, sum) in self.column.chunks_exact_mut(len).take(GRID).enumerate() {
                let taps = [0, 1, 2, 3].map(|row| tap(row, column));
                interpolate(sum, vertical, taps);
            }
            let sum = &mut self.state[..len];
            let lanes = &mut self.lanes[..len];
            let samples = self.values.len();
            let mut index = 0;
            while index < samples {
                let sample = index as i64 - 1;
                let coarse = sample.div_euclid(GRID_STEP);
                let quarter = sample.rem_euclid(GRID_STEP) as usize;
                let first = (coarse - 1 - GRID_FIRST) as usize;
                let taps = [0, 1, 2, 3].map(|tap| {
                    let at = (first + tap) * len;
                    &self.column[at..at + len]
                });
                if quarter == 0 && index + 4 <= samples {
                    interpolate4(lanes, taps);
                    let xs = Vec4::from_slice(&self.xs[index..index + 4]);
                    tables
                        .finish4(xs, z, lanes, octaves, fine.clone())
                        .write_to_slice(&mut self.values[index..index + 4]);
                    index += 4;
                } else {
                    interpolate(sum, QUARTER_WEIGHTS[quarter], taps);
                    self.values[index] =
                        tables.finish(self.xs[index], z, sum, octaves, fine.clone());
                    index += 1;
                }
            }
            self.layers.apply_noise(layer, &mut self.row, &self.values);
        }
        let stride = TILE_STRIDE as usize;
        let start = (row + 1) as usize * stride;
        self.heights[start..start + stride].copy_from_slice(&self.row);
    }
}

impl Layers {
    pub fn height_cost(&self) -> u64 {
        (0..self.layers().len())
            .map(|layer| {
                if self.noise(layer).is_some() {
                    16 * COARSE_COST + NOISE_SAMPLE_COST
                } else {
                    1
                }
            })
            .sum()
    }

    pub fn sample_height(&self, pyramid: &Pyramid, level: u32, sample: [i64; 2]) -> f32 {
        let spacing = pyramid.spacing(level);
        let [dx, dz] = self.offset_samples(spacing);
        let sample = [sample[0] + dx, sample[1] + dz];
        let x = sample[0] as f32 * spacing;
        let z = sample[1] as f32 * spacing;
        let [cx, cz] = sample.map(|index| index.div_euclid(GRID_STEP));
        let horizontal = QUARTER_WEIGHTS[sample[0].rem_euclid(GRID_STEP) as usize];
        let vertical = QUARTER_WEIGHTS[sample[1].rem_euclid(GRID_STEP) as usize];
        let point = Vec2::new(x, z);
        let mut height = [0.0_f32];
        for layer in 0..self.layers().len() {
            let Some(tables) = self.noise(layer) else {
                self.apply_local(layer, &mut height, &[x], z, (point, point), spacing);
                continue;
            };
            let (octaves, fine, gullies) = tables.bands(spacing);
            let len = state_len(octaves);
            let mut points = vec![0.0_f32; 16 * len];
            for (index, point) in points.chunks_exact_mut(len).enumerate() {
                let column = (cx - 1 + (index / 4) as i64) * GRID_STEP;
                let row = (cz - 1 + (index % 4) as i64) * GRID_STEP;
                tables.coarse(
                    column as f32 * spacing,
                    row as f32 * spacing,
                    octaves,
                    gullies,
                    point,
                );
            }
            let mut columns = vec![0.0_f32; 4 * len];
            for (column, value) in columns.chunks_exact_mut(len).enumerate() {
                let taps = [0, 1, 2, 3].map(|row| {
                    let at = (column * 4 + row) * len;
                    &points[at..at + len]
                });
                interpolate(value, vertical, taps);
            }
            let mut sum = vec![0.0_f32; len];
            let taps = [0, 1, 2, 3].map(|column| &columns[column * len..(column + 1) * len]);
            interpolate(&mut sum, horizontal, taps);
            self.apply_noise(
                layer,
                &mut height,
                &[tables.finish(x, z, &sum, octaves, fine)],
            );
        }
        height[0]
    }

    pub fn height_at(&self, pyramid: &Pyramid, point: Vec2) -> f32 {
        let spacing = pyramid.spacing(0);
        let u = point.x / spacing;
        let v = point.y / spacing;
        let (i, j) = (u.floor(), v.floor());
        let (fu, fv) = (u - i, v - j);
        let (i, j) = (i as i64, j as i64);
        let h = |di: i64, dj: i64| self.sample_height(pyramid, 0, [i + di, j + dj]);
        if fu + fv <= 1.0 {
            let h00 = h(0, 0);
            h00 + fu * (h(1, 0) - h00) + fv * (h(0, 1) - h00)
        } else {
            let h11 = h(1, 1);
            h11 + (1.0 - fu) * (h(0, 1) - h11) + (1.0 - fv) * (h(1, 0) - h11)
        }
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
        out.put(&self.placed).put(&self.placing).put(&self.water);
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
        let placed: Option<Arc<[crate::Placed]>> = input.get()?;
        let placing = input.get()?;
        let water: Option<TileWater> = input.get()?;
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
        tile.hold(water);
        tile.placed = placed;
        tile.placing = placing;
        Ok(tile)
    }
}
