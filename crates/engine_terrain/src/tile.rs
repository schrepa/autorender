//! Tiles: their samples, bounds and holes, sampling within them, and their
//! generation from a layer stack, a step at a time.

use std::ops::Range;
use std::sync::Arc;

use glam::{Vec2, Vec3, Vec4};

use engine_math::{clamp, max, min};

use crate::hydrology::RIVER_FAR;
use crate::layers::{Layer, Layers, state_len};
use crate::rivers::RiverRaster;
use crate::surface::{BiomeRules, Control, RowInputs, RowScratch, SampleShape, smooth_gradient};
use crate::tile_water::{TileWater, WaterPoint, WaterState};
use crate::{
    HydrologySample, PATCH_QUADS, PATCHES_PER_TILE, PathSample, Placing, Pyramid, TILE_QUADS,
    TILE_SAMPLES, TILE_STRIDE, TileKey, TileShading,
};

mod beds;

pub(crate) use beds::RiverInputs;

/// Cells with a hole bit: those starting at samples 0..=256, so tiles agree
/// on the cells along their shared edges.
const HOLE_CELLS: usize = TILE_SAMPLES as usize;
const HOLE_WORDS: usize = (HOLE_CELLS * HOLE_CELLS).div_ceil(64);

/// A tile holds each sample's meters to the nearest river's edge, and the
/// height of the bankfull flood of the river holding it above its ground,
/// as steps of `RIVER_EDGE_STEP` meters counted from `RIVER_EDGE_ZERO`, the
/// edge and the ground; `RIVER_EDGE_FAR` stands for no river within reach,
/// and for no river's bed. Its surface masks, placement and the GPU's
/// control values all read these steps.
pub const RIVER_EDGE_STEP: f32 = 0.0625;
pub const RIVER_EDGE_ZERO: u16 = 32_768;
pub const RIVER_EDGE_FAR: u16 = u16::MAX;

/// The step of a tile's river values ([`RIVER_EDGE_STEP`]) nearest
/// `meters`, within those short of [`RIVER_EDGE_FAR`]; that one at
/// [`RIVER_FAR`] and beyond, where no river reaches.
pub(crate) fn river_step(meters: f32) -> u16 {
    if meters >= RIVER_FAR {
        return RIVER_EDGE_FAR;
    }
    let step = (meters / RIVER_EDGE_STEP).round() + f32::from(RIVER_EDGE_ZERO);
    clamp(step, 0.0, f32::from(RIVER_EDGE_FAR - 1)) as u16
}

/// [`river_step`] undone: meters from the nearest river's edge, negative
/// in its channel; [`RIVER_FAR`] where no river reaches.
pub(crate) fn river_meters(step: u16) -> f32 {
    if step == RIVER_EDGE_FAR {
        RIVER_FAR
    } else {
        (f32::from(step) - f32::from(RIVER_EDGE_ZERO)) * RIVER_EDGE_STEP
    }
}

/// A tile's samples: heights in meters above the terrain's origin of
/// samples -1 to 257 along x and z (its samples and a border of one), row
/// by row along z, with their bounds, where a stamp cut them holes, and
/// with a surface each sample's control value; each sample's meters to the
/// nearest river's edge where rivers reach it, and the bed of the river
/// holding it where it lies in one; what culling knows of each of its
/// patches, made with it; and for a terrain's tiles of level 1 and coarser,
/// once it arrived, the shading they take from a record of their own, level
/// 0's averaged.
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
    // The water standing on it, the world's one water there.
    water: Option<TileWater>,
    shading: Option<Arc<TileShading>>,
    bounds: [f32; 2],
    patch_bounds: Box<[[f32; 2]]>,
    patch_holes: [u64; 4],
    patch_culling: Box<[PatchCulling]>,
}

/// What culling knows of a patch, made with its tile: the disk holding the
/// slopes (rise per meter along x and z) of all its triangles, which says
/// where a view sees none of their fronts; and the lowest ground of each of
/// its quarters (rows along z), which hides what lies below sight over it.
/// Slopes are in 256ths, the disk's radius rounded up over its center's
/// rounding; floors in 255ths of the patch's heights up from its lowest,
/// rounded down: neither tells more than the ground holds.
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

/// Slopes' steps: 256 a unit of rise per meter.
const SLOPE_STEPS: f32 = 256.0;

/// The surface at a point: its height on its triangle, the normal of the
/// smooth surface there ([`SampleShape`]) and whether its cell is a hole.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileSample {
    pub height: f32,
    pub normal: Vec3,
    pub hole: bool,
}

impl Tile {
    /// A tile of `key` at sample spacing `spacing` from its
    /// [`TILE_STRIDE`]-squared `heights` and optional hole bits (one per
    /// cell starting at samples 0..=256, row by row), with what culling
    /// knows of its patches. `None` when the sizes do not match or a height
    /// is not finite.
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

    /// [`Self::new`] but for what culling knows of its patches, which a
    /// tile read back holds already.
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

    /// What culling knows of patch `(x, z)`: the disk around its triangles'
    /// slopes, each cell split from its (x, z+1) corner to its (x+1, z) one,
    /// and its quarters' lowest ground.
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
            // A step lower still, past what rounding could add back.
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

    /// The tile with `control`: one packed [`Control`] per sample, the
    /// layout of its heights. `None` when the size does not match.
    pub fn with_control(mut self, control: Box<[u32]>) -> Option<Self> {
        let stride = TILE_STRIDE as usize;
        (control.len() == stride * stride).then(|| {
            self.control = Some(control);
            self
        })
    }

    /// Every sample's meters to the nearest river's edge as steps of
    /// [`RIVER_EDGE_STEP`] from [`RIVER_EDGE_ZERO`] ([`RIVER_EDGE_FAR`]
    /// beyond every river's reach), the border included, row by row: the
    /// layout of its heights. `None` where no river reaches the tile.
    pub fn river_edges(&self) -> Option<&[u16]> {
        self.river_edges.as_deref()
    }

    /// The tile with the water its samples' `surface` and words `info`
    /// ([`TileWater::info`]) stand at: none where no sample's surface
    /// stands over its ground or the sizes do not match.
    #[must_use]
    pub(crate) fn with_water(mut self, surface: Box<[f32]>, info: Box<[[u32; 3]]>) -> Self {
        self.water = TileWater::new(surface, info, &self.heights, self.spacing);
        self
    }

    /// The water standing on the tile, if any does.
    pub fn water(&self) -> Option<&TileWater> {
        self.water.as_ref()
    }

    /// The tile holding `water` in place of its own: water made over its
    /// samples as its record holds them, as a job of its settled water
    /// makes it ([`Tile::steadied`], [`Tile::flowing`]). Its ground,
    /// objects and shading stay as they are.
    #[must_use]
    pub fn holding(&self, water: Option<TileWater>) -> Self {
        Self {
            key: self.key,
            spacing: self.spacing,
            heights: self.heights.clone(),
            holes: self.holes.clone(),
            control: self.control.clone(),
            river_edges: self.river_edges.clone(),
            river_beds: self.river_beds.clone(),
            placed: self.placed.clone(),
            placing: self.placing,
            water,
            shading: self.shading.clone(),
            bounds: self.bounds,
            patch_bounds: self.patch_bounds.clone(),
            patch_holes: self.patch_holes,
            patch_culling: self.patch_culling.clone(),
        }
    }

    /// The shading the tile shades with, once its record arrived: level 0's
    /// averaged, which a terrain's tiles of level 1 and coarser take from
    /// it ([`TileShading::shades`]); `None` before, and for other tiles.
    pub fn shading(&self) -> Option<&Arc<TileShading>> {
        self.shading.as_ref()
    }

    /// The tile shading with `shading` in place of what it held, as its
    /// record arrives. Its ground, water and objects stay as they are.
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

    /// Meters between samples.
    pub fn spacing(&self) -> f32 {
        self.spacing
    }

    /// Every sample's packed control value, the border included, row by
    /// row: the layout GPU tiles upload. `None` without a surface, and for
    /// tiles the GPU or their shading gives theirs.
    pub fn control(&self) -> Option<&[u32]> {
        self.control.as_deref()
    }

    /// The objects the surface's placement layers of this tile's level
    /// stand on it.
    pub fn placed(&self) -> &[crate::Placed] {
        self.placed.as_deref().unwrap_or(&[])
    }

    /// Which of its level's objects the tile holds, all of them or
    /// those of one distance: `None` before any were placed.
    pub fn placing(&self) -> Option<Placing> {
        self.placing
    }

    /// The tile with `placed` standing on it: the objects `placing` places.
    #[must_use]
    pub fn with_placed(mut self, placed: Vec<crate::Placed>, placing: Placing) -> Self {
        self.placed = (!placed.is_empty()).then(|| Arc::from(placed));
        self.placing = Some(placing);
        self
    }

    /// The tile with none of its objects.
    #[must_use]
    pub fn without_placed(mut self) -> Self {
        self.placed = None;
        self.placing = None;
        self
    }

    /// The control value of sample `(x, z)`, -1 to 257 along each axis.
    pub fn control_at(&self, x: i32, z: i32) -> Option<Control> {
        let stride = TILE_STRIDE as i32;
        let x = x.clamp(-1, stride - 2);
        let z = z.clamp(-1, stride - 2);
        self.control
            .as_ref()
            .map(|control| Control::unpack(control[((z + 1) * stride + x + 1) as usize]))
    }

    /// Height of sample `(x, z)`, -1 to 257 along each axis.
    #[inline]
    pub fn height(&self, x: i32, z: i32) -> f32 {
        let stride = TILE_STRIDE as i32;
        self.heights[((z + 1) * stride + x + 1) as usize]
    }

    /// The largest difference, in meters, between this tile's heights and
    /// those of `parent`, the tile of the next coarser level holding it,
    /// between its samples: what this level's patches move by as they
    /// morph into the parent level's. Every third sample along each axis
    /// is compared, alternately on and between the parent's.
    pub fn morph_error(&self, parent: &Tile) -> f32 {
        let offset = [
            i32::from(self.key.x % 2) * 128,
            i32::from(self.key.z % 2) * 128,
        ];
        let mut largest = 0.0_f32;
        for z in (0..=TILE_QUADS as i32).step_by(3) {
            for x in (0..=TILE_QUADS as i32).step_by(3) {
                // The parent's samples around this one, half as far apart.
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

    /// Every sample, the border included, row by row: the layout GPU tiles
    /// upload.
    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    /// Lowest and highest height of samples 0 to 256.
    pub fn bounds(&self) -> [f32; 2] {
        self.bounds
    }

    /// Lowest and highest height of patch `(x, z)`'s samples.
    pub fn patch_bounds(&self, x: u32, z: u32) -> [f32; 2] {
        self.patch_bounds[(z * PATCHES_PER_TILE + x) as usize]
    }

    /// The disk holding patch `(x, z)`'s triangles' slopes (rise per meter
    /// along x and z): its center and its radius, never smaller than they
    /// need.
    pub fn patch_slopes(&self, x: u32, z: u32) -> [f32; 3] {
        let cull = self.patch_culling[(z * PATCHES_PER_TILE + x) as usize];
        [
            f32::from(cull.slope[0]) / SLOPE_STEPS,
            f32::from(cull.slope[1]) / SLOPE_STEPS,
            f32::from(cull.spread) / SLOPE_STEPS,
        ]
    }

    /// The lowest ground of each quarter of patch `(x, z)`, rows along z,
    /// never above it.
    pub fn patch_floors(&self, x: u32, z: u32) -> [f32; 4] {
        let index = (z * PATCHES_PER_TILE + x) as usize;
        let [low, high] = self.patch_bounds[index];
        self.patch_culling[index]
            .floors
            .map(|floor| min(low + (high - low) * (f32::from(floor) / 255.0), high))
    }

    /// Whether any cell of patch `(x, z)` is a hole.
    pub fn patch_has_holes(&self, x: u32, z: u32) -> bool {
        let index = (z * PATCHES_PER_TILE + x) as usize;
        self.patch_holes[index / 64] & (1 << (index % 64)) != 0
    }

    /// Hole bits of cells starting at samples 0..=256, row by row; `None`
    /// without holes.
    pub fn hole_bits(&self) -> Option<&[u64]> {
        self.holes.as_deref()
    }

    /// Whether the cell starting at sample `(x, z)` is a hole.
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

    /// The cell holding `local`, meters from the tile's first corner: its
    /// first sample's column and row and where in it `local` lies, `fu` and
    /// `fv`; `None` outside the tile.
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

    /// The surface at `local`, meters from the tile's first corner: its
    /// height on the triangles every consumer splits cells into, and the
    /// normal of the smooth surface there, the gradients at the cell's four
    /// samples blended across it as renderers shade the ground; `None`
    /// outside the tile.
    pub fn sample(&self, local: Vec2) -> Option<TileSample> {
        let (i, j, fu, fv) = self.cell(local)?;
        let corners = [
            self.height(i, j),
            self.height(i + 1, j),
            self.height(i, j + 1),
            self.height(i + 1, j + 1),
        ];
        // The 4 × 4 samples around the cell, each of its corners' gradients
        // from the 3 × 3 around that corner.
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

    /// The water at `local`, meters from the tile's first corner, on the
    /// triangles [`Self::sample`] reads the ground on: its surface, its
    /// depth over that ground, its flow, whitewater, kind and state (its
    /// reach the first corner's); `None` where no water reaches the
    /// triangle's corners or its surface stands at or below the ground.
    pub fn water_at(&self, local: Vec2) -> Option<WaterPoint> {
        let water = self.water.as_ref()?;
        let (i, j, fu, fv) = self.cell(local)?;
        let at = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)];
        let surfaces = at.map(|(x, z)| water.surface(x, z));
        // The corners of the triangle holding it.
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
                on_triangle(info.map(|(flow, _, _, _)| flow.x), fu, fv),
                on_triangle(info.map(|(flow, _, _, _)| flow.y), fu, fv),
            ),
            white: on_triangle(info.map(|(_, white, _, _)| white), fu, fv),
            lake: info[corners[0]].2,
            state: WaterState {
                turbulence: on_triangle(states.map(|state| state.turbulence), fu, fv),
                breaking: on_triangle(states.map(|state| state.breaking), fu, fv),
                shear: on_triangle(states.map(|state| state.shear), fu, fv),
                reach: states[corners[0]].reach,
            },
        })
    }

    /// What the bed holds under the water at `local`, meters from the
    /// tile's first corner, as the water's sample nearest it holds it: its
    /// reach's code and its nearest obstacle, offset from `local`
    /// (`TileWater::bed_words`); none outside the tile or where it holds
    /// no water.
    pub fn bed_words(&self, local: Vec2) -> [u32; 2] {
        let (Some(water), Some((i, j, fu, fv))) = (self.water.as_ref(), self.cell(local)) else {
            return [0; 2];
        };
        let (x, z) = (i + i32::from(fu >= 0.5), j + i32::from(fv >= 0.5));
        let sample = Vec2::new(x as f32, z as f32) * self.spacing;
        water.bed_words(x, z, local - sample)
    }
}

/// The value at `(fu, fv)` in a cell whose corners, from its first, hold
/// `[v00, v10, v01, v11]`, on the triangle holding it: the cell's diagonal
/// runs from its (x, z+1) corner to its (x+1, z) corner.
#[inline]
pub(crate) fn on_triangle([v00, v10, v01, v11]: [f32; 4], fu: f32, fv: f32) -> f32 {
    if fu + fv <= 1.0 {
        v00 + fu * (v10 - v00) + fv * (v01 - v00)
    } else {
        v11 + (1.0 - fu) * (v01 - v11) + (1.0 - fv) * (v10 - v11)
    }
}

/// Coarse grid points along each axis: every fourth sample from sample -8
/// to 264, enough for Catmull-Rom interpolation at samples -1 to 257.
const GRID: usize = 69;
const GRID_FIRST: i64 = -2;
const GRID_STEP: i64 = 4;
/// Work units of a coarse grid point and of a noise layer's sample,
/// against one per sample of other layers: about their time.
const COARSE_COST: u64 = 24;
const NOISE_SAMPLE_COST: u64 = 4;

/// Four coarse states `taps` combined with Catmull-Rom `weights` into
/// `out`, value by value, in one order wherever states are interpolated.
#[inline(always)]
fn interpolate(out: &mut [f32], weights: [f32; 4], taps: [&[f32]; 4]) {
    let [a, b, c, d] = taps;
    for ((((value, a), b), c), d) in out.iter_mut().zip(a).zip(b).zip(c).zip(d) {
        *value = weights[0] * a + weights[1] * b + weights[2] * c + weights[3] * d;
    }
}

/// Catmull-Rom weights at quarter steps, exact in `f32`.
const QUARTER_WEIGHTS: [[f32; 4]; 4] = [
    [0.0, 1.0, 0.0, 0.0],
    [-0.070_312_5, 0.867_187_5, 0.226_562_5, -0.023_437_5],
    [-0.0625, 0.5625, 0.5625, -0.0625],
    [-0.023_437_5, 0.226_562_5, 0.867_187_5, -0.070_312_5],
];

/// [`QUARTER_WEIGHTS`] by tap, a coarse cell's four samples in lanes.
const LANE_WEIGHTS: [Vec4; 4] = {
    let w = QUARTER_WEIGHTS;
    [
        Vec4::new(w[0][0], w[1][0], w[2][0], w[3][0]),
        Vec4::new(w[0][1], w[1][1], w[2][1], w[3][1]),
        Vec4::new(w[0][2], w[1][2], w[2][2], w[3][2]),
        Vec4::new(w[0][3], w[1][3], w[2][3], w[3][3]),
    ]
};

/// [`interpolate`] for the four samples of a coarse cell at once,
/// sample `k` in lane `k` with [`QUARTER_WEIGHTS`]`[k]`.
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

/// Generates one tile from a layer stack in steps, which workers run a
/// slice at a time between their messages. Every sample is a pure function
/// of its position
/// and level: noise octaves at least 32 samples long are evaluated on a
/// grid of every fourth sample and interpolated, shorter ones down to four
/// samples at every sample, so neighboring tiles agree on shared samples.
pub struct Generator {
    layers: Arc<Layers>,
    key: TileKey,
    spacing: f32,
    first: [i64; 2],
    bands: Vec<Option<(usize, Range<usize>, usize)>>,
    // Each noise layer's coarse states, grid point by grid point.
    grids: Vec<Option<Box<[f32]>>>,
    stage: Stage,
    heights: Box<[f32]>,
    holes: Option<Box<[u64]>>,
    xs: Box<[f32]>,
    row: Box<[f32]>,
    values: Box<[f32]>,
    // A row's coarse states interpolated along z, grid column by column,
    // and one sample's, and a coarse cell's four samples' in lanes.
    column: Box<[f32]>,
    state: Box<[f32]>,
    lanes: Box<[Vec4]>,
    min: Vec2,
    max: Vec2,
    // The surface whose control values the tile gets, and their samples.
    surface: Option<Arc<BiomeRules>>,
    control: Option<Box<[u32]>>,
    // The finished tile while its objects are placed, a row
    // of candidate squares at a time, and the objects placed so far.
    tile: Option<Tile>,
    placed: Vec<crate::Placed>,
    // Each hydrology layer's rivers rasterized over the tile's samples
    // once, by layer, until its water is made; the first baked one's meters
    // to rivers' edges and, over the tile's heights, its rivers' beds, which
    // the tile keeps; its stored rows' z; a row's inputs to the surface's
    // masks.
    rasters: Vec<Option<RiverRaster>>,
    river_edges: Option<Box<[u16]>>,
    river_beds: Option<Box<[u16]>>,
    zs: Box<[f32]>,
    inputs: SurfaceInputs,
    places: crate::placement::PlaceScratch,
    // Whether it gives its tile control values; the GPU, or its shading,
    // gives them where not.
    materials: bool,
}

/// A row's inputs to a surface's masks, and scratch space for evaluating
/// them and for one sample's material weights, reused row after row.
#[derive(Default)]
pub(crate) struct SurfaceInputs {
    zs: Vec<f32>,
    height: Vec<f32>,
    gradient: Vec<f32>,
    curvature: Vec<f32>,
    water: Vec<HydrologySample>,
    path: Vec<PathSample>,
    scratch: RowScratch,
    weights: Vec<(u8, f32)>,
}

impl SurfaceInputs {
    fn clear(&mut self, len: usize) {
        for row in [&mut self.height, &mut self.gradient, &mut self.curvature] {
            row.clear();
            row.reserve(len);
        }
        self.water.clear();
        self.path.clear();
    }
}

/// The control values `rules` over `layers` give a tile's stored row `row`
/// (0 to 258, samples -1 to 257) at its stored `columns`, into `out`, one a
/// column: from the smooth surface through the tile's stored `heights`
/// around each sample ([`SampleShape`]), border samples reading their
/// nearest neighbors inside, and from each sample's meters to the nearest
/// river's edge and its bed the tile holds (`rivers`, its edges' steps and
/// its beds' planes); `xs` holds each stored column's x and `z` is the
/// row's, in the layers' frame, its samples `spacing` meters apart. What the
/// rules give a tile's samples on the CPU, for generators to give their
/// tiles and for tiles' shading to average.
pub(crate) fn control_row(
    (rules, layers): (&BiomeRules, &Layers),
    (heights, rivers): (&[f32], beds::RiverRows<'_>),
    (xs, z, spacing): (&[f32], f32, f32),
    (row, columns): (usize, Range<usize>),
    (out, inputs): (&mut [u32], &mut SurfaceInputs),
) {
    let stride = TILE_STRIDE as usize;
    // Inputs the masks never read are left unread.
    let reads = rules.reads();
    let (up, down) = (row.saturating_sub(1), (row + 1).min(stride - 1));
    // The row's inputs, then its control values all at once.
    inputs.clear(columns.len());
    for column in columns.clone() {
        let held = RiverInputs::held(rivers, row * stride + column);
        let (left, right) = (column.saturating_sub(1), (column + 1).min(stride - 1));
        let at = |r: usize, c: usize| heights[r * stride + c];
        let height = at(row, column);
        let shape = SampleShape::of(
            [up, row, down].map(|r| [left, column, right].map(|c| at(r, c))),
            spacing,
        );
        let x = xs[column];
        // Surroundings lose the masks of the bakes and
        // roads they fade out.
        let shown = layers.shown(x, z);
        let water = match layers.hydrology() {
            Some(water) if shown > 0.0 && reads.water => held.with_maps(water, [x, z], spacing),
            _ => HydrologySample::default(),
        };
        inputs.height.push(height);
        inputs.gradient.push(shape.slope());
        inputs.curvature.push(shape.curvature);
        inputs.water.push(HydrologySample {
            flow: water.flow * shown,
            wetness: water.wetness * shown,
            lake: water.lake * shown,
            ..water
        });
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
        path: &inputs.path,
    };
    rules.control_row(&row_inputs, out, (&mut inputs.scratch, &mut inputs.weights));
}

impl Generator {
    pub fn new(layers: Arc<Layers>, pyramid: &Pyramid, key: TileKey) -> Self {
        let level = u32::from(key.level);
        let spacing = pyramid.spacing(level);
        // Surroundings read their stack where their samples
        // lie in the ground it was drawn for.
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

    /// The generator leaving its tile without control values (where
    /// `materials` is false), which the GPU, or the tile's shading, gives to
    /// rendering and the rules to gameplay at the samples it asks for; it
    /// still places objects.
    #[must_use]
    pub fn with_materials(mut self, materials: bool) -> Self {
        self.materials = materials;
        self
    }

    /// The sample rows it samples heights of: every row, -1 to 257.
    const HEIGHT_ROWS: (i32, i32) = (-1, TILE_SAMPLES as i32 + 1);

    /// The coarse grid rows its height rows read: each sample row reads
    /// four from the one after its grid point's.
    fn grid_rows() -> (usize, usize) {
        let (first, end) = Self::HEIGHT_ROWS;
        let low = i64::from(first).div_euclid(GRID_STEP) - GRID_FIRST - 1;
        let high = i64::from(end - 1).div_euclid(GRID_STEP) - GRID_FIRST + 3;
        (low as usize, std::cmp::min(high as usize, GRID))
    }

    /// The generator giving its tile `surface`'s control values.
    #[must_use]
    pub fn with_surface(mut self, surface: Arc<BiomeRules>) -> Self {
        self.surface = Some(surface);
        self
    }

    pub fn key(&self) -> TileKey {
        self.key
    }

    /// Whether generation stopped without a tile: its heights or control
    /// values were invalid.
    pub fn failed(&self) -> bool {
        matches!(self.stage, Stage::Failed)
    }

    /// The stage after the samples and holes: the surface's control values
    /// unless the GPU gives them, else done.
    fn after_holes(&self) -> Stage {
        match &self.surface {
            Some(_) if self.materials => Stage::Biomes { row: 0 },
            _ => Stage::Done,
        }
    }

    /// Control values of stored row `row` (0 to 258, samples -1 to 257), as
    /// the rules give a tile's samples ([`control_row`]), each sample's
    /// meters to the nearest river's edge and its bed those the tile keeps.
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

    /// Generates until the tile is done or `budget` work units run out
    /// (about one per sample), returning the tile once done. A step that
    /// starts always finishes, so a budget can go slightly negative in
    /// effect; callers pass what is left.
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
                    let grid = self.grids[layer].as_mut().expect("noise layers have grids");
                    let len = state_len(coarse);
                    let (first, spacing) = (self.first[0], self.spacing);
                    let x = |column: usize| {
                        (first + (GRID_FIRST + column as i64) * GRID_STEP) as f32 * spacing
                    };
                    // Four columns at once, the rest one by
                    // one, to the same bits.
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
                    let bits = self.holes.as_mut().expect("holes stage has bits");
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
                        Some((surface, info)) => tile.with_water(surface.into(), info.into()),
                        None => tile,
                    };
                    self.stage = Stage::Done;
                    // Objects stand on the tiles of their layers' levels,
                    // placed in the stage after.
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
                            // Then what authors stand on it,
                            // with its own level's objects.
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

    /// Rasterizes layer `layer`'s rivers, if it holds any, over the tile's
    /// samples within `budget` (samples' worth), for the tile's rows, water
    /// and masks to read. Returns the stage after: this layer's again until
    /// its rivers are stamped, then the next layer's, and once every
    /// layer's are, the samples', the rasters no river reaches let go and
    /// the first baked hydrology layer's meters to rivers' edges kept for
    /// the tile.
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

    /// The water of every stored row (0 to 258) as the stack's hydrology
    /// bake holds its lakes and rivers over them
    /// ([`crate::tile_water::water_rows`]), its rivers' from their raster,
    /// which it then lets go: their samples' surfaces and words, row by
    /// row; `None` without the bake.
    fn water_rows(&mut self, budget: &mut u64) -> Option<(Vec<f32>, Vec<[u32; 3]>)> {
        let stride = TILE_STRIDE as usize;
        let rasters = std::mem::take(&mut self.rasters);
        let (index, hydrology) = self.layers.hydrology_layer()?;
        let raster = rasters.get(index).and_then(Option::as_ref);
        let count = stride * stride;
        let (mut surface, mut info) = (vec![f32::NAN; count], vec![[0_u32; 3]; count]);
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

    /// Heights of sample row `row` (-1 to 257), every layer applied in
    /// order.
    fn sample_row(&mut self, row: i32) {
        let z = (self.first[1] + i64::from(row)) as f32 * self.spacing;
        self.row.fill(0.0);
        let coarse_row = i64::from(row).div_euclid(GRID_STEP);
        let vertical = QUARTER_WEIGHTS[i64::from(row).rem_euclid(GRID_STEP) as usize];
        for layer in 0..self.layers.layers().len() {
            let Some(tables) = self.layers.noise(layer) else {
                // Hydrology layers cut the rivers reaching the tile from its
                // raster.
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
            let grid = self.grids[layer].as_ref().expect("noise layers have grids");
            // The coarse state along z first, for every grid column.
            let base = (coarse_row - 1 - GRID_FIRST) as usize;
            let tap = |row: usize, column: usize| {
                let at = ((base + row) * GRID + column) * len;
                &grid[at..at + len]
            };
            for (column, sum) in self.column.chunks_exact_mut(len).take(GRID).enumerate() {
                let taps = [0, 1, 2, 3].map(|row| tap(row, column));
                interpolate(sum, vertical, taps);
            }
            // Then along x for every sample, which sums every octave: the
            // four samples of a coarse cell at once, the
            // rest one by one, to the same bits.
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
    /// Work units of one [`Self::sample_height`], against one per sample of
    /// a layer: sixteen coarse grid points and a sample per noise layer.
    pub(crate) fn height_cost(&self) -> u64 {
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

    /// The height of global sample `sample` (x and z, counted from the
    /// terrain's first corner) at `level`, exactly as a generated tile holds
    /// it: the same coarse grid points, interpolation and layer order.
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
            // The coarse state along z first for each column, then along x,
            // as tiles interpolate it.
            let len = state_len(octaves);
            // The sixteen grid points' states, column by column.
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

    /// The height at terrain-local `point` on level 0's triangles, as
    /// generated tiles, renderers and collision hold it: four samples of
    /// [`Self::sample_height`].
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

/// A tile's key, spacing, samples, holes, control values, meters to
/// rivers' edges and their beds, placed objects and which of them, and its
/// water as its own record holds it ([`TileWater`]'s); reading rebuilds its
/// bounds as [`Tile::new`] does. Its shading is a record of its own
/// ([`TileShading`]), which a tile read back holds none of.
/// Tiles travel with their heights and water surfaces at their nearest
/// steps of the height grid ([`crate::codec::HEIGHT_STEP`]): a tile read
/// back is the record every host shows.
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

    fn read(input: &mut crate::codec::Reader<'_>) -> Result<Self, crate::codec::CodecError> {
        let invalid = crate::codec::CodecError("invalid tile");
        let stride = TILE_STRIDE as usize;
        let key = input.get()?;
        let spacing = input.get()?;
        let heights = crate::codec::get_grid(input, stride)?;
        let holes = input.get()?;
        let culling: Box<[PatchCulling]> = input.get()?;
        let control: Option<Box<[u32]>> = input.get()?;
        // Meters to rivers' edges, then their beds.
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
        let mut tile = Self::shaped(key, spacing, heights, holes).ok_or(invalid)?;
        if culling.len() != tile.patch_bounds.len() {
            return Err(invalid);
        }
        tile.patch_culling = culling;
        if let Some(control) = control {
            tile = tile.with_control(control).ok_or(invalid)?;
        }
        tile = tile.with_rivers(river_edges, river_beds).ok_or(invalid)?;
        tile.water = water;
        tile.placed = placed;
        tile.placing = placing;
        Ok(tile)
    }
}
