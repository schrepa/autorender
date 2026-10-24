//! The beds of the rivers over a tile's samples, as the tile holds them
//! beside its meters to their edges, and what surface masks read of both:
//! the same codes on the CPU and in the GPU's texels, so both read the same
//! values.

use glam::Vec2;

use engine_math::clamp;

use super::{
    Generator, RIVER_EDGE_FAR, RIVER_EDGE_STEP, RIVER_EDGE_ZERO, Stage, Tile, on_triangle,
    river_meters, river_step,
};
use crate::hydrology::RIVER_FAR;
use crate::{Bed, Hydrology, HydrologySample, Reach, Rivers, TILE_STRIDE};

/// Samples of each of a tile's planes: its samples and their border.
const SAMPLES: usize = (TILE_STRIDE * TILE_STRIDE) as usize;

/// A tile holds the bed of the river holding each sample in four planes of
/// the layout of its heights, one after another ([`Tile::river_beds`]): the
/// median size of its grains (d50, meters) and the river's stream power at
/// its bankfull flood (watts per square meter of bed), each the high half of
/// the `f32` nearest it ([`half`]), 0 outside every river's bed; the height
/// of that flood's level above the sample's ground in steps of
/// [`RIVER_EDGE_STEP`] from [`RIVER_EDGE_ZERO`], [`RIVER_EDGE_FAR`] outside
/// every river's bed; and the bed's form ([`form`]).
const GRAIN: usize = 0;
const POWER: usize = 1;
const FLOOD: usize = 2;
const FORM: usize = 3;
const PLANES: usize = 4;

/// The steps a bed's form holds its d84 in, as a multiple of its d50 above
/// one, and the most its twelve bits hold: up to nine times its d50.
const RATIO_STEP: f32 = 1.0 / 512.0;
const RATIO_STEPS: f32 = 4095.0;

/// `value` as the high half of the `f32` nearest it that one holds, ties to
/// even: what a tile holds of a bed's grain size and power, which the CPU
/// and the GPU read back alike, as the high half of an `f32` ([`from_half`]);
/// 0 for 0, less or NaN.
fn half(value: f32) -> u16 {
    if value.is_nan() || value <= 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// The value [`half`] held as `code`.
fn from_half(code: u16) -> f32 {
    f32::from_bits(u32::from(code) << 16)
}

/// Meters a sample's ground stands above the flood level a tile holds
/// `step` steps above that ground ([`river_step`]), as the GPU reads them:
/// [`RIVER_FAR`] for [`RIVER_EDGE_FAR`], no bed.
fn flood_meters(step: u16) -> f32 {
    if step == RIVER_EDGE_FAR {
        RIVER_FAR
    } else {
        (f32::from(RIVER_EDGE_ZERO) - f32::from(step)) * RIVER_EDGE_STEP
    }
}

/// `bed`'s form as a tile holds it: its reach in bits 0 to 2, whether steps
/// or knobs are carved into it in bit 3, and from bit 4 its d84 as a
/// multiple of its d50 above one, in steps of [`RATIO_STEP`].
fn form(bed: &Bed) -> u16 {
    let ratio = if bed.d50 > 0.0 {
        bed.d84 / bed.d50
    } else {
        1.0
    };
    let steps = clamp(((ratio - 1.0) / RATIO_STEP).round(), 0.0, RATIO_STEPS) as u16;
    bed.reach as u16 | u16::from(bed.carved) << 3 | steps << 4
}

/// The reach a bed's form holds ([`form`]).
fn reach(form: u16) -> Reach {
    match form & 7 {
        0 => Reach::Pool,
        1 => Reach::Plane,
        2 => Reach::Steps,
        3 => Reach::Cascade,
        _ => Reach::Falls,
    }
}

/// Planes of a tile's samples' beds, none in any.
fn no_beds() -> Box<[u16]> {
    let mut planes = vec![0; PLANES * SAMPLES];
    planes[FLOOD * SAMPLES..(FLOOD + 1) * SAMPLES].fill(RIVER_EDGE_FAR);
    planes.into_boxed_slice()
}

/// A tile's rivers as surface masks read them, row by row with the border:
/// its edges (steps, [`river_step`]) and the planes of its beds, each
/// missing where the tile has none.
pub(crate) type RiverRows<'a> = (Option<&'a [u16]>, Option<&'a [u16]>);

/// What surface masks read of the rivers at a point of a tile's ground:
/// meters from the nearest river's edge, and the grain, power and flood of
/// the bed of the river holding it, as [`HydrologySample`] holds them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RiverInputs {
    pub(crate) river: f32,
    pub(crate) grain: f32,
    pub(crate) power: f32,
    pub(crate) flood: f32,
}

impl RiverInputs {
    /// No river within reach.
    pub(crate) const FAR: Self = Self {
        river: RIVER_FAR,
        grain: 0.0,
        power: 0.0,
        flood: RIVER_FAR,
    };

    /// Sample `at` of a tile's `edges` (steps, [`river_step`]) and planes of
    /// `beds`, row by row with the border, as the GPU reads its texel.
    pub(crate) fn held((edges, beds): RiverRows<'_>, at: usize) -> Self {
        let plane = |index: usize| beds.map(|beds| beds[index * SAMPLES + at]);
        Self {
            river: edges.map_or(RIVER_FAR, |edges| river_meters(edges[at])),
            grain: plane(GRAIN).map_or(0.0, from_half),
            power: plane(POWER).map_or(0.0, from_half),
            flood: plane(FLOOD).map_or(RIVER_FAR, flood_meters),
        }
    }

    /// What `rivers` give at terrain-local `(x, z)` over ground `ground`: the
    /// meters to the nearest river's edge from the square `spacing` meters a
    /// side about it, as a tile's sample there measures them, and the bed of
    /// the river holding the point.
    pub(crate) fn at(rivers: &Rivers, [x, z]: [f32; 2], (ground, spacing): (f32, f32)) -> Self {
        let bed = rivers.bed_at(x, z);
        Self {
            river: rivers.edge(x, z, spacing),
            grain: bed.map_or(0.0, |bed| bed.d50),
            power: bed.map_or(0.0, |bed| bed.power),
            flood: bed.map_or(RIVER_FAR, |bed| ground - bed.flood),
        }
    }

    /// What `hydrology`'s maps give at terrain-local `(x, z)` as samples
    /// `spacing` meters apart see them, with these rivers' inputs.
    pub(crate) fn with_maps(
        self,
        hydrology: &Hydrology,
        [x, z]: [f32; 2],
        spacing: f32,
    ) -> HydrologySample {
        HydrologySample {
            grain: self.grain,
            power: self.power,
            flood: self.flood,
            ..hydrology.masks_at(x, z, spacing, self.river)
        }
    }
}

impl Tile {
    /// The tile with each sample's meters to the nearest river's edge in
    /// steps ([`river_step`]) and its samples' beds' planes
    /// ([`Tile::river_beds`]), the layout of its heights: `None` when a size
    /// does not match.
    pub(crate) fn with_rivers(
        mut self,
        edges: Option<Box<[u16]>>,
        beds: Option<Box<[u16]>>,
    ) -> Option<Self> {
        if edges.as_ref().is_some_and(|edges| edges.len() != SAMPLES)
            || beds
                .as_ref()
                .is_some_and(|beds| beds.len() != PLANES * SAMPLES)
        {
            return None;
        }
        self.river_edges = edges;
        self.river_beds = beds;
        Some(self)
    }

    /// The bed of the river holding each sample, the border included, in
    /// four planes of the layout of its heights one after another: the
    /// median size of its grains and the river's stream power at its
    /// bankfull flood, each the high half of its `f32` (0 outside every
    /// river's bed); the height of that flood's level above the sample's
    /// ground in steps of [`RIVER_EDGE_STEP`] from [`RIVER_EDGE_ZERO`]
    /// ([`RIVER_EDGE_FAR`] outside every river's bed); and its reach,
    /// whether steps or knobs are carved into it, and its d84 as a multiple
    /// of its d50. `None` where no sample lies in a river's bed.
    pub fn river_beds(&self) -> Option<&[u16]> {
        self.river_beds.as_deref()
    }

    /// Every sample's meters to the nearest river's edge, then the grain,
    /// power and flood of the bed of the river holding it as
    /// [`Self::river_beds`] holds them ([`RIVER_EDGE_FAR`], 0, 0 and
    /// [`RIVER_EDGE_FAR`] where none reaches), the border included, row by
    /// row: the texels GPU tiles upload. `None` where no river reaches the
    /// tile.
    pub fn river_texels(&self) -> Option<Box<[[u16; 4]]>> {
        let (edges, beds) = (self.river_edges.as_deref(), self.river_beds.as_deref());
        if edges.is_none() && beds.is_none() {
            return None;
        }
        let plane = |index: usize, at: usize, none: u16| {
            beds.map_or(none, |beds| beds[index * SAMPLES + at])
        };
        Some(
            (0..SAMPLES)
                .map(|at| {
                    [
                        edges.map_or(RIVER_EDGE_FAR, |edges| edges[at]),
                        plane(GRAIN, at, 0),
                        plane(POWER, at, 0),
                        plane(FLOOD, at, RIVER_EDGE_FAR),
                    ]
                })
                .collect(),
        )
    }

    /// The bed of the river holding the sample nearest `local` (meters from
    /// the tile's first sample, as [`Self::sample`] takes it), its flood the
    /// level of the sample's ground plus the height the tile holds it at
    /// above that ground: `None` outside every river's bed, and outside the
    /// tile.
    pub fn bed(&self, local: Vec2) -> Option<Bed> {
        let beds = self.river_beds.as_deref()?;
        let (i, j, fu, fv) = self.cell(local)?;
        let (x, z) = (i + i32::from(fu >= 0.5), j + i32::from(fv >= 0.5));
        let at = (z as usize + 1) * TILE_STRIDE as usize + x as usize + 1;
        let step = beds[FLOOD * SAMPLES + at];
        if step == RIVER_EDGE_FAR {
            return None;
        }
        let form = beds[FORM * SAMPLES + at];
        let d50 = from_half(beds[GRAIN * SAMPLES + at]);
        Some(Bed {
            reach: reach(form),
            d50,
            d84: d50 * (1.0 + f32::from(form >> 4) * RATIO_STEP),
            power: from_half(beds[POWER * SAMPLES + at]),
            flood: self.height(x, z) + river_meters(step),
            carved: form & 8 != 0,
        })
    }

    /// What masks read of the rivers at sample `(x, z)`, -1 to 257 along
    /// each axis, as the GPU reads its texel.
    pub(crate) fn river_inputs(&self, x: i32, z: i32) -> RiverInputs {
        let stride = TILE_STRIDE as i32;
        let (x, z) = (x.clamp(-1, stride - 2), z.clamp(-1, stride - 2));
        let rivers = (self.river_edges.as_deref(), self.river_beds.as_deref());
        RiverInputs::held(rivers, ((z + 1) * stride + x + 1) as usize)
    }

    /// What masks read of the rivers at `local`, meters from the tile's
    /// first corner, on the triangles [`Self::sample`] reads the ground on:
    /// the meters to the nearest river's edge and the flood, [`RIVER_FAR`]
    /// where a corner of the triangle holding it lies beyond every river's
    /// reach or bed, or outside the tile; the grain and power, 0 at such
    /// corners.
    pub(crate) fn river_at(&self, local: Vec2) -> RiverInputs {
        let Some((i, j, fu, fv)) = self.cell(local) else {
            return RiverInputs::FAR;
        };
        let held =
            [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)].map(|(x, z)| self.river_inputs(x, z));
        let corners = if fu + fv <= 1.0 { [0, 1, 2] } else { [3, 2, 1] };
        let blend = |value: fn(RiverInputs) -> f32| on_triangle(held.map(value), fu, fv);
        let reached = |value: fn(RiverInputs) -> f32| {
            if corners
                .iter()
                .all(|&corner| value(held[corner]) < RIVER_FAR)
            {
                blend(value)
            } else {
                RIVER_FAR
            }
        };
        RiverInputs {
            river: reached(|inputs| inputs.river),
            grain: blend(|inputs| inputs.grain),
            power: blend(|inputs| inputs.power),
            flood: reached(|inputs| inputs.flood),
        }
    }
}

impl Generator {
    /// Each stored sample's bed (0 to 258 along each axis) as the first
    /// baked hydrology layer's raster reads it, over the heights it now
    /// holds ([`Tile::river_beds`]): none where no sample lies in a river's
    /// bed. Returns the stage after: the holes', else what follows them.
    pub(super) fn beds(&mut self, budget: &mut u64) -> Stage {
        if let Some((index, hydrology)) = self.layers.hydrology_layer()
            && let Some(raster) = self.rasters.get(index).and_then(Option::as_ref)
        {
            let stride = TILE_STRIDE as usize;
            let mut beds: Option<Box<[u16]>> = None;
            for at in 0..SAMPLES {
                let Some(bed) = raster.bed(&hydrology.rivers, at / stride, at % stride) else {
                    continue;
                };
                let planes = beds.get_or_insert_with(no_beds);
                planes[GRAIN * SAMPLES + at] = half(bed.d50);
                planes[POWER * SAMPLES + at] = half(bed.power);
                planes[FLOOD * SAMPLES + at] = river_step(bed.flood - self.heights[at]);
                planes[FORM * SAMPLES + at] = form(&bed);
            }
            self.river_beds = beds;
            *budget = budget.saturating_sub(SAMPLES as u64);
        }
        if self.holes.is_some() {
            Stage::Holes { row: 0 }
        } else {
            self.after_holes()
        }
    }
}
