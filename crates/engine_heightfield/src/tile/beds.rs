use std::sync::Arc;

use glam::Vec2;

use engine_math::clamp;

use super::{
    FLOOD, FORM, GRAIN, Generator, POWER, RATIO_STEP, RIVER_EDGE_FAR, RIVER_EDGE_STEP,
    RIVER_EDGE_ZERO, SAMPLES, Stage, Tile, bed_at, from_half, on_triangle, river_meters, river_step,
};
use crate::hydrology::RIVER_FAR;
use crate::{Bed, HydrologyBake as Hydrology, HydrologySample, Rivers, TILE_STRIDE};

const PLANES: usize = 4;

const RATIO_STEPS: f32 = 4095.0;

fn half(value: f32) -> u16 {
    if value.is_nan() || value <= 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn flood_meters(step: u16) -> f32 {
    if step == RIVER_EDGE_FAR {
        RIVER_FAR
    } else {
        (f32::from(RIVER_EDGE_ZERO) - f32::from(step)) * RIVER_EDGE_STEP
    }
}

fn form(bed: &Bed) -> u16 {
    let ratio = if bed.d50 > 0.0 {
        bed.d84 / bed.d50
    } else {
        1.0
    };
    let steps = clamp(((ratio - 1.0) / RATIO_STEP).round(), 0.0, RATIO_STEPS) as u16;
    bed.reach as u16 | u16::from(bed.carved) << 3 | steps << 4
}

fn no_beds() -> Box<[u16]> {
    let mut planes = vec![0; PLANES * SAMPLES];
    planes[FLOOD * SAMPLES..(FLOOD + 1) * SAMPLES].fill(RIVER_EDGE_FAR);
    planes.into_boxed_slice()
}

pub type RiverRows<'a> = (Option<&'a [u16]>, Option<&'a [u16]>);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiverInputs {
    pub river: f32,
    pub grain: f32,
    pub power: f32,
    pub flood: f32,
}

impl RiverInputs {
    pub const FAR: Self = Self {
        river: RIVER_FAR,
        grain: 0.0,
        power: 0.0,
        flood: RIVER_FAR,
    };

    pub fn held((edges, beds): RiverRows<'_>, at: usize) -> Self {
        let plane = |index: usize| beds.map(|beds| beds[index * SAMPLES + at]);
        Self {
            river: edges.map_or(RIVER_FAR, |edges| river_meters(edges[at])),
            grain: plane(GRAIN).map_or(0.0, from_half),
            power: plane(POWER).map_or(0.0, from_half),
            flood: plane(FLOOD).map_or(RIVER_FAR, flood_meters),
        }
    }

    pub fn at(rivers: &Rivers, [x, z]: [f32; 2], (ground, spacing): (f32, f32)) -> Self {
        let bed = rivers.bed_at(x, z);
        Self {
            river: rivers.edge(x, z, spacing),
            grain: bed.map_or(0.0, |bed| bed.d50),
            power: bed.map_or(0.0, |bed| bed.power),
            flood: bed.map_or(RIVER_FAR, |bed| ground - bed.flood),
        }
    }

    pub fn with_maps(
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
        self.river_edges = edges.map(Arc::from);
        self.river_beds = beds.map(Arc::from);
        Some(self)
    }

    pub fn river_beds(&self) -> Option<&[u16]> {
        self.river_beds.as_deref()
    }

    pub fn river_texels(&self) -> Option<Arc<[[u16; 4]]>> {
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

    pub fn bed(&self, local: Vec2) -> Option<Bed> {
        bed_at(self.river_beds.as_deref()?, &self.heights, self.spacing, local)
    }

    pub(crate) fn river_inputs(&self, x: i32, z: i32) -> RiverInputs {
        let stride = TILE_STRIDE as i32;
        let (x, z) = (x.clamp(-1, stride - 2), z.clamp(-1, stride - 2));
        let rivers = (self.river_edges.as_deref(), self.river_beds.as_deref());
        RiverInputs::held(rivers, ((z + 1) * stride + x + 1) as usize)
    }

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
