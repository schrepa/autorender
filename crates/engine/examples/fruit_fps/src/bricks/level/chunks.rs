use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use engine::prelude::*;
use engine_core::{
    AssetLibrary, DrawKind, Drawn, DrawnTable, GeometryHandle, Layers, MaterialHandle, PassBit,
    PassMask, Projection, ViewFacts,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::sets::SetView;
use crate::play::light::DetailLimit;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LevelPiece {
    pub largest_m: f32,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub(super) struct PieceSizes {
    sizes: BTreeMap<GeometryHandle, f32>,
}

pub(super) fn pieces(found: Query<(&Drawn, &LevelPiece)>, mut sizes: ResMut<PieceSizes>) {
    let next: BTreeMap<GeometryHandle, f32> = found
        .iter()
        .filter_map(|(drawn, piece)| drawn.geometry.map(|mesh| (mesh, piece.largest_m)))
        .collect();
    if sizes.sizes != next {
        sizes.sizes = next;
    }
}

#[derive(Clone, Debug)]
pub(super) struct Chunk {
    pub mesh: Option<GeometryHandle>,
    pub material: Option<MaterialHandle>,
    pub instances: Range<u32>,
    pub passes: Option<PassMask>,
    pub layers: Layers,
    pub bounds: Option<[Vec3; 2]>,
    pub vertices: u64,
    pub largest_m: Option<f32>,
}

impl Chunk {
    pub fn drawn_in(&self, pass: Option<PassBit>) -> bool {
        self.passes
            .is_none_or(|mask| pass.is_some_and(|bit| mask.has(bit)))
    }

    fn nearest(&self, eye: Vec3) -> f32 {
        self.bounds.map_or(0.0, |[low, high]| {
            eye.max(low).min(high).distance_squared(eye)
        })
    }
}

pub(super) fn listed(
    (table, library, pieces): (&DrawnTable, Option<&AssetLibrary>, &PieceSizes),
    kind: DrawKind,
    into: &mut Vec<Chunk>,
) {
    into.clear();
    into.extend(table.batches(kind).map(|batch| {
        let mesh = batch
            .geometry
            .zip(library)
            .and_then(|(geometry, library)| library.mesh(geometry));
        let count = batch.instances.len() as u64;
        Chunk {
            mesh: batch.geometry,
            material: batch.material,
            instances: batch.instances.clone(),
            passes: batch.passes,
            layers: batch.layers,
            bounds: mesh.filter(|_| count == 1).map(|mesh| {
                let bounds = mesh.bounds();
                [bounds.min(), bounds.max()]
            }),
            vertices: mesh.map_or(0, |mesh| mesh.indices().len() as u64) * count,
            largest_m: batch
                .geometry
                .filter(|_| count == 1)
                .and_then(|geometry| pieces.sizes.get(&geometry).copied()),
        }
    }));
}

fn outside(view_projection: Mat4, [low, high]: [Vec3; 2]) -> bool {
    let mut beyond = [true; 5];
    for corner in 0..8_u32 {
        let point = Vec3::new(
            if corner & 1 == 0 { low.x } else { high.x },
            if corner & 2 == 0 { low.y } else { high.y },
            if corner & 4 == 0 { low.z } else { high.z },
        );
        let clip = view_projection * point.extend(1.0);
        beyond[0] &= clip.x < -clip.w;
        beyond[1] &= clip.x > clip.w;
        beyond[2] &= clip.y < -clip.w;
        beyond[3] &= clip.y > clip.w;
        beyond[4] &= clip.w <= 0.0;
    }
    beyond.contains(&true)
}

fn pixel_scale(facts: &ViewFacts) -> Option<(f32, bool)> {
    let half = facts.pixels[3] as f32 * 0.5;
    match facts.lens.projection {
        Projection::Perspective {
            vertical_fov_radians,
        } => {
            let slope = (vertical_fov_radians * 0.5).tan();
            (slope > 0.0).then(|| (half / slope, false))
        }
        Projection::Fitted { clip } => Some((half * clip[1][1].abs(), false)),
        Projection::Orthographic { height } => (height > 0.0).then(|| (half * 2.0 / height, true)),
    }
}

pub(super) struct Sized<'a> {
    pub limit: DetailLimit,
    pub held: Option<&'a mut BTreeSet<GeometryHandle>>,
    pub set: Option<SetView<'a>>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Counts {
    pub framed: u32,
    pub hidden: u32,
    pub small: u32,
}

impl Sized<'_> {
    fn shown(&mut self, chunk: &Chunk, near_squared: f32, scale: Option<(f32, bool)>) -> bool {
        let limit = self.limit;
        let (Some(size), Some((scale, flat))) = (chunk.largest_m, scale) else {
            return true;
        };
        if size >= limit.never_m || limit.least_px <= 0.0 || near_squared <= 0.0 {
            return true;
        }
        let pixels = if flat {
            size * scale
        } else {
            size * scale / near_squared.sqrt()
        };
        let was = self
            .held
            .as_deref()
            .zip(chunk.mesh)
            .is_some_and(|(held, mesh)| held.contains(&mesh));
        let least = if was {
            limit.least_px * (1.0 - limit.overlap)
        } else {
            limit.least_px
        };
        let shown = pixels >= least;
        if let (Some(held), Some(mesh)) = (self.held.as_deref_mut(), chunk.mesh) {
            if shown {
                held.insert(mesh);
            } else {
                held.remove(&mesh);
            }
        }
        shown
    }
}

pub(super) fn ordered(
    chunks: &[Chunk],
    facts: &ViewFacts,
    sized: &mut Sized<'_>,
    into: &mut Vec<u32>,
) -> Counts {
    into.clear();
    let scale = pixel_scale(facts);
    let mut counts = Counts::default();
    for (at, chunk) in chunks.iter().enumerate() {
        let seen = chunk.layers.overlaps(facts.layers)
            && chunk
                .bounds
                .is_none_or(|bounds| !outside(facts.view_projection, bounds));
        if !seen {
            continue;
        }
        counts.framed += 1;
        if let Some(set) = sized.set
            && set.hides(chunk.mesh) == Some(true)
        {
            counts.hidden += 1;
            continue;
        }
        if !sized.shown(chunk, chunk.nearest(facts.eye), scale) {
            counts.small += 1;
            continue;
        }
        into.push(at as u32);
    }
    into.sort_by(|&a, &b| {
        let near = |at: u32| chunks[at as usize].nearest(facts.eye);
        near(a).total_cmp(&near(b))
    });
    counts
}
