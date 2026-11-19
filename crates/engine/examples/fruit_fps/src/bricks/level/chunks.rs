use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use engine::prelude::*;
use engine_core::{AssetLibrary, Drawn, GeometryHandle, Projection, ViewFacts};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::LEVEL_DETAIL;
use super::sets::SetView;
use crate::bricks::figures::LEVEL;
use crate::play::light::DetailLimit;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LevelPiece {
    pub largest_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Piece {
    bounds: [Vec3; 2],
    pub largest_m: Option<f32>,
}

#[derive(Resource, Clone, Debug, Default)]
pub(super) struct Pieces {
    found: BTreeMap<GeometryHandle, Piece>,
}

impl Pieces {
    pub fn single(&self, mesh: Option<GeometryHandle>, instances: &Range<u32>) -> Option<&Piece> {
        if instances.len() != 1 {
            return None;
        }
        self.found.get(&mesh?)
    }

    pub fn sized(&self) -> impl Iterator<Item = (GeometryHandle, &Piece, f32)> + '_ {
        self.found
            .iter()
            .filter_map(|(mesh, piece)| Some((*mesh, piece, piece.largest_m?)))
    }
}

type Touched<'w, 's> = Query<'w, 's, (), Or<(Changed<Drawn>, Changed<LevelPiece>)>>;

pub(super) fn pieces(
    found: Query<(&Drawn, Option<&LevelPiece>)>,
    touched: Touched,
    library: Option<Res<AssetLibrary>>,
    mut pieces: ResMut<Pieces>,
) {
    if touched.is_empty() && !library.as_ref().is_some_and(DetectChanges::is_changed) {
        return;
    }
    let library = library.as_deref();
    let next: BTreeMap<GeometryHandle, Piece> = found
        .iter()
        .filter(|(drawn, _)| drawn.kind == LEVEL || drawn.kind == LEVEL_DETAIL)
        .filter_map(|(drawn, piece)| {
            let mesh = drawn.geometry?;
            let bounds = library?.mesh(mesh)?.bounds();
            Some((
                mesh,
                Piece {
                    bounds: [bounds.min(), bounds.max()],
                    largest_m: piece.map(|piece| piece.largest_m),
                },
            ))
        })
        .collect();
    if pieces.found != next {
        pieces.found = next;
    }
}

fn nearest([low, high]: [Vec3; 2], eye: Vec3) -> f32 {
    eye.max(low).min(high).distance_squared(eye)
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

pub(super) fn pixel_scale(facts: &ViewFacts) -> Option<(f32, bool)> {
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

#[derive(Clone, Copy)]
pub(super) enum Seen {
    Outside,
    Hidden,
    Near(f32),
}

pub(super) fn seen(
    piece: Option<&Piece>,
    mesh: Option<GeometryHandle>,
    facts: &ViewFacts,
    set: Option<SetView<'_>>,
) -> Seen {
    if piece.is_some_and(|piece| outside(facts.view_projection, piece.bounds)) {
        return Seen::Outside;
    }
    if set.is_some_and(|set| set.hides(mesh) == Some(true)) {
        return Seen::Hidden;
    }
    Seen::Near(piece.map_or(0.0, |piece| nearest(piece.bounds, facts.eye)))
}

pub(super) struct Sized<'a> {
    pub limit: DetailLimit,
    pub held: Option<&'a mut BTreeSet<GeometryHandle>>,
}

impl Sized<'_> {
    pub fn shown(
        &mut self,
        (mesh, size): (GeometryHandle, f32),
        near_squared: f32,
        scale: Option<(f32, bool)>,
    ) -> bool {
        let limit = self.limit;
        let Some((scale, flat)) = scale else {
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
            .is_some_and(|held| held.contains(&mesh));
        let least = if was {
            limit.least_px * (1.0 - limit.overlap)
        } else {
            limit.least_px
        };
        let shown = pixels >= least;
        if let Some(held) = self.held.as_deref_mut() {
            if shown {
                held.insert(mesh);
            } else {
                held.remove(&mesh);
            }
        }
        shown
    }
}
