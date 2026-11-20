use std::ops::Range;

use engine::prelude::*;
use engine_core::collections::{Cursor, Since};
use engine_core::{Asset, Assets, Drawn, MeshFile, ViewFacts};
use engine_drawing::{nearest_squared, outside};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::sets::SetView;
use super::{LEVEL, LEVEL_DETAIL};

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
    found: Vec<Option<(Asset<MeshFile>, Piece)>>,
}

impl Pieces {
    pub fn single(
        &self,
        mesh: Option<Asset<MeshFile>>,
        instances: &Range<u32>,
    ) -> Option<&Piece> {
        if instances.len() != 1 {
            return None;
        }
        let mesh = mesh?;
        let (known, piece) = self.found.get(mesh.slot())?.as_ref()?;
        (*known == mesh).then_some(piece)
    }
}

type Touched<'w, 's> = Query<'w, 's, &'static Drawn, Or<(Changed<Drawn>, Changed<LevelPiece>)>>;

fn level(drawn: &Drawn) -> bool {
    drawn.kind == LEVEL || drawn.kind == LEVEL_DETAIL
}

pub(super) fn pieces(
    found: Query<(&Drawn, Option<&LevelPiece>)>,
    touched: Touched,
    assets: Res<Assets>,
    mut seen: Local<Cursor>,
    mut pieces: ResMut<Pieces>,
) {
    let meshes = match assets.changed(&mut seen) {
        Since::Same => false,
        Since::Listed(ids) => ids.iter().any(|&id| assets.asset::<MeshFile>(id).is_some()),
        Since::All => true,
    };
    if !meshes && !touched.iter().any(level) {
        return;
    }
    let mut next: Vec<Option<(Asset<MeshFile>, Piece)>> = Vec::new();
    for (drawn, piece) in &found {
        if !level(drawn) {
            continue;
        }
        let Some(mesh) = drawn.geometry else {
            continue;
        };
        let Ok(loaded) = assets.get(mesh) else {
            continue;
        };
        let bounds = loaded.description().bounds;
        let slot = mesh.slot();
        if next.len() <= slot {
            next.resize(slot + 1, None);
        }
        next[slot] = Some((
            mesh,
            Piece {
                bounds: [bounds.min, bounds.max],
                largest_m: piece.map(|piece| piece.largest_m),
            },
        ));
    }
    if pieces.found != next {
        pieces.found = next;
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
    mesh: Option<Asset<MeshFile>>,
    facts: &ViewFacts,
    set: Option<SetView<'_>>,
) -> Seen {
    if piece.is_some_and(|piece| outside(facts.view_projection, piece.bounds)) {
        return Seen::Outside;
    }
    if set.is_some_and(|set| set.hides(mesh) == Some(true)) {
        return Seen::Hidden;
    }
    Seen::Near(piece.map_or(0.0, |piece| nearest_squared(piece.bounds, facts.eye)))
}

pub(super) fn near(
    piece: Option<&Piece>,
    mesh: Option<Asset<MeshFile>>,
    facts: &ViewFacts,
    set: Option<SetView<'_>>,
) -> Option<f32> {
    if set.is_some_and(|set| set.hides(mesh) == Some(true)) {
        return None;
    }
    if piece.is_some_and(|piece| outside(facts.view_projection, piece.bounds)) {
        return None;
    }
    Some(piece.map_or(0.0, |piece| nearest_squared(piece.bounds, facts.eye)))
}
