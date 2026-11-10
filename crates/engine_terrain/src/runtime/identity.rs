use std::sync::Arc;

use super::content_hash;
use super::{Terrain, TerrainEdit, TerrainSurface};
use crate::Stamp;
use engine_core::RecordKey;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Generation(u128);

impl Generation {
    pub(super) fn of(edits: &[TerrainEdit], recipe: Option<RecordKey>) -> Self {
        let stamps: Vec<Stamp> = edits.iter().map(|edit| edit.stamp).collect();
        let mut bytes = crate::codec::encode(&stamps);
        bytes.extend(recipe.map_or([0; 32], |recipe| recipe.hash()));
        Self(content_hash(&bytes))
    }
}

impl Terrain {
    pub(super) fn same(&self, other: &Self) -> bool {
        let Self {
            origin,
            pyramid,
            layers,
            surface,
            collision,
            edits: _,
            rivers: _,
            surroundings: _,
        } = self;
        let TerrainSurface {
            rules,
            materials,
            assets: _,
            bridge: _,
        } = surface;
        Arc::ptr_eq(layers, &other.layers)
            && *origin == other.origin
            && *pyramid == other.pyramid
            && (Arc::ptr_eq(rules, &other.surface.rules) || *rules == other.surface.rules)
            && *materials == other.surface.materials
            && *collision == other.collision
    }
}
