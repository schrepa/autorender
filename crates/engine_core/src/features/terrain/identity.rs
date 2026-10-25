//! When a terrain changed so much that its tiles must go, when it is the
//! same terrain whose tiles stay, and which of those tiles are what it
//! makes now: their generation.

use std::sync::Arc;

use engine_terrain::{GivenRiver, Stamp};

use super::{Terrain, TerrainEdit, TerrainRiver, TerrainSurface};
use crate::content_hash;

/// What a terrain's tiles are made from, by content: a hash of its edits'
/// stamps, of the rivers drawn over the layers its bakes hold and of
/// whether tiles hold control values, which is all that changes while a
/// terrain streams ([`Terrain::same`] starts it over for anything else).
/// Resident tiles and the jobs making them carry the generation they were
/// made from; another generation's tiles are not the ground now, while the
/// same content always has the same generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Generation(u128);

impl Generation {
    /// The generation of tiles made under `edits` over bakes holding the
    /// rivers `drawn`, holding control values where `materials`.
    pub(super) fn of(edits: &[TerrainEdit], drawn: &[TerrainRiver], materials: bool) -> Self {
        let stamps: Vec<Stamp> = edits.iter().map(|edit| edit.stamp).collect();
        let rivers: Vec<GivenRiver> = drawn.iter().map(|drawn| drawn.river.clone()).collect();
        let mut bytes = engine_terrain::codec::encode(&stamps);
        bytes.extend(engine_terrain::codec::encode(&rivers));
        bytes.push(u8::from(materials));
        Self(content_hash(&bytes))
    }
}

impl Terrain {
    /// Whether `other` describes the same terrain: equal fields and the
    /// same shared source, so streaming keeps its tiles; edits apply over
    /// it tile by tile. Every field is named, so one added says whether it
    /// keeps the tiles or fails to build.
    pub(super) fn same(&self, other: &Self) -> bool {
        let Self {
            origin,
            pyramid,
            layers,
            surface,
            collision,
            // Applied over the tiles kept, tile by tile.
            edits: _,
            // Baked again, the tiles kept where its rivers stay.
            rivers: _,
            // Followed on their own.
            surroundings: _,
        } = self;
        let TerrainSurface {
            rules,
            materials,
            // Taken in place: tiles place objects by the rules alone.
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
