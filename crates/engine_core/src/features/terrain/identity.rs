//! When a terrain changed so much that its tiles must go, when it is the
//! same terrain whose tiles stay, and which of those tiles are what it
//! makes now: their generation.

use std::sync::Arc;

use engine_terrain::Stamp;

use super::{Terrain, TerrainEdit, TerrainSource, TerrainSurface};
use crate::content_hash;

/// What a terrain's tiles are made from, by content: a hash of its edits'
/// stamps and of whether tiles hold control values, which is all that
/// changes while a terrain streams ([`Terrain::same`] starts it over for
/// anything else). Resident tiles and the jobs making them carry the
/// generation they were made from; another generation's tiles are not
/// the ground now, while the same content always has the same generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Generation(u128);

impl Generation {
    /// The generation of tiles made under `edits`, holding control values
    /// where `materials`.
    pub(super) fn of(edits: &[TerrainEdit], materials: bool) -> Self {
        let stamps: Vec<Stamp> = edits.iter().map(|edit| edit.stamp).collect();
        let mut bytes = engine_terrain::codec::encode(&stamps);
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
            source,
            surface,
            collision,
            // Applied over the tiles kept, tile by tile.
            edits: _,
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
        let source = match (source, &other.source) {
            (TerrainSource::Generated(a), TerrainSource::Generated(b)) => Arc::ptr_eq(a, b),
            (
                TerrainSource::File {
                    header: a,
                    url: u,
                    data: _,
                },
                TerrainSource::File {
                    header: b,
                    url: v,
                    data: _,
                },
            ) => Arc::ptr_eq(a, b) && u == v,
            (TerrainSource::Generated(_), TerrainSource::File { .. })
            | (TerrainSource::File { .. }, TerrainSource::Generated(_)) => false,
        };
        source
            && *origin == other.origin
            && *pyramid == other.pyramid
            && (Arc::ptr_eq(rules, &other.surface.rules) || *rules == other.surface.rules)
            && *materials == other.surface.materials
            && *collision == other.collision
    }
}
