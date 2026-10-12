//! When a terrain changed so much that its tiles must go, and when it is
//! the same terrain whose tiles stay.

use std::sync::Arc;

use super::{Terrain, TerrainSource, TerrainSurface};

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
            // Not compared.
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
