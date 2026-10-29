use engine_render::graph::Targets;
use engine_render::shaders::Entry;

use crate::shaders::entry::{terrain, terrain_shadow};

pub(crate) fn terrain(targets: Option<Targets>) -> (Entry, Entry) {
    match targets {
        Some(Targets::Canvas) => (terrain::vs_terrain, terrain::fs_terrain),
        Some(Targets::Post) => (terrain::vs_terrain, terrain::fs_terrain_post),
        Some(Targets::Temporal) => (terrain::vs_terrain_temporal, terrain::fs_terrain_temporal),
        Some(Targets::Debug) => (terrain::vs_terrain, terrain::fs_debug_terrain),
        None => (terrain::vs_terrain, terrain::fs_terrain_probe),
    }
}

pub(crate) const TERRAIN_SHADOW_VERTEX: Entry = terrain_shadow::vs_terrain_shadow;
