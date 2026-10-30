use engine_render::shaders::Entry;

use crate::shaders::entry::{terrain, terrain_shadow};

pub(crate) const TERRAIN_DEBUG: (Entry, Entry) = (terrain::vs_terrain, terrain::fs_debug_terrain);

pub(crate) const TERRAIN_VIEW: (Entry, Entry) = (terrain::vs_terrain, terrain::fs_terrain_probe);

pub(crate) const TERRAIN_SHADOW_VERTEX: Entry = terrain_shadow::vs_terrain_shadow;
