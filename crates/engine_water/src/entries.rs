use engine_render::graph::Targets;
use engine_render::shaders::Entry;

use crate::shaders::entry::meshes;

pub(crate) fn water(targets: Targets) -> (Entry, Entry) {
    match targets {
        Targets::Post => (meshes::vs_water, meshes::fs_water_post),
        Targets::Temporal => (meshes::vs_water, meshes::fs_water_temporal),
        Targets::Debug => (meshes::vs_water, meshes::fs_debug_water),
    }
}

pub(crate) fn falls(targets: Targets) -> (Entry, Entry) {
    match targets {
        Targets::Post => (meshes::vs_fall, meshes::fs_fall_post),
        Targets::Temporal => (meshes::vs_fall_temporal, meshes::fs_fall_temporal),
        Targets::Debug => (meshes::vs_fall, meshes::fs_debug_fall),
    }
}
