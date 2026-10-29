use engine_render::graph::Targets;
use engine_render::shaders::Entry;

engine_render::shader_modules! {
    "cover::draw" in "ground_cover" = "cover.wgsl";
    "cover::debug" in "ground_cover" = "cover_debug.wgsl";
    "cover::grow" in "ground_cover" = "cover_grow.wgsl";
}

engine_render::entry_points! {
    cover {
        vertex: [vs_cover @ "cover::draw", vs_cover_temporal @ "cover::draw"],
        fragment: [
            fs_cover @ "cover::draw",
            fs_cover_post @ "cover::draw",
            fs_cover_temporal @ "cover::draw",
            fs_debug_cover @ "cover::debug",
        ],
    }
    cover_grow {
        compute: [grow_cover @ "cover::grow", finish_cover @ "cover::grow"],
    }
}

pub(crate) fn cover(targets: Targets) -> (Entry, Entry) {
    match targets {
        Targets::Canvas => (entry::cover::vs_cover, entry::cover::fs_cover),
        Targets::Post => (entry::cover::vs_cover, entry::cover::fs_cover_post),
        Targets::Temporal => (
            entry::cover::vs_cover_temporal,
            entry::cover::fs_cover_temporal,
        ),
        Targets::Debug => (entry::cover::vs_cover, entry::cover::fs_debug_cover),
    }
}
