engine_render::shader_modules! {
    "post::reflections" in "reflections" = "reflections.wgsl";
    "reflections::levels" in "reflections" = "levels.wgsl";
    "reflections::piece" in "reflections" = "piece.wgsl";
    "face::reflections" in "reflections" = "faces/reflections.wgsl";
}

engine_render::entry_points! {
    reflections {
        vertex: [vs_fullscreen @ "engine::fullscreen"],
        fragment: [
            fs_reflections @ "post::reflections",
            fs_levels_first @ "reflections::levels",
        ],
    }
}
