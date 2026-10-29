engine_render::shader_modules! {
    "ambient_occlusion::occlusion" in "ambient_occlusion" = "occlusion.wgsl";
}

engine_render::entry_points! {
    ambient_occlusion {
        vertex: [vs_fullscreen @ "engine::fullscreen"],
        fragment: [
            fs_occlusion @ "ambient_occlusion::occlusion",
            fs_effects_filter @ "ambient_occlusion::occlusion",
        ],
    }
}
