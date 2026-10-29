engine_render::shader_modules! {
    "taa::resolve" in "taa" = "resolve.wgsl";
}

engine_render::entry_points! {
    temporal {
        vertex: [vs_fullscreen @ "engine::fullscreen"],
    }
}
