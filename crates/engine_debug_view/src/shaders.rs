engine_render::shader_modules! {
    "debug::view" in "debug_view" = "debug_view.wgsl";
}

engine_render::entry_points! {
    debug_view {
        vertex: [vs_fullscreen @ "engine::fullscreen"],
        fragment: [fs_debug_view @ "debug::view"],
    }
}
