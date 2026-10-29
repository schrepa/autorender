engine_render::shader_modules! {
    "pond::surface" in "pond" = "surface.wgsl";
}

engine_render::entry_points! {
    pond {
        vertex: [vs_pond @ "pond::surface"],
        fragment: [
            fs_pond @ "pond::surface",
            fs_pond_post @ "pond::surface",
            fs_pond_temporal @ "pond::surface",
        ],
    }
}
