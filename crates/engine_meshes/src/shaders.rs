engine_render::shader_modules! {
    "meshes::shade" in "meshes" = "shade.wgsl";
    "meshes::forward" in "meshes" = "forward.wgsl";
    "meshes::impostors" in "meshes" = "impostors.wgsl";
    "meshes::debug_draw" in "meshes" = "debug_draw.wgsl";
}

engine_render::entry_points! {
    meshes {
        vertex: [
            vs_impostor_bake @ "meshes::impostors",
            vs_impostor_bake_extended @ "meshes::impostors",
            vs_impostor @ "meshes::impostors",
            vs_impostor_temporal @ "meshes::impostors",
            vs_debug_impostor @ "meshes::debug_draw",
        ],
        fragment: [
            fs_post @ "meshes::forward",
            fs_temporal @ "meshes::forward",
            fs_probe @ "meshes::forward",
            fs_probe_extended @ "meshes::forward",
            fs_post_extended @ "meshes::forward",
            fs_temporal_extended @ "meshes::forward",
            fs_masked_depth @ "meshes::forward",
            fs_masked_depth_extended @ "meshes::forward",
            fs_masked_depth_temporal_extended @ "meshes::forward",
            fs_impostor_shade @ "meshes::impostors",
            fs_impostor_shade_extended @ "meshes::impostors",
            fs_impostor_bake @ "meshes::impostors",
            fs_impostor_bake_extended @ "meshes::impostors",
            fs_impostor_post @ "meshes::impostors",
            fs_impostor_temporal @ "meshes::impostors",
            fs_debug_mesh @ "meshes::debug_draw",
            fs_debug_impostor @ "meshes::debug_draw",
        ],
    }
}
