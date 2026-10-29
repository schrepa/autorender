engine_render::shader_modules! {
    "bloom::piece" in "bloom" = "piece.wgsl";
    "face::bloom" in "bloom" = "faces/bloom.wgsl";
    "bloom::chain" in "bloom" = "chain.wgsl";
    "bloom::apply" in "bloom" = "apply.wgsl";
}

engine_render::entry_points! {
    bloom {
        vertex: [vs_fullscreen @ "engine::fullscreen"],
        fragment: [
            fs_bloom_first @ "bloom::chain",
            fs_bloom_up @ "bloom::chain",
        ],
    }
}
