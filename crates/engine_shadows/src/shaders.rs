engine_render::shader_modules! {
    "shadows::sample" in "shadows" = "shadows/sample.wgsl";
    "shadows::maps" in "shadows" = "shadows.wgsl";
    "shadows::marks" in "shadows" = "shadow_marks.wgsl";
    "face::shadows" in "shadows" = "faces/shadows.wgsl";
    "shadows::piece" in "shadows" = "piece.wgsl";
}

engine_render::entry_points! {
    shadow {
        vertex: [
            vs_affine @ "shadows::maps",
            vs_affine_masked @ "shadows::maps",
            vs_impostor_card @ "shadows::maps",
            vs_compose @ "shadows::maps",
            vs_moving @ "shadows::maps",
            vs_moving_masked @ "shadows::maps",
            vs_clear_moving @ "shadows::maps",
        ],
        fragment: [
            fs_impostor_card @ "shadows::maps",
            fs_masked @ "shadows::maps",
            fs_compose @ "shadows::maps",
            fs_moving @ "shadows::maps",
            fs_moving_masked @ "shadows::maps",
        ],
        compute: [select_casters @ "shadows::maps", count_casters @ "shadows::maps"],
    }
    shadow_marks {
        compute: [mark_shadow_tiles @ "shadows::marks"],
    }
}
