gpu_consts! {
    pub const FLAG_VISIBLE: u32 = 1 => ["VISIBLE" in "presentation::cull"];
    pub const FLAG_PHASE1: u32 = 8 => ["PHASE1" in "presentation::cull"];
    pub const FLAG_PHASE2: u32 = 16 => ["PHASE2" in "presentation::cull"];
    pub const FLAG_PROBE: u32 = 128 => ["PROBE" in "presentation::cull"];
    pub const STILL_FRAMES: u32 = 16 => ["STILL_FRAMES"];
    pub const MOVED_SLOT: u32 = 1 << 31 => ["MOVED_SLOT"];
    pub const LOD_LEVELS: usize = 3 => ["LOD_LEVELS" in "presentation::cull", "IMPOSTOR" in "presentation::cull"];
    pub const LOD_STATE: u32 = 1 << 31 => ["LOD_STATE" in "presentation::cull"];
    pub const DRAW_ARGS_BYTES: u64 = 20 => ["size_of DrawArgs" in "presentation::cull"];
    pub const COMPOSED_BYTES: u64 = 112 => ["size_of Composed" in "presentation::cull"];

    pub const PRESENTATION_GROUP: u32 = 256 => ["@shift_previous" in "presentation::cull", "@cull" in "presentation::cull", "@count_still" in "presentation::cull", "@fade_levels" in "presentation::cull", "@occlusion_test" in "presentation::cull", "@probe_cull" in "presentation::cull"];
    pub const SCAN_GROUP: u32 = 64 => ["GROUP" in "presentation::cull", "@sum_groups" in "presentation::cull", "@scan_groups" in "presentation::cull", "@scatter" in "presentation::cull"];
    pub const PARENT_GROUP: u32 = 64 => ["@compose_parents" in "presentation::cull"];
    pub const ARGS_GROUP: u32 = 64 => ["@batch_args" in "presentation::cull"];

    pub const IMPOSTOR_FRAMES_SHIFT: u32 = 19 => ["IMPOSTOR_FRAMES_SHIFT"];
    pub const IMPOSTOR_HEMISPHERE: u32 = 1 << 16 => ["IMPOSTOR_HEMISPHERE"];
    pub const IMPOSTOR_MASKED: u32 = 1 << 17 => ["IMPOSTOR_MASKED"];
    pub const IMPOSTOR_FOLIAGE: u32 = 1 << 24 => ["IMPOSTOR_FOLIAGE"];
    pub const MASK_MIP_SCALE: f32 = 0.25 => ["MASK_MIP_SCALE"];

    pub const MATERIAL_ANISOTROPY: u16 = 16 => ["DETAIL_ANISOTROPY"];

    pub const PYRAMID_GROUP: u32 = 8 => ["@pyramid_first" in "engine::pyramid", "@pyramid_down" in "engine::pyramid", "@pyramid_first_nearest" in "engine::pyramid", "@pyramid_down_nearest" in "engine::pyramid"];

    pub const PYRAMID_EXTENT_BYTES: usize = 16 => ["size_of Extent" in "engine::pyramid"];

    pub const VARIANT_MAPS: u32 = 1 => ["VARIANT_MAPS" in "engine::mesh::declarations"];
    pub const VARIANT_NORMAL_MAP: u32 = 2 => ["VARIANT_NORMAL_MAP" in "engine::mesh::declarations"];
    pub const VARIANT_DOUBLE_SIDED: u32 = 8 => ["VARIANT_DOUBLE_SIDED" in "engine::mesh::declarations"];
    pub const VARIANT_METALLIC_ROUGHNESS_MAP: u32 = 128 => ["VARIANT_METALLIC_ROUGHNESS_MAP" in "engine::mesh::declarations"];
    pub const VARIANT_OCCLUSION_MAP: u32 = 256 => ["VARIANT_OCCLUSION_MAP" in "engine::mesh::declarations"];
    pub const VARIANT_EMISSIVE_MAP: u32 = 512 => ["VARIANT_EMISSIVE_MAP" in "engine::mesh::declarations"];
    pub const VARIANT_PACKED_OCCLUSION: u32 = 1024 => ["VARIANT_PACKED_OCCLUSION" in "engine::mesh::declarations"];
    pub const VARIANT_WORLD_MAPPING: u32 = 4096 => ["VARIANT_WORLD_MAPPING" in "engine::mesh::declarations"];
    pub const VARIANT_TOP_LAYER: u32 = 8192 => ["VARIANT_TOP_LAYER" in "engine::mesh::declarations"];
    pub const VARIANT_FOLIAGE: u32 = 65536 => ["VARIANT_FOLIAGE" in "engine::mesh::declarations"];

    pub const CARD_FADING: u32 = 1 << 31 => ["IMPOSTOR_FADING", "CARD_FADING" in "presentation::cull"];
    pub const CARD_FADING_OUT: u32 = 1 << 30 => ["IMPOSTOR_FADING_OUT", "CARD_FADING_OUT" in "presentation::cull"];
    pub const CARD_OVER: u32 = 1 << 15 => ["IMPOSTOR_OVER", "CARD_OVER" in "presentation::cull"];
}

pub const IMPOSTOR_KINDS: u32 = 4;
