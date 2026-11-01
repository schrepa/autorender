use super::scene::MAX_LEVELS;

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

    pub const CARD_VERTICES: u32 = 6 => ["IMPOSTOR_VERTICES" in "presentation::cull"];

    pub const IMPOSTOR_FRAMES_SHIFT: u32 = 19 => ["IMPOSTOR_FRAMES_SHIFT"];
    pub const IMPOSTOR_HEMISPHERE: u32 = 1 << 16 => ["IMPOSTOR_HEMISPHERE"];
    pub const IMPOSTOR_MASKED: u32 = 1 << 17 => ["IMPOSTOR_MASKED"];
    pub const IMPOSTOR_FOLIAGE: u32 = 1 << 24 => ["IMPOSTOR_FOLIAGE"];
    pub const MASK_MIP_SCALE: f32 = 0.25 => ["MASK_MIP_SCALE"];

    pub const MIRRORED_BITS: u32 = 14 => ["MIRRORED_BITS"];

    pub const MATERIAL_ANISOTROPY: u16 = 16 => ["DETAIL_ANISOTROPY"];

    pub const PYRAMID_GROUP: u32 = 8 => ["@pyramid_first" in "engine::pyramid", "@pyramid_down" in "engine::pyramid", "@pyramid_first_nearest" in "engine::pyramid", "@pyramid_down_nearest" in "engine::pyramid"];

    pub const PYRAMID_EXTENT_BYTES: usize = 16 => ["size_of Extent" in "engine::pyramid"];

    pub const VARIANT_MAPS: u32 = 1 => ["VARIANT_MAPS" in "meshes::declarations"];
    pub const VARIANT_NORMAL_MAP: u32 = 2 => ["VARIANT_NORMAL_MAP" in "meshes::declarations"];
    pub const VARIANT_DOUBLE_SIDED: u32 = 8 => ["VARIANT_DOUBLE_SIDED" in "meshes::declarations"];
    pub const VARIANT_METALLIC_ROUGHNESS_MAP: u32 = 128 => ["VARIANT_METALLIC_ROUGHNESS_MAP" in "meshes::declarations"];
    pub const VARIANT_OCCLUSION_MAP: u32 = 256 => ["VARIANT_OCCLUSION_MAP" in "meshes::declarations"];
    pub const VARIANT_EMISSIVE_MAP: u32 = 512 => ["VARIANT_EMISSIVE_MAP" in "meshes::declarations"];
    pub const VARIANT_PACKED_OCCLUSION: u32 = 1024 => ["VARIANT_PACKED_OCCLUSION" in "meshes::declarations"];
    pub const VARIANT_WORLD_MAPPING: u32 = 4096 => ["VARIANT_WORLD_MAPPING" in "meshes::declarations"];
    pub const VARIANT_TOP_LAYER: u32 = 8192 => ["VARIANT_TOP_LAYER" in "meshes::declarations"];
    pub const VARIANT_FOLIAGE: u32 = 65536 => ["VARIANT_FOLIAGE" in "meshes::declarations"];

    pub const DEBUG_SPAN_STEPS: f32 = 254.0 => ["DEBUG_SPAN_STEPS"];
    pub const WATER_SPAN_LEAST: f32 = 0.25 => ["WATER_SPAN_LEAST" in "place::heightfield"];
    pub const WATER_SPAN_MOST: f32 = 4096.0 => ["WATER_SPAN_MOST" in "place::heightfield"];
    pub const DEBUG_KIND_MESH: u32 = 1 => ["DEBUG_KIND_MESH"];
    pub const DEBUG_KIND_PLACED: u32 = 2 => ["DEBUG_KIND_PLACED"];
    pub const DEBUG_KIND_CARD: u32 = 3 => ["DEBUG_KIND_CARD"];
    pub const DEBUG_KIND_PLACED_CARD: u32 = 4 => ["DEBUG_KIND_PLACED_CARD"];
    pub const DEBUG_KIND_TERRAIN: u32 = 6 => ["DEBUG_KIND_TERRAIN"];
    pub const DEBUG_KIND_COVER: u32 = 7 => ["DEBUG_KIND_COVER"];
    pub const DEBUG_KIND_WATER: u32 = 8 => ["DEBUG_KIND_WATER"];
    pub const DEBUG_KIND_FALLS: u32 = 10 => ["DEBUG_KIND_FALLS"];
    pub const DEBUG_KIND_SKY: u32 = 11 => ["DEBUG_KIND_SKY"];
    pub const DEBUG_LEVEL_SHIFT: u32 = 4 => ["DEBUG_LEVEL_SHIFT"];
    pub const DEBUG_FADING: u32 = 0x100 => ["DEBUG_FADING"];
    pub const DEBUG_CASTS: u32 = 0x200 => ["DEBUG_CASTS"];
    pub const DEBUG_SOURCE_SHIFT: u32 = 16 => ["DEBUG_SOURCE_SHIFT"];
    pub const DEBUG_SUN_NONE: u32 = 0 => ["DEBUG_SUN_NONE"];
    pub const DEBUG_SUN_LEVEL: u32 = 1 => ["DEBUG_SUN_LEVEL"];
    pub const DEBUG_SUN_HORIZON: u32 = DEBUG_SUN_LEVEL + MAX_LEVELS as u32 => ["DEBUG_SUN_HORIZON"];
    pub const DEBUG_SUN_BEYOND: u32 = DEBUG_SUN_HORIZON + 1 => ["DEBUG_SUN_BEYOND"];
    pub const DEBUG_SHADOWED: u32 = 0x80 => ["DEBUG_SHADOWED"];
    pub const DEBUG_TILE_EDGE: u32 = 0x800 => ["DEBUG_TILE_EDGE"];
    pub const DEBUG_PATCH_EDGE: u32 = 0x1000 => ["DEBUG_PATCH_EDGE"];
    pub const DEBUG_PATCH_SHIFT: u32 = 24 => ["DEBUG_PATCH_SHIFT"];
    pub const DEBUG_NO_MATERIAL: u32 = 0xff => ["DEBUG_NO_MATERIAL"];

    pub const WIND_FULL_SPEED: f32 = 12.0 => ["WIND_FULL_SPEED"];
    pub const WIND_SPAN: f32 = 10.0 => ["WIND_SPAN"];
    pub const WIND_TRUNK_FLUTTER: f32 = 0.12 => ["WIND_TRUNK_FLUTTER"];
    pub const WIND_TRUNK_RATE: f32 = 4.7 => ["WIND_TRUNK_RATE"];
    pub const WIND_BRANCH: f32 = 4.0 => ["WIND_BRANCH"];
    pub const WIND_BRANCH_RATE: f32 = 3.1 => ["WIND_BRANCH_RATE"];
    pub const WIND_BRANCH_RISE: f32 = 2.1 => ["WIND_BRANCH_RISE"];
    pub const WIND_FLUTTER: f32 = 0.2 => ["WIND_FLUTTER"];
    pub const WIND_FLUTTER_RATE: f32 = 13.0 => ["WIND_FLUTTER_RATE"];
    pub const WIND_SHAKE_WHOLE: f32 = 1.0 => ["WIND_SHAKE_WHOLE"];
    pub const WIND_SHAKE_STILL: f32 = 0.5 => ["WIND_SHAKE_STILL"];

    pub const CARD_FADING: u32 = 1 << 31 => ["IMPOSTOR_FADING", "CARD_FADING" in "presentation::cull"];
    pub const CARD_FADING_OUT: u32 = 1 << 30 => ["IMPOSTOR_FADING_OUT", "CARD_FADING_OUT" in "presentation::cull"];
    pub const CARD_OVER: u32 = 1 << 15 => ["IMPOSTOR_OVER", "CARD_OVER" in "presentation::cull"];
}

pub const IMPOSTOR_KINDS: u32 = 4;

gpu_mirrors! {
    engine_core::NO_PARENT => ["NO_PARENT" in "presentation::cull"];
    engine_core::NODE_VALID => ["NODE_VALID" in "presentation::cull"];
    engine_core::NODE_BLEND => ["NODE_BLEND" in "presentation::cull"];
    engine_core::NODE_ACTIVE => ["NODE_ACTIVE" in "presentation::cull"];
    engine_core::NODE_PARENT_ACTIVE => ["NODE_PARENT_ACTIVE" in "presentation::cull"];
    engine_core::presentation_bounds::MAX_COEFFICIENT => ["MAX_COEFFICIENT" in "presentation::cull"];
    engine_core::presentation_bounds::ROUNDING_MARGIN => ["ROUNDING_MARGIN" in "presentation::cull"];
    engine_core::presentation_bounds::MAX_PRODUCT_BOUND => ["MAX_PRODUCT_BOUND" in "presentation::cull"];
    engine_core::presentation_bounds::COEFFICIENT_LIMIT => ["COEFFICIENT_LIMIT" in "presentation::cull"];
    engine_core::presentation_bounds::MIN_LINEAR_SCALE => ["MIN_LINEAR_SCALE" in "presentation::cull"];
    engine_core::presentation_bounds::COMMON_DETERMINANT_MARGIN => ["COMMON_DETERMINANT_MARGIN" in "presentation::cull"];
    engine_core::presentation_bounds::LINEAR_ROTATION_DOT_THRESHOLD => ["LINEAR_ROTATION_DOT" in "presentation::cull"];
}
