use super::scene::{MAX_LEVELS, MAX_MOVING_TILES};
use super::terrain::PaletteEntry;

gpu_consts! {
    pub(crate) const FLAG_VISIBLE: u32 = 1 => ["VISIBLE" in PRESENTATION_SHADER];
    pub(crate) const FLAG_PHASE1: u32 = 8 => ["PHASE1" in PRESENTATION_SHADER];
    pub(crate) const FLAG_PHASE2: u32 = 16 => ["PHASE2" in PRESENTATION_SHADER];
    pub(crate) const FLAG_PROBE: u32 = 128 => ["PROBE" in PRESENTATION_SHADER];
    pub(crate) const STILL_FRAMES: u32 = 16 => ["STILL_FRAMES"];
    pub(crate) const MOVED_SLOT: u32 = 1 << 31 => ["MOVED_SLOT"];
    pub(crate) const LOD_LEVELS: usize = 3 => ["LOD_LEVELS" in PRESENTATION_SHADER, "IMPOSTOR" in PRESENTATION_SHADER];
    pub(crate) const LOD_STATE: u32 = 1 << 31 => ["LOD_STATE" in PRESENTATION_SHADER];
    pub(crate) const DRAW_ARGS_BYTES: u64 = 20 => ["size_of DrawArgs" in PRESENTATION_SHADER, "size_of DrawArgs" in SHADOW_SHADER];
    pub(crate) const COMPOSED_BYTES: u64 = 112 => ["size_of Composed" in PRESENTATION_SHADER];

    pub(crate) const PRESENTATION_GROUP: u32 = 256 => ["@shift_previous" in PRESENTATION_SHADER, "@cull" in PRESENTATION_SHADER, "@count_still" in PRESENTATION_SHADER, "@fade_levels" in PRESENTATION_SHADER, "@occlusion_test" in PRESENTATION_SHADER, "@probe_cull" in PRESENTATION_SHADER];
    pub(crate) const SCAN_GROUP: u32 = 64 => ["GROUP" in PRESENTATION_SHADER, "@sum_groups" in PRESENTATION_SHADER, "@scan_groups" in PRESENTATION_SHADER, "@scatter" in PRESENTATION_SHADER];
    pub(crate) const PARENT_GROUP: u32 = 64 => ["@compose_parents" in PRESENTATION_SHADER];
    pub(crate) const ARGS_GROUP: u32 = 64 => ["@batch_args" in PRESENTATION_SHADER];

    pub(crate) const CARD_VERTICES: u32 = 6 => ["IMPOSTOR_VERTICES" in PRESENTATION_SHADER];

    pub(crate) const SCATTER_LEVELS: u32 = 5 => ["SHOWN_CARD" in SCATTER_CULL_SHADER];
    pub(crate) const SHOWN_NONE: u32 = 6 => ["SHOWN_NONE" in SCATTER_CULL_SHADER];
    pub(crate) const FADED: u32 = 65535 => ["FADED" in SCATTER_CULL_SHADER];
    pub(crate) const STATE_ADOPT: u32 = u32::MAX => ["STATE_ADOPT" in SCATTER_CULL_SHADER];
    pub(crate) const DEAD: u32 = u32::MAX => ["DEAD"];
    pub(crate) const TERRAIN_INDEX: u32 = 0x1ff => ["TERRAIN_INDEX"];
    pub(crate) const LAYER_SHIFT: u32 = 9 => ["LAYER_SHIFT"];
    pub(crate) const NO_LAYER: u32 = 0x7f => ["NO_LAYER"];
    pub(crate) const NEAR_LEVEL_SHIFT: u32 = 16 => ["NEAR_LEVEL_SHIFT"];
    pub(crate) const RANK_SHIFT: u32 = 21 => ["RANK_SHIFT" in SCATTER_CULL_SHADER];

    pub(crate) const SHADOW_REGIONS: usize = 7 => ["REGIONS" in SHADOW_SHADER];
    pub(crate) const MOVING_MAP: usize = SHADOW_REGIONS => ["MOVING_MAP" in SHADOW_SHADER];
    pub(crate) const SKY_MAP: usize = SHADOW_REGIONS + 1 => ["SKY_MAP" in SHADOW_SHADER];
    pub(crate) const MOVING_PAIRS: u32 = 8 => ["MOVING_PAIRS" in SHADOW_SHADER];
    pub(crate) const MOVING_TILES: usize = MAX_MOVING_TILES => ["MOVING_TILES" in SHADOW_SHADER];
    pub(crate) const SHADOW_LEVELS: usize = MAX_LEVELS => ["SHADOW_LEVELS"];
    pub(crate) const SHADOW_TOTALS: usize = 4 => ["TOTALS" in SHADOW_SHADER];
    pub(crate) const SHADOW_GROUP: u32 = 256 => ["@select_casters" in SHADOW_SHADER];
    pub(crate) const SHADOW_MARKS_GROUP: u32 = 8 => ["@mark_shadow_tiles" in SHADOW_MARKS_SHADER];
    pub(crate) const STILLNESS: u32 = 1 => ["STILLNESS" in SHADOW_SHADER];
    pub(crate) const SKY_REFRESH: u32 = 2 => ["SKY_REFRESH" in SHADOW_SHADER];
    pub(crate) const SKY_DIRECTIONS: usize = 13 => ["SKY_DIRECTIONS" in POST_SHADER];
    pub(crate) const SKY_LAYERS: usize = 2 * SKY_DIRECTIONS => ["SKY_LAYERS" in POST_SHADER];

    pub(crate) const MIRRORED_BITS: u32 = 14 => ["MIRRORED_BITS"];
    pub(crate) const BODY_SOLID: u32 = 0 => ["BODY_SOLID"];
    pub(crate) const BODY_THIN: u32 = 1 => ["BODY_THIN"];
    pub(crate) const BODY_GROUND: u32 = 2 => ["BODY_GROUND"];
    pub(crate) const BODY_WATER: u32 = 3 => ["BODY_WATER"];

    pub(crate) const NO_PLACED: u32 = u32::MAX => ["NO_PLACED" in REFLECTIONS_SHADER];

    pub(crate) const GROUND_TEXELS: f32 = 2.0 => ["KEPT_GROUND_TEXELS" in SHADER];
    pub(crate) const SHADOW_APRON: u32 = 1 => ["SHADOW_APRON"];

    pub(crate) const NO_SLOT: u32 = u32::MAX => ["NO_SLOT" in LIGHT_SHADER];
    pub(crate) const GROUP_LIGHTS: u32 = 8 => ["LIGHTS" in LIGHT_SHADER];
    pub(crate) const LIGHT_BYTES: u64 = 64 => ["size_of Light"];
    pub(crate) const SHAPED_LIGHTS: u32 = 1 => ["SHAPED_LIGHTS" in LIGHT_SHADER, "SHAPED_LIGHTS" in SHADER];
    pub(crate) const KEPT_REGIONS: u32 = 2 => ["KEPT_REGIONS" in SHADER];
    pub(crate) const LIGHT_INTERIOR: u32 = 1 => ["LIGHT_INTERIOR" in SHADER];
    pub(crate) const LIGHT_EXTERIOR: u32 = 2 => ["LIGHT_EXTERIOR" in SHADER];

    pub(crate) const IMPOSTOR_FRAMES_SHIFT: u32 = 19 => ["IMPOSTOR_FRAMES_SHIFT"];
    pub(crate) const IMPOSTOR_HEMISPHERE: u32 = 1 << 16 => ["IMPOSTOR_HEMISPHERE"];
    pub(crate) const IMPOSTOR_MASKED: u32 = 1 << 17 => ["IMPOSTOR_MASKED"];
    pub(crate) const IMPOSTOR_FOLIAGE: u32 = 1 << 24 => ["IMPOSTOR_FOLIAGE"];
    pub(crate) const IMPOSTOR_KINDS: u32 = 4 => ["IMPOSTOR_KINDS" in REFLECTIONS_SHADER];
    pub(crate) const MASK_MIP_SCALE: f32 = 0.25 => ["MASK_MIP_SCALE"];

    pub(crate) const NOT_RESIDENT: u32 = u32::MAX => ["NOT_RESIDENT"];
    pub(crate) const ERROR_MIPS: u32 = engine_core::terrain_data::ERROR_MIPS => ["SURFACE_ERROR_MIPS", "PEAK_MIPS" in REFLECTIONS_SHADER];
    pub(crate) const TERRAIN_GRID: u32 = engine_core::terrain_data::PATCH_QUADS + 1 => ["TERRAIN_GRID"];
    pub(crate) const TERRAIN_GRID_VERTICES: u32 = TERRAIN_GRID * TERRAIN_GRID => ["TERRAIN_GRID_VERTICES"];

    pub(crate) const SURFACE_MESHLET_LEVEL: u32 = 2 => ["SURFACE_MESHLET_LEVEL"];
    pub(crate) const SURFACE_MESHLET_VERTICES: u32 = ((1 << SURFACE_MESHLET_LEVEL) + 1) * ((1 << SURFACE_MESHLET_LEVEL) + 2) / 2 => ["SURFACE_MESHLET_VERTICES"];
    pub(crate) const SURFACE_MESHLET_INDICES: u32 = 3 << (2 * SURFACE_MESHLET_LEVEL) => ["SURFACE_MESHLET_INDICES"];
    pub(crate) const SUBDIVISION_GROUP: u32 = 64 => ["SUBDIVISION_GROUP", "@subdivide_decide", "@subdivide_apply"];
    pub(crate) const SUBDIVISION_ARGS: u32 = 16 => ["SUBDIVISION_ARGS"];
    pub(crate) const SUBDIVISION_LEAVES: u32 = (size_of::<wgpu::util::DispatchIndirectArgs>() / 4) as u32 => ["SUBDIVISION_LEAVES"];
    pub(crate) const SUBDIVISION_DRAW: u32 = SUBDIVISION_LEAVES + 1 => ["SUBDIVISION_DRAW"];
    pub(crate) const SUBDIVISION_RESERVED: u32 = SUBDIVISION_DRAW + (size_of::<wgpu::util::DrawIndexedIndirectArgs>() / 4) as u32 => ["SUBDIVISION_RESERVED"];
    pub(crate) const SUBDIVISION_REFUSED: u32 = SUBDIVISION_RESERVED + 1 => ["SUBDIVISION_REFUSED"];
    pub(crate) const SUBDIVISION_COUNTED: u32 = 8 => ["SUBDIVISION_COUNTED"];

    pub(crate) const HORIZON_RECORD_VECTORS: u32 = 4 + engine_core::MAX_BOUNCE_LEVELS / 4 => ["HORIZON_RECORD_VECTORS"];
    pub(crate) const HORIZON_RECORDS: usize = 64 * HORIZON_RECORD_VECTORS as usize => ["HORIZON_RECORDS"];

    pub(crate) const BOUNCE_RECORD_WORDS: usize = engine_core::terrain_data::bounce::PROBE_BYTES / 4 => ["BOUNCE_RECORD_WORDS" in BOUNCE_SHADER];

    pub(crate) const MATERIAL_ANISOTROPY: u16 = 16 => ["DETAIL_ANISOTROPY"];

    pub(crate) const DEFORMATION_BLOCK: u32 = 16 => ["BLOCK" in DEFORMATION_SHADER];

    pub(crate) const PYRAMID_GROUP: u32 = 8 => ["@pyramid_first" in DEPTH_PYRAMID_SHADER, "@pyramid_down" in DEPTH_PYRAMID_SHADER, "@pyramid_first_nearest" in DEPTH_PYRAMID_SHADER, "@pyramid_down_nearest" in DEPTH_PYRAMID_SHADER];

    pub(crate) const WATER_BLOCK: u32 = TERRAIN_GRID + 2 => ["WATER_BLOCK"];
    pub(crate) const WATER_BLOCKS_SIDE: u32 = 16 => ["WATER_BLOCKS_SIDE"];
    pub(crate) const NO_WATER_BLOCK: u32 = u32::MAX => ["NO_WATER_BLOCK"];
    pub(crate) const NO_WATER_SURFACE: f32 = -1.0e30 => ["NO_WATER_SURFACE", "SIM_NO_SURFACE" in WATER_SIM_SHADER];
    pub(crate) const WATER_TILE_PATCHES: u32 = engine_core::terrain_data::PATCHES_PER_TILE => ["WATER_TILE_PATCHES"];
    pub(crate) const WATER_TILE_WORDS: u32 = WATER_TILE_PATCHES * WATER_TILE_PATCHES => ["WATER_TILE_WORDS"];
    pub(crate) const WATER_FLOW_RANGE: f32 = engine_core::terrain_data::WATER_FLOW_RANGE => ["WATER_FLOW_RANGE", "SIM_FLOW_RANGE" in WATER_SIM_SHADER];
    pub(crate) const WATER_SPAN_LEAST: f32 = engine_core::terrain_data::WATER_SPAN_LEAST => ["WATER_SPAN_LEAST"];
    pub(crate) const WATER_SPAN_MOST: f32 = engine_core::terrain_data::WATER_SPAN_MOST => ["WATER_SPAN_MOST"];
    pub(crate) const WATER_SHEAR_RANGE: f32 = engine_core::terrain_data::WATER_SHEAR_RANGE => ["WATER_SHEAR_RANGE", "SIM_SHEAR_RANGE" in WATER_SIM_SHADER];
    pub(crate) const WATER_GRADIENT_RANGE: f32 = 4.0 => ["WATER_GRADIENT_RANGE"];

    pub(crate) const WATER_SIM_GROUP: u32 = engine_core::terrain_data::BLOCK_SIDE => ["WATER_SIM_GROUP" in WATER_SIM_SHADER, "WATER_LIVE_BLOCK" in SHADER, "@sim_exchange" in WATER_SIM_SHADER, "@sim_waves" in WATER_SIM_SHADER, "@sim_seconds" in WATER_SIM_SHADER, "@sim_drain" in WATER_SIM_SHADER, "@sim_update" in WATER_SIM_SHADER, "@sim_touch" in WATER_SIM_SHADER, "@sim_burst" in WATER_SIM_SHADER, "@sim_push" in WATER_SIM_SHADER, "@sim_ended" in WATER_SIM_SHADER, "@sim_tally" in WATER_SIM_SHADER, "@sim_activity" in WATER_SIM_SHADER, "@sim_look" in WATER_SIM_SHADER, "@sim_drawn" in WATER_SIM_SHADER];
    pub(crate) const SIM_SLOTS: usize = 64 => ["SIM_SLOTS" in WATER_SIM_SHADER];
    pub(crate) const SIM_CELL_SUMS: u64 = 4 => ["SIM_CELL_SUMS" in WATER_SIM_SHADER];
    pub(crate) const SIM_BLOCK_LOOKS: u64 = 3 => ["SIM_BLOCK_LOOKS" in WATER_SIM_SHADER];
    pub(crate) const SIM_LIVE_LOOKS: u64 = 2 => ["SIM_LIVE_LOOKS" in WATER_SIM_SHADER, "WATER_LIVE_LOOKS"];
    pub(crate) const SIM_STEPPING: u32 = 1 => ["SIM_STEPPING" in WATER_SIM_SHADER];
    pub(crate) const SIM_LOCAL: u32 = 2 => ["SIM_LOCAL" in WATER_SIM_SHADER];
    pub(crate) const SIM_OPEN: u32 = 4 => ["SIM_OPEN" in WATER_SIM_SHADER];
    pub(crate) const SIM_SUMMING: u32 = 8 => ["SIM_SUMMING" in WATER_SIM_SHADER];
    pub(crate) const SIM_LOOKING: u32 = 16 => ["SIM_LOOKING" in WATER_SIM_SHADER];
    pub(crate) const SIM_SPARSE: u32 = 32 => ["SIM_SPARSE" in WATER_SIM_SHADER];
    pub(crate) const SIM_LIVE: u32 = 64 => ["SIM_LIVE" in WATER_SIM_SHADER];
    pub(crate) const SIM_BURSTING: u32 = 128 => ["SIM_BURSTING" in WATER_SIM_SHADER];
    pub(crate) const SIM_KEEPING: u32 = 256 => ["SIM_KEEPING" in WATER_SIM_SHADER];
    pub(crate) const SIM_RIM: u32 = engine_core::terrain_data::BLOCK_SIDE => ["SIM_RIM" in WATER_SIM_SHADER, "WATER_LIVE_RIM"];
    pub(crate) const WATER_SIM_SOURCE_GROUP: u32 = 64 => ["WATER_SIM_SOURCE_GROUP" in WATER_SIM_SHADER, "@sim_source" in WATER_SIM_SHADER];
    pub(crate) const WATER_SETUP_RING_GROUP: u32 = 64 => ["SETUP_RING_GROUP", "@setup_ring"];
    pub(crate) const SETUP_SOLID_ENTRIES: usize = 4 => ["SETUP_SOLID"];
    pub(crate) const SETUP_WHOLE: u32 = 0 => ["SETUP_WHOLE"];
    pub(crate) const SETUP_RISEN: u32 = 1 => ["SETUP_RISEN"];
    pub(crate) const SETUP_STANDS: u32 = 2 => ["SETUP_STANDS"];
    pub(crate) const SIM_STANDING: u32 = 1 << 31 => ["SIM_STANDING" in WATER_SIM_SHADER];
    pub(crate) const LIVE_NONE: u32 = u32::MAX => ["SETUP_NO_LIVE"];
    pub(crate) const LIVE_HEAD: u32 = 4 => ["SETUP_LIVE_HEAD"];
    pub(crate) const LIVE_CELL: u32 = 3 => ["SETUP_LIVE_CELL"];
    pub(crate) const LIVE_HELD: f32 = 1.0e30 => ["SETUP_LIVE_HELD"];

    pub(crate) const PARTICLE_GROUP: u32 = 256 => ["PARTICLE_GROUP", "@particles_age", "@particles_sum", "@particles_scan", "@particles_place", "@particles_splat"];
    pub(crate) const PARTICLE_SIDE: u32 = 8 => ["PARTICLE_SIDE", "@particles_gather", "@particles_currents", "@particles_births", "@particles_look", "@particles_down"];
    pub(crate) const PARTICLE_LATTICE: u32 = 8 => ["PARTICLE_LATTICE"];
    pub(crate) const PARTICLE_MARGIN: u32 = 2 => ["PARTICLE_MARGIN"];
    pub(crate) const PARTICLE_FOAM_LIVE: u32 = 0 => ["PARTICLE_FOAM_LIVE"];
    pub(crate) const PARTICLE_WAVES_LIVE: u32 = 1 => ["PARTICLE_WAVES_LIVE"];
    pub(crate) const PARTICLE_FOAM_BORN: u32 = 2 => ["PARTICLE_FOAM_BORN"];
    pub(crate) const PARTICLE_WAVES_BORN: u32 = 3 => ["PARTICLE_WAVES_BORN"];
    pub(crate) const PARTICLE_FOAM_DROPPED: u32 = 4 => ["PARTICLE_FOAM_DROPPED"];
    pub(crate) const PARTICLE_WAVES_DROPPED: u32 = 5 => ["PARTICLE_WAVES_DROPPED"];
    pub(crate) const PARTICLE_FOAM_BEFORE: u32 = 6 => ["PARTICLE_FOAM_BEFORE"];
    pub(crate) const PARTICLE_WAVES_BEFORE: u32 = 7 => ["PARTICLE_WAVES_BEFORE"];
    pub(crate) const PARTICLE_ARGS: u32 = 8 => ["PARTICLE_ARGS"];
    pub(crate) const PARTICLE_DRAW: u32 = 14 => ["PARTICLE_DRAW"];
    pub(crate) const PARTICLE_ITEMS: u32 = 18 => ["PARTICLE_ITEMS"];

    pub(crate) const BLADE_BYTES: u64 = 32 => ["size_of CoverBlade"];
    pub(crate) const COVER_LEAVES: u32 = 5 => ["COVER_LEAVES"];
    pub(crate) const COVER_LUSH_RATIO: f32 = 2.0 => ["COVER_LUSH_RATIO"];
    pub(crate) const STRAIGHT_JOINTS: u32 = 3 => ["STRAIGHT_JOINTS"];
    pub(crate) const STRAIGHT_FIRST_VERTEX: u32 = 1024 => ["STRAIGHT_FIRST_VERTEX"];

    pub(crate) const PYRAMID_EXTENT_BYTES: usize = 16 => ["size_of Extent" in DEPTH_PYRAMID_SHADER];

    pub(crate) const PALETTE_ENTRIES: usize = 16_384 / size_of::<PaletteEntry>() => ["TERRAIN_PALETTE_ENTRIES" in SHADER, "PALETTE_ENTRIES" in COVER_GROW_SHADER, "PALETTE_ENTRIES" in BOUNCE_SHADER, "PALETTE_ENTRIES" in SCATTER_SNAP_SHADER, "PALETTE_ENTRIES" in REFLECTIONS_SHADER];

    pub(crate) const SHADOW_CELLS: u32 = 128 => ["CLOUD_SHADOW_CELLS"];
    pub(crate) const LIGHT_TEXELS: u32 = 384 => ["LIGHT_TEXELS" in CLOUDS_SHADER];
    pub(crate) const LIGHT_HEIGHTS: u32 = 16 => ["LIGHT_HEIGHTS" in CLOUDS_SHADER];
    pub(crate) const EYE_LAYER: u32 = 0 => ["EYE_LAYER"];
    pub(crate) const MIRROR_LAYER: u32 = 1 => ["MIRROR_LAYER"];

    pub(crate) const NOON_LIGHT: f32 = 1.1 => ["NOON_LIGHT" in SKY_SHADER];
    pub(crate) const ADAPTATION_POWER: f32 = 0.75 => ["ADAPTATION_POWER" in SKY_SHADER];
    pub(crate) const ADAPTATION_MOST: f32 = 16.0 => ["ADAPTATION_MOST" in SKY_SHADER];

    pub(crate) const EXPOSURE_BINS: u32 = 128 => ["EXPOSURE_BINS" in EXPOSURE_SHADER];
    pub(crate) const EXPOSURE_GRID: u32 = 128 => ["EXPOSURE_GRID" in EXPOSURE_SHADER];
    pub(crate) const EXPOSURE_GROUP: u32 = 16 => ["@meter_exposure" in EXPOSURE_SHADER];

    pub(crate) const VARIANT_MAPS: u32 = 1 => ["VARIANT_MAPS" in SHADER];
    pub(crate) const VARIANT_NORMAL_MAP: u32 = 2 => ["VARIANT_NORMAL_MAP" in SHADER];
    pub(crate) const VARIANT_DOUBLE_SIDED: u32 = 8 => ["VARIANT_DOUBLE_SIDED" in SHADER];
    pub(crate) const VARIANT_METALLIC_ROUGHNESS_MAP: u32 = 128 => ["VARIANT_METALLIC_ROUGHNESS_MAP" in SHADER];
    pub(crate) const VARIANT_OCCLUSION_MAP: u32 = 256 => ["VARIANT_OCCLUSION_MAP" in SHADER];
    pub(crate) const VARIANT_EMISSIVE_MAP: u32 = 512 => ["VARIANT_EMISSIVE_MAP" in SHADER];
    pub(crate) const VARIANT_PACKED_OCCLUSION: u32 = 1024 => ["VARIANT_PACKED_OCCLUSION" in SHADER];
    pub(crate) const VARIANT_WORLD_MAPPING: u32 = 4096 => ["VARIANT_WORLD_MAPPING" in SHADER];
    pub(crate) const VARIANT_TOP_LAYER: u32 = 8192 => ["VARIANT_TOP_LAYER" in SHADER];
    pub(crate) const VARIANT_FOLIAGE: u32 = 65536 => ["VARIANT_FOLIAGE" in SHADER];

    pub(crate) const PROBE_ENTRY_WORDS: usize = 3 => ["PROBE_ENTRY_WORDS" in SHADER];

    pub(crate) const DEBUG_VIEW_KINDS: u32 = 1 => ["DEBUG_VIEW_KINDS" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_LEVELS: u32 = 2 => ["DEBUG_VIEW_LEVELS" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_CARDS: u32 = 3 => ["DEBUG_VIEW_CARDS" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_SHADOWS: u32 = 4 => ["DEBUG_VIEW_SHADOWS" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_CASTERS: u32 = 5 => ["DEBUG_VIEW_CASTERS" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_OVERDRAW: u32 = 6 => ["DEBUG_VIEW_OVERDRAW" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_NORMALS: u32 = 7 => ["DEBUG_VIEW_NORMALS" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_TILES: u32 = 8 => ["DEBUG_VIEW_TILES" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_MATERIALS: u32 = 9 => ["DEBUG_VIEW_MATERIALS" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_SKY: u32 = 10 => ["DEBUG_VIEW_SKY" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_DETAIL: u32 = 11 => ["DEBUG_VIEW_DETAIL" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_WATER_DEPTH: u32 = 12 => ["DEBUG_VIEW_WATER_DEPTH" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_WATER_SPEED: u32 = 13 => ["DEBUG_VIEW_WATER_SPEED" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_WATER_FROUDE: u32 = 14 => ["DEBUG_VIEW_WATER_FROUDE" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_WATER_TURBULENCE: u32 = 15 => ["DEBUG_VIEW_WATER_TURBULENCE" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_WATER_WHITE: u32 = 16 => ["DEBUG_VIEW_WATER_WHITE" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_WATER_REACH: u32 = 17 => ["DEBUG_VIEW_WATER_REACH" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_WATER_SPAN: u32 = 19 => ["DEBUG_VIEW_WATER_SPAN" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_WATER_EDDIES: u32 = 20 => ["DEBUG_VIEW_WATER_EDDIES" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_WATER_ROUGHNESS: u32 = 21 => ["DEBUG_VIEW_WATER_ROUGHNESS" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_GRAVITY: f32 = engine_core::terrain_data::hydraulics::GRAVITY => ["DEBUG_GRAVITY" in SHADER];
    pub(crate) const DEBUG_DRY: f32 = engine_core::terrain_data::shallow_water::DRY => ["DEBUG_DRY" in SHADER];
    pub(crate) const DEBUG_SPAN_STEPS: f32 = 254.0 => ["DEBUG_SPAN_STEPS"];
    pub(crate) const DEBUG_KIND_MESH: u32 = 1 => ["DEBUG_KIND_MESH"];
    pub(crate) const DEBUG_KIND_PLACED: u32 = 2 => ["DEBUG_KIND_PLACED"];
    pub(crate) const DEBUG_KIND_CARD: u32 = 3 => ["DEBUG_KIND_CARD"];
    pub(crate) const DEBUG_KIND_PLACED_CARD: u32 = 4 => ["DEBUG_KIND_PLACED_CARD"];
    pub(crate) const DEBUG_KIND_TERRAIN: u32 = 6 => ["DEBUG_KIND_TERRAIN"];
    pub(crate) const DEBUG_KIND_COVER: u32 = 7 => ["DEBUG_KIND_COVER"];
    pub(crate) const DEBUG_KIND_WATER: u32 = 8 => ["DEBUG_KIND_WATER"];
    pub(crate) const DEBUG_KIND_SEA: u32 = 9 => ["DEBUG_KIND_SEA"];
    pub(crate) const DEBUG_KIND_FALLS: u32 = 10 => ["DEBUG_KIND_FALLS"];
    pub(crate) const DEBUG_KIND_SKY: u32 = 11 => ["DEBUG_KIND_SKY"];
    pub(crate) const DEBUG_LEVEL_SHIFT: u32 = 4 => ["DEBUG_LEVEL_SHIFT"];
    pub(crate) const DEBUG_FADING: u32 = 0x100 => ["DEBUG_FADING"];
    pub(crate) const DEBUG_CASTS: u32 = 0x200 => ["DEBUG_CASTS"];
    pub(crate) const DEBUG_SOURCE_SHIFT: u32 = 16 => ["DEBUG_SOURCE_SHIFT"];
    pub(crate) const DEBUG_SUN_NONE: u32 = 0 => ["DEBUG_SUN_NONE"];
    pub(crate) const DEBUG_SUN_LEVEL: u32 = 1 => ["DEBUG_SUN_LEVEL"];
    pub(crate) const DEBUG_SUN_HORIZON: u32 = DEBUG_SUN_LEVEL + MAX_LEVELS as u32 => ["DEBUG_SUN_HORIZON"];
    pub(crate) const DEBUG_SUN_BEYOND: u32 = DEBUG_SUN_HORIZON + 1 => ["DEBUG_SUN_BEYOND"];
    pub(crate) const DEBUG_SHADOWED: u32 = 0x80 => ["DEBUG_SHADOWED"];
    pub(crate) const DEBUG_TILE_EDGE: u32 = 0x800 => ["DEBUG_TILE_EDGE"];
    pub(crate) const DEBUG_PATCH_EDGE: u32 = 0x1000 => ["DEBUG_PATCH_EDGE"];
    pub(crate) const DEBUG_PATCH_SHIFT: u32 = 24 => ["DEBUG_PATCH_SHIFT"];
    pub(crate) const DEBUG_NO_MATERIAL: u32 = 0xff => ["DEBUG_NO_MATERIAL"];

    pub(crate) const OCCUPANCY_CELLS: u32 = 16 => ["CELLS" in OCCUPANCY_SHADER];
    pub(crate) const OCCUPANCY_QUAD: u32 = 6 => ["QUAD" in OCCUPANCY_SHADER];

    pub(crate) const SCATTER_GROUP: u32 = 64 => ["@snap_scatter" in SCATTER_SNAP_SHADER, "@cull_scatter" in SCATTER_CULL_SHADER];
    pub(crate) const OCCUPANCY_GROUP: u32 = 64 => ["@select_covers" in OCCUPANCY_SHADER];
    pub(crate) const DEFORMATION_GROUP: u32 = 8 => ["@press" in DEFORMATION_SHADER, "@slopes" in DEFORMATION_SHADER, "@scroll" in DEFORMATION_SHADER, "@wipe" in DEFORMATION_SHADER, "@rebase" in DEFORMATION_SHADER];
    pub(crate) const CLOUD_GROUP: u32 = 8 => ["@march_view" in CLOUDS_SHADER, "@march_clouds" in CLOUDS_SHADER, "@shadow_clouds" in CLOUDS_SHADER, "@light_clouds" in CLOUDS_SHADER];
    pub(crate) const SKY_TABLE_GROUP: u32 = 8 => ["@build_sky_view" in SKY_SHADER];
    pub(crate) const CONTROL_GROUP: u32 = 8 => ["@tile_control" in TERRAIN_CONTROL_SHADER];
    pub(crate) const BOUNCE_GROUP: u32 = 8 => ["@bounce_ground" in BOUNCE_SHADER, "@bounce_down" in BOUNCE_SHADER, "@bounce_gather" in BOUNCE_SHADER];
    pub(crate) const NORMALS_GROUP: u32 = 16 => ["@tile_normals" in TERRAIN_NORMALS_SHADER];

    pub(crate) const WIND_FULL_SPEED: f32 = 12.0 => ["WIND_FULL_SPEED"];
    pub(crate) const WIND_SPAN: f32 = 10.0 => ["WIND_SPAN"];
    pub(crate) const WIND_TRUNK_FLUTTER: f32 = 0.12 => ["WIND_TRUNK_FLUTTER"];
    pub(crate) const WIND_TRUNK_RATE: f32 = 4.7 => ["WIND_TRUNK_RATE"];
    pub(crate) const WIND_BRANCH: f32 = 4.0 => ["WIND_BRANCH"];
    pub(crate) const WIND_BRANCH_RATE: f32 = 3.1 => ["WIND_BRANCH_RATE"];
    pub(crate) const WIND_BRANCH_RISE: f32 = 2.1 => ["WIND_BRANCH_RISE"];
    pub(crate) const WIND_FLUTTER: f32 = 0.2 => ["WIND_FLUTTER"];
    pub(crate) const WIND_FLUTTER_RATE: f32 = 13.0 => ["WIND_FLUTTER_RATE"];
    pub(crate) const WIND_SHAKE_WHOLE: f32 = 1.0 => ["WIND_SHAKE_WHOLE"];
    pub(crate) const WIND_SHAKE_STILL: f32 = 0.5 => ["WIND_SHAKE_STILL"];

    pub(crate) const CARD_FADING: u32 = 1 << 31 => ["FADING" in SCATTER_CULL_SHADER, "IMPOSTOR_FADING", "CARD_FADING" in PRESENTATION_SHADER];
    pub(crate) const CARD_FADING_OUT: u32 = 1 << 30 => ["FADING_OUT" in SCATTER_CULL_SHADER, "IMPOSTOR_FADING_OUT", "CARD_FADING_OUT" in PRESENTATION_SHADER];
    pub(crate) const CARD_OVER: u32 = 1 << 15 => ["IMPOSTOR_OVER", "CARD_OVER" in PRESENTATION_SHADER];
    pub(crate) const CLOUD_HORIZON: f32 = 0.05 => ["HORIZON" in CLOUDS_SHADER, "CLOUD_HORIZON"];
    pub(crate) const CLOUD_REACH: f32 = 60000.0 => ["REACH" in CLOUDS_SHADER, "CLOUD_PARALLAX_REACH"];
    pub(crate) const SURFACE_BEND: f32 = 7.0 => ["WATER_SURFACE_BEND"];
    pub(crate) const SURFACE_BEND_MOST: f32 = 1.0 => ["WATER_SURFACE_BEND_MOST"];
    pub(crate) const SURFACE_WIND_DRIFT: f32 = 0.03 => ["WATER_SURFACE_DRIFT"];
}

gpu_mirrors! {
    engine_core::table_data::sky::MULTIPLE_SIDE => ["SKY_MULTIPLE_SIZE"];
    engine_core::terrain_data::hydraulics::GRAVITY => ["SIM_GRAVITY" in WATER_SIM_SHADER, "WATER_GRAVITY"];
    engine_core::terrain_data::shallow_water::THETA => ["SIM_THETA" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::THIN => ["SIM_THIN" in WATER_SIM_SHADER, "SETUP_THIN"];
    engine_core::terrain_data::shallow_water::DRY => ["SIM_DRY" in WATER_SIM_SHADER, "SETUP_DRY"];
    engine_core::terrain_data::shallow_water::FASTEST => ["SIM_FASTEST" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::SOURCE_SPEED => ["SIM_SOURCE_SPEED" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::UNHELD => ["SIM_UNHELD" in WATER_SIM_SHADER, "SETUP_UNHELD"];
    engine_core::terrain_data::shallow_water::DENSITY => ["SIM_DENSITY" in WATER_SIM_SHADER, "WATER_DENSITY"];
    engine_core::terrain_data::shallow_water::FERGUSON_A1 => ["SIM_FERGUSON_A1" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::FERGUSON_A2 => ["SIM_FERGUSON_A2" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::SHALLOWEST => ["SIM_SHALLOWEST" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::DEEPEST => ["SIM_DEEPEST" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::STRICKLER => ["SIM_STRICKLER" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::CELL_MEANS => ["SIM_CELL_MEANS" in WATER_SIM_SHADER];
    engine_core::terrain_data::NO_BLOCK => ["SIM_NO_BLOCK" in WATER_SIM_SHADER];
    engine_core::terrain_data::BLOCK_GROUND => ["SIM_BLOCK_GROUND" in WATER_SIM_SHADER, "SETUP_GROUND"];
    engine_core::terrain_data::shallow_water::GROUND_D84 => ["SETUP_GROUND_D84"];
    engine_core::terrain_data::COVER_POINTS => ["SETUP_COVER_POINTS"];
    engine_core::terrain_data::shallow_water::WET_MOST => ["SIM_WET_MOST" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::FOAM_LIFE => ["WATER_FOAM_LIFE"];
    engine_core::terrain_data::shallow_water::FOAM_WET => ["WATER_FOAM_WET"];
    engine_core::terrain_data::shallow_water::FOAM_LOSS => ["WATER_FOAM_LOSS"];
    engine_core::terrain_data::shallow_water::FOAM_POWER => ["WATER_FOAM_POWER"];
    engine_core::terrain_data::shallow_water::COURANT => ["SIM_COURANT" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::LOCAL_SHARE => ["SIM_LOCAL_SHARE" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::SLOWEST => ["SIM_SLOWEST" in WATER_SIM_SHADER];
    engine_core::SIM_TOUCHES => ["SIM_TOUCHES" in WATER_SIM_SHADER];
    engine_core::terrain_data::TURBULENT_SPEED => ["WATER_TURBULENT_SPEED", "SIM_TURBULENT_SPEED" in WATER_SIM_SHADER];
    engine_core::terrain_data::TURBULENT_SWIRL => ["SIM_TURBULENT_SWIRL" in WATER_SIM_SHADER];
    engine_core::terrain_data::BREAK_ONSET => ["DEBUG_BREAK_ONSET" in DEBUG_VIEW_SHADER, "SIM_BREAK_ONSET" in WATER_SIM_SHADER, "WATER_BREAK_ONSET"];
    engine_core::terrain_data::BREAK_FULL => ["DEBUG_BREAK_FULL" in DEBUG_VIEW_SHADER, "SIM_BREAK_FULL" in WATER_SIM_SHADER, "WATER_BREAK_FULL"];
    engine_core::terrain_data::BREAK_THINNEST => ["WATER_BREAK_THINNEST", "SIM_BREAK_THINNEST" in WATER_SIM_SHADER];
    engine_core::NO_PARENT => ["NO_PARENT" in PRESENTATION_SHADER];
    engine_core::NODE_VALID => ["NODE_VALID" in PRESENTATION_SHADER];
    engine_core::NODE_BLEND => ["NODE_BLEND" in PRESENTATION_SHADER];
    engine_core::NODE_ACTIVE => ["NODE_ACTIVE" in PRESENTATION_SHADER];
    engine_core::NODE_PARENT_ACTIVE => ["NODE_PARENT_ACTIVE" in PRESENTATION_SHADER];
    engine_core::presentation_bounds::MAX_COEFFICIENT => ["MAX_COEFFICIENT" in PRESENTATION_SHADER];
    engine_core::presentation_bounds::ROUNDING_MARGIN => ["ROUNDING_MARGIN" in PRESENTATION_SHADER];
    engine_core::presentation_bounds::MAX_PRODUCT_BOUND => ["MAX_PRODUCT_BOUND" in PRESENTATION_SHADER];
    engine_core::presentation_bounds::COEFFICIENT_LIMIT => ["COEFFICIENT_LIMIT" in PRESENTATION_SHADER];
    engine_core::presentation_bounds::MIN_LINEAR_SCALE => ["MIN_LINEAR_SCALE" in PRESENTATION_SHADER];
    engine_core::presentation_bounds::COMMON_DETERMINANT_MARGIN => ["COMMON_DETERMINANT_MARGIN" in PRESENTATION_SHADER];
    engine_core::presentation_bounds::LINEAR_ROTATION_DOT_THRESHOLD => ["LINEAR_ROTATION_DOT" in PRESENTATION_SHADER];
    engine_core::RIM_START => ["RIM_START" in DEFORMATION_SHADER];
    engine_core::GUST_SLOW_SHARE => ["GUST_SLOW_SHARE"];
    engine_core::GUST_SLOW_RATE => ["GUST_SLOW_RATE"];
    engine_core::GUST_SLOW_TRAVEL => ["GUST_SLOW_TRAVEL"];
    engine_core::GUST_QUICK_SHARE => ["GUST_QUICK_SHARE"];
    engine_core::GUST_QUICK_RATE => ["GUST_QUICK_RATE"];
    engine_core::GUST_QUICK_TRAVEL => ["GUST_QUICK_TRAVEL"];
    engine_core::GUST_QUICK_ACROSS => ["GUST_QUICK_ACROSS"];
    engine_core::RIM_REACH => ["RIM_REACH" in DEFORMATION_SHADER];
    engine_core::terrain_data::lighting::DIRECTIONS => ["HORIZON_DIRECTIONS"];
    engine_core::terrain_data::lighting::HORIZON_LAYERS => ["HORIZON_LAYERS"];
    engine_core::terrain_data::bounce::BINS => ["BOUNCE_BINS" in BOUNCE_SHADER];
    engine_core::terrain_data::bounce::LAYERS => ["BOUNCE_PROBE_LAYERS" in BOUNCE_SHADER];
    engine_core::terrain_data::bounce::LOW_PROBE => ["BOUNCE_LOW"];
    engine_core::terrain_data::bounce::HIGH_PROBE => ["BOUNCE_HIGH"];
    engine_core::terrain_data::bounce::MAX_PROBES => ["BOUNCE_WINDOW_ROW"];
    engine_core::terrain_data::bounce::NEAREST => ["BOUNCE_NEAREST" in BOUNCE_SHADER];
    engine_core::terrain_data::bounce::DISTANCE_STEPS => ["BOUNCE_STEPS" in BOUNCE_SHADER];
    engine_core::terrain_data::TILE_STRIDE => ["TILE_STRIDE" in TERRAIN_CONTROL_SHADER, "TILE_STRIDE" in TERRAIN_NORMALS_SHADER];
    engine_core::terrain_data::TILE_QUADS => ["TILE_QUADS" in REFLECTIONS_SHADER];
    engine_core::terrain_data::ERROR_CELLS => ["PEAK_CELLS" in REFLECTIONS_SHADER];
    engine_core::terrain_data::RIVER_FAR => ["RIVER_FAR" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::RIVER_EDGE_STEP => ["RIVER_EDGE_STEP" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::RIVER_EDGE_ZERO => ["RIVER_EDGE_ZERO" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::RIVER_EDGE_FAR => ["RIVER_EDGE_FAR" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::PATH_FAR => ["PATH_FAR" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::MAX_STACK => ["MAX_STACK" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::MAX_MATERIALS => ["MAX_MATERIALS" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::PROGRAM => ["H_PROGRAM" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::BIOMES => ["H_BIOMES" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::READS => ["H_READS" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::HYDROLOGY => ["H_HYDROLOGY" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::PATHS => ["H_PATHS" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::FADE => ["H_FADE" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::WET_BIOMES => ["H_WET_BIOMES" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::EROSION => ["H_EROSION" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::BIOME_WORDS => ["BIOME_WORDS" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::CONSTANT => ["OP_CONSTANT" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::HEIGHT => ["OP_HEIGHT" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::GRADIENT => ["OP_GRADIENT" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::CURVATURE => ["OP_CURVATURE" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::FLOW => ["OP_FLOW" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::WETNESS => ["OP_WETNESS" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::LAKE => ["OP_LAKE" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::RIVER => ["OP_RIVER" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::PATH => ["OP_PATH" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::NOISE => ["OP_NOISE" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::MULTIPLY => ["OP_MULTIPLY" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::ADD => ["OP_ADD" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::MIN => ["OP_MIN" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::MAX => ["OP_MAX" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::INVERT => ["OP_INVERT" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::REMAP => ["OP_REMAP" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::GRAIN => ["OP_GRAIN" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::POWER => ["OP_POWER" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::FLOOD => ["OP_FLOOD" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::WORN => ["OP_WORN" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::DEPOSIT => ["OP_DEPOSIT" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::FACING => ["OP_FACING" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::END => ["OP_END" in TERRAIN_CONTROL_SHADER];
}
