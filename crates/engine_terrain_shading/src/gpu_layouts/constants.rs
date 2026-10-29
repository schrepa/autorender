engine_render::gpu_consts! {
    pub(crate) const GROUND_TEXELS: f32 = engine_faces::shadows::GROUND_TEXELS => ["KEPT_GROUND_TEXELS" in "terrain::draw"];

    pub(crate) const NOT_RESIDENT: u32 = u32::MAX => ["NOT_RESIDENT"];
    pub(crate) const ERROR_MIPS: u32 = engine_terrain::ERROR_MIPS => ["SURFACE_ERROR_MIPS", "PEAK_MIPS" in "post::reflections"];
    pub(crate) const TERRAIN_GRID: u32 = engine_terrain::PATCH_QUADS + 1 => ["TERRAIN_GRID"];
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

    pub(crate) const HORIZON_RECORD_VECTORS: u32 = 4 + engine_terrain::MAX_BOUNCE_LEVELS / 4 => ["HORIZON_RECORD_VECTORS"];
    pub(crate) const HORIZON_RECORDS: usize = 64 * HORIZON_RECORD_VECTORS as usize => ["HORIZON_RECORDS"];
    pub(crate) const HORIZON_RECORD_SLOTS: usize = HORIZON_RECORDS + 1 => ["HORIZON_RECORD_SLOTS"];

    pub(crate) const BOUNCE_RECORD_WORDS: usize = engine_terrain::bounce::PROBE_BYTES / 4 => ["BOUNCE_RECORD_WORDS" in "terrain::bounce_maps"];

    pub(crate) const WATER_BLOCK: u32 = TERRAIN_GRID + 2 => ["WATER_BLOCK"];
    pub(crate) const WATER_BLOCKS_SIDE: u32 = 16 => ["WATER_BLOCKS_SIDE"];
    pub(crate) const NO_WATER_BLOCK: u32 = u32::MAX => ["NO_WATER_BLOCK"];
    pub(crate) const NO_WATER_SURFACE: f32 = -1.0e30 => ["NO_WATER_SURFACE", "SIM_NO_SURFACE" in "water::sim"];
    pub(crate) const WATER_TILE_PATCHES: u32 = engine_terrain::PATCHES_PER_TILE => ["WATER_TILE_PATCHES"];
    pub(crate) const WATER_TILE_WORDS: u32 = WATER_TILE_PATCHES * WATER_TILE_PATCHES => ["WATER_TILE_WORDS"];

    pub(crate) const PALETTE_ENTRIES: usize = engine_faces::heightfield::PALETTE_ENTRIES => ["TERRAIN_PALETTE_ENTRIES" in "terrain::draw", "PALETTE_ENTRIES" in "cover::grow", "PALETTE_ENTRIES" in "terrain::bounce_maps", "PALETTE_ENTRIES" in "scatter::snap", "PALETTE_ENTRIES" in "post::reflections"];

    pub(crate) const CONTROL_GROUP: u32 = 8 => ["@tile_control" in "terrain::control"];
    pub(crate) const BOUNCE_GROUP: u32 = 8 => ["@bounce_ground" in "terrain::bounce_maps", "@bounce_down" in "terrain::bounce_maps", "@bounce_gather" in "terrain::bounce_maps"];
    pub(crate) const NORMALS_GROUP: u32 = 16 => ["@tile_normals" in "terrain::normals"];
}

engine_render::gpu_mirrors! {
    engine_terrain::BLOCK_SIDE => ["WATER_LIVE_BLOCK" in "face::heightfield"];
    engine_terrain::lighting::DIRECTIONS => ["HORIZON_DIRECTIONS"];
    engine_terrain::lighting::HORIZON_LAYERS => ["HORIZON_LAYERS"];
    engine_terrain::bounce::BINS => ["BOUNCE_BINS" in "terrain::bounce_maps"];
    engine_terrain::bounce::LAYERS => ["BOUNCE_PROBE_LAYERS" in "terrain::bounce_maps"];
    engine_terrain::bounce::LOW_PROBE => ["BOUNCE_LOW"];
    engine_terrain::bounce::HIGH_PROBE => ["BOUNCE_HIGH"];
    engine_terrain::bounce::MAX_PROBES => ["BOUNCE_WINDOW_ROW"];
    engine_terrain::bounce::NEAREST => ["BOUNCE_NEAREST" in "terrain::bounce_maps"];
    engine_terrain::bounce::DISTANCE_STEPS => ["BOUNCE_STEPS" in "terrain::bounce_maps"];
    engine_terrain::TILE_STRIDE => ["TILE_STRIDE" in "terrain::tile", "TILE_STRIDE" in "terrain::tile"];
    engine_terrain::RIVER_FAR => ["RIVER_FAR" in "terrain::control"];
    engine_terrain::RIVER_EDGE_STEP => ["RIVER_EDGE_STEP" in "terrain::control"];
    engine_terrain::RIVER_EDGE_ZERO => ["RIVER_EDGE_ZERO" in "terrain::control"];
    engine_terrain::RIVER_EDGE_FAR => ["RIVER_EDGE_FAR" in "terrain::control"];
    engine_terrain::PATH_FAR => ["PATH_FAR" in "terrain::control"];
    engine_terrain::gpu::MAX_STACK => ["MAX_STACK" in "terrain::control"];
    engine_terrain::gpu::MAX_MATERIALS => ["MAX_MATERIALS" in "terrain::control"];
    engine_terrain::gpu::header::PROGRAM => ["H_PROGRAM" in "terrain::control"];
    engine_terrain::gpu::header::BIOMES => ["H_BIOMES" in "terrain::control"];
    engine_terrain::gpu::header::READS => ["H_READS" in "terrain::control"];
    engine_terrain::gpu::header::HYDROLOGY => ["H_HYDROLOGY" in "terrain::control"];
    engine_terrain::gpu::header::PATHS => ["H_PATHS" in "terrain::control"];
    engine_terrain::gpu::header::FADE => ["H_FADE" in "terrain::control"];
    engine_terrain::gpu::header::WET_BIOMES => ["H_WET_BIOMES" in "terrain::control"];
    engine_terrain::gpu::header::EROSION => ["H_EROSION" in "terrain::control"];
    engine_terrain::gpu::BIOME_WORDS => ["BIOME_WORDS" in "terrain::control"];
    engine_terrain::gpu::op::CONSTANT => ["OP_CONSTANT" in "terrain::control"];
    engine_terrain::gpu::op::HEIGHT => ["OP_HEIGHT" in "terrain::control"];
    engine_terrain::gpu::op::GRADIENT => ["OP_GRADIENT" in "terrain::control"];
    engine_terrain::gpu::op::CURVATURE => ["OP_CURVATURE" in "terrain::control"];
    engine_terrain::gpu::op::FLOW => ["OP_FLOW" in "terrain::control"];
    engine_terrain::gpu::op::WETNESS => ["OP_WETNESS" in "terrain::control"];
    engine_terrain::gpu::op::LAKE => ["OP_LAKE" in "terrain::control"];
    engine_terrain::gpu::op::RIVER => ["OP_RIVER" in "terrain::control"];
    engine_terrain::gpu::op::PATH => ["OP_PATH" in "terrain::control"];
    engine_terrain::gpu::op::NOISE => ["OP_NOISE" in "terrain::control"];
    engine_terrain::gpu::op::MULTIPLY => ["OP_MULTIPLY" in "terrain::control"];
    engine_terrain::gpu::op::ADD => ["OP_ADD" in "terrain::control"];
    engine_terrain::gpu::op::MIN => ["OP_MIN" in "terrain::control"];
    engine_terrain::gpu::op::MAX => ["OP_MAX" in "terrain::control"];
    engine_terrain::gpu::op::INVERT => ["OP_INVERT" in "terrain::control"];
    engine_terrain::gpu::op::REMAP => ["OP_REMAP" in "terrain::control"];
    engine_terrain::gpu::op::GRAIN => ["OP_GRAIN" in "terrain::control"];
    engine_terrain::gpu::op::POWER => ["OP_POWER" in "terrain::control"];
    engine_terrain::gpu::op::FLOOD => ["OP_FLOOD" in "terrain::control"];
    engine_terrain::gpu::op::WORN => ["OP_WORN" in "terrain::control"];
    engine_terrain::gpu::op::DEPOSIT => ["OP_DEPOSIT" in "terrain::control"];
    engine_terrain::gpu::op::FACING => ["OP_FACING" in "terrain::control"];
    engine_terrain::gpu::op::VALUE => ["OP_VALUE" in "terrain::control"];
    engine_terrain::gpu::op::END => ["OP_END" in "terrain::control"];
}
