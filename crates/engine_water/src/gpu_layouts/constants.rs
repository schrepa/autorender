engine_render::gpu_consts! {
    pub(crate) const NO_WATER_SURFACE: f32 = -1.0e30 => ["SIM_NO_SURFACE" in "water::sim"];
    pub(crate) const WATER_FLOW_RANGE: f32 = engine_terrain::WATER_FLOW_RANGE => ["SIM_FLOW_RANGE" in "water::sim"];
    pub(crate) const WATER_SHEAR_RANGE: f32 = engine_terrain::WATER_SHEAR_RANGE => ["SIM_SHEAR_RANGE" in "water::sim"];
    pub(crate) const WATER_GRADIENT_RANGE: f32 = 4.0 => ["WATER_GRADIENT_RANGE"];

    pub(crate) const WATER_SIM_GROUP: u32 = engine_terrain::BLOCK_SIDE => ["WATER_SIM_GROUP" in "water::sim", "WATER_LIVE_BLOCK" in "face::heightfield", "@sim_exchange" in "water::sim", "@sim_waves" in "water::sim", "@sim_seconds" in "water::sim", "@sim_drain" in "water::sim", "@sim_update" in "water::sim", "@sim_touch" in "water::sim", "@sim_burst" in "water::sim", "@sim_push" in "water::sim", "@sim_ended" in "water::sim", "@sim_tally" in "water::sim", "@sim_activity" in "water::sim", "@sim_look" in "water::sim", "@sim_drawn" in "water::sim"];
    pub(crate) const SIM_SLOTS: usize = 64 => ["SIM_SLOTS" in "water::sim"];
    pub(crate) const SIM_CELL_SUMS: u64 = 4 => ["SIM_CELL_SUMS" in "water::sim"];
    pub(crate) const SIM_BLOCK_LOOKS: u64 = 3 => ["SIM_BLOCK_LOOKS" in "water::sim"];
    pub(crate) const SIM_LIVE_LOOKS: u64 = 2 => ["SIM_LIVE_LOOKS" in "water::sim", "WATER_LIVE_LOOKS"];
    pub(crate) const SIM_STEPPING: u32 = 1 => ["SIM_STEPPING" in "water::sim"];
    pub(crate) const SIM_LOCAL: u32 = 2 => ["SIM_LOCAL" in "water::sim"];
    pub(crate) const SIM_OPEN: u32 = 4 => ["SIM_OPEN" in "water::sim"];
    pub(crate) const SIM_SUMMING: u32 = 8 => ["SIM_SUMMING" in "water::sim"];
    pub(crate) const SIM_LOOKING: u32 = 16 => ["SIM_LOOKING" in "water::sim"];
    pub(crate) const SIM_SPARSE: u32 = 32 => ["SIM_SPARSE" in "water::sim"];
    pub(crate) const SIM_LIVE: u32 = 64 => ["SIM_LIVE" in "water::sim"];
    pub(crate) const SIM_BURSTING: u32 = 128 => ["SIM_BURSTING" in "water::sim"];
    pub(crate) const SIM_KEEPING: u32 = 256 => ["SIM_KEEPING" in "water::sim"];
    pub(crate) const SIM_RIM: u32 = engine_terrain::BLOCK_SIDE => ["SIM_RIM" in "water::sim", "WATER_LIVE_RIM"];
    pub(crate) const WATER_SIM_SOURCE_GROUP: u32 = 64 => ["WATER_SIM_SOURCE_GROUP" in "water::sim", "@sim_source" in "water::sim"];
    pub(crate) const WATER_SETUP_RING_GROUP: u32 = 64 => ["SETUP_RING_GROUP", "@setup_ring"];
    pub(crate) const SETUP_SOLID_ENTRIES: usize = 4 => ["SETUP_SOLID"];
    pub(crate) const SETUP_WHOLE: u32 = 0 => ["SETUP_WHOLE"];
    pub(crate) const SETUP_RISEN: u32 = 1 => ["SETUP_RISEN"];
    pub(crate) const SETUP_STANDS: u32 = 2 => ["SETUP_STANDS"];
    pub(crate) const SIM_STANDING: u32 = 1 << 31 => ["SIM_STANDING" in "water::sim"];
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

    pub(crate) const DEBUG_GRAVITY: f32 = engine_terrain::hydraulics::GRAVITY => ["DEBUG_GRAVITY" in "water::debug"];
    pub(crate) const DEBUG_DRY: f32 = engine_terrain::shallow_water::DRY => ["DEBUG_DRY" in "water::debug"];

    pub(crate) const SURFACE_BEND: f32 = 7.0 => ["WATER_SURFACE_BEND"];
    pub(crate) const SURFACE_BEND_MOST: f32 = 1.0 => ["WATER_SURFACE_BEND_MOST"];
    pub(crate) const SURFACE_WIND_DRIFT: f32 = 0.03 => ["WATER_SURFACE_DRIFT"];
}

engine_render::gpu_mirrors! {
    engine_terrain::hydraulics::GRAVITY => ["SIM_GRAVITY" in "water::sim", "WATER_GRAVITY"];
    engine_terrain::shallow_water::THETA => ["SIM_THETA" in "water::sim"];
    engine_terrain::shallow_water::THIN => ["SIM_THIN" in "water::sim", "SETUP_THIN"];
    engine_terrain::shallow_water::DRY => ["SIM_DRY" in "water::sim", "SETUP_DRY"];
    engine_terrain::shallow_water::FASTEST => ["SIM_FASTEST" in "water::sim"];
    engine_terrain::shallow_water::SOURCE_SPEED => ["SIM_SOURCE_SPEED" in "water::sim"];
    engine_terrain::shallow_water::UNHELD => ["SIM_UNHELD" in "water::sim", "SETUP_UNHELD"];
    engine_terrain::shallow_water::DENSITY => ["SIM_DENSITY" in "water::sim", "WATER_DENSITY"];
    engine_terrain::shallow_water::FERGUSON_A1 => ["SIM_FERGUSON_A1" in "water::sim"];
    engine_terrain::shallow_water::FERGUSON_A2 => ["SIM_FERGUSON_A2" in "water::sim"];
    engine_terrain::shallow_water::SHALLOWEST => ["SIM_SHALLOWEST" in "water::sim"];
    engine_terrain::shallow_water::DEEPEST => ["SIM_DEEPEST" in "water::sim"];
    engine_terrain::shallow_water::STRICKLER => ["SIM_STRICKLER" in "water::sim"];
    engine_terrain::shallow_water::CELL_MEANS => ["SIM_CELL_MEANS" in "water::sim"];
    engine_terrain::NO_BLOCK => ["SIM_NO_BLOCK" in "water::sim"];
    engine_terrain::BLOCK_GROUND => ["SIM_BLOCK_GROUND" in "water::sim", "SETUP_GROUND"];
    engine_terrain::shallow_water::GROUND_D84 => ["SETUP_GROUND_D84"];
    engine_terrain::COVER_POINTS => ["SETUP_COVER_POINTS"];
    engine_terrain::shallow_water::WET_MOST => ["SIM_WET_MOST" in "water::sim"];
    engine_terrain::shallow_water::FOAM_LIFE => ["WATER_FOAM_LIFE"];
    engine_terrain::shallow_water::FOAM_WET => ["WATER_FOAM_WET"];
    engine_terrain::shallow_water::FOAM_LOSS => ["WATER_FOAM_LOSS"];
    engine_terrain::shallow_water::FOAM_POWER => ["WATER_FOAM_POWER"];
    engine_terrain::shallow_water::COURANT => ["SIM_COURANT" in "water::sim"];
    engine_terrain::shallow_water::LOCAL_SHARE => ["SIM_LOCAL_SHARE" in "water::sim"];
    engine_terrain::shallow_water::SLOWEST => ["SIM_SLOWEST" in "water::sim"];
    engine_terrain::SIM_TOUCHES => ["SIM_TOUCHES" in "water::sim"];
    engine_terrain::TURBULENT_SPEED => ["WATER_TURBULENT_SPEED", "SIM_TURBULENT_SPEED" in "water::sim"];
    engine_terrain::TURBULENT_SWIRL => ["SIM_TURBULENT_SWIRL" in "water::sim"];
    engine_terrain::BREAK_ONSET => ["SIM_BREAK_ONSET" in "water::sim", "WATER_BREAK_ONSET"];
    engine_terrain::BREAK_FULL => ["SIM_BREAK_FULL" in "water::sim", "WATER_BREAK_FULL"];
    engine_terrain::BREAK_THINNEST => ["WATER_BREAK_THINNEST", "SIM_BREAK_THINNEST" in "water::sim"];
}
