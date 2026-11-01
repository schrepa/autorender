use engine_core::{Cut, settings, stats};

pub const MAX_BOUNCE_LEVELS: u32 = 8;

settings! {
    #[section("terrain_shading")]
    pub struct TerrainShading {
        pub enabled: bool = true => any,
        pub subdivision_error: f32 = 2.0 => range(0.25, 64.0),
        pub subdivision_depth: u32 = 27 => range(10, 29),
        pub subdivision_leaves: u32 = 131_072 => range(1024, 4_000_000),
        pub relief_steps: u32 = 8 => range(2, 64),
        pub sun_shadows: bool = true => any,
        pub shadow_resolution: u32 = 512 => range(64, 4096),
        pub shadow_softness: f32 = 0.01 => range(0.0, 0.2),
        pub sky_visibility: bool = true => any,
        pub sky_resolution: u32 = 1024 => range(64, 4096),
        pub bounce: bool = true => any,
        pub bounce_spacing: f32 = 16.0 => range(1.0, 1024.0),
        pub bounce_columns: u32 = 64 => multiple(2, 8, 1024),
        pub bounce_levels: u32 = MAX_BOUNCE_LEVELS => range(1, MAX_BOUNCE_LEVELS),
        pub macro_resolution: u32 = 2048 => range(64, 4096),
    }
}

impl TerrainShading {
    pub fn lit(&self) -> bool {
        self.sky_visibility || self.sun_shadows
    }

    pub fn bounces(&self) -> bool {
        self.sun_shadows && self.bounce
    }
}

stats! {
    #[section("terrain_shading_stats")]
    pub struct TerrainShadingStats {
        #[pending] pub pending: u32,
        pub triangles: u32,
        pub leaves: Cut,
        pub shallow_subdivisions: u32,
        pub shadow_patches: u32,
        pub tiles: u32,
        pub tiles_dropped: Cut,
        pub gpu_controls: u32,
        pub horizons: u32,
        pub sky_maps: u32,
        pub lighting_uploads: u32,
        pub bounce_relit: u32,
        pub bounce_probes: u32,
        pub bounce_levels: u32,
        pub bounce_dropped: u32,
        pub macro_maps: u32,
        pub dropped_materials: u32,
        pub shaded_tiles: u32,
        pub shading_records: u32,
        pub lit: u32,
        pub lighting: u32,
        pub bounced: u32,
        pub bouncing: u32,
        pub colored: u32,
        pub coloring: u32,
        pub surface_slots: Cut,
        pub failed_jobs: u64,
        pub latest_job_failure: Option<String>,
    }
}
