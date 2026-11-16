use engine_core::{Cut, settings, stats};

use crate::gpu::{LookUniform, SUN_LEVELS};

pub const MAX_BOUNCE_LEVELS: u32 = 8;

settings! {
    #[section("terrain_shading")]
    #[switch(on)]
    pub struct TerrainShading {
        pub subdivision_error: f32 = 2.0 => range(0.25, 64.0),
        pub subdivision_unseen: f32 = 8.0 => range(1.0, 64.0),
        pub subdivision_depth: u32 = 27 => range(10, 29),
        pub subdivision_leaves: u32 = 131_072 => range(1024, 4_000_000),
        pub relief_steps: u32 = 8 => range(2, 64),
        pub relief_pixels: f32 = 0.5 => range(0.5, 4096.0),
        pub relief_anisotropy: f32 = 16.0 => range(1.0, 16.0),
        pub relief_refine_pixels: f32 = 0.0 => range(0.0, 64.0),
        pub relief_reach: f32 = 16.0 => range(0.0, 16.0),
        pub maps_pixels: f32 = 0.5 => range(0.5, 4096.0),
        pub albedo_pixels: f32 = 0.5 => range(0.5, 4096.0),
        pub far_plane_cut: f32 = 0.0 => range(0.0, 0.3),
        pub rung_anisotropy: f32 = 16.0 => range(1.0, 16.0),
        pub rung_detail: f32 = 1.0 => range(0.125, 64.0),
        pub site_lush: [f32; 3] = [1.0, 1.0, 1.0] => range(0.0, 4.0),
        pub site_bare: [f32; 3] = [1.0, 1.0, 1.0] => range(0.0, 4.0),
        pub strata_dip: f32 = 0.0 => range(0.0, 89.0),
        pub strata_heading: f32 = 0.0 => range(-360.0, 360.0),
        pub strata_run: f32 = 100.0 => range(0.1, 100_000.0),
        pub strata_warp: f32 = 0.0 => range(0.0, 1000.0),
        pub strata_warp_size: f32 = 100.0 => range(1.0, 100_000.0),
        pub strata_slope_start: f32 = 0.0 => range(0.0, 90.0),
        pub strata_slope_full: f32 = 0.0 => range(0.0, 90.0),
        pub height_blend: f32 = 1.0 => range(0.001, 1.0),
        pub tiling_hexes: f32 = 3.5 => range(0.25, 64.0),
        pub tiling_blend: f32 = 0.2 => range(0.01, 1.0),
        pub tiling_turn: f32 = 0.0 => range(0.0, 1.0),
        pub plane_sharpness: f32 = 1.0 => range(1.0, 64.0),
        pub plane_cut: f32 = 0.0 => range(0.0, 0.3),
        pub edge_shade: f32 = 0.0 => range(0.0, 1.0),
        pub edge_soil: f32 = 0.0 => range(0.0, 1.0),
        pub edge_width: f32 = 1.0 => range(0.001, 16.0),
        pub wet_porous_from: f32 = 0.0 => range(0.0, 1.0),
        pub wet_porous_range: f32 = 1.0 => range(0.01, 1.0),
        pub wet_darkening: f32 = 1.0 => range(0.0, 1.0),
        pub wet_gloss: f32 = 1.0 => range(0.0, 1.0),
        pub wet_spacing: f32 = 1.0e6 => range(0.01, 1.0e6),
        pub shore_fringe: f32 = 0.001 => range(0.001, 100.0),
        pub arrival_seconds: f32 = 0.001 => range(0.001, 60.0),
        pub sun_shadows: bool = true => any,
        pub shadow_resolution: u32 = 512 => range(64, 4096),
        pub sun_levels: u32 = SUN_LEVELS => range(1, SUN_LEVELS),
        pub sun_texel: f32 = 1.0 => range(0.25, 64.0),
        pub sun_rows: u32 = 256 => multiple(8, 8, 32768),
        pub sun_march_steps: u32 = 2048 => range(256, 65536),
        pub sun_sample_spread: f32 = 1.0 => range(0.0, 16.0),
        pub sun_march_slack: f32 = 0.25 => range(0.0, 4.0),
        pub sun_tall: f32 = 32.0 => range(1.0, 1000.0),
        pub sun_slide_seconds: f32 = 0.25 => range(0.0, 10.0),
        pub sky_visibility: bool = true => any,
        pub sky_resolution: u32 = 1024 => range(64, 4096),
        pub bounce: bool = true => any,
        pub bounce_spacing: f32 = 16.0 => range(1.0, 1024.0),
        pub bounce_columns: u32 = 64 => multiple(2, 8, 1024),
        pub bounce_levels: u32 = MAX_BOUNCE_LEVELS => range(1, MAX_BOUNCE_LEVELS),
        pub macro_resolution: u32 = 2048 => range(64, 4096),
        pub tile_feed: u32 = 4096 => at_least(1),
    }
}

impl TerrainShading {
    pub fn lit(&self) -> bool {
        self.sky_visibility || self.sun_shadows
    }

    pub fn bounces(&self) -> bool {
        self.sun_shadows && self.bounce
    }

    pub fn look(&self) -> LookUniform {
        let (dip_sin, dip_cos) = engine_math::sin_cos(self.strata_dip.to_radians());
        let (heading_sin, heading_cos) = engine_math::sin_cos(self.strata_heading.to_radians());
        let start = engine_math::sin(self.strata_slope_start.to_radians());
        let full = engine_math::sin(self.strata_slope_full.to_radians());
        LookUniform {
            reach: [
                self.relief_pixels,
                self.maps_pixels,
                self.albedo_pixels,
                self.rung_detail,
            ],
            far: [
                0.0,
                self.far_plane_cut,
                self.edge_shade,
                self.edge_soil,
            ],
            march: [
                self.relief_anisotropy,
                self.relief_refine_pixels,
                self.rung_anisotropy,
                self.relief_reach,
            ],
            lush: [self.site_lush[0], self.site_lush[1], self.site_lush[2], 0.0],
            bedding: [
                dip_sin * heading_cos,
                dip_cos,
                dip_sin * heading_sin,
                1.0 / self.strata_run,
            ],
            strike: [
                heading_sin,
                -heading_cos,
                self.strata_warp,
                1.0 / self.strata_warp_size,
            ],
            slopes: [
                start,
                engine_math::max(full, start + f32::EPSILON),
                1.0 / self.edge_width,
                0.0,
            ],
            tiling: [
                self.tiling_hexes,
                1.0 / self.tiling_blend,
                self.tiling_turn,
                self.height_blend,
            ],
            planes: [
                self.plane_sharpness,
                self.plane_cut,
                0.0,
                0.0,
            ],
            wet: [
                self.wet_porous_from,
                1.0 / self.wet_porous_range,
                self.wet_darkening,
                self.wet_gloss,
            ],
            shore: [
                self.shore_fringe,
                self.wet_spacing,
                self.arrival_seconds,
                0.0,
            ],
            bare: [self.site_bare[0], self.site_bare[1], self.site_bare[2], 0.0],
        }
    }
}

stats! {
    #[section("terrain_shading_stats")]
    pub struct TerrainShadingStats {
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
        pub trace_pool: Cut,
        pub trace_turns: Cut,
        pub sun_walks: Cut,
        pub sun_lost: u32,
        pub failed_jobs: u64,
        pub latest_job_failure: Option<String>,
    }
}
