//! Terrain settings: level of detail, streaming, motion and the viewport
//! level of detail measures.

use super::*;

/// The size in pixels of the viewport frames render to, which terrain level
/// of detail measures pixels on; hosts set it each frame. Without it,
/// 1920x1080.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
        }
    }
}

settings! {
    /// Terrain settings.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneTerrainSettings")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct TerrainSettings {
        /// Whether terrains draw.
        pub enabled: bool = true => any,
        /// Pixels per sample before coarsening; smaller values draw more triangles.
        pub pixel_error: f32 = 6.0 => range(1.0, 64.0),
        /// Coarsest levels kept resident everywhere so the terrain always draws whole.
        pub pinned_levels: u32 = 2 => range(1, 8),
        /// Resident tile budget (about 270 KB CPU, twice that GPU, each); least important excess tiles drop.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_tiles: Option<u32> = None => budget,
        /// Work per clock check, in samples (67,000 per decoded tile, about 500,000 per generated tile); the whole frame budget without a clock.
        pub samples_per_frame: u32 = 24_576 => at_least(1),
        /// Milliseconds of terrain work per frame until the view first loads whole.
        pub loading_work_ms: f32 = 50.0 => range(0.1, 1000.0),
        /// Milliseconds of terrain work per frame after the view loads.
        pub work_ms: f32 = 4.0 => range(0.1, 1000.0),
        /// Level-0 collision radius around the camera and focus entities, meters.
        pub collision_radius: f32 = 192.0 => range(0.0, 4096.0),
        /// With shadows on, horizon maps cast long sun shadows and shadow maps cover nearby terrain. Off uses shadow maps alone.
        pub sun_shadows: bool = true => any,
        /// Texels along a horizon map's sides, 8 bytes each.
        pub shadow_resolution: u32 = 1024 => range(64, 4096),
        /// Shadow-edge height per meter to its casting slope; the sun's disc gives 0.0093.
        pub shadow_softness: f32 = 0.01 => range(0.0, 0.2),
        /// Scale ambient light by terrain sky visibility, baked once the finest pinned level is resident. Sheltered valleys take ground light instead.
        pub sky_visibility: bool = true => any,
        /// Texels along a sky visibility map's sides, 4 bytes each.
        pub sky_resolution: u32 = 1024 => range(64, 4096),
        /// Distance for material-height ray marching, meters; 0 disables relief.
        pub relief_distance: f32 = 24.0 => range(0.0, 200.0),
        /// The most steps a relief ray takes.
        pub relief_steps: u32 = 8 => range(2, 64),
        /// Stream generated surroundings; off ends each terrain at its edges.
        pub surroundings: bool = true => any,
        /// Optional detail reduction during fast camera motion; absent by default.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub motion_detail: Option<MotionDetail> = None => any,
        /// Tile cache bytes (about 600 KB per tile). Least recently used excess tiles drop; 0 caches none.
        pub cache_bytes: u64 = 256 << 20 => any,
        /// Tile bytes admitted per frame (about 540 KB each). Excess tiles wait; the first tile always fits.
        pub upload_bytes: u32 = 8 << 20 => at_least(1),
        /// Worker prefetch distance in level-range multiples while nothing is missing; none within the 2.5 keeping reach.
        pub idle_reach: f32 = 3.0 => range(0.0, 8.0),
        /// Most row bands per view tile on spare workers; 1 generates whole tiles.
        pub max_bands: u32 = 4 => range(1, 16),
        /// Compute tile material controls on GPU upload; rounding may cross rule thresholds. Gameplay queries evaluate rules. Off computes controls on CPU.
        pub gpu_materials: bool = true => any,
        /// Bake requested material pages for pixel lookups. Steep or finer ground and missing pages evaluate materials directly.
        pub virtual_texture: bool = true => any,
        /// Optional page budget (about 340 KB each), also device-limited. Unrequested pages yield space; excess requested pages drop to direct shading.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub virtual_texture_pages: Option<u32> = None => budget,
        /// Pages it bakes in a frame at most, about 0.1 ms each on an
        /// integrated GPU.
        pub virtual_texture_bakes: u32 = 16 => at_least(1),
    }
}

settings! {
    extra_rules;
    /// How terrain loads while the camera moves fast: the
    /// finest levels' tiles in its view wait, more of them the faster it
    /// moves, and tiles out of its view wait until it settles. The coarsest
    /// pinned levels, collision's tiles and those under focus entities never
    /// wait, and tiles resident stay.
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneTerrainMotionDetail")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct MotionDetail {
        /// Meters per second above which the camera moves fast.
        pub speed: f32 = 30.0 => range(0.0, 100_000.0),
        /// Meters per second at which the finest `levels` levels wait, at least
        /// `speed`; between, fewer wait, at least one.
        pub fastest: f32 = 300.0 => range(0.0, 100_000.0),
        /// Finest levels whose tiles wait at `fastest`.
        pub levels: u32 = 2 => range(0, 8),
        /// Seconds after the camera slows below `speed` before tiles out of its
        /// view load.
        pub settle: f32 = 0.2 => range(0.0, 10.0),
    }
}

impl ExtraRules for MotionDetail {
    /// The fastest speed is at least the fast one.
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.fastest >= self.speed {
            Ok(())
        } else {
            Err(InvalidSetting::new("fastest", "must be at least speed"))
        }
    }

    fn sanitize_extra(self) -> Self {
        Self {
            fastest: crate::math::max(self.fastest, self.speed),
            ..self
        }
    }
}

impl MotionDetail {
    /// The finest levels whose tiles wait at `speed` meters per second.
    pub fn waiting(&self, speed: f32) -> u32 {
        if speed <= self.speed || speed.is_nan() || self.levels == 0 {
            return 0;
        }
        let share = if self.fastest > self.speed {
            crate::math::clamp((speed - self.speed) / (self.fastest - self.speed), 0.0, 1.0)
        } else {
            1.0
        };
        ((share * self.levels as f32).ceil() as u32).clamp(1, self.levels)
    }
}

impl TerrainSettings {
    /// What a world without these settings reads as: no terrain draws.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
