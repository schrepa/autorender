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

/// Most levels of probes a terrain's bounce light relights
/// ([`TerrainSettings::bounce_levels`]).
pub const MAX_BOUNCE_LEVELS: u32 = 8;

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
        /// Pixels a level's samples span before the next coarser level takes
        /// over; smaller draws more triangles.
        pub pixel_error: f32 = 6.0 => range(1.0, 64.0),
        /// The coarsest levels kept resident over the whole terrain, so it
        /// always draws whole.
        pub pinned_levels: u32 = 2 => range(1, 8),
        /// An optional budget of resident tiles (about 270 KB each, twice that
        /// on the GPU); the least important beyond it drop.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_tiles: Option<u32> = None => budget,
        /// Work decoded between checks of the frame's work time, in samples'
        /// worth (67,000 a file's tile, the rivers' water on a tile about as
        /// much); a frame's whole work where hosts give terrain no clock.
        pub samples_per_frame: u32 = 24_576 => at_least(1),
        /// Milliseconds a frame works on terrain until the camera's view first
        /// loads whole: loading takes large slices of frames.
        pub loading_work_ms: f32 = 50.0 => range(0.1, 1000.0),
        /// Milliseconds a frame works on terrain after: play takes small
        /// slices.
        pub work_ms: f32 = 4.0 => range(0.1, 1000.0),
        /// Meters around the camera and focus entities within which level-0
        /// tiles load for collision.
        pub collision_radius: f32 = 192.0 => range(0.0, 4096.0),
        /// Long sun shadows through each terrain's horizon map,
        /// with shadows on: the near shadow maps then draw terrain near their
        /// receivers, the kept shadow levels hold its ground as far as they
        /// reach, and the horizon map shadows the rest. Off, terrain casts
        /// into the shadow maps alone. The map holds the
        /// horizons in sixteen directions, cooked with the terrain's lighting
        /// from its finest pinned level, so the sun moves freely.
        pub sun_shadows: bool = true => any,
        /// Texels along a horizon map's sides, 72 bytes each on the GPU.
        pub shadow_resolution: u32 = 512 => range(64, 4096),
        /// Shadow edge height per meter to the slope casting it (the sun's disc
        /// gives 0.0093).
        pub shadow_softness: f32 = 0.01 => range(0.0, 0.2),
        /// Ambient light on terrain scaled by how much sky its ground sees,
        /// and taken from the sky it sees: each terrain's sky visibility map,
        /// cooked with its lighting from its finest pinned level, where
        /// valleys and hollows see less sky and take the light of the ground
        /// around them instead, as bounce light gathers it.
        pub sky_visibility: bool = true => any,
        /// Texels along a sky visibility map's sides, 4 bytes each.
        pub sky_resolution: u32 = 1024 => range(64, 4096),
        /// Bounce light, with long sun shadows: probes over each terrain
        /// whose transfer is cooked from its finest pinned level and its
        /// surface's rules, nothing of the sun or sky in it, gather every
        /// frame the light the ground around sends them as the sun, its
        /// shadows and the sky light it now, so slopes the sun lights light
        /// the valleys beside them through the day. They relight in nested
        /// levels around the detail eye, the coarsest over the whole terrain,
        /// so every place takes the same light wherever the eye stands.
        /// Terrain takes its light from the ground around it there, and what
        /// stands on terrain that light and the share of the sky the ground
        /// leaves it. Surroundings, and all ground with bounce light off,
        /// take none.
        pub bounce: bool = true => any,
        /// Meters between the finest level's probe columns at most: a terrain
        /// takes the power of two of them along its longer side within this
        /// spacing, at most 512 (larger terrains spread theirs wider). Each
        /// block of two by two columns holds one probe where the ground around
        /// stays open and four where it rises around them, in two layers:
        /// about 100 bytes a probe in the record, 7 to 27 MB a terrain of 512
        /// columns.
        pub bounce_spacing: f32 = 16.0 => range(1.0, 1024.0),
        /// Probe columns along each side of every level of probes relit
        /// around the detail eye, an even count: the finest level at the
        /// terrain's own columns, each coarser one twice as far apart and
        /// reaching twice as far, the coarsest holding the whole terrain. A
        /// level shades within half its side, less two and a half columns, of
        /// the eye. Each level's window takes a square of these texels of the
        /// horizon maps (`shadow_resolution` a side); a terrain whose levels
        /// they cannot hold takes no bounce light.
        pub bounce_columns: u32 = 64 => multiple(2, 8, 1024),
        /// Levels of probes a terrain relights at most: as many as its
        /// coarsest needs to hold the whole terrain, whose columns grow past
        /// `bounce_columns` where these leave it short.
        pub bounce_levels: u32 = MAX_BOUNCE_LEVELS => range(1, MAX_BOUNCE_LEVELS),
        /// Texels along the sides of each terrain's macro color map (the tint
        /// and growth its bakes give its ground over its whole extent),
        /// rounded up to a multiple of four: a byte each with a third more for
        /// their mips on devices sampling compressed textures, four elsewhere.
        pub macro_resolution: u32 = 2048 => range(64, 4096),
        /// The most steps a ray takes through the ground's relief. Relief
        /// shows by its size on screen, its view rays marching through its
        /// materials' heights: whole while its depth moves the ground two
        /// pixels or more, fading out by half a pixel, with about a step per
        /// pixel it moves.
        pub relief_steps: u32 = 8 => range(2, 64),
        /// Ground generated around terrains with surroundings
        /// streams and draws; off, every terrain ends at its edges.
        pub surroundings: bool = true => any,
        /// Detail lowered while the camera moves fast: off by
        /// default, as it lowers detail in motion on purpose.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub motion_detail: Option<MotionDetail> = None => any,
        /// Bytes of tiles kept in memory after the view leaves them,
        /// about 600 KB each, to return at once when it comes back (and
        /// those places games name load into): the least recently used
        /// beyond drop, and 0 keeps none.
        pub cache_bytes: u64 = 256 << 20 => any,
        /// Bytes of tiles becoming resident in a frame, which the GPU takes in
        /// that frame, about 540 KB a tile: tiles beyond wait for the next,
        /// though each frame's first arrives whatever its size.
        pub upload_bytes: u32 = 8 << 20 => at_least(1),
        /// Level-0 tiles' and surroundings' tiles' surface materials from
        /// the GPU: those tiles generate without control values, which a
        /// compute pass gives each as it uploads, the rules' own but where GPU
        /// rounding crosses a threshold; gameplay reads the rules at the
        /// points it asks. Off, the CPU gives them. Either way a terrain's
        /// coarser tiles take theirs from their shading, level 0's averaged,
        /// made in the cook.
        pub gpu_materials: bool = true => any,
        /// The runtime virtual texture: the ground's materials
        /// baked into pages the views ask for, which pixels read in place of
        /// computing them; ground finer than its texels, steep ground and pages
        /// not baked yet compute theirs.
        pub virtual_texture: bool = true => any,
        /// An optional budget of pages it keeps (about 340 KB each). Omitted,
        /// it keeps as many as the views ask for, up to the device's texture
        /// array layers; beyond either, pages unasked for a while make room,
        /// and those asked for with none left drop (the ground computes
        /// itself).
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
