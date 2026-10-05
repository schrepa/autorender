use engine_core::{
    AmbientOcclusion, Bloom, Decals, GroundCover, LevelOfDetail, LocalLights, Occupancy, Particles,
    ReflectionProbes, Reflections, Refraction, Shadows, SunShafts, TerrainSettings,
    VirtualGeometry, Wind,
};
use engine_scene::SceneAtmosphere;
use serde::{Deserialize, Serialize};

/// Versioned, read-only requests processed synchronously between frames.
/// Transport/session identity and execution controls belong to the host.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum DevRequest {
    Status {
        version: u32,
    },
    Inspect {
        version: u32,
        #[serde(default)]
        entity: Option<String>,
        /// Select direct children by their parent's authored ID; excludes entity.
        #[serde(default)]
        parent: Option<String>,
        #[serde(default)]
        component: Option<String>,
        #[serde(default)]
        offset: usize,
        #[serde(default = "default_limit")]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = 1, max = 100)))]
        limit: usize,
    },
    Assets {
        version: u32,
    },
    Describe {
        version: u32,
        #[serde(default)]
        name: Option<String>,
    },
    /// Closest authoritative hit; rejects incomplete native candidate scans.
    Raycast {
        version: u32,
        origin: [f64; 3],
        direction: [f64; 3],
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(range(min = 0.0, max = 2_000_000.0))
        )]
        max_distance: f64,
        #[serde(default)]
        filter: CollisionFilterRequest,
    },
    /// Complete bounded shape overlap, using a world-space rigid query pose.
    Overlap {
        version: u32,
        shape: CollisionShapeRequest,
        position: [f64; 3],
        #[serde(default = "identity_rotation")]
        rotation: [f64; 4],
        #[serde(default)]
        filter: CollisionFilterRequest,
        #[serde(default = "default_limit")]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = 1, max = 100)))]
        limit: usize,
    },
    /// Closest translation-only shape cast; orientation stays fixed along the ray.
    ShapeCast {
        version: u32,
        shape: CollisionShapeRequest,
        position: [f64; 3],
        #[serde(default = "identity_rotation")]
        rotation: [f64; 4],
        direction: [f64; 3],
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(range(min = 0.0, max = 2_000_000.0))
        )]
        max_distance: f64,
        #[serde(default)]
        filter: CollisionFilterRequest,
    },
    /// Terrain surfaces (decision 051) under world points `[x, z]`, 1 to
    /// 256: each one's terrain, height, normal, hole and the level of the
    /// resident tile answering, or null where no tile is resident.
    TerrainHeight {
        version: u32,
        #[cfg_attr(not(target_arch = "wasm32"), schemars(length(min = 1, max = 256)))]
        points: Vec<[f64; 2]>,
    },
    /// Compares, over `samples` by `samples` points across `half_extent`
    /// meters of world `center` `[x, z]`, what terrain queries (the tiles
    /// renderers draw), collision rays and generated terrains' layers say
    /// the ground is (decision 051): the largest differences and where.
    TerrainSweep {
        version: u32,
        center: [f64; 2],
        #[serde(default = "default_half_extent")]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = 0.0, max = 4096.0)))]
        half_extent: f64,
        #[serde(default = "default_sweep_samples")]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = 2, max = 64)))]
        samples: u32,
    },
    /// Non-consuming page of the latest fixed-tick sensor transition snapshot.
    CollisionEvents {
        version: u32,
        #[serde(default)]
        entity: Option<String>,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(max = 131072)))]
        offset: usize,
        #[serde(default = "default_limit")]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = 1, max = 100)))]
        limit: usize,
    },
}

/// Query geometry dimensions are world units. No scale or rotation changes occur
/// during a cast; sphere/cuboid transform admission matches native collision.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CollisionShapeRequest {
    Cuboid {
        half_extents: [f64; 3],
    },
    Sphere {
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(range(min = 0.0001, max = 10000.0))
        )]
        radius: f64,
    },
}

/// Layer matching is symmetric: query memberships intersect collider filter,
/// and collider memberships intersect query filter. Exclusion uses a live ID
/// in the supplied active scene; no runtime entity handle is accepted.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct CollisionFilterRequest {
    pub memberships: u32,
    pub filter: u32,
    pub sensors: CollisionSensorFilter,
    pub exclude: Option<String>,
}

impl Default for CollisionFilterRequest {
    fn default() -> Self {
        Self {
            memberships: 1,
            filter: u32::MAX,
            sensors: CollisionSensorFilter::Include,
            exclude: None,
        }
    }
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum CollisionSensorFilter {
    #[default]
    Include,
    Exclude,
    Only,
}

fn identity_rotation() -> [f64; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

fn default_half_extent() -> f64 {
    64.0
}

fn default_sweep_samples() -> u32 {
    16
}

/// What the terrain control shows (decision 051).
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum TerrainViewRequest {
    Shaded,
    Levels,
    Normals,
    Tiles,
    Materials,
    Sky,
}

fn default_limit() -> usize {
    20
}

impl DevRequest {
    pub fn version(&self) -> u32 {
        match self {
            Self::Status { version }
            | Self::Inspect { version, .. }
            | Self::Assets { version }
            | Self::Describe { version, .. }
            | Self::Raycast { version, .. }
            | Self::Overlap { version, .. }
            | Self::ShapeCast { version, .. }
            | Self::TerrainHeight { version, .. }
            | Self::TerrainSweep { version, .. }
            | Self::CollisionEvents { version, .. } => *version,
        }
    }
}

/// Host controls available alongside read-only [`DevRequest`] operations.
/// The browser host enforces pause/lifecycle rules. Deserialization validates
/// structure; the host also validates version, finite values, ranges, and u64
/// revision parsing before changing state.
// One short-lived request at a time: the environment variant's size does not
// matter, and boxing its sections would complicate the typed schema.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum BrowserControl {
    /// Preview fixed-tick poses at a chosen fraction without advancing clocks.
    /// Requires pause; ordinary step/render/capture retain authoritative poses.
    Present {
        version: u32,
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = 0.0, max = 1.0)))]
        alpha: f64,
        /// Present through GPU blending/composition/culling (decision 014)
        /// instead of the exact CPU path; for comparisons.
        #[serde(default)]
        gpu: bool,
    },
    /// Temporarily toggle visibility culling and render without advancing clocks.
    Culling {
        version: u32,
        enabled: bool,
    },
    /// Temporarily replace environment sections (decision 022) and render
    /// while paused. Sections are a v5 scene `environment`'s, validated alike;
    /// omitted sections stay. A v5 scene with an environment restores its own
    /// on reload; persist a look by saving it there.
    Environment {
        version: u32,
        #[serde(default)]
        atmosphere: Option<SceneAtmosphere>,
        #[serde(default)]
        shadows: Option<Shadows>,
        #[serde(default)]
        ambient_occlusion: Option<AmbientOcclusion>,
        #[serde(default)]
        bloom: Option<Bloom>,
        #[serde(default)]
        local_lights: Option<LocalLights>,
        #[serde(default)]
        sun_shafts: Option<SunShafts>,
        #[serde(default)]
        reflections: Option<Reflections>,
        #[serde(default)]
        particles: Option<Particles>,
        #[serde(default)]
        refraction: Option<Refraction>,
        #[serde(default)]
        decals: Option<Decals>,
        #[serde(default)]
        reflection_probes: Option<ReflectionProbes>,
        #[serde(default)]
        virtual_geometry: Option<VirtualGeometry>,
        #[serde(default)]
        level_of_detail: Option<LevelOfDetail>,
        #[serde(default)]
        terrain: Option<TerrainSettings>,
        #[serde(default)]
        ground_cover: Option<GroundCover>,
        #[serde(default)]
        wind: Option<Wind>,
        #[serde(default)]
        occupancy: Option<Occupancy>,
    },
    /// Temporarily set terrain's level of detail and view (decision 051) and
    /// render while paused: `pixel_error` replaces the settings' (1 to
    /// 64), `freeze` keeps streaming and patches following the camera's
    /// current position while the camera moves on, and `view` shows terrain
    /// shaded or its levels, normals or tiles and patches. Omitted fields
    /// stay as they are.
    Terrain {
        version: u32,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = 1.0, max = 64.0)))]
        pixel_error: Option<f32>,
        #[serde(default)]
        freeze: Option<bool>,
        #[serde(default)]
        view: Option<TerrainViewRequest>,
    },
    /// Set a complete temporary world-space camera and render while paused.
    Camera {
        version: u32,
        position: [f32; 3],
        target: [f32; 3],
        vertical_fov_radians: f32,
        near: f32,
        far: f32,
    },
    /// RAF-paced host sampling, bounded to ten seconds; advances one tick per
    /// sample in step mode and no gameplay in render/present modes. Present mode
    /// samples existing fixed-tick history at alpha 0.5. Clears pending input.
    Benchmark {
        version: u32,
        #[serde(default = "default_warmup")]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = 0, max = 120)))]
        warmup: u32,
        #[serde(default = "default_samples")]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = 1, max = 240)))]
        samples: u32,
        #[serde(default)]
        mode: BenchmarkMode,
    },
    /// Advance exactly 1–120 fixed ticks, then one presentation/render boundary.
    Step {
        version: u32,
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = 1, max = 120)))]
        ticks: u32,
    },
    /// Replace the temporary batch applied on the next explicit step.
    Input {
        version: u32,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = -1.0, max = 1.0)))]
        forward: f32,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = -1.0, max = 1.0)))]
        right: f32,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = -1.0, max = 1.0)))]
        up: f32,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = -1000.0, max = 1000.0)))]
        yaw: f32,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = -1000.0, max = 1000.0)))]
        pitch: f32,
        #[serde(default)]
        held: u32,
        #[serde(default)]
        pressed: u32,
        #[serde(default)]
        released: u32,
    },
    /// Extract and render current state without advancing gameplay.
    Render {
        version: u32,
    },
    Pause {
        version: u32,
    },
    Resume {
        version: u32,
    },
    /// Render while paused and wait for presentation before host capture.
    /// Omitted alpha renders authoritative current state; a supplied fraction
    /// previews existing fixed-tick history without advancing gameplay.
    Capture {
        version: u32,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_optional_alpha"
        )]
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(with = "f64", range(min = 0.0, max = 1.0))
        )]
        alpha: Option<f64>,
        /// With `alpha`, capture GPU presentation (decision 014) instead of
        /// the exact CPU path.
        #[serde(default)]
        gpu: bool,
    },
    /// Fetch saved source and require both expected runtime and source revisions.
    Reload {
        version: u32,
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(regex(pattern = r"^(0|[1-9][0-9]*)$"))
        )]
        expected_revision: String,
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(regex(pattern = r"^(0|[1-9][0-9]*)$"))
        )]
        revision: String,
    },
}

impl BrowserControl {
    /// The operation of every control, as requests name it.
    pub const OPS: [&'static str; 13] = [
        "present",
        "culling",
        "environment",
        "terrain",
        "camera",
        "benchmark",
        "step",
        "input",
        "render",
        "pause",
        "resume",
        "capture",
        "reload",
    ];

    pub fn version(&self) -> u32 {
        match self {
            Self::Present { version, .. }
            | Self::Culling { version, .. }
            | Self::Environment { version, .. }
            | Self::Terrain { version, .. }
            | Self::Camera { version, .. }
            | Self::Benchmark { version, .. }
            | Self::Step { version, .. }
            | Self::Input { version, .. }
            | Self::Render { version }
            | Self::Pause { version }
            | Self::Resume { version }
            | Self::Capture { version, .. }
            | Self::Reload { version, .. } => *version,
        }
    }
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkMode {
    #[default]
    Step,
    Render,
    Present,
}

fn default_warmup() -> u32 {
    30
}

fn deserialize_optional_alpha<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<f64>, D::Error> {
    f64::deserialize(deserializer).map(Some)
}
fn default_samples() -> u32 {
    120
}
