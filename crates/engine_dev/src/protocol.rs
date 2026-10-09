use engine_core::{
    AmbientOcclusion, Bloom, Clouds, ContactShadows, Decals, Deformation, GroundCover,
    LevelOfDetail, LocalLights, Navigation, Occupancy, Ocean, Particles, ReflectionProbes,
    Reflections, Refraction, Shadows, Sky, SunShafts, TerrainSettings, TimeOfDay, VirtualGeometry,
    Water, Wind, WorldStreaming,
};
use engine_core::{OcclusionCulling, SubpixelCulling, TemporalAntialiasing};
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
    /// Terrain surfaces under world points `[x, z]`, 1 to
    /// 256: each one's terrain, height, normal, hole and the level of the
    /// resident tile answering, what its hydrology bake found there and the
    /// water standing there, and the meters from the
    /// nearest road's edge (negative on roads, null beyond
    /// their masks' reach), or null where no tile is resident.
    TerrainHeight {
        version: u32,
        #[cfg_attr(not(target_arch = "wasm32"), schemars(length(min = 1, max = 256)))]
        points: Vec<[f64; 2]>,
    },
    /// The shortest walk over a generated terrain from world `from` to `to`
    /// `[x, z]`: across ground an agent stands on (no holes,
    /// no water deeper than `max_water`, no closer than `radius` to placed
    /// objects that collide on resident tiles), each step no steeper than
    /// `max_slope` (rise per meter), pulled taut and following the ground.
    /// It walks pyramid `level` (0 by default) unless the area within
    /// `margin` meters of the ends holds more than `max_samples` samples,
    /// then the finest level within. Returns its points (world, rounded to
    /// centimeters), length, the level walked and samples searched, or a
    /// null walk where none reaches. Defaults: max_slope 0.8, radius 0.5,
    /// max_water 0.6, margin 96, max_samples 1048576.
    NavPath {
        version: u32,
        from: [f64; 2],
        to: [f64; 2],
        #[serde(default)]
        options: Option<engine_core::terrain_data::NavOptions>,
    },
    /// The sea's surface over world points `[x, z]`, 1 to
    /// 256, at simulation time `seconds` (by default the time the last frame
    /// presented), as frames draw it: its height, normal, how much the waves
    /// compress it there (1 still, toward 0 on sharpening crests, where foam
    /// gathers) and the still sea's point the waves moved there, or null
    /// without a sea; the ground below where a terrain tile is resident,
    /// and the water's depth over it.
    WaterQuery {
        version: u32,
        #[cfg_attr(not(target_arch = "wasm32"), schemars(length(min = 1, max = 256)))]
        points: Vec<[f64; 2]>,
        #[serde(default)]
        seconds: Option<f64>,
    },
    /// The tracks deformers and agents pressed at world points
    /// `[x, z]`, 1 to 256, refilled to simulation time `seconds` (by default
    /// the time the last frame presented), as the log holds them: how deep
    /// the deepest sinks deformable ground (`track`) and how high rims raise
    /// it (`rim`); where a tile is resident, the most its surface materials
    /// let tracks sink it (`deform`), the depth it shows (`depth`, the
    /// lesser) and the ground's height. Also the stamps the log holds and
    /// those its budget dropped.
    DeformationQuery {
        version: u32,
        #[cfg_attr(not(target_arch = "wasm32"), schemars(length(min = 1, max = 256)))]
        points: Vec<[f64; 2]>,
        #[serde(default)]
        seconds: Option<f64>,
    },
    /// Compares, over `samples` by `samples` points across `half_extent`
    /// meters of world `center` `[x, z]`, what terrain queries (the tiles
    /// renderers draw), collision rays and generated terrains' layers say
    /// the ground is: the largest differences and where.
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
    /// The cheapest route over the generated terrain under world `from`
    /// `[x, z]` to world `to`, over its layers with their
    /// bakes, as a path layer would follow it: walking between the samples
    /// of pyramid `level`, each step's cost growing by `grade_cost` times
    /// its squared grade and by `water_cost` where water stands, each turn
    /// costing `turn_cost` meters per unit of one less its angle's cosine,
    /// refusing grades steeper than `max_grade` (rise per meter). Defaults:
    /// level 3, max_grade 0.12, grade_cost 100, water_cost 50, turn_cost 40.
    /// Returns the terrain's origin `[x, z]` and the route's points
    /// terrain-local (a path layer's `points`; add the origin for world
    /// points), its length, its steepest grade within the limit and the
    /// turning steps of its hairpins crossing steeper ground (up to 45
    /// degrees), or a null route where none keeps within the limits. Its
    /// work grows with the square of the ends' distance.
    TerrainRoute {
        version: u32,
        from: [f64; 2],
        to: [f64; 2],
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(max = 15)))]
        level: Option<u32>,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = 0.001, max = 10.0)))]
        max_grade: Option<f64>,
        #[serde(default)]
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(range(min = 0.0, max = 1000000.0))
        )]
        grade_cost: Option<f64>,
        #[serde(default)]
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(range(min = 0.0, max = 1000000.0))
        )]
        water_cost: Option<f64>,
        #[serde(default)]
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(range(min = 0.0, max = 1000000.0))
        )]
        turn_cost: Option<f64>,
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

/// What the terrain control shows.
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
    Detail,
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
            | Self::TerrainRoute { version, .. }
            | Self::WaterQuery { version, .. }
            | Self::DeformationQuery { version, .. }
            | Self::NavPath { version, .. }
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
        /// Present through GPU blending/composition/culling
        /// instead of the exact CPU path; for comparisons.
        #[serde(default)]
        gpu: bool,
    },
    /// Temporarily toggle visibility culling and render without advancing clocks.
    Culling {
        version: u32,
        enabled: bool,
    },
    /// Temporarily replace environment sections and render
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
        #[serde(default)]
        contact_shadows: Option<ContactShadows>,
        #[serde(default)]
        water: Option<Water>,
        #[serde(default)]
        ocean: Option<Ocean>,
        #[serde(default)]
        deformation: Option<Deformation>,
        #[serde(default)]
        clouds: Option<Clouds>,
        #[serde(default)]
        world_streaming: Option<WorldStreaming>,
        #[serde(default)]
        sky: Option<Sky>,
        #[serde(default)]
        time_of_day: Option<TimeOfDay>,
        #[serde(default)]
        navigation: Option<Navigation>,
        #[serde(default)]
        occlusion_culling: Option<OcclusionCulling>,
        #[serde(default)]
        subpixel_culling: Option<SubpixelCulling>,
        #[serde(default)]
        temporal_antialiasing: Option<TemporalAntialiasing>,
    },
    /// Temporarily set terrain's level of detail and view and
    /// render while paused: `pixel_error` replaces the settings' (1 to
    /// 64), `freeze` keeps streaming and patches following the camera's
    /// current position while the camera moves on, and `view` shows terrain
    /// shaded or its levels, normals, tiles and patches, materials, sky
    /// visibility or detail. Omitted fields stay as they
    /// are.
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
    /// Temporarily edit a generated terrain over its layers and bakes
    /// and render while paused: `change` raises or lowers the
    /// ground by meters, levels it to a world height or cuts holes in it, at
    /// full strength within `radius` meters of world `center` `[x, z]` and
    /// fading over `falloff` more, under `id` (1 to 64 letters, digits, `_`,
    /// `-` and `.`), which repeating leaves the terrain alone. Only the tiles
    /// under it regenerate. Returns whether it applied, the edits standing,
    /// where tiles regenerate (world `min` and `max` `[x, z]`) and the most
    /// the ground moved (meters). `terrain` names a scene's terrain entity
    /// (the first terrain when omitted). The `Terrain` inspector lists the
    /// edits as stamp layers to save into the scene.
    TerrainEdit {
        version: u32,
        #[serde(default)]
        terrain: Option<String>,
        id: String,
        center: [f64; 2],
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(range(min = 0.0, max = 100000.0))
        )]
        radius: f64,
        #[serde(default)]
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(range(min = 0.0, max = 100000.0))
        )]
        falloff: f64,
        change: engine_core::terrain_data::StampOp,
    },
    /// Take the terrain's last edit away and render while
    /// paused; `terrain` as for `terrain_edit`.
    TerrainUndo {
        version: u32,
        #[serde(default)]
        terrain: Option<String>,
    },
    /// Put back the last edit undo took away, unless an edit came since,
    /// and render while paused.
    TerrainRedo {
        version: u32,
        #[serde(default)]
        terrain: Option<String>,
    },
    /// Press a track as a deformer would, now, and render
    /// while paused: a capsule `radius` meters across (0.01 to 50) from world
    /// `from` `[x, z]` to `to` (a footprint at `from` when omitted), sinking
    /// ground as deformable as that `depth` meters along its middle (0 to
    /// 10), round (`flatness` 0) to flat (1; 0.5 by default). `clear`
    /// forgets every track first. Not a source change: the log and the map
    /// keep it until cleared or refilled. Returns the stamp, the stamps the
    /// log holds and whether deformation is enabled.
    Deform {
        version: u32,
        from: [f64; 2],
        #[serde(default)]
        to: Option<[f64; 2]>,
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = 0.01, max = 50.0)))]
        radius: f64,
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = 0.0, max = 10.0)))]
        depth: f64,
        #[serde(default = "default_flatness")]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(range(min = 0.0, max = 1.0)))]
        flatness: f64,
        #[serde(default)]
        clear: bool,
    },
    /// Send navigation agent `entity` (authored ID) walking to world
    /// `destination` `[x, z]`, or stop it with null, and
    /// render while paused; steps then walk it. Not a source change.
    /// Returns its status.
    NavGo {
        version: u32,
        entity: String,
        #[serde(default)]
        destination: Option<[f32; 2]>,
    },
    /// Remove streamed items for good, or restore them, by
    /// ID (`scene/root` for scenes; at most 256 each), and render while
    /// paused: `clear` restores every item first. Loaded cells despawn and
    /// spawn them at once. Not a source change: WorldChanges keeps them,
    /// as a game's save would. Returns how many items are removed.
    WorldChanges {
        version: u32,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(length(max = 256)))]
        remove: Vec<String>,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(length(max = 256)))]
        restore: Vec<String>,
        #[serde(default)]
        clear: bool,
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
        /// With `alpha`, capture GPU presentation instead of
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
    pub const OPS: [&'static str; 19] = [
        "present",
        "culling",
        "environment",
        "terrain",
        "terrain_edit",
        "terrain_undo",
        "terrain_redo",
        "deform",
        "world_changes",
        "nav_go",
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
            | Self::TerrainEdit { version, .. }
            | Self::TerrainUndo { version, .. }
            | Self::TerrainRedo { version, .. }
            | Self::Deform { version, .. }
            | Self::WorldChanges { version, .. }
            | Self::NavGo { version, .. }
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

fn default_flatness() -> f64 {
    0.5
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
