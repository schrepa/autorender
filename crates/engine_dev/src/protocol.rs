use engine_scene::{SceneAtmosphere, SceneEnvironment};
use serde::{Deserialize, Serialize};

#[cfg(not(target_arch = "wasm32"))]
use engine_settings::{rule::same, stated};

/// The bounds requests keep: their schemas state them and their handlers
/// check them, so neither restates a number.
pub mod bounds {
    use engine_settings::Rule;

    /// Entries a page holds at most.
    pub const MAX_PAGE_SIZE: usize = 100;
    /// Bytes of an inspector's description at most: its type's doc comment.
    pub const MAX_DESCRIPTION_BYTES: usize = 512;
    /// A page's size.
    pub const PAGE: Rule = Rule::Range(1.0, MAX_PAGE_SIZE as f64);
    /// Points a height, water or deformation query names at most.
    pub const MAX_POINTS: usize = 256;
    /// World coordinates requests name, as collision queries bound them.
    pub const COORDINATE: Rule = Rule::Range(-1.0e6, 1.0e6);
    /// How far a ray or a cast reaches.
    pub const DISTANCE: Rule = Rule::Range(0.0, 2.0e6);
    /// Where a page of sensor events starts.
    pub const EVENT_OFFSET: Rule = Rule::Range(0.0, 131_072.0);
    /// A terrain sweep's half extent, meters.
    pub const SWEEP_EXTENT: Rule = Rule::Range(0.0, 4096.0);
    /// A terrain sweep's samples along a side.
    pub const SWEEP_SAMPLES: Rule = Rule::Range(2.0, 64.0);
    /// A presentation fraction.
    pub const ALPHA: Rule = Rule::Range(0.0, 1.0);
    /// Benchmark warm-up frames.
    pub const WARMUP: Rule = Rule::Range(0.0, 120.0);
    /// Benchmark samples.
    pub const SAMPLES: Rule = Rule::Range(1.0, 240.0);
    /// Fixed ticks a step advances.
    pub const TICKS: Rule = Rule::Range(1.0, engine_core::MAX_DEV_TICKS as f64);
    /// An input axis.
    pub const AXIS: Rule = Rule::Range(-1.0, 1.0);
    /// An input's look delta.
    pub const TURN: Rule = Rule::Range(-1000.0, 1000.0);
    /// IDs a world change names at most, in each list.
    pub const MAX_CHANGES: usize = 256;
    /// Bytes of a terrain edit's ID at most.
    pub const MAX_EDIT_ID: usize = 64;
}

/// The schema transform stating a stamp's `field` rule on the request field
/// of that name.
#[cfg(not(target_arch = "wasm32"))]
fn stamp_rule(field: &'static str) -> impl Fn(&mut schemars::Schema) {
    engine_settings::stated(engine_core::Stamp::rule_of(field).expect("stamps declare the field"))
}

/// The schema transform stating route option `field`'s rule on the request
/// field of that name.
#[cfg(not(target_arch = "wasm32"))]
fn route_rule(field: &'static str) -> impl Fn(&mut schemars::Schema) {
    let state =
        stated(engine_core::RouteOptions::rule_of(field).expect("route options declare the field"));
    let default = serde_json::to_value(engine_core::RouteOptions::default())
        .expect("route options serialize")[field]
        .clone();
    move |schema| {
        state(schema);
        schema.insert("default".into(), default.clone());
    }
}

/// The development protocol's version, independent of scene format versions.
pub const PROTOCOL_VERSION: u32 = 1;
/// Bytes of one request's JSON.
pub const MAX_REQUEST_BYTES: usize = 16 * 1024;
/// Bytes of one reply's JSON.
pub const MAX_RESPONSE_BYTES: usize = 64 * 1024;

/// Where a query runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handler {
    /// Status, inspection, assets and descriptions.
    Tools,
    /// The collision world's queries.
    Collision,
    /// Terrain, water, deformation and navigation queries.
    Terrain,
}

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
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::PAGE)))]
        limit: usize,
    },
    Assets {
        version: u32,
    },
    Describe {
        version: u32,
        #[serde(default)]
        name: Option<String>,
        /// With `SceneDocument`: one definition of the scene schema, which
        /// is too large for one response; without it, the schema's top
        /// level and the names of its definitions.
        #[serde(default)]
        definition: Option<String>,
    },
    /// Closest authoritative hit; rejects incomplete native candidate scans.
    Raycast {
        version: u32,
        origin: [f64; 3],
        direction: [f64; 3],
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::DISTANCE)))]
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
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::PAGE)))]
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
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::DISTANCE)))]
        max_distance: f64,
        #[serde(default)]
        filter: CollisionFilterRequest,
    },
    /// Terrain surfaces under world points `[x, z]`: each one's terrain, height, normal, hole and the level of the
    /// resident tile answering, what its hydrology bake found there and the
    /// water standing there, and the meters from the
    /// nearest road's edge (negative on roads, null beyond
    /// their masks' reach), or null where no tile is resident.
    TerrainHeight {
        version: u32,
        #[cfg_attr(not(target_arch = "wasm32"), schemars(length(min = 1, max = bounds::MAX_POINTS)))]
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
    /// null walk where none reaches.
    NavPath {
        version: u32,
        from: [f64; 2],
        to: [f64; 2],
        #[serde(default)]
        options: Option<engine_core::terrain_data::NavOptions>,
    },
    /// The sea's surface over world points `[x, z]` at simulation time `seconds` (by default the time the last frame
    /// presented), as frames draw it: its height, normal, how much the waves
    /// compress it there (1 still, toward 0 on sharpening crests, where foam
    /// gathers) and the still sea's point the waves moved there, or null
    /// without a sea; the ground below where a terrain tile is resident,
    /// and the water's depth over it.
    WaterQuery {
        version: u32,
        #[cfg_attr(not(target_arch = "wasm32"), schemars(length(min = 1, max = bounds::MAX_POINTS)))]
        points: Vec<[f64; 2]>,
        #[serde(default)]
        seconds: Option<f64>,
    },
    /// The tracks deformers and agents pressed at world points
    /// `[x, z]`, refilled to simulation time `seconds` (by default
    /// the time the last frame presented), as the log holds them: how deep
    /// the deepest sinks deformable ground (`track`) and how high rims raise
    /// it (`rim`); where a tile is resident, the most its surface materials
    /// let tracks sink it (`deform`), the depth it shows (`depth`, the
    /// lesser) and the ground's height. Also the stamps the log holds and
    /// those its budget dropped.
    DeformationQuery {
        version: u32,
        #[cfg_attr(not(target_arch = "wasm32"), schemars(length(min = 1, max = bounds::MAX_POINTS)))]
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
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::SWEEP_EXTENT)))]
        half_extent: f64,
        #[serde(default = "default_sweep_samples")]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::SWEEP_SAMPLES)))]
        samples: u32,
    },
    /// The cheapest route over the generated terrain under world `from`
    /// `[x, z]` to world `to`, over its layers with their
    /// bakes, as a path layer would follow it: walking between the samples
    /// of pyramid `level`, each step's cost growing by `grade_cost` times
    /// its squared grade and by `water_cost` where water stands, each turn
    /// costing `turn_cost` meters per unit of one less its angle's cosine,
    /// refusing grades steeper than `max_grade` (rise per meter).
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
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = route_rule("level")))]
        level: Option<u32>,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = route_rule("max_grade")))]
        max_grade: Option<f64>,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = route_rule("grade_cost")))]
        grade_cost: Option<f64>,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = route_rule("water_cost")))]
        water_cost: Option<f64>,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = route_rule("turn_cost")))]
        turn_cost: Option<f64>,
    },
    /// Non-consuming page of the latest fixed-tick sensor transition snapshot.
    CollisionEvents {
        version: u32,
        #[serde(default)]
        entity: Option<String>,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::EVENT_OFFSET)))]
        offset: usize,
        #[serde(default = "default_limit")]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::PAGE)))]
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
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(transform = stated(engine_settings::rule::range(engine_core::MIN_COLLIDER_EXTENT, engine_core::MAX_COLLIDER_EXTENT)))
        )]
        half_extents: [f64; 3],
    },
    Sphere {
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(engine_settings::rule::range(engine_core::MIN_COLLIDER_EXTENT, engine_core::MAX_COLLIDER_EXTENT))))]
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
    /// Where the query runs: the one routing table of queries.
    pub fn handler(&self) -> Handler {
        match self {
            Self::Status { .. }
            | Self::Inspect { .. }
            | Self::Assets { .. }
            | Self::Describe { .. } => Handler::Tools,
            Self::Raycast { .. }
            | Self::Overlap { .. }
            | Self::ShapeCast { .. }
            | Self::CollisionEvents { .. } => Handler::Collision,
            Self::TerrainHeight { .. }
            | Self::TerrainSweep { .. }
            | Self::TerrainRoute { .. }
            | Self::WaterQuery { .. }
            | Self::DeformationQuery { .. }
            | Self::NavPath { .. } => Handler::Terrain,
        }
    }

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
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::ALPHA)))]
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
    /// Temporarily replace environment sections and render while paused.
    /// Sections are a scene `environment`'s, validated alike; omitted
    /// sections stay. A scene with an environment restores its own on
    /// reload; persist a look by saving it there.
    Environment(EnvironmentControl),
    /// Temporarily set terrain's level of detail and view and
    /// render while paused: `pixel_error` replaces the settings', `freeze`
    /// keeps streaming and patches following the camera's
    /// current position while the camera moves on, and `view` shows terrain
    /// shaded or its levels, normals, tiles and patches, materials, sky
    /// visibility or detail. Omitted fields stay as they
    /// are.
    Terrain {
        version: u32,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(same(engine_core::TerrainSettings::rule_of("pixel_error")))))]
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
    /// fading over `falloff` more, under `id` (letters, digits, `_`, `-` and
    /// `.`), which repeating leaves the terrain alone. Only the tiles
    /// under it regenerate. Returns whether it applied, the edits standing,
    /// where tiles regenerate (world `min` and `max` `[x, z]`) and the most
    /// the ground moved (meters). `terrain` names a scene's terrain entity
    /// (the first terrain when omitted). The `Terrain` inspector lists the
    /// edits as stamp layers to save into the scene.
    TerrainEdit {
        version: u32,
        #[serde(default)]
        terrain: Option<String>,
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(length(min = 1, max = bounds::MAX_EDIT_ID), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
        )]
        id: String,
        center: [f64; 2],
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stamp_rule("radius")))]
        radius: f64,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stamp_rule("falloff")))]
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
    /// while paused: a capsule `radius` meters across from world `from`
    /// `[x, z]` to `to` (a footprint at `from` when omitted), sinking ground
    /// as deformable as that `depth` meters along its middle, round
    /// (`flatness` 0) to flat (1). `clear`
    /// forgets every track first. Not a source change: the log and the map
    /// keep it until cleared or refilled. Returns the stamp, the stamps the
    /// log holds and whether deformation is enabled.
    Deform {
        version: u32,
        from: [f64; 2],
        #[serde(default)]
        to: Option<[f64; 2]>,
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(same(engine_core::Deformer::rule_of("radius")))))]
        radius: f64,
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(same(engine_core::Deformer::rule_of("depth")))))]
        depth: f64,
        #[serde(default = "default_flatness")]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(same(engine_core::Deformer::rule_of("flatness")))))]
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
    /// Set an authored entity's animation player and render while paused:
    /// its clip (an animation ID of the scene), time in seconds, speed and
    /// looping; omitted values stay. Returns them before and after.
    Animation {
        version: u32,
        entity: String,
        #[serde(default)]
        clip: Option<String>,
        #[serde(default)]
        time: Option<f32>,
        #[serde(default)]
        speed: Option<f32>,
        #[serde(default)]
        looping: Option<bool>,
    },
    /// Remove streamed items for good, or restore them, by
    /// ID (`scene/root` for scenes), and render while
    /// paused: `clear` restores every item first. Loaded cells despawn and
    /// spawn them at once. Not a source change: WorldChanges keeps them,
    /// as a game's save would. Returns how many items are removed.
    WorldChanges {
        version: u32,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(length(max = bounds::MAX_CHANGES)))]
        remove: Vec<String>,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(length(max = bounds::MAX_CHANGES)))]
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
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::WARMUP)))]
        warmup: u32,
        #[serde(default = "default_samples")]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::SAMPLES)))]
        samples: u32,
        #[serde(default)]
        mode: BenchmarkMode,
    },
    /// Advance exactly `ticks` fixed ticks, then one presentation/render
    /// boundary.
    Step {
        version: u32,
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::TICKS)))]
        ticks: u32,
    },
    /// Replace the temporary batch applied on the next explicit step.
    Input {
        version: u32,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::AXIS)))]
        forward: f32,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::AXIS)))]
        right: f32,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::AXIS)))]
        up: f32,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::TURN)))]
        yaw: f32,
        #[serde(default)]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::TURN)))]
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
            schemars(with = "f64", transform = stated(bounds::ALPHA))
        )]
        alpha: Option<f64>,
        /// With `alpha`, capture GPU presentation instead of
        /// the exact CPU path.
        #[serde(default)]
        gpu: bool,
    },
    /// Runs the game's own control `name` with `args`, as the page
    /// registers it (a Stress Lab workload, look or view), then renders.
    /// The bridge's describe lists the game's controls.
    Game {
        version: u32,
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(regex(pattern = r"^[a-z][a-z0-9_]{0,63}$"))
        )]
        name: String,
        /// The control's arguments, a JSON object.
        #[serde(default)]
        args: serde_json::Map<String, serde_json::Value>,
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
    /// Whether the control needs a paused host: every one but pausing and
    /// resuming.
    pub fn needs_pause(&self) -> bool {
        match self {
            Self::Present { .. }
            | Self::Culling { .. }
            | Self::Environment(..)
            | Self::Terrain { .. }
            | Self::TerrainEdit { .. }
            | Self::TerrainUndo { .. }
            | Self::TerrainRedo { .. }
            | Self::Deform { .. }
            | Self::WorldChanges { .. }
            | Self::NavGo { .. }
            | Self::Animation { .. }
            | Self::Camera { .. }
            | Self::Benchmark { .. }
            | Self::Step { .. }
            | Self::Input { .. }
            | Self::Render { .. }
            | Self::Capture { .. }
            | Self::Game { .. }
            | Self::Reload { .. } => true,
            Self::Pause { .. } | Self::Resume { .. } => false,
        }
    }

    pub fn version(&self) -> u32 {
        match self {
            Self::Present { version, .. }
            | Self::Culling { version, .. }
            | Self::Terrain { version, .. }
            | Self::TerrainEdit { version, .. }
            | Self::TerrainUndo { version, .. }
            | Self::TerrainRedo { version, .. }
            | Self::Deform { version, .. }
            | Self::WorldChanges { version, .. }
            | Self::NavGo { version, .. }
            | Self::Animation { version, .. }
            | Self::Camera { version, .. }
            | Self::Benchmark { version, .. }
            | Self::Step { version, .. }
            | Self::Input { version, .. }
            | Self::Render { version }
            | Self::Pause { version }
            | Self::Resume { version }
            | Self::Capture { version, .. }
            | Self::Game { version, .. }
            | Self::Reload { version, .. } => *version,
            Self::Environment(control) => control.version,
        }
    }
}

/// engine_dev's part of [`engine_core::environment_sections`]: the
/// environment control's request.
macro_rules! control_sections {
    ($($key:ident: $settings:ident { since: $since:literal, opt_in: $opt_in:literal }),* $(,)?) => {
        /// The environment control: the protocol version and the sections it
        /// replaces, each in its scene environment shape.
        #[derive(Debug, Deserialize, Serialize)]
        #[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
        #[serde(deny_unknown_fields)]
        pub struct EnvironmentControl {
            pub version: u32,
            /// Whether a named section's omitted fields keep their current
            /// values instead of taking their defaults.
            #[serde(default)]
            pub merge: bool,
            #[serde(default)]
            pub atmosphere: Option<SceneAtmosphere>,
            $(
                #[serde(default)]
                pub $key: Option<engine_core::$settings>,
            )*
        }

        impl EnvironmentControl {
            /// The sections it names, as a scene environment.
            pub fn sections(&self) -> SceneEnvironment {
                SceneEnvironment {
                    atmosphere: self.atmosphere,
                    $($key: self.$key,)*
                }
            }
        }
    };
}

engine_core::environment_sections!(control_sections);

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
