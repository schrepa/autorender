use engine_scene::{SceneAtmosphere, SceneEnvironment};
use engine_settings::ErrorCode;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::DevError;

#[cfg(not(target_arch = "wasm32"))]
use engine_settings::stated;

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
    /// Ordinary frames one `frames` control runs.
    pub const FRAMES: Rule = Rule::Range(1.0, 240.0);
    /// Milliseconds between the frames a `frames` control runs.
    pub const FRAME_INTERVAL: Rule = Rule::Range(1.0, 1000.0);
    /// An input axis.
    pub const AXIS: Rule = Rule::Range(-1.0, 1.0);
    /// An input's look delta.
    pub const TURN: Rule = Rule::Range(-1000.0, 1000.0);
    /// IDs a world change names at most, in each list.
    pub const MAX_CHANGES: usize = 256;
    /// Values a distribution request names at most, within the request's
    /// bytes (`MAX_REQUEST_BYTES`).
    pub const MAX_VALUES: usize = 1024;
}

/// The schema transform stating a stamp's `field` rule on the request field
/// of that name.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn stamp_rule(field: &'static str) -> impl Fn(&mut schemars::Schema) {
    engine_settings::stated(engine_core::Stamp::rule_of(field).expect("stamps declare the field"))
}

use crate::features::{
    EngineControl, EngineQuery,
    animation::AnimationControl,
    collision::{CollisionEventsQuery, OverlapQuery, RaycastQuery, ShapeCastQuery},
    deformation::{DeformControl, DeformationQuery},
    navigation::{NavGoControl, NavPathQuery},
    ocean::WaterQuery,
    terrain::{
        TerrainControl, TerrainEditControl, TerrainHeightQuery, TerrainRedoControl,
        TerrainRouteQuery, TerrainSweepQuery, TerrainUndoControl,
    },
    world_streaming::WorldChangesControl,
};

/// The development protocol's version, independent of scene format versions.
/// Every request carries it beside its op's fields, as `version`
/// ([`parse_request`]).
pub const PROTOCOL_VERSION: u32 = 1;
/// Bytes of one request's JSON.
pub const MAX_REQUEST_BYTES: usize = 16 * 1024;
/// Bytes of one reply's JSON.
pub const MAX_RESPONSE_BYTES: usize = 64 * 1024;

/// Where each control's effect shows: the inspector reading the state it
/// changes, `status` for the host's own, `GameControls` for the game's, or
/// nothing for controls that act without leaving state to read. Every
/// control is listed once and every inspector named exists: the build
/// checks both.
pub const CONTROL_STATE: &[(&str, &str)] = &[
    ("present", "PresentationState"),
    ("culling", "status"),
    ("environment", "Environment"),
    ("terrain", "TerrainDebug"),
    ("terrain_edit", "Terrain"),
    ("terrain_undo", "Terrain"),
    ("terrain_redo", "Terrain"),
    ("deform", "Tracks"),
    ("nav_go", "NavAgent"),
    ("animation", "AnimationPlayer"),
    ("world_changes", "WorldChanges"),
    ("camera", "Camera"),
    ("benchmark", ""),
    ("step", "PresentationState"),
    ("frames", "PresentationState"),
    ("input", "Input"),
    ("render", ""),
    ("pause", "status"),
    ("resume", "status"),
    ("capture", ""),
    ("game", "GameControls"),
    ("reload", "status"),
    ("debug_panic", "status"),
    ("debug_fault", "status"),
];

/// A query, as the handler it runs in takes it: [`DevRequest::route`].
pub(crate) enum Route {
    /// Status, inspection, assets, the digest and descriptions.
    Tools(ToolsQuery),
    /// A feature's query, which its development file answers.
    Feature(Box<dyn EngineQuery>),
}

/// The development tools' own queries ([`DevRequest`]'s, which document
/// them).
pub(crate) enum ToolsQuery {
    Status,
    Inspect {
        entity: Option<String>,
        parent: Option<String>,
        component: Option<String>,
        offset: usize,
        limit: usize,
    },
    Assets,
    Digest,
    Describe {
        name: Option<String>,
        definition: Option<String>,
    },
    Distribution {
        values: Vec<f64>,
    },
}

/// A control as the host runs it: [`BrowserControl::route`].
#[expect(
    clippy::large_enum_variant,
    reason = "one short-lived control at a time, as BrowserControl"
)]
pub enum Control {
    /// A feature's control, which its development file applies; the host
    /// renders after it.
    Feature(Box<dyn EngineControl>),
    /// One the host runs itself.
    Host(HostControl),
}

/// The controls the host runs itself ([`BrowserControl`]'s, which document
/// them).
pub enum HostControl {
    Present {
        alpha: f64,
        gpu: bool,
    },
    Culling {
        enabled: bool,
    },
    Environment(EnvironmentControl),
    Camera {
        position: [f32; 3],
        target: [f32; 3],
        vertical_fov_radians: f32,
        near: f32,
        far: f32,
    },
    Step {
        ticks: u32,
    },
    Input {
        forward: f32,
        right: f32,
        up: f32,
        yaw: f32,
        pitch: f32,
        held: u32,
        pressed: u32,
        released: u32,
    },
    Render,
    DebugPanic,
    DebugFault {
        feature: String,
    },
    /// The ones the host's loop runs, as the request named them:
    /// benchmarks, frames, pausing, resuming, captures, the game's controls
    /// and reloads.
    Loop(BrowserControl),
}

/// Versioned, read-only requests processed synchronously between frames.
/// Transport/session identity and execution controls belong to the host.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    derive(schemars::JsonSchema),
    schemars(transform = with_version)
)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum DevRequest {
    Status {},
    Inspect {
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
    Assets {},
    /// The world's content as hashes that depend on neither entity ids nor
    /// spawn order: per component inspector, the entities holding it and a
    /// hash of their views; per kind of asset, the count and a hash of
    /// their contents; and one hash of both. After the same ticks of the
    /// same scene, equal digests mean equal state.
    Digest {},
    /// The spread of measured `values` as every host computes it: the
    /// smallest and largest, and the median, 95th and 99th percentiles,
    /// each interpolated linearly between the two sorted values around its
    /// rank; `null` without values, or with one that is not a finite number
    /// of at least 0.
    Distribution {
        #[cfg_attr(not(target_arch = "wasm32"), schemars(length(max = bounds::MAX_VALUES)))]
        values: Vec<f64>,
    },
    Describe {
        #[serde(default)]
        name: Option<String>,
        /// With `SceneDocument`: one definition of the scene schema, which
        /// is too large for one response; without it, the schema's top
        /// level and the names of its definitions.
        #[serde(default)]
        definition: Option<String>,
    },
    Raycast(RaycastQuery),
    Overlap(OverlapQuery),
    ShapeCast(ShapeCastQuery),
    TerrainHeight(TerrainHeightQuery),
    NavPath(NavPathQuery),
    WaterQuery(WaterQuery),
    DeformationQuery(DeformationQuery),
    TerrainSweep(TerrainSweepQuery),
    TerrainRoute(TerrainRouteQuery),
    CollisionEvents(CollisionEventsQuery),
}

/// `source`, a request, as the `T` its op names: its JSON parsed, its
/// fields but `version` taken as `T` takes them, then the protocol version
/// every request carries beside them, which must be [`PROTOCOL_VERSION`].
/// No request type declares the version: this envelope takes it, and
/// `with_version` states it in their schemas.
pub fn parse_request<T: DeserializeOwned>(source: &str) -> Result<T, DevError> {
    #[derive(Deserialize)]
    struct Version {
        version: u32,
    }
    let mut request: serde_json::Value = engine_scene::parse_json(source)?;
    let version = request
        .as_object_mut()
        .and_then(|fields| fields.remove("version"));
    let request = engine_scene::parse_value(request)?;
    let Some(version) = version else {
        return Err(DevError::new(
            ErrorCode::MissingField,
            "$.version",
            "missing field `version`",
        ));
    };
    let Version { version } = engine_scene::parse_value(serde_json::json!({ "version": version }))?;
    if version != PROTOCOL_VERSION {
        return Err(DevError::new(
            ErrorCode::UnsupportedVersion,
            "$.version",
            format!("accepted protocol version: {PROTOCOL_VERSION}"),
        ));
    }
    Ok(request)
}

/// States the version every request carries ([`parse_request`]) in
/// `schema`, one of requests: on each, beside its op.
#[cfg(not(target_arch = "wasm32"))]
fn with_version(schema: &mut schemars::Schema) {
    use serde_json::Value;
    let version = serde_json::to_value(schemars::SchemaGenerator::default().subschema_for::<u32>())
        .expect("schemas serialize");
    let requests = schema
        .get_mut("oneOf")
        .and_then(Value::as_array_mut)
        .expect("requests are one of their ops");
    for request in requests {
        request
            .get_mut("properties")
            .and_then(Value::as_object_mut)
            .expect("requests have fields")
            .insert("version".to_owned(), version.clone());
        let required = request
            .get_mut("required")
            .and_then(Value::as_array_mut)
            .expect("requests require their op");
        let op = required
            .iter()
            .position(|name| name == "op")
            .expect("requests require their op");
        required.insert(op + 1, "version".into());
    }
}

/// A page's size when a request names none.
pub(crate) fn default_limit() -> usize {
    20
}

impl DevRequest {
    /// The query as the handler it runs in takes it: the one routing table
    /// of queries. A feature's query goes whole to its development file;
    /// every field of the tools' own is named, so one added reaches the
    /// tools or fails to build.
    pub(crate) fn route(self) -> Route {
        match self {
            Self::Status {} => Route::Tools(ToolsQuery::Status),
            Self::Inspect {
                entity,
                parent,
                component,
                offset,
                limit,
            } => Route::Tools(ToolsQuery::Inspect {
                entity,
                parent,
                component,
                offset,
                limit,
            }),
            Self::Assets {} => Route::Tools(ToolsQuery::Assets),
            Self::Digest {} => Route::Tools(ToolsQuery::Digest),
            Self::Distribution { values } => Route::Tools(ToolsQuery::Distribution { values }),
            Self::Describe { name, definition } => {
                Route::Tools(ToolsQuery::Describe { name, definition })
            }
            Self::Raycast(query) => Route::Feature(Box::new(query)),
            Self::Overlap(query) => Route::Feature(Box::new(query)),
            Self::ShapeCast(query) => Route::Feature(Box::new(query)),
            Self::CollisionEvents(query) => Route::Feature(Box::new(query)),
            Self::TerrainHeight(query) => Route::Feature(Box::new(query)),
            Self::NavPath(query) => Route::Feature(Box::new(query)),
            Self::WaterQuery(query) => Route::Feature(Box::new(query)),
            Self::DeformationQuery(query) => Route::Feature(Box::new(query)),
            Self::TerrainSweep(query) => Route::Feature(Box::new(query)),
            Self::TerrainRoute(query) => Route::Feature(Box::new(query)),
        }
    }
}

/// Host controls available alongside read-only [`DevRequest`] operations.
/// The browser host enforces pause/lifecycle rules. Parsing validates
/// structure and the protocol version; the host also validates finite values,
/// ranges, and u64 revision parsing before changing state.
#[expect(
    clippy::large_enum_variant,
    reason = "one short-lived request at a time: the environment variant's size does not matter, and boxing its sections would complicate the typed schema"
)]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    derive(schemars::JsonSchema),
    schemars(transform = with_version)
)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum BrowserControl {
    /// Preview fixed-tick poses at a chosen fraction without advancing clocks.
    /// Requires pause; ordinary step/render/capture retain authoritative poses.
    Present {
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::ALPHA)))]
        alpha: f64,
        /// Present through GPU blending/composition/culling
        /// instead of the exact CPU path; for comparisons.
        #[serde(default)]
        gpu: bool,
    },
    /// Temporarily toggle visibility culling and render without advancing clocks.
    Culling {
        enabled: bool,
    },
    /// Temporarily replace environment sections and render while paused.
    /// Sections are a scene `environment`'s, validated alike; omitted
    /// sections stay. A scene with an environment restores its own on
    /// reload; persist a look by saving it there.
    Environment(EnvironmentControl),
    Terrain(TerrainControl),
    TerrainEdit(TerrainEditControl),
    TerrainUndo(TerrainUndoControl),
    TerrainRedo(TerrainRedoControl),
    Deform(DeformControl),
    NavGo(NavGoControl),
    Animation(AnimationControl),
    WorldChanges(WorldChangesControl),
    /// Set a complete temporary world-space camera and render while paused.
    /// Its values keep the rules of scenes' cameras (`SceneCamera`), the
    /// field of view in radians.
    Camera {
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(transform = stated(engine_settings::rule::same(engine_core::Camera::rule_of("position"))))
        )]
        position: [f32; 3],
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(transform = stated(engine_settings::rule::same(engine_core::Camera::rule_of("target"))))
        )]
        target: [f32; 3],
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(transform = stated(engine_settings::rule::same(engine_core::Camera::rule_of("vertical_fov_radians"))))
        )]
        vertical_fov_radians: f32,
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(transform = stated(engine_settings::rule::same(engine_core::Camera::rule_of("near"))))
        )]
        near: f32,
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(transform = stated(engine_settings::rule::same(engine_core::Camera::rule_of("far"))))
        )]
        far: f32,
    },
    /// RAF-paced host sampling, bounded to ten seconds; advances one tick per
    /// sample in step mode and no gameplay in render/present modes. Present mode
    /// samples existing fixed-tick history at alpha 0.5. Clears pending input.
    Benchmark {
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
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::TICKS)))]
        ticks: u32,
    },
    /// Run `count` ordinary frames `interval_ms` apart while paused, as play
    /// runs them after a pause: the first advances no time, each later one
    /// `interval_ms`, without input (pending input is cleared), presented
    /// and antialiased as play's are; then stay paused. With `capture`,
    /// the host captures the last frame as it was drawn, without drawing
    /// again, as `capture` keeps its files.
    Frames {
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::FRAMES)))]
        count: u32,
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::FRAME_INTERVAL)))]
        interval_ms: f64,
        #[serde(default)]
        capture: bool,
    },
    /// Replace the temporary batch applied on the next explicit step.
    Input {
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
    Render {},
    Pause {},
    Resume {},
    /// Render while paused and wait for presentation before host capture.
    /// Omitted alpha renders authoritative current state; a supplied fraction
    /// previews existing fixed-tick history without advancing gameplay.
    Capture {
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
    /// Panics the engine, as a bug in it would, to check what the host does
    /// then: the engine stops for good, and every later request answers
    /// `runtime_panicked` with the panic's message and where it happened.
    DebugPanic {},
    /// Fails the renderer's lazily created feature `feature` as a GPU error
    /// would: its next creation makes an invalid object, which its first
    /// frame uses. The renderer disables it alone; then renders.
    DebugFault {
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(regex(pattern = r"^[a-z_]{1,64}$"))
        )]
        feature: String,
    },
}

impl BrowserControl {
    /// The control as the host runs it: the one routing table of
    /// controls. Every field of the host's own is named, so one added
    /// reaches the host or fails to build.
    pub fn route(self) -> Control {
        match self {
            Self::Present { alpha, gpu } => Control::Host(HostControl::Present { alpha, gpu }),
            Self::Culling { enabled } => Control::Host(HostControl::Culling { enabled }),
            Self::Environment(control) => Control::Host(HostControl::Environment(control)),
            Self::Terrain(control) => Control::Feature(Box::new(control)),
            Self::TerrainEdit(control) => Control::Feature(Box::new(control)),
            Self::TerrainUndo(control) => Control::Feature(Box::new(control)),
            Self::TerrainRedo(control) => Control::Feature(Box::new(control)),
            Self::Deform(control) => Control::Feature(Box::new(control)),
            Self::NavGo(control) => Control::Feature(Box::new(control)),
            Self::Animation(control) => Control::Feature(Box::new(control)),
            Self::WorldChanges(control) => Control::Feature(Box::new(control)),
            Self::Camera {
                position,
                target,
                vertical_fov_radians,
                near,
                far,
            } => Control::Host(HostControl::Camera {
                position,
                target,
                vertical_fov_radians,
                near,
                far,
            }),
            Self::Step { ticks } => Control::Host(HostControl::Step { ticks }),
            Self::Input {
                forward,
                right,
                up,
                yaw,
                pitch,
                held,
                pressed,
                released,
            } => Control::Host(HostControl::Input {
                forward,
                right,
                up,
                yaw,
                pitch,
                held,
                pressed,
                released,
            }),
            Self::Render {} => Control::Host(HostControl::Render),
            Self::DebugPanic {} => Control::Host(HostControl::DebugPanic),
            Self::DebugFault { feature } => Control::Host(HostControl::DebugFault { feature }),
            control @ (Self::Benchmark { .. }
            | Self::Frames { .. }
            | Self::Pause { .. }
            | Self::Resume { .. }
            | Self::Capture { .. }
            | Self::Game { .. }
            | Self::Reload { .. }) => Control::Host(HostControl::Loop(control)),
        }
    }

    /// Whether the control needs a paused host: every one but pausing and
    /// resuming.
    pub fn needs_pause(&self) -> bool {
        match self {
            Self::Present { .. }
            | Self::Culling { .. }
            | Self::Environment(..)
            | Self::Terrain(_)
            | Self::TerrainEdit(_)
            | Self::TerrainUndo(_)
            | Self::TerrainRedo(_)
            | Self::Deform(_)
            | Self::WorldChanges(_)
            | Self::NavGo(_)
            | Self::Animation(_)
            | Self::Camera { .. }
            | Self::Benchmark { .. }
            | Self::Step { .. }
            | Self::Frames { .. }
            | Self::Input { .. }
            | Self::Render { .. }
            | Self::Capture { .. }
            | Self::Game { .. }
            | Self::Reload { .. }
            | Self::DebugPanic { .. }
            | Self::DebugFault { .. } => true,
            Self::Pause { .. } | Self::Resume { .. } => false,
        }
    }
}

/// engine_dev's part of [`engine_core::environment_sections`]: the
/// environment control's request.
macro_rules! control_sections {
    ($($key:ident: $settings:ident { opt_in: $opt_in:literal, feature: $feature:ident }),* $(,)?) => {
        /// The environment control: the sections it replaces, each in its
        /// scene environment shape.
        #[derive(Debug, Deserialize, Serialize)]
        #[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
        #[serde(deny_unknown_fields)]
        pub struct EnvironmentControl {
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
