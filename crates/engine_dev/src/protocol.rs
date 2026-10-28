use engine_scene::{SceneAtmosphere, SceneEnvironment};
use engine_settings::ErrorCode;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::DevError;

#[cfg(not(target_arch = "wasm32"))]
use engine_settings::stated;

pub mod bounds {
    use engine_settings::Rule;

    pub const MAX_PAGE_SIZE: usize = 100;
    pub const MAX_DESCRIPTION_BYTES: usize = 512;
    pub const PAGE: Rule = Rule::Range(1.0, MAX_PAGE_SIZE as f64);
    pub const MAX_POINTS: usize = 256;
    pub const COORDINATE: Rule = Rule::Range(-1.0e6, 1.0e6);
    pub const DISTANCE: Rule = Rule::Range(0.0, 2.0e6);
    pub const EVENT_OFFSET: Rule = Rule::Range(0.0, 131_072.0);
    pub const SWEEP_EXTENT: Rule = Rule::Range(0.0, 4096.0);
    pub const SWEEP_SAMPLES: Rule = Rule::Range(2.0, 64.0);
    pub const ALPHA: Rule = Rule::Range(0.0, 1.0);
    pub const WARMUP: Rule = Rule::Range(0.0, 120.0);
    pub const SAMPLES: Rule = Rule::Range(1.0, 240.0);
    pub const TICKS: Rule = Rule::Range(1.0, engine_core::MAX_DEV_TICKS as f64);
    pub const FRAMES: Rule = Rule::Range(1.0, 240.0);
    pub const FRAME_INTERVAL: Rule = Rule::Range(1.0, 1000.0);
    pub const AXIS: Rule = Rule::Range(-1.0, 1.0);
    pub const TURN: Rule = Rule::Range(-1000.0, 1000.0);
    pub const MAX_CHANGES: usize = 256;
    pub const MAX_VALUES: usize = 1024;
    pub const PIXEL: Rule = Rule::Range(0.0, 16_384.0);
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn stamp_rule(field: &'static str) -> impl Fn(&mut schemars::Schema) {
    engine_settings::stated(engine_core::Stamp::rule_of(field).expect("stamps declare the field"))
}

use crate::features::{
    EngineControl, EngineQuery,
    collision::{CollisionEventsQuery, OverlapQuery, RaycastQuery, ShapeCastQuery},
    deformation::{DeformControl, DeformationQuery},
    navigation::{NavGoControl, NavPathQuery},
    ocean::WaterQuery,
    terrain::{
        TerrainControl, TerrainEditControl, TerrainHeightQuery, TerrainRedoControl,
        TerrainRiverControl, TerrainRouteQuery, TerrainSweepQuery, TerrainUndoControl,
    },
    world_streaming::WorldChangesControl,
};

pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_REQUEST_BYTES: usize = 16 * 1024;
pub const MAX_RESPONSE_BYTES: usize = 64 * 1024;

pub const CONTROL_STATE: &[(&str, &str)] = &[
    ("present", "PresentationState"),
    ("culling", "status"),
    ("environment", "Environment"),
    ("terrain", "TerrainDebug"),
    ("terrain_edit", "Terrain"),
    ("terrain_undo", "Terrain"),
    ("terrain_redo", "Terrain"),
    ("terrain_river", "Terrain"),
    ("deform", "Tracks"),
    ("nav_go", "NavAgent"),
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
    ("debug_device_loss", "status"),
    ("debug_missing_asset", ""),
    ("probe", ""),
];

pub(crate) enum Route {
    Tools(ToolsQuery),
    Feature(Box<dyn EngineQuery>),
}

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

#[expect(
    clippy::large_enum_variant,
    reason = "one short-lived control at a time, as BrowserControl"
)]
pub enum Control {
    Feature(Box<dyn EngineControl>),
    Host(HostControl),
}

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
    Loop(BrowserControl),
}

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
    Digest {},
    Distribution {
        #[cfg_attr(not(target_arch = "wasm32"), schemars(length(max = bounds::MAX_VALUES)))]
        values: Vec<f64>,
    },
    Describe {
        #[serde(default)]
        name: Option<String>,
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

pub(crate) fn default_limit() -> usize {
    20
}

impl DevRequest {
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
    Present {
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::ALPHA)))]
        alpha: f64,
        #[serde(default)]
        gpu: bool,
    },
    Culling {
        enabled: bool,
    },
    Environment(EnvironmentControl),
    Terrain(TerrainControl),
    TerrainEdit(TerrainEditControl),
    TerrainUndo(TerrainUndoControl),
    TerrainRedo(TerrainRedoControl),
    TerrainRiver(TerrainRiverControl),
    Deform(DeformControl),
    NavGo(NavGoControl),
    WorldChanges(WorldChangesControl),
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
    Step {
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::TICKS)))]
        ticks: u32,
    },
    Frames {
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::FRAMES)))]
        count: u32,
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::FRAME_INTERVAL)))]
        interval_ms: f64,
        #[serde(default)]
        capture: bool,
    },
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
    Render {},
    Pause {},
    Resume {},
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
        #[serde(default)]
        gpu: bool,
    },
    Game {
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(regex(pattern = r"^[a-z][a-z0-9_]{0,63}$"))
        )]
        name: String,
        #[serde(default)]
        args: serde_json::Map<String, serde_json::Value>,
    },
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
    DebugPanic {},
    DebugFault {
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(regex(pattern = r"^[a-z_]{1,64}$"))
        )]
        feature: String,
    },
    DebugDeviceLoss {},
    DebugMissingAsset {
        #[cfg_attr(not(target_arch = "wasm32"), schemars(length(min = 1, max = 192)))]
        entity: String,
    },
    Probe {
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::PIXEL)))]
        x: u32,
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::PIXEL)))]
        y: u32,
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
        #[serde(default)]
        gpu: bool,
    },
}

impl BrowserControl {
    pub fn route(self) -> Control {
        match self {
            Self::Present { alpha, gpu } => Control::Host(HostControl::Present { alpha, gpu }),
            Self::Culling { enabled } => Control::Host(HostControl::Culling { enabled }),
            Self::Environment(control) => Control::Host(HostControl::Environment(control)),
            Self::Terrain(control) => Control::Feature(Box::new(control)),
            Self::TerrainEdit(control) => Control::Feature(Box::new(control)),
            Self::TerrainUndo(control) => Control::Feature(Box::new(control)),
            Self::TerrainRedo(control) => Control::Feature(Box::new(control)),
            Self::TerrainRiver(control) => Control::Feature(Box::new(control)),
            Self::Deform(control) => Control::Feature(Box::new(control)),
            Self::NavGo(control) => Control::Feature(Box::new(control)),
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
            Self::DebugMissingAsset { entity } => {
                Control::Feature(Box::new(crate::controls::MissingAsset { entity }))
            }
            control @ (Self::Benchmark { .. }
            | Self::Frames { .. }
            | Self::Pause { .. }
            | Self::Resume { .. }
            | Self::Capture { .. }
            | Self::Probe { .. }
            | Self::DebugDeviceLoss { .. }
            | Self::Game { .. }
            | Self::Reload { .. }) => Control::Host(HostControl::Loop(control)),
        }
    }

    pub fn needs_pause(&self) -> bool {
        match self {
            Self::Present { .. }
            | Self::Culling { .. }
            | Self::Environment(..)
            | Self::Terrain(_)
            | Self::TerrainEdit(_)
            | Self::TerrainUndo(_)
            | Self::TerrainRedo(_)
            | Self::TerrainRiver(_)
            | Self::Deform(_)
            | Self::WorldChanges(_)
            | Self::NavGo(_)
            | Self::Camera { .. }
            | Self::Benchmark { .. }
            | Self::Step { .. }
            | Self::Frames { .. }
            | Self::Input { .. }
            | Self::Render { .. }
            | Self::Capture { .. }
            | Self::Probe { .. }
            | Self::Game { .. }
            | Self::Reload { .. }
            | Self::DebugPanic { .. }
            | Self::DebugFault { .. }
            | Self::DebugDeviceLoss { .. }
            | Self::DebugMissingAsset { .. } => true,
            Self::Pause { .. } | Self::Resume { .. } => false,
        }
    }
}

macro_rules! control_sections {
    ($($key:ident: $settings:ident { opt_in: $opt_in:literal, feature: $feature:ident }),* $(,)?) => {
        #[derive(Debug, Deserialize, Serialize)]
        #[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
        #[serde(deny_unknown_fields)]
        pub struct EnvironmentControl {
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
