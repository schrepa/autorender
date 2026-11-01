use engine_scene::SceneEnvironment;
use engine_settings::{Error, ErrorCode, Expected};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

#[cfg(not(target_arch = "wasm32"))]
use engine_settings::stated;

pub use engine_core::bounds;
pub(crate) use engine_core::bounds::default_limit;

pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_REQUEST_BYTES: usize = 16 * 1024;
pub const MAX_RESPONSE_BYTES: usize = 64 * 1024;
pub const MAX_WATCHED: usize = 32;
pub const MAX_EXPLAINED: usize = 64;

pub const CONTROL_STATE: &[(&str, &str)] = &[
    ("present", "PresentationState"),
    ("culling", "status"),
    ("environment", "Environment"),
    ("camera", "Camera"),
    ("benchmark", ""),
    ("step", "PresentationState"),
    ("frames", "PresentationState"),
    ("input", "Input"),
    ("render", ""),
    ("pause", "status"),
    ("resume", "status"),
    ("game", "GameControls"),
    ("reload", "status"),
    ("debug_panic", "status"),
    ("debug_fault", "status"),
    ("debug_device_loss", "status"),
    ("debug_missing_asset", ""),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum Verb {
    Describe,
    Inspect,
    Query,
    Control,
    Capture,
    Profile,
    Explain,
    Watch,
    Check,
}

impl Verb {
    pub const ALL: [Self; 9] = [
        Self::Describe,
        Self::Inspect,
        Self::Query,
        Self::Control,
        Self::Capture,
        Self::Profile,
        Self::Explain,
        Self::Watch,
        Self::Check,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Describe => "describe",
            Self::Inspect => "inspect",
            Self::Query => "query",
            Self::Control => "control",
            Self::Capture => "capture",
            Self::Profile => "profile",
            Self::Explain => "explain",
            Self::Watch => "watch",
            Self::Check => "check",
        }
    }

    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|verb| verb.name() == name)
    }

    pub const fn names_operations(self) -> bool {
        matches!(self, Self::Query | Self::Control)
    }

    pub fn names() -> impl Iterator<Item = &'static str> {
        Self::ALL.into_iter().map(Self::name)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum Detail {
    #[default]
    Short,
    Full,
}

#[derive(Clone, Debug)]
pub struct Envelope {
    pub verb: Verb,
    pub fields: Map<String, Value>,
}

impl Envelope {
    pub fn parse(source: &str) -> Result<Self, Error> {
        if source.len() > MAX_REQUEST_BYTES {
            return Err(Error::new(ErrorCode::RequestTooLarge, "$")
                .at_most(MAX_REQUEST_BYTES as u64)
                .got(source.len()));
        }
        let request: Value = engine_settings::parse_json(source)?;
        let Value::Object(mut fields) = request else {
            return Err(Error::new(ErrorCode::InvalidType, "$").expected(Expected::Object));
        };
        check_version(fields.remove("version"))?;
        let verb = match fields.remove("verb") {
            None => {
                return Err(Error::new(ErrorCode::MissingField, "$.verb")
                    .expected(Expected::Present)
                    .variants(Verb::names()));
            }
            Some(Value::String(name)) => Verb::named(&name).ok_or_else(|| {
                Error::new(ErrorCode::UnknownVariant, "$.verb")
                    .got(&name)
                    .variants(Verb::names())
            })?,
            Some(other) => {
                return Err(Error::new(ErrorCode::InvalidType, "$.verb")
                    .expected(Expected::String)
                    .got(other)
                    .variants(Verb::names()));
            }
        };
        Ok(Self { verb, fields })
    }

    pub fn name(&self) -> Result<&str, Error> {
        match self.fields.get("name") {
            Some(Value::String(name)) => Ok(name),
            Some(other) => Err(Error::new(ErrorCode::InvalidType, "$.name")
                .expected(Expected::String)
                .got(other)),
            None => Err(Error::new(ErrorCode::MissingField, "$.name").expected(Expected::Present)),
        }
    }

    pub fn args(&self) -> Result<Value, Error> {
        let extra = self
            .fields
            .keys()
            .find(|key| !["name", "args"].contains(&key.as_str()));
        if let Some(extra) = extra {
            return Err(Error::new(ErrorCode::UnknownField, format!("$.{extra}"))
                .expected(Expected::Absent)
                .got(extra)
                .variants(["name", "args"]));
        }
        match self.fields.get("args") {
            None => Ok(Value::Object(Map::new())),
            Some(args @ Value::Object(_)) => Ok(args.clone()),
            Some(other) => Err(Error::new(ErrorCode::InvalidType, "$.args")
                .expected(Expected::Object)
                .got(other)),
        }
    }

    pub fn body<T: DeserializeOwned>(&self) -> Result<T, Error> {
        engine_settings::parse_value(Value::Object(self.fields.clone()))
    }
}

pub fn request(verb: Verb, fields: Value) -> String {
    let mut request = match fields {
        Value::Object(fields) => fields,
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) | Value::Array(_) => {
            Map::new()
        }
    };
    request.insert("verb".to_owned(), Value::from(verb.name()));
    request.insert("version".to_owned(), Value::from(PROTOCOL_VERSION));
    Value::Object(request).to_string()
}

pub fn named(verb: Verb, name: &str, args: Value) -> String {
    request(verb, serde_json::json!({ "name": name, "args": args }))
}

pub(crate) fn parse_named<T: DeserializeOwned>(name: &str, args: Value) -> Result<T, Error> {
    let mut fields = match args {
        Value::Object(fields) => fields,
        other @ (Value::Null
        | Value::Bool(_)
        | Value::Number(_)
        | Value::String(_)
        | Value::Array(_)) => {
            return Err(Error::new(ErrorCode::InvalidType, "$.args")
                .expected(Expected::Object)
                .got(other));
        }
    };
    fields.insert("op".to_owned(), Value::from(name));
    engine_settings::parse_value(Value::Object(fields)).map_err(|mut error| {
        let path = match error.path.strip_prefix('$') {
            Some("" | ".op") => "$.args".to_owned(),
            Some(rest) => format!("$.args{rest}"),
            None => std::mem::take(&mut error.path),
        };
        error.path = path;
        error
    })
}

fn check_version(version: Option<Value>) -> Result<(), Error> {
    #[derive(Deserialize)]
    struct Version {
        version: u32,
    }
    let accepted = f64::from(PROTOCOL_VERSION);
    let Some(version) = version else {
        return Err(Error::new(ErrorCode::MissingField, "$.version")
            .expected(Expected::Present)
            .range(accepted, accepted));
    };
    let Version { version } = engine_settings::parse_value(serde_json::json!({ "version": version }))?;
    if version != PROTOCOL_VERSION {
        return Err(Error::new(ErrorCode::UnsupportedVersion, "$.version")
            .got(version)
            .range(accepted, accepted));
    }
    Ok(())
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Query {
    Status {},
    Assets {},
    Digest {},
    Distribution {
        #[cfg_attr(not(target_arch = "wasm32"), schemars(length(max = bounds::MAX_VALUES)))]
        values: Vec<f64>,
    },
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct DescribeRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub definition: Option<String>,
    #[serde(default)]
    pub detail: Detail,
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct InspectRequest {
    #[serde(default)]
    pub entity: Option<String>,
    #[serde(default)]
    pub parent: Option<String>,
    #[serde(default)]
    pub component: Option<String>,
    #[serde(default)]
    pub offset: usize,
    #[serde(default = "default_limit")]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::PAGE)))]
    pub limit: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct CaptureRequest {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_alpha"
    )]
    #[cfg_attr(
        not(target_arch = "wasm32"),
        schemars(with = "f64", transform = stated(bounds::ALPHA))
    )]
    pub alpha: Option<f64>,
    #[serde(default)]
    pub digest: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compare: Option<CompareRequest>,
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct CompareRequest {
    #[cfg_attr(not(target_arch = "wasm32"), schemars(length(min = 1, max = 512)))]
    pub a: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub b: Option<String>,
    #[serde(default)]
    pub tolerance: u8,
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct ProfileRequest {
    #[serde(default = "default_profile_frames")]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::FRAMES)))]
    pub frames: u32,
    #[serde(default = "default_interval_ms")]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::FRAME_INTERVAL)))]
    pub interval_ms: f64,
    #[serde(default = "default_top")]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::PAGE)))]
    pub top: usize,
    #[serde(default)]
    pub detail: Detail,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deep: Option<DeepProfile>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct DeepProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_events: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_samples: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor: Option<Vendor>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum Vendor {
    Pix,
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(tag = "about", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExplainQuestion {
    Entity {
        #[cfg_attr(not(target_arch = "wasm32"), schemars(length(min = 1, max = 64)))]
        entity: String,
        #[serde(default)]
        detail: Detail,
    },
    Frame {
        #[serde(default = "default_top")]
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::PAGE)))]
        top: usize,
        #[serde(default)]
        detail: Detail,
    },
    Scene {},
}

impl Query {
    pub const NAMES: [&'static str; 4] = ["status", "assets", "digest", "distribution"];
}

impl ExplainQuestion {
    pub fn needs_pause(&self) -> bool {
        matches!(self, Self::Entity { .. })
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct WatchRequest {
    #[serde(default)]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(length(max = MAX_WATCHED)))]
    pub stats: Vec<String>,
    #[serde(default)]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(length(max = MAX_WATCHED)))]
    pub entities: Vec<WatchedComponent>,
    #[serde(default = "default_watch_frames")]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::FRAMES)))]
    pub frames: u32,
    #[serde(default = "default_interval_ms")]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::FRAME_INTERVAL)))]
    pub interval_ms: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct WatchedComponent {
    #[cfg_attr(not(target_arch = "wasm32"), schemars(length(min = 1, max = 64)))]
    pub entity: String,
    #[cfg_attr(not(target_arch = "wasm32"), schemars(length(min = 1, max = 64)))]
    pub component: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct CheckRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<CheckLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum CheckLevel {
    Quick,
    Pr,
    Merge,
}

impl CheckLevel {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Quick => "quick",
            Self::Pr => "pr",
            Self::Merge => "merge",
        }
    }
}

pub enum Control {
    Operation(crate::controls::OperationCall),
    Host(HostControl),
}

pub enum HostControl {
    Present {
        alpha: f64,
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
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum BrowserControl {
    Present {
        #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::ALPHA)))]
        alpha: f64,
    },
    Culling {
        enabled: bool,
    },
    Environment(EnvironmentControl),
    Camera {
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(transform = stated(bounds::COORDINATE))
        )]
        position: [f32; 3],
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(transform = stated(bounds::COORDINATE))
        )]
        target: [f32; 3],
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(transform = stated(camera_fov()))
        )]
        vertical_fov_radians: f32,
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(transform = stated(camera_depth()))
        )]
        near: f32,
        #[cfg_attr(
            not(target_arch = "wasm32"),
            schemars(transform = stated(camera_depth()))
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
}

impl BrowserControl {
    pub const NAMES: [&'static str; 17] = [
        "present",
        "culling",
        "environment",
        "camera",
        "benchmark",
        "step",
        "frames",
        "input",
        "render",
        "pause",
        "resume",
        "game",
        "reload",
        "debug_panic",
        "debug_fault",
        "debug_device_loss",
        "debug_missing_asset",
    ];

    pub fn route(self) -> Control {
        match self {
            Self::Present { alpha } => Control::Host(HostControl::Present { alpha }),
            Self::Culling { enabled } => Control::Host(HostControl::Culling { enabled }),
            Self::Environment(control) => Control::Host(HostControl::Environment(control)),
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
                Control::Operation(crate::controls::OperationCall::MissingAsset { entity })
            }
            control @ (Self::Benchmark { .. }
            | Self::Frames { .. }
            | Self::Pause { .. }
            | Self::Resume { .. }
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
            | Self::Camera { .. }
            | Self::Benchmark { .. }
            | Self::Step { .. }
            | Self::Frames { .. }
            | Self::Input { .. }
            | Self::Render { .. }
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

#[derive(Debug, Deserialize, Serialize)]
pub struct EnvironmentControl {
    #[serde(default)]
    pub merge: bool,
    #[serde(flatten)]
    pub sections: BTreeMap<String, serde_json::Value>,
}

impl EnvironmentControl {
    pub fn sections(&self) -> SceneEnvironment {
        SceneEnvironment {
            sections: self.sections.clone(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl schemars::JsonSchema for EnvironmentControl {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "EnvironmentControl".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "object",
            "properties": {
                "merge": { "type": "boolean", "default": false },
            },
            "additionalProperties": false,
        })
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

fn default_samples() -> u32 {
    120
}

fn default_profile_frames() -> u32 {
    60
}

fn default_watch_frames() -> u32 {
    60
}

fn default_interval_ms() -> f64 {
    1000.0 / 60.0
}

pub(crate) fn default_top() -> usize {
    8
}

#[cfg(not(target_arch = "wasm32"))]
fn camera_fov() -> engine_settings::Rule {
    engine_settings::Rule::range(
        f64::from(engine_core::Camera::MIN_FOV_DEGREES.to_radians()),
        f64::from(engine_core::Camera::MAX_FOV_DEGREES.to_radians()),
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn camera_depth() -> engine_settings::Rule {
    engine_settings::Rule::range(
        f64::from(engine_core::Camera::MIN_DEPTH),
        f64::from(engine_core::Camera::REACH),
    )
}
