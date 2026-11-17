use engine_scene::SceneEnvironment;
use engine_settings::{Error, ErrorCode, Expected};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

use engine_settings::stated;

pub(crate) use bounds::default_limit;

pub mod bounds {
    use engine_settings::Rule;

    pub const MAX_PAGE_SIZE: usize = 100;
    pub const MAX_SCAN_ENTITIES: usize = 10_000;
    pub const PAGE: Rule = Rule::Range(1.0, MAX_PAGE_SIZE as f64);
    pub const COORDINATE: Rule = Rule::Range(-1.0e6, 1.0e6);

    pub fn default_limit() -> usize {
        20
    }

    pub const ALPHA: Rule = Rule::Range(0.0, 1.0);
    pub const TICKS: Rule = Rule::AtLeast(1.0);
    pub const FRAMES: Rule = Rule::Range(1.0, 240.0);
    pub const FRAME_INTERVAL: Rule = Rule::Range(1.0, 1000.0);
    pub const TURN: Rule = Rule::Range(-1000.0, 1000.0);
}

pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_RESPONSE_BYTES: usize = 64 * 1024;

pub const CONTROL_STATE: &[(&str, &str)] = &[
    ("present", "PresentationState"),
    ("environment", "Environment"),
    ("camera", "GlobalTransform"),
    ("step", "PresentationState"),
    ("frames", "PresentationState"),
    ("input", "Devices"),
    ("render", ""),
    ("pause", "status"),
    ("resume", "status"),
    ("game", "GameControls"),
    ("reload", "status"),
    ("parent", ""),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Verb {
    Describe,
    Inspect,
    Query,
    Control,
    Capture,
    Profile,
}

impl Verb {
    pub const ALL: [Self; 6] = [
        Self::Describe,
        Self::Inspect,
        Self::Query,
        Self::Control,
        Self::Capture,
        Self::Profile,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Describe => "describe",
            Self::Inspect => "inspect",
            Self::Query => "query",
            Self::Control => "control",
            Self::Capture => "capture",
            Self::Profile => "profile",
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

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize, schemars::JsonSchema,
)]
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
    let Version { version } =
        engine_settings::parse_value(serde_json::json!({ "version": version }))?;
    if version != PROTOCOL_VERSION {
        return Err(Error::new(ErrorCode::UnsupportedVersion, "$.version")
            .got(version)
            .range(accepted, accepted));
    }
    Ok(())
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Query {
    Status {},
    Assets {},
    Digest {},
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DescribeRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub definition: Option<String>,
    #[serde(default)]
    pub detail: Detail,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
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
    #[schemars(transform = stated(bounds::PAGE))]
    pub limit: usize,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CaptureRequest {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_alpha"
    )]
    #[schemars(with = "f64", transform = stated(bounds::ALPHA))]
    pub alpha: Option<f64>,
    #[serde(default)]
    pub digest: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compare: Option<CompareRequest>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CompareRequest {
    #[schemars(length(min = 1, max = 512))]
    pub a: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub b: Option<String>,
    #[serde(default)]
    pub tolerance: u8,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProfileRequest {
    #[serde(default = "default_profile_frames")]
    #[schemars(transform = stated(bounds::FRAMES))]
    pub frames: u32,
    #[serde(default = "default_interval_ms")]
    #[schemars(transform = stated(bounds::FRAME_INTERVAL))]
    pub interval_ms: f64,
    #[serde(default = "default_top")]
    #[schemars(transform = stated(bounds::PAGE))]
    pub top: usize,
    #[serde(default)]
    pub detail: Detail,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deep: Option<DeepProfile>,
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeepProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_events: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_samples: Option<bool>,
}

impl Query {
    pub const NAMES: [&'static str; 3] = ["status", "assets", "digest"];
}

pub enum Control {
    Operation(crate::controls::OperationCall),
    Host(BrowserControl),
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PointerControl {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "[f32; 2]", transform = stated(bounds::COORDINATE))]
    pub position_pixels: Option<[f32; 2]>,
    #[serde(default)]
    #[schemars(transform = stated(bounds::TURN))]
    pub motion_pixels: [f32; 2],
    #[serde(default)]
    #[schemars(transform = stated(bounds::TURN))]
    pub wheel_notches: [f32; 2],
    #[serde(default)]
    pub buttons: Vec<engine_core::MouseButton>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TouchControl {
    pub id: u32,
    #[schemars(transform = stated(bounds::COORDINATE))]
    pub position_pixels: [f32; 2],
    #[serde(default)]
    #[schemars(transform = stated(bounds::TURN))]
    pub motion_pixels: [f32; 2],
    pub phase: engine_core::TouchPhase,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum BrowserControl {
    Present {
        #[schemars(transform = stated(bounds::ALPHA))]
        alpha: f64,
    },
    Environment(EnvironmentControl),
    Camera {
        #[schemars(transform = stated(bounds::COORDINATE))]
        position: [f32; 3],
        #[schemars(transform = stated(bounds::COORDINATE))]
        target: [f32; 3],
        #[schemars(transform = stated(camera_fov()))]
        vertical_fov_radians: f32,
        #[schemars(transform = stated(camera_depth()))]
        near: f32,
        #[schemars(transform = stated(camera_depth()))]
        far: f32,
    },
    Step {
        #[schemars(transform = stated(bounds::TICKS))]
        ticks: u32,
    },
    Frames {
        #[schemars(transform = stated(bounds::FRAMES))]
        count: u32,
        #[schemars(transform = stated(bounds::FRAME_INTERVAL))]
        interval_ms: f64,
        #[serde(default)]
        capture: bool,
    },
    Input {
        #[serde(default)]
        keys: Vec<String>,
        #[serde(default)]
        pointer: PointerControl,
        #[serde(default)]
        touches: Vec<TouchControl>,
    },
    Render {},
    Pause {},
    Resume {},
    Game {
        #[schemars(regex(pattern = r"^[a-z][a-z0-9_]{0,63}$"))]
        name: String,
        #[serde(default)]
        args: serde_json::Map<String, serde_json::Value>,
    },
    Reload {
        #[schemars(regex(pattern = r"^(0|[1-9][0-9]*)$"))]
        expected_revision: String,
        #[schemars(regex(pattern = r"^(0|[1-9][0-9]*)$"))]
        revision: String,
    },
    Parent {
        #[schemars(length(min = 1, max = 192))]
        entity: String,
        #[schemars(length(min = 1, max = 192))]
        parent: String,
    },
}

impl BrowserControl {
    pub const NAMES: [&'static str; 12] = [
        "present",
        "environment",
        "camera",
        "step",
        "frames",
        "input",
        "render",
        "pause",
        "resume",
        "game",
        "reload",
        "parent",
    ];

    pub fn route(self) -> Control {
        match self {
            Self::Parent { entity, parent } => {
                Control::Operation(crate::controls::OperationCall::Parent { entity, parent })
            }
            Self::Environment(control) => {
                Control::Operation(crate::controls::OperationCall::Environment(control))
            }
            control @ (Self::Present { .. }
            | Self::Camera { .. }
            | Self::Step { .. }
            | Self::Input { .. }
            | Self::Render {}
            | Self::Frames { .. }
            | Self::Pause { .. }
            | Self::Resume { .. }
            | Self::Game { .. }
            | Self::Reload { .. }) => Control::Host(control),
        }
    }

    pub fn needs_pause(&self) -> bool {
        match self {
            Self::Present { .. }
            | Self::Environment(..)
            | Self::Camera { .. }
            | Self::Step { .. }
            | Self::Frames { .. }
            | Self::Input { .. }
            | Self::Render { .. }
            | Self::Game { .. }
            | Self::Reload { .. }
            | Self::Parent { .. } => true,
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

fn default_profile_frames() -> u32 {
    60
}

fn default_interval_ms() -> f64 {
    1000.0 / 60.0
}

pub(crate) fn default_top() -> usize {
    8
}

fn deserialize_optional_alpha<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<f64>, D::Error> {
    f64::deserialize(deserializer).map(Some)
}

fn camera_fov() -> engine_settings::Rule {
    engine_settings::Rule::range(
        f64::from(engine_core::Camera::MIN_FOV_DEGREES.to_radians()),
        f64::from(engine_core::Camera::MAX_FOV_DEGREES.to_radians()),
    )
}

fn camera_depth() -> engine_settings::Rule {
    engine_settings::Rule::range(
        f64::from(engine_core::Camera::MIN_DEPTH),
        f64::from(engine_core::Camera::REACH),
    )
}
