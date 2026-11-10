use serde::{Serialize, Serializer};

use crate::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum DropReason {
    UnknownPipeline,
    PassMismatch,
    UnfilledHoles,
    GeometryNotResident,
    GeometryNotHeld,
    GeometryWaiting,
    GeometryFailed,
    MaterialNotResident,
    MaterialNotHeld,
    MaterialWaiting,
    MaterialFailed,
    GeometryLacksStreams,
    NoRenderPass,
    PipelineOverLimit,
    PipelineCompiling,
    PipelineRefused,
    MissingInput,
    MissingBindGroup,
    RowIncomplete,
}

impl DropReason {
    pub const fn faults(self) -> bool {
        match self {
            Self::UnknownPipeline
            | Self::PassMismatch
            | Self::UnfilledHoles
            | Self::GeometryNotResident
            | Self::GeometryNotHeld
            | Self::MaterialNotResident
            | Self::MaterialNotHeld
            | Self::GeometryLacksStreams
            | Self::NoRenderPass
            | Self::PipelineRefused
            | Self::MissingInput
            | Self::MissingBindGroup => true,
            Self::GeometryWaiting
            | Self::GeometryFailed
            | Self::MaterialWaiting
            | Self::MaterialFailed
            | Self::PipelineOverLimit
            | Self::PipelineCompiling
            | Self::RowIncomplete => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DroppedDraws {
    pub part: &'static str,
    pub kind: Option<&'static str>,
    pub pass: &'static str,
    pub reason: DropReason,
    pub faults: bool,
    pub count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum KindPassState {
    Drew { draws: u32 },
    Dropped { reason: DropReason, draws: u32 },
    Skipped { reason: &'static str },
    Silent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct KindPass {
    pub kind: &'static str,
    pub pass: &'static str,
    pub state: KindPassState,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RenderStats {
    pub draw_calls: u32,
    pub upload_bytes: u64,
    pub instances: u32,
    #[serde(skip)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    pub gpu_visible: Option<u32>,
    pub triangles: Option<u32>,
    pub read_back_age: Option<u32>,
    pub resident_meshes: u32,
    pub mesh_upload_bytes: u64,
    pub resident_mesh_bytes: u64,
    pub resident_textures: u32,
    pub texture_upload_bytes: u64,
    pub resident_texture_bytes: u64,
    pub pipelines: u32,
    pub pipelines_created: u32,
    pub render_passes: Names,
    pub linked: LinkedStats,
    #[serde(skip_serializing_if = "none_overflowed")]
    #[cfg_attr(feature = "schema", schemars(skip))]
    pub readback_overflows: u64,
    pub kind_passes: Vec<KindPass>,
    pub kind_problems: Vec<Error>,
    pub dropped_draws: Vec<DroppedDraws>,
}

fn none_overflowed(overflows: &u64) -> bool {
    *overflows == 0
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinkedStats {
    pub faces: &'static [(&'static str, &'static str)],
    pub nobody: &'static [&'static str],
    pub lists: &'static [(&'static str, &'static [&'static str])],
}

struct FaceBricks(&'static [(&'static str, &'static str)]);

impl Serialize for FaceBricks {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.iter().copied())
    }
}

struct ListBricks(&'static [(&'static str, &'static [&'static str])]);

impl Serialize for ListBricks {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.iter().copied())
    }
}

#[derive(Serialize)]
struct LinkedJson {
    faces: FaceBricks,
    nobody: &'static [&'static str],
    lists: ListBricks,
}

impl Serialize for LinkedStats {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        LinkedJson {
            faces: FaceBricks(self.faces),
            nobody: self.nobody,
            lists: ListBricks(self.lists),
        }
        .serialize(serializer)
    }
}

#[cfg(feature = "schema")]
impl schemars::JsonSchema for LinkedStats {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> std::borrow::Cow<'static, str> {
        "LinkedStats".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let faces = generator.subschema_for::<std::collections::BTreeMap<String, String>>();
        let nobody = generator.subschema_for::<Vec<String>>();
        let lists = generator.subschema_for::<std::collections::BTreeMap<String, Vec<String>>>();
        schemars::json_schema!({
            "type": "object",
            "properties": {
                "faces": faces,
                "nobody": nobody,
                "lists": lists,
            },
            "required": ["faces", "nobody", "lists"],
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Names {
    pub bits: u128,
    pub names: &'static [&'static str],
}

impl Serialize for Names {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(
            self.names
                .iter()
                .enumerate()
                .filter(|(index, _)| {
                    self.bits & 1_u128.checked_shl(*index as u32).unwrap_or(0) != 0
                })
                .map(|(_, name)| name),
        )
    }
}

#[cfg(feature = "schema")]
impl schemars::JsonSchema for Names {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Names".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        <Vec<String>>::json_schema(generator)
    }
}
