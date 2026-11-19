use bevy_ecs::resource::Resource;
use serde::{Serialize, Serializer};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
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
    PipelineCompiling,
    PipelineRefused,
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
            | Self::MissingBindGroup => true,
            Self::GeometryWaiting
            | Self::GeometryFailed
            | Self::MaterialWaiting
            | Self::MaterialFailed
            | Self::PipelineCompiling
            | Self::RowIncomplete => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
pub struct DroppedDraws {
    pub brick: &'static str,
    pub part: &'static str,
    pub kind: Option<&'static str>,
    pub pass: &'static str,
    pub reason: DropReason,
    pub faults: bool,
    pub count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum KindPassState {
    Drew { draws: u32 },
    Dropped { reason: DropReason, draws: u32 },
    Skipped { reason: &'static str },
    Silent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
pub struct KindPass {
    pub brick: &'static str,
    pub kind: &'static str,
    pub pass: &'static str,
    pub state: KindPassState,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
#[component(immutable)]
pub struct DrawReports {
    pub kinds: Vec<KindPass>,
    pub dropped: Vec<DroppedDraws>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "dev", derive(schemars::JsonSchema))]
pub struct RenderStats {
    pub draw_calls: u64,
    pub instances: u64,
    pub upload_bytes: u64,
    pub mesh_upload_bytes: u64,
    pub texture_upload_bytes: u64,
    pub pipelines: u32,
    pub pipelines_created: u32,
    pub render_passes: Names,
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

#[cfg(feature = "dev")]
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
