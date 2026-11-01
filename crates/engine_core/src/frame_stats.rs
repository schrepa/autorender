use std::collections::BTreeMap;

use serde::{Serialize, Serializer};

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct FrameStats {
    pub frame: u64,
    pub tick: u64,
    pub simulation_ms: f64,
    pub extraction_ms: f64,
    pub render_ms: f64,
    pub fixed_steps: u32,
    pub dropped_seconds: f64,
    pub pipelines_compiling: bool,
    pub pipelines_listed: u32,
    pub pipelines_unlisted: u32,
    pub loading: bool,
    pub resolution: ResolutionStats,
    pub dynamic_resolution: DynamicResolutionStats,
    pub jobs: crate::JobStats,
    pub disabled_features: Names,
    #[serde(flatten)]
    pub render: RenderStats,
    pub bricks_cost: Option<engine_registry::BrickCosts>,
    #[serde(flatten)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    pub bricks: BTreeMap<&'static str, serde_json::Value>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
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
    #[serde(skip)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    pub submitted: bool,
    pub pipelines: u32,
    pub pipelines_created: u32,
    pub render_passes: Names,
    pub linked: LinkedStats,
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
        let faces = generator.subschema_for::<BTreeMap<String, String>>();
        let nobody = generator.subschema_for::<Vec<String>>();
        let lists = generator.subschema_for::<BTreeMap<String, Vec<String>>>();
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
    pub bits: u64,
    pub names: &'static [&'static str],
}

impl Serialize for Names {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(
            self.names
                .iter()
                .enumerate()
                .filter(|(index, _)| self.bits & (1 << index) != 0)
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ResolutionStats {
    pub display_width: u32,
    pub display_height: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub dropped_pixels: u64,
    pub gpu_ms: Option<f32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicResolutionStats {
    pub active: bool,
    pub scale: f32,
    pub budget_ms: f32,
}
