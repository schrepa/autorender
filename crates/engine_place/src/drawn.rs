use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use engine_assets::{Attributes, StreamShape};
use serde::{Deserialize, Serialize};

use crate::{GeometryHandle, MaterialHandle, PassMask};

pub type NodeId = Entity;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct DrawKind(pub &'static str);

impl DrawKind {
    pub const fn name(self) -> &'static str {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct KindDecl {
    pub kind: DrawKind,
    pub needs: Attributes,
    pub streams: &'static [StreamShape],
    pub passes: &'static [&'static str],
}

impl KindDecl {
    pub fn joins(&self, name: &str) -> bool {
        self.passes.contains(&name)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Layers(pub u32);

impl Layers {
    pub const NONE: Self = Self(0);
    pub const FIRST: Self = Self(1);
    pub const ALL: Self = Self(u32::MAX);

    pub const fn overlaps(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

impl Default for Layers {
    fn default() -> Self {
        Self::FIRST
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct InstanceValues {
    pub tint: [f32; 4],
    pub emission: [f32; 3],
    pub floats: Vec<f32>,
}

impl Default for InstanceValues {
    fn default() -> Self {
        Self {
            tint: [1.0; 4],
            emission: [0.0; 3],
            floats: Vec::new(),
        }
    }
}

#[derive(Component, Clone, Debug, PartialEq)]
#[component(immutable)]
pub struct Drawn {
    pub kind: DrawKind,
    pub geometry: Option<GeometryHandle>,
    pub material: Option<MaterialHandle>,
    pub passes: Option<PassMask>,
    pub layers: Layers,
    pub order: i16,
    pub values: InstanceValues,
}

impl Drawn {
    pub fn new(kind: DrawKind, geometry: GeometryHandle, material: MaterialHandle) -> Self {
        Self {
            geometry: Some(geometry),
            material: Some(material),
            ..Self::procedural(kind)
        }
    }

    pub fn procedural(kind: DrawKind) -> Self {
        Self {
            kind,
            geometry: None,
            material: None,
            passes: None,
            layers: Layers::FIRST,
            order: 0,
            values: InstanceValues::default(),
        }
    }
}
