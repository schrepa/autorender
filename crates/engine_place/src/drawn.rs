use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use engine_assets::Attributes;
use serde::{Deserialize, Serialize};

use crate::{MaterialHandle, MeshHandle, PassMask};

pub type NodeId = Entity;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct DrawKind(pub &'static str);

impl DrawKind {
    pub const MESH: Self = Self("mesh");
    pub const SKIN: Self = Self("skin");

    pub const fn name(self) -> &'static str {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct KindDecl {
    pub kind: DrawKind,
    pub vertex: &'static str,
    pub needs: Attributes,
    pub casts_shadows: bool,
    pub seen_by_probes: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionReach {
    pub module: &'static str,
    pub reach: f32,
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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DrawnProjection {
    pub vertical_fov: f32,
    pub depth_scale: f32,
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

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct DrawnBounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Component, Clone, Debug, PartialEq)]
pub struct Drawn {
    pub kind: DrawKind,
    pub geometry: Option<MeshHandle>,
    pub material: Option<MaterialHandle>,
    pub node: NodeId,
    pub passes: Option<PassMask>,
    pub layers: Layers,
    pub order: i16,
    pub projection: Option<DrawnProjection>,
    pub values: InstanceValues,
}

impl Drawn {
    pub fn new(
        kind: DrawKind,
        geometry: MeshHandle,
        material: MaterialHandle,
        node: NodeId,
    ) -> Self {
        Self {
            geometry: Some(geometry),
            material: Some(material),
            ..Self::procedural(kind, node)
        }
    }

    pub fn procedural(kind: DrawKind, node: NodeId) -> Self {
        Self {
            kind,
            geometry: None,
            material: None,
            node,
            passes: None,
            layers: Layers::FIRST,
            order: 0,
            projection: None,
            values: InstanceValues::default(),
        }
    }
}
