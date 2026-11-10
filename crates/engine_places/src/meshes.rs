use bevy_ecs::prelude::{Component, Resource};
use engine_place::{DrawKind, MaterialHandle, MeshHandle, Slot};

pub const MESH: DrawKind = DrawKind("mesh");
pub const SKIN: DrawKind = DrawKind("skin");

pub const LAST_LEVEL: u32 = 3;
pub const DEBUG_LEVEL_AT: u32 = 8;
pub const DEBUG_LEVEL_MASK: u32 = 15;
pub const DEBUG_FADING: u32 = 1 << 12;

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Instances;

impl Slot for Instances {
    const NAME: &'static str = "instances";
    const WGSL: Option<&'static str> = Some(crate::contract!(
        "places/instances.wgsl",
        LAST_LEVEL,
        DEBUG_LEVEL_AT,
        DEBUG_LEVEL_MASK,
        DEBUG_FADING,
    ));
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelBatch {
    pub kind: DrawKind,
    pub mesh: MeshHandle,
    pub material: Option<MaterialHandle>,
}

#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct LevelBatches {
    pub generation: u64,
    pub batches: Vec<LevelBatch>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LastLevel {
    pub error: f32,
    pub triangles: u32,
}

#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct LastLevels {
    pub generation: u64,
    pub levels: Vec<LastLevel>,
}
