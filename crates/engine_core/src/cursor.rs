use bevy_ecs::prelude::Resource;
use serde::Serialize;

#[derive(Resource, Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[repr(u32)]
pub enum CursorMode {
    Drag = 0,
    #[default]
    Free = 1,
    Captured = 2,
}

impl CursorMode {
    pub const ALL: [(Self, &'static str); 3] = [
        (Self::Drag, "drag"),
        (Self::Free, "free"),
        (Self::Captured, "captured"),
    ];

    pub const fn name(self) -> &'static str {
        Self::ALL[self as usize].1
    }
}
