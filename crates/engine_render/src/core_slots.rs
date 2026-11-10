use std::any::TypeId;

use bevy_ecs::prelude::Resource;
use engine_core::{DrawKind, List, Slot, TargetName};

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Resolve;

impl Slot for Resolve {
    const NAME: &'static str = "resolve";
    const WGSL: Option<&'static str> = Some(include_str!("places/resolve.wgsl"));
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Lit;

impl Slot for Lit {
    const NAME: &'static str = "lit";
    const WGSL: Option<&'static str> = Some(include_str!("places/lit.wgsl"));
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DebugShown {
    pub owner: Option<TypeId>,
    pub local: u32,
}

pub const DEBUG_TARGETS: [TargetName; 3] = [
    TargetName("debug_ids"),
    TargetName("debug_count"),
    TargetName("debug_words"),
];

pub const DEBUG_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

impl Slot for DebugShown {
    const NAME: &'static str = "debug";
    const WGSL: Option<&'static str> = Some(include_str!("places/debug.wgsl"));
}

pub struct DebugMode {
    pub name: &'static str,
    pub decode: fn(kind: Option<DrawKind>, ids: [u32; 4], color: [f32; 4]) -> serde_json::Value,
    pub fields: fn() -> serde_json::Value,
}

pub struct DebugModes;

impl List for DebugModes {
    const NAME: &'static str = "debug_modes";
    type Member = DebugMode;
    const WGSL: Option<&'static str> = Some(include_str!("places/debug_modes.wgsl"));
}
