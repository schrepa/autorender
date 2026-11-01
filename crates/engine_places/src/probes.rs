use bevy_ecs::prelude::{Component, Resource};
use engine_place::{DrawPoint, Slot};
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Interior {
    pub sky: f32,
    pub sun: f32,
}

impl Default for Interior {
    fn default() -> Self {
        Self { sky: 1.0, sun: 1.0 }
    }
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReflectionProbe {
    pub extent: Vec3,
    #[serde(default = "full_blend")]
    pub blend: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interior: Option<Interior>,
}

fn full_blend() -> f32 {
    1.0
}

impl ReflectionProbe {
    pub fn new(extent: Vec3) -> Self {
        Self {
            extent,
            blend: 1.0,
            interior: None,
        }
    }

    #[must_use]
    pub fn with_blend(mut self, blend: f32) -> Self {
        self.blend = blend;
        self
    }

    #[must_use]
    pub fn with_interior(mut self, interior: Interior) -> Self {
        self.interior = Some(interior);
        self
    }
}

#[derive(Resource, Default)]
pub struct ProbeViews;

impl Slot for ProbeViews {
    const NAME: &'static str = "probe_views";
}

impl DrawPoint for ProbeViews {
    type Layer = ();
    type Args = ProbeView;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeView {
    pub face: u32,
    pub center: Vec3,
    pub still: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeFace {
    pub face: u32,
    pub center: Vec3,
    pub view: Mat4,
    pub cube: u32,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct ProbeCaptures {
    pub faces: Vec<ProbeFace>,
    pub size: u32,
    pub still: bool,
}

impl Slot for ProbeCaptures {
    const NAME: &'static str = "probe_captures";
}

#[derive(Resource, Default)]
pub struct ProbeLight;

impl Slot for ProbeLight {
    const NAME: &'static str = "probe_light";
    const WGSL: Option<&'static str> = Some(include_str!("places/probe_light.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::PROBE_LIGHT];
}
