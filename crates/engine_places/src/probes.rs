use bevy_ecs::prelude::{Component, Resource};
use engine_place::{DrawPoint, Slot};
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

pub const MIRRORED: bool = true;

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
#[serde(from = "AuthoredProbe", into = "AuthoredProbe")]
pub struct ReflectionProbe {
    pub extent: Vec3,
    pub blend: f32,
    pub interior: Option<Interior>,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthoredProbe {
    extent: [f32; 3],
    #[serde(default = "full_blend")]
    blend: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    interior: Option<Interior>,
}

fn full_blend() -> f32 {
    1.0
}

impl From<AuthoredProbe> for ReflectionProbe {
    fn from(authored: AuthoredProbe) -> Self {
        Self {
            extent: Vec3::from_array(authored.extent),
            blend: authored.blend,
            interior: authored.interior,
        }
    }
}

impl From<ReflectionProbe> for AuthoredProbe {
    fn from(probe: ReflectionProbe) -> Self {
        Self {
            extent: probe.extent.to_array(),
            blend: probe.blend,
            interior: probe.interior,
        }
    }
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
    pub views: bool,
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
