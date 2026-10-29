use std::sync::{Mutex, PoisonError};

use bevy_ecs::{
    prelude::{Component, Entity, Resource, World},
    query::QueryState,
};
use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::bricks::{Brick, Bricks};
use crate::extraction::Nearest;
use crate::sections::environment_section;
use crate::{GlobalTransforms, RenderSnapshot, TransformIndex};
use engine_settings::{InvalidSetting, Rule, settings};

pub struct ReflectionProbes;

impl Brick for ReflectionProbes {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<ReflectionProbeSettings>()
            .resource::<ProbeExtractor>()
            .extract(extract_probes)
            .passes(&["probe_captures"]);
    }
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct ReflectionProbe {
    pub extent: Vec3,
    pub blend: f32,
    pub interior: Option<Interior>,
}

settings! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct Interior {
        pub sky: f32 = 1.0 => range(0.0, 1.0),
        pub sun: f32 = 1.0 => range(0.0, 1.0),
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

    pub fn validate(&self) -> Result<(), InvalidSetting> {
        if !(self.extent.is_finite() && self.extent.cmpgt(Vec3::ZERO).all()) {
            return Err(
                InvalidSetting::broken("extent", Rule::above(0.0)).got(self.extent.to_array())
            );
        }
        if !(self.blend.is_finite() && self.blend >= 0.0) {
            return Err(InvalidSetting::broken("blend", Rule::at_least(0.0)).got(self.blend));
        }
        match self.interior {
            Some(interior) => interior
                .validate()
                .map_err(|invalid| invalid.within("interior")),
            None => Ok(()),
        }
    }
}

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneReflectionProbes")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct ReflectionProbeSettings {
        pub enabled: bool = true => any,
        pub resolution: u32 = 128 => power_of_two(16, 1024),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_probes: Option<u32> = None => budget,
        pub faces_per_frame: u32 = 1 => at_least(1),
    }
}

environment_section!(ReflectionProbeSettings, "reflection_probes", opt_in: false);

impl ReflectionProbeSettings {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderReflectionProbe {
    pub key: u64,
    pub center: Vec3,
    pub extent: Vec3,
    pub blend: f32,
    pub interior: Option<Interior>,
}

#[derive(Clone, Debug, Default)]
pub struct RenderReflectionProbes {
    pub list: Vec<RenderReflectionProbe>,
    pub dropped: usize,
}

#[derive(Resource, Default)]
pub(crate) struct ProbeExtractor(Mutex<ProbeExtraction>);

fn extract_probes(world: &World, snapshot: &mut RenderSnapshot) {
    world
        .resource::<ProbeExtractor>()
        .0
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .extract(world, snapshot);
}

#[derive(Default)]
struct ProbeExtraction {
    query: Option<QueryState<(Entity, &'static ReflectionProbe)>>,
    hints: Vec<TransformIndex>,
    candidates: Vec<(f32, RenderReflectionProbe)>,
    nearest: Nearest,
}

impl ProbeExtraction {
    fn extract(&mut self, world: &World, snapshot: &mut RenderSnapshot) {
        let settings = snapshot.environment.reflection_probes;
        snapshot.reflection_probes.list.clear();
        snapshot.reflection_probes.dropped = 0;
        if !settings.enabled {
            return;
        }
        if self.query.is_none() {
            self.query = world.try_query();
        }
        let globals = world.resource::<GlobalTransforms>();
        let eye = snapshot.detail_eye;
        self.candidates.clear();
        if let Some(query) = &mut self.query {
            for (position, (entity, probe)) in query.iter(world).enumerate() {
                if probe.validate().is_err() {
                    continue;
                }
                if self.hints.len() <= position {
                    self.hints.resize(position + 1, TransformIndex::default());
                }
                let Some(center) = globals
                    .get_indexed(entity, &mut self.hints[position])
                    .and_then(|pose| pose.matrix())
                    .map(|matrix| matrix.w_axis.truncate())
                    .filter(|center| center.is_finite())
                else {
                    continue;
                };
                let outside = ((eye - center).abs() - probe.extent).max(Vec3::ZERO);
                self.candidates.push((
                    outside.length_squared(),
                    RenderReflectionProbe {
                        key: entity.to_bits(),
                        center,
                        extent: probe.extent,
                        blend: probe.blend,
                        interior: probe.interior,
                    },
                ));
            }
        }
        let kept = settings
            .max_probes
            .map_or(usize::MAX, |budget| budget as usize);
        let candidates = &self.candidates;
        let ranked = self
            .nearest
            .nearest_first(candidates.iter().map(|(distance, _)| *distance), kept);
        snapshot.reflection_probes.dropped = candidates.len() - ranked.len();
        snapshot
            .reflection_probes
            .list
            .extend(ranked.map(|position| candidates[position].1));
    }
}
