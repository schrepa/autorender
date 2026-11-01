use std::sync::{Mutex, PoisonError};

use bevy_ecs::{
    prelude::{Entity, Resource, World},
    query::QueryState,
};
use glam::Vec3;
use serde::{Deserialize, Serialize};

use engine_core::extraction::Nearest;
use engine_core::sections::environment_section;
use engine_core::{GlobalTransforms, RenderSnapshot, TransformIndex};
use engine_settings::settings;

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
    pub cut: usize,
}

impl RenderSnapshot {
    pub fn probe_list(&self) -> &[RenderReflectionProbe] {
        if self.section::<ReflectionProbeSettings>().enabled {
            &self.reflection_probes.list
        } else {
            &[]
        }
    }
}

#[derive(Resource, Default)]
pub struct ProbeExtractor(Mutex<ProbeExtraction>);

pub fn extract_probes(world: &World, snapshot: &mut RenderSnapshot) {
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
        let settings = engine_core::section_in::<ReflectionProbeSettings>(world);
        snapshot.reflection_probes.list.clear();
        snapshot.reflection_probes.cut = 0;
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
        snapshot.reflection_probes.cut = candidates.len() - ranked.len();
        snapshot
            .reflection_probes
            .list
            .extend(ranked.map(|position| candidates[position].1));
    }
}
