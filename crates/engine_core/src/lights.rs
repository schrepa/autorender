use bevy_ecs::{
    change_detection::DetectChanges,
    prelude::{Component, Entity, Ref, Resource, World},
    query::QueryState,
};
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

use crate::extraction::{Nearest, Poses};
use crate::ticks::ScanClock;
use crate::{GlobalTransforms, RenderSnapshot};
use engine_settings::{ExtraRules, InvalidSetting, Setting, settings};

pub const MAX_SPOT_DEGREES: f32 = 89.0;
pub const MIN_SPOT_DEGREES: f32 = 0.1;
pub const MAX_LIGHT_RANGE: f32 = 10_000.0;

settings! {
    unauthored;
    extra_rules;
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Spot {
        pub direction: Vec3 = Vec3::NEG_Y => finite,
        pub inner_angle: f32 = 25.0_f32.to_radians() => range(0.0, MAX_SPOT_DEGREES.to_radians()),
        pub outer_angle: f32 = 40.0_f32.to_radians()
            => range(MIN_SPOT_DEGREES.to_radians(), MAX_SPOT_DEGREES.to_radians()),
    }
}

impl ExtraRules for Spot {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.direction.length_squared() == 0.0 {
            return Err(InvalidSetting::new("direction", "must be nonzero"));
        }
        if self.outer_angle < self.inner_angle {
            return Err(InvalidSetting::new(
                "outer_angle",
                "must be at least the inner angle",
            ));
        }
        Ok(())
    }
}

settings! {
    unauthored;
    #[derive(Component, Clone, Copy, Debug, PartialEq)]
    pub struct LocalLight {
        pub color: Vec3 = Vec3::new(1.0, 0.85, 0.6) => at_least(0.0),
        pub intensity: f32 = 20.0 => above(0.0),
        pub range: f32 = 12.0 => above_to(0.0, MAX_LIGHT_RANGE),
        pub offset: Vec3 = Vec3::ZERO => range(-1.0e6, 1.0e6),
        pub spot: Option<Spot> = None => any,
        pub radius: f32 = 0.0 => range(0.0, MAX_LIGHT_RANGE),
        pub length: f32 = 0.0 => range(0.0, MAX_LIGHT_RANGE),
        pub region: LightRegion = LightRegion::Both => any,
    }
}

impl Setting for LightRegion {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LightRegion {
    #[default]
    Both,
    Interior,
    Exterior,
}

impl LocalLight {
    pub fn point(color: Vec3, intensity: f32, range: f32) -> Self {
        Self {
            color,
            intensity,
            range,
            offset: Vec3::ZERO,
            spot: None,
            radius: 0.0,
            length: 0.0,
            region: LightRegion::Both,
        }
    }

    pub fn spot(
        color: Vec3,
        intensity: f32,
        range: f32,
        direction: Vec3,
        inner_angle: f32,
        outer_angle: f32,
    ) -> Self {
        Self {
            spot: Some(Spot {
                direction,
                inner_angle,
                outer_angle,
            }),
            ..Self::point(color, intensity, range)
        }
    }

    #[must_use]
    pub fn with_offset(mut self, offset: Vec3) -> Self {
        self.offset = offset;
        self
    }

    #[must_use]
    pub fn with_shape(mut self, radius: f32, length: f32) -> Self {
        self.radius = radius;
        self.length = length;
        self
    }

    #[must_use]
    pub fn with_region(mut self, region: LightRegion) -> Self {
        self.region = region;
        self
    }
}

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneLocalLights")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct LocalLightSettings {
        pub enabled: bool = true => any,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_lights: Option<u32> = None => budget,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub lights_per_cluster: Option<u32> = None => budget,
        pub intensity: f32 = 1.0 => range(0.0, 100.0),
        pub distance: f32 = 250.0 => range(1.0, 100_000.0),
    }
}

impl LocalLightSettings {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    pub fn active(&self) -> bool {
        self.enabled && self.intensity > 0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderLight {
    pub position: Vec3,
    pub range: f32,
    pub color: Vec3,
    pub direction: Vec3,
    pub cone: [f32; 2],
    pub shape: [f32; 2],
    pub region: LightRegion,
    pub slot: Option<u32>,
}

impl RenderLight {
    pub const POINT: [f32; 2] = [-2.0, -2.0];

    fn new(light: &LocalLight, scale: f32, model: Option<Mat4>, slot: Option<u32>) -> Self {
        let (position, direction) = match light.spot {
            Some(spot) => (light.offset, spot.direction),
            None if light.length > 0.0 => (light.offset, Vec3::Y),
            None => (light.offset, Vec3::ZERO),
        };
        let (position, direction, size) = match model {
            Some(model) => (
                model.transform_point3(position),
                model.transform_vector3(direction),
                model.transform_vector3(Vec3::Y).length(),
            ),
            None => (position, direction, 1.0),
        };
        let cone = light.spot.map_or(Self::POINT, |spot| {
            let outer = crate::math::clamp(spot.outer_angle, 1.0e-3, 1.55);
            let inner = crate::math::clamp(spot.inner_angle, 0.0, outer);
            [
                crate::math::max(crate::math::cos(inner), crate::math::cos(outer) + 1.0e-4),
                crate::math::cos(outer),
            ]
        });
        let length = if light.spot.is_some() {
            0.0
        } else {
            light.length
        };
        Self {
            position,
            range: light.range,
            color: light.color * (light.intensity * scale),
            direction: direction.normalize_or_zero(),
            cone,
            shape: [light.radius * size, length * size],
            region: light.region,
            slot,
        }
    }

    pub(crate) fn posed(&self, model: Mat4) -> Self {
        let size = model.transform_vector3(Vec3::Y).length();
        Self {
            position: model.transform_point3(self.position),
            direction: model.transform_vector3(self.direction).normalize_or_zero(),
            shape: self.shape.map(|value| value * size),
            slot: None,
            ..*self
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RenderLights {
    pub list: Vec<RenderLight>,
    pub dropped: usize,
    pub invalid: usize,
}

pub(crate) struct LightExtraction {
    query: QueryState<(Entity, Ref<'static, LocalLight>)>,
    scans: ScanClock,
    count: usize,
    source_revision: Option<u64>,
    settings: Option<LocalLightSettings>,
    lights: Vec<(Entity, LocalLight, Option<u32>)>,
    world_lights: Vec<usize>,
    invalid: usize,
    ranked: Vec<(f32, usize)>,
    nearest: Nearest,
    pub(crate) revision: u64,
}

impl LightExtraction {
    pub(crate) fn new(world: &mut World) -> Self {
        Self {
            query: world.query::<(Entity, Ref<LocalLight>)>(),
            scans: ScanClock::default(),
            count: 0,
            source_revision: None,
            settings: None,
            lights: Vec::new(),
            world_lights: Vec::new(),
            invalid: 0,
            ranked: Vec::new(),
            nearest: Nearest::default(),
            revision: 0,
        }
    }

    pub(crate) fn extract(
        &mut self,
        world: &World,
        snapshot: &mut RenderSnapshot,
        slot_of: impl Fn(Entity) -> Option<u32>,
        source_revision: u64,
        gpu: bool,
    ) {
        let settings = snapshot.environment.local_lights;
        snapshot.lights.dropped = 0;
        snapshot.lights.invalid = 0;
        if !settings.active() {
            snapshot.lights.list.clear();
            self.scans.forget();
            self.revision += 1;
            return;
        }
        let this_run = world.read_change_tick();
        let mut count = 0;
        let mut changed = self.scans.aged(this_run)
            || self.source_revision != Some(source_revision)
            || self.settings != Some(settings);
        for (_, light) in self.query.iter(world) {
            count += 1;
            changed |= self.scans.changed(light.last_changed(), this_run);
        }
        changed |= count != self.count;
        if changed {
            self.lights.clear();
            self.world_lights.clear();
            self.invalid = 0;
            for (entity, light) in self.query.iter(world) {
                if light.validate().is_err() {
                    self.invalid += 1;
                    continue;
                }
                let slot = slot_of(entity);
                if slot.is_none() {
                    self.world_lights.push(self.lights.len());
                }
                self.lights.push((entity, *light, slot));
            }
            self.count = count;
            self.source_revision = Some(source_revision);
            self.settings = Some(settings);
            self.scans.scanned(this_run);
        } else if self.world_lights.is_empty()
            && settings
                .max_lights
                .is_none_or(|budget| self.lights.len() <= budget as usize)
        {
            snapshot.lights.invalid = self.invalid;
            return;
        }
        snapshot.lights.invalid = self.invalid;
        self.revision += 1;
        let globals = world.resource::<GlobalTransforms>();
        let poses = Poses::of(world, gpu);
        let budget = settings
            .max_lights
            .map_or(usize::MAX, |budget| budget as usize);
        snapshot.lights.list.clear();
        if self.lights.len() > budget {
            let eye = snapshot.detail_eye;
            self.ranked.clear();
            for (index, (entity, light, _)) in self.lights.iter().enumerate() {
                let Some(model) = globals.get(*entity).and_then(|pose| pose.matrix()) else {
                    continue;
                };
                let distance = model.transform_point3(light.offset).distance_squared(eye)
                    - light.range * light.range;
                self.ranked.push((distance, index));
            }
            let kept = self
                .nearest
                .kept_in_order(self.ranked.iter().map(|&(distance, _)| distance), budget);
            snapshot.lights.dropped = self.lights.len() - kept.len();
            for position in kept {
                let (entity, light, slot) = &self.lights[self.ranked[position].1];
                let model = match slot {
                    Some(_) => None,
                    None => match poses.model(*entity) {
                        Some(model) => Some(model),
                        None => continue,
                    },
                };
                snapshot.lights.list.push(RenderLight::new(
                    light,
                    settings.intensity,
                    model,
                    *slot,
                ));
            }
            return;
        }
        for (entity, light, slot) in &self.lights {
            let model = match slot {
                Some(_) => None,
                None => match poses.model(*entity) {
                    Some(model) => Some(model),
                    None => {
                        snapshot.lights.dropped += 1;
                        continue;
                    }
                },
            };
            snapshot
                .lights
                .list
                .push(RenderLight::new(light, settings.intensity, model, *slot));
        }
    }
}
