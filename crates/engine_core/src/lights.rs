//! Local lights: point and spot lights carried by entities.
//!
//! A light on an entity that renders a mesh follows that mesh's displayed
//! pose, interpolation included: retained frames name the mesh's slot and the
//! renderer reads the pose it already has, so such lights cost no per-frame
//! CPU work while they are unchanged. Lights on other entities follow the
//! authoritative world pose. Renderers bin lights into view clusters and shade
//! each pixel with its cluster's lights; frames without lights pay nothing.
//!
//! The [`LocalLightSettings`] are inserted by default; without them, or
//! when disabled, frames shade no local lights. Beyond `max_lights` the
//! lights nearest the camera win, chosen every frame, and the rest count as
//! dropped; beyond `lights_per_cluster` a cluster's later-binned lights are
//! lost and counted as cluster overflow. Light fades out over the last tenth
//! of `distance`.

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

/// The widest a spot's cone opens, in degrees from its axis.
pub const MAX_SPOT_DEGREES: f32 = 89.0;
/// The narrowest a spot's outer angle closes, in degrees.
pub const MIN_SPOT_DEGREES: f32 = 0.1;
/// The farthest a local light reaches, world units.
pub const MAX_LIGHT_RANGE: f32 = 10_000.0;

settings! {
    unauthored;
    extra_rules;
    /// The cone of a spot light.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Spot {
        /// Direction the cone points, in the entity's local space; any
        /// nonzero length.
        pub direction: Vec3 = Vec3::NEG_Y => finite,
        /// Half-angle in radians inside which the light is full.
        pub inner_angle: f32 = 25.0_f32.to_radians() => range(0.0, MAX_SPOT_DEGREES.to_radians()),
        /// Half-angle in radians beyond which it is dark: at least
        /// `inner_angle`.
        pub outer_angle: f32 = 40.0_f32.to_radians()
            => range(MIN_SPOT_DEGREES.to_radians(), MAX_SPOT_DEGREES.to_radians()),
    }
}

impl ExtraRules for Spot {
    /// The cone points somewhere, and opens at least as wide outside as in.
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
    /// A point or spot light at its entity's origin plus `offset`.
    ///
    /// Light reaching a surface `d` world units away is `color * intensity /
    /// d²`, smoothly windowed to nothing at `range`. On an entity that renders
    /// a mesh the light follows the mesh's displayed pose; elsewhere it
    /// follows the authoritative world pose, so put lights on moving objects
    /// on the moving mesh itself. Lights breaking their rules are skipped.
    #[derive(Component, Clone, Copy, Debug, PartialEq)]
    pub struct LocalLight {
        /// Linear color.
        pub color: Vec3 = Vec3::new(1.0, 0.85, 0.6) => at_least(0.0),
        /// Light at one world unit.
        pub intensity: f32 = 20.0 => above(0.0),
        /// World distance beyond which the light has no effect.
        pub range: f32 = 12.0 => above_to(0.0, MAX_LIGHT_RANGE),
        /// Position in the entity's local space, scaled with it.
        pub offset: Vec3 = Vec3::ZERO => range(-1.0e6, 1.0e6),
        /// A cone, or `None` for light in every direction.
        pub spot: Option<Spot> = None => any,
        /// The radius of the sphere it shines from while [`crate::Highlights`]
        /// are on, scaled with its entity (by its local y axis's scale);
        /// their `light_radius` where this is smaller.
        pub radius: f32 = 0.0 => range(0.0, MAX_LIGHT_RANGE),
        /// The length of the tube it shines from, up its entity's local y axis
        /// from its position, scaled with it; 0 for none. Spots shine from
        /// their sphere alone. Lights take it while [`crate::Highlights`] are
        /// on.
        pub length: f32 = 0.0 => range(0.0, MAX_LIGHT_RANGE),
        /// Where it shines: everywhere, or only inside or only outside
        /// interiors.
        pub region: LightRegion = LightRegion::Both => any,
    }
}

impl Setting for LightRegion {}

/// Where a local light shines. Interiors are the boxes of interior
/// reflection probes ([`crate::ReflectionProbe::interior`]): with no
/// shadows of their own, lights would otherwise shine through the walls
/// and rock around them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LightRegion {
    /// Inside and outside interiors alike.
    #[default]
    Both,
    /// Inside interiors only: a cave's or a hall's lamps, which then stop
    /// lighting the ground over them.
    Interior,
    /// Outside interiors only: lamps beside a cave's mouth.
    Exterior,
}

impl LocalLight {
    /// Light in every direction from the entity's origin.
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

    /// A cone pointing along local `direction`, full within `inner_angle`
    /// and dark beyond `outer_angle` (radians).
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

    /// This light at local `offset` from its entity's origin.
    #[must_use]
    pub fn with_offset(mut self, offset: Vec3) -> Self {
        self.offset = offset;
        self
    }

    /// This light shining from a sphere of `radius` and a tube of `length`
    /// up its entity's local y axis from its position.
    #[must_use]
    pub fn with_shape(mut self, radius: f32, length: f32) -> Self {
        self.radius = radius;
        self.length = length;
        self
    }

    /// This light shining in `region` alone.
    #[must_use]
    pub fn with_region(mut self, region: LightRegion) -> Self {
        self.region = region;
        self
    }
}

settings! {
    /// Local lighting settings.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneLocalLights")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct LocalLightSettings {
        /// Whether frames shade local lights.
        pub enabled: bool = true => any,
        /// An optional budget of lights a frame shades; beyond it the nearest to
        /// the camera win. Omitted, every light shades.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_lights: Option<u32> = None => budget,
        /// An optional budget of lights one view cluster lists. Omitted, every
        /// light reaching a cluster shades there.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub lights_per_cluster: Option<u32> = None => budget,
        /// Scales every light's intensity, e.g. to dim lamps by day.
        pub intensity: f32 = 1.0 => range(0.0, 100.0),
        /// View distance beyond which local light fades out: distant lights
        /// cost no shading.
        pub distance: f32 = 250.0 => range(1.0, 100_000.0),
    }
}

impl LocalLightSettings {
    /// What a world without these settings reads as: no local lights.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Whether frames shade local lights at all.
    pub fn active(&self) -> bool {
        self.enabled && self.intensity > 0.0
    }
}

/// One light of a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderLight {
    /// World position; with `slot`, the position in that slot's model space.
    pub position: Vec3,
    /// World distance beyond which the light has no effect.
    pub range: f32,
    /// Linear color times intensity and the settings' scale.
    pub color: Vec3,
    /// Unit cone direction in the same space as `position`; for lights
    /// without a cone, their tube's axis, or zero without one.
    pub direction: Vec3,
    /// Cosines of the inner and outer cone half-angles; `[-2, -2]` for
    /// point lights.
    pub cone: [f32; 2],
    /// The radius of the sphere and the length of the tube it shines from,
    /// in the units of `position`; with `slot`, renderers scale them by the
    /// pose's y axis, as they turn `direction`.
    pub shape: [f32; 2],
    /// Where it shines.
    pub region: LightRegion,
    /// Frames: the renderable slot whose displayed pose carries this light.
    /// Renderers transform `position` and `direction` by that pose. Owned
    /// snapshots ([`crate::Engine::extract`]) hold every light in world space.
    pub slot: Option<u32>,
}

impl RenderLight {
    /// Cone cosines of point lights: below any cosine.
    pub const POINT: [f32; 2] = [-2.0, -2.0];

    fn new(light: &LocalLight, scale: f32, model: Option<Mat4>, slot: Option<u32>) -> Self {
        // A tube stands along the entity's y axis.
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
            // A full cone still fades over a sliver, avoiding a division by
            // zero in renderers' smooth steps.
            [
                crate::math::max(crate::math::cos(inner), crate::math::cos(outer) + 1.0e-4),
                crate::math::cos(outer),
            ]
        });
        // Spots shine from their sphere alone.
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

    /// This light of a slot in world space, through the slot's `model`:
    /// its sphere and tube scale with the model's y axis.
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

/// A frame's local lights ([`RenderSnapshot::lights`]), filled in place.
#[derive(Clone, Debug, Default)]
pub struct RenderLights {
    /// Its local lights, within the settings' budget.
    pub list: Vec<RenderLight>,
    /// Usable lights beyond the budget, or without a valid pose.
    pub dropped: usize,
    /// Lights breaking their rules, which no frame draws.
    pub invalid: usize,
}

/// Retained light records and what they were built from.
pub(crate) struct LightExtraction {
    query: QueryState<(Entity, Ref<'static, LocalLight>)>,
    // Scan tick of the last rebuild, the lights it saw and the slot source
    // revision it resolved slots against.
    scans: ScanClock,
    count: usize,
    source_revision: Option<u64>,
    settings: Option<LocalLightSettings>,
    // Every usable light: entity, component, and slot when it has one.
    lights: Vec<(Entity, LocalLight, Option<u32>)>,
    // Indices into `lights` of those without a slot: their world poses are
    // read every frame.
    world_lights: Vec<usize>,
    // Lights the last rebuild found breaking their rules.
    invalid: usize,
    // Lights with an authoritative pose, by distance, when lights exceed the
    // budget: their distance and index.
    ranked: Vec<(f32, usize)>,
    nearest: Nearest,
    // Advances whenever an extraction rewrites the snapshot's lights.
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

    /// Lights on entities that own slot `slot_of(entity)` stay in that slot's
    /// model space. `source_revision` changes when slot ownership does; `gpu`
    /// frames read authoritative poses for the others.
    pub(crate) fn extract(
        &mut self,
        world: &World,
        snapshot: &mut RenderSnapshot,
        slot_of: impl Fn(Entity) -> Option<u32>,
        source_revision: u64,
        gpu: bool,
    ) {
        // Extracted sanitized with the environment, before this.
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
            // Unchanged lights that all follow slots: the records stand.
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
            // Nearest to the camera first, by authoritative position.
            let eye = snapshot.camera_position;
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
