//! Local lights: point and spot lights carried by entities.
//!
//! A light on an entity that renders a mesh follows that mesh's displayed
//! pose, interpolation included: retained frames name the mesh's slot and the
//! renderer reads the pose it already has, so such lights cost no per-frame
//! CPU work while they are unchanged. Lights on other entities follow the
//! authoritative world pose. Renderers bin lights into view clusters and shade
//! each pixel with its cluster's lights; frames without lights pay nothing.
//!
//! The [`LocalLights`] settings are inserted by default; without them, or
//! when disabled, frames shade no local lights. Beyond `max_lights` the
//! lights nearest the camera win, chosen every frame, and the rest count as
//! dropped; beyond `lights_per_cluster` a cluster's later-binned lights are
//! lost and counted as cluster overflow. Light fades out over the last tenth
//! of `distance`.

use bevy_ecs::{
    change_detection::{DetectChanges, Tick},
    prelude::{Component, Entity, Ref, Resource, World},
    query::QueryState,
};
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

use crate::environment::{InvalidSetting, check_budget, check_range};
use crate::extraction::{Nearest, Poses};
use crate::{GlobalTransforms, RenderSnapshot};

/// The cone of a spot light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spot {
    /// Direction the cone points, in the entity's local space; any nonzero
    /// length.
    pub direction: Vec3,
    /// Half-angle in radians inside which the light is full.
    pub inner_angle: f32,
    /// Half-angle in radians beyond which it is dark: at least
    /// `inner_angle` and below a quarter turn.
    pub outer_angle: f32,
}

/// A point or spot light at its entity's origin plus `offset`.
///
/// Light reaching a surface `d` world units away is `color * intensity /
/// d²`, smoothly windowed to nothing at `range`. On an entity that renders a
/// mesh the light follows the mesh's displayed pose; elsewhere it follows the
/// authoritative world pose, so put lights on moving objects on the moving
/// mesh itself. Unusable values (nonfinite or negative color, intensity or
/// range that is not positive, a zero spot direction) skip the light.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct LocalLight {
    /// Linear color.
    pub color: Vec3,
    /// Light at one world unit.
    pub intensity: f32,
    /// World distance beyond which the light has no effect.
    pub range: f32,
    /// Position in the entity's local space, scaled with it.
    pub offset: Vec3,
    /// A cone, or `None` for light in every direction.
    pub spot: Option<Spot>,
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
    pub fn with_offset(mut self, offset: Vec3) -> Self {
        self.offset = offset;
        self
    }

    /// Whether renderers can shade this light (see the type's documentation).
    pub fn is_usable(&self) -> bool {
        let spot_usable = self.spot.is_none_or(|spot| {
            spot.direction.is_finite()
                && spot.direction.length_squared() > 0.0
                && spot.inner_angle.is_finite()
                && spot.outer_angle.is_finite()
                && spot.inner_angle >= 0.0
                && spot.outer_angle > 0.0
        });
        self.color.is_finite()
            && self.color.min_element() >= 0.0
            && self.intensity.is_finite()
            && self.intensity > 0.0
            && self.range.is_finite()
            && self.range > 0.0
            && self.offset.is_finite()
            && spot_usable
    }
}

/// Local lighting settings. Requires version 6.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneLocalLights")
)]
#[serde(default, deny_unknown_fields)]
pub struct LocalLights {
    pub enabled: bool,
    /// An optional budget of lights a frame shades; beyond it the nearest to
    /// the camera win. Omitted, every light shades.
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_lights: Option<u32>,
    /// An optional budget of lights one view cluster lists. Omitted, every
    /// light reaching a cluster shades there.
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lights_per_cluster: Option<u32>,
    /// Scales every light's intensity, e.g. to dim lamps by day.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 100.0)))]
    pub intensity: f32,
    /// View distance beyond which local light fades out: distant lights
    /// cost no shading.
    #[cfg_attr(feature = "schema", schemars(range(min = 1.0, max = 100000.0)))]
    pub distance: f32,
}

impl Default for LocalLights {
    fn default() -> Self {
        Self {
            enabled: true,
            max_lights: None,
            lights_per_cluster: None,
            intensity: 1.0,
            distance: 250.0,
        }
    }
}

impl LocalLights {
    /// What a world without these settings reads as: no local lights.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Ok for budgets of at least one light, an intensity in `[0, 100]` and
    /// a distance in `[1, 100000]`.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_budget("max_lights", self.max_lights)?;
        check_budget("lights_per_cluster", self.lights_per_cluster)?;
        check_range("intensity", self.intensity, 0.0..=100.0)?;
        check_range("distance", self.distance, 1.0..=100_000.0)
    }

    /// Budgets and cluster lists of at least one light; an unusable
    /// intensity scale reads as 1 and an unusable distance as the default.
    pub fn sanitized(self) -> Self {
        Self {
            enabled: self.enabled,
            max_lights: self.max_lights.map(|budget| budget.max(1)),
            lights_per_cluster: self.lights_per_cluster.map(|budget| budget.max(1)),
            intensity: if self.intensity.is_finite() && self.intensity >= 0.0 {
                self.intensity
            } else {
                1.0
            },
            distance: if self.distance.is_finite() && self.distance > 0.0 {
                self.distance
            } else {
                Self::default().distance
            },
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
    /// Unit cone direction in the same space as `position`; zero for point
    /// lights.
    pub direction: Vec3,
    /// Cosines of the inner and outer cone half-angles; `[-2, -2]` for
    /// point lights.
    pub cone: [f32; 2],
    /// Frames: the renderable slot whose displayed pose carries this light.
    /// Renderers transform `position` and `direction` by that pose. Owned
    /// snapshots ([`crate::Engine::extract`]) hold every light in world space.
    pub slot: Option<u32>,
}

impl RenderLight {
    /// Cone cosines of point lights: below any cosine.
    pub const POINT: [f32; 2] = [-2.0, -2.0];

    fn new(light: &LocalLight, scale: f32, model: Option<Mat4>, slot: Option<u32>) -> Self {
        let (position, direction) = match light.spot {
            Some(spot) => (light.offset, spot.direction),
            None => (light.offset, Vec3::ZERO),
        };
        let (position, direction) = match model {
            Some(model) => (
                model.transform_point3(position),
                model.transform_vector3(direction),
            ),
            None => (position, direction),
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
        Self {
            position,
            range: light.range,
            color: light.color * (light.intensity * scale),
            direction: direction.normalize_or_zero(),
            cone,
            slot,
        }
    }
}

/// Retained light records and what they were built from.
pub(crate) struct LightExtraction {
    query: QueryState<(Entity, Ref<'static, LocalLight>)>,
    // Scan tick of the last rebuild, the lights it saw and the slot source
    // revision it resolved slots against.
    last_scan: Option<Tick>,
    count: usize,
    source_revision: Option<u64>,
    settings: Option<LocalLights>,
    // Every usable light: entity, component, and slot when it has one.
    lights: Vec<(Entity, LocalLight, Option<u32>)>,
    // Indices into `lights` of those without a slot: their world poses are
    // read every frame.
    world_lights: Vec<usize>,
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
            last_scan: None,
            count: 0,
            source_revision: None,
            settings: None,
            lights: Vec::new(),
            world_lights: Vec::new(),
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
        snapshot.dropped_lights = 0;
        if !settings.active() {
            snapshot.lights.clear();
            self.last_scan = None;
            self.revision += 1;
            return;
        }
        let this_run = world.read_change_tick();
        let mut count = 0;
        let mut changed = self.last_scan.is_none()
            || self.source_revision != Some(source_revision)
            || self.settings != Some(settings);
        for (_, light) in self.query.iter(world) {
            count += 1;
            if let Some(last) = self.last_scan
                && light.last_changed().is_newer_than(last, this_run)
            {
                changed = true;
            }
        }
        changed |= count != self.count;
        if changed {
            self.lights.clear();
            self.world_lights.clear();
            for (entity, light) in self.query.iter(world) {
                if !light.is_usable() {
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
            self.last_scan = Some(this_run);
        } else if self.world_lights.is_empty()
            && settings
                .max_lights
                .is_none_or(|budget| self.lights.len() <= budget as usize)
        {
            // Unchanged lights that all follow slots: the records stand.
            return;
        }
        self.revision += 1;
        let globals = world.resource::<GlobalTransforms>();
        let poses = Poses::of(world, gpu);
        let budget = settings
            .max_lights
            .map_or(usize::MAX, |budget| budget as usize);
        snapshot.lights.clear();
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
            snapshot.dropped_lights = self.lights.len() - kept.len();
            for position in kept {
                let (entity, light, slot) = &self.lights[self.ranked[position].1];
                let model = match slot {
                    Some(_) => None,
                    None => match poses.model(*entity) {
                        Some(model) => Some(model),
                        None => continue,
                    },
                };
                snapshot
                    .lights
                    .push(RenderLight::new(light, settings.intensity, model, *slot));
            }
            return;
        }
        for (entity, light, slot) in &self.lights {
            let model = match slot {
                Some(_) => None,
                None => match poses.model(*entity) {
                    Some(model) => Some(model),
                    None => {
                        snapshot.dropped_lights += 1;
                        continue;
                    }
                },
            };
            snapshot
                .lights
                .push(RenderLight::new(light, settings.intensity, model, *slot));
        }
    }
}
