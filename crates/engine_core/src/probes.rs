//! Reflection probes: boxes whose surroundings the renderer
//! captures into environment maps, which surfaces inside them reflect in
//! place of the sky's.
//!
//! Extraction lists the usable probes nearest the camera each frame; the
//! renderer keeps what it captured of each until the probe changes or the
//! sky does.
//!
//! The [`ReflectionProbes`] settings are inserted by default; without them,
//! or when disabled, frames capture and reflect no probes. Beyond
//! `max_probes` the probes nearest the camera win and the rest count as
//! dropped; without a budget, as many stay as the GPU's texture arrays hold.
//!
//! A probe may hold an interior: a cave or a hall whose box
//! lets in only shares of the sky's light and the sun's, so surfaces inside
//! it darken as the rock or walls around them would make them, and reflect
//! what the probe captured inside.

use bevy_ecs::{
    prelude::{Component, Entity, Resource, World},
    query::QueryState,
};
use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::extraction::Nearest;
use crate::{GlobalTransforms, RenderSnapshot, TransformIndex};
use engine_settings::{ExtraRules, settings};

/// Captures its surroundings from its entity's world position into an
/// environment map, which surfaces inside its box reflect.
///
/// The box is axis-aligned in world space, centered on the entity's world
/// position, reaching `extent` from it along each axis. Reflections inside it
/// land on its walls (parallax correction), so walls, ground and objects near
/// them line up with what they reflect; toward its faces the probe fades out
/// over `blend` into the probes around it or the sky. Probes capture still
/// objects and still lights only: moving ones reflect through screen-space
/// reflections alone.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct ReflectionProbe {
    /// Half the box's size along each world axis, in world units.
    pub extent: Vec3,
    /// World distance inside the box's faces over which the probe fades out.
    pub blend: f32,
    /// What the box lets in, when it is an interior.
    pub interior: Option<Interior>,
}

settings! {
    /// An interior's light, blending in over the probe's blend distance inside
    /// its faces. Requires version 9.
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct Interior {
        /// The share of the sky's light that reaches surfaces inside the box.
        pub sky: f32 = 1.0 => range(0.0, 1.0),
        /// The share of the sun's light that reaches surfaces inside the box.
        pub sun: f32 = 1.0 => range(0.0, 1.0),
    }
}

impl ExtraRules for Interior {}

impl ReflectionProbe {
    /// A probe covering the box `extent` from its entity along each axis,
    /// fading over its outer world unit.
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

    /// The probe as an interior letting in `interior`'s light.
    #[must_use]
    pub fn with_interior(mut self, interior: Interior) -> Self {
        self.interior = Some(interior);
        self
    }

    /// Whether renderers can capture and reflect this probe.
    pub fn is_usable(&self) -> bool {
        self.extent.is_finite()
            && self.extent.cmpgt(Vec3::ZERO).all()
            && self.blend.is_finite()
            && self.blend >= 0.0
            && self
                .interior
                .is_none_or(|interior| interior.validate().is_ok())
    }
}

settings! {
    /// Reflection probe settings. Requires version 8.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneReflectionProbes")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct ReflectionProbeSettings {
        /// Whether surfaces reflect probes.
        pub enabled: bool = true => any,
        /// Texels across each face of every environment map, the sky's
        /// included.
        pub resolution: u32 = 128 => power_of_two(16, 1024),
        /// An optional budget of probes kept captured; beyond it the probes
        /// nearest the camera win. Omitted, as many as the GPU's texture arrays
        /// hold.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_probes: Option<u32> = None => budget,
        /// Cube faces captured per frame while probes wait: more fill new places
        /// sooner, fewer keep those frames lighter.
        pub faces_per_frame: u32 = 1 => at_least(1),
    }
}

impl ExtraRules for ReflectionProbeSettings {}

impl ReflectionProbeSettings {
    /// What a world without these settings reads as: no reflection probes.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

/// One reflection probe of a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderReflectionProbe {
    /// The probe's identity across frames (its entity's bits).
    pub key: u64,
    /// The world position it captures from: the center of its box.
    pub center: Vec3,
    pub extent: Vec3,
    pub blend: f32,
    /// What its box lets in, when it is an interior.
    pub interior: Option<Interior>,
}

/// Reflection probe records of each frame.
pub(crate) struct ProbeExtraction {
    query: QueryState<(Entity, &'static ReflectionProbe)>,
    hints: Vec<TransformIndex>,
    // This frame's usable probes and their squared distances from the eye to
    // their boxes.
    candidates: Vec<(f32, RenderReflectionProbe)>,
    nearest: Nearest,
}

impl ProbeExtraction {
    pub(crate) fn new(world: &mut World) -> Self {
        Self {
            query: world.query::<(Entity, &ReflectionProbe)>(),
            hints: Vec::new(),
            candidates: Vec::new(),
            nearest: Nearest::default(),
        }
    }

    /// Fills the snapshot's probes, nearest the camera first, from their
    /// entities' authoritative poses; beyond a budget the nearest win.
    pub(crate) fn extract(&mut self, world: &World, snapshot: &mut RenderSnapshot) {
        // Extracted sanitized with the environment, before this.
        let settings = snapshot.environment.reflection_probes;
        snapshot.reflection_probes.clear();
        snapshot.dropped_reflection_probes = 0;
        if !settings.enabled {
            return;
        }
        let globals = world.resource::<GlobalTransforms>();
        let eye = snapshot.camera_position;
        self.candidates.clear();
        for (position, (entity, probe)) in self.query.iter(world).enumerate() {
            if !probe.is_usable() {
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
        // Nearest first; equally near probes keep their order.
        let kept = settings
            .max_probes
            .map_or(usize::MAX, |budget| budget as usize);
        let candidates = &self.candidates;
        let ranked = self
            .nearest
            .nearest_first(candidates.iter().map(|(distance, _)| *distance), kept);
        snapshot.dropped_reflection_probes = candidates.len() - ranked.len();
        snapshot
            .reflection_probes
            .extend(ranked.map(|position| candidates[position].1));
    }
}
