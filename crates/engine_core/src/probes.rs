//! Reflection probes (decision 043): boxes whose surroundings the renderer
//! captures into environment maps, which surfaces inside them reflect in
//! place of the sky's.
//!
//! Extraction lists the usable probes nearest the camera each frame; the
//! renderer keeps what it captured of each until the probe changes or the
//! sky does.

use bevy_ecs::{
    prelude::{Component, Entity, Resource, World},
    query::QueryState,
};
use glam::Vec3;

use crate::{GlobalTransforms, RenderSnapshot, TransformIndex};

/// Captures its surroundings from its entity's world position into an
/// environment map (decision 043), which surfaces inside its box reflect.
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
}

impl ReflectionProbe {
    /// A probe covering the box `extent` from its entity along each axis,
    /// fading over its outer world unit.
    pub fn new(extent: Vec3) -> Self {
        Self { extent, blend: 1.0 }
    }

    pub fn with_blend(mut self, blend: f32) -> Self {
        self.blend = blend;
        self
    }

    /// Whether renderers can capture and reflect this probe.
    pub fn is_usable(&self) -> bool {
        self.extent.is_finite()
            && self.extent.cmpgt(Vec3::ZERO).all()
            && self.blend.is_finite()
            && self.blend >= 0.0
    }
}

/// Reflection probe settings (decision 043), inserted by default. Without
/// this resource, or when disabled, frames capture and reflect no probes.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct ReflectionProbes {
    pub enabled: bool,
    /// Texels across each face of every environment map, the sky's included:
    /// a power of two in `[16, 1024]`.
    pub resolution: u32,
    /// An optional budget of probes kept captured: beyond it the probes
    /// nearest the camera win and the rest count as dropped. `None`, the
    /// default, keeps as many as the GPU's texture arrays hold.
    pub max_probes: Option<u32>,
    /// Cube faces captured per frame while probes wait: more fill new places
    /// sooner, fewer keep those frames lighter.
    pub faces_per_frame: u32,
}

impl Default for ReflectionProbes {
    fn default() -> Self {
        Self {
            enabled: true,
            resolution: 128,
            max_probes: None,
            faces_per_frame: 1,
        }
    }
}

impl ReflectionProbes {
    /// The resolution rounded up to a power of two in `[16, 1024]`, a budget
    /// of at least one probe and at least one face per frame.
    pub fn sanitized(self) -> Self {
        Self {
            enabled: self.enabled,
            resolution: self.resolution.clamp(16, 1024).next_power_of_two(),
            max_probes: self.max_probes.map(|budget| budget.max(1)),
            faces_per_frame: self.faces_per_frame.max(1),
        }
    }
}

/// One reflection probe of a frame (decision 043).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderReflectionProbe {
    /// The probe's identity across frames (its entity's bits).
    pub key: u64,
    /// The world position it captures from: the center of its box.
    pub center: Vec3,
    pub extent: Vec3,
    pub blend: f32,
}

/// Reflection probe records of each frame.
pub(crate) struct ProbeExtraction {
    query: QueryState<(Entity, &'static ReflectionProbe)>,
    hints: Vec<TransformIndex>,
    // This frame's usable probes and their squared distances from the eye to
    // their boxes.
    candidates: Vec<(f32, RenderReflectionProbe)>,
    // Candidates by distance: each distance's total order in the high bits,
    // its candidate's position in the low ones.
    order: Vec<u64>,
}

impl ProbeExtraction {
    pub(crate) fn new(world: &mut World) -> Self {
        Self {
            query: world.query::<(Entity, &ReflectionProbe)>(),
            hints: Vec::new(),
            candidates: Vec::new(),
            order: Vec::new(),
        }
    }

    /// Fills the snapshot's probes, nearest the camera first, from their
    /// entities' authoritative poses; beyond a budget the nearest win.
    pub(crate) fn extract(&mut self, world: &World, snapshot: &mut RenderSnapshot) {
        let settings = world.get_resource::<ReflectionProbes>().map_or(
            ReflectionProbes {
                enabled: false,
                ..ReflectionProbes::default()
            },
            |value| value.sanitized(),
        );
        snapshot.reflection_probe_settings = settings;
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
                },
            ));
        }
        // Nearest first; equally near probes keep their order (sorting
        // eight-byte keys, not the records).
        self.order.clear();
        self.order.extend(
            self.candidates
                .iter()
                .enumerate()
                .map(|(position, (distance, _))| {
                    u64::from(total_order(*distance)) << 32 | position as u64
                }),
        );
        self.order.sort_unstable();
        let kept = settings.max_probes.map_or(self.order.len(), |budget| {
            self.order.len().min(budget as usize)
        });
        snapshot.dropped_reflection_probes = self.order.len() - kept;
        let candidates = &self.candidates;
        snapshot.reflection_probes.extend(
            self.order[..kept]
                .iter()
                .map(|key| candidates[(key & u64::from(u32::MAX)) as usize].1),
        );
    }
}

/// `value`'s bits ordered as [`f32::total_cmp`] orders values.
fn total_order(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits >> 31 == 1 {
        !bits
    } else {
        bits | 1 << 31
    }
}
