//! Decals (decision 039): materials projected onto the opaque surfaces inside
//! an entity's box, drawn after the opaque pass from the depth buffer.
//!
//! Extraction reads one record per decal each frame; decals keep no GPU
//! state of their own, so frames without decals pay nothing.
//!
//! The [`Decals`] settings are inserted by default; without them, or when
//! disabled, frames draw no decals. Beyond `max_decals` the decals nearest
//! the camera win and the rest count as dropped.

use std::sync::Arc;

use bevy_ecs::{
    prelude::{Component, Entity, Resource, World},
    query::QueryState,
};
use glam::Mat4;
use serde::{Deserialize, Serialize};

use crate::environment::{InvalidSetting, check_budget};
use crate::{
    AssetLibrary, GlobalTransforms, MaterialHandle, PresentedTransforms, RenderJoint,
    RenderMaterial, RenderSnapshot, RenderTexture, Surface, TransformIndex,
    retained_render::{IDENTITY_ROWS, pose_rows},
};

/// How a decal combines with the surface it lies on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DecalBlend {
    /// Covering the surface by its alpha with its own color, lit by the sun,
    /// sky and local lights like any surface: paint, posters, road markings.
    #[default]
    Alpha,
    /// Multiplying the surface's color by its own, by its alpha: dirt,
    /// cracks, stains and scorch marks keep the surface's lighting.
    Multiply,
}

/// Projects a material onto the opaque surfaces inside its entity's box
/// (decision 039).
///
/// The box is the entity's local `[-0.5, 0.5]` cube placed by its world
/// pose: scale the entity to size the decal. The material's base color and
/// its map span the box's local x (u) and z (v), projected down its local y
/// onto surfaces facing its +y; surfaces turning away fade out from
/// `fade_angle` to a right angle. Its alpha is the material's opacity, times
/// its base color texture's alpha and `opacity`. Every opaque surface in the
/// box receives it, moving or not; transparent surfaces, particles and the
/// sky do not. Decals read the material's color, base color map, roughness,
/// metalness and emission.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Decal {
    pub material: MaterialHandle,
    pub blend: DecalBlend,
    /// Multiplies the decal's alpha, in `[0, 1]`: fade decals in and out.
    pub opacity: f32,
    /// Radians between a surface's normal and the box's +y beyond which the
    /// decal starts to fade, in `[0, π/2]`.
    pub fade_angle: f32,
}

impl Decal {
    /// An alpha decal of `material` at full opacity, fading from 60 degrees.
    pub fn new(material: MaterialHandle) -> Self {
        Self {
            material,
            blend: DecalBlend::Alpha,
            opacity: 1.0,
            fade_angle: 60.0_f32.to_radians(),
        }
    }

    pub fn with_blend(mut self, blend: DecalBlend) -> Self {
        self.blend = blend;
        self
    }

    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    pub fn with_fade_angle(mut self, radians: f32) -> Self {
        self.fade_angle = radians;
        self
    }

    /// Whether renderers can draw this decal.
    pub fn is_usable(&self) -> bool {
        self.opacity.is_finite()
            && (0.0..=1.0).contains(&self.opacity)
            && self.fade_angle.is_finite()
            && (0.0..=std::f32::consts::FRAC_PI_2).contains(&self.fade_angle)
    }
}

/// Decal settings (decision 039). Requires version 8.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneDecals")
)]
#[serde(default, deny_unknown_fields)]
pub struct Decals {
    pub enabled: bool,
    /// An optional budget of decals a frame draws; beyond it the decals
    /// nearest the camera win. Omitted, every decal draws.
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_decals: Option<u32>,
}

impl Default for Decals {
    fn default() -> Self {
        Self {
            enabled: true,
            max_decals: None,
        }
    }
}

impl Decals {
    /// Ok for a budget of at least one decal.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_budget("max_decals", self.max_decals)
    }

    /// A budget of at least one decal.
    pub fn sanitized(self) -> Self {
        Self {
            enabled: self.enabled,
            max_decals: self.max_decals.map(|budget| budget.max(1)),
        }
    }
}

/// One decal of a frame (decision 039).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderDecal {
    /// World matrix of the decal's box: its prepared pose, or its
    /// authoritative pose on GPU-presented frames.
    pub model: Mat4,
    /// Retained frames: the decal's entry in [`RenderSnapshot::joints`],
    /// whose palette holds its displayed pose (GPU-presented frames compose
    /// it as they do slots'). Owned snapshots draw `model`.
    pub joint: Option<u32>,
    /// Its material when it has maps, listed in
    /// [`RenderSnapshot::decal_materials`]; `None` draws from `color` alone.
    pub material: Option<MaterialHandle>,
    /// Linear color and alpha: the material's, alpha times the decal's
    /// opacity.
    pub color: [f32; 4],
    /// The material's surface, which covering decals shade with.
    pub surface: Surface,
    pub blend: DecalBlend,
    /// Cosine of the fade angle: surfaces whose normals turn further from
    /// the box's +y fade.
    pub fade: f32,
}

/// Where a frame's decal poses come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecalPoses {
    /// Owned snapshots draw each decal's prepared matrix.
    Owned,
    /// Retained frames list each decal's pose among the joints: prepared
    /// matrices, or on `gpu` frames the nodes the GPU composes.
    Retained { gpu: bool },
}

/// Decal records of each frame.
pub(crate) struct DecalExtraction {
    query: QueryState<(Entity, &'static Decal)>,
    hints: Vec<TransformIndex>,
    // This frame's usable decals: squared distance to the eye, world pose,
    // presentation node (GPU frames) and component.
    candidates: Vec<(f32, Mat4, u32, Decal)>,
}

impl DecalExtraction {
    pub(crate) fn new(world: &mut World) -> Self {
        Self {
            query: world.query::<(Entity, &Decal)>(),
            hints: Vec::new(),
            candidates: Vec::new(),
        }
    }

    /// Fills the snapshot's decals and their materials and textures from
    /// `poses`; retained frames append their poses to the joints. Beyond a
    /// budget the nearest win.
    pub(crate) fn extract(
        &mut self,
        world: &World,
        snapshot: &mut RenderSnapshot,
        poses: DecalPoses,
    ) {
        // Extracted sanitized with the environment, before this.
        let settings = snapshot.environment.decals;
        snapshot.decals.clear();
        snapshot.decal_materials.clear();
        snapshot.decal_textures.clear();
        snapshot.dropped_decals = 0;
        if !settings.enabled {
            return;
        }
        let globals = world.resource::<GlobalTransforms>();
        let presented = world.resource::<PresentedTransforms>();
        let assets = world.resource::<AssetLibrary>();
        let eye = snapshot.camera_position;
        let gpu = poses == DecalPoses::Retained { gpu: true };
        self.candidates.clear();
        for (position, (entity, decal)) in self.query.iter(world).enumerate() {
            if !decal.is_usable() {
                continue;
            }
            if self.hints.len() <= position {
                self.hints.resize(position + 1, TransformIndex::default());
            }
            let hint = &mut self.hints[position];
            let model = if gpu {
                globals
                    .get_indexed(entity, hint)
                    .and_then(|pose| pose.matrix())
            } else {
                presented
                    .get_indexed(entity, hint)
                    .and_then(|pose| pose.matrix())
            };
            let Some(model) = model else {
                continue;
            };
            let node = if gpu { hint.0 as u32 } else { u32::MAX };
            let distance = model.w_axis.truncate().distance_squared(eye);
            self.candidates.push((distance, model, node, *decal));
        }
        if let Some(budget) = settings.max_decals
            && self.candidates.len() > budget as usize
        {
            // Stable: equally near decals keep their order.
            self.candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
            snapshot.dropped_decals = self.candidates.len() - budget as usize;
            self.candidates.truncate(budget as usize);
        }
        for &(_, model, node, decal) in &self.candidates {
            let Some(material) = assets.material(decal.material) else {
                snapshot.missing_assets += 1;
                continue;
            };
            // Materials with maps bind their own group: list them and their
            // textures once.
            let mapped = material.has_maps();
            if mapped {
                if material
                    .textures()
                    .any(|texture| assets.texture(texture).is_none())
                {
                    snapshot.missing_assets += 1;
                    continue;
                }
                if !snapshot
                    .decal_materials
                    .iter()
                    .any(|listed| listed.handle == decal.material)
                {
                    snapshot.decal_materials.push(RenderMaterial {
                        handle: decal.material,
                        material: *material,
                    });
                    for texture in material.textures() {
                        if !snapshot
                            .decal_textures
                            .iter()
                            .any(|listed| listed.handle == texture)
                        {
                            snapshot.decal_textures.push(RenderTexture {
                                handle: texture,
                                texture: Arc::clone(
                                    assets.texture(texture).expect("checked above"),
                                ),
                            });
                        }
                    }
                }
            }
            let joint = match poses {
                DecalPoses::Owned => None,
                DecalPoses::Retained { gpu } => {
                    snapshot.joints.push(RenderJoint {
                        node,
                        inverse_bind: IDENTITY_ROWS,
                        matrix: if gpu { IDENTITY_ROWS } else { pose_rows(model) },
                    });
                    Some(snapshot.joints.len() as u32 - 1)
                }
            };
            let mut color = material.color();
            color[3] *= decal.opacity;
            snapshot.decals.push(RenderDecal {
                model,
                joint,
                material: mapped.then_some(decal.material),
                color,
                surface: material.surface(),
                blend: decal.blend,
                fade: crate::math::cos(decal.fade_angle),
            });
        }
    }
}
