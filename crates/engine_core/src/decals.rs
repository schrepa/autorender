//! Decals: materials projected onto the opaque surfaces inside
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

use crate::{
    AssetLibrary, MaterialHandle, RenderJoint, RenderMaterial, RenderSnapshot, RenderTexture,
    Surface, TransformIndex,
    extraction::{Nearest, Poses},
    retained_render::{IDENTITY_ROWS, pose_rows},
};
use engine_settings::{InvalidSetting, Rule, Setting, settings};

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

/// Projects a material onto the opaque surfaces inside its entity's box.
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

    #[must_use]
    pub fn with_blend(mut self, blend: DecalBlend) -> Self {
        self.blend = blend;
        self
    }

    #[must_use]
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    #[must_use]
    pub fn with_fade_angle(mut self, radians: f32) -> Self {
        self.fade_angle = radians;
        self
    }

    /// The rule of [`Self::opacity`].
    pub const OPACITY: Rule = Rule::Range(0.0, 1.0);
    /// The widest [`Self::fade_angle`], in degrees: a right angle.
    pub const MAX_FADE_DEGREES: f32 = 90.0;

    /// The first value breaking its rule, if any: frames draw only valid
    /// decals and count the others.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        self.opacity.check("opacity", Self::OPACITY)?;
        self.fade_angle.check(
            "fade_angle",
            Rule::Range(0.0, f64::from(Self::MAX_FADE_DEGREES.to_radians())),
        )
    }
}

settings! {
    /// Decal settings.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneDecals")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct DecalSettings {
        /// Whether frames draw decals.
        pub enabled: bool = true => any,
        /// An optional budget of decals a frame draws; beyond it the decals
        /// nearest the camera win. Omitted, every decal draws.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_decals: Option<u32> = None => budget,
    }
}

impl DecalSettings {
    /// What a world without these settings reads as: no decals.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

/// One decal of a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderDecal {
    /// World matrix of the decal's box: its prepared pose, or its
    /// authoritative pose on GPU-presented frames.
    pub model: Mat4,
    /// The decal's entry in [`RenderSnapshot::joints`], whose palette holds
    /// its displayed pose (GPU-presented frames compose it as they do
    /// slots'); `None` draws `model`.
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

/// Decal records of each frame.
pub(crate) struct DecalExtraction {
    query: QueryState<(Entity, &'static Decal)>,
    hints: Vec<TransformIndex>,
    // This frame's usable decals: squared distance to the eye, world pose,
    // presentation node (GPU frames) and component.
    candidates: Vec<(f32, Mat4, u32, Decal)>,
    nearest: Nearest,
    // The joints the last extraction listed.
    joints: Vec<RenderJoint>,
}

impl DecalExtraction {
    pub(crate) fn new(world: &mut World) -> Self {
        Self {
            query: world.query::<(Entity, &Decal)>(),
            hints: Vec::new(),
            candidates: Vec::new(),
            nearest: Nearest::default(),
            joints: Vec::new(),
        }
    }

    /// Fills the snapshot's decals and their materials and textures, and
    /// appends their poses to the joints: prepared matrices, or on `gpu`
    /// frames the nodes the GPU composes. Beyond a budget the nearest win.
    /// Returns whether the joints they listed changed since the last call.
    pub(crate) fn extract(
        &mut self,
        world: &World,
        snapshot: &mut RenderSnapshot,
        gpu: bool,
    ) -> bool {
        let first = snapshot.joints.len();
        self.list(world, snapshot, gpu);
        let listed = &snapshot.joints[first..];
        if listed == self.joints.as_slice() {
            return false;
        }
        self.joints.clear();
        self.joints.extend_from_slice(listed);
        true
    }

    fn list(&mut self, world: &World, snapshot: &mut RenderSnapshot, gpu: bool) {
        // Extracted sanitized with the environment, before this.
        let settings = snapshot.environment.decals;
        snapshot.decals.clear();
        snapshot.decal_materials.clear();
        snapshot.decal_textures.clear();
        snapshot.dropped_decals = 0;
        snapshot.invalid_decals = 0;
        if !settings.enabled {
            return;
        }
        let poses = Poses::of(world, gpu);
        let assets = world.resource::<AssetLibrary>();
        let eye = snapshot.camera_position;
        self.candidates.clear();
        for (position, (entity, decal)) in self.query.iter(world).enumerate() {
            if decal.validate().is_err() {
                snapshot.invalid_decals += 1;
                continue;
            }
            if self.hints.len() <= position {
                self.hints.resize(position + 1, TransformIndex::default());
            }
            let hint = &mut self.hints[position];
            let Some(model) = poses.model_indexed(entity, hint) else {
                continue;
            };
            let node = if gpu { hint.0 as u32 } else { u32::MAX };
            let distance = model.w_axis.truncate().distance_squared(eye);
            self.candidates.push((distance, model, node, *decal));
        }
        let kept = settings
            .max_decals
            .map_or(usize::MAX, |budget| budget as usize);
        snapshot.dropped_decals = self.candidates.len().saturating_sub(kept);
        // Beyond the budget the nearest win; kept decals blend in their order.
        let candidates = &self.candidates;
        for position in self
            .nearest
            .kept_in_order(candidates.iter().map(|candidate| candidate.0), kept)
        {
            let (_, model, node, decal) = candidates[position];
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
                        version: assets.material_version(decal.material).unwrap_or(0),
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
            snapshot.joints.push(RenderJoint {
                node,
                inverse_bind: IDENTITY_ROWS,
                matrix: if gpu { IDENTITY_ROWS } else { pose_rows(model) },
            });
            let joint = Some(snapshot.joints.len() as u32 - 1);
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
