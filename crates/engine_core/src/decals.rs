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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DecalBlend {
    #[default]
    Alpha,
    Multiply,
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Decal {
    pub material: MaterialHandle,
    pub blend: DecalBlend,
    pub opacity: f32,
    pub fade_angle: f32,
}

impl Decal {
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

    pub const OPACITY: Rule = Rule::Range(0.0, 1.0);
    pub const MAX_FADE_DEGREES: f32 = 90.0;

    pub fn validate(&self) -> Result<(), InvalidSetting> {
        self.opacity.check("opacity", Self::OPACITY)?;
        self.fade_angle.check(
            "fade_angle",
            Rule::Range(0.0, f64::from(Self::MAX_FADE_DEGREES.to_radians())),
        )
    }
}

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneDecals")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct DecalSettings {
        pub enabled: bool = true => any,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_decals: Option<u32> = None => budget,
    }
}

impl DecalSettings {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderDecal {
    pub model: Mat4,
    pub joint: Option<u32>,
    pub material: Option<MaterialHandle>,
    pub color: [f32; 4],
    pub surface: Surface,
    pub blend: DecalBlend,
    pub fade: f32,
}

#[derive(Clone, Debug, Default)]
pub struct RenderDecals {
    pub list: Vec<RenderDecal>,
    pub materials: Vec<RenderMaterial>,
    pub textures: Vec<RenderTexture>,
    pub dropped: usize,
    pub invalid: usize,
}

pub(crate) struct DecalExtraction {
    query: QueryState<(Entity, &'static Decal)>,
    hints: Vec<TransformIndex>,
    candidates: Vec<(f32, Mat4, u32, Decal)>,
    nearest: Nearest,
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
        let settings = snapshot.environment.decals;
        snapshot.decals.list.clear();
        snapshot.decals.materials.clear();
        snapshot.decals.textures.clear();
        snapshot.decals.dropped = 0;
        snapshot.decals.invalid = 0;
        if !settings.enabled {
            return;
        }
        let poses = Poses::of(world, gpu);
        let assets = world.resource::<AssetLibrary>();
        let eye = snapshot.detail_eye;
        self.candidates.clear();
        for (position, (entity, decal)) in self.query.iter(world).enumerate() {
            if decal.validate().is_err() {
                snapshot.decals.invalid += 1;
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
        snapshot.decals.dropped = self.candidates.len().saturating_sub(kept);
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
                    .decals
                    .materials
                    .iter()
                    .any(|listed| listed.handle == decal.material)
                {
                    snapshot.decals.materials.push(RenderMaterial {
                        handle: decal.material,
                        material: *material,
                        version: assets.material_version(decal.material).unwrap_or(0),
                    });
                    for texture in material.textures() {
                        if !snapshot
                            .decals
                            .textures
                            .iter()
                            .any(|listed| listed.handle == texture)
                        {
                            snapshot.decals.textures.push(RenderTexture {
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
            snapshot.decals.list.push(RenderDecal {
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
