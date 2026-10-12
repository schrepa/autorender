//! Scene decals: an entity field projecting one of
//! the document's materials onto the opaque surfaces inside the entity's
//! box. Angles are in degrees.

use engine_core::{Decal, DecalBlend, MaterialHandle, bevy_ecs::world::EntityWorldMut, glam::Vec3};
use engine_settings::{Setting, settings};
use serde::{Deserialize, Serialize};

use crate::components::{Placement, Reach, SceneComponent};
use crate::instance::Lookups;
use crate::{SceneDocument, SceneError, SceneFiles, SceneMaterial};

impl Setting for SceneDecalBlend {}

/// How a decal combines with the surface it lies on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SceneDecalBlend {
    /// Covering by alpha, lit like any surface: paint, posters, markings.
    #[default]
    Alpha,
    /// Multiplying the surface's color, keeping its light: dirt, cracks,
    /// stains and scorch marks.
    Multiply,
}

settings! {
    /// A decal on this entity.
    ///
    /// Its box is the entity's local unit cube (`[-0.5, 0.5]` on each axis):
    /// scale the entity to size it. The material's color, base color map,
    /// roughness, metalness and emission span the box's x and z, projected
    /// down its y onto the opaque surfaces inside it that face its +y.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct SceneDecal {
        /// ID of the projected material in this document.
        #[cfg_attr(
            feature = "schema",
            schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
        )]
        pub material: String = String::new() => any,
        /// How the decal combines with the surface it lies on.
        #[serde(default)]
        pub blend: SceneDecalBlend = SceneDecalBlend::Alpha => any,
        /// Multiplies the decal's alpha.
        #[serde(default = "full_opacity")]
        pub opacity: f32 = full_opacity() => same(Some(Decal::OPACITY)),
        /// Degrees between a surface's normal and the box's +y beyond which
        /// the decal fades out, gone at 90.
        #[serde(default = "default_fade")]
        pub fade_degrees: f32 = default_fade() => range(0.0, Decal::MAX_FADE_DEGREES),
    }
}

fn full_opacity() -> f32 {
    1.0
}

fn default_fade() -> f32 {
    60.0
}

/// A validated decal of the document's `material`th material.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PreparedDecal {
    pub material: usize,
    blend: DecalBlend,
    opacity: f32,
    fade_angle: f32,
}

impl PreparedDecal {
    /// The component, with the material's resident handle.
    pub(crate) fn runtime(&self, material: MaterialHandle) -> Decal {
        Decal::new(material)
            .with_blend(self.blend)
            .with_opacity(self.opacity)
            .with_fade_angle(self.fade_angle)
    }
}

impl SceneDecal {
    /// An alpha decal of material `material` at full opacity, fading from
    /// 60 degrees.
    pub fn new(material: impl Into<String>) -> Self {
        Self {
            material: material.into(),
            blend: SceneDecalBlend::Alpha,
            opacity: full_opacity(),
            fade_degrees: default_fade(),
        }
    }

    /// The field of `decal`, whose material has authored ID `material`.
    pub fn from_runtime(decal: &Decal, material: impl Into<String>) -> Self {
        Self {
            material: material.into(),
            blend: match decal.blend {
                DecalBlend::Alpha => SceneDecalBlend::Alpha,
                DecalBlend::Multiply => SceneDecalBlend::Multiply,
            },
            opacity: decal.opacity,
            fade_degrees: decal.fade_angle.to_degrees(),
        }
    }

    /// The decal with its material's index in the document's `materials`;
    /// its rules and material reference are checked with the entity's.
    pub(crate) fn prepare(&self, materials: &[SceneMaterial]) -> PreparedDecal {
        let material = materials
            .iter()
            .position(|material| material.id == self.material)
            .expect("the entity's checks found the decal's material");
        PreparedDecal {
            material,
            blend: match self.blend {
                SceneDecalBlend::Alpha => DecalBlend::Alpha,
                SceneDecalBlend::Multiply => DecalBlend::Multiply,
            },
            opacity: self.opacity,
            // The runtime bound is exactly the widest fade in radians.
            fade_angle: engine_core::math::min(
                self.fade_degrees.to_radians(),
                Decal::MAX_FADE_DEGREES.to_radians(),
            ),
        }
    }
}

impl SceneComponent for SceneDecal {
    type Prepared = PreparedDecal;

    const PLACEMENT: Placement = Placement::Anywhere;

    fn prepare(
        &self,
        _: &str,
        document: &SceneDocument,
        _: Option<&SceneFiles>,
    ) -> Result<PreparedDecal, SceneError> {
        Ok(SceneDecal::prepare(self, &document.materials))
    }

    fn spawn(prepared: &PreparedDecal, entity: &mut EntityWorldMut<'_>, lookups: &Lookups) {
        entity.insert(prepared.runtime(lookups.owned_materials[prepared.material]));
    }

    // It projects inside its entity's unit box.
    fn reach(_: &PreparedDecal, reach: &Reach<'_>, extend: &mut impl FnMut(f32)) {
        extend(reach.of_box(Vec3::splat(-0.5), Vec3::splat(0.5)));
    }
}
