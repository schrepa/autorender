//! Scene version 8 decals: an entity field projecting one of
//! the document's materials onto the opaque surfaces inside the entity's
//! box. Angles are in degrees.

use engine_core::{Decal, DecalBlend, MaterialHandle};
use serde::{Deserialize, Serialize};

use crate::{SceneError, SceneMaterial};

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

/// A decal on this entity. Requires version 8.
///
/// Its box is the entity's local unit cube (`[-0.5, 0.5]` on each axis):
/// scale the entity to size it. The material's color, base color map,
/// roughness, metalness and emission span the box's x and z, projected down
/// its y onto the opaque surfaces inside it that face its +y.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneDecal {
    /// ID of the projected material in this document.
    #[cfg_attr(
        feature = "schema",
        schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
    )]
    pub material: String,
    #[serde(default)]
    pub blend: SceneDecalBlend,
    /// Multiplies the decal's alpha.
    #[serde(default = "full_opacity")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub opacity: f32,
    /// Degrees between a surface's normal and the box's +y beyond which the
    /// decal fades out, gone at 90.
    #[serde(default = "default_fade")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 90.0)))]
    pub fade_degrees: f32,
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

    /// Validates the decal at authored path `path` against the document's
    /// `materials`; its material reference is checked with the entity.
    pub(crate) fn prepare(
        &self,
        path: &str,
        materials: &[SceneMaterial],
    ) -> Result<PreparedDecal, SceneError> {
        let invalid = |field: &str, rule: &str| {
            SceneError::new("invalid_decal", format!("{path}.{field}"), rule.to_owned())
        };
        if !(self.opacity.is_finite() && (0.0..=1.0).contains(&self.opacity)) {
            return Err(invalid("opacity", "must be finite and in [0, 1]"));
        }
        if !(self.fade_degrees.is_finite() && (0.0..=90.0).contains(&self.fade_degrees)) {
            return Err(invalid("fade_degrees", "must be finite and in [0, 90]"));
        }
        let material = materials
            .iter()
            .position(|material| material.id == self.material)
            .ok_or_else(|| invalid("material", "must name a material of this document"))?;
        Ok(PreparedDecal {
            material,
            blend: match self.blend {
                SceneDecalBlend::Alpha => DecalBlend::Alpha,
                SceneDecalBlend::Multiply => DecalBlend::Multiply,
            },
            opacity: self.opacity,
            // The runtime bound is exactly π/2.
            fade_angle: engine_core::math::min(
                self.fade_degrees.to_radians(),
                std::f32::consts::FRAC_PI_2,
            ),
        })
    }
}
