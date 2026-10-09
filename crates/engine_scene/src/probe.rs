//! Scene version 8 reflection probes: an entity field
//! capturing the entity's surroundings into an environment map, which
//! surfaces inside its world box reflect. It mirrors the runtime component
//! field for field, so inspected values paste back into scenes.

use engine_core::{Interior, ReflectionProbe, glam::Vec3};
use serde::{Deserialize, Serialize};

use crate::SceneError;

/// A reflection probe on this entity. Requires version 8.
///
/// Its box is axis-aligned in world space and centered on the entity's world
/// position, so the entity's rotation and scale leave it unchanged. Surfaces
/// inside it reflect what the probe captured from its center (still objects
/// and still lights), fading out over `blend` toward its faces into the
/// probes around it or the sky.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneReflectionProbe {
    /// Half the box's size along each world axis in world units, each
    /// finite and positive.
    pub extent: [f32; 3],
    /// World distance inside the box's faces over which the probe fades out;
    /// finite and nonnegative.
    #[serde(default = "default_blend")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0)))]
    pub blend: f32,
    /// Makes the box an interior: a cave or a hall letting in
    /// only these shares of the sky's light and the sun's. Requires
    /// version 9.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interior: Option<Interior>,
}

fn default_blend() -> f32 {
    1.0
}

impl SceneReflectionProbe {
    /// A probe covering the box `extent` from its entity along each world
    /// axis, fading over its outer world unit.
    pub fn new(extent: [f32; 3]) -> Self {
        Self {
            extent,
            blend: default_blend(),
            interior: None,
        }
    }

    /// The component at authored path `path`, validated.
    pub fn runtime(&self, path: &str) -> Result<ReflectionProbe, SceneError> {
        let invalid = |field: &str, rule: &str| {
            SceneError::new(
                "invalid_reflection_probe",
                format!("{path}.{field}"),
                rule.to_owned(),
            )
        };
        if !self
            .extent
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        {
            return Err(invalid(
                "extent",
                "must be finite and positive on every axis",
            ));
        }
        if !(self.blend.is_finite() && self.blend >= 0.0) {
            return Err(invalid("blend", "must be finite and nonnegative"));
        }
        let mut probe = ReflectionProbe::new(Vec3::from_array(self.extent)).with_blend(self.blend);
        if let Some(interior) = self.interior {
            if !interior.is_usable() {
                return Err(invalid("interior", "sky and sun must be in [0, 1]"));
            }
            probe = probe.with_interior(interior);
        }
        Ok(probe)
    }
}

impl From<&ReflectionProbe> for SceneReflectionProbe {
    fn from(probe: &ReflectionProbe) -> Self {
        Self {
            extent: probe.extent.to_array(),
            blend: probe.blend,
            interior: probe.interior,
        }
    }
}
