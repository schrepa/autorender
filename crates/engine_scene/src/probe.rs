//! Scene reflection probes: an entity field
//! capturing the entity's surroundings into an environment map, which
//! surfaces inside its world box reflect. It mirrors the runtime component
//! field for field, so inspected values paste back into scenes.

use engine_core::{Interior, ReflectionProbe, glam::Vec3};
use engine_settings::ErrorCode;
use engine_settings::settings;
use serde::{Deserialize, Serialize};

use crate::SceneError;
use crate::error::broken_rule;

settings! {
    /// A reflection probe on this entity.
    ///
    /// Its box is axis-aligned in world space and centered on the entity's
    /// world position, so the entity's rotation and scale leave it unchanged.
    /// Surfaces inside it reflect what the probe captured from its center
    /// (still objects and still lights), fading out over `blend` toward its
    /// faces into the probes around it or the sky.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct SceneReflectionProbe {
        /// Half the box's size along each world axis in world units.
        pub extent: [f32; 3] = [1.0; 3] => above(0.0),
        /// World distance inside the box's faces over which the probe fades
        /// out.
        #[serde(default = "default_blend")]
        pub blend: f32 = default_blend() => at_least(0.0),
        /// Makes the box an interior: a cave or a hall letting in
        /// only these shares of the sky's light and the sun's.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub interior: Option<Interior> = None => any,
    }
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
        self.validate()
            .map_err(|invalid| broken_rule(ErrorCode::InvalidReflectionProbe, path, invalid))?;
        let mut probe = ReflectionProbe::new(Vec3::from_array(self.extent)).with_blend(self.blend);
        if let Some(interior) = self.interior {
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
