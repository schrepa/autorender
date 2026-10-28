use engine_core::{Interior, ReflectionProbe, bevy_ecs::world::EntityWorldMut, glam::Vec3};
use engine_settings::settings;
use serde::{Deserialize, Serialize};

use crate::components::{Placement, Reach, SceneComponent};
use crate::instance::Lookups;
use crate::{SceneDocument, SceneError, SceneFiles};

settings! {
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct SceneReflectionProbe {
        pub extent: [f32; 3] = [1.0; 3] => above(0.0),
        #[serde(default = "default_blend")]
        pub blend: f32 = default_blend() => at_least(0.0),
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub interior: Option<Interior> = None => any,
    }
}

fn default_blend() -> f32 {
    1.0
}

impl SceneReflectionProbe {
    pub fn new(extent: [f32; 3]) -> Self {
        Self {
            extent,
            blend: default_blend(),
            interior: None,
        }
    }

    pub(crate) fn runtime(&self) -> ReflectionProbe {
        let mut probe = ReflectionProbe::new(Vec3::from_array(self.extent)).with_blend(self.blend);
        if let Some(interior) = self.interior {
            probe = probe.with_interior(interior);
        }
        probe
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

impl SceneComponent for SceneReflectionProbe {
    type Prepared = ReflectionProbe;

    const PLACEMENT: Placement = Placement::Anywhere;

    fn prepare(
        &self,
        _: &str,
        _: &SceneDocument,
        _: Option<&SceneFiles>,
    ) -> Result<ReflectionProbe, SceneError> {
        Ok(self.runtime())
    }

    fn spawn(prepared: &ReflectionProbe, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*prepared);
    }

    fn reach(prepared: &ReflectionProbe, reach: &Reach<'_>, extend: &mut impl FnMut(f32)) {
        extend(reach.from + prepared.extent.length());
    }
}
