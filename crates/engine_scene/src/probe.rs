use engine_core::{Error, Interior, ReflectionProbe, bevy_ecs::world::EntityWorldMut, glam::Vec3};
use engine_settings::{Rule, settings};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::components::{
    ComponentContext, ComponentKind, Placement, Prepared, PreparedComponent, Reach, admitted,
    no_files, no_paths, parsed, ruled, valid_world,
};
use crate::instance::Lookups;

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

pub(crate) static REFLECTION_PROBE: ComponentKind = ComponentKind {
    name: "reflection_probe",
    shows: "ReflectionProbe",
    placement: Placement::Anywhere,
    schema: crate::component_schema!(SceneReflectionProbe),
    example: probe_example,
    validate: validate_probe,
    validate_world: valid_world,
    prepare: prepare_probe,
    files: no_files,
    admit: admitted,
    paths: no_paths,
};

fn probe_example() -> Value {
    json!({ "extent": [1.0, 1.0, 1.0] })
}

fn validate_probe(value: &Value, path: &str) -> Result<(), Error> {
    ruled::<SceneReflectionProbe>(value, path, REFLECTION_PROBE.name, Rule::Any).map(drop)
}

fn prepare_probe(value: &Value, context: &ComponentContext<'_>) -> Prepared {
    Ok(Box::new(
        parsed::<SceneReflectionProbe>(value, context.path)?.runtime(),
    ))
}

impl PreparedComponent for ReflectionProbe {
    fn spawn(&self, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*self);
    }

    fn reach(&self, reach: &Reach<'_>, extend: &mut dyn FnMut(f32)) {
        extend(reach.from + self.extent.length());
    }
}
