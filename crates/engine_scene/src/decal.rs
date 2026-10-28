use engine_core::{Decal, DecalBlend, MaterialHandle, bevy_ecs::world::EntityWorldMut, glam::Vec3};
use engine_settings::{Setting, settings};
use serde::{Deserialize, Serialize};

use crate::components::{Placement, Reach, SceneComponent};
use crate::instance::Lookups;
use crate::{SceneDocument, SceneError, SceneFiles, SceneMaterial};

impl Setting for SceneDecalBlend {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SceneDecalBlend {
    #[default]
    Alpha,
    Multiply,
}

settings! {
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct SceneDecal {
        #[cfg_attr(
            feature = "schema",
            schemars(transform = engine_settings::stated(engine_settings::rule::id()))
        )]
        pub material: String = String::new() => any,
        #[serde(default)]
        pub blend: SceneDecalBlend = SceneDecalBlend::Alpha => any,
        #[serde(default = "full_opacity")]
        pub opacity: f32 = full_opacity() => same(Some(Decal::OPACITY)),
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct PreparedDecal {
    pub material: usize,
    blend: DecalBlend,
    opacity: f32,
    fade_angle: f32,
}

impl PreparedDecal {
    pub(crate) fn runtime(&self, material: MaterialHandle) -> Decal {
        Decal::new(material)
            .with_blend(self.blend)
            .with_opacity(self.opacity)
            .with_fade_angle(self.fade_angle)
    }
}

impl SceneDecal {
    pub fn new(material: impl Into<String>) -> Self {
        Self {
            material: material.into(),
            blend: SceneDecalBlend::Alpha,
            opacity: full_opacity(),
            fade_degrees: default_fade(),
        }
    }

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

    fn reach(_: &PreparedDecal, reach: &Reach<'_>, extend: &mut impl FnMut(f32)) {
        extend(reach.of_box(Vec3::splat(-0.5), Vec3::splat(0.5)));
    }
}
