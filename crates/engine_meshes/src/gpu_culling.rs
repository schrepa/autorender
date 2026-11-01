use bevy_ecs::prelude::{Resource, World};
use serde::{Deserialize, Serialize};

use engine_core::bricks::{Brick, Bricks, DrawState, ExplainRecord, ExplainRequest};
use engine_core::sections::environment_section;
use engine_settings::settings;

pub struct Culling;

impl Brick for Culling {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<OcclusionCulling>()
            .settings::<SubpixelCulling>()
            .passes(&["occlusion_test", "opaque_phase_two"])
            .explain(explain_culled)
            .smoke(
                "culling_settings",
                true,
                crate::brick_smokes::culling_settings,
            );
    }
}

fn explain_culled(world: &World, request: &ExplainRequest) -> Option<ExplainRecord> {
    let ExplainRequest::Entity(entity) = request else {
        return None;
    };
    if entity.drawn != DrawState::Culled {
        return None;
    }
    Some(
        ExplainRecord::new("culled")
            .with("entity", &entity.id)
            .with("slot", entity.slot)
            .with(
                "occlusion_culling",
                world
                    .get_resource::<OcclusionCulling>()
                    .is_some_and(|culling| culling.enabled),
            )
            .with(
                "subpixel_culling",
                world
                    .get_resource::<SubpixelCulling>()
                    .is_some_and(|culling| culling.enabled),
            ),
    )
}

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneOcclusionCulling")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct OcclusionCulling {
        pub enabled: bool = true => any,
    }
}

environment_section!(OcclusionCulling, "occlusion_culling", opt_in: false);

impl OcclusionCulling {
    pub fn off() -> Self {
        Self { enabled: false }
    }
}

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneSubpixelCulling")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct SubpixelCulling {
        pub enabled: bool = true => any,
    }
}

environment_section!(SubpixelCulling, "subpixel_culling", opt_in: false);

impl SubpixelCulling {
    pub fn off() -> Self {
        Self { enabled: false }
    }
}
