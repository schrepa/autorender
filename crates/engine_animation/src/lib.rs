#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod animator;
mod import;
mod models;
mod scene;
mod skins;
mod views;

use bevy_ecs::prelude::*;
use engine_core::{
    Asked, Brick, Bricks, ComponentKind, GpuStruct, SceneComponents, Schedule, Source, settings,
};
use engine_places::skin::Skin;
use engine_places_rendering::animation::{Animate, Animator, Attachment, Model};
use engine_render::RenderBricks;

pub use import::{ImportShared, Model as ImportedModel, import_glb};
pub use models::{MODEL_PART, ModelLoaded};
pub use views::{
    AnimationAim, AnimationLayer, AnimationPlay, AnimationState, AnimationStateView, ClipView,
    LayerView, PlayingView,
};

const GENERATED: &[&str] = &[
    <skins::JointRows as GpuStruct>::WGSL,
    <skins::SkinSlot as GpuStruct>::WGSL,
];

settings! {
    #[section("animation")]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneAnimation")]
    pub struct AnimationSettings {
        pub enabled: bool = true => any,
    }
}

pub struct AnimationBrick;

impl Brick for AnimationBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|animation: Res<AnimationSettings>| animation.enabled)
            .settings::<AnimationSettings>()
            .resource::<animator::Books>()
            .resource::<skins::Skins>()
            .resource::<models::ModelLoads>()
            .fill::<Asked<Animate>>(None)
            .fill::<Skin>(Some(shaders::animation::faces::skin::MODULE))
            .join::<SceneComponents>(
                ComponentKind::of::<Animator>("animator")
                    .check(scene::check_animator)
                    .schema(scene::animator_schema),
                None,
            )
            .join::<SceneComponents>(
                ComponentKind::of::<Attachment>("attachment")
                    .schema(scene::attachment_schema),
                None,
            )
            .join::<SceneComponents>(
                ComponentKind::of::<Model>("model")
                    .check(models::check_model)
                    .files(models::model_files)
                    .prepare(models::prepare_model)
                    .schema(models::model_schema),
                None,
            )
            .moving::<Animator>()
            .operation::<AnimationPlay>()
            .operation::<AnimationLayer>()
            .operation::<AnimationAim>()
            .observation::<AnimationState>()
            .on_remove::<Animator, _>(animator::release)
            .on_remove::<Model, _>(models::release)
            .systems(
                Schedule::FixedPreSimulation,
                (animator::command, animator::advance),
            )
            .systems(
                Schedule::Presentation,
                (models::load_models, animator::sync, animator::advance_real),
            )
            .systems(Schedule::ViewSync, skins::pose_skins)
            .generated(GENERATED)
            .part::<skins::SkinGpu>();
    }
}
