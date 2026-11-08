#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod animator;
mod import;
mod scene;
mod skins;
mod smokes;
mod views;

use bevy_ecs::prelude::*;
use engine_core::{
    Asked, Brick, Bricks, ComponentKind, GpuStruct, Runs, SceneComponents, Schedule, Smoke, Source,
    settings,
};
use engine_places::skin::Skin;
use engine_places_rendering::animation::{Animate, Animator, Attachment};
use engine_render::RenderBricks;

pub use import::{ImportShared, Model, import_glb};
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
    const SMOKES: &'static [Smoke] = &[Smoke {
        name: "animation_plays",
        runs: Runs::Headless,
        run: smokes::plays,
    }];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|animation: Res<AnimationSettings>| animation.enabled)
            .settings::<AnimationSettings>()
            .resource::<animator::Books>()
            .resource::<skins::Skins>()
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
            .moving::<Animator>()
            .operation::<AnimationPlay>()
            .operation::<AnimationLayer>()
            .operation::<AnimationAim>()
            .observation::<AnimationState>()
            .on_remove::<Animator, _>(animator::release)
            .systems(
                Schedule::FixedPreSimulation,
                (animator::command, animator::advance),
            )
            .systems(
                Schedule::Presentation,
                (animator::sync, animator::advance_real),
            )
            .systems(Schedule::ViewSync, skins::pose_skins)
            .generated(GENERATED)
            .part::<skins::SkinGpu>();
    }
}
