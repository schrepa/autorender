#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod animator;
mod import;
mod part;
mod scene;
mod smokes;
mod views;

use bevy_ecs::prelude::*;
use engine_core::{
    Brick, Bricks, ComponentKind, GpuStruct, Runs, SceneComponents, Schedule, Smoke, Source,
};
use engine_places::animation::{Animator, Attachment};
use engine_places::shadows::ShadowCasters;
use engine_render::{Layer, MainView, RenderBricks};

pub use import::{ImportShared, Model, import_glb};
pub use views::{
    AnimationAim, AnimationLayer, AnimationPlay, AnimationState, AnimationStateView, ClipView,
    LayerView, PlayingView,
};

const GENERATED: &[&str] = &[
    part::WGSL,
    <part::SkinFrame as GpuStruct>::WGSL,
    <part::SkinPart as GpuStruct>::WGSL,
    <part::RestVertex as GpuStruct>::WGSL,
    <part::JointRows as GpuStruct>::WGSL,
];

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
            .when(animated)
            .resource::<part::Books>()
            .join::<SceneComponents>(
                ComponentKind::of::<Animator>("animator")
                    .check(scene::check_animator)
                    .schema(scene::animator_schema)
                    .example(scene::animator_example),
                None,
            )
            .join::<SceneComponents>(
                ComponentKind::of::<Attachment>("attachment")
                    .schema(scene::attachment_schema)
                    .example(scene::attachment_example),
                None,
            )
            .operation::<AnimationPlay>()
            .operation::<AnimationLayer>()
            .operation::<AnimationAim>()
            .observation::<AnimationState>()
            .systems(
                Schedule::FixedPostSimulation,
                (animator::advance, animator::follow_joints),
            )
            .systems(Schedule::Presentation, animator::publish_assets)
            .off(animator::forget)
            .generated(GENERATED)
            .draws::<part::SkinningGpu, MainView>(Layer::Opaque, part::SkinningGpu::draw_lit)
            .draws::<part::SkinningGpu, ShadowCasters>((), part::SkinningGpu::draw_casters);
    }
}

fn animated(animators: Query<(), With<Animator>>) -> bool {
    !animators.is_empty()
}
