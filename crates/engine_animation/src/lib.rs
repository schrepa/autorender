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
    Asked, Brick, Bricks, ComponentKind, GpuStruct, Runs, SceneComponents, Schedule, Smoke, Source,
};
use engine_places_rendering::animation::{Animate, Animator, Attachment};
use engine_places_rendering::shadows::ShadowCasters;
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
            .resource::<animator::Books>()
            .fill::<Asked<Animate>>(None)
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
                (
                    animator::sync,
                    animator::advance_real,
                    animator::publish_assets,
                ),
            )
            .generated(GENERATED)
            .draws::<part::SkinningGpu, MainView>(Layer::Opaque, part::SkinningGpu::draw_lit)
            .draws::<part::SkinningGpu, ShadowCasters>((), part::SkinningGpu::draw_casters);
    }
}

fn animated(animators: Query<(), With<Animator>>) -> bool {
    !animators.is_empty()
}
