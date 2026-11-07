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
    Asked, Attributes, Brick, Bricks, ComponentKind, GpuStruct, KindDecl, Runs, SceneComponents,
    Schedule, Smoke, Source, settings,
};
use engine_places_rendering::animation::{Animate, Animator, Attachment};
use engine_render::RenderBricks;

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
    <part::SkinBound as GpuStruct>::WGSL,
    <part::SkinView as GpuStruct>::WGSL,
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
                (animator::sync, animator::advance_real),
            )
            .generated(GENERATED)
            .draw_kind(KindDecl {
                kind: part::SKIN,
                vertex: shaders::animation::draw::VS_SKINNED.name(),
                needs: Attributes::BASE,
            })
            .draws::<part::SkinningGpu>(part::SKIN, part::SkinningGpu::draw);
    }
}
