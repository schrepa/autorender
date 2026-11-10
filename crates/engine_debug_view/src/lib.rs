#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod gpu;
mod modes;
mod probe;
mod settings;

use bevy_ecs::prelude::Res;
use engine_core::{
    AttachPoint, Brick, Bricks, Pass, PassName, PassPurpose, PassViews, Schedule, SlotSet, Source,
    Writes, every,
};
use engine_render::{
    DEBUG_TARGETS, DebugModes, DebugShown, RenderBricks, TargetDecl, TargetName, wgpu,
};

pub use settings::DebugView;

use gpu::{DebugViewPiece, OverlayGpu};
use probe::{Probe, ProbeGpu, ProbePixel, ProbedPixel};

pub(crate) struct DebugViewPass;

impl PassName for DebugViewPass {
    const NAME: &'static str = "debug_view";
    const PASS: Pass = Pass {
        writes: Writes(&[
            DEBUG_TARGETS[0],
            DEBUG_TARGETS[1],
            DEBUG_TARGETS[2],
            TargetName::DEPTH,
        ]),
        reads: SlotSet::ALL,
        views: PassViews::Main,
        at: AttachPoint::Opaque,
        purpose: PassPurpose::Debug,
        takes: every,
    };
}

const fn shown_target(name: TargetName, format: wgpu::TextureFormat) -> TargetDecl {
    TargetDecl {
        name,
        format,
        written: &[],
        module: None,
    }
}

const TARGETS: [TargetDecl; 3] = [
    shown_target(DEBUG_TARGETS[0], wgpu::TextureFormat::Rgba32Uint),
    shown_target(DEBUG_TARGETS[1], wgpu::TextureFormat::R16Float),
    shown_target(DEBUG_TARGETS[2], wgpu::TextureFormat::Rg32Uint),
];

fn debugging(view: Res<DebugView>, probe: Option<Res<Probe>>) -> bool {
    view.showing() || probe.is_some_and(|probe| probe.probing())
}

pub struct DebugViewBrick;

impl Brick for DebugViewBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<DebugView>()
            .named::<DebugView, DebugModes>("view", &[settings::OFF], modes::name)
            .resource::<Probe>()
            .resource::<modes::Whole>()
            .piece::<DebugViewPiece>()
            .generated(gpu::GENERATED)
            .fill::<DebugShown>(Some(shaders::debug_view::shown::MODULE))
            .within(debugging, |bricks| {
                bricks.pass::<DebugViewPass>();
            })
            .part::<OverlayGpu>();
        for target in TARGETS {
            bricks.view_target(target);
        }
        for (index, (name, _)) in modes::OWN.into_iter().enumerate() {
            let wgsl = (index == 0).then_some(shaders::debug_view::modes::MODULE);
            bricks.join::<DebugModes>(modes::member(name), wgsl);
        }
        if engine_render::DEV {
            bricks
                .resource::<probe::ProbeSteps>()
                .systems(Schedule::ViewSync, (probe::advance, modes::publish))
                .operation::<ProbePixel>()
                .observation::<ProbedPixel>()
                .part::<ProbeGpu>();
        } else {
            bricks.systems(Schedule::ViewSync, modes::publish);
        }
    }
}
