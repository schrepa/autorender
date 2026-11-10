#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod gpu;
mod part;
mod settings;

use bevy_ecs::prelude::*;
use engine_core::{
    Brick, Bricks, Pass, PassName, PassPurpose, PassViews, Schedule, SlotSet, Source, Writes, never,
};
use engine_render::{AttachPoint, RenderBricks, TargetDecl, TargetName, wgpu};

pub use settings::{Reflections, ReflectionsStats};

use gpu::ReflectionsPiece;

pub(crate) struct ReflectionsPass;

impl PassName for ReflectionsPass {
    const NAME: &'static str = "reflections";
    const PASS: Pass = Pass {
        writes: Writes::NOTHING,
        reads: SlotSet::NONE,
        views: PassViews::Main,
        at: AttachPoint::Post,
        purpose: PassPurpose::Scene,
        takes: never,
    };
}

const SURFACE_POINTS: &[AttachPoint] = &[AttachPoint::Opaque, AttachPoint::LateOpaque];

const SURFACE: TargetDecl = TargetDecl {
    name: TargetName("surface"),
    format: wgpu::TextureFormat::Rgba16Uint,
    written: SURFACE_POINTS,
    module: Some(shaders::reflections::surface::MODULE),
};

const REFLECTED: TargetDecl = TargetDecl {
    name: TargetName("reflected"),
    format: wgpu::TextureFormat::Rgba16Float,
    written: SURFACE_POINTS,
    module: Some(shaders::reflections::surface::MODULE),
};

pub struct ReflectionsBrick;

impl Brick for ReflectionsBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<Reflections>()
            .stats::<ReflectionsStats>()
            .piece::<ReflectionsPiece>()
            .generated(gpu::GENERATED)
            .view_target(SURFACE)
            .view_target(REFLECTED)
            .pass::<ReflectionsPass>()
            .systems(Schedule::ViewSync, publish_piece)
            .within(reflecting, |bricks| {
                bricks
                    .join::<engine_places_rendering::reflections::TraceReaders>((), None)
                    .part::<part::TraceGpu>();
            });
    }
}

fn reflecting(reflections: Res<Reflections>) -> bool {
    reflections.enabled
}

fn publish_piece(
    reflections: Res<Reflections>,
    mut piece: ResMut<ReflectionsPiece>,
    mut stats: ResMut<ReflectionsStats>,
) {
    let active = reflections.enabled;
    piece.set_if_neq(ReflectionsPiece {
        surface: [reflections.max_roughness, 0.0, 0.0, 0.0],
        trace: [reflections.distance, 0.0, 0.0, 0.0],
        applied: [
            reflections.distance,
            reflections.intensity,
            if active { 1.0 } else { 0.0 },
            0.0,
        ],
    });
    stats.set_if_neq(ReflectionsStats { active });
}
