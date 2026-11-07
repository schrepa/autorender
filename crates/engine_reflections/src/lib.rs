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
    Brick, Bricks, Pass, PassName, PassViews, PrimaryView, Schedule, SlotSet, Source, Writes, never,
};
use engine_render::{AttachPoint, RenderBricks, TargetDecl, TargetName, Written, wgpu};

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
        takes: never,
    };
}

const SURFACE_POINTS: &[AttachPoint] = &[AttachPoint::Opaque, AttachPoint::Sky];

const SURFACE: TargetDecl = TargetDecl {
    name: TargetName("surface"),
    format: wgpu::TextureFormat::Rgba16Uint,
    written: Written::At(SURFACE_POINTS),
    module: Some(shaders::reflections::surface::MODULE),
};

const REFLECTIVITY: TargetDecl = TargetDecl {
    name: TargetName("reflectivity"),
    format: wgpu::TextureFormat::R8Unorm,
    written: Written::At(SURFACE_POINTS),
    module: Some(shaders::reflections::surface::MODULE),
};

pub struct ReflectionsBrick;

impl Brick for ReflectionsBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        use shaders::reflections::apply;
        bricks
            .settings::<Reflections>()
            .stats::<ReflectionsStats>()
            .piece::<ReflectionsPiece>()
            .generated(gpu::GENERATED)
            .view_target(SURFACE)
            .view_target(REFLECTIVITY)
            .pass::<ReflectionsPass>()
            .systems(Schedule::ViewSync, publish_piece)
            .within(reflecting, |bricks| {
                bricks
                    .applies::<ReflectionsPass>(apply::MODULE)
                    .join::<engine_places_rendering::reflections::TraceReaders>((), None)
                    .part::<part::TraceGpu>();
            });
    }
}

fn reflecting(reflections: Res<Reflections>) -> bool {
    reflections.reflects()
}

fn publish_piece(
    reflections: Res<Reflections>,
    view: Res<PrimaryView>,
    mut piece: ResMut<ReflectionsPiece>,
    mut stats: ResMut<ReflectionsStats>,
) {
    let active = reflections.reflects() && !view.debug();
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
