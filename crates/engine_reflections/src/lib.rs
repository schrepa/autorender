#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod gpu;
mod part;
mod settings;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Schedule, Source, View};
use engine_render::{Layer, Output, PostScene, RenderBricks, ViewOutputs, wgpu};

pub use settings::{Reflections, ReflectionsStats};

use gpu::ReflectionsPiece;

const SURFACE_LAYERS: &[Layer] = &[Layer::Opaque, Layer::Late, Layer::Backdrop, Layer::Media];

const SURFACE: Output = Output {
    name: "surface",
    format: wgpu::TextureFormat::Rgba16Uint,
    layers: SURFACE_LAYERS,
};

const REFLECTIVITY: Output = Output {
    name: "reflectivity",
    format: wgpu::TextureFormat::R8Unorm,
    layers: SURFACE_LAYERS,
};

pub struct ReflectionsBrick;

impl Brick for ReflectionsBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        use shaders::reflections::{apply, surface};
        bricks
            .settings::<Reflections>()
            .stats::<ReflectionsStats>()
            .piece::<ReflectionsPiece>()
            .generated(gpu::GENERATED)
            .join::<ViewOutputs>(SURFACE, Some(surface::MODULE))
            .join::<ViewOutputs>(REFLECTIVITY, Some(surface::MODULE))
            .systems(Schedule::ViewSync, publish_piece)
            .within(reflecting, |bricks| {
                bricks
                    .join::<PostScene>((), Some(apply::MODULE))
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
    view: Res<View>,
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
