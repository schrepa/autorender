#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod gpu;
mod part;
mod settings;
mod world_trace;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Schedule, Source};
use engine_places::ground::{GroundTiles, TerrainTiles};
use engine_places::reflections::Reflected;
use engine_render::{DebugShown, Layer, Output, PostScene, RenderBricks, ViewOutputs, wgpu};

pub use settings::{Reflections, ReflectionsStats};

use gpu::ReflectionsPiece;

const SURFACE: Output = Output {
    name: "surface",
    format: wgpu::TextureFormat::Rgba16Uint,
    layers: &[Layer::Opaque, Layer::Late, Layer::Backdrop, Layer::Media],
};

pub struct ReflectionsBrick;

impl Brick for ReflectionsBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        use shaders::reflections::{apply, faces, surface};
        bricks
            .settings::<Reflections>()
            .stats::<ReflectionsStats>()
            .piece::<ReflectionsPiece>()
            .join::<ViewOutputs>(SURFACE, Some(surface::MODULE))
            .systems(Schedule::ViewSync, publish_piece)
            .within(reflecting, |bricks| {
                bricks
                    .fill::<Reflected>(Some(faces::reflections::MODULE))
                    .join::<PostScene>((), Some(apply::MODULE))
                    .part::<part::TraceGpu>();
            });
    }
}

fn reflecting(reflections: Res<Reflections>) -> bool {
    reflections.reflects()
}

fn publish_piece(
    reflections: Res<Reflections>,
    tiles: Res<TerrainTiles>,
    ground: Res<GroundTiles>,
    shown: Res<DebugShown>,
    mut piece: ResMut<ReflectionsPiece>,
    mut stats: ResMut<ReflectionsStats>,
) {
    let active = reflections.reflects() && shown.owner.is_none();
    let walked = world_trace::traced(&tiles, &ground).count() as f32;
    piece.set_if_neq(ReflectionsPiece {
        surface: [
            reflections.max_roughness,
            if reflections.reflects() { 1.0 } else { 0.0 },
            0.0,
            0.0,
        ],
        trace: [reflections.distance, 0.0, 0.0, walked],
        applied: [
            reflections.distance,
            reflections.intensity,
            if active { 1.0 } else { 0.0 },
            0.0,
        ],
    });
    stats.set_if_neq(ReflectionsStats { active });
}
