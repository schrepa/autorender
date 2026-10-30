#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod bindings;
mod clouds;
mod environment;
mod gpu_layouts;
mod haze;
mod pipeline_layouts;
pub mod shaders;
mod sky;

use engine_core::{Brick, Bricks, Place};
use engine_faces::slots::SkySlot;
use engine_render::render_bricks::AddRenderParts;

pub struct SkyBrick;

impl Brick for SkyBrick {
    const SLOT: Option<Place> = Some(Place::slot::<SkySlot>());

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<engine_core::Clouds>()
            .settings::<engine_core::Sky>()
            .settings::<engine_core::Haze>()
            .settings::<engine_core::NightSky>()
            .render_part(&sky::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES)
            .pipeline_layouts(pipeline_layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::LAYOUTS, gpu_layouts::DECLARED)
            .mirrors(&[], gpu_layouts::MIRRORED);
    }
}
