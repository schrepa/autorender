#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

mod bindings;
mod exposure;
mod gpu_layouts;
mod pipeline_layouts;
pub mod shaders;

use engine_core::{Brick, Bricks, Place};
use engine_faces::slots::ExposureSlot;
use engine_render::render_bricks::AddRenderParts;

pub struct AutoExposureBrick;

impl Brick for AutoExposureBrick {
    const SLOT: Option<Place> = Some(Place::slot::<ExposureSlot>());

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<engine_core::AutoExposure>()
            .render_part(&exposure::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES)
            .pipeline_layouts(pipeline_layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::LAYOUTS, gpu_layouts::DECLARED);
    }
}
