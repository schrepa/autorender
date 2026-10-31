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
mod settings;
pub mod shaders;

pub use settings::AutoExposure;

use engine_core::{Brick, Bricks, Fill};
use engine_faces::places::ExposureSlot;
use engine_render::render_bricks::AddRenderParts;

pub struct AutoExposureBrick;

impl Brick for AutoExposureBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<AutoExposure>()
            .render_part(&exposure::PART)
            .shaders(shaders::MODULES)
            .places(engine_faces::places::PLACES)
            .fill(Fill::<ExposureSlot>::new().gpu(exposure::OFFER))
            .pipeline_layouts(pipeline_layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::LAYOUTS, gpu_layouts::DECLARED);
    }
}
