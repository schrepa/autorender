#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

mod antialiasing;
mod bindings;
pub mod brick_smokes;
mod gpu_layouts;
mod pipeline_layouts;
mod previous;
pub mod shaders;
mod temporal;

use engine_core::{Brick, Bricks, Fill};
use engine_faces::places::AntialiasingSlot;
use engine_render::render_bricks::AddRenderParts;

pub struct TaaBrick;

impl Brick for TaaBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .fill(Fill::<AntialiasingSlot>::new())
            .settings::<engine_core::TemporalAntialiasing>()
            .smoke(
                "temporal_antialiasing",
                true,
                engine_core::brick_smokes::temporal_antialiasing,
            )
            .render_part(&temporal::PART)
            .shaders(shaders::MODULES)
            .places(engine_faces::places::PLACES)
            .pipeline_layouts(pipeline_layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::LAYOUTS, &[]);
    }
}
