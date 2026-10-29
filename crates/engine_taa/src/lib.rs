#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

mod bindings;
mod gpu_layouts;
mod pipeline_layouts;
pub mod shaders;
mod temporal;

use engine_core::{Brick, Bricks, Slot};
use engine_faces::slots::AntialiasingSlot;
use engine_render::render_bricks::AddRenderParts;

pub struct TaaBrick;

impl Brick for TaaBrick {
    const SLOT: Option<&'static str> = Some(AntialiasingSlot::NAME);

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<engine_core::TemporalAntialiasing>()
            .smoke(
                "temporal_antialiasing",
                true,
                engine_core::brick_smokes::temporal_antialiasing,
            )
            .render_part(&temporal::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES)
            .pipeline_layouts(pipeline_layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::LAYOUTS, &[]);
    }
}
