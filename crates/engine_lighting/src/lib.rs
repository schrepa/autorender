#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine_core::{Brick, Bricks, Fill};
use engine_faces::places::LightingSlot;
use engine_render::render_bricks::AddRenderParts;

mod gpu_layouts;
mod part;
mod shaders;

pub struct LightingModelBrick;

impl Brick for LightingModelBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<engine_core::SurfaceLight>()
            .settings::<engine_core::Highlights>()
            .render_part(&part::PART)
            .shaders(shaders::MODULES)
            .fill(Fill::<LightingSlot>::new().wgsl("lighting::faces::lighting"))
            .places(engine_faces::places::PLACES)
            .mirrors(gpu_layouts::LAYOUTS, &[]);
    }
}
