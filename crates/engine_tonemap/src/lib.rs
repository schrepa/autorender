#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

mod gpu_layouts;
mod light;
pub mod shaders;
mod tonemap;

use engine_core::{Brick, Bricks, Fill};
use engine_faces::places::TonemapSlot;
use engine_render::render_bricks::AddRenderParts;

pub struct TonemapBrick;

impl Brick for TonemapBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<engine_core::Grading>()
            .render_part(&tonemap::PART)
            .shaders(shaders::MODULES)
            .places(engine_faces::places::PLACES)
            .fill(Fill::<TonemapSlot>::new().wgsl("tonemap::faces::tonemap"))
            .mirrors(gpu_layouts::LAYOUTS, &[]);
    }
}
