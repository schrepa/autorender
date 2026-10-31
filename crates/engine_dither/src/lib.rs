#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine_core::{Brick, Bricks, Fill, Join};
use engine_faces::places::{DitherSlot, PostEffectsList};
use engine_render::composite::PostDisplay;
use engine_render::render_bricks::AddRenderParts;

pub mod gpu_layouts;
mod part;
pub mod shaders;

pub struct DitherBrick;

impl Brick for DitherBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .fill(Fill::<DitherSlot>::new().wgsl("dither::faces::dither"))
            .join(Join::<PostDisplay>::new().wgsl(part::MEMBER))
            .join(Join::<PostEffectsList>::new())
            .settings::<engine_core::Dither>()
            .render_part(&part::PART)
            .shaders(shaders::MODULES)
            .places(engine_faces::places::PLACES)
            .mirrors(gpu_layouts::LAYOUTS, &[]);
    }
}
