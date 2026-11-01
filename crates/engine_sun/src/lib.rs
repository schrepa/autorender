#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine_core::{Brick, Bricks, Fill};
use engine_faces::places::SunSlot;
use engine_render::render_bricks::AddRenderParts;

mod gpu_layouts;
mod light;
mod part;
mod shaders;

pub struct SunBrick;

impl Brick for SunBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<engine_core::Sunlight>()
            .render_part(&part::PART)
            .shaders(shaders::MODULES)
            .places(engine_faces::places::PLACES)
            .fill(
                Fill::<SunSlot>::new()
                    .wgsl("sun::faces::sun")
                    .gpu(part::OFFER),
            )
            .mirrors(gpu_layouts::LAYOUTS, &[]);
    }
}
