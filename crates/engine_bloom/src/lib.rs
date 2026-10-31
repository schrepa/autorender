#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine_core::{Brick, Bricks, Fill, Join};
use engine_faces::places::{BloomSlot, PostEffectsList};
use engine_render::composite::PostLight;
use engine_render::render_bricks::AddRenderParts;

pub mod gpu_layouts;
pub mod layouts;
mod part;
mod settings;
pub mod shaders;

pub use settings::Bloom;

pub struct BloomBrick;

impl Brick for BloomBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .fill(Fill::<BloomSlot>::new().wgsl("bloom::faces::bloom"))
            .join(Join::<PostLight>::new().wgsl(part::MEMBER))
            .join(Join::<PostEffectsList>::new())
            .settings::<Bloom>()
            .render_part(&part::PART)
            .shaders(shaders::MODULES)
            .places(engine_faces::places::PLACES)
            .pipeline_layouts(layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::LAYOUTS, &[]);
    }
}
