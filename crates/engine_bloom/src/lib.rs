use engine_core::{Brick, Bricks};
use engine_faces::slots::PostEffectsList;
use engine_render::render_bricks::AddRenderParts;

pub mod layouts;
mod part;
pub mod piece;
pub mod shaders;

pub struct BloomBrick;

impl Brick for BloomBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .list::<PostEffectsList>()
            .settings::<engine_core::Bloom>()
            .render_part(&part::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES)
            .pipeline_layouts(layouts::PIPELINE_LAYOUTS)
            .mirrors(piece::LAYOUTS, &[]);
    }
}
