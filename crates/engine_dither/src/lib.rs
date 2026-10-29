use engine_core::{Brick, Bricks};
use engine_faces::slots::PostEffectsList;
use engine_render::render_bricks::AddRenderParts;

pub mod gpu_layouts;
mod part;
pub mod shaders;

pub struct DitherBrick;

impl Brick for DitherBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .list::<PostEffectsList>()
            .settings::<engine_core::Dither>()
            .render_part(&part::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES)
            .mirrors(gpu_layouts::LAYOUTS, &[]);
    }
}
