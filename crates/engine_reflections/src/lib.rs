#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine_core::{Brick, Bricks};
use engine_faces::slots::PostEffectsList;
use engine_render::render_bricks::AddRenderParts;

mod gpu_layouts;
mod layouts;
mod part;
mod shaders;
mod world_trace;

pub struct ReflectionsBrick;

impl Brick for ReflectionsBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .list::<PostEffectsList>()
            .settings::<engine_core::Reflections>()
            .render_part(&part::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES)
            .pipeline_layouts(layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::LAYOUTS, gpu_layouts::DECLARED);
    }
}
