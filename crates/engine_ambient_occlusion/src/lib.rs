#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine_core::{Brick, Bricks};
use engine_faces::slots::PostEffectsList;
use engine_render::render_bricks::AddRenderParts;

pub mod layouts;
mod part;
pub mod shaders;

pub struct AmbientOcclusionBrick;

impl Brick for AmbientOcclusionBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .list::<PostEffectsList>()
            .settings::<engine_core::AmbientOcclusion>()
            .render_part(&part::PART)
            .shaders(shaders::MODULES)
            .pipeline_layouts(layouts::PIPELINE_LAYOUTS);
    }
}
