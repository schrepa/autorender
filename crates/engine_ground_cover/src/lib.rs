#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine_core::{Brick, Bricks, DetailConsumer};
use engine_faces::slots::GroundCoverList;
use engine_render::render_bricks::AddRenderParts;
use engine_terrain::GroundCover;

mod bindings;
pub mod gpu_layouts;
mod ground_cover;
pub mod pipeline_layouts;
pub mod shaders;

pub struct GroundCoverBrick;

impl Brick for GroundCoverBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .list::<GroundCoverList>()
            .settings::<GroundCover>()
            .detail_consumer(DetailConsumer {
                name: "blades",
                setting: "ground_cover.max_blades",
                stat: "ground_cover.blade_cut",
            })
            .render_part(&ground_cover::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES)
            .pipeline_layouts(pipeline_layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::LAYOUTS, gpu_layouts::DECLARED);
    }
}
