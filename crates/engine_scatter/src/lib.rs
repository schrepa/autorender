#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod bindings;
mod gpu_layouts;
mod occupancy;
mod pipeline_layouts;
mod scatter;
mod scatter_draws;
mod scatter_pool;
mod shaders;

use engine_core::{Brick, Bricks};
use engine_faces::slots::ScatterList;
use engine_render::render_bricks::AddRenderParts;

pub struct ScatterBrick;

impl Brick for ScatterBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .list::<ScatterList>()
            .settings::<engine_core::Foliage>()
            .smoke(
                "foliage_settings",
                true,
                engine_core::brick_smokes::foliage_settings,
            )
            .render_part(&scatter::PART)
            .render_part(&occupancy::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES)
            .pipeline_layouts(pipeline_layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::LAYOUTS, gpu_layouts::DECLARED)
            .mirrors(engine_faces::placed::LAYOUTS, &[]);
    }
}
