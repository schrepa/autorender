#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

pub mod debug_view;
pub mod gpu_layouts;
pub mod pipeline_layouts;
pub mod shaders;

use engine_core::{Brick, Bricks, DebugView, Join};
use engine_faces::places::DebugViewList;
use engine_render::render_bricks::AddRenderParts;

pub struct DebugViewBrick;

impl Brick for DebugViewBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join(Join::<DebugViewList>::new())
            .settings::<DebugView>()
            .smoke("debug_view", true, engine_core::debug_view_smoke::run)
            .render_part(&debug_view::PART)
            .shaders(shaders::MODULES)
            .pipeline_layouts(pipeline_layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::LAYOUTS, gpu_layouts::DECLARED);
    }
}
