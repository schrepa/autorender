#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

use engine_core::{Brick, Bricks};
use engine_faces::slots::WaterList;
use engine_render::render_bricks::AddRenderParts;

include!(concat!(env!("OUT_DIR"), "/source.rs"));

mod bindings;
mod debug;
mod entries;
pub mod gpu_layouts;
pub mod pipeline_layouts;
pub mod shaders;
mod water;
mod water_falls;
mod water_kernels;
mod water_particles;
mod water_settle;
mod water_sim;

pub struct WaterBrick;

impl Brick for WaterBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .list::<WaterList>()
            .render_part(&water::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES)
            .pipeline_layouts(pipeline_layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::water::LAYOUTS, gpu_layouts::constants::DECLARED)
            .mirrors(&[], gpu_layouts::constants::MIRRORED);
    }
}
