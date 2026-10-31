#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

use engine_core::{Brick, Bricks, DetailConsumer, DetailConsumersList, Join};
use engine_faces::places::WaterList;
use engine_render::render_bricks::AddRenderParts;
use engine_terrain::Water;

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
            .join(Join::<WaterList>::new())
            .settings::<Water>()
            .join(Join::<DetailConsumersList>::new().cpu(DetailConsumer {
                name: "live_cells",
                setting: "water.live_budget",
                stat: "water.cell_cut",
            }))
            .join(Join::<DetailConsumersList>::new().cpu(DetailConsumer {
                name: "foam_points",
                setting: "water.foam_points",
                stat: "water.foam_cut",
            }))
            .join(Join::<DetailConsumersList>::new().cpu(DetailConsumer {
                name: "water_leaves",
                setting: "water.subdivision_leaves",
                stat: "water.leaf_cut",
            }))
            .render_part(&water::PART)
            .shaders(shaders::MODULES)
            .places(engine_faces::places::PLACES)
            .pipeline_layouts(pipeline_layouts::PIPELINE_LAYOUTS)
            .mirrors(
                gpu_layouts::water::LAYOUTS,
                gpu_layouts::constants::DECLARED,
            )
            .mirrors(&[], gpu_layouts::constants::MIRRORED);
    }
}
