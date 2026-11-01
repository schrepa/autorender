#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod bindings;
pub mod brick_smokes;
mod foliage;
mod gpu_layouts;
mod occupancy;
mod pipeline_layouts;
mod runtime {
    mod occupancy;
}
mod scatter;
mod scatter_draws;
mod scatter_pool;
mod shaders;

use engine_core::{Brick, Bricks, Fill, Join};
use engine_faces::occupancy::OccupancyFace;
use engine_faces::placed::PlacedFace;
use engine_faces::places::{OccupancySlot, PlacedSlot, ScatterList};
use engine_render::render_bricks::AddRenderParts;

use crate::occupancy::OccupancyPass;
use crate::scatter::{PlacedCovers, ScatterCoversSlot, ScatterPass};

pub struct ScatterBrick;

impl Brick for ScatterBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join(Join::<ScatterList>::new())
            .fill(
                Fill::<OccupancySlot>::new()
                    .wgsl("scatter::faces::occupancy")
                    .gpu(engine_render::offer_face!(OccupancyPass => dyn OccupancyFace)),
            )
            .fill(
                Fill::<PlacedSlot>::new()
                    .wgsl("scatter::faces::placed")
                    .gpu(engine_render::offer_face!(ScatterPass => dyn PlacedFace)),
            )
            .fill(
                Fill::<ScatterCoversSlot>::new()
                    .gpu(engine_render::offer_face!(ScatterPass => dyn PlacedCovers)),
            )
            .settings::<engine_core::Foliage>()
            .settings::<engine_terrain::Occupancy>()
            .smoke(
                "foliage_settings",
                true,
                engine_core::brick_smokes::foliage_settings,
            )
            .render_part(&scatter::PART)
            .render_part(&occupancy::PART)
            .shaders(shaders::MODULES)
            .places(engine_faces::places::PLACES)
            .pipeline_layouts(pipeline_layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::LAYOUTS, gpu_layouts::DECLARED)
            .mirrors(engine_faces::placed::LAYOUTS, &[]);
    }
}
