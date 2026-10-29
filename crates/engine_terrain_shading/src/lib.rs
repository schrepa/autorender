#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod bindings;
mod entries;
pub mod gpu_layouts;
mod pipeline_layouts;
pub mod shaders;
mod terrain;
mod terrain_bounce;
mod terrain_horizons;
mod terrain_macro;
mod terrain_maps;
mod terrain_sky;

use engine_core::{Brick, Bricks};
use engine_faces::slots::TerrainShadingList;
use engine_render::render_bricks::AddRenderParts;

pub struct TerrainShadingBrick;

impl Brick for TerrainShadingBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .list::<TerrainShadingList>()
            .render_part(&terrain::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES)
            .pipeline_layouts(pipeline_layouts::PIPELINE_LAYOUTS)
            .mirrors(
                engine_faces::heightfield::LAYOUTS,
                gpu_layouts::constants::DECLARED,
            )
            .mirrors(
                gpu_layouts::terrain::LAYOUTS,
                gpu_layouts::constants::MIRRORED,
            )
            .mirrors(gpu_layouts::bounce::LAYOUTS, &[]);
    }
}
