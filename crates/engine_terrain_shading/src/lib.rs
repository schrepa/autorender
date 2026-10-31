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

use engine_core::{Brick, Bricks, Fill, Join};
use engine_faces::heightfield::HeightfieldFace;
use engine_faces::places::{GroundTilesSlot, HeightfieldSlot, HorizonsSlot, TerrainShadingList};
use engine_render::render_bricks::AddRenderParts;

use crate::terrain::TerrainPass;

pub struct TerrainShadingBrick;

impl Brick for TerrainShadingBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join(Join::<TerrainShadingList>::new())
            .fill(Fill::<GroundTilesSlot>::new().wgsl("terrain::faces::ground_tiles"))
            .fill(
                Fill::<HeightfieldSlot>::new()
                    .wgsl("terrain::faces::heightfield")
                    .gpu(engine_render::offer_face!(TerrainPass => dyn HeightfieldFace)),
            )
            .fill(Fill::<HorizonsSlot>::new().wgsl("terrain::faces::horizons"))
            .render_part(&terrain::PART)
            .shaders(shaders::MODULES)
            .places(engine_faces::places::PLACES)
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
