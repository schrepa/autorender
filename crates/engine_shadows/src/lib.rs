#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

mod bindings;
mod constants;
mod effects;
mod gpu_layouts;
mod pipeline_layouts;
pub mod shaders;
mod shadow_fit;
mod shadows;
mod sky_occlusion;

use engine_core::{Brick, Bricks, DetailConsumer, DetailConsumersList, Fill, Join};
use engine_faces::places::ShadowsSlot;
use engine_render::render_bricks::AddRenderParts;

pub struct ShadowsBrick;

impl Brick for ShadowsBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<engine_core::Shadows>()
            .settings::<engine_core::SkyOcclusion>()
            .join(Join::<DetailConsumersList>::new().cpu(DetailConsumer {
                name: "shadow_tiles",
                setting: "shadows.atlas_slots",
                stat: "shadows.tile_cut",
            }))
            .join(Join::<DetailConsumersList>::new().cpu(DetailConsumer {
                name: "moving_tiles",
                setting: "shadows.moving_tiles",
                stat: "shadows.moving_cut",
            }))
            .render_part(&shadows::PART)
            .shaders(shaders::MODULES)
            .fill(
                Fill::<ShadowsSlot>::new()
                    .wgsl("shadows::faces::shadows")
                    .gpu(shadows::SHADOW_CASTERS),
            )
            .places(engine_faces::places::PLACES)
            .pipeline_layouts(pipeline_layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::LAYOUTS, gpu_layouts::DECLARED)
            .mirrors(&[], gpu_layouts::MIRRORED);
    }
}
