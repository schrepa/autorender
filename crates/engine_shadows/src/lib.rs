#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

mod bindings;
mod gpu_layouts;
mod pipeline_layouts;
pub mod shaders;
mod shadow_fit;
mod shadows;

use engine_core::{Brick, Bricks, DetailConsumer, Place};
use engine_faces::slots::ShadowsSlot;
use engine_render::render_bricks::AddRenderParts;

pub struct ShadowsBrick;

impl Brick for ShadowsBrick {
    const SLOT: Option<Place> = Some(Place::slot::<ShadowsSlot>());

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<engine_core::Shadows>()
            .settings::<engine_core::SkyOcclusion>()
            .detail_consumer(DetailConsumer {
                name: "shadow_tiles",
                setting: "shadows.atlas_slots",
                stat: "shadows.tile_cut",
            })
            .detail_consumer(DetailConsumer {
                name: "moving_tiles",
                setting: "shadows.moving_tiles",
                stat: "shadows.moving_cut",
            })
            .render_part(&shadows::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES)
            .pipeline_layouts(pipeline_layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::LAYOUTS, gpu_layouts::DECLARED)
            .mirrors(&[], gpu_layouts::MIRRORED);
    }
}
