#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine_core::{Brick, Bricks, Place};
use engine_faces::slots::LightsSlot;
use engine_render::render_bricks::AddRenderParts;

mod gpu_layouts;
mod part;
mod pipeline_layouts;
pub mod shaders;

pub struct LightsBrick;

impl Brick for LightsBrick {
    const SLOT: Option<Place> = Some(Place::slot::<LightsSlot>());

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<engine_core::LocalLightSettings>()
            .resource::<engine_core::LightExtractor>()
            .extract(engine_core::extract_lights)
            .render_part(&part::PART)
            .shaders(shaders::MODULES)
            .pipeline_layouts(pipeline_layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::LAYOUTS, gpu_layouts::DECLARED);
    }
}
