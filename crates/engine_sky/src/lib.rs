#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod bindings;
mod clouds;
mod environment;
mod from_core {
    mod clouds;
    mod sky;
}
mod gpu_layouts;
mod haze;
mod light;
mod pipeline_layouts;
pub mod shaders;
mod sky;
mod tables;
mod under_eye;
mod views;

use engine_core::{Brick, Bricks, Fill, Join, Member};
use engine_faces::places::SkySlot;
use engine_render::composite::PostLight;
use engine_render::render_bricks::AddRenderParts;

pub struct SkyBrick;

impl Brick for SkyBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<engine_core::Clouds>()
            .settings::<engine_core::Sky>()
            .settings::<engine_core::Haze>()
            .settings::<engine_core::NightSky>()
            .settings::<engine_core::SkyLighting>()
            .extract(under_eye::extract_surface_under_eye)
            .render_part(&sky::PART)
            .shaders(shaders::MODULES)
            .places(engine_faces::places::PLACES)
            .fill(
                Fill::<SkySlot>::new()
                    .wgsl("sky::faces::sky")
                    .gpu(sky::OFFER),
            )
            .join(Join::<PostLight>::new().wgsl(Member {
                module: "sky::haze",
                functions: &["apply"],
                reads: &["haze"],
            }))
            .pipeline_layouts(pipeline_layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::LAYOUTS, gpu_layouts::DECLARED)
            .mirrors(&[], gpu_layouts::MIRRORED);
    }
}
