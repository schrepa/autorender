#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

mod gpu_layouts;
pub mod shaders;
mod tonemap;

use engine_core::{Brick, Bricks, Place};
use engine_faces::slots::TonemapSlot;
use engine_render::render_bricks::AddRenderParts;

pub struct TonemapBrick;

impl Brick for TonemapBrick {
    const SLOT: Option<Place> = Some(Place::slot::<TonemapSlot>());

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<engine_core::Grading>()
            .render_part(&tonemap::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES)
            .mirrors(gpu_layouts::LAYOUTS, &[]);
    }
}
