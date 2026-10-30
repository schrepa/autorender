#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine_core::{Brick, Bricks, Place};
use engine_faces::slots::SunSlot;
use engine_render::render_bricks::AddRenderParts;

mod gpu_layouts;
mod part;
mod shaders;

pub struct SunBrick;

impl Brick for SunBrick {
    const SLOT: Option<Place> = Some(Place::slot::<SunSlot>());

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .resource::<engine_core::Atmosphere>()
            .render_part(&part::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES)
            .mirrors(gpu_layouts::LAYOUTS, &[]);
    }
}
