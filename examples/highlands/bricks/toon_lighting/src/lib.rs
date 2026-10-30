#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod shaders;

use engine_core::{Brick, Bricks, Place, SurfaceLight};
use engine_faces::slots::LightingSlot;
use engine_render::render_bricks::AddRenderParts;

pub struct ToonLightingBrick;

impl Brick for ToonLightingBrick {
    const SLOT: Option<Place> = Some(Place::slot::<LightingSlot>());

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<SurfaceLight>()
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES);
    }
}
