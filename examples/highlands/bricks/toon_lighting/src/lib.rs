#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod shaders;

use engine_core::{Brick, Bricks, Fill, SurfaceLight};
use engine_faces::places::LightingSlot;
use engine_render::render_bricks::AddRenderParts;

pub struct ToonLightingBrick;

impl Brick for ToonLightingBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .fill(Fill::<LightingSlot>::new().wgsl("toon::faces::lighting"))
            .settings::<SurfaceLight>()
            .shaders(shaders::MODULES)
            .places(engine_faces::places::PLACES);
    }
}
