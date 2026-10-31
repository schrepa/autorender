#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod entries;
mod part;
pub mod shaders;

use engine_core::{Brick, Bricks, Fill};
use engine_faces::places::MeshesSlot;
use engine_render::render_bricks::AddRenderParts;

pub struct MeshesBrick;

impl Brick for MeshesBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .fill(Fill::<MeshesSlot>::new())
            .render_part(&part::PART)
            .shaders(shaders::MODULES)
            .places(engine_faces::places::PLACES);
    }
}
