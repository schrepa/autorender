#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod batches;
pub mod brick_smokes;
mod depth_pyramid;
mod entries;
mod extra_views;
mod frame;
mod from_render {
    mod meshes;
}
mod gpu_culling;
mod gpu_presentation;
mod impostor_draws;
mod instances;
mod level_of_detail;
mod meshes;
mod opaque;
mod part;
mod previous;
mod retained_uploads;
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
