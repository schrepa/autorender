mod entries;
mod part;
pub mod shaders;

use engine_core::{Brick, Bricks, Slot};
use engine_faces::slots::MeshesSlot;
use engine_render::render_bricks::AddRenderParts;

pub struct MeshesBrick;

impl Brick for MeshesBrick {
    const SLOT: Option<&'static str> = Some(MeshesSlot::NAME);

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .render_part(&part::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES);
    }
}
