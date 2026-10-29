use engine_core::{Brick, Bricks, Slot};
use engine_faces::slots::SunSlot;
use engine_render::render_bricks::AddRenderParts;

mod part;
mod piece;
mod shaders;

pub struct SunBrick;

impl Brick for SunBrick {
    const SLOT: Option<&'static str> = Some(SunSlot::NAME);

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .resource::<engine_core::Atmosphere>()
            .render_part(&part::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES)
            .mirrors(piece::LAYOUTS, &[]);
    }
}
