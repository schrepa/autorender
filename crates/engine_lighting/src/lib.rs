use engine_core::{Brick, Bricks, Slot};
use engine_faces::slots::LightingSlot;
use engine_render::render_bricks::AddRenderParts;

mod part;
mod piece;
mod shaders;

pub struct LightingModelBrick;

impl Brick for LightingModelBrick {
    const SLOT: Option<&'static str> = Some(LightingSlot::NAME);

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<engine_core::Highlights>()
            .smoke(
                "foliage_settings",
                true,
                engine_core::brick_smokes::foliage_settings,
            )
            .render_part(&part::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES)
            .mirrors(piece::LAYOUTS, &[]);
    }
}
