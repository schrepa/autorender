mod shaders;

use engine_core::{Brick, Bricks, Slot};
use engine_faces::slots::LightingSlot;
use engine_render::render_bricks::AddRenderParts;

pub struct ToonLightingBrick;

impl Brick for ToonLightingBrick {
    const SLOT: Option<&'static str> = Some(LightingSlot::NAME);

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES);
    }
}
