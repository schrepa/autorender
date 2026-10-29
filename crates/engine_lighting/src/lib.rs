use engine_core::{Brick, Bricks, Slot};
use engine_faces::slots::LightingSlot;

pub struct LightingModelBrick;

impl Brick for LightingModelBrick {
    const SLOT: Option<&'static str> = Some(LightingSlot::NAME);

    fn register(_bricks: &mut Bricks<'_>) {}
}
