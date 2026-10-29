use engine_core::{Brick, Bricks, Slot};
use engine_faces::slots::LightingSlot;

pub struct ToonLightingBrick;

impl Brick for ToonLightingBrick {
    const SLOT: Option<&'static str> = Some(LightingSlot::NAME);

    fn register(_bricks: &mut Bricks<'_>) {}
}
