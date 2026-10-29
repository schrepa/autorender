use engine_core::{Brick, Bricks, Slot};
use engine_faces::slots::LightsSlot;

pub struct LightsBrick;

impl Brick for LightsBrick {
    const SLOT: Option<&'static str> = Some(LightsSlot::NAME);

    fn register(_bricks: &mut Bricks<'_>) {}
}
