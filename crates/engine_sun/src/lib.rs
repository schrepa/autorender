use engine_core::{Brick, Bricks, Slot};
use engine_faces::slots::SunSlot;

pub struct SunBrick;

impl Brick for SunBrick {
    const SLOT: Option<&'static str> = Some(SunSlot::NAME);

    fn register(_bricks: &mut Bricks<'_>) {}
}
