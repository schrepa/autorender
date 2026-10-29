use engine_core::{Brick, Bricks, Slot};
use engine_faces::slots::SkySlot;

pub struct SkyBrick;

impl Brick for SkyBrick {
    const SLOT: Option<&'static str> = Some(SkySlot::NAME);

    fn register(_bricks: &mut Bricks<'_>) {}
}
