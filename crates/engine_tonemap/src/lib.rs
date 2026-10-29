use engine_core::{Brick, Bricks, Slot};
use engine_faces::slots::TonemapSlot;

pub struct TonemapBrick;

impl Brick for TonemapBrick {
    const SLOT: Option<&'static str> = Some(TonemapSlot::NAME);

    fn register(_bricks: &mut Bricks<'_>) {}
}
