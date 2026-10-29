use engine_core::{Brick, Bricks, Slot};
use engine_faces::slots::AntialiasingSlot;

pub struct TaaBrick;

impl Brick for TaaBrick {
    const SLOT: Option<&'static str> = Some(AntialiasingSlot::NAME);

    fn register(_bricks: &mut Bricks<'_>) {}
}
