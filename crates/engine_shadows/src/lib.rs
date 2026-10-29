use engine_core::{Brick, Bricks, Slot};
use engine_faces::slots::ShadowsSlot;

pub struct ShadowsBrick;

impl Brick for ShadowsBrick {
    const SLOT: Option<&'static str> = Some(ShadowsSlot::NAME);

    fn register(_bricks: &mut Bricks<'_>) {}
}
