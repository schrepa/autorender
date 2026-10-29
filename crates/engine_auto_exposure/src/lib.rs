use engine_core::{Brick, Bricks, Slot};
use engine_faces::slots::ExposureSlot;

pub struct AutoExposureBrick;

impl Brick for AutoExposureBrick {
    const SLOT: Option<&'static str> = Some(ExposureSlot::NAME);

    fn register(_bricks: &mut Bricks<'_>) {}
}
