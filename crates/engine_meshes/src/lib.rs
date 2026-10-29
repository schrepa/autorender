use engine_core::{Brick, Bricks, Slot};
use engine_faces::slots::MeshesSlot;

pub struct MeshesBrick;

impl Brick for MeshesBrick {
    const SLOT: Option<&'static str> = Some(MeshesSlot::NAME);

    fn register(_bricks: &mut Bricks<'_>) {}
}
