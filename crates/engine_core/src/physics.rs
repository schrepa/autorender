use crate::Slot;
use crate::smoke::{SmokeEntry, SmokeResult, SmokeSession};

pub struct PhysicsSlot;

impl Slot for PhysicsSlot {
    const NAME: &'static str = "physics";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_physics",
        brick: Self::NAME,
        gpu: false,
        run: physics,
    }];
}

fn physics(_session: &mut dyn SmokeSession) -> SmokeResult {
    Ok(())
}
