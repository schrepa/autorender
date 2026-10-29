use engine_core::{Brick, Bricks};
use engine_faces::slots::ProbesList;

pub struct ReflectionProbesBrick;

impl Brick for ReflectionProbesBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.list::<ProbesList>();
    }
}
