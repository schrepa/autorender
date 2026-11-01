include!(concat!(env!("OUT_DIR"), "/bricks/toon_lighting.rs"));

use engine_core::{Brick, Bricks, Source};
use engine_places::lighting::Lighting;

pub struct ToonLightingBrick;

impl Brick for ToonLightingBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks.fill::<Lighting>(Some(shaders::toon_lighting::faces::lighting::MODULE));
    }
}
