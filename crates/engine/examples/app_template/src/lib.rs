use engine::prelude::*;

pub mod bricks {
    pub mod glimmer;
    pub mod turning;
}

use crate::bricks::glimmer::GlimmerBrick;
use crate::bricks::turning::TurningBrick;

#[derive(Resource, Default)]
pub struct Template;

impl Game for Template {}

pub fn bricks<L: Listing>(list: L) -> L {
    list.with::<TurningBrick>().with::<GlimmerBrick>()
}

engine::game!(
    pub struct TemplateGame(Template),
    "Template",
    bricks,
    scene = "template.scene.json"
);
