use bevy_ecs::prelude::Resource;
use engine_place::Slot;

#[derive(Resource, Default)]
pub struct DitherPlace;

impl Slot for DitherPlace {
    const NAME: &'static str = "dither";
    const WGSL: Option<&'static str> = Some(include_str!("places/dither.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::DITHER];
}
