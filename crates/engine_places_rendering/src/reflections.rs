use bevy_ecs::prelude::Resource;
use engine_place::Slot;

#[derive(Resource, Default)]
pub struct TraceSource;

impl Slot for TraceSource {
    const NAME: &'static str = "trace_source";
    const WGSL: Option<&'static str> = Some(include_str!("places/trace_source.wgsl"));
}
