use bevy_ecs::prelude::Resource;
use engine_place::{List, Slot};

#[derive(Resource, Default)]
pub struct TraceSource;

impl Slot for TraceSource {
    const NAME: &'static str = "trace_source";
    const WGSL: Option<&'static str> = Some(include_str!("places/trace_source.wgsl"));
}

pub struct TraceReaders;

impl List for TraceReaders {
    const NAME: &'static str = "trace_readers";
    type Member = ();
}
