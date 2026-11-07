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

pub const TRACE_DEPTH_SOLID: u32 = 0;
pub const TRACE_DEPTH_THIN: u32 = 1;
pub const TRACE_DEPTH_SURFACE: u32 = 2;
pub const TRACE_DEPTH_OPAQUE: u32 = 3;

pub struct TraceDepths;

impl List for TraceDepths {
    const NAME: &'static str = "trace_depths";
    type Member = ();
    const WGSL: Option<&'static str> = Some(engine_places::contract!(
        "places/trace_depths.wgsl",
        TRACE_DEPTH_SOLID,
        TRACE_DEPTH_THIN,
        TRACE_DEPTH_SURFACE,
        TRACE_DEPTH_OPAQUE,
    ));
}
