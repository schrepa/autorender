import package::engine::surfaces::SURFACE_WATER;
import package::place::trace_depths::TRACE_DEPTH_OPAQUE;

fn trace_depth(depth: u32, kind: u32) -> u32 {
    return select(depth, TRACE_DEPTH_OPAQUE, kind == SURFACE_WATER);
}
