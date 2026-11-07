import package::engine::surfaces::SURFACE_GROUND;
import package::place::trace_depths::TRACE_DEPTH_SURFACE;

fn trace_depth(depth: u32, kind: u32) -> u32 {
    return select(depth, TRACE_DEPTH_SURFACE, kind == SURFACE_GROUND);
}
