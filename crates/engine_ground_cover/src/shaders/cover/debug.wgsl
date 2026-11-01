import package::cover::draw::{CoverOutput, cover_surface};
import package::face::debug::{DEBUG_KIND_COVER, DebugTargets, debug_targets};
import package::face::lighting::sun_debug;

@fragment
fn fs_debug_cover(input: CoverOutput, @builtin(front_facing) front: bool) -> DebugTargets {
    let point = cover_surface(input, front);
    let sun = sun_debug(point.world + point.shadow_offset, point.geometric, point.world_dx,
        point.world_dy, point.on_terrain);
    return debug_targets(DEBUG_KIND_COVER, 0u, 0u, point.normal, sun);
}
