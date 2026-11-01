import package::cover::draw::{CoverOutput, cover_surface};
import package::face::debug::{DEBUG_KIND_COVER, DebugTargets, debug_targets};
import package::face::lighting::{sun_debug, surface_sun};

@fragment
fn fs_debug_cover(input: CoverOutput, @builtin(front_facing) front: bool) -> DebugTargets {
    let point = cover_surface(input, front);
    let sun = sun_debug(surface_sun(point));
    return debug_targets(DEBUG_KIND_COVER, 0u, 0u, point.normal, sun);
}
