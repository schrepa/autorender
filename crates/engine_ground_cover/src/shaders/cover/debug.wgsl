import package::cover::draw::{CoverOutput, cover_surface};
import package::face::debug::{DebugTargets, debug_record, debug_targets};
import package::face::lighting::{sun_debug, surface_sun};
import package::generated::kinds::KIND_COVER;

@fragment
fn fs_debug_cover(input: CoverOutput, @builtin(front_facing) front: bool) -> DebugTargets {
    let point = cover_surface(input, front);
    let sun = sun_debug(surface_sun(point));
    return debug_targets(debug_record(KIND_COVER, 0u, false, false, 0u), 0u, 0u, point.normal,
        sun);
}
