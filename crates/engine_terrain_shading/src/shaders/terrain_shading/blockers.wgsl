import package::engine::surfaces::SURFACE_GROUND;
import package::place::sun_blockers::{DEBUG_SUN_HORIZON, SunAt, SunOpen};
import package::generated::terrain_shading::sun_shown;
import package::terrain_shading::faces::horizons::terrain_visibility;

fn sun_open(open: SunOpen, at: SunAt) -> SunOpen {
    if open.blocked <= 0.0 {
        return open;
    }
    var result = open;
    result.blocked = min(open.blocked, terrain_visibility(at.world, at.kind == SURFACE_GROUND));
    return result;
}

fn sun_flags(flags: u32, at: SunAt) -> u32 {
    if terrain_visibility(at.world, at.kind == SURFACE_GROUND) <= 0.0 {
        return sun_shown(DEBUG_SUN_HORIZON, true);
    }
    return flags;
}
