import package::place::sun_blockers::{DEBUG_SHADOWED, DEBUG_SUN_HORIZON, SunAt, SunOpen};
import package::terrain_shading::faces::horizons::terrain_visibility;

fn sun_open(open: SunOpen, at: SunAt) -> SunOpen {
    if open.blocked <= 0.0 {
        return open;
    }
    var result = open;
    result.blocked = min(open.blocked, terrain_visibility(at.world, at.ground));
    return result;
}

fn sun_flags(flags: u32, at: SunAt) -> u32 {
    if terrain_visibility(at.world, at.ground) <= 0.0 {
        return DEBUG_SUN_HORIZON | DEBUG_SHADOWED;
    }
    return flags;
}
