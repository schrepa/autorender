import package::place::sun_blockers::{DEBUG_SHADOWED, DEBUG_SUN_BEYOND, DEBUG_SUN_HORIZON,
    DEBUG_SUN_LEVEL, SunAt, SunOpen};
import package::shadows::sample::{map_lookup, map_visibility, sun_shadowed};

const SHADOWED_BELOW: f32 = 0.5;

fn sun_open(open: SunOpen, at: SunAt) -> SunOpen {
    if !sun_shadowed() || open.blocked <= 0.0 {
        return open;
    }
    var result = open;
    result.blocked = min(open.blocked, map_visibility(at));
    return result;
}

fn sun_flags(flags: u32, at: SunAt) -> u32 {
    if !sun_shadowed() || (flags & (DEBUG_SHADOWED - 1u)) == DEBUG_SUN_HORIZON {
        return flags;
    }
    var plain = at;
    plain.softness = 0.0;
    let found = map_lookup(plain);
    if found.y < 0.0 {
        return DEBUG_SUN_BEYOND;
    }
    return (DEBUG_SUN_LEVEL + u32(found.y)) | select(0u, DEBUG_SHADOWED, found.x < SHADOWED_BELOW);
}
