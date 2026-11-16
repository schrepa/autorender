import package::place::sun_blockers::{DEBUG_SUN_BEYOND, DEBUG_SUN_HORIZON, DEBUG_SUN_LEVEL,
    SunAt};
import package::generated::shadows::{sun_shown, sun_shown_source};
import package::shadows::sample::{air_visibility, map_lookup, map_visibility, sun_shadowed};

const FOLD_COST: u32 = 1u;
const SHADOWED_BELOW: f32 = 0.5;

fn sun_open(open: f32, at: SunAt) -> f32 {
    if !sun_shadowed() || bitcast<u32>(open) == 0u {
        return open;
    }
    let seen = min(1.0, map_visibility(at));
    if seen <= 0.0 {
        return seen;
    }
    return min(seen, open);
}

fn sun_air(open: f32, world: vec3<f32>) -> f32 {
    if !sun_shadowed() || bitcast<u32>(open) == 0u {
        return open;
    }
    let seen = min(1.0, air_visibility(world));
    if seen <= 0.0 {
        return seen;
    }
    return min(seen, open);
}

fn sun_flags(flags: u32, at: SunAt) -> u32 {
    if !sun_shadowed() || sun_shown_source(flags) == DEBUG_SUN_HORIZON {
        return flags;
    }
    var plain = at;
    plain.softness = 0.0;
    let found = map_lookup(plain);
    if found.y < 0.0 {
        return sun_shown(DEBUG_SUN_BEYOND, false);
    }
    return sun_shown(DEBUG_SUN_LEVEL + u32(found.y), found.x < SHADOWED_BELOW);
}
