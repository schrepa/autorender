import package::place::sun_blockers::{DEBUG_SUN_BEYOND, DEBUG_SUN_HORIZON, DEBUG_SUN_LEVEL,
    SunAt};
import package::generated::shadows::{sun_shown, sun_shown_source};
import package::shadows::sample::{air_visibility, map_lookup, map_visibility, sun_shadowed};

const SHADOWED_BELOW: f32 = 0.5;

fn sun_open(open: f32, at: SunAt) -> f32 {
    if !sun_shadowed() || open <= 0.0 {
        return open;
    }
    return min(open, map_visibility(at));
}

fn sun_air(open: f32, world: vec3<f32>) -> f32 {
    if !sun_shadowed() || open <= 0.0 {
        return open;
    }
    return min(open, air_visibility(world));
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
