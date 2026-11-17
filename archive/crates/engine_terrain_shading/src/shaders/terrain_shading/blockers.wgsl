import package::engine::surfaces::SURFACE_GROUND;
import package::place::sun_blockers::{DEBUG_SUN_HORIZON, SunAt};
import package::generated::terrain_shading::sun_shown;
import package::terrain_shading::faces::horizons::terrain_visibility;

fn sun_open(open: f32, at: SunAt) -> f32 {
    if open <= 0.0 {
        return open;
    }
    return min(open, terrain_visibility(at.world, at.kind == SURFACE_GROUND));
}

fn sun_air(open: f32, world: vec3<f32>) -> f32 {
    if open <= 0.0 {
        return open;
    }
    return min(open, terrain_visibility(world, false));
}

fn sun_flags(flags: u32, at: SunAt) -> u32 {
    if terrain_visibility(at.world, at.kind == SURFACE_GROUND) <= 0.0 {
        return sun_shown(DEBUG_SUN_HORIZON, true);
    }
    return flags;
}
