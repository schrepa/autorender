import package::engine::math::PI;
import package::face::sun::sun_direction;
import package::generated::pieces::terrain_shading::piece;
import package::generated::terrain_shading::{SUN_RIM, SUN_STATE};
import package::terrain_shading::horizons::{terrain_horizon_sampler, terrain_horizons};

struct SunHorizon {
    weight: f32,
    crests: vec4<f32>,
    ground: vec4<f32>,
}

fn sun_horizon(pick: u32, world: vec2<f32>, lifted: bool) -> SunHorizon {
    var found: SunHorizon;
    let levels = u32(piece().sun[SUN_STATE].z);
    for (var level = 0u; level < levels && found.weight < 1.0; level++) {
        let record = piece().sun[2u * level + pick];
        let uv = (world - record.xy) * record.z;
        let inside = min(min(uv.x, uv.y), 1.0 - max(uv.x, uv.y));
        let share = saturate(inside * SUN_RIM) * (1.0 - found.weight);
        if share <= 0.0 {
            continue;
        }
        let layer = u32(record.w);
        found.crests += share * textureSampleLevel(terrain_horizons, terrain_horizon_sampler, uv,
            layer, 0.0);
        if lifted {
            found.ground += share * textureSampleLevel(terrain_horizons, terrain_horizon_sampler,
                uv, layer + 1u, 0.0);
        }
        found.weight += share;
    }
    return found;
}

fn sun_disc(clear: f32) -> f32 {
    let x = clamp(clear, -1.0, 1.0);
    return 0.5 + (asin(x) + x * sqrt(1.0 - x * x)) / PI;
}

fn terrain_visibility(world: vec3<f32>, on_terrain: bool) -> f32 {
    let sun = sun_direction();
    let level = dot(sun.xz, sun.xz);
    if level < 1.0e-8 {
        return 1.0;
    }
    let state = piece().sun[SUN_STATE];
    var found = sun_horizon(0u, world.xz, !on_terrain);
    if state.x > 0.0 {
        let next = sun_horizon(1u, world.xz, !on_terrain);
        found.crests = mix(found.crests, next.crests, state.x);
        found.ground = mix(found.ground, next.ground, state.x);
        found.weight = mix(found.weight, next.weight, state.x);
    }
    if found.weight <= 0.0 {
        return 1.0;
    }
    let crests = found.crests / found.weight;
    var slope = crests.x + crests.y;
    if !on_terrain {
        let ground = found.ground / found.weight;
        let above = world.y - (ground.x + ground.y);
        let far = crests.z - (above - state.y) * ground.w;
        slope = max(slope - max(above, 0.0) * ground.z, far);
    }
    let rise = sun.y * inverseSqrt(level);
    let clear = (rise - slope) * level
        / (state.w * sqrt(1.0 + level * crests.w * crests.w));
    return mix(1.0, sun_disc(clear), found.weight);
}
