import package::shadows::sample::{SHADOW_MAPS, air_visibility, crown_light, kept_lift, map_lookup,
    map_visibility, shadow, shadow_level};

struct KeptTexels {
    first: u32,
    count: u32,
    texels: vec4<f32>,
}

struct KeptLift {
    first: u32,
    lifts: vec4<f32>,
}

fn sun_shadowed() -> bool {
    return SHADOW_MAPS && shadow.state.x != 0.0;
}

fn sun_shadow(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>, crown: f32)
        -> f32 {
    crown_light = crown;
    let visible = map_visibility(world, normal, dx, dy);
    crown_light = 0.0;
    return visible;
}

fn air_shadow(world: vec3<f32>) -> f32 {
    return air_visibility(world);
}

fn shadow_lookup(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>)
        -> vec2<f32> {
    return map_lookup(world, normal, dx, dy);
}

fn kept_texels(world: vec3<f32>) -> KeptTexels {
    var kept: KeptTexels;
    kept.first = shadow_level(world).level;
    kept.count = u32(shadow.level_state.x);
    for (var index = 0u; index < 4u && kept.first + index < kept.count; index++) {
        kept.texels[index] = shadow.windows[kept.first + index].w;
    }
    return kept;
}

fn keep_lift(lift: KeptLift) {
    kept_lift = lift;
}
