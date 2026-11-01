import package::generated::pieces::shadows::piece;
import package::generated::shadows::KEPT_GROUND_TEXELS;
import package::place::shadows::{DEBUG_SUN_LEVEL, KeptLift, KeptTexels};
import package::shadows::sample::{air_visibility, crown_light, kept_lift, map_lookup,
    map_visibility, shadow_level};
import package::shadows::sky::sky_occlusion;

fn sun_shadowed() -> bool {
    return piece().state.x != 0.0;
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
    kept.count = u32(piece().level_state.x);
    for (var index = 0u; index < 4u && kept.first + index < kept.count; index++) {
        kept.texels[index] = piece().windows[kept.first + index].w;
    }
    return kept;
}

fn keep_lift(lift: KeptLift) {
    kept_lift = lift;
}

fn kept_level(texel: f32, spacing: f32, levels: u32) -> u32 {
    var level = 0u;
    while level + 1u < levels && spacing * f32(1u << level) < KEPT_GROUND_TEXELS * texel {
        level++;
    }
    return level;
}

fn sky_visibility(world: vec3<f32>, normal: vec3<f32>) -> f32 {
    return sky_occlusion(world, normal);
}

fn debug_level_offset() -> u32 {
    return DEBUG_SUN_LEVEL;
}
