import package::generated::pieces::shadows::piece;
import package::generated::shadows::KEPT_GROUND_TEXELS;
import package::place::shadows::KeptTexels;
import package::shadows::sample::{air_visibility, shadow_level, sun_shadowed};

fn air_shadow(world: vec3<f32>) -> f32 {
    if !sun_shadowed() {
        return 1.0;
    }
    return air_visibility(world);
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

fn kept_level(texel: f32, spacing: f32, levels: u32) -> u32 {
    var level = 0u;
    while level + 1u < levels && spacing * f32(1u << level) < KEPT_GROUND_TEXELS * texel {
        level++;
    }
    return level;
}
