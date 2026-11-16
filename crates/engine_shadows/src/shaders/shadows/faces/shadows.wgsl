import package::face::sun_blockers::sun_air;
import package::place::shadows::KeptTexels;
import package::shadows::sample::{level_count, level_texel, shadow_level};

fn air_shadow(world: vec3<f32>) -> f32 {
    return sun_air(1.0, world);
}

fn kept_texels(world: vec3<f32>) -> KeptTexels {
    var kept: KeptTexels;
    kept.first = shadow_level(world).level;
    kept.count = level_count();
    for (var index = 0u; index < 4u && kept.first + index < kept.count; index++) {
        kept.texels[index] = level_texel(kept.first + index);
    }
    return kept;
}
