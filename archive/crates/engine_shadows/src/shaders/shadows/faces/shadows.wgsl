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
    if kept.first < kept.count {
        kept.texels.x = level_texel(kept.first);
    }
    if kept.first + 1u < kept.count {
        kept.texels.y = level_texel(kept.first + 1u);
    }
    if kept.first + 2u < kept.count {
        kept.texels.z = level_texel(kept.first + 2u);
    }
    if kept.first + 3u < kept.count {
        kept.texels.w = level_texel(kept.first + 3u);
    }
    return kept;
}
