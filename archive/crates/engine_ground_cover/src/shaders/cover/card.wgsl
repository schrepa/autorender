import package::cover::faces::cover::{cover_card_look, cover_header, cover_light, cover_merged};
import package::cover::laws::LIT_LEAST;
import package::engine::mapped::cutoff as mapped_cutoff;
import package::engine::mapped::{base_texel, deflected_normal, mask_coverage};
import package::engine::materials::{MAP_NORMAL, map_present, map_uv, normal_map, normal_sampler,
    normal_scale};
import package::engine::shading::{SurfaceInput, SurfacePoint};
import package::engine::view::{fade_noise, stable_normalize};
import package::face::foliage::foliage;
import package::face::lighting::SURFACE_THIN;

fn base_color(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    return base_texel(input).rgb * input.color.rgb;
}

fn opacity(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return base_texel(input).a;
}

fn cutoff(input: SurfaceInput, point: SurfacePoint) -> f32 {
    if foliage().z == 0.0 {
        return mapped_cutoff(input, point);
    }
    return max(fade_noise(input.pixel), 1.0 / 255.0) / mask_coverage(input);
}

fn normal(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    let toward = stable_normalize(input.normal);
    var shading = select(-toward, toward, input.front);
    if map_present(MAP_NORMAL) {
        let uv = map_uv(MAP_NORMAL, input.uv, input.uv1);
        let sampled = textureSampleBias(normal_map, normal_sampler, uv, input.bias).xyz
            * 2.0 - 1.0;
        shading = deflected_normal(shading, input.world_dx, input.world_dy, dpdx(uv), dpdy(uv),
            sampled, normal_scale());
    }
    return stable_normalize(mix(shading, vec3<f32>(0.0, 1.0, 0.0), cover_header().shading.y));
}

fn roughness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return cover_header().shading.w;
}

fn occlusion(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return pow(max(input.surface.x, LIT_LEAST), 1.0 - input.color.a);
}

fn transmission(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return foliage().x * mix(cover_card_look().form.z, 1.0, input.color.a);
}

fn mirrored(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return 0.0;
}

fn kind(input: SurfaceInput, point: SurfacePoint) -> u32 {
    return SURFACE_THIN;
}

fn tangled(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return cover_merged(cover_light(), distance(input.world, input.eye));
}
