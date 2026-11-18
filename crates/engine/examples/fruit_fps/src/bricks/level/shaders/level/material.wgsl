import package::generated::level::LevelSurface;

@group(1) @binding(0) var level_base_color_map: texture_2d<f32>;
@group(1) @binding(1) var level_base_color_sampler: sampler;
@group(1) @binding(2) var level_normal_map: texture_2d<f32>;
@group(1) @binding(3) var level_normal_sampler: sampler;
@group(1) @binding(4) var level_orm_map: texture_2d<f32>;
@group(1) @binding(5) var level_orm_sampler: sampler;
@group(1) @binding(6) var level_lightmap: texture_2d<f32>;
@group(1) @binding(7) var level_lightmap_sampler: sampler;
@group(1) @binding(8) var level_macro_map: texture_2d<f32>;
@group(1) @binding(9) var level_macro_sampler: sampler;
@group(1) @binding(10) var<uniform> level_surface: LevelSurface;
@group(1) @binding(11) var level_sky_map: texture_2d<f32>;
@group(1) @binding(12) var level_sky_sampler: sampler;

struct LevelTexels {
    base_color: vec3<f32>,
    alpha: f32,
    normal: vec3<f32>,
    occlusion: f32,
    roughness: f32,
    metallic: f32,
    glow: f32,
}

fn level_texels(uv: vec2<f32>) -> LevelTexels {
    var texels: LevelTexels;
    let base = textureSample(level_base_color_map, level_base_color_sampler, uv) * level_surface.tint;
    texels.base_color = base.rgb;
    texels.alpha = base.a;
    let across = textureSample(level_normal_map, level_normal_sampler, uv).xy * 2.0 - 1.0;
    texels.normal = vec3<f32>(across, sqrt(max(1.0 - dot(across, across), 0.0)));
    let orm = textureSample(level_orm_map, level_orm_sampler, uv);
    texels.occlusion = orm.r;
    texels.roughness = orm.g;
    texels.metallic = orm.b;
    texels.glow = orm.a;
    return texels;
}

fn level_baked(uv1: vec2<f32>) -> vec4<f32> {
    return textureSample(level_lightmap, level_lightmap_sampler, uv1);
}

fn level_broken(texels: LevelTexels, world: vec3<f32>, facing: vec3<f32>) -> LevelTexels {
    let broken = level_surface.broken;
    var out = texels;
    if broken.x + broken.y > 0.0 {
        let axis = abs(facing);
        let across = select(select(world.xy, world.zy, axis.x > axis.z), world.xz,
            axis.y > max(axis.x, axis.z));
        let both = textureSample(level_macro_map, level_macro_sampler, across * broken.z).rg - 0.5;
        let shift = both.x + both.y;
        out.base_color = texels.base_color * max(1.0 + broken.x * shift, 0.0);
        out.roughness = saturate(texels.roughness + broken.y * shift);
    }
    return out;
}
