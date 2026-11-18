@group(1) @binding(0) var base_color_map: texture_2d<f32>;
@group(1) @binding(1) var base_color_sampler: sampler;
@group(1) @binding(2) var normal_map: texture_2d<f32>;
@group(1) @binding(3) var normal_sampler: sampler;
@group(1) @binding(4) var orm_map: texture_2d<f32>;
@group(1) @binding(5) var orm_sampler: sampler;
@group(1) @binding(6) var figure_field_map: texture_2d<f32>;
@group(1) @binding(7) var figure_field_sampler: sampler;

struct MaterialTexels {
    base_color: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
    metallic: f32,
}

fn material_texels(uv: vec2<f32>) -> MaterialTexels {
    var texels: MaterialTexels;
    texels.base_color = textureSample(base_color_map, base_color_sampler, uv).rgb;
    let across = textureSample(normal_map, normal_sampler, uv).xy * 2.0 - 1.0;
    texels.normal = vec3<f32>(across, sqrt(max(1.0 - dot(across, across), 0.0)));
    let orm = textureSample(orm_map, orm_sampler, uv);
    texels.roughness = orm.g;
    texels.metallic = orm.b;
    return texels;
}

fn field_texel(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(figure_field_map, figure_field_sampler, uv, 0.0);
}
