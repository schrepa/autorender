fn light_field(map: texture_2d<f32>, map_sampler: sampler, at: vec3<f32>, to_sun: vec3<f32>,
        rect: vec4<f32>, shape: vec4<f32>) -> vec4<f32> {
    let along = to_sun.xz * ((at.y - shape.x) / max(to_sun.y, 0.05));
    let uv = (at.xz - along - rect.xy) * rect.zw;
    let stored = textureSampleLevel(map, map_sampler, uv, 0.0);
    return vec4<f32>(stored.rgb * stored.rgb * shape.y, stored.a);
}
