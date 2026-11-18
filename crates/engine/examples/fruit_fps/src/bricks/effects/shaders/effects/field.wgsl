import package::effects::items::effects_frame;

@group(1) @binding(0) var effects_field_map: texture_2d<f32>;
@group(1) @binding(1) var effects_field_sampler: sampler;

fn effect_field(world: vec3<f32>) -> vec4<f32> {
    let rect = effects_frame.field;
    let shape = effects_frame.field_shape;
    let to_sun = effects_frame.sun.xyz;
    let along = to_sun.xz * ((world.y - shape.x) / max(to_sun.y, 0.05));
    let uv = (world.xz - along - rect.xy) * rect.zw;
    let stored = textureSampleLevel(effects_field_map, effects_field_sampler, uv, 0.0);
    let sampled = vec4<f32>(stored.rgb * stored.rgb * shape.y, stored.a);
    return select(vec4<f32>(effects_frame.sky_color.rgb, 1.0), sampled, shape.z > 0.0);
}
