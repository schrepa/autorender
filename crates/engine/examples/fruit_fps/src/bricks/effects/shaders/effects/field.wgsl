import package::effects::items::effects_frame;
import package::play::light::light_field;

@group(1) @binding(0) var effects_field_map: texture_2d<f32>;
@group(1) @binding(1) var effects_field_sampler: sampler;

fn effect_field(world: vec3<f32>) -> vec4<f32> {
    let shape = effects_frame.field_shape;
    let sampled = light_field(effects_field_map, effects_field_sampler, world,
        effects_frame.sun.xyz, effects_frame.field, shape);
    return select(vec4<f32>(effects_frame.sky_color.rgb, 1.0), sampled, shape.z > 0.0);
}
