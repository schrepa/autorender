import package::engine::materials::{VALUE_RIPPLES, material_value};
import package::engine::shading::{SurfaceInput, SurfacePoint};
import package::engine::standard::normal as standard_normal;

fn normal(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    let p = input.world.xz;
    let a = dot(p, vec2<f32>(0.8, 0.6)) * 0.7 + input.time * 1.1;
    let b = dot(p, vec2<f32>(-0.6, 0.8)) * 1.6 + input.time * 1.7;
    let c = dot(p, vec2<f32>(0.1, -1.0)) * 3.3 + input.time * 2.3;
    let slope = vec2<f32>(0.8, 0.6) * (0.126 * cos(a))
        + vec2<f32>(-0.6, 0.8) * (0.112 * cos(b))
        + vec2<f32>(0.1, -1.0) * (0.083 * cos(c));
    return normalize(standard_normal(input, point)
        - vec3<f32>(slope.x, 0.0, slope.y) * material_value(VALUE_RIPPLES));
}
