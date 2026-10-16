@group(0) @binding(0) var<uniform> params: QueueParams;
@group(0) @binding(1) var<storage, read> raw: array<vec4<f32>>;

// Keep the original hardware conversion into Rgba8UnormSrgb.
@fragment
fn export_reflections(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = vec2<u32>(position.xy);
    return raw[texel.y * params.size.x + texel.x];
}
