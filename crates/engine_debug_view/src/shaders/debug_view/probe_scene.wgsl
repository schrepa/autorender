import package::engine::color::encode_srgb;

@group(1) @binding(0) var<uniform> probe_pixel: vec4<u32>;
@group(1) @binding(1) var<storage, read_write> shaded: vec4<u32>;
@group(1) @binding(2) var probe_scene: texture_2d<f32>;

@compute @workgroup_size(1)
fn cs_probe_scene() {
    let last = max(textureDimensions(probe_scene), vec2<u32>(1u)) - 1u;
    let color = textureLoad(probe_scene, vec2<i32>(min(probe_pixel.xy, last)), 0);
    shaded = vec4<u32>(pack4x8unorm(vec4<f32>(encode_srgb(color.rgb), color.a)), 0u, 0u, 0u);
}
