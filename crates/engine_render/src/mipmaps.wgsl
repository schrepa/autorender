// Mip generation for base color textures (see mipmaps.rs). Each level is the
// bilinear sample of the level above at this level's texel centers: exactly
// the average of each 2x2 block for even sizes. Sampling an sRGB view decodes
// before filtering and the sRGB target encodes after, so averages are linear.

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

struct Varyings {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// One triangle covering the level, with texture coordinates 0..1 across it
// (top-left origin): clip (-1, 1), (3, 1), (-1, -3).
@vertex
fn vs_mip(@builtin(vertex_index) index: u32) -> Varyings {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var output: Varyings;
    output.position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    output.uv = uv;
    return output;
}

@fragment
fn fs_mip(input: Varyings) -> @location(0) vec4<f32> {
    return textureSampleLevel(source, source_sampler, input.uv, 0.0);
}
