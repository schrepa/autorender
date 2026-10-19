// Color helpers of the modules that shade and present color.

// The luminance of linear Rec. 709 `color`.
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Linear `color` encoded as sRGB, for targets that store what they are
// given (WebGPU canvases are UNORM).
fn encode_srgb(color: vec3<f32>) -> vec3<f32> {
    let low = color * 12.92;
    let high = 1.055 * pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, color <= vec3<f32>(0.0031308));
}

// sRGB-encoded `encoded` as linear color: `encode_srgb`'s inverse.
fn decode_srgb(encoded: vec3<f32>) -> vec3<f32> {
    let low = encoded * (1.0 / 12.92);
    let high = pow((max(encoded, vec3<f32>(0.0)) + 0.055) * (1.0 / 1.055), vec3<f32>(2.4));
    return select(high, low, encoded <= vec3<f32>(0.04045));
}

// Three independent 32-bit hashes of `seed` (Jarzynski and Olano's pcg3d,
// "Hash Functions for GPU Rendering", JCGT 2020).
fn dither_hash(seed: vec3<u32>) -> vec3<u32> {
    var v = seed * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3<u32>(16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// Encoded color `encoded` (in [0, 1]) dithered for an 8-bit target at pixel
// `pixel`: each channel gains noise of a triangular distribution reaching
// one step of the target either side, which leaves the average of what the
// target stores exactly the value, and its error independent of it, so
// smooth gradients show no bands. Within half a step of black or white the
// noise narrows to an even spread of half a step either side, which the
// target's clamp cannot bias. `turn` changes the noise (temporally
// antialiased frames turn it every frame, which their history averages);
// `salt` gives each write of the same pixel its own.
fn dithered(encoded: vec3<f32>, pixel: vec2<f32>, turn: f32, salt: u32) -> vec3<f32> {
    let bits = dither_hash(vec3<u32>(vec2<u32>(max(pixel, vec2<f32>(0.0))),
        bitcast<u32>(turn) ^ salt));
    let first = vec3<f32>(bits >> vec3<u32>(16u)) * (1.0 / 65536.0) - 0.5;
    let second = vec3<f32>(bits & vec3<u32>(0xffffu)) * (1.0 / 65536.0) - 0.5;
    let room = saturate(min(encoded, vec3<f32>(1.0) - encoded) * 510.0);
    return encoded + (first + second * room) * (1.0 / 255.0);
}
