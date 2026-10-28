fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn encode_srgb(color: vec3<f32>) -> vec3<f32> {
    let low = color * 12.92;
    let high = 1.055 * pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, color <= vec3<f32>(0.0031308));
}

fn decode_srgb(encoded: vec3<f32>) -> vec3<f32> {
    let low = encoded * (1.0 / 12.92);
    let high = pow((max(encoded, vec3<f32>(0.0)) + 0.055) * (1.0 / 1.055), vec3<f32>(2.4));
    return select(high, low, encoded <= vec3<f32>(0.04045));
}

fn filmic(x: vec3<f32>) -> vec3<f32> {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

fn unfilmic(shown: f32) -> f32 {
    let a = 2.51 - 2.43 * shown;
    let b = 0.03 - 0.59 * shown;
    return (sqrt(b * b + 0.56 * a * shown) - b) / (2.0 * a);
}

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

fn dithered(encoded: vec3<f32>, pixel: vec2<f32>, turn: f32, salt: u32) -> vec3<f32> {
    let bits = dither_hash(vec3<u32>(vec2<u32>(max(pixel, vec2<f32>(0.0))),
        bitcast<u32>(turn) ^ salt));
    let first = vec3<f32>(bits >> vec3<u32>(16u)) * (1.0 / 65536.0) - 0.5;
    let second = vec3<f32>(bits & vec3<u32>(0xffffu)) * (1.0 / 65536.0) - 0.5;
    let room = saturate(min(encoded, vec3<f32>(1.0) - encoded) * 510.0);
    return encoded + (first + second * room) * (1.0 / 255.0);
}
