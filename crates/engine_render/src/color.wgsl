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
