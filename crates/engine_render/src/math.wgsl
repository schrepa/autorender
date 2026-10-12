// Numeric helpers several modules share.

// The largest magnitude of `value`'s components.
fn maximum_abs(value: vec3<f32>) -> f32 {
    let positive = abs(value);
    return max(positive.x, max(positive.y, positive.z));
}

// A hash's top 24 bits as a number in [0, 1).
fn unit(value: u32) -> f32 {
    return f32(value >> 8u) * (1.0 / 16777216.0);
}
