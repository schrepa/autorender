const FIRST_PERSON_DEPTH: f32 = 0.00390625;
const TAU: f32 = 6.2831853;

fn maximum_abs(value: vec3<f32>) -> f32 {
    let positive = abs(value);
    return max(positive.x, max(positive.y, positive.z));
}

fn unit(value: u32) -> f32 {
    return f32(value >> 8u) * (1.0 / 16777216.0);
}

fn pcg(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
