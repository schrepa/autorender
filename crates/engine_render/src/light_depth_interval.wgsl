// Conservative binary32 intervals for depth occupancy. Callers validate the
// input domain before doing arithmetic. Integer steps between operations are
// observable: an affine dot must not replace this enclosure construction.

fn depth_domain(value: f32, bound: u32) -> bool {
    let magnitude = bitcast<u32>(value) & 0x7fffffffu;
    return magnitude <= bound && (magnitude == 0u || magnitude >= 0x00800000u);
}

fn depth_position_valid(position: vec3<f32>) -> bool {
    return depth_domain(position.x, 0x49800000u)
        && depth_domain(position.y, 0x49800000u)
        && depth_domain(position.z, 0x49800000u);
}

fn depth_plane_valid(plane: vec4<f32>) -> bool {
    return depth_domain(plane.x, 0x40000000u)
        && depth_domain(plane.y, 0x40000000u)
        && depth_domain(plane.z, 0x40000000u)
        && depth_domain(plane.w, 0x4a000000u);
}

fn depth_light_valid(position: vec3<f32>, radius: f32) -> bool {
    let bits = bitcast<u32>(radius);
    return depth_position_valid(position) && bits >= 0x35800000u && bits <= 0x49800000u;
}

// A rounded operation may round either way or flush a subnormal result to
// zero. Keep every endpoint normal or zero, including at the normal boundary.
fn depth_outward(value: f32) -> vec2<f32> {
    let bits = bitcast<u32>(value);
    if (bits & 0x7fffffffu) <= 0x00800000u {
        return vec2<f32>(-0x1p-125f, 0x1p-125f);
    }
    if (bits & 0x80000000u) != 0u {
        return vec2<f32>(bitcast<f32>(bits + 1u), bitcast<f32>(bits - 1u));
    }
    return vec2<f32>(bitcast<f32>(bits - 1u), bitcast<f32>(bits + 1u));
}

fn depth_add(left: vec2<f32>, right: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(depth_outward(left.x + right.x).x,
        depth_outward(left.y + right.y).y);
}

fn depth_multiply(interval: vec2<f32>, coefficient: f32) -> vec2<f32> {
    if coefficient == 0.0 {
        return vec2<f32>(0.0);
    }
    let ordered = select(interval, interval.yx, coefficient < 0.0);
    return vec2<f32>(depth_outward(ordered.x * coefficient).x,
        depth_outward(ordered.y * coefficient).y);
}

fn depth_project_box(low: vec3<f32>, high: vec3<f32>, plane: vec4<f32>) -> vec2<f32> {
    let x = depth_multiply(vec2<f32>(low.x, high.x), plane.x);
    let y = depth_multiply(vec2<f32>(low.y, high.y), plane.y);
    let z = depth_multiply(vec2<f32>(low.z, high.z), plane.z);
    return depth_add(depth_add(depth_add(x, y), z), vec2<f32>(plane.w));
}

fn depth_point_interval(position: vec3<f32>, plane: vec4<f32>) -> vec2<f32> {
    return depth_project_box(position, position, plane);
}

// The original range test's nonnegative squared distance is retained. Within
// the checked domains, any point passing it is inside this deliberately loose
// box: 4*(1-2^-23)^5 > 1+2^-23. A spot uses its original radius too.
fn depth_light_interval(position: vec3<f32>, radius: f32, plane: vec4<f32>) -> vec2<f32> {
    let diameter = radius * 2.0;
    let x_low = depth_outward(position.x - diameter).x;
    let y_low = depth_outward(position.y - diameter).x;
    let z_low = depth_outward(position.z - diameter).x;
    let x_high = depth_outward(position.x + diameter).y;
    let y_high = depth_outward(position.y + diameter).y;
    let z_high = depth_outward(position.z + diameter).y;
    return depth_project_box(vec3<f32>(x_low, y_low, z_low),
        vec3<f32>(x_high, y_high, z_high), plane);
}

// Both inclusive bin endpoints are in 0..31, so full-width masks never shift
// by 32. Equality with a threshold belongs to the upper bin.
fn depth_interval_bits(first: u32, last: u32) -> u32 {
    return (0xffffffffu << first) & (0xffffffffu >> (31u - last));
}

fn depth_guard(mask: u32, fragment: u32, generation: u32, current: u32) -> bool {
    return generation != 0u && generation == current && mask != 0u
        && fragment != 0u && (fragment & mask) == fragment;
}
