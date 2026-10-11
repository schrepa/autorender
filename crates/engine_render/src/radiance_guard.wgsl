// A finite half-target boundary has explicit headroom below 65504. This
// does not promise recovery or complete detection after upstream overflow.
const RADIANCE_GUARD: f32 = 60000.0;

fn guard_radiance(value: vec4<f32>, bit: u32) -> vec4<f32> {
    var bounded = value;
    var limited = false;
    for (var i = 0u; i < 4u; i += 1u) {
        if (bitcast<u32>(value[i]) & 0x7f800000u) != 0x7f800000u {
            limited = limited || value[i] < 0.0 || value[i] >= RADIANCE_GUARD;
            bounded[i] = clamp(value[i], 0.0, RADIANCE_GUARD);
        }
    }
    if limited {
        atomicOr(&radiance_limits, bit);
    }
    return bounded;
}
