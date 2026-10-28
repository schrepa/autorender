const WATER_BURST_LONGEST: u32 = 16u;

fn water_burst_bits(site: vec2<i32>, stream: u32, slot: vec2<u32>) -> vec4<u32> {
    var v = vec4<u32>(bitcast<vec2<u32>>(site), slot.x, (stream << 5u) | slot.y) * 1664525u
        + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v ^= v >> vec4<u32>(16u);
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}

fn water_burst_slot(site: vec2<i32>, stream: u32, step: u32, life: f32, seconds: f32,
        least: u32) -> vec2<u32> {
    let steps = max(life / max(seconds, 1.0e-9), 1.0);
    let nearest = u32(clamp(round(log2(steps)), 0.0, f32(WATER_BURST_LONGEST)));
    let octaves = max(nearest, firstTrailingBit(max(least, 1u)));
    let shift = water_burst_bits(site, stream, vec2<u32>(0u, 31u)).x;
    return vec2<u32>((step + shift) >> octaves, octaves);
}

fn water_burst_life(slot: vec2<u32>, seconds: f32) -> f32 {
    return f32(1u << slot.y) * seconds;
}

fn water_burst_weight(mean: f32, drawn: f32) -> f32 {
    let fires = 1.0 - exp(-mean);
    if mean <= 0.0 || drawn >= fires {
        return 0.0;
    }
    return mean / fires;
}

struct WaterBurst {
    starts: bool,
    within: vec2<f32>,
    turn: f32,
}

const WATER_BURST_LIVE: u32 = 16u;

fn water_burst_draws(cell: vec2<i32>, tick: u32) -> vec4<f32> {
    let bits = water_burst_bits(cell, WATER_BURST_LIVE, vec2<u32>(tick, 0u));
    return vec4<f32>(bits >> vec4<u32>(8u)) * (1.0 / 16777216.0);
}

fn water_burst(cell: vec2<i32>, tick: u32, mean: f32) -> WaterBurst {
    let draws = water_burst_draws(cell, tick);
    return WaterBurst(draws.x < 1.0 - exp(-max(mean, 0.0)), draws.yz, draws.w);
}
