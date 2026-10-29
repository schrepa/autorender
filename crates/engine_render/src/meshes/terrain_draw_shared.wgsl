fn terrain_value(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * (3.0 - 2.0 * f);
    let q = vec2<u32>(vec2<i32>(cell));
    let x0 = q.x * 0x8da6b343u;
    let x1 = x0 + 0x8da6b343u;
    let y0 = q.y * 0xd8163841u;
    let y1 = y0 + 0xd8163841u;
    let a = terrain_hash_of(x0 ^ y0);
    let b = terrain_hash_of(x1 ^ y0);
    let c = terrain_hash_of(x0 ^ y1);
    let d = terrain_hash_of(x1 ^ y1);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn terrain_hash_of(mixed: u32) -> f32 {
    var h = mixed;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return f32((h ^ (h >> 16u)) >> 8u) * (1.0 / 16777216.0);
}

fn terrain_hash2(mixed: u32) -> vec2<f32> {
    var h = mixed;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return vec2<f32>(f32(h >> 16u), f32(h & 0xffffu)) * (1.0 / 65535.0);
}

fn terrain_value2(p: vec2<f32>) -> vec2<f32> {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * (3.0 - 2.0 * f);
    let q = vec2<u32>(vec2<i32>(cell));
    let x0 = q.x * 0x8da6b343u;
    let x1 = x0 + 0x8da6b343u;
    let y0 = q.y * 0xd8163841u;
    let y1 = y0 + 0xd8163841u;
    return mix(mix(terrain_hash2(x0 ^ y0), terrain_hash2(x1 ^ y0), u.x),
        mix(terrain_hash2(x0 ^ y1), terrain_hash2(x1 ^ y1), u.x), u.y);
}
