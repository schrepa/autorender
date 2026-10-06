// Ground variety (decision 090): the tint a terrain's large-scale map gives
// its ground at every distance and the blades growing on it, the same in
// both (`meshes/terrain.wgsl`, `cover_grow.wgsl`). The map holds where the
// ground lies in a hollow (0.5 on level ground, toward 1 in hollows) and on
// a ridge (0.5 level, toward 1 on ridges); broad patches vary it further.

fn variety_hash(cell: vec2<i32>) -> f32 {
    var h = (u32(cell.x) * 0x8da6b343u) ^ (u32(cell.y) * 0xd8163841u);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return f32((h ^ (h >> 16u)) >> 8u) * (1.0 / 16777216.0);
}

// Value noise at `p` (0 to 1), its lattice blended by smooth steps.
fn variety_noise(p: vec2<f32>) -> f32 {
    let cell = vec2<i32>(floor(p));
    let f = fract(p);
    let s = f * f * (3.0 - 2.0 * f);
    let a = variety_hash(cell);
    let b = variety_hash(cell + vec2<i32>(1, 0));
    let c = variety_hash(cell + vec2<i32>(0, 1));
    let d = variety_hash(cell + vec2<i32>(1, 1));
    return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

// The tint of ground at world `world` (x and z) whose map texel holds
// `map` (hollow, ridge), `vegetated` the share of it growing plants:
// plants grow lusher and darker where water gathers and drier and paler on
// exposed ground and in broad patches; bare ground and rock only weather,
// paler on ridges.
fn ground_variety(map: vec2<f32>, world: vec2<f32>, vegetated: f32) -> vec3<f32> {
    let patches = variety_noise(world / 340.0 + vec2<f32>(13.1, -7.7)) * 0.65
        + variety_noise(world / 130.0 + vec2<f32>(-3.3, 21.9)) * 0.35;
    let lush = saturate(0.5 + 1.6 * (map.x - 0.5) - 1.2 * (map.y - 0.5) + 0.8 * (patches - 0.5));
    let growth = mix(vec3<f32>(1.05, 1.02, 0.88), vec3<f32>(0.85, 0.97, 0.82), lush);
    let weather = vec3<f32>(1.0 + 0.14 * (map.y - 0.5) - 0.08 * (patches - 0.5));
    return mix(weather, growth, vegetated);
}
