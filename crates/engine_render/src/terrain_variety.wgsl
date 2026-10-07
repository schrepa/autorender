// Ground variety (decision 090): the tint a terrain's large-scale map gives
// its ground at every distance and the blades growing on it, the same in
// both (`meshes/terrain.wgsl`, `cover_grow.wgsl`). The map holds where the
// ground lies in a hollow (0.5 on level ground, toward 1 in hollows) and on
// a ridge (0.5 level, toward 1 on ridges); broad patches vary it further.
// Ground cover's density, which both read too (decision 098), follows.

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

// Ground cover's density (decision 056) as a share of its densest, `away`
// meters from the eye, for cover growing densest within `radius` and in
// tiers out to `reach` (0 without tiers; decision 090), the same for its
// blades and the canopy shown where they thin (decision 098): all of it
// within a fifth of the radius, then the fifth over the distance; past the
// radius, a fifth of the radius's cube over the distance's, which meets it
// there. Without tiers it thins to none over the radius's outer half; with
// them, over the last three tenths of `end`, the reach drawn in by world
// noise (`cover_end`) or, to bound it, the reach itself.
fn cover_share(away: f32, radius: f32, reach: f32, end: f32) -> f32 {
    let distance = max(away, 1.0e-3);
    var share = min(1.0, 0.2 * radius / distance);
    if distance > radius {
        let ratio = radius / distance;
        share = 0.2 * ratio * ratio * ratio;
    }
    if reach > radius {
        return share * (1.0 - smoothstep(0.7 * end, end, away));
    }
    return share * (1.0 - smoothstep(0.5 * radius, radius, away));
}

// Where cover's tiers end at world `world` (x and z): their reach, drawn in
// by up to a fifth in patches tens of meters across (decision 098), so
// cover ends raggedly instead of in a circle around the eye.
fn cover_end(world: vec2<f32>, reach: f32) -> f32 {
    return reach * (0.8 + 0.2 * variety_noise(world / 37.0 + vec2<f32>(5.3, -8.1)));
}

// How much of the leaves the densest cover would show blades growing at
// `share` of it, `away` meters from the eye, show (decision 098): blades
// widen as they thin within the radius, so there all of it, and past it
// they thin faster than they widen. The ground shows the rest as its
// canopy.
fn cover_shown(away: f32, share: f32, radius: f32) -> f32 {
    return min(share * max(1.0, away / (0.2 * radius)), 1.0);
}
