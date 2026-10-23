// Ground variety: the tint a terrain's macro colors give its
// ground and the blades growing on it, the same in both
// (`meshes/terrain.wgsl`, `cover_grow.wgsl`). Ground cover's density, which
// both read too, follows.

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

// The meters a pixel spans of level ground `relative` to the eye, a pixel
// spanning `pixel` meters a meter away: as `pixel_footprint` measures it,
// the view's depth taken as the distance, for passes binding no view
// (blades and placed objects take their ground's tint by it). Views more
// grazing than a sixteenth count as that, as there.
fn ground_footprint(relative: vec3<f32>, pixel: f32) -> f32 {
    let distance = max(length(relative), 1.0e-6);
    return distance * pixel * inverseSqrt(max(abs(relative.y) / distance, 0.0625));
}

// The tint of ground whose macro colors' texel holds `colors` (the tint, 128
// of 255 leaving it as it is, and how lush plants grow there:
// `engine_terrain::macro_colors`), `vegetated` the share of it growing
// plants, `strength` of it shown (`macro_strength`): every ground takes the
// tint, and plants grow lusher and darker where it holds them lush, drier
// and paler where dry.
fn macro_tint(colors: vec4<f32>, vegetated: f32, strength: f32) -> vec3<f32> {
    let growth = mix(vec3<f32>(1.05, 1.02, 0.88), vec3<f32>(0.85, 0.97, 0.82), colors.a);
    let tint = colors.rgb * (255.0 / 128.0) * mix(vec3<f32>(1.0), growth, vegetated);
    return mix(vec3<f32>(1.0), tint, strength);
}

// How much of their tint macro colors give ground where a pixel spans
// `footprint` meters of it (a page's texel its own size): a little where
// its materials' own detail shows, all of it once a pixel spans a third of
// a meter, where those repeat too small to vary it, rising between 4 and 35
// centimeters (40 and 400 meters away at a view facing the ground at
// 1080p).
fn macro_strength(footprint: f32) -> f32 {
    return 0.35 + 0.65 * smoothstep(0.04, 0.35, footprint);
}

// Ground cover's density as a share of its densest, `away`
// meters from the eye, for cover growing densest within `radius` and in
// tiers out to `reach` (0 without tiers), the same for its
// blades and the canopy shown where they thin: all of it
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
// by up to a fifth in patches tens of meters across, so
// cover ends raggedly instead of in a circle around the eye.
fn cover_end(world: vec2<f32>, reach: f32) -> f32 {
    return reach * (0.8 + 0.2 * variety_noise(world / 37.0 + vec2<f32>(5.3, -8.1)));
}

// How much of the leaves the densest cover would show blades growing at
// `share` of it, `away` meters from the eye, show: blades
// widen as they thin within the radius, so there all of it, and past it
// they thin faster than they widen. The ground shows the rest as its
// canopy.
fn cover_shown(away: f32, share: f32, radius: f32) -> f32 {
    return min(share * max(1.0, away / (0.2 * radius)), 1.0);
}
