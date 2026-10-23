// Ground variety: the tint a terrain's macro colors give its
// ground and the blades growing on it, the same in both
// (`meshes/terrain.wgsl`, `cover_grow.wgsl`). Ground cover's density, which
// both read too, follows, with how things retire by their size on screen
// (ground cover's blades and placed objects alike) and what a view sees of
// blades, which their canopy shows where they retire.

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

// The tint of ground whose macro colors' texel holds `colors` (the tint, 128
// of 255 leaving it as it is, and how lush plants grow there:
// `engine_terrain::macro_colors`), `vegetated` the share of it growing
// plants, `strength` of it shown (1 in the distance): every ground takes the
// tint, and plants grow lusher and darker where it holds them lush, drier
// and paler where dry.
fn macro_tint(colors: vec4<f32>, vegetated: f32, strength: f32) -> vec3<f32> {
    let growth = mix(vec3<f32>(1.05, 1.02, 0.88), vec3<f32>(0.85, 0.97, 0.82), colors.a);
    let tint = colors.rgb * (255.0 / 128.0) * mix(vec3<f32>(1.0), growth, vegetated);
    return mix(vec3<f32>(1.0), tint, strength);
}

// How much of their tint macro colors give ground `distance` meters from the
// eye: a little near, where its materials' own detail shows, all of it from
// a few hundred meters, where those repeat too small to vary it.
fn macro_strength(distance: f32) -> f32 {
    return 0.35 + 0.65 * smoothstep(40.0, 400.0, distance);
}

// The share of the radius within which ground cover grows densest
// (`gpu_layouts::constants::COVER_CORE`).
const COVER_CORE: f32 = 0.2;

// Ground cover's blades as `meshes/cover.wgsl` shades them, which their
// canopy takes on average: the share of their color those seen through
// from the side away from the sun show, the light reaching their roots,
// rising to all of it at their tips, and their roughness.
const COVER_FAR_SIDE: f32 = 0.8;
const COVER_ROOT_OCCLUSION: f32 = 0.35;
const COVER_ROUGHNESS: f32 = 0.7;

// The share of things retiring by their size on screen that still show
// where each spans `pixels` pixels: each shows while it spans more than a
// pixel times a factor of its own, from a half to two by its rank
// (2^(2 rank - 1)), so those ranked under (1 + log2 pixels) / 2 do: none
// under half a pixel, all from two. Ground cover's blades and the objects
// of thinning placement layers retire so, in their ranks' order, and the
// ground takes on what they leave.
fn seen_share(pixels: f32) -> f32 {
    return saturate(0.5 + 0.5 * log2(max(pixels, 1.0e-6)));
}

// How many times wider than their roots ground cover's blades grow `away`
// meters from the detail eye, for cover growing densest within `radius`:
// as many times as they thin within it, so they cover as much ground as the
// densest cover's; past it no wider than there.
fn cover_widening(away: f32, radius: f32) -> f32 {
    return clamp(away / (COVER_CORE * radius), 1.0, 1.0 / COVER_CORE);
}

// Ground cover's density as a share of its densest, `away` meters from the
// detail eye, for cover growing densest within `radius` whose blades span
// the settings' pixels `span` meters away at the width the radius gives
// them (0: cover within the radius alone), the same for its blades and the
// canopy shown where they thin. All of it within a fifth of the radius,
// then the fifth over the distance; past the radius, where blades widen no
// more, a fifth of the radius's square over the distance's, which meets it
// there, so each pixel of the view holds about as many blades as at the
// radius. Of those, the share still wide enough on screen stands
// (`seen_share`): within the radius blades span as many pixels as at its
// fifth, and past it fewer as they recede, until none stands from twice
// `span`. Without tiers cover thins to none over the radius's outer half.
fn cover_share(away: f32, radius: f32, span: f32) -> f32 {
    let distance = max(away, 1.0e-3);
    var share = min(1.0, COVER_CORE * radius / distance);
    if span <= 0.0 {
        return share * (1.0 - smoothstep(0.5 * radius, radius, away));
    }
    if distance > radius {
        let ratio = radius / distance;
        share = COVER_CORE * ratio * ratio;
    }
    // How many times the settings' pixels a blade spans, as wide as it
    // grows there.
    let spanned = span * COVER_CORE * cover_widening(distance, radius) / distance;
    return share * seen_share(spanned);
}

// How much of the leaves the densest cover would show blades growing at
// `share` of it, `away` meters from the detail eye, show: blades widen as
// they thin within the radius, so there all those the share keeps, and past
// it they thin without widening. The ground shows the rest as its canopy.
fn cover_shown(away: f32, share: f32, radius: f32) -> f32 {
    return min(share * cover_widening(away, radius), 1.0);
}

// What a view sees of ground cover's blades on average: the share of their
// tips' color in the colors it sees, and the share of light their occlusion
// leaves. Each blade tapers from its root as 1 - along^1.5 up its height,
// turns from its root's color to its tip's by smoothstep(0, 0.7, along) and
// from COVER_ROOT_OCCLUSION to no occlusion at its tip
// (`meshes/cover.wgsl`). A view whose path through the blades crosses
// `depth` of their leaves (`meshes/terrain.wgsl`'s `terrain_canopy_depth`)
// passes the share F of their leaf area lying above a height as often as
// exp(-depth F) (Beer-Lambert), so it meets each band of their height first
// as often as that falls across the band. The bands end at along 0.35, 0.7
// and 0.85, where F, the integral of 1 - t^1.5 from there to the tip over
// its whole 0.6, is 0.465, 0.107 and 0.0274, and each shows its own means
// over its leaf area: 0.177, 0.786, 1 and 1 of the tip's color, 0.168,
// 0.507, 0.767 and 0.900 of the way up. Seen down through sparse blades,
// the tips make 0.48 of the color; along dense ones, nearly all of it.
struct CoverSeen {
    tip: f32,
    occlusion: f32,
}

fn cover_seen(depth: f32) -> CoverSeen {
    let crossed = max(depth, 1.0e-3);
    let root = exp(-crossed);
    let low = exp(-0.465 * crossed);
    let middle = exp(-0.107 * crossed);
    let high = exp(-0.0274 * crossed);
    let met = 1.0 - root;
    var seen: CoverSeen;
    seen.tip = ((low - root) * 0.177 + (middle - low) * 0.786 + (1.0 - middle)) / met;
    let along = ((low - root) * 0.168 + (middle - low) * 0.507 + (high - middle) * 0.767
        + (1.0 - high) * 0.900) / met;
    seen.occlusion = mix(COVER_ROOT_OCCLUSION, 1.0, along);
    return seen;
}
