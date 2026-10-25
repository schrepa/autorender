// Ground variety: the tint a terrain's own detail and macro colors give
// its ground and the blades growing on it, the same in both and at every
// distance (`meshes/terrain.wgsl`, `cover_grow.wgsl`). Ground cover's
// density, which both read too, follows, with how things retire by their
// size on screen (ground cover's blades and placed objects alike), how
// placement layers thin with distance, and what a view sees of blades as
// `meshes/cover.wgsl` shades them, which their canopy shows where they
// retire.

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
// plants: every ground takes the tint as the macro colors hold it, near and
// far alike, and plants grow lusher and darker where it holds them lush,
// drier and paler where dry.
fn macro_tint(colors: vec4<f32>, vegetated: f32) -> vec3<f32> {
    let growth = mix(vec3<f32>(1.05, 1.02, 0.88), vec3<f32>(0.85, 0.97, 0.82), colors.a);
    return colors.rgb * (255.0 / 128.0) * mix(vec3<f32>(1.0), growth, vegetated);
}

// Value noise of the ground's detail, 0 to 1.
fn terrain_value(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * (3.0 - 2.0 * f);
    // The four corners' hashes: the next cell's products are the cell's
    // plus each multiplier (wrapping integers), one product per axis.
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

// The hash of a lattice cell, 0 to 1, from its coordinates' products, x
// times 0x8da6b343 and y times 0xd8163841 (wrapping), mixed.
fn terrain_hash_of(mixed: u32) -> f32 {
    var h = mixed;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return f32((h ^ (h >> 16u)) >> 8u) * (1.0 / 16777216.0);
}

// Noise of feature size `size` meters at `p`, as a pixel spanning
// `footprint` meters averages it: whole while the pixel spans half a
// feature or less, its mean once it spans a whole one. A noise computed
// where it shows has no mipmaps to filter it, and a pixel spanning its
// features would otherwise show one of them at random.
fn terrain_detail(p: vec2<f32>, size: f32, footprint: f32) -> f32 {
    let kept = saturate(2.0 - 2.0 * footprint / size);
    // Averaged away entirely, it is its mean.
    if kept == 0.0 {
        return 0.5;
    }
    return 0.5 + (terrain_value(p / size) - 0.5) * kept;
}

// How the ground's own detail varies its color at `world` (its x and z),
// the share `mapped` of it having maps, as a pixel spanning `footprint`
// meters averages it: plain colors vary at every scale, maps carry their
// own detail and vary broadly. Blades growing on the ground take it at
// their roots (`cover_grow.wgsl`), so the blades a pixel holds average to
// what the ground's pixel shows.
fn ground_detail(world: vec2<f32>, mapped: f32, footprint: f32) -> f32 {
    let broad = terrain_detail(world + vec2<f32>(-31.7, 12.3), 60.0, footprint);
    let patchy = terrain_detail(world + vec2<f32>(17.3, 41.9), 9.0, footprint);
    let fine = terrain_detail(world + vec2<f32>(-5.1, 3.7), 1.6, footprint);
    let plain = (0.8 + 0.4 * patchy) * (0.85 + 0.3 * fine);
    let textured = (0.86 + 0.28 * broad) * (0.92 + 0.16 * patchy);
    return mix(plain, textured, mapped);
}

// The mean square slope, both axes summed, of unit normals whose mean is
// `shortened` long, their spread about it shortening it so (Toksvig 2005,
// "Mipmapping normal maps"): 2 (1 - length) / length. Added to the square
// of a surface's alpha (its roughness squared), it widens its highlights as
// those slopes do.
fn spread_slopes(shortened: f32) -> f32 {
    let kept = clamp(shortened, 1.0e-4, 1.0);
    return 2.0 * (1.0 - kept) / kept;
}

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

// The share of a placement layer's objects still standing `away` meters from
// the detail eye as the layer thins with distance, `thin` holding the share
// standing far away, the meters its own tiles and its far tiles reach (0
// where they stay resident everywhere) and 1 where it thins at all: all of
// them within seven tenths of its tiles' reach, falling to its far share
// across most of the rest, then to none across the same part of its far
// tiles' reach. Objects ranked past it hide before their tiles leave, so
// density never steps where tiles end (`scatter_cull.wgsl`), and the ground
// takes on those it hides as it does those retiring by their size
// (`meshes/terrain.wgsl`'s `terrain_placed_share`).
fn placed_thinned(thin: vec4<f32>, away: f32) -> f32 {
    if thin.w == 0.0 {
        return 1.0;
    }
    var keep = 1.0;
    if thin.y > 0.0 {
        keep = mix(1.0, thin.x, smoothstep(0.7 * thin.y, 0.95 * thin.y, away));
    }
    if thin.z > 0.0 {
        keep *= 1.0 - smoothstep(0.7 * thin.z, 0.95 * thin.z, away);
    }
    return keep;
}

// How many times wider than their roots ground cover's blades stand `away`
// meters from the detail eye, for cover whose blades as wide as their roots
// span the settings' pixels `thin` meters away: as wide as their roots
// within that, and past it as wide as spans those pixels on screen, so they
// never stand narrower; stochastic simplification of aggregate detail
// (Cook, Halstead, Planck and Ryu 2007) keeps as small a share of them,
// by their fixed ranks, so they cover as much ground (`cover_share`).
fn cover_widening(away: f32, thin: f32) -> f32 {
    return max(away / max(thin, 1.0e-6), 1.0);
}

// The share of ground cover's leaves its blades show `away` meters from the
// detail eye, for cover whose blades' height spans the settings' pixels
// `retire` meters away: those still tall enough on screen to stand
// (`seen_share`), all of them within half `retire` and none from twice it,
// by their size on screen alone. The ground shows the rest as its canopy,
// their average (`meshes/terrain.wgsl`'s `terrain_thinned`).
fn cover_shown(away: f32, retire: f32) -> f32 {
    return seen_share(retire / max(away, 1.0e-3));
}

// Ground cover's density as a share of its densest, `away` meters from the
// detail eye, for cover whose blades thin past `thin` meters and retire
// around `retire` (`cover_widening`, `cover_shown`): the leaves they show
// over how much wider they stand, so the blades standing there, by their
// fixed ranks and widened, cover just as much ground as all of them would.
fn cover_share(away: f32, thin: f32, retire: f32) -> f32 {
    return cover_shown(away, retire) / cover_widening(away, thin);
}

// Ground cover's blades as `meshes/cover.wgsl` shades them, which their
// canopy takes on average: the share of their color those seen through
// from the side away from the sun show, the light reaching their roots,
// rising to all of it at their tips, and their roughness; how far up from
// their lit faces their normals lean, how far across their width those
// turn so the blades read round, and the share of their length over which
// their color turns from their roots' to their tips'.
const COVER_FAR_SIDE: f32 = 0.8;
const COVER_ROOT_OCCLUSION: f32 = 0.35;
const COVER_ROUGHNESS: f32 = 0.7;
const COVER_UPRIGHT: f32 = 0.6;
const COVER_ROUND: f32 = 0.45;
const COVER_TIP_ALONG: f32 = 0.7;

// The share of their leaves' area blades turn toward a view along the
// ground and toward one from straight above (Ross's G): blades curving up
// from upright at their roots to leaning at their tips
// (`meshes/cover.wgsl`'s `cover_point` at its mean curl), their width
// tapering as 1 - along^1.5, their faces turned every way across the
// ground: 2 / pi of the upright share of their area, and its leaning share.
// Views between meet them as an ellipsoid of leaf angles does (Campbell
// 1986), within about a hundredth.
const COVER_SIDE_PROJECTION: f32 = 0.6196;
const COVER_TOP_PROJECTION: f32 = 0.1793;

// How far the faces of blades turn across the ground on average, which
// their lit sides keep of a lean toward the sun: their curve's upright
// share of their area, 0.9733, times their width's turn across it,
// asinh(COVER_ROUND) / COVER_ROUND.
const COVER_LIT_ACROSS: f32 = 0.9431;

// The share of their leaves' area blades turn toward a view rising `rise`
// above the horizon (the sine of its angle), and the share of that their
// upright sides turn to it, the rest their leaning faces from above.
fn cover_projection(rise: f32) -> vec2<f32> {
    let side = COVER_SIDE_PROJECTION * sqrt(max(1.0 - rise * rise, 0.0));
    let top = COVER_TOP_PROJECTION * abs(rise);
    return vec2<f32>(sqrt(side * side + top * top), side / max(side + top, 1.0e-6));
}

// What a view sees of ground cover's blades on average: the share of their
// tips' color in the colors it sees, the share of light their occlusion
// leaves, and how often it meets each of four bands of their length first,
// root to tip. Each blade curves up from its root (`meshes/cover.wgsl`'s
// `cover_point`, at its mean curl), its width tapering as 1 - along^1.5,
// turning from its root's color to its tip's by smoothstep(0,
// COVER_TIP_ALONG, along) and from COVER_ROOT_OCCLUSION to no occlusion at
// its tip. A view whose path through the blades crosses `depth` of their
// leaves (`meshes/terrain.wgsl`'s `terrain_canopy`) passes the share F of
// their leaf area lying above a point along them as often as exp(-depth F)
// (Beer-Lambert), so it meets each band of their length first as often as
// that falls across the band. The bands end at along 0.35, 0.7 and 0.85,
// where F, their area from there to the tip along the curve over all of
// it, is 0.4274, 0.0915 and 0.0230, and each shows its own means over its
// area: COVER_BAND_TIP of the tip's color, COVER_BAND_ALONG of the way up.
// Seen down through sparse blades, the tips make 0.45 of the color; along
// dense ones, nearly all of it.
struct CoverSeen {
    tip: f32,
    occlusion: f32,
    bands: vec4<f32>,
}

const COVER_BAND_TIP: vec4<f32> = vec4<f32>(0.1709, 0.7804, 1.0, 1.0);
const COVER_BAND_ALONG: vec4<f32> = vec4<f32>(0.1640, 0.5036, 0.7667, 0.9001);

fn cover_seen(depth: f32) -> CoverSeen {
    let crossed = max(depth, 1.0e-3);
    let root = exp(-crossed);
    let low = exp(-0.4274 * crossed);
    let middle = exp(-0.0915 * crossed);
    let high = exp(-0.0230 * crossed);
    var seen: CoverSeen;
    seen.bands = vec4<f32>(low - root, middle - low, high - middle, 1.0 - high)
        / (1.0 - root);
    seen.tip = dot(seen.bands, COVER_BAND_TIP);
    seen.occlusion = mix(COVER_ROOT_OCCLUSION, 1.0, dot(seen.bands, COVER_BAND_ALONG));
    return seen;
}

// The share of the ground's normal that the blades a view sees (`seen`),
// `height` meters tall on average, light with where foliage's ground blend
// turns their normals to the ground's up to `reach` meters above it
// (`meshes/cover.wgsl`'s `cover_surface`): all of it at the ground, none
// from `reach` up, each band at its mean height; none without the blend.
fn cover_grounded(seen: CoverSeen, height: f32, reach: f32) -> f32 {
    if reach <= 0.0 {
        return 0.0;
    }
    let kept = smoothstep(vec4<f32>(0.0), vec4<f32>(reach), COVER_BAND_ALONG * height);
    return dot(seen.bands, vec4<f32>(1.0) - kept);
}
