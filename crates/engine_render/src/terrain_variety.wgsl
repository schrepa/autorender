// Ground variety: the tint a terrain's macro colors give its
// ground and the blades growing on it, how lush ground grows them, and the
// clumps they grow in, the same in both and at every distance
// (`meshes/terrain.wgsl`, `cover_grow.wgsl`). Ground cover's law, which both
// read too, follows, with how things retire as they shrink under a pixel on
// screen (ground cover's tufts and placed objects alike), and what a view
// sees of blades, which their canopy shows where they retire.

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

// How many times as tall, and as dense, ground cover grows on a terrain's
// lushest ground as on its driest (`gpu_layouts::constants::COVER_LUSH_RATIO`).
const COVER_LUSH_RATIO: f32 = 2.0;

// The light ground cover's blades reflect on a terrain's lushest ground over
// what the same material's reflect on its driest, red, green and blue.
// Leaves short of water lose chlorophyll and reflect more of the light, most
// from green to red and least in the blue (Carter 1993, "Responses of leaf
// spectral reflectance to plant stress"): the lushest reflect two thirds of
// the driest's red, four fifths of their green and nine tenths of their blue,
// so dry grass shows paler and yellower, lush grass darker and greener, the
// driest reflecting about 30% more light than the lushest.
const COVER_LUSH_REFLECTANCE: vec3<f32> = vec3<f32>(0.67, 0.8, 0.89);

// How ground cover grows where the macro colors hold plants `lush` (0 dry,
// 1 lush), as shares of the cover its material names: as tall and as dense,
// and each channel of the light its blades reflect (`tint`). Lushness moves
// them between the driest ground's and the lushest's geometrically about the
// material's own, which grows halfway (and before the colors arrive).
struct CoverGrowth {
    height: f32,
    density: f32,
    tint: vec3<f32>,
}

fn cover_growth(lush: f32) -> CoverGrowth {
    let toward = saturate(lush) - 0.5;
    var growth: CoverGrowth;
    growth.height = pow(COVER_LUSH_RATIO, toward);
    growth.density = growth.height;
    growth.tint = pow(COVER_LUSH_REFLECTANCE, vec3<f32>(toward));
    return growth;
}

// The tint of ground whose macro colors' texel holds `colors` (the tint, 128
// of 255 leaving it as it is, and how lush plants grow there:
// `engine_terrain::macro_colors`), `vegetated` the share of it growing
// plants: every ground takes the tint as the macro colors hold it, near and
// far alike, and plants the light they reflect as lush as it holds them
// (`cover_growth`): darker and greener where lush, paler and yellower where
// dry.
fn macro_tint(colors: vec4<f32>, vegetated: f32) -> vec3<f32> {
    let growth = cover_growth(colors.a);
    return colors.rgb * (255.0 / 128.0) * mix(vec3<f32>(1.0), growth.tint, vegetated);
}

// Grass grows in tussocks gathered into clumps, each standing taller or
// shorter than its neighbors, leaning its own way, a little yellower or
// bluer, drier or lusher, brighter or darker. The clump field's two
// lattices (meters between their points): tussocks a few decimeters across,
// and clumps of them a meter or two across.
const COVER_CLUMP_LATTICES: vec2<f32> = vec2<f32>(0.4, 1.6);

// How far a clump's blades grow from the cover around them where the clump
// field stands at an extreme: two fifths taller or shorter; their leaves as
// pale and yellow, or as dark and green, as ground two fifths of the way
// from a terrain's driest to its lushest drier or lusher grows them
// (`cover_growth`'s tint); a tenth more red and less blue or the other way
// about; three tenths brighter or darker; and leaning along x and z each as
// far as a blade's own curl leans it on average (`meshes/cover.wgsl`'s
// `cover_point`).
const COVER_CLUMP_HEIGHT: f32 = 0.4;
const COVER_CLUMP_DRY: f32 = 0.4;
const COVER_CLUMP_HUE: f32 = 0.1;
const COVER_CLUMP_BRIGHT: f32 = 0.3;
const COVER_CLUMP_LEAN: f32 = 0.35;

// The meters between the clump field's lattice points in layer `layer`: its
// tussocks' in even layers, its clumps' in odd ones.
fn cover_clump_spacing(layer: u32) -> f32 {
    return select(COVER_CLUMP_LATTICES.x, COVER_CLUMP_LATTICES.y, (layer & 1u) == 1u);
}

// Four values (0 to 1, a byte each) of the clump field's lattice point
// `cell` in layer `layer`, each layer its own, and none the ground's noise
// shares.
fn cover_clump_hash(cell: vec2<i32>, layer: u32) -> vec4<f32> {
    var h = (u32(cell.x) * 0x8da6b343u) ^ (u32(cell.y) * 0xd8163841u)
        ^ ((layer + 1u) * 0x2c1b3c6du);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return unpack4x8unorm(h ^ (h >> 16u));
}

// Layer `layer` of the clump field at `at` (world x and z): value noise of
// four channels on its lattice (`cover_clump_spacing`), each -1 to 1 and 0
// on average, its points' hashes blended by smooth steps.
fn cover_clump_layer(at: vec2<f32>, layer: u32) -> vec4<f32> {
    let p = at / cover_clump_spacing(layer);
    let cell = vec2<i32>(floor(p));
    let f = p - floor(p);
    let s = f * f * (3.0 - 2.0 * f);
    let a = cover_clump_hash(cell, layer);
    let b = cover_clump_hash(cell + vec2<i32>(1, 0), layer);
    let c = cover_clump_hash(cell + vec2<i32>(0, 1), layer);
    let d = cover_clump_hash(cell + vec2<i32>(1, 1), layer);
    return mix(mix(a, b, s.x), mix(c, d, s.x), s.y) * 2.0 - 1.0;
}

// The look of ground cover's clump at `at` (world x and z), half its
// tussocks' layer and half its clumps': how much taller (x), drier (y),
// yellower (z) and brighter (w) its blades grow than the cover around them,
// each -1 to 1 and 0 on average (`cover_clumped`). Blades take it at their
// roots, and their canopy as a pixel's footprint averages it
// (`meshes/terrain.wgsl`'s `terrain_canopy_clump`).
fn cover_clump(at: vec2<f32>) -> vec4<f32> {
    return 0.5 * (cover_clump_layer(at, 0u) + cover_clump_layer(at, 1u));
}

// The lean of ground cover's blades in the clump at `at` (world x and z), as
// a blade's curl leans it (`meshes/cover.wgsl`'s `cover_point`): along x and
// z, up to COVER_CLUMP_LEAN each, from the clump field's next two layers as
// `cover_clump` lays its look.
fn cover_clump_lean(at: vec2<f32>) -> vec2<f32> {
    let layers = cover_clump_layer(at, 2u).xy + cover_clump_layer(at, 3u).xy;
    return 0.5 * COVER_CLUMP_LEAN * layers;
}

// How ground cover grows in a clump whose look is `look` (`cover_clump`, or
// its mean over a footprint), as shares of the cover around it: as tall,
// and each channel of the light its blades' leaves reflect, their tips' and
// their canopy's alike. The mean look grows it as it is.
struct CoverClumped {
    height: f32,
    tint: vec3<f32>,
}

fn cover_clumped(look: vec4<f32>) -> CoverClumped {
    var clumped: CoverClumped;
    clumped.height = 1.0 + COVER_CLUMP_HEIGHT * look.x;
    let dry = pow(COVER_LUSH_REFLECTANCE, vec3<f32>(-COVER_CLUMP_DRY * look.y));
    let hue = vec3<f32>(1.0 + COVER_CLUMP_HUE * look.z, 1.0, 1.0 - COVER_CLUMP_HUE * look.z);
    clumped.tint = dry * hue * (1.0 + COVER_CLUMP_BRIGHT * look.w);
    return clumped;
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

// The most leaves a tuft of ground cover holds, and so the most blades it
// stands for (`cover_tuft`, `gpu_layouts::constants::COVER_LEAVES`); and the
// steps of a leaf a blade's byte of leaves counts them in, as many as a byte
// holds for them all.
const COVER_LEAVES: u32 = 5u;
const COVER_LEAF_STEPS: u32 = 255u / COVER_LEAVES;

// The widest a leaf grows for its height and still reads as a grass leaf:
// the leaf blades of common meadow grasses are 3 to 10 mm wide and 10 to 50
// cm long (Hubbard 1984, "Grasses"), the widest of them a tenth as wide as
// they are long.
const COVER_LEAF_ASPECT: f32 = 0.1;

// Ground cover's blades as `meshes/cover.wgsl` shades them, which their
// canopy takes on average: the share of their color those seen through
// from the side away from the sun show, the light reaching their roots,
// rising to all of it at their tips, and their roughness.
const COVER_FAR_SIDE: f32 = 0.8;
const COVER_ROOT_OCCLUSION: f32 = 0.35;
const COVER_ROUGHNESS: f32 = 0.7;

// The share of things retiring by their size on screen that still show
// where each spans `pixels` pixels: each shows while it spans more than a
// pixel times a factor of its own, from a half to one by its rank
// (2^(rank - 1)), so those ranked under 1 + log2 pixels do: all from a
// pixel, none under half of one. Ground cover's tufts and the objects of
// placement layers retire so, in their ranks' order, and the ground takes
// on what they leave.
fn seen_share(pixels: f32) -> f32 {
    return saturate(1.0 + log2(max(pixels, 1.0e-6)));
}

// Ground cover's law on a terrain, which its blades and their canopy follow
// alike (`ground_cover::CoverLaw`): the meters from the detail eye within
// which a blade of its densest cover spans a pixel or more across the view,
// its roots' width over the angle a pixel spans.
struct CoverLaw {
    span: f32,
}

// How many blades each tuft of ground cover stands for `away` meters from
// the detail eye under `law`, holding their leaves (`cover_tuft`): one while
// a blade spans a pixel or more across the view, then as many as a pixel
// holds of them as they shrink under one, so each tuft spans a pixel and
// every pixel keeps the leaves the field gives it, up to COVER_LEAVES.
fn cover_widening(away: f32, law: CoverLaw) -> f32 {
    return clamp(away / max(law.span, 1.0e-6), 1.0, f32(COVER_LEAVES));
}

// A blade of ground cover `height` meters tall whose roots are `root`
// meters wide, `away` meters from the detail eye under `law`, as the tuft it
// stands as there: as wide at its roots in all as the blades it stands for
// (`cover_widening`), so it holds their leaves, in leaves as wide as their
// roots or as a leaf of its height grows (COVER_LEAF_ASPECT), whichever is
// wider, and no wider than the tuft. Each leaf's width (x, meters) and how
// many leaves it holds (y, 1 to COVER_LEAVES), the last of them, past the
// whole ones, standing that share of its height, so leaves rise one by one
// as their tuft widens.
fn cover_tuft(away: f32, height: f32, root: f32, law: CoverLaw) -> vec2<f32> {
    let tuft = root * cover_widening(away, law);
    let leaf = min(tuft, max(root, COVER_LEAF_ASPECT * height));
    return vec2<f32>(leaf, clamp(tuft / leaf, 1.0, f32(COVER_LEAVES)));
}

// Ground cover's density as a share of the density its cover names, `away`
// meters from the detail eye under `law`, the same for its tufts and the
// canopy shown where they retire. All of it while a blade spans a pixel or
// more across the view; then one blade in as many as each tuft stands for
// (`cover_widening`), so tufts a pixel wide hold the field's leaves; past
// COVER_LEAVES times `law.span`, where tufts hold all the leaves they can,
// they narrow under a pixel and retire by their size on screen
// (`seen_share`), none standing from twice that.
fn cover_share(away: f32, law: CoverLaw) -> f32 {
    let widening = cover_widening(away, law);
    // The pixels a tuft spans across the view, as wide as it grows there.
    return seen_share(law.span * widening / max(away, 1.0e-3)) / widening;
}

// How much of the leaves the field would show tufts growing at `share` of
// it, `away` meters from the detail eye under `law`, show: all those the
// share keeps while tufts widen as blades thin, fewer where tufts retire by
// their size on screen. The ground shows the rest as its canopy.
fn cover_shown(away: f32, share: f32, law: CoverLaw) -> f32 {
    return min(share * cover_widening(away, law), 1.0);
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
