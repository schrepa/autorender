fn variety_hash(cell: vec2<i32>) -> f32 {
    var h = (u32(cell.x) * 0x8da6b343u) ^ (u32(cell.y) * 0xd8163841u);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return f32((h ^ (h >> 16u)) >> 8u) * (1.0 / 16777216.0);
}

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

const COVER_LUSH_RATIO: f32 = 2.0;

const COVER_LUSH_REFLECTANCE: vec3<f32> = vec3<f32>(0.67, 0.8, 0.89);

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

fn macro_tint(colors: vec4<f32>, vegetated: f32) -> vec3<f32> {
    let growth = cover_growth(colors.a);
    return colors.rgb * (255.0 / 128.0) * mix(vec3<f32>(1.0), growth.tint, vegetated);
}

const COVER_CLUMP_LATTICES: vec2<f32> = vec2<f32>(0.4, 1.6);

const COVER_CLUMP_HEIGHT: f32 = 0.4;
const COVER_CLUMP_DRY: f32 = 0.4;
const COVER_CLUMP_HUE: f32 = 0.1;
const COVER_CLUMP_BRIGHT: f32 = 0.3;
const COVER_CLUMP_LEAN: f32 = 0.35;

fn cover_clump_spacing(layer: u32) -> f32 {
    return select(COVER_CLUMP_LATTICES.x, COVER_CLUMP_LATTICES.y, (layer & 1u) == 1u);
}

fn cover_clump_hash(cell: vec2<i32>, layer: u32) -> vec4<f32> {
    var h = (u32(cell.x) * 0x8da6b343u) ^ (u32(cell.y) * 0xd8163841u)
        ^ ((layer + 1u) * 0x2c1b3c6du);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return unpack4x8unorm(h ^ (h >> 16u));
}

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

fn cover_clump(at: vec2<f32>) -> vec4<f32> {
    return 0.5 * (cover_clump_layer(at, 0u) + cover_clump_layer(at, 1u));
}

fn cover_clump_lean(at: vec2<f32>) -> vec2<f32> {
    let layers = cover_clump_layer(at, 2u).xy + cover_clump_layer(at, 3u).xy;
    return 0.5 * COVER_CLUMP_LEAN * layers;
}

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

fn spread_slopes(shortened: f32) -> f32 {
    let kept = clamp(shortened, 1.0e-4, 1.0);
    return 2.0 * (1.0 - kept) / kept;
}

const COVER_LEAVES: u32 = 5u;
const COVER_LEAF_STEPS: u32 = 255u / COVER_LEAVES;

const COVER_LEAF_ASPECT: f32 = 0.1;

const COVER_FAR_SIDE: f32 = 0.8;
const COVER_ROOT_OCCLUSION: f32 = 0.35;
const COVER_ROUGHNESS: f32 = 0.7;

fn seen_share(pixels: f32) -> f32 {
    return saturate(1.0 + log2(max(pixels, 1.0e-6)));
}

struct CoverLaw {
    span: f32,
}

fn cover_widening(away: f32, law: CoverLaw) -> f32 {
    return clamp(away / max(law.span, 1.0e-6), 1.0, f32(COVER_LEAVES));
}

fn cover_tuft(away: f32, height: f32, root: f32, law: CoverLaw) -> vec2<f32> {
    let tuft = root * cover_widening(away, law);
    let leaf = min(tuft, max(root, COVER_LEAF_ASPECT * height));
    return vec2<f32>(leaf, clamp(tuft / leaf, 1.0, f32(COVER_LEAVES)));
}

fn cover_share(away: f32, law: CoverLaw) -> f32 {
    let widening = cover_widening(away, law);
    return seen_share(law.span * widening / max(away, 1.0e-3)) / widening;
}

fn cover_shown(away: f32, share: f32, law: CoverLaw) -> f32 {
    return min(share * cover_widening(away, law), 1.0);
}

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
