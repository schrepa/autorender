import package::face::cover::{CoverLaw, cover_clump_layer};

const COVER_LUSH_RATIO: f32 = 2.0;

const COVER_LUSH_REFLECTANCE: vec3<f32> = vec3<f32>(0.67, 0.8, 0.89);

const COVER_CLUMP_LATTICES: vec2<f32> = vec2<f32>(0.4, 1.6);

const COVER_CLUMP_HEIGHT: f32 = 0.4;
const COVER_CLUMP_DRY: f32 = 0.4;
const COVER_CLUMP_HUE: f32 = 0.1;
const COVER_CLUMP_BRIGHT: f32 = 0.3;
const COVER_CLUMP_LEAN: f32 = 0.35;

const COVER_LEAVES: u32 = 5u;
const COVER_LEAF_STEPS: u32 = 255u / COVER_LEAVES;

const COVER_LEAF_ASPECT: f32 = 0.1;

const COVER_ROOT_OCCLUSION: f32 = 0.35;

fn cover_clump_hash(cell: vec2<i32>, layer: u32) -> vec4<f32> {
    var h = (u32(cell.x) * 0x8da6b343u) ^ (u32(cell.y) * 0xd8163841u)
        ^ ((layer + 1u) * 0x2c1b3c6du);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return unpack4x8unorm(h ^ (h >> 16u));
}

fn cover_clump(at: vec2<f32>) -> vec4<f32> {
    return 0.5 * (cover_clump_layer(at, 0u) + cover_clump_layer(at, 1u));
}

fn cover_clump_lean(at: vec2<f32>) -> vec2<f32> {
    let layers = cover_clump_layer(at, 2u).xy + cover_clump_layer(at, 3u).xy;
    return 0.5 * COVER_CLUMP_LEAN * layers;
}

fn cover_widening(away: f32, law: CoverLaw) -> f32 {
    return clamp(away / max(law.span, 1.0e-6), 1.0, f32(COVER_LEAVES));
}

fn cover_tuft(away: f32, height: f32, root: f32, law: CoverLaw) -> vec2<f32> {
    let tuft = root * cover_widening(away, law);
    let leaf = min(tuft, max(root, COVER_LEAF_ASPECT * height));
    return vec2<f32>(leaf, clamp(tuft / leaf, 1.0, f32(COVER_LEAVES)));
}
