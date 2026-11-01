import package::cover::laws::{COVER_CLUMP_BRIGHT, COVER_CLUMP_DRY, COVER_CLUMP_HEIGHT,
    COVER_CLUMP_HUE, COVER_CLUMP_LATTICES, COVER_LUSH_REFLECTANCE, COVER_ROOT_OCCLUSION,
    cover_clump_hash, cover_widening};
import package::engine::view::seen_share;
import package::face::ground_tiles::PALETTE_ENTRIES;
import package::generated::cover::{COVER_LUSH_RATIO, COVER_MATERIAL_ROWS, CoverMaterial};
import package::place::cover::{CoverClumped, CoverGrowth, CoverLaw, CoverSeen};

const COVER_FAR_SIDE: f32 = 0.8;
const COVER_ROUGHNESS: f32 = 0.7;

const COVER_FIRST_TERRAIN: u32 = COVER_MATERIAL_ROWS * PALETTE_ENTRIES;

@group(0) @binding(0) var<storage, read> cover_table: array<vec4<u32>>;

fn cover_growth(lush: f32) -> CoverGrowth {
    let toward = saturate(lush) - 0.5;
    var growth: CoverGrowth;
    growth.height = pow(COVER_LUSH_RATIO, toward);
    growth.density = growth.height;
    growth.tint = pow(COVER_LUSH_REFLECTANCE, vec3<f32>(toward));
    return growth;
}

fn cover_tint(lush: f32, vegetated: f32) -> vec3<f32> {
    let growth = cover_growth(lush);
    return mix(vec3<f32>(1.0), growth.tint, vegetated);
}

fn cover_clump_spacing(layer: u32) -> f32 {
    return select(COVER_CLUMP_LATTICES.x, COVER_CLUMP_LATTICES.y, (layer & 1u) == 1u);
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

fn cover_clumped(look: vec4<f32>) -> CoverClumped {
    var clumped: CoverClumped;
    clumped.height = 1.0 + COVER_CLUMP_HEIGHT * look.x;
    let dry = pow(COVER_LUSH_REFLECTANCE, vec3<f32>(-COVER_CLUMP_DRY * look.y));
    let hue = vec3<f32>(1.0 + COVER_CLUMP_HUE * look.z, 1.0, 1.0 - COVER_CLUMP_HUE * look.z);
    clumped.tint = dry * hue * (1.0 + COVER_CLUMP_BRIGHT * look.w);
    return clumped;
}

fn cover_share(away: f32, law: CoverLaw) -> f32 {
    let widening = cover_widening(away, law);
    return seen_share(law.span * widening / max(away, 1.0e-3)) / widening;
}

fn cover_shown(away: f32, share: f32, law: CoverLaw) -> f32 {
    return min(share * cover_widening(away, law), 1.0);
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

fn material_cover(base: u32, material: u32) -> CoverMaterial {
    let at = COVER_MATERIAL_ROWS * min(base + material, PALETTE_ENTRIES - 1u);
    var spec: CoverMaterial;
    spec.shape = bitcast<vec4<f32>>(cover_table[at]);
    spec.color = bitcast<vec4<f32>>(cover_table[at + 1u]);
    spec.canopy = bitcast<vec4<f32>>(cover_table[at + 2u]);
    return spec;
}

fn material_canopy(terrain: u32, material: u32) -> vec4<f32> {
    return material_cover(cover_table[COVER_FIRST_TERRAIN + terrain].x, material).canopy;
}

fn cover_terms(terrain: u32) -> vec2<f32> {
    return bitcast<vec2<f32>>(cover_table[COVER_FIRST_TERRAIN + terrain].yz);
}
