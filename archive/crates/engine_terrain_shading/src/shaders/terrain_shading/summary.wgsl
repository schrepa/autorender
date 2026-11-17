import package::generated::terrain_shading::{CONTROL_PLANE, SITE_PLANE, SUMMARY_FIRST, SUMMARY_GROUP,
    SUMMARY_PLANES, TILE_PLANES, TILE_STRIDE, control_a, control_b, control_blend};

@group(1) @binding(0) var<storage, read> summary_layers: array<u32>;
@group(1) @binding(1) var summary_in: texture_2d_array<u32>;
@group(1) @binding(2) var summary_out: texture_storage_2d_array<rgba32uint, write>;
@group(1) @binding(3) var summary_heights: texture_2d_array<f32>;

const SUMMARY_SPAN: u32 = 1u << SUMMARY_FIRST;
const LAST_SAMPLE: u32 = TILE_STRIDE - 3u;
const BYTE_SHIFTS: vec4<u32> = vec4<u32>(0u, 8u, 16u, 24u);

struct Summary {
    heights: vec2<f32>,
    wide_heights: vec2<f32>,
    materials: vec2<u32>,
    low: vec4<u32>,
    high: vec4<u32>,
    wide_low: vec4<u32>,
    wide_high: vec4<u32>,
}

fn summary_from(heights: vec2<f32>) -> Summary {
    var summary: Summary;
    summary.heights = heights;
    summary.wide_heights = heights;
    summary.low = vec4<u32>(255u);
    summary.wide_low = vec4<u32>(255u);
    return summary;
}

fn material_bit(material: u32) -> u32 {
    return 1u << min(material, 31u);
}

fn site_bytes(word: u32) -> vec4<u32> {
    return (vec4<u32>(word) >> BYTE_SHIFTS) & vec4<u32>(255u);
}

fn site_word(bytes: vec4<u32>) -> u32 {
    return bytes.x | (bytes.y << 8u) | (bytes.z << 16u) | (bytes.w << 24u);
}

fn summary_store(cell: vec2<u32>, layer: u32, summary: Summary) {
    let plane = layer * SUMMARY_PLANES;
    textureStore(summary_out, cell, plane,
        vec4<u32>(bitcast<vec2<u32>>(summary.wide_heights), summary.materials.yx));
    textureStore(summary_out, cell, plane + 1u,
        vec4<u32>(site_word(summary.low), site_word(summary.high), site_word(summary.wide_low),
            site_word(summary.wide_high)));
    textureStore(summary_out, cell, plane + 2u,
        vec4<u32>(bitcast<vec2<u32>>(summary.heights), 0u, 0u));
}

@compute @workgroup_size(SUMMARY_GROUP, SUMMARY_GROUP)
fn tile_summary_first(@builtin(global_invocation_id) id: vec3<u32>) {
    let side = textureDimensions(summary_out).x;
    if id.x >= side || id.y >= side {
        return;
    }
    let layer = summary_layers[id.z];
    let first = id.xy * SUMMARY_SPAN;
    let last = first + vec2<u32>(SUMMARY_SPAN);
    let start = max(first, vec2<u32>(1u)) - vec2<u32>(1u);
    let end = min(last + vec2<u32>(1u), vec2<u32>(LAST_SAMPLE));
    var summary = summary_from(vec2<f32>(textureLoad(summary_heights, first + vec2<u32>(1u), layer,
        0).r));
    for (var z = start.y; z <= end.y; z++) {
        for (var x = start.x; x <= end.x; x++) {
            let sample = vec2<u32>(x, z);
            let texel = sample + vec2<u32>(1u);
            let control = textureLoad(summary_in, texel, layer * TILE_PLANES + CONTROL_PLANE, 0).x;
            let share = control_blend(control);
            var present = 0u;
            if share != 255u {
                present |= material_bit(control_a(control));
            }
            if share != 0u {
                present |= material_bit(control_b(control));
            }
            let site = site_bytes(textureLoad(summary_in, texel, layer * TILE_PLANES + SITE_PLANE,
                0).x);
            let height = textureLoad(summary_heights, texel, layer, 0).r;
            summary.wide_heights = vec2<f32>(min(summary.wide_heights.x, height),
                max(summary.wide_heights.y, height));
            summary.materials.y |= present;
            summary.wide_low = min(summary.wide_low, site);
            summary.wide_high = max(summary.wide_high, site);
            if all(sample >= first) && all(sample <= last) {
                summary.heights = vec2<f32>(min(summary.heights.x, height),
                    max(summary.heights.y, height));
                summary.materials.x |= present;
                summary.low = min(summary.low, site);
                summary.high = max(summary.high, site);
            }
        }
    }
    summary_store(id.xy, layer, summary);
}

@compute @workgroup_size(SUMMARY_GROUP, SUMMARY_GROUP)
fn tile_summary_down(@builtin(global_invocation_id) id: vec3<u32>) {
    let side = textureDimensions(summary_out).x;
    if id.x >= side || id.y >= side {
        return;
    }
    let layer = summary_layers[id.z];
    let plane = layer * SUMMARY_PLANES;
    let seed = textureLoad(summary_in, 2u * id.xy, plane + 2u, 0);
    var summary = summary_from(bitcast<vec2<f32>>(seed.xy));
    for (var index = 0u; index < 4u; index++) {
        let at = 2u * id.xy + vec2<u32>(index & 1u, index >> 1u);
        let wide = textureLoad(summary_in, at, plane, 0);
        let sites = textureLoad(summary_in, at, plane + 1u, 0);
        let own = textureLoad(summary_in, at, plane + 2u, 0);
        summary.heights = vec2<f32>(min(summary.heights.x, bitcast<f32>(own.x)),
            max(summary.heights.y, bitcast<f32>(own.y)));
        summary.wide_heights = vec2<f32>(min(summary.wide_heights.x, bitcast<f32>(wide.x)),
            max(summary.wide_heights.y, bitcast<f32>(wide.y)));
        summary.materials |= wide.wz;
        summary.low = min(summary.low, site_bytes(sites.x));
        summary.high = max(summary.high, site_bytes(sites.y));
        summary.wide_low = min(summary.wide_low, site_bytes(sites.z));
        summary.wide_high = max(summary.wide_high, site_bytes(sites.w));
    }
    summary_store(id.xy, layer, summary);
}
