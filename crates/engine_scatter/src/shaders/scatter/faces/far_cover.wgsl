import package::generated::scatter::PlacedFar;
import package::place::far_cover::{GroundAt, PALETTE_ENTRIES};

@group(0) @binding(0) var<storage, read> placed_far: array<vec4<u32>>;

fn placed_share(entry: PlacedFar, across: f32) -> f32 {
    let covered = entry.placed.w;
    if covered <= 0.0 || across <= 0.0 {
        return 0.0;
    }
    let first = entry.size.x;
    let octaves = max(first - entry.size.y, 0.0);
    let kept = saturate((1.0 + first - log2(across)) / (1.0 + octaves));
    var shown = kept;
    if octaves > 1.0e-3 {
        shown = (1.0 - exp2(-2.0 * octaves * kept)) / (1.0 - exp2(-2.0 * octaves));
    }
    return covered * (1.0 - shown);
}

fn placed_far_entry(row: u32) -> PlacedFar {
    return PlacedFar(bitcast<vec4<f32>>(placed_far[row]),
        bitcast<vec4<f32>>(placed_far[row + 1u]));
}

fn far_color(color: vec3<f32>, at: GroundAt) -> vec3<f32> {
    let counts = placed_far[0];
    if at.terrain >= counts.x {
        return color;
    }
    let base = placed_far[1u + at.terrain / 4u][at.terrain % 4u];
    let index = min(base + at.material, PALETTE_ENTRIES - 1u);
    if index >= counts.z {
        return color;
    }
    let entry = placed_far_entry(counts.y + 2u * index);
    let placed = placed_share(entry, at.across);
    if placed > 0.0 {
        return mix(color, entry.placed.rgb, placed);
    }
    return color;
}
