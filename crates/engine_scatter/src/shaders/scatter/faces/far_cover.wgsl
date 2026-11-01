import package::generated::scatter::PlacedFar;
import package::place::far_cover::{GroundAt, PALETTE_ENTRIES};

@group(0) @binding(0) var<storage, read> placed_far_bases: array<u32>;
@group(0) @binding(0) var<storage, read> placed_far: array<PlacedFar>;

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

fn far_color(color: vec3<f32>, at: GroundAt) -> vec3<f32> {
    if at.terrain >= arrayLength(&placed_far_bases) {
        return color;
    }
    let index = min(placed_far_bases[at.terrain] + at.material, PALETTE_ENTRIES - 1u);
    if index >= arrayLength(&placed_far) {
        return color;
    }
    let entry = placed_far[index];
    let placed = placed_share(entry, at.across);
    if placed > 0.0 {
        return mix(color, entry.placed.rgb, placed);
    }
    return color;
}
