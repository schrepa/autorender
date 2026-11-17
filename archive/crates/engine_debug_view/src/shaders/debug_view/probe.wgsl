import package::generated::debug_view::{PROBED_COUNT, PROBED_IDS, PROBED_WORDS};
import package::generated::pieces::debug_view::piece;

@group(1) @binding(0) var<storage, read_write> probed: array<u32>;
@group(1) @binding(1) var debug_ids: texture_2d<u32>;
@group(1) @binding(2) var debug_count: texture_2d<f32>;
@group(1) @binding(3) var debug_words: texture_2d<u32>;

@compute @workgroup_size(1)
fn cs_debug_probe() {
    let pixel = min(piece().probe.xy, textureDimensions(debug_ids) - vec2<u32>(1u));
    let ids = textureLoad(debug_ids, pixel, 0);
    let words = textureLoad(debug_words, pixel, 0).xy;
    probed[PROBED_IDS] = ids.x;
    probed[PROBED_IDS + 1u] = ids.y;
    probed[PROBED_IDS + 2u] = ids.z;
    probed[PROBED_IDS + 3u] = ids.w;
    probed[PROBED_COUNT] = pack2x16float(vec2<f32>(textureLoad(debug_count, pixel, 0).r, 0.0));
    probed[PROBED_WORDS] = words.x;
    probed[PROBED_WORDS + 1u] = words.y;
}
