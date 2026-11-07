import package::generated::debug_view::{PROBED_COUNT, PROBED_IDS, PROBED_WORDS};
import package::generated::pieces::debug_view::piece;
import package::generated::targets::{output_debug_count, output_debug_ids, output_debug_words};

@group(1) @binding(0) var<storage, read_write> probed: array<u32>;

@compute @workgroup_size(1)
fn cs_debug_probe() {
    let pixel = min(piece().probe.xy, textureDimensions(output_debug_ids) - vec2<u32>(1u));
    let ids = textureLoad(output_debug_ids, pixel, 0);
    let words = textureLoad(output_debug_words, pixel, 0).xy;
    probed[PROBED_IDS] = ids.x;
    probed[PROBED_IDS + 1u] = ids.y;
    probed[PROBED_IDS + 2u] = ids.z;
    probed[PROBED_IDS + 3u] = ids.w;
    probed[PROBED_COUNT] = pack2x16float(vec2<f32>(textureLoad(output_debug_count, pixel, 0).r, 0.0));
    probed[PROBED_WORDS] = words.x;
    probed[PROBED_WORDS + 1u] = words.y;
}
