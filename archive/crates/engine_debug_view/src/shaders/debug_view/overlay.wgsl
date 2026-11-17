import package::face::debug::{DebugShade, DebugTargets, shown};
import package::face::debug_modes::shade_kind;

@group(1) @binding(0) var<uniform> placed: vec4<u32>;
@group(1) @binding(1) var debug_ids: texture_2d<u32>;
@group(1) @binding(2) var debug_count: texture_2d<f32>;
@group(1) @binding(3) var debug_words: texture_2d<u32>;

@fragment
fn fs_debug_overlay(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy) - vec2<i32>(placed.xy);
    var targets: DebugTargets;
    targets.ids = textureLoad(debug_ids, pixel, 0);
    targets.count = textureLoad(debug_count, pixel, 0).r;
    targets.words = textureLoad(debug_words, pixel, 0).xy;
    var shade: DebugShade;
    return shown(shade_kind(shade, targets), targets);
}
