// Temporal antialiasing resolve, prepended with effects.wgsl.
// One full-screen triangle per display frame blends this frame's jittered
// color with the previous output, reprojected by per-pixel motion. The
// history is clamped to the box of the current pixel and its four edge
// neighbors in YCoCg, which bounds ghosting from disocclusion, changed
// shading and approximate motion. Pixels whose reprojection leaves the
// viewport, and every pixel after a reset, show this frame's color alone.
// Ambient occlusion darkens this frame's color before blending, so the
// history averages its noise; bloom reaches the canvas only.
// Retain total and coarse color; signed detail is their difference. Each
// alpha stores sqrt(uncertainty), linear UNORM even though RGB is sRGB.
// Writes the canvas and both next histories together. Canvas alpha stays 1.

struct Resolve {
    // Size of one target pixel in texture coordinates.
    texel: vec2<f32>,
    // Keep new samples contributing after uncertainty has fallen.
    minimum_current_weight: f32,
    // Zero after a reset: the history must not be read.
    history_valid: u32,
}

@group(0) @binding(0) var<uniform> resolve: Resolve;
// This frame's linear color (an sRGB target), read texel by texel.
@group(0) @binding(1) var scene: texture_2d<f32>;
// Offset from each pixel to its position in the previous frame, in texture
// coordinates, from the opaque pass.
@group(0) @binding(2) var motion: texture_2d<f32>;
@group(0) @binding(3) var history: texture_2d<f32>;
@group(0) @binding(4) var history_sampler: sampler;
// The filtered ambient occlusion's visibility and the sun shafts' light,
// at half resolution.
@group(0) @binding(5) var occlusion: texture_2d<f32>;
@group(0) @binding(6) var bloom: texture_2d<f32>;
@group(0) @binding(7) var<uniform> post: Post;
// Reflections at half resolution; like bloom and sun shafts,
// they reach the canvas only.
@group(0) @binding(8) var reflections: texture_2d<f32>;
@group(0) @binding(9) var coarse_history: texture_2d<f32>;

// Below this motion (a 64th of a pixel) the history is read at the pixel
// itself: static pixels skip resampling.
const STILL_PIXELS: f32 = 0x1p-6f;
// Nonzero uncertainty must survive the 8-bit history alpha channel.
const MIN_UNCERTAINTY: f32 = 1.0 / (255.0 * 255.0);
const MIN_SAMPLE_UNCERTAINTY: f32 = 0x1p-12f;

struct Outputs {
    @location(0) canvas: vec4<f32>,
    @location(1) history: vec4<f32>,
    @location(2) coarse_history: vec4<f32>,
}

fn ycocg(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.25 * color.r + 0.5 * color.g + 0.25 * color.b,
        0.5 * color.r - 0.5 * color.b,
        -0.25 * color.r + 0.5 * color.g - 0.25 * color.b,
    );
}

fn rgb(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(color.x + color.y - color.z, color.x + color.z, color.x - color.y - color.z);
}

// The history at `uv`: Catmull-Rom over the surrounding 4x4 texels, with the
// middle pairs of each axis merged into one bilinear fetch and the four
// corners (under 3% of the weight) dropped. Plain bilinear fetches would
// accumulate a blur of over a pixel (standard deviation) on moving content at
// the default weight.
struct Footprint {
    middle: vec2<f32>,
    before: vec2<f32>,
    after: vec2<f32>,
    weights: vec4<f32>,
    middle_weight: f32,
}

fn footprint(uv: vec2<f32>) -> Footprint {
    let size = 1.0 / resolve.texel;
    let position = uv * size;
    let center = floor(position - 0.5) + 0.5;
    let f = position - center;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let middle = (center + w2 / w12) * resolve.texel;
    let before = (center - 1.0) * resolve.texel;
    let after = (center + 2.0) * resolve.texel;
    let weights = vec4<f32>(w12.x * w0.y, w0.x * w12.y, w3.x * w12.y, w12.x * w3.y);
    let middle_weight = w12.x * w12.y;
    return Footprint(middle, before, after, weights, middle_weight);
}

fn sample_history(source: texture_2d<f32>, at: Footprint) -> vec4<f32> {
    let sum = at.middle_weight + at.weights.x + at.weights.y + at.weights.z + at.weights.w;
    let middle_sample = textureSampleLevel(source, history_sampler, at.middle, 0.0);
    let up = textureSampleLevel(source, history_sampler, vec2<f32>(at.middle.x, at.before.y), 0.0);
    let left = textureSampleLevel(source, history_sampler, vec2<f32>(at.before.x, at.middle.y), 0.0);
    let right = textureSampleLevel(source, history_sampler, vec2<f32>(at.after.x, at.middle.y), 0.0);
    let down = textureSampleLevel(source, history_sampler, vec2<f32>(at.middle.x, at.after.y), 0.0);
    let color = middle_sample.rgb * at.middle_weight + up.rgb * at.weights.x
        + left.rgb * at.weights.y + right.rgb * at.weights.z + down.rgb * at.weights.w;
    // Signed reconstruction still sharpens RGB. Absolute weights keep the
    // resampled uncertainty nonnegative; this is a correlated-sample proxy.
    let positive = abs(at.weights);
    let middle_positive = abs(at.middle_weight);
    let alphas = vec4<f32>(up.a, left.a, right.a, down.a);
    let uncertainty = (middle_sample.a * middle_sample.a * middle_positive
        + dot(alphas * alphas, positive))
        / (middle_positive + positive.x + positive.y + positive.z + positive.w);
    return vec4<f32>(color / sum, uncertainty);
}

fn finite_sample(value: vec4<f32>) -> bool {
    return all((bitcast<vec4<u32>>(value) & vec4<u32>(0x7f800000u)) != vec4<u32>(0x7f800000u));
}

fn valid_history(value: vec4<f32>) -> bool {
    // Guard the arithmetic domain before subtraction, squaring or division.
    // The signed five-tap RGB reconstruction lies inside these bounds.
    return finite_sample(value) && all(value.rgb >= vec3<f32>(-1.0))
        && all(value.rgb <= vec3<f32>(2.0)) && value.a > 0.0 && value.a <= 1.0;
}

// Gain and posterior uncertainty. Guarded bands have |residual| <= 4,
// P <= 1 and R in [2^-12, 1], so products and denominators stay finite.
fn update_band(residual: f32, raw_uncertainty: f32, sample_uncertainty: f32) -> vec2<f32> {
    let uncertainty = clamp(raw_uncertainty, MIN_UNCERTAINTY, 1.0);
    let process = max(residual * residual - uncertainty - sample_uncertainty, 0.0);
    let predicted = clamp(uncertainty + process, MIN_UNCERTAINTY, 1.0);
    let gain = clamp(predicted / (predicted + sample_uncertainty), resolve.minimum_current_weight, 1.0);
    let remaining = 1.0 - gain;
    return vec2<f32>(gain, remaining * remaining * predicted + gain * gain * sample_uncertainty);
}

@fragment
fn fs_resolve(@builtin(position) position: vec4<f32>) -> Outputs {
    let pixel = vec2<i32>(position.xy);
    let texel = textureLoad(scene, pixel, 0);
    let screen = position.xy * resolve.texel;
    // The neighborhood darkens with this pixel: its box scales alike.
    let effects = screen_effects(occlusion, history_sampler, screen, post);
    let shade = effect_shade(texel.a, effects, post);
    let current = texel.rgb * shade;
    let last = vec2<i32>(textureDimensions(scene)) - 1;
    let left_rgb = textureLoad(scene, max(pixel - vec2<i32>(1, 0), vec2<i32>(0)), 0).rgb * shade;
    let right_rgb = textureLoad(scene, min(pixel + vec2<i32>(1, 0), last), 0).rgb * shade;
    let up_rgb = textureLoad(scene, max(pixel - vec2<i32>(0, 1), vec2<i32>(0)), 0).rgb * shade;
    let down_rgb = textureLoad(scene, min(pixel + vec2<i32>(0, 1), last), 0).rgb * shade;
    let upper_left_rgb = textureLoad(scene, clamp(pixel + vec2<i32>(-1, -1), vec2<i32>(0), last), 0).rgb * shade;
    let upper_right_rgb = textureLoad(scene, clamp(pixel + vec2<i32>(1, -1), vec2<i32>(0), last), 0).rgb * shade;
    let lower_left_rgb = textureLoad(scene, clamp(pixel + vec2<i32>(-1, 1), vec2<i32>(0), last), 0).rgb * shade;
    let lower_right_rgb = textureLoad(scene, clamp(pixel + vec2<i32>(1, 1), vec2<i32>(0), last), 0).rgb * shade;
    let coarse = (4.0 * current + 2.0 * (left_rgb + right_rgb + up_rgb + down_rgb)
        + upper_left_rgb + upper_right_rgb + lower_left_rgb + lower_right_rgb) / 16.0;
    let detail = current - coarse;
    let center = ycocg(current);
    let left = ycocg(left_rgb);
    let right = ycocg(right_rgb);
    let up = ycocg(up_rgb);
    let down = ycocg(down_rgb);
    let upper_left = ycocg(upper_left_rgb);
    let upper_right = ycocg(upper_right_rgb);
    let lower_left = ycocg(lower_left_rgb);
    let lower_right = ycocg(lower_right_rgb);
    let mean = (center.x + left.x + right.x + up.x + down.x
        + upper_left.x + upper_right.x + lower_left.x + lower_right.x) / 9.0;
    let deviation = vec4<f32>(left.x, right.x, up.x, down.x) - vec4<f32>(mean);
    let corner_deviation = vec4<f32>(upper_left.x, upper_right.x, lower_left.x, lower_right.x) - vec4<f32>(mean);
    let center_deviation = center.x - mean;
    let variance = (center_deviation * center_deviation + dot(deviation, deviation)
        + dot(corner_deviation, corner_deviation)) / 9.0;
    let coarse_uncertainty = clamp((9.0 / 64.0) * variance, MIN_SAMPLE_UNCERTAINTY, 1.0);
    let detail_uncertainty = clamp((41.0 / 64.0) * variance, MIN_SAMPLE_UNCERTAINTY, 1.0);
    var color = current;
    var total_next = current;
    var coarse_next = coarse;
    var next_coarse_uncertainty = coarse_uncertainty;
    var next_detail_uncertainty = detail_uncertainty;
    let offset = textureLoad(motion, pixel, 0).xy;
    let uv = position.xy * resolve.texel + offset;
    // Reject nonfinite motion explicitly before any history fetch.
    if resolve.history_valid != 0u && finite_sample(vec4<f32>(offset, 0.0, 0.0))
        && all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)) {
        let low = min(center, min(min(left, right), min(up, down)));
        let high = max(center, max(max(left, right), max(up, down)));
        let low9 = min(low, min(min(upper_left, upper_right), min(lower_left, lower_right)));
        let high9 = max(high, max(max(upper_left, upper_right), max(lower_left, lower_right)));
        var past: vec4<f32>;
        var past_coarse: vec4<f32>;
        if all(abs(offset) < resolve.texel * STILL_PIXELS) {
            let stored = textureLoad(history, pixel, 0);
            past = vec4<f32>(stored.rgb, stored.a * stored.a);
            let stored_coarse = textureLoad(coarse_history, pixel, 0);
            past_coarse = vec4<f32>(stored_coarse.rgb, stored_coarse.a * stored_coarse.a);
        } else {
            let at = footprint(uv);
            past = sample_history(history, at);
            past_coarse = sample_history(coarse_history, at);
        }
        if valid_history(past) && valid_history(past_coarse) {
            // Both innovations precede neighborhood clipping.
            let previous_detail = past.rgb - past_coarse.rgb;
            let coarse_update = update_band(ycocg(coarse).x - ycocg(past_coarse.rgb).x,
                past_coarse.a, coarse_uncertainty);
            let detail_update = update_band(ycocg(detail).x - ycocg(previous_detail).x,
                past.a, detail_uncertainty);
            let safe_coarse = clamp(rgb(clamp(ycocg(past_coarse.rgb), low9, high9)), vec3<f32>(0.0), vec3<f32>(1.0));
            let safe_detail = rgb(clamp(ycocg(past.rgb), low, high)) - safe_coarse;
            let raw_coarse = mix(safe_coarse, coarse, coarse_update.x);
            let raw_detail = mix(safe_detail, detail, detail_update.x);
            color = max(rgb(clamp(ycocg(raw_coarse + raw_detail), low, high)), vec3<f32>(0.0));
            total_next = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
            coarse_next = clamp(raw_coarse, vec3<f32>(0.0), vec3<f32>(1.0));
            // The stored means define next frame's implicit signed detail.
            // Account for clipping both means before retaining uncertainty.
            let coarse_correction = ycocg(coarse_next - raw_coarse).x;
            let detail_correction = ycocg((total_next - coarse_next) - raw_detail).x;
            next_coarse_uncertainty = clamp(coarse_update.y + coarse_correction * coarse_correction, MIN_UNCERTAINTY, 1.0);
            next_detail_uncertainty = clamp(detail_update.y + detail_correction * detail_correction, MIN_UNCERTAINTY, 1.0);
        }
    }
    color = max(color, vec3<f32>(0.0));
    var output: Outputs;
    output.history = vec4<f32>(total_next, sqrt(next_detail_uncertainty));
    output.coarse_history = vec4<f32>(coarse_next, sqrt(next_coarse_uncertainty));
    let shown = effect_light(color, effects, reflections, bloom, history_sampler, screen, post);
    // The UNORM canvas stores encoded color.
    output.canvas = vec4<f32>(encode_srgb(shown), 1.0);
    return output;
}
