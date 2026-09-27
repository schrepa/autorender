// Temporal antialiasing resolve (decision 017). One full-screen triangle per
// display frame blends this frame's jittered color with the previous output,
// reprojected by per-pixel motion. The history is clamped to the box of the
// current pixel and its four edge neighbors in YCoCg, which bounds ghosting
// from disocclusion, changed shading and approximate motion. Pixels whose
// reprojection leaves the viewport, and every pixel after a reset, show this
// frame's color alone. Writes the canvas and the next history together.

struct Resolve {
    // Size of one target pixel in texture coordinates.
    texel: vec2<f32>,
    // Weight of this frame's color against the reprojected history.
    current_weight: f32,
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

// sRGB targets apply output encoding themselves; UNORM canvas targets do not.
override SURFACE_IS_SRGB: bool = false;

// Below this motion (a 64th of a pixel) the history is read at the pixel
// itself: static pixels skip resampling.
const STILL_PIXELS: f32 = 0x1p-6f;

struct Outputs {
    @location(0) canvas: vec4<f32>,
    @location(1) history: vec4<f32>,
}

// One triangle covering the viewport: (-1, -1), (3, -1) and (-1, 3).
@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
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
fn sample_history(uv: vec2<f32>) -> vec3<f32> {
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
    let sum = middle_weight + weights.x + weights.y + weights.z + weights.w;
    let color = textureSampleLevel(history, history_sampler, middle, 0.0).rgb * middle_weight
        + textureSampleLevel(history, history_sampler, vec2<f32>(middle.x, before.y), 0.0).rgb * weights.x
        + textureSampleLevel(history, history_sampler, vec2<f32>(before.x, middle.y), 0.0).rgb * weights.y
        + textureSampleLevel(history, history_sampler, vec2<f32>(after.x, middle.y), 0.0).rgb * weights.z
        + textureSampleLevel(history, history_sampler, vec2<f32>(middle.x, after.y), 0.0).rgb * weights.w;
    return color / sum;
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let low = color * 12.92;
    let high = 1.055 * pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, color <= vec3<f32>(0.0031308));
}

@fragment
fn fs_resolve(@builtin(position) position: vec4<f32>) -> Outputs {
    let pixel = vec2<i32>(position.xy);
    let current = textureLoad(scene, pixel, 0).rgb;
    var color = current;
    let offset = textureLoad(motion, pixel, 0).xy;
    let uv = position.xy * resolve.texel + offset;
    // NaN motion fails these comparisons too.
    if resolve.history_valid != 0u && all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)) {
        let last = vec2<i32>(textureDimensions(scene)) - 1;
        let center = ycocg(current);
        let left = ycocg(textureLoad(scene, max(pixel - vec2<i32>(1, 0), vec2<i32>(0)), 0).rgb);
        let right = ycocg(textureLoad(scene, min(pixel + vec2<i32>(1, 0), last), 0).rgb);
        let up = ycocg(textureLoad(scene, max(pixel - vec2<i32>(0, 1), vec2<i32>(0)), 0).rgb);
        let down = ycocg(textureLoad(scene, min(pixel + vec2<i32>(0, 1), last), 0).rgb);
        let low = min(center, min(min(left, right), min(up, down)));
        let high = max(center, max(max(left, right), max(up, down)));
        var past: vec3<f32>;
        if all(abs(offset) < resolve.texel * STILL_PIXELS) {
            past = textureLoad(history, pixel, 0).rgb;
        } else {
            past = sample_history(uv);
        }
        color = mix(rgb(clamp(ycocg(past), low, high)), current, resolve.current_weight);
    }
    color = max(color, vec3<f32>(0.0));
    var output: Outputs;
    output.history = vec4<f32>(color, 1.0);
    if SURFACE_IS_SRGB {
        output.canvas = vec4<f32>(color, 1.0);
    } else {
        output.canvas = vec4<f32>(linear_to_srgb(color), 1.0);
    }
    return output;
}
