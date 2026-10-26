// Temporal antialiasing resolve, prepended with effects.wgsl: a temporal
// upscaler. One full-screen triangle per display frame, over the display,
// blends this frame's jittered color, drawn at its render size, with the
// previous output, reprojected by per-pixel motion, or by that of the
// nearest surface around each pixel. Each display pixel takes the render
// sample whose jittered position lies nearest its center, weighing it by how
// near that is (fully within half a display pixel, none from a whole one),
// so over the jitter's cycle the history gathers every display pixel's own
// samples. The history is clamped to the box of the sample's render pixel
// and its four edge neighbors in YCoCg, or one rounded toward its eight
// neighbors' (Karis, "High Quality Temporal Supersampling", 2014), which
// bounds ghosting from disocclusion, changed shading and approximate motion.
// Where particles cover a pixel, this frame's color may weigh more (as FSR
// 2's reactive mask does). Pixels whose reprojection leaves the
// viewport, and every pixel after a reset, show this frame's color alone:
// its nearest sample where one lies within half a display pixel, else its
// samples around blended bilinearly. Ambient occlusion darkens this frame's
// color and the reflections are laid over it before blending, each render
// pixel taking those at its own center, so the history averages the
// occlusion's noise and gathers the reflections traced from another pixel of
// each 2x2 block every frame; the haze and bloom reach the canvas only.
// Writes the canvas and the next history together, both at the display's
// size.

struct Resolve {
    // Size of one display pixel in texture coordinates: the canvas's and the
    // history's.
    texel: vec2<f32>,
    // Weight of this frame's color against the reprojected history.
    current_weight: f32,
    // Zero after a reset: the history must not be read.
    history_valid: u32,
    // This frame's turn of the dither (the camera's `options.z`); 1 when
    // pixels follow the motion of the nearest surface around them; 1 when
    // the history keeps within the rounded box; and the least weight of this
    // frame's color per unit of particles' coverage (0: none).
    options: vec4<f32>,
    // The render size this frame drew its targets at, in pixels, from their
    // top-left corner (xy), and how far its samples moved off their pixels'
    // centers, in render pixels (zw): render pixel p shows the scene at
    // p + 0.5 - zw.
    render: vec4<f32>,
}

@group(0) @binding(0) var<uniform> resolve: Resolve;
// This frame's linear color (an sRGB target), read texel by texel.
@group(0) @binding(1) var scene: texture_2d<f32>;
// Offset from each pixel to its position in the previous frame, in texture
// coordinates of the view, from the opaque pass.
@group(0) @binding(2) var motion: texture_2d<f32>;
@group(0) @binding(3) var history: texture_2d<f32>;
@group(0) @binding(4) var history_sampler: sampler;
// The filtered ambient occlusion's visibility at half resolution.
@group(0) @binding(5) var occlusion: texture_2d<f32>;
@group(0) @binding(6) var bloom: texture_2d<f32>;
@group(0) @binding(7) var<uniform> post: Post;
// Reflections at half resolution, traced at render pixels: each render
// pixel's color takes them before the blend, so they enter the history.
@group(0) @binding(8) var reflections: texture_2d<f32>;
// The depth buffer (reversed: nearer surfaces are greater), for the nearest
// surface around each pixel.
@group(0) @binding(9) var depth_texture: texture_depth_2d;
// Transparency's revealage: y is the share of each pixel particles leave
// visible (1 where none drew); one texel leaving everything visible in
// frames drawing nothing transparent.
@group(0) @binding(10) var revealed: texture_2d<f32>;
// The haze filtered over time at half resolution (post.wgsl's), which the
// canvas alone shows.
@group(0) @binding(11) var haze: texture_2d<f32>;

// Below this motion (a 64th of a display pixel) the history is read at the
// pixel itself: static pixels skip resampling.
const STILL_PIXELS: f32 = 0x1p-6f;

struct Outputs {
    @location(0) canvas: vec4<f32>,
    @location(1) history: vec4<f32>,
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

// The motion of the nearest surface among render pixel `pixel` and its
// eight neighbors (`last` the last render pixel): where a nearer surface's
// edge moves over a farther one, the pixels around the edge follow it, so
// its history comes along instead of trailing behind.
fn nearest_motion(pixel: vec2<i32>, last: vec2<i32>) -> vec2<f32> {
    var nearest = pixel;
    var depth = textureLoad(depth_texture, pixel, 0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let at = clamp(pixel + vec2<i32>(x, y), vec2<i32>(0), last);
            let here = textureLoad(depth_texture, at, 0);
            if here > depth {
                depth = here;
                nearest = at;
            }
        }
    }
    return textureLoad(motion, nearest, 0).xy;
}

// `color`, this frame's at `position` (render pixels of the jittered
// samples, continuous) darkened by its shade, with the reflections there,
// which were traced at render pixels at half resolution: a render pixel's
// are the bilinear read at its center.
fn reflected_at(color: vec3<f32>, position: vec2<f32>) -> vec3<f32> {
    return reflected(color, reflections, history_sampler, position / resolve.render.xy, post);
}

// This frame's color at render pixel `pixel` offset by `offset` (within the
// pixels drawn, `last` the last), darkened by `shade` and with its
// reflections, in YCoCg.
fn neighbor(pixel: vec2<i32>, offset: vec2<i32>, last: vec2<i32>, shade: f32) -> vec3<f32> {
    let at = clamp(pixel + offset, vec2<i32>(0), last);
    return ycocg(reflected_at(textureLoad(scene, at, 0).rgb * shade, vec2<f32>(at) + 0.5));
}

// This frame's color at `position` (render pixels of the jittered samples,
// continuous) between the four samples around it, bilinearly, within the
// pixels drawn (`last` the last).
fn upsampled(position: vec2<f32>, last: vec2<i32>) -> vec3<f32> {
    let base = position - 0.5;
    let first = vec2<i32>(floor(base));
    let f = base - floor(base);
    let a = textureLoad(scene, clamp(first, vec2<i32>(0), last), 0).rgb;
    let b = textureLoad(scene, clamp(first + vec2<i32>(1, 0), vec2<i32>(0), last), 0).rgb;
    let c = textureLoad(scene, clamp(first + vec2<i32>(0, 1), vec2<i32>(0), last), 0).rgb;
    let d = textureLoad(scene, clamp(first + vec2<i32>(1, 1), vec2<i32>(0), last), 0).rgb;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

@fragment
fn fs_resolve(@builtin(position) position: vec4<f32>) -> Outputs {
    // This display pixel's center in the view, and in render pixels; the
    // render pixel whose jittered sample lies nearest it.
    let screen = position.xy * resolve.texel;
    let rendered = screen * resolve.render.xy;
    let jittered = rendered + resolve.render.zw;
    let last = vec2<i32>(resolve.render.xy) - 1;
    let pixel = clamp(vec2<i32>(floor(jittered)), vec2<i32>(0), last);
    let texel = textureLoad(scene, pixel, 0);
    // The neighborhood darkens with this pixel: its box scales alike. Each
    // render pixel takes its own reflections, so the history holds them and
    // the box spans how they vary.
    let shade = effect_shade(texel.a, occlusion_at(occlusion, depth_texture, pixel, post), post);
    let current = reflected_at(texel.rgb * shade, vec2<f32>(pixel) + 0.5);
    // How far that sample lies from this pixel's center, in display pixels:
    // within half of one it counts fully, from a whole one not at all.
    let away = (vec2<f32>(pixel) + 0.5 - resolve.render.zw - rendered)
        / (resolve.render.xy * resolve.texel);
    let near = saturate(2.0 - 2.0 * max(abs(away.x), abs(away.y)));
    var color = current;
    if near < 1.0 {
        color = reflected_at(upsampled(jittered, last) * shade, jittered);
    }
    var offset = textureLoad(motion, pixel, 0).xy;
    if resolve.options.y != 0.0 {
        offset = nearest_motion(pixel, last);
    }
    let uv = screen + offset;
    // NaN motion fails these comparisons too.
    if resolve.history_valid != 0u && all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)) {
        let center = ycocg(current);
        let left = neighbor(pixel, vec2<i32>(-1, 0), last, shade);
        let right = neighbor(pixel, vec2<i32>(1, 0), last, shade);
        let up = neighbor(pixel, vec2<i32>(0, -1), last, shade);
        let down = neighbor(pixel, vec2<i32>(0, 1), last, shade);
        var low = min(center, min(min(left, right), min(up, down)));
        var high = max(center, max(max(left, right), max(up, down)));
        // Rounded: halfway from the cross's range to that of all eight
        // neighbors, which a thin diagonal feature's history stays inside.
        if resolve.options.z != 0.0 {
            let a = neighbor(pixel, vec2<i32>(-1, -1), last, shade);
            let b = neighbor(pixel, vec2<i32>(1, -1), last, shade);
            let c = neighbor(pixel, vec2<i32>(-1, 1), last, shade);
            let d = neighbor(pixel, vec2<i32>(1, 1), last, shade);
            low = 0.5 * (low + min(low, min(min(a, b), min(c, d))));
            high = 0.5 * (high + max(high, max(max(a, b), max(c, d))));
        }
        var past: vec3<f32>;
        if all(abs(offset) < resolve.texel * STILL_PIXELS) {
            past = textureLoad(history, vec2<i32>(position.xy), 0).rgb;
        } else {
            past = sample_history(uv);
        }
        // Particles have no motion of their own: where they cover the
        // pixel, this frame's color weighs at least their coverage times
        // the response.
        var weight = resolve.current_weight * near;
        if resolve.options.w > 0.0 {
            // Frames drawing nothing transparent bind one texel for all.
            let covered = min(pixel, vec2<i32>(textureDimensions(revealed)) - 1);
            let particles = 1.0 - textureLoad(revealed, covered, 0).y;
            weight = max(weight, min(resolve.options.w * particles, 1.0));
        }
        color = mix(rgb(clamp(ycocg(past), low, high)), current, weight);
    }
    color = max(color, vec3<f32>(0.0));
    var output: Outputs;
    output.history = vec4<f32>(color, 1.0);
    // The haze and bloom reach the canvas only.
    let light = haze_at(haze, depth_texture, pixel, post);
    let shown = effect_light(color, light, bloom, history_sampler, screen, post);
    // The UNORM canvas stores encoded color.
    var encoded = encode_srgb(shown);
    // Dithered 8-bit writes turn their noise every frame: the history then
    // averages to the color, where rounding would stop it short by up to
    // half a step over this frame's weight (five steps at 0.1), and the
    // canvas shows no bands.
    if post.sun_direction.w != 0.0 {
        let turn = resolve.options.x;
        output.history = vec4<f32>(decode_srgb(dithered(encode_srgb(color), position.xy, turn,
            1u)), 1.0);
        encoded = dithered(encoded, position.xy, turn, 2u);
    }
    output.canvas = vec4<f32>(encoded, 1.0);
    return output;
}
