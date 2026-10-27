// Screen effects, prepended with effects.wgsl and screen_points.wgsl. Every
// one but bloom runs at the render size, each pixel by its own depth,
// normal and ray. Ambient occlusion samples the depth buffer at every
// pixel, and one filter smooths it on each surface; the haze's march
// (haze.wgsl) is filtered over time with the last frame's result
// reprojected; bloom filters the scene's glowing light down a chain of
// levels from half the render size and back up, the passes halving it also
// halving the scene into the levels rough reflections read
// (reflections.wgsl, whose pass binds groups of its own); the composite
// applies them to the scene target, upscaled to the display, and writes the
// canvas. Temporally antialiased frames apply them in their resolve
// instead. Every pass reads within what the frame drew: its render size of
// the screen-sized targets, from their top-left corner, and as much of the
// others.

@group(0) @binding(0) var<uniform> post: Post;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
// The scene target: linear color, and the ambient share of its light in alpha.
@group(0) @binding(2) var scene: texture_2d<f32>;
// Glow weight of each pixel: its exposed brightness above the bloom
// threshold, as x / (1 + x).
@group(0) @binding(3) var glow: texture_2d<f32>;
// The level a bloom pass reads.
@group(0) @binding(4) var source: texture_2d<f32>;
// The filtered ambient occlusion's visibility.
@group(0) @binding(5) var occlusion: texture_2d<f32>;
@group(0) @binding(6) var bloom: texture_2d<f32>;
@group(0) @binding(7) var linear: sampler;
// Reflected color and its weight.
@group(0) @binding(8) var reflections: texture_2d<f32>;
// The haze's march this frame (haze.wgsl): its exposed light, and the
// inverse distance of the surface its ray ended at (w; 0 for sky).
@group(0) @binding(9) var marched: texture_2d<f32>;
// Ambient occlusion before its filter: visibility, and the distance from
// the eye (0 where only sky is).
@group(0) @binding(12) var raw_occlusion: texture_2d<f32>;
// The bloom level a pass halving both chains reads beside the scene level
// in `source`.
@group(0) @binding(15) var bloom_source: texture_2d<f32>;

// Sky occlusion's maps (`shadows::sky`): still meshes seen from fixed
// directions of the sky, one layer each near the camera, then one each
// reaching farther; their comparison sampler; and where each looks.
const SKY_DIRECTIONS: u32 = 13u;
const SKY_LAYERS: u32 = 26u;
struct Sky {
    // World to each map's clip space.
    clips: array<mat4x4<f32>, SKY_LAYERS>,
    // Each map's unit direction toward the sky, and (w) how much it counts:
    // 0 until drawn, fading in to 1 over its first frames.
    directions: array<vec4<f32>, SKY_LAYERS>,
    // Directions in use, the near and the far maps' texels in world units,
    // and their reach over their distances.
    counts: vec4<f32>,
}
@group(0) @binding(16) var sky_maps: texture_depth_2d_array;
@group(0) @binding(17) var sky_sampler: sampler_comparison;
@group(0) @binding(18) var<uniform> sky: Sky;
// The haze filtered over time, as marched: the last frame's, the history
// its filter reprojects, or this frame's, which the composite lays over the
// scene.
@group(0) @binding(19) var haze: texture_2d<f32>;

const TAU: f32 = 6.2831853;
const SAMPLES: u32 = 8u;
// Unit directions of the occlusion samples: one per eighth of a turn, in the
// order of a spiral of seven turns, each at a larger fraction of the radius.
const SPIRAL: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
    vec2<f32>(-0.9238795, 0.38268343), vec2<f32>(-0.38268343, 0.9238795),
    vec2<f32>(0.38268343, 0.9238795), vec2<f32>(0.9238795, 0.38268343),
    vec2<f32>(0.9238795, -0.38268343), vec2<f32>(0.38268343, -0.9238795),
    vec2<f32>(-0.38268343, -0.9238795), vec2<f32>(-0.9238795, -0.38268343),
);
// The largest sampling radius, in pixels: close surfaces keep their
// samples in cache.
const MAX_PIXELS: f32 = 96.0;

// The world radius surfaces occlude within: the same at every distance, so
// a place darkens alike wherever the eye stands.
fn occlusion_radius() -> f32 {
    return post.occlusion.x;
}

// How much of its occlusion a surface takes whose occlusion radius spans
// `pixels` pixels: only what the terrains' probes and the occupancy maps
// are too coarse to hold. The radius hands over as a level finer than the
// finest occupancy map would: all of it while it spans twice the pixels
// that map's texel spans at the edge of its reach (`post.occlusion.z`
// times the pixels a meter spans a meter away), none once it spans no
// more, over the octave between, as the maps' levels hand over
// (`scene.wgsl`'s `detail_level`).
fn occlusion_share(pixels: f32) -> f32 {
    let handover = post.occlusion.z * post.eye.w;
    return saturate(log2(max(pixels, 1.0e-6) / max(handover, 1.0e-6)));
}

// Scalable ambient obscurance (McGuire et al.) over eight samples: each
// sample closer to the surface's tangent plane than the radius occludes by
// how far above that plane, and how close, it is, at the share of its
// strength the radius's span on screen leaves (`occlusion_share`), seen from
// the detail eye; 1 without occlusion.
fn ambient_visibility(pixel: vec2<i32>, p: vec3<f32>, normal: vec3<f32>, distance: f32,
        noise: f32) -> f32 {
    if post.occlusion.w == 0.0 {
        return 1.0;
    }
    let last = vec2<i32>(post.size.xy) - 1;
    let radius = occlusion_radius();
    let seen = radius * post.eye.w / max(length(p - post.detail.xyz), 1.0e-4);
    let strength = post.occlusion.y * occlusion_share(seen);
    let pixels = min(radius * post.eye.w / distance, MAX_PIXELS);
    if strength <= 0.0 || pixels < 1.0 {
        return 1.0;
    }
    let turn = vec2<f32>(cos(noise * TAU), sin(noise * TAU));
    let radius2 = radius * radius;
    // Keeps flat surfaces from occluding themselves as depth precision falls.
    let bias = 0.002 * distance;
    var sum = 0.0;
    for (var i = 0u; i < SAMPLES; i++) {
        let fraction = (f32(i) + 0.5) / f32(SAMPLES);
        let d = SPIRAL[i];
        let offset = vec2<f32>(d.x * turn.x - d.y * turn.y, d.x * turn.y + d.y * turn.x)
            * (fraction * pixels);
        let probe = world_at(clamp(pixel + vec2<i32>(round(offset)), vec2<i32>(0), last));
        let v = probe.xyz - p;
        let vv = dot(v, v);
        let falloff = max(radius2 - vv, 0.0);
        sum += probe.w * falloff * falloff * falloff
            * max((dot(v, normal) - bias) / (vv + 0.01 * radius2), 0.0);
    }
    let radius6 = radius2 * radius2 * radius2;
    let obscurance = clamp(sum * (5.0 / f32(SAMPLES)) / radius6, 0.0, 1.0);
    return 1.0 - strength * obscurance;
}

// How much of map `layer` sees the sky along its direction from `p`, a
// surface point facing `normal`, as (seen, counted): seen the share of the
// 2x2 texels around it no caster lies above, lifted a texel toward the sky
// and off the surface so its own surface hides none of it; counted 0 where
// the map does not reach or was never drawn, fading in from its edges.
fn sky_seen(layer: u32, p: vec3<f32>, normal: vec3<f32>, texel: f32) -> vec2<f32> {
    let toward = sky.directions[layer];
    if toward.w <= 0.0 {
        return vec2<f32>(1.0, 0.0);
    }
    let lifted = p + normal * (1.5 * texel) + toward.xyz * texel;
    let clip = sky.clips[layer] * vec4<f32>(lifted, 1.0);
    let edge = max(abs(clip.x), abs(clip.y));
    let counted = toward.w * (1.0 - smoothstep(0.85, 1.0, edge)) * select(1.0, 0.0, clip.z >= 1.0);
    if counted <= 0.0 {
        return vec2<f32>(1.0, 0.0);
    }
    let uv = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
    let seen = textureSampleCompareLevel(sky_maps, sky_sampler, uv, layer, clip.z);
    return vec2<f32>(seen, counted);
}

// The share of the sky's light that reaches `p`, a surface point facing
// `normal`: each direction in use weighted by how squarely the surface faces
// it, seen through the near map where it reaches, the far map beyond and
// open sky past both; at least sky occlusion's floor.
fn sky_visibility(p: vec3<f32>, normal: vec3<f32>) -> f32 {
    let count = u32(sky.counts.x);
    if post.sky.w == 0.0 || count == 0u {
        return 1.0;
    }
    var sum = 0.0;
    var total = 0.0;
    for (var direction = 0u; direction < count; direction++) {
        let weight = max(dot(normal, sky.directions[direction].xyz), 0.0) + 0.05;
        let near = sky_seen(direction, p, normal, sky.counts.y);
        var seen = near.x;
        if near.y < 1.0 {
            let far = sky_seen(direction + SKY_DIRECTIONS, p, normal, sky.counts.z);
            seen = mix(mix(1.0, far.x, far.y), near.x, near.y);
        }
        sum += weight * seen;
        total += weight;
    }
    let visible = max(sum / total, post.sky.y);
    return 1.0 - post.sky.x * (1.0 - visible);
}

// Ambient occlusion at every pixel: writes the occlusion's visibility (with
// what sky occlusion leaves of the sky's light) and the distance from the
// eye, which the filter compares.
@fragment
fn fs_occlusion(@builtin(position) position: vec4<f32>) -> @location(0) vec2<f32> {
    let pixel = vec2<i32>(position.xy);
    let center = world_at(pixel);
    if center.w == 0.0 {
        return vec2<f32>(1.0, 0.0);
    }
    let p = center.xyz;
    let distance = length(post.eye.xyz - p);
    let normal = depth_normal(pixel, p);
    if all(normal == vec3<f32>(0.0)) {
        return vec2<f32>(1.0, distance);
    }
    // Interleaved gradient noise turns the spiral per pixel. It stays the
    // same every frame: changing it would let the temporal resolve's clamp
    // chase changing darkness, which flickers.
    let noise = fract(52.9829189 * fract(dot(position.xy, vec2<f32>(0.06711056, 0.00583715))));
    var visible = sky_visibility(p, normal);
    if post.occlusion.w != 0.0 {
        visible *= ambient_visibility(pixel, p, normal, distance, noise);
    }
    return vec2<f32>(visible, distance);
}

// How far, relative to its own, a neighbor's inverse distance may lie from
// the center's plane before the filter stops counting it.
const OCCLUSION_PLANE: f32 = 0.01;

// The inverse distance of raw occlusion `sample`: 0 for sky, which lies
// infinitely far.
fn inverse_distance(sample: vec4<f32>) -> f32 {
    return select(0.0, 1.0 / sample.y, sample.y > 0.0);
}

// The occlusion over each pixel's 5x5 neighbors on its surface, weighted
// by a Gaussian: the eight samples' per-pixel turns average out instead of
// mottling what they shade. Inverse distance changes linearly across a
// plane on screen, so a neighbor counts while its inverse distance lies
// near the center's extrapolated by the center's slope toward its nearer
// neighbor on each axis. Each neighbor is read once, the slope's too.
fn filtered_occlusion(pixel: vec2<i32>) -> f32 {
    let last = vec2<i32>(post.size.xy) - 1;
    let center = textureLoad(raw_occlusion, pixel, 0);
    if !(center.y > 0.0) {
        return center.x;
    }
    // Row by row from (-2, -2); the center is the thirteenth.
    var samples: array<vec4<f32>, 25>;
    var inverses: array<f32, 25>;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let index = (y + 2) * 5 + x + 2;
            var sample = center;
            if x != 0 || y != 0 {
                sample = textureLoad(raw_occlusion, clamp(pixel + vec2<i32>(x, y), vec2<i32>(0),
                    last), 0);
            }
            samples[index] = sample;
            inverses[index] = inverse_distance(sample);
        }
    }
    let inverse = inverses[12];
    let right = inverses[13] - inverse;
    let left = inverse - inverses[11];
    let down = inverses[17] - inverse;
    let up = inverse - inverses[7];
    let slope = vec2<f32>(
        select(right, left, abs(left) < abs(right)),
        select(down, up, abs(up) < abs(down)),
    );
    let tolerance = OCCLUSION_PLANE * inverse;
    var sum = 0.0;
    var total = 0.0;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let index = (y + 2) * 5 + x + 2;
            let offset = vec2<i32>(x, y);
            let expected = inverse + dot(slope, vec2<f32>(offset));
            let plane = saturate(1.25 - abs(inverses[index] - expected) / tolerance);
            let weight = exp(-f32(x * x + y * y) / 8.0) * plane;
            sum += samples[index].x * weight;
            total += weight;
        }
    }
    return sum / total;
}

// The occlusion filtered at each pixel, which the composite and the
// resolve read at the same pixel.
@fragment
fn fs_effects_filter(@builtin(position) position: vec4<f32>) -> @location(0) f32 {
    return filtered_occlusion(vec2<i32>(position.xy));
}

// The weight of this frame's march against the haze's reprojected history:
// an exponential average over about ten frames, as temporal antialiasing's,
// over which the march's jitter turns through its steps.
const HAZE_CURRENT: f32 = 0.1;

// How near a neighbor's surface, or the history's, must lie to a pixel's,
// as a share of the pixel's inverse distance from the eye, for the haze's
// filter to count it.
const HAZE_PLANE: f32 = 0.1;

// Last frame's filtered haze where it showed the point the depth buffer
// holds at `pixel`, whose inverse distance from the eye is `own` (0 for
// sky): the four history pixels around the point's place in last frame's
// view, weighted bilinearly and by how near the distance of their surface
// from last frame's eye lies to the point's, so history from across a
// depth discontinuity is left out. w is the weight left: 0 where none is,
// where the point lay off last frame's view, or where no history precedes
// this frame.
fn reprojected_haze(pixel: vec2<i32>, own: f32) -> vec4<f32> {
    let drawn = post.haze.yz;
    if drawn.x <= 0.0 {
        return vec4<f32>(0.0);
    }
    let point = world_at(pixel);
    let clip = post.previous_view_projection * vec4<f32>(point.xyz, 1.0);
    if !(clip.w > 0.0) {
        return vec4<f32>(0.0);
    }
    let ndc = clip.xy / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return vec4<f32>(0.0);
    }
    // What a history texel on the point's surface holds: its inverse
    // distance from last frame's eye.
    var expected = 0.0;
    if own > 0.0 {
        expected = 1.0 / max(length(point.xyz - post.previous_eye.xyz), 1.0e-4);
    }
    let tolerance = max(HAZE_PLANE * expected, 1.0e-7);
    // Last frame's pixels, at their centers, within those it drew.
    let at = uv * drawn - 0.5;
    let first = vec2<i32>(floor(at));
    let f = at - floor(at);
    let last = vec2<i32>(drawn) - 1;
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            let tap = textureLoad(haze, clamp(first + vec2<i32>(x, y), vec2<i32>(0), last), 0);
            let bilinear = select(1.0 - f.x, f.x, x == 1) * select(1.0 - f.y, f.y, y == 1);
            let weight = bilinear * saturate(1.25 - abs(tap.w - expected) / tolerance);
            sum += tap.rgb * weight;
            total += weight;
        }
    }
    if total < 1.0e-3 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(sum / total, total);
}

// The haze filtered over time at pixel `position`: this frame's march there
// blended into the history reprojected to it by the camera's motion
// (`reprojected_haze`), the history clamped to the box of this frame's
// march over the 3x3 pixels around, HAZE_CURRENT of the new each frame.
// Where no history reaches the pixel (off last frame's view, across a depth
// discontinuity, or in a frame no history precedes, as exact frames are)
// it takes this frame's march alone, averaged over the 3x3 pixels around on
// its surface. w keeps the inverse distance of the pixel's surface.
@fragment
fn fs_haze_filter(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let last = vec2<i32>(post.size.xy) - 1;
    let center = textureLoad(marched, pixel, 0);
    let own = center.w;
    let tolerance = max(HAZE_PLANE * own, 1.0e-7);
    var low = center.rgb;
    var high = center.rgb;
    var surface = center.rgb;
    var count = 1.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            if x == 0 && y == 0 {
                continue;
            }
            let neighbor = textureLoad(marched,
                clamp(pixel + vec2<i32>(x, y), vec2<i32>(0), last), 0);
            low = min(low, neighbor.rgb);
            high = max(high, neighbor.rgb);
            let same = saturate(1.25 - abs(neighbor.w - own) / tolerance);
            surface += neighbor.rgb * same;
            count += same;
        }
    }
    let history = reprojected_haze(pixel, own);
    if history.w <= 0.0 {
        return vec4<f32>(surface / count, own);
    }
    return vec4<f32>(mix(clamp(history.rgb, low, high), center.rgb, HAZE_CURRENT), own);
}

// `level`'s color at `at` (its texels, continuous; `size` of them, `drawn`
// of them drawn), bilinearly, kept half a texel within the drawn ones.
fn level_tap(level: texture_2d<f32>, at: vec2<f32>, drawn: vec2<f32>, size: vec2<f32>)
        -> vec3<f32> {
    return textureSampleLevel(level, linear, clamp(at, vec2<f32>(0.5), drawn - 0.5) / size,
        0.0).rgb;
}

// A pixel's glowing light: its color times its decoded glow.
fn glow_light(uv: vec2<f32>) -> vec3<f32> {
    let color = textureSampleLevel(scene, linear, uv, 0.0).rgb;
    let weight = min(textureSampleLevel(glow, linear, uv, 0.0).r, 0.98);
    return color * (weight / (1.0 - weight));
}

// The first bloom level, half the render size, as a blur needs no more:
// the scene's glowing light over each 2x2 block and its diagonal
// neighbors, each tap weighted by 1 / (1 + luminance) so single bright
// pixels cannot flicker as fireflies.
@fragment
fn fs_bloom_first(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let center = position.xy * 2.0;
    var sum = vec3<f32>(0.0);
    var weights = 0.0;
    for (var i = 0u; i < 5u; i++) {
        var offset = vec2<f32>(0.0);
        if i > 0u {
            offset = vec2<f32>(f32(i & 1u) * 2.0 - 1.0, f32((i >> 1u) & 1u) * 2.0 - 1.0);
        }
        let light = glow_light(drawn_uv(center + offset, post));
        let weight = select(1.0, 4.0, i == 0u) / (1.0 + luminance(light));
        sum += light * weight;
        weights += weight;
    }
    return vec4<f32>(sum / weights, 1.0);
}

// Half the resolution of `level` at `position`: its 2x2 block weighted four
// times against the four diagonal neighbors (a dual filter), within the
// texels the frame drew of it.
fn halved(level: texture_2d<f32>, position: vec2<f32>) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(level));
    let drawn = drawn_texels(size, post);
    let at = position * 2.0;
    let sum = level_tap(level, at, drawn, size) * 4.0
        + level_tap(level, at - 1.0, drawn, size)
        + level_tap(level, at + 1.0, drawn, size)
        + level_tap(level, at + vec2<f32>(1.0, -1.0), drawn, size)
        + level_tap(level, at + vec2<f32>(-1.0, 1.0), drawn, size);
    return vec4<f32>(sum * 0.125, 1.0);
}

@fragment
fn fs_bloom_down(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return halved(source, position.xy);
}

// Frames that both reflect and bloom halve the scene into its blurred
// levels for rough reflections and the bloom's glowing light in the same
// passes, the two chains' levels sharing their sizes.
struct Levels {
    @location(0) scene: vec4<f32>,
    @location(1) bloom: vec4<f32>,
}

// The scene's first blurred level and bloom's first from the same five taps
// of the scene: fs_bloom_down's of its color and fs_bloom_first's of its
// glowing light.
@fragment
fn fs_levels_first(@builtin(position) position: vec4<f32>) -> Levels {
    let center = position.xy * 2.0;
    var colors: array<vec3<f32>, 5>;
    var sum = vec3<f32>(0.0);
    var weights = 0.0;
    for (var i = 0u; i < 5u; i++) {
        var offset = vec2<f32>(0.0);
        if i > 0u {
            offset = vec2<f32>(f32(i & 1u) * 2.0 - 1.0, f32((i >> 1u) & 1u) * 2.0 - 1.0);
        }
        let at = drawn_uv(center + offset, post);
        let color = textureSampleLevel(scene, linear, at, 0.0).rgb;
        let decoded = min(textureSampleLevel(glow, linear, at, 0.0).r, 0.98);
        let light = color * (decoded / (1.0 - decoded));
        let weight = select(1.0, 4.0, i == 0u) / (1.0 + luminance(light));
        sum += light * weight;
        weights += weight;
        colors[i] = color;
    }
    var levels: Levels;
    // `halved`'s order: offsets (-1, -1), (1, 1), (1, -1) and (-1, 1).
    levels.scene = vec4<f32>(
        (colors[0] * 4.0 + colors[4] + colors[3] + colors[1] + colors[2]) * 0.125, 1.0);
    levels.bloom = vec4<f32>(sum / weights, 1.0);
    return levels;
}

// Both chains' next levels, which share their size.
@fragment
fn fs_levels_down(@builtin(position) position: vec4<f32>) -> Levels {
    var levels: Levels;
    levels.scene = halved(source, position.xy);
    levels.bloom = halved(bloom_source, position.xy);
    return levels;
}

// Twice the resolution of `source`, added to the level below: a tent of
// eight taps (the dual filter's upsampling), within the texels the frame
// drew of it.
@fragment
fn fs_bloom_up(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(source));
    let drawn = drawn_texels(size, post);
    let at = position.xy * 0.5;
    let sum = (level_tap(source, at + vec2<f32>(-2.0, 0.0), drawn, size)
        + level_tap(source, at + vec2<f32>(2.0, 0.0), drawn, size)
        + level_tap(source, at + vec2<f32>(0.0, -2.0), drawn, size)
        + level_tap(source, at + vec2<f32>(0.0, 2.0), drawn, size))
        + (level_tap(source, at - 1.0, drawn, size)
        + level_tap(source, at + 1.0, drawn, size)
        + level_tap(source, at + vec2<f32>(1.0, -1.0), drawn, size)
        + level_tap(source, at + vec2<f32>(-1.0, 1.0), drawn, size)) * 2.0;
    return vec4<f32>(sum / 12.0, 1.0);
}

// The scene at `pixel` (render pixels, continuous) as the canvas shows it
// upscaled: Catmull-Rom over the 4x4 pixels around it, its middle pairs on
// each axis merged into one bilinear fetch and its corners dropped (as the
// temporal resolve reads its history), within the pixels the frame drew.
fn upscaled_scene(pixel: vec2<f32>) -> vec4<f32> {
    let center = floor(pixel - 0.5) + 0.5;
    let f = pixel - center;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let middle = center + w2 / w12;
    let before = center - 1.0;
    let after = center + 2.0;
    let weights = vec4<f32>(w12.x * w0.y, w0.x * w12.y, w3.x * w12.y, w12.x * w3.y);
    let middle_weight = w12.x * w12.y;
    let sum = middle_weight + weights.x + weights.y + weights.z + weights.w;
    let color = scene_tap(middle) * middle_weight
        + scene_tap(vec2<f32>(middle.x, before.y)) * weights.x
        + scene_tap(vec2<f32>(before.x, middle.y)) * weights.y
        + scene_tap(vec2<f32>(after.x, middle.y)) * weights.z
        + scene_tap(vec2<f32>(middle.x, after.y)) * weights.w;
    // Catmull-Rom's negative lobes overshoot at edges.
    return clamp(color / sum, vec4<f32>(0.0), vec4<f32>(1.0));
}

// The scene at `pixel` (render pixels, continuous), bilinearly, within the
// pixels the frame drew.
fn scene_tap(pixel: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(scene, linear, drawn_uv(pixel, post), 0.0);
}

// Frames without temporal antialiasing: the scene upscaled to the canvas,
// with its effects.
@fragment
fn fs_composite(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    // This canvas pixel's center in the view, and in the pixels the frame
    // drew: their own where it drew the whole display, whose color it reads
    // as it is.
    let uv = position.xy * post.extent.zw;
    let whole = all(post.size.xy == post.extent.xy);
    let rendered = select(uv * post.size.xy, position.xy, whole);
    let pixel = min(vec2<i32>(rendered), vec2<i32>(post.size.xy) - 1);
    var texel = textureLoad(scene, pixel, 0);
    if !whole {
        texel = upscaled_scene(rendered);
    }
    let visible = occlusion_at(occlusion, pixel, post);
    let shaded = texel.rgb * effect_shade(texel.a, visible, post);
    let mirrored = reflected(shaded, reflections, linear, uv, post);
    let color = effect_light(mirrored, haze_at(haze, pixel, post), bloom, linear, uv, post);
    // The UNORM canvas stores encoded color, dithered when 8-bit writes are.
    var encoded = encode_srgb(color);
    if post.sun_direction.w != 0.0 {
        encoded = dithered(encoded, position.xy, 0.0, 1u);
    }
    return vec4<f32>(encoded, 1.0);
}
