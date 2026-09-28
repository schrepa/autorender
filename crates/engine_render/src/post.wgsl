// Screen effects (decisions 021 and 025), prepended with effects.wgsl.
// Ambient occlusion samples the depth buffer at half resolution and filters
// the result;
// screen-space reflections trace reflected rays through it at half
// resolution; bloom filters the scene's glowing light down a chain of
// half-resolution levels and back up; the composite applies them and sun
// shafts (shafts.wgsl) to the scene target and writes the canvas.
// Temporally antialiased frames apply them in their resolve instead.

@group(0) @binding(0) var<uniform> post: Post;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
// The scene target: linear color, and the ambient share of its light in alpha.
@group(0) @binding(2) var scene: texture_2d<f32>;
// Glow weight of each pixel: its exposed brightness above the bloom
// threshold, as x / (1 + x).
@group(0) @binding(3) var glow: texture_2d<f32>;
// The level a bloom pass reads.
@group(0) @binding(4) var source: texture_2d<f32>;
@group(0) @binding(5) var occlusion: texture_2d<f32>;
@group(0) @binding(6) var bloom: texture_2d<f32>;
@group(0) @binding(7) var linear: sampler;
// Reflected color and its weight, and scattered sunlight, at half resolution.
@group(0) @binding(8) var reflections: texture_2d<f32>;
@group(0) @binding(9) var shafts: texture_2d<f32>;
// Reflections (decision 025): the nearest depth of each 2x2 block of the
// depth buffer, then of each block of the level below, which rays skip
// across; and the traced reflections the filter smooths.
@group(0) @binding(10) var nearest_depth: texture_2d<f32>;
@group(0) @binding(11) var traced: texture_2d<f32>;
// Ambient occlusion before its filter: visibility, and the distance from
// the eye (0 where only sky is).
@group(0) @binding(12) var raw_occlusion: texture_2d<f32>;

// sRGB targets apply output encoding themselves; UNORM canvas targets do not.
override SURFACE_IS_SRGB: bool = false;

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
// The largest sampling radius, in full-resolution pixels: close surfaces
// keep their samples in cache.
const MAX_PIXELS: f32 = 96.0;

// World position of a depth-buffer pixel; w is 0 where only sky is (the
// cleared depth, 0 with reversed depth, decision 042).
fn world_at(pixel: vec2<i32>) -> vec4<f32> {
    let depth = textureLoad(depth_texture, pixel, 0);
    let uv = (vec2<f32>(pixel) + 0.5) * post.size.zw;
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = post.inverse_view_projection * clip;
    return vec4<f32>(world.xyz / world.w, select(0.0, 1.0, depth > 0.0));
}

// Scalable ambient obscurance (McGuire et al.) over eight samples: each
// sample closer to the surface's tangent plane than the radius occludes by
// how far above that plane, and how close, it is. Writes the visibility and
// the distance from the eye, which the filter compares.
@fragment
fn fs_occlusion(@builtin(position) position: vec4<f32>) -> @location(0) vec2<f32> {
    let last = vec2<i32>(post.size.xy) - 1;
    let pixel = min(vec2<i32>(position.xy) * 2, last);
    let center = world_at(pixel);
    if center.w == 0.0 {
        return vec2<f32>(1.0, 0.0);
    }
    let p = center.xyz;
    // The normal from the nearer neighbor on each axis keeps silhouettes
    // from bending it.
    let right = world_at(min(pixel + vec2<i32>(1, 0), last)).xyz - p;
    let left = p - world_at(max(pixel - vec2<i32>(1, 0), vec2<i32>(0))).xyz;
    let down = world_at(min(pixel + vec2<i32>(0, 1), last)).xyz - p;
    let up = p - world_at(max(pixel - vec2<i32>(0, 1), vec2<i32>(0))).xyz;
    let across = select(right, left, dot(left, left) < dot(right, right));
    let along = select(down, up, dot(up, up) < dot(down, down));
    let to_eye = post.eye.xyz - p;
    let distance = length(to_eye);
    var normal = cross(along, across);
    let length_squared = dot(normal, normal);
    if !(length_squared > 0.0) {
        return vec2<f32>(1.0, distance);
    }
    normal = normal * inverseSqrt(length_squared);
    if dot(normal, to_eye) < 0.0 {
        normal = -normal;
    }
    let radius = post.occlusion.x;
    let pixels = min(radius * post.eye.w / distance, MAX_PIXELS);
    if pixels < 1.0 {
        return vec2<f32>(1.0, distance);
    }
    // Interleaved gradient noise turns the spiral per pixel. It stays the
    // same every frame: turning it would let the temporal resolve's clamp
    // chase changing darkness, which flickers.
    let noise = fract(52.9829189 * fract(dot(position.xy, vec2<f32>(0.06711056, 0.00583715))));
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
    return vec2<f32>(1.0 - post.occlusion.y * obscurance, distance);
}

// How far, relative to its own, a neighbor's inverse distance may lie from
// the center's plane before the filter stops counting it.
const OCCLUSION_PLANE: f32 = 0.01;

// The inverse distance of a raw occlusion texel: 0 for sky, which lies
// infinitely far.
fn inverse_distance(texel: vec2<i32>) -> f32 {
    let distance = textureLoad(raw_occlusion, texel, 0).y;
    return select(0.0, 1.0 / distance, distance > 0.0);
}

// The occlusion over each half-resolution texel's 5x5 neighbors on its
// surface, weighted by a Gaussian: the eight samples' per-pixel turns average
// out instead of mottling what the ambient light mostly shades (metal in
// shadow). Inverse distance changes linearly across a plane on screen, so a
// neighbor counts while its inverse distance lies near the center's
// extrapolated by the center's slope toward its nearer neighbor on each axis.
@fragment
fn fs_occlusion_filter(@builtin(position) position: vec4<f32>) -> @location(0) f32 {
    let texel = vec2<i32>(position.xy);
    let last = vec2<i32>(textureDimensions(raw_occlusion)) - 1;
    let center = textureLoad(raw_occlusion, texel, 0);
    if !(center.y > 0.0) {
        return center.x;
    }
    let inverse = 1.0 / center.y;
    let right = inverse_distance(min(texel + vec2<i32>(1, 0), last)) - inverse;
    let left = inverse - inverse_distance(max(texel - vec2<i32>(1, 0), vec2<i32>(0)));
    let down = inverse_distance(min(texel + vec2<i32>(0, 1), last)) - inverse;
    let up = inverse - inverse_distance(max(texel - vec2<i32>(0, 1), vec2<i32>(0)));
    let slope = vec2<f32>(
        select(right, left, abs(left) < abs(right)),
        select(down, up, abs(up) < abs(down)),
    );
    let tolerance = OCCLUSION_PLANE * inverse;
    var sum = 0.0;
    var total = 0.0;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let offset = vec2<i32>(x, y);
            let neighbor = clamp(texel + offset, vec2<i32>(0), last);
            let sample = textureLoad(raw_occlusion, neighbor, 0);
            let sample_inverse = select(0.0, 1.0 / sample.y, sample.y > 0.0);
            let expected = inverse + dot(slope, vec2<f32>(offset));
            let plane = saturate(1.25 - abs(sample_inverse - expected) / tolerance);
            let weight = exp(-f32(x * x + y * y) / 8.0) * plane;
            sum += sample.x * weight;
            total += weight;
        }
    }
    return sum / total;
}

// Reflection rays walk at most this many cells of the depth pyramid.
const REFLECTION_STEPS: u32 = 96u;
// A reflected surface counts where the ray passes within this distance of
// it, in world units, plus this share of the ray's length.
const REFLECTION_THICKNESS: f32 = 0.3;
const REFLECTION_THICKNESS_SHARE: f32 = 0.05;
const FLOAT_MAX: f32 = 0x1.fffffep+127f;
// The pixel of each 2x2 block the half-resolution reflection passes trace:
// the first without temporal antialiasing; with it, another each frame
// (`post.reflections.w`), so the resolve's history gathers every block.
const BLOCK_PIXELS = array<vec2<i32>, 4>(
    vec2<i32>(0, 0), vec2<i32>(1, 1), vec2<i32>(1, 0), vec2<i32>(0, 1),
);

// The full-resolution pixel half-resolution texel `texel` traces.
fn traced_pixel(texel: vec2<i32>) -> vec2<i32> {
    let last = vec2<i32>(post.size.xy) - 1;
    return min(texel * 2 + BLOCK_PIXELS[u32(post.reflections.w) & 3u], last);
}

// The nearest depth (the largest, reversed) of cell `cell` of mip `mip`:
// mip 0 is the depth buffer, mip m > 0 the pyramid's level m - 1, whose
// texels cover 2^m pixels a side (the pyramid repeats the buffer's edge).
fn nearest_depth_at(cell: vec2<i32>, mip: i32) -> f32 {
    if mip == 0 {
        let last = vec2<i32>(post.size.xy) - 1;
        return textureLoad(depth_texture, clamp(cell, vec2<i32>(0), last), 0);
    }
    let last = vec2<i32>(textureDimensions(nearest_depth, mip - 1)) - 1;
    return textureLoad(nearest_depth, clamp(cell, vec2<i32>(0), last), mip - 1).r;
}

// Pixel position (rows downward) and depth of world point `world`, and its
// clip w.
fn screen_point(world: vec3<f32>) -> vec4<f32> {
    let clip = post.view_projection * vec4<f32>(world, 1.0);
    let ndc = clip.xyz / clip.w;
    return vec4<f32>((ndc.x * 0.5 + 0.5) * post.size.x, (0.5 - ndc.y * 0.5) * post.size.y,
        ndc.z, clip.w);
}

// The world point at pixel position `pixel` (continuous) and depth `depth`.
fn world_point(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
    let uv = pixel * post.size.zw;
    let world = post.inverse_view_projection
        * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return world.xyz / world.w;
}

// Screen-space reflections (decision 025) at half resolution: from each
// reflective pixel (the glow target's second channel), a ray along the
// mirror direction of its depth normal crosses the screen as a straight
// line in pixels and depth. It walks the pyramid of nearest depths: a cell
// the ray crosses in front of everything is skipped and the next one is
// coarser; a cell it may meet is refined, down to single pixels (FidelityFX
// SSSR's hierarchical march), so every pixel along the ray near a surface is
// visited and thin objects are found alike from pixel to pixel. The first
// pixel the ray passes behind is its hit, reflected where the ray passes
// close to that surface, fading toward the screen edges, the ray's end and
// the surface's thickness; misses weigh nothing.
@fragment
fn fs_reflections(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let last = vec2<i32>(post.size.xy) - 1;
    let pixel = traced_pixel(vec2<i32>(position.xy));
    let reflectivity = textureLoad(glow, pixel, 0).g;
    if reflectivity < 0.01 {
        return vec4<f32>(0.0);
    }
    let center = world_at(pixel);
    if center.w == 0.0 {
        return vec4<f32>(0.0);
    }
    let p = center.xyz;
    let right = world_at(min(pixel + vec2<i32>(1, 0), last)).xyz - p;
    let left = p - world_at(max(pixel - vec2<i32>(1, 0), vec2<i32>(0))).xyz;
    let down = world_at(min(pixel + vec2<i32>(0, 1), last)).xyz - p;
    let up = p - world_at(max(pixel - vec2<i32>(0, 1), vec2<i32>(0))).xyz;
    let across = select(right, left, dot(left, left) < dot(right, right));
    let along = select(down, up, dot(up, up) < dot(down, down));
    var normal = cross(along, across);
    let length_squared = dot(normal, normal);
    if !(length_squared > 0.0) {
        return vec4<f32>(0.0);
    }
    normal = normal * inverseSqrt(length_squared);
    let to_eye = normalize(post.eye.xyz - p);
    if dot(normal, to_eye) < 0.0 {
        normal = -normal;
    }
    let direction = reflect(-to_eye, normal);
    // The ray's segment: `distance` long, ending nine tenths of the way to
    // the eye plane when it heads toward the camera.
    let start = screen_point(p);
    let rows = transpose(post.view_projection);
    let w_rate = dot(rows[3].xyz, direction);
    var reach = post.reflections.x;
    if w_rate < 0.0 {
        reach = min(reach, 0.9 * start.w / -w_rate);
    }
    let end = screen_point(p + direction * reach);
    if !(start.w > 0.0 && end.w > 0.0) {
        return vec4<f32>(0.0);
    }
    let origin = start.xyz;
    let ray = end.xyz - start.xyz;
    let moving = abs(ray.xy) > vec2<f32>(1.0e-9);
    let forward = ray.xy >= vec2<f32>(0.0);
    // Cell edges ahead of the ray, and a little past them so the next cell
    // starts inside.
    let ahead = select(vec2<f32>(0.0), vec2<f32>(1.0), forward);
    let past = select(vec2<f32>(-0.005), vec2<f32>(0.005), forward);
    let top = i32(textureNumLevels(nearest_depth));
    // Leave the starting pixel first: a surface does not reflect itself.
    var t = FLOAT_MAX;
    let exits = select(vec2<f32>(FLOAT_MAX),
        (floor(origin.xy) + ahead + past - origin.xy) / select(vec2<f32>(1.0), ray.xy, moving),
        moving);
    t = min(exits.x, exits.y);
    var point = origin + ray * t;
    var mip = 0;
    for (var step = 0u; step < REFLECTION_STEPS; step++) {
        if mip < 0 || t > 1.0 || any(point.xy < vec2<f32>(0.0))
                || any(point.xy >= post.size.xy) {
            break;
        }
        let size = exp2(f32(mip));
        let cell = floor(point.xy / size);
        let surface = nearest_depth_at(vec2<i32>(cell), mip);
        let edges = select(vec2<f32>(FLOAT_MAX),
            ((cell + ahead) * size + past - origin.xy) / select(vec2<f32>(1.0), ray.xy, moving),
            moving);
        let leave = min(edges.x, edges.y);
        // Reversed depth: the ray meets the cell's nearest depth only while
        // it heads away from the eye.
        var meet = FLOAT_MAX;
        if ray.z < 0.0 {
            meet = (surface - origin.z) / ray.z;
        }
        // In front of everything in the cell: advance to where it leaves the
        // cell (coarser next) or meets its nearest depth (finer next).
        let above = surface < point.z;
        let skipped = above && leave < meet;
        if above {
            t = min(leave, meet);
            point = origin + ray * t;
        }
        if skipped {
            mip = min(mip + 1, top);
        } else {
            mip -= 1;
        }
    }
    if mip >= 0 || t > 1.0 {
        return vec4<f32>(0.0);
    }
    let hit = vec2<i32>(floor(point.xy));
    if any(hit < vec2<i32>(0)) || any(hit > last) || all(abs(point.xy - origin.xy) < vec2<f32>(2.0)) {
        return vec4<f32>(0.0);
    }
    let surface = world_at(hit);
    if surface.w == 0.0 {
        return vec4<f32>(0.0);
    }
    // How far behind that surface the ray passes, against a thickness that
    // grows with the ray's length.
    let ray_point = world_point(point.xy, point.z);
    let travelled = distance(ray_point, p);
    let behind = distance(ray_point, surface.xyz);
    let thickness = REFLECTION_THICKNESS + REFLECTION_THICKNESS_SHARE * travelled;
    var confidence = 1.0 - smoothstep(0.0, thickness, behind);
    confidence *= confidence;
    let uv = point.xy * post.size.zw;
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    let fraction = min(travelled / max(post.reflections.x, 1.0e-6), 1.0);
    let fade = saturate(edge * 10.0) * (1.0 - fraction * fraction) * confidence;
    let color = textureSampleLevel(scene, linear, uv, 0.0).rgb;
    return vec4<f32>(color, reflectivity * fade);
}

// The traced reflections over each half-resolution texel's 3x3 neighbors
// on the same surface (depth within 2% of its own), weighted by a tent and
// by each sample's reflection weight: neighboring rays' hits and misses
// blend instead of speckling. Frames with temporal antialiasing then also
// gather each block's four pixels over frames.
@fragment
fn fs_reflections_filter(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(position.xy);
    let last = vec2<i32>(textureDimensions(traced)) - 1;
    let depth = textureLoad(depth_texture, traced_pixel(texel), 0);
    var sum = vec4<f32>(0.0);
    var total = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbor = clamp(texel + vec2<i32>(x, y), vec2<i32>(0), last);
            let neighbor_depth = textureLoad(depth_texture, traced_pixel(neighbor), 0);
            // Reversed depth is about inversely proportional to distance:
            // relative differences of the two agree.
            let same = abs(neighbor_depth - depth) <= 0.02 * max(depth, neighbor_depth);
            let weight = f32((2 - abs(x)) * (2 - abs(y))) * select(0.0, 1.0, same);
            let sample = textureLoad(traced, neighbor, 0);
            sum += vec4<f32>(sample.rgb * sample.a, sample.a) * weight;
            total += weight;
        }
    }
    if !(sum.a > 0.0) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(sum.rgb / sum.a, sum.a / total);
}

// A pixel's glowing light: its color times its decoded glow.
fn glow_light(uv: vec2<f32>) -> vec3<f32> {
    let color = textureSampleLevel(scene, linear, uv, 0.0).rgb;
    let weight = min(textureSampleLevel(glow, linear, uv, 0.0).r, 0.98);
    return color * (weight / (1.0 - weight));
}

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// The first bloom level, at half resolution: the scene's glowing light over
// each 2x2 block and its diagonal neighbors, each tap weighted by
// 1 / (1 + luminance) so single bright pixels cannot flicker as fireflies.
@fragment
fn fs_bloom_first(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = post.size.zw;
    let uv = position.xy * 2.0 * texel;
    var sum = vec3<f32>(0.0);
    var weights = 0.0;
    for (var i = 0u; i < 5u; i++) {
        var offset = vec2<f32>(0.0);
        if i > 0u {
            offset = vec2<f32>(f32(i & 1u) * 2.0 - 1.0, f32((i >> 1u) & 1u) * 2.0 - 1.0);
        }
        let light = glow_light(uv + offset * texel);
        let weight = select(1.0, 4.0, i == 0u) / (1.0 + luminance(light));
        sum += light * weight;
        weights += weight;
    }
    return vec4<f32>(sum / weights, 1.0);
}

// Half the resolution of `source`: its 2x2 block weighted four times against
// the four diagonal neighbors (a dual filter).
@fragment
fn fs_bloom_down(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(source));
    let uv = position.xy * 2.0 * texel;
    let sum = textureSampleLevel(source, linear, uv, 0.0).rgb * 4.0
        + textureSampleLevel(source, linear, uv - texel, 0.0).rgb
        + textureSampleLevel(source, linear, uv + texel, 0.0).rgb
        + textureSampleLevel(source, linear, uv + vec2<f32>(texel.x, -texel.y), 0.0).rgb
        + textureSampleLevel(source, linear, uv + vec2<f32>(-texel.x, texel.y), 0.0).rgb;
    return vec4<f32>(sum * 0.125, 1.0);
}

// Twice the resolution of `source`, added to the level below: a tent of
// eight taps (the dual filter's upsampling).
@fragment
fn fs_bloom_up(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(source));
    let uv = position.xy * 0.5 * texel;
    let sum = (textureSampleLevel(source, linear, uv + vec2<f32>(-2.0, 0.0) * texel, 0.0).rgb
        + textureSampleLevel(source, linear, uv + vec2<f32>(2.0, 0.0) * texel, 0.0).rgb
        + textureSampleLevel(source, linear, uv + vec2<f32>(0.0, -2.0) * texel, 0.0).rgb
        + textureSampleLevel(source, linear, uv + vec2<f32>(0.0, 2.0) * texel, 0.0).rgb)
        + (textureSampleLevel(source, linear, uv - texel, 0.0).rgb
        + textureSampleLevel(source, linear, uv + texel, 0.0).rgb
        + textureSampleLevel(source, linear, uv + vec2<f32>(texel.x, -texel.y), 0.0).rgb
        + textureSampleLevel(source, linear, uv + vec2<f32>(-texel.x, texel.y), 0.0).rgb) * 2.0;
    return vec4<f32>(sum / 12.0, 1.0);
}

// Frames without temporal antialiasing: the scene with its effects.
@fragment
fn fs_composite(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = textureLoad(scene, vec2<i32>(position.xy), 0);
    let uv = position.xy * post.size.zw;
    let shade = occlusion_factor(texel.a, occlusion, linear, uv, post);
    let mirrored = reflected(texel.rgb * shade, reflections, linear, uv, post);
    let lit = shafted(mirrored, shafts, linear, uv, post);
    let color = glowing(lit, bloom, linear, uv, post);
    if SURFACE_IS_SRGB {
        return vec4<f32>(color, 1.0);
    }
    return vec4<f32>(encode_srgb(color), 1.0);
}
