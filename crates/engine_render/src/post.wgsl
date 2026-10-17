// Screen effects, prepended with effects.wgsl.
// Ambient occlusion samples the depth buffer at half resolution, and one
// half-resolution filter smooths it and brings the quarter-resolution sun
// shafts (shafts.wgsl) up beside it; screen-space reflections trace
// reflected rays through it at half resolution; bloom filters the scene's
// glowing light down a chain of half-resolution levels and back up; the
// composite applies them to the scene target and writes the canvas.
// Temporally antialiased frames apply them in their resolve instead.

// The level a bloom pass reads.
@group(0) @binding(4) var source: texture_2d<f32>;
// The filtered ambient occlusion's visibility and the sun shafts' light at
// half resolution.
@group(0) @binding(5) var occlusion: texture_2d<f32>;
@group(0) @binding(6) var bloom: texture_2d<f32>;
// Reflected color and its weight at half resolution.
@group(0) @binding(8) var reflections: texture_2d<f32>;
// Sun shafts at quarter resolution: scattered light, and the
// length of the ray that gathered it.
@group(0) @binding(9) var shafts: texture_2d<f32>;
// The traced reflections the filter smooths.
@group(0) @binding(11) var traced: texture_2d<f32>;
// Ambient occlusion before its filter: visibility, and the distance from
// the eye (0 where only sky is).
@group(0) @binding(12) var raw_occlusion: texture_2d<f32>;
// The bloom level a pass halving both chains reads beside the scene level
// in `source`.
@group(0) @binding(15) var bloom_source: texture_2d<f32>;

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

// Steps along each contact shadow ray.
const CONTACT_STEPS: u32 = 12u;

// How much of the sun reaches `p` (with unit `normal`, `away` meters from
// the eye) past what the depth buffer holds within `post.contact.x` meters
// toward the sun: a surface counts where the ray passes
// behind it by less than a thickness growing along the ray, so thin things
// shade and the fronts of distant ones do not. Hits nearer the ray's start
// shade more, so its reach leaves no edge, and they shade less over the
// last third of the distance surfaces trace within, so its
// end leaves none either.
fn contact_visibility(p: vec3<f32>, normal: vec3<f32>, away: f32, noise: f32) -> f32 {
    let toward = post.sun_direction.xyz;
    let strength = post.contact.y * (1.0 - smoothstep(0.67 * post.contact.z, post.contact.z,
        away));
    if post.contact.w == 0.0 || strength <= 0.0 || dot(normal, toward) <= 0.0 {
        return 1.0;
    }
    let reach = post.contact.x;
    let last = vec2<i32>(post.size.xy) - 1;
    // Off the surface by what depth precision there needs.
    let start = p + normal * (0.01 + 0.001 * away);
    for (var i = 0u; i < CONTACT_STEPS; i++) {
        let along = (f32(i) + noise) / f32(CONTACT_STEPS) * reach;
        let q = start + toward * along;
        let clip = post.view_projection * vec4<f32>(q, 1.0);
        if clip.w <= 0.0 {
            break;
        }
        let uv = vec2<f32>(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
        if any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) {
            break;
        }
        let surface = world_at(min(vec2<i32>(uv * post.size.xy), last));
        if surface.w == 0.0 {
            continue;
        }
        let behind = length(q - post.eye.xyz) - length(surface.xyz - post.eye.xyz);
        if behind > 0.002 * away && behind < 0.1 + along {
            return 1.0 - strength * (1.0 - along / reach);
        }
    }
    return 1.0;
}

// Scalable ambient obscurance (McGuire et al.) over eight samples: each
// sample closer to the surface's tangent plane than the radius occludes by
// how far above that plane, and how close, it is; 1 without occlusion.
fn ambient_visibility(pixel: vec2<i32>, p: vec3<f32>, normal: vec3<f32>, distance: f32,
        noise: f32) -> f32 {
    if post.occlusion.w == 0.0 {
        return 1.0;
    }
    let last = vec2<i32>(post.size.xy) - 1;
    let radius = post.occlusion.x;
    let pixels = min(radius * post.eye.w / distance, MAX_PIXELS);
    if pixels < 1.0 {
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
    return 1.0 - post.occlusion.y * obscurance;
}

// Ambient occlusion and contact shadows at half resolution: writes the
// occlusion's visibility, the distance from the eye, which the filter
// compares, and the contact shadows' visibility.
@fragment
fn fs_occlusion(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let last = vec2<i32>(post.size.xy) - 1;
    let pixel = min(vec2<i32>(position.xy) * 2, last);
    let center = world_at(pixel);
    if center.w == 0.0 {
        return vec4<f32>(1.0, 0.0, 1.0, 0.0);
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
        return vec4<f32>(1.0, distance, 1.0, 0.0);
    }
    normal = normal * inverseSqrt(length_squared);
    if dot(normal, to_eye) < 0.0 {
        normal = -normal;
    }
    // Interleaved gradient noise turns the spiral and offsets the contact
    // rays' steps per pixel. It stays the same every frame: changing it
    // would let the temporal resolve's clamp chase changing darkness, which
    // flickers.
    let noise = fract(52.9829189 * fract(dot(position.xy, vec2<f32>(0.06711056, 0.00583715))));
    return vec4<f32>(ambient_visibility(pixel, p, normal, distance, noise), distance,
        contact_visibility(p, normal, distance, noise), 0.0);
}

// How far, relative to its own, a neighbor's inverse distance may lie from
// the center's plane before the filter stops counting it.
const OCCLUSION_PLANE: f32 = 0.01;

// The inverse distance of raw occlusion `sample`: 0 for sky, which lies
// infinitely far.
fn inverse_distance(sample: vec4<f32>) -> f32 {
    return select(0.0, 1.0 / sample.y, sample.y > 0.0);
}

// The occlusion and contact shadows (x and y) over each half-resolution
// texel's 5x5 neighbors on its surface, weighted by a Gaussian: the eight
// samples' per-pixel turns and the rays' offsets average out instead of
// mottling what they shade. Inverse distance changes linearly across a
// plane on screen, so a neighbor counts while its inverse distance lies near
// the center's extrapolated by the center's slope toward its nearer
// neighbor on each axis. Each neighbor is read once, the slope's too.
fn filtered_occlusion(texel: vec2<i32>) -> vec2<f32> {
    let last = vec2<i32>(textureDimensions(raw_occlusion)) - 1;
    let center = textureLoad(raw_occlusion, texel, 0);
    if !(center.y > 0.0) {
        return center.xz;
    }
    // Row by row from (-2, -2); the center is the thirteenth.
    var samples: array<vec4<f32>, 25>;
    var inverses: array<f32, 25>;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let index = (y + 2) * 5 + x + 2;
            var sample = center;
            if x != 0 || y != 0 {
                sample = textureLoad(raw_occlusion, clamp(texel + vec2<i32>(x, y), vec2<i32>(0),
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
    var sum = vec2<f32>(0.0);
    var total = 0.0;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let index = (y + 2) * 5 + x + 2;
            let offset = vec2<i32>(x, y);
            let expected = inverse + dot(slope, vec2<f32>(offset));
            let plane = saturate(1.25 - abs(inverses[index] - expected) / tolerance);
            let weight = exp(-f32(x * x + y * y) / 8.0) * plane;
            sum += samples[index].xz * weight;
            total += weight;
        }
    }
    return sum / total;
}

// How far a quarter-resolution shaft ray's length may lie from a texel's
// own, as a share of it, before the filter stops counting it.
const SHAFT_REACH: f32 = 0.1;

// The sun shafts at half-resolution texel `texel`: the four
// quarter-resolution rays around the pixel it stands for, weighted
// bilinearly and by how near their lengths are to that pixel's ray, so light
// gathered past a silhouette does not spill onto the surface in front of it.
// Where no ray is near, the bilinear weights alone remain.
fn upsampled_shafts(texel: vec2<i32>) -> f32 {
    let last = vec2<i32>(post.size.xy) - 1;
    let pixel = min(texel * 2, last);
    let depth = textureLoad(depth_texture, pixel, 0);
    let uv = (vec2<f32>(pixel) + 0.5) * post.size.zw;
    let far_point = post.inverse_view_projection
        * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let reach = min(length(far_point.xyz / far_point.w - post.eye.xyz), post.shafts.z);
    // Quarter-resolution texels in shaft_pixel's centers.
    let position = (vec2<f32>(pixel) - 1.0) * 0.25;
    let first = vec2<i32>(floor(position));
    let f = position - floor(position);
    let shaft_last = vec2<i32>(textureDimensions(shafts)) - 1;
    let tolerance = SHAFT_REACH * max(reach, 1.0e-3);
    var sum = 0.0;
    var total = 0.0;
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            let sample = textureLoad(shafts,
                clamp(first + vec2<i32>(x, y), vec2<i32>(0), shaft_last), 0);
            let bilinear = select(1.0 - f.x, f.x, x == 1) * select(1.0 - f.y, f.y, y == 1);
            let near = saturate(1.25 - abs(sample.y - reach) / tolerance);
            let weight = bilinear * (near + 1.0e-3);
            sum += sample.x * weight;
            total += weight;
        }
    }
    return sum / max(total, 1.0e-8);
}

// One half-resolution pass for the screen effects that need smoothing: the
// filtered occlusion (x; 1 without occlusion), the sun shafts brought up
// from quarter resolution (y; 0 without shafts) and the filtered contact
// shadows (z; 1 without them).
@fragment
fn fs_effects_filter(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(position.xy);
    var visible = vec2<f32>(1.0);
    if post.occlusion.w != 0.0 || post.contact.w != 0.0 {
        visible = filtered_occlusion(texel);
    }
    var shafts_light = 0.0;
    if post.shafts.w != 0.0 {
        shafts_light = upsampled_shafts(texel);
    }
    return vec4<f32>(visible.x, shafts_light, visible.y, 0.0);
}

@fragment
fn fs_reflections(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return reflection_at(vec2<i32>(position.xy));
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
            // The center's depth is already read.
            var neighbor_depth = depth;
            if x != 0 || y != 0 {
                neighbor_depth = textureLoad(depth_texture, traced_pixel(neighbor), 0);
            }
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

// Half the resolution of `level` at `position`: its 2x2 block weighted four
// times against the four diagonal neighbors (a dual filter).
fn halved(level: texture_2d<f32>, position: vec2<f32>) -> vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(level));
    let uv = position * 2.0 * texel;
    let sum = textureSampleLevel(level, linear, uv, 0.0).rgb * 4.0
        + textureSampleLevel(level, linear, uv - texel, 0.0).rgb
        + textureSampleLevel(level, linear, uv + texel, 0.0).rgb
        + textureSampleLevel(level, linear, uv + vec2<f32>(texel.x, -texel.y), 0.0).rgb
        + textureSampleLevel(level, linear, uv + vec2<f32>(-texel.x, texel.y), 0.0).rgb;
    return vec4<f32>(sum * 0.125, 1.0);
}

@fragment
fn fs_bloom_down(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return halved(source, position.xy);
}

// Frames that both reflect and bloom halve the scene for rough reflections
// and the bloom's glowing light in the same passes.
struct Levels {
    @location(0) scene: vec4<f32>,
    @location(1) bloom: vec4<f32>,
}

// The first level of both chains from the same five taps of the scene:
// fs_bloom_down's of its color and fs_bloom_first's of its glowing light.
@fragment
fn fs_levels_first(@builtin(position) position: vec4<f32>) -> Levels {
    let texel = post.size.zw;
    let uv = position.xy * 2.0 * texel;
    var colors: array<vec3<f32>, 5>;
    var sum = vec3<f32>(0.0);
    var weights = 0.0;
    for (var i = 0u; i < 5u; i++) {
        var offset = vec2<f32>(0.0);
        if i > 0u {
            offset = vec2<f32>(f32(i & 1u) * 2.0 - 1.0, f32((i >> 1u) & 1u) * 2.0 - 1.0);
        }
        let at = uv + offset * texel;
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
    let effects = screen_effects(occlusion, linear, uv, post);
    let shaded = texel.rgb * effect_shade(texel.a, effects, post);
    let color = effect_light(shaded, effects, reflections, bloom, linear, uv, post);
    // The UNORM canvas stores encoded color.
    return vec4<f32>(encode_srgb(color), 1.0);
}
