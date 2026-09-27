// Screen effects (decisions 021 and 025), prepended with effects.wgsl.
// Ambient occlusion samples the depth buffer at half resolution;
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

// World position of a depth-buffer pixel; w is 0 where only sky is.
fn world_at(pixel: vec2<i32>) -> vec4<f32> {
    let depth = textureLoad(depth_texture, pixel, 0);
    let uv = (vec2<f32>(pixel) + 0.5) * post.size.zw;
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = post.inverse_view_projection * clip;
    return vec4<f32>(world.xyz / world.w, select(0.0, 1.0, depth < 1.0));
}

// Scalable ambient obscurance (McGuire et al.) over eight samples: each
// sample closer to the surface's tangent plane than the radius occludes by
// how far above that plane, and how close, it is.
@fragment
fn fs_occlusion(@builtin(position) position: vec4<f32>) -> @location(0) f32 {
    let last = vec2<i32>(post.size.xy) - 1;
    let pixel = min(vec2<i32>(position.xy) * 2, last);
    let center = world_at(pixel);
    if center.w == 0.0 {
        return 1.0;
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
    var normal = cross(along, across);
    let length_squared = dot(normal, normal);
    if !(length_squared > 0.0) {
        return 1.0;
    }
    normal = normal * inverseSqrt(length_squared);
    if dot(normal, to_eye) < 0.0 {
        normal = -normal;
    }
    let distance = length(to_eye);
    let radius = post.occlusion.x;
    let pixels = min(radius * post.eye.w / distance, MAX_PIXELS);
    if pixels < 1.0 {
        return 1.0;
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
    return 1.0 - post.occlusion.y * obscurance;
}

// Reflection rays take this many steps, spaced closer near the surface.
const REFLECTION_STEPS: u32 = 24u;

// Screen-space reflections (decision 025) at half resolution: from each
// reflective pixel (the glow target's second channel), a ray along the
// mirror direction of its depth normal steps across the depth buffer; the
// first step behind a surface, refined by bisection, is the reflected point.
// The result is its scene color and the pixel's reflectivity faded toward
// the screen edges and the ray's end; misses weigh nothing.
@fragment
fn fs_reflections(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let last = vec2<i32>(post.size.xy) - 1;
    let pixel = min(vec2<i32>(position.xy) * 2, last);
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
    let distance = post.reflections.x;
    let noise = fract(52.9829189 * fract(dot(position.xy, vec2<f32>(0.06711056, 0.00583715))));
    // Start a little off the surface so it does not hit itself.
    let origin = p + normal * (0.002 * length(post.eye.xyz - p) + 0.02);
    var previous = 0.0;
    for (var i = 0u; i < REFLECTION_STEPS; i++) {
        let fraction = (f32(i) + noise) / f32(REFLECTION_STEPS);
        let t = distance * fraction * fraction;
        let clip = post.view_projection * vec4<f32>(origin + direction * t, 1.0);
        if clip.w <= 0.0 {
            break;
        }
        let ndc = clip.xyz / clip.w;
        if any(abs(ndc.xy) >= vec2<f32>(1.0)) {
            break;
        }
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        let sample_pixel = min(vec2<i32>(uv * post.size.xy), last);
        let depth = textureLoad(depth_texture, sample_pixel, 0);
        if ndc.z > depth && depth < 1.0 {
            // Behind a surface: bisect between the last two steps, then keep
            // it only if the ray passed close behind that surface.
            var low = previous;
            var high = t;
            var hit = uv;
            for (var j = 0u; j < 4u; j++) {
                let middle = (low + high) * 0.5;
                let probe = post.view_projection * vec4<f32>(origin + direction * middle, 1.0);
                let probe_ndc = probe.xyz / probe.w;
                let probe_uv = vec2<f32>(probe_ndc.x * 0.5 + 0.5, 0.5 - probe_ndc.y * 0.5);
                let probe_depth = textureLoad(depth_texture,
                    min(vec2<i32>(probe_uv * post.size.xy), last), 0);
                if probe_ndc.z > probe_depth {
                    high = middle;
                    hit = probe_uv;
                } else {
                    low = middle;
                }
            }
            let surface = world_at(min(vec2<i32>(hit * post.size.xy), last)).xyz;
            let ray_point = origin + direction * high;
            let behind = length(ray_point - post.eye.xyz) - length(surface - post.eye.xyz);
            if behind > 0.3 + 0.05 * high {
                return vec4<f32>(0.0);
            }
            let edge = min(min(hit.x, 1.0 - hit.x), min(hit.y, 1.0 - hit.y));
            let fade = saturate(edge * 10.0) * (1.0 - fraction * fraction);
            let color = textureSampleLevel(scene, linear, hit, 0.0).rgb;
            return vec4<f32>(color, reflectivity * fade);
        }
        previous = t;
    }
    return vec4<f32>(0.0);
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
