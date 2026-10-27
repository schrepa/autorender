// Screen effects, prepended with effects.wgsl. Every one but bloom runs at
// the render size, each pixel by its own depth, normal and ray.
// Ambient occlusion samples the depth buffer at every pixel, and one
// filter smooths it on each surface; the haze's march (haze.wgsl) is
// filtered over time with the last frame's result reprojected; screen-space
// reflections trace each pixel's reflected ray through the depth buffer,
// reading the scene through its own levels, each half the one before;
// bloom filters the scene's glowing light down a chain of levels from half
// the render size and back up; the composite applies them to the scene
// target, upscaled to the display, and writes the canvas. Temporally
// antialiased frames apply them in their resolve instead. Every pass reads
// within what the frame drew: its render size of the screen-sized targets,
// from their top-left corner, and as much of the others.

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
// Reflections: the nearest depth of each 2x2 block of the
// depth buffer, then of each block of the level below, which rays skip
// across.
@group(0) @binding(10) var nearest_depth: texture_2d<f32>;
// Ambient occlusion before its filter: visibility, and the distance from
// the eye (0 where only sky is).
@group(0) @binding(12) var raw_occlusion: texture_2d<f32>;
// Reflections: the surface the opaque pass shaded (`surface_texel`: its
// shading normal, its roughness along the view and across it, and the
// share of its light its mirrored environment makes up), and the scene
// with its levels, the first the scene itself at the render size and each
// next one half the one before, blurred for rough surfaces.
@group(0) @binding(13) var surface_texture: texture_2d<u32>;
@group(0) @binding(14) var scene_levels: texture_2d<f32>;
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

// World position of a depth-buffer pixel; w is 0 where only sky is (the
// cleared depth, 0 with reversed depth).
fn world_at(pixel: vec2<i32>) -> vec4<f32> {
    let depth = textureLoad(depth_texture, pixel, 0);
    let uv = (vec2<f32>(pixel) + 0.5) * post.size.zw;
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = post.inverse_view_projection * clip;
    return vec4<f32>(world.xyz / world.w, select(0.0, 1.0, depth > 0.0));
}

// The unit normal, facing the eye, of the surface the depth buffer holds at
// `pixel`, whose world position is `p`; none (zero) where its neighbors
// give no direction. The nearer neighbor on each axis keeps silhouettes
// from bending it. On the screen's edges the neighbor past the edge is the
// pixel itself, which gives no direction: the one inside stands in.
fn depth_normal(pixel: vec2<i32>, p: vec3<f32>) -> vec3<f32> {
    let last = vec2<i32>(post.size.xy) - 1;
    let right = world_at(min(pixel + vec2<i32>(1, 0), last)).xyz - p;
    let left = p - world_at(max(pixel - vec2<i32>(1, 0), vec2<i32>(0))).xyz;
    let down = world_at(min(pixel + vec2<i32>(0, 1), last)).xyz - p;
    let up = p - world_at(max(pixel - vec2<i32>(0, 1), vec2<i32>(0))).xyz;
    let across = select(right, left,
        pixel.x == last.x || (pixel.x > 0 && dot(left, left) < dot(right, right)));
    let along = select(down, up,
        pixel.y == last.y || (pixel.y > 0 && dot(up, up) < dot(down, down)));
    let normal = cross(along, across);
    let length_squared = dot(normal, normal);
    if !(length_squared > 0.0) {
        return vec3<f32>(0.0);
    }
    let unit = normal * inverseSqrt(length_squared);
    return select(unit, -unit, dot(unit, post.eye.xyz - p) < 0.0);
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

// Reflection rays walk at most this many cells of the depth pyramid.
const REFLECTION_STEPS: u32 = 96u;
// A reflected surface counts where the ray passes within this distance of
// it, in world units, plus this share of the ray's length.
const REFLECTION_THICKNESS: f32 = 0.3;
const REFLECTION_THICKNESS_SHARE: f32 = 0.05;
// Rays heading toward the camera reach surfaces from their hidden side: the
// visible surface stands in for it through this much more depth (a solid
// object's), fading over it.
const REFLECTION_SOLID: f32 = 1.0;
// A roughness lobe's half-width in radians per GGX alpha within the plane of
// incidence, the Phong lobe matching it narrowing to REFLECTION_LOBE_EDGE of
// its peak there (as in cone-traced reflections); across that plane, that
// times the cosine of the view with the normal, as slopes across the view
// turn the mirrored ray out of the plane by that much less than slopes
// along it turn it within (Bruneton, Neyret and Holzschuch 2010).
const REFLECTION_CONE: f32 = 1.19;
const REFLECTION_LOBE_EDGE: f32 = 0.244;
// The most taps a lobe is read with along its long axis on screen: as a
// texture unit's anisotropic filter takes at most as many (8x) along a
// footprint's longer side, past which they read a coarser level.
const REFLECTION_TAPS: u32 = 8u;
// The least cosine with the view a reflecting surface's shading normal
// keeps where the surface itself faces the eye at least as squarely: bumps
// turned past the view bend back to it, so their rays leave the surface.
const REFLECTION_FACING: f32 = 0.05;
// The share of the screen at its edges, and of a ray's reach at its end,
// over which reflections fade out, so neither edge shows; hits nearer keep
// all their weight.
const REFLECTION_EDGE: f32 = 0.1;
const FLOAT_MAX: f32 = 0x1.fffffep+127f;

// The nearest depth (the largest, reversed) of cell `cell` of mip `mip`:
// mip 0 is the depth buffer, mip m > 0 the pyramid's level m - 1, whose
// texels cover 2^m pixels a side (the pyramid repeats the drawn pixels'
// edge), within the cells it reduced: the render size over 2^m, rounded up.
fn nearest_depth_at(cell: vec2<i32>, mip: i32) -> f32 {
    if mip == 0 {
        let last = vec2<i32>(post.size.xy) - 1;
        return textureLoad(depth_texture, clamp(cell, vec2<i32>(0), last), 0);
    }
    let span = 1 << u32(mip);
    let last = max((vec2<i32>(post.size.xy) + span - 1) / span, vec2<i32>(1)) - 1;
    return textureLoad(nearest_depth, clamp(cell, vec2<i32>(0), last), mip - 1).r;
}

// `level`'s color at `at` (its texels, continuous; `size` of them, `drawn`
// of them drawn), bilinearly, kept half a texel within the drawn ones.
fn level_tap(level: texture_2d<f32>, at: vec2<f32>, drawn: vec2<f32>, size: vec2<f32>)
        -> vec3<f32> {
    return textureSampleLevel(level, linear, clamp(at, vec2<f32>(0.5), drawn - 0.5) / size,
        0.0).rgb;
}

// Pixel position (rows downward) and depth of world point `world`, and its
// clip w.
fn screen_point(world: vec3<f32>) -> vec4<f32> {
    let clip = post.view_projection * vec4<f32>(world, 1.0);
    let ndc = clip.xyz / clip.w;
    return vec4<f32>((ndc.x * 0.5 + 0.5) * post.size.x, (0.5 - ndc.y * 0.5) * post.size.y,
        ndc.z, clip.w);
}

// The unit normal at octahedral coordinates `encoded` in [-1, 1].
fn unfolded_normal(encoded: vec2<f32>) -> vec3<f32> {
    let z = 1.0 - abs(encoded.x) - abs(encoded.y);
    let fold = max(-z, 0.0);
    let shift = vec2<f32>(select(-fold, fold, encoded.x >= 0.0),
        select(-fold, fold, encoded.y >= 0.0));
    return normalize(vec3<f32>(encoded - shift, z));
}

// The world point at pixel position `pixel` (continuous) and depth `depth`.
fn world_point(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
    let uv = pixel * post.size.zw;
    let world = post.inverse_view_projection
        * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return world.xyz / world.w;
}

// The surface the opaque pass shaded at a pixel (`surface_texel`'s words):
// its shading normal, its roughness along the view and across it, and the
// share of its light its mirrored environment makes up.
struct Mirroring {
    normal: vec3<f32>,
    roughness: vec2<f32>,
    mirrored: f32,
}

fn mirroring_at(pixel: vec2<i32>) -> Mirroring {
    let words = textureLoad(surface_texture, pixel, 0);
    var mirroring: Mirroring;
    mirroring.normal = unfolded_normal(vec2<f32>(words.xy) / 65535.0 * 2.0 - 1.0);
    mirroring.roughness = unpack4x8unorm(words.z).xy;
    mirroring.mirrored = f32(words.w) / 65535.0;
    return mirroring;
}

// The scene at `at` (render pixels, continuous) as a lobe whose blur spans
// 2^`level` pixels reads it: the scene's level `level`, whose texels span
// as many pixels, mirrors reading the first, the scene itself at the render
// size, and wider lobes blending the two levels around theirs as trilinear
// filtering blends mipmaps.
fn blurred_scene(at: vec2<f32>, level: f32) -> vec3<f32> {
    return textureSampleLevel(scene_levels, linear, drawn_uv(at, post), level).rgb;
}

// The scene's light over a reflection's lobe where it lands about `hit`
// (render pixels, continuous), its half-lengths there `lengths` (pixels):
// along the ray's way on screen `way`, the plane of incidence seen edge on
// from the eye it passes through, and across it. Read at taps along the
// long axis, as many as it spans the short one's width (a pixel at least)
// over, up to REFLECTION_TAPS, each at the level whose blur spans the lobe
// across or the taps' spacing, whichever is wider, and weighted by the
// lobe's Gaussian profile, REFLECTION_LOBE_EDGE of its peak at the
// half-length. A tap counts only where what it shows stands above the
// plane the reflecting surface at `p` lies in (`plane`, its normal: none
// counts every tap) by more than a pixel spans there: the lobe's rays leave
// that plane, so they never meet what lies on it, the reflecting surface
// itself, or under it. Where no tap counts, the hit's own stands in.
fn lobe_light(hit: vec2<f32>, way: vec2<f32>, lengths: vec2<f32>, p: vec3<f32>,
        plane: vec3<f32>) -> vec3<f32> {
    let last = vec2<i32>(post.size.xy) - 1;
    let squared = dot(way, way);
    let along = select(vec2<f32>(0.0, 1.0), way * inverseSqrt(max(squared, 1.0e-30)),
        squared > 1.0e-12);
    let count = u32(clamp(ceil(lengths.x / max(lengths.y, 0.5)), 1.0, f32(REFLECTION_TAPS)));
    let spacing = 2.0 * lengths.x / f32(count);
    let level = log2(max(max(2.0 * lengths.y, spacing), 1.0));
    if count == 1u {
        return blurred_scene(hit, level);
    }
    let no_plane = all(plane == vec3<f32>(0.0));
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var tap = 0u; tap < count; tap++) {
        let offset = f32(2u * tap + 1u) / f32(count) - 1.0;
        let at = hit + along * (offset * lengths.x);
        let seen = world_at(clamp(vec2<i32>(floor(at)), vec2<i32>(0), last));
        let above = dot(seen.xyz - p, plane) > distance(seen.xyz, post.eye.xyz) / post.eye.w;
        if seen.w != 0.0 && !above && !no_plane {
            continue;
        }
        let weight = pow(REFLECTION_LOBE_EDGE, offset * offset);
        sum += blurred_scene(at, level) * weight;
        total += weight;
    }
    if total <= 0.0 {
        return blurred_scene(hit, level);
    }
    return sum / total;
}

// Screen-space reflections at every pixel, every frame: from each
// reflective pixel (the glow target's second channel), its own ray, along
// the mirror direction of its own shading normal, crosses the screen as a
// straight line in pixels and depth. It walks the pyramid of nearest
// depths: a cell the ray crosses in front of everything is skipped and the
// next one is coarser; a cell it may meet is refined, down to single
// pixels (FidelityFX SSSR's hierarchical march), so every pixel along the
// ray near a surface is visited and thin objects are found alike from pixel
// to pixel. The first pixel the ray passes behind is its hit, reflected
// where the ray passes close to that surface (or, heading toward the
// camera, within a solid object's depth of it), fading toward the screen
// edges, the ray's end and the surface's thickness; misses weigh nothing,
// leaving the sky the surface's own shading mirrors (`finish`).
// What it reflects is the scene over the lobe its own roughness spreads the
// ray into (`lobe_light`): long along the ray's way on screen, short across
// it, so grazing views draw it out into streaks. It writes, both
// premultiplied by how surely the ray met the scene, the scene's light as
// the surface's reflectivity mirrors it, and the share of the surface's own
// light its mirrored environment made up, which that light replaces
// (`reflected`).
@fragment
fn fs_reflections(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let last = vec2<i32>(post.size.xy) - 1;
    let pixel = vec2<i32>(position.xy);
    let reflectivity = textureLoad(glow, pixel, 0).g;
    if reflectivity < 0.01 {
        return vec4<f32>(0.0);
    }
    let center = world_at(pixel);
    if center.w == 0.0 {
        return vec4<f32>(0.0);
    }
    let p = center.xyz;
    // The normal the lighting shaded (with normal maps and waves) and the
    // roughness, bent back toward the eye where bumps turn it past the view
    // so the ray leaves the surface: to REFLECTION_FACING or, where the
    // surface the depth buffer holds (`plane`, its normal) faces the eye less
    // squarely, as squarely as it does, so surfaces seen grazing keep their
    // own mirror image.
    let mirroring = mirroring_at(pixel);
    let to_eye = normalize(post.eye.xyz - p);
    let plane = depth_normal(pixel, p);
    let facing = min(REFLECTION_FACING, max(dot(plane, to_eye), 0.0));
    var normal = mirroring.normal;
    normal = normalize(normal + max(facing - dot(normal, to_eye), 0.0) * to_eye);
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
    if ray.z > 0.0 {
        // Heading toward the camera (reversed depth grows), the ray reaches
        // what it meets from the side the camera cannot see, and crosses
        // much of the world per pixel: the surface in front counts fully
        // through the thickness and the ray's length across a pixel there,
        // then fades over a solid object's depth.
        let pixels = max(abs(ray.x), abs(ray.y));
        let stride = distance(
            world_point(point.xy + ray.xy / pixels, point.z + ray.z / pixels), ray_point);
        let near = thickness + stride;
        confidence = 1.0 - smoothstep(near, near + REFLECTION_SOLID, behind);
    }
    let uv = point.xy * post.size.zw;
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    let fraction = min(travelled / max(post.reflections.x, 1.0e-6), 1.0);
    let fade = saturate(edge / REFLECTION_EDGE) * saturate((1.0 - fraction) / REFLECTION_EDGE)
        * confidence;
    // The light reflected over the roughness lobe: its half-lengths where it
    // lands, in pixels, along the plane of incidence and across it
    // (REFLECTION_CONE), the cosine of the view with the normal narrowing it
    // across, so grazing views draw it out along the ray's way.
    let pixels = travelled * post.eye.w / max(distance(ray_point, post.eye.xyz), 1.0e-4);
    let alphas = mirroring.roughness * mirroring.roughness;
    let lengths = REFLECTION_CONE * pixels
        * vec2<f32>(alphas.x, alphas.y * max(dot(normal, to_eye), 0.0));
    let color = lobe_light(point.xy, ray.xy, lengths, p, plane);
    return vec4<f32>(color * (reflectivity * fade), mirroring.mirrored * fade);
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
