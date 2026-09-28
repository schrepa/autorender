// Shared scene declarations, prepended to the mesh, particle and sun shaft
// shaders: the first bind group's camera, atmosphere (decision 018) and sun
// shadows (decision 020), and the lighting functions every path shades with.

// Uniform: three column-major mat4 and three vec4, 16-byte alignment,
// exactly 240 bytes. `view_projection` is the rasterized projection; temporally
// antialiased frames (decision 017) jitter it by `jitter.xy` clip units per w.
// Their motion uses the previous frame's unjittered `previous_view_projection`
// and `jitter.zw`, two over the viewport size in pixels. The unjittered
// inverse turns pixels into sky rays; `position.xyz` is the eye and
// `position.w` presentation time in seconds (decision 019). `options.x` is 1
// while a reflection probe captures a cube face (decision 043).
struct Camera {
    view_projection: mat4x4<f32>,
    previous_view_projection: mat4x4<f32>,
    inverse_view_projection: mat4x4<f32>,
    position: vec4<f32>,
    jitter: vec4<f32>,
    options: vec4<f32>,
}
// The scene's atmosphere (decision 018) in linear color; scalars ride in w.
struct Lighting {
    // Toward the sun; w: rim strength.
    sun: vec4<f32>,
    // w: specular scale (1 is physically based).
    sun_color: vec4<f32>,
    // w: ambient scale.
    zenith: vec4<f32>,
    // w: exposure.
    horizon: vec4<f32>,
    // w: saturation.
    ground: vec4<f32>,
    // Start, density, height falloff and maximum.
    fog: vec4<f32>,
    // Emission scale (decision 019), the exposed brightness above which
    // light glows (decision 021), the roughness beyond which surfaces
    // reflect only the sky, and 1 when screen-space reflections run
    // (decision 025).
    effects: vec4<f32>,
    // Contrast and vibrance (decision 027).
    grade: vec4<f32>,
    // Split toning: luminance-normalized gains of shadows and highlights.
    shadow_tint: vec4<f32>,
    highlight_tint: vec4<f32>,
    // Refraction through glass (decision 038): the bend's strength (0 without
    // refraction) and the blur radius in pixels of fully rough glass.
    refraction: vec4<f32>,
}
// Sun shadows (decisions 020 and 028): world to each near cascade and to both
// far maps (x and y in [-1, 1] across a map, z in [0, 1]).
struct Shadow {
    near: array<mat4x4<f32>, 4>,
    far: mat4x4<f32>,
    // x: shadows on; y: the kept far map holds casters; z: near cascades in
    // use; w: distance where shadows begin to fade.
    state: vec4<f32>,
    // View distance each near cascade reaches; unreachable beyond those in
    // use.
    reach: vec4<f32>,
    // Normal offsets (world units) of lookups in each near cascade.
    near_offsets: vec4<f32>,
    // Steepest receiver plane (depth per texel) of each near cascade.
    near_limits: vec4<f32>,
    // x: distance where shadows are gone; y: normal offset of far lookups;
    // z: steepest receiver plane of the far maps.
    offsets: vec4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lighting: Lighting;
@group(0) @binding(2) var<uniform> shadow: Shadow;
// The near cascades, one layer each.
@group(0) @binding(3) var near_shadows: texture_depth_2d_array;
// The far map of still casters, and the one redrawn every frame.
@group(0) @binding(4) var kept_shadows: texture_depth_2d;
@group(0) @binding(5) var moving_shadows: texture_depth_2d;
@group(0) @binding(6) var shadow_sampler: sampler_comparison;
// Unfiltered reads of four shadow texels, for receiver planes.
@group(0) @binding(7) var shadow_texels: sampler;

// sRGB targets apply output encoding themselves; UNORM canvas targets do not.
override SURFACE_IS_SRGB: bool = false;

const PI: f32 = 3.14159265;

fn maximum_abs(value: vec3<f32>) -> f32 {
    let positive = abs(value);
    return max(positive.x, max(positive.y, positive.z));
}

fn stable_normalize(value: vec3<f32>) -> vec3<f32> {
    let scale = maximum_abs(value);
    // Interpolation between opposing vertex normals can produce a zero vector.
    if scale == 0.0 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(value / scale);
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let low = color * 12.92;
    let high = 1.055 * pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, color <= vec3<f32>(0.0031308));
}

// Graded linear color as the target stores it.
fn encoded(color: vec3<f32>) -> vec3<f32> {
    if SURFACE_IS_SRGB {
        return color;
    }
    return linear_to_srgb(color);
}

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// The sky's light in `direction` without the sun's disc: horizon to zenith
// above; below, haze darkening only partway toward the ground, as distant
// ground seen through air does; and a wide halo toward the sun. Fog and
// reflections take this color.
fn haze(direction: vec3<f32>) -> vec3<f32> {
    let up = direction.y;
    var color = mix(lighting.horizon.rgb, lighting.zenith.rgb, sqrt(max(up, 0.0)));
    color = mix(color, lighting.ground.rgb, min(max(-up, 0.0) * 1.5, 0.35));
    let toward = max(dot(direction, lighting.sun.xyz), 0.0);
    let square = toward * toward;
    return color + lighting.sun_color.rgb * (0.12 * square * square * square);
}

// The sky itself: its haze and the sun's small disc.
fn sky(direction: vec3<f32>) -> vec3<f32> {
    let toward = max(dot(direction, lighting.sun.xyz), 0.0);
    return haze(direction) + lighting.sun_color.rgb * (6.0 * pow(toward, 900.0));
}

// The sky's average light over the hemisphere around `normal`: the zenith
// from above, half horizon and half zenith from the side (cooler than the
// sun, so shade contrasts with light) and the ground from below.
fn sky_average(normal: vec3<f32>) -> vec3<f32> {
    let side = mix(lighting.horizon.rgb, lighting.zenith.rgb, 0.5);
    let above = mix(side, lighting.zenith.rgb, max(normal.y, 0.0));
    return mix(above, lighting.ground.rgb, max(-normal.y, 0.0));
}

// Fog covering `point` seen from `eye`: exponential in the distance beyond
// the start, with a density decaying exponentially with height, integrated
// along the ray.
fn fog(eye: vec3<f32>, point: vec3<f32>) -> f32 {
    let ray = point - eye;
    let distance = length(ray);
    let start = lighting.fog.x;
    if distance <= start {
        return 0.0;
    }
    let falloff = lighting.fog.z;
    // Far below the ground the density would overflow; it is saturated there.
    let from_height = max(eye.y + ray.y * (start / distance), -1000.0);
    let to_height = max(point.y, -1000.0);
    let rise = falloff * (to_height - from_height);
    let near = exp(-falloff * from_height);
    let far = exp(-falloff * to_height);
    // The mean density along the ray; its limit where the ray is level.
    let mean = select(near, (near - far) / rise, abs(rise) > 1.0e-4);
    let depth = lighting.fog.y * (distance - start) * mean;
    return min(1.0 - exp(-depth), lighting.fog.w);
}

// Narkowicz's fit of the ACES filmic curve.
fn filmic(x: vec3<f32>) -> vec3<f32> {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

// The grade of every shaded color (decisions 018 and 027): exposure; the
// filmic curve, half per channel and half on the brightest channel, which
// keeps the hue and saturation of bright colors; contrast as an S-curve in a
// perceptual (square root) space that keeps black and white, so dark shade
// deepens without crushing; split toning of shadows and highlights; then
// saturation, and vibrance that lifts muted colors more than vivid ones.
fn grade(color: vec3<f32>) -> vec3<f32> {
    let exposed = max(color, vec3<f32>(0.0)) * lighting.horizon.w;
    let peak = max(exposed.r, max(exposed.g, exposed.b));
    let kept = exposed * (filmic(vec3<f32>(peak)).r / max(peak, 1.0e-6));
    let curved = mix(filmic(exposed), kept, 0.5);
    let root = sqrt(curved);
    let perceptual = root + lighting.grade.x * (root * root * (3.0 - 2.0 * root) - root);
    var mapped = perceptual * perceptual;
    let tone = smoothstep(0.2, 0.8, sqrt(luminance(mapped)));
    mapped *= mix(lighting.shadow_tint.rgb, lighting.highlight_tint.rgb, tone);
    let top = max(mapped.r, max(mapped.g, mapped.b));
    let bottom = min(mapped.r, min(mapped.g, mapped.b));
    let muted = 1.0 - (top - bottom) / max(top, 1.0e-6);
    let amount = lighting.ground.w + lighting.grade.y * muted;
    return saturate(mix(vec3<f32>(luminance(mapped)), mapped, amount));
}

// Glow weight of light `color`: its exposed brightness above the bloom
// threshold, as x / (1 + x) (decision 021).
fn glow_weight(color: vec3<f32>) -> f32 {
    let bright = max(luminance(color) * lighting.horizon.w - lighting.effects.y, 0.0);
    return bright / (1.0 + bright);
}

// Visibility of the sun at `world` in the far maps: the kept one of still
// casters and the one redrawn every frame; 1 outside them.
fn far_visibility(world: vec3<f32>) -> f32 {
    let clip = shadow.far * vec4<f32>(world, 1.0);
    if any(abs(clip.xy) >= vec2<f32>(1.0)) || clip.z >= 1.0 {
        return 1.0;
    }
    let uv = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
    var visible = textureSampleCompareLevel(moving_shadows, shadow_sampler, uv, clip.z);
    if shadow.state.y != 0.0 {
        visible = min(visible, textureSampleCompareLevel(kept_shadows, shadow_sampler, uv, clip.z));
    }
    return visible;
}

// Receiver planes (decision 028): a surface's lookup compares each of the
// four texels around it with the surface's own plane at that texel's center.
// A tilted surface then matches its own depths, so lookups need no margin
// that grows with texel size, which would keep every caster's lowest part
// (all of a low one) from shadowing what it stands on. Coarser maps then
// only make shadow edges blockier. The margin left covers depth rounding.
const RECEIVER_MARGIN: f32 = 2.0 / 65535.0;

// A receiver in a shadow map: its position in texels, its depth, and how its
// depth changes per texel along the map's axes.
struct MapPoint {
    texel: vec2<f32>,
    depth: f32,
    slope: vec2<f32>,
}

// `world` in the map `clip_from_world`, `size` texels a side. The plane
// comes from how the surface moves per pixel (`dx` and `dy`); its slope stays
// within `limit` depth per texel where the surface turns edge-on to the sun.
fn map_point(clip_from_world: mat4x4<f32>, world: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>,
        size: f32, limit: f32) -> MapPoint {
    let clip = clip_from_world * vec4<f32>(world, 1.0);
    let clip_dx = clip_from_world * vec4<f32>(dx, 0.0);
    let clip_dy = clip_from_world * vec4<f32>(dy, 0.0);
    // Texel rows run downward.
    let scale = vec2<f32>(0.5, -0.5) * size;
    let texel_dx = clip_dx.xy * scale;
    let texel_dy = clip_dy.xy * scale;
    let det = texel_dx.x * texel_dy.y - texel_dx.y * texel_dy.x;
    var slope = vec2<f32>(0.0);
    if abs(det) > 1.0e-12 {
        slope = vec2<f32>(clip_dx.z * texel_dy.y - clip_dy.z * texel_dx.y,
            clip_dy.z * texel_dx.x - clip_dx.z * texel_dy.x) / det;
    }
    var point: MapPoint;
    point.texel = (clip.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5)) * size;
    point.depth = clip.z;
    point.slope = clamp(slope, vec2<f32>(-limit), vec2<f32>(limit));
    return point;
}

// Where to gather the four texels around `point` in a map `size` texels a
// side: their shared corner, so the hardware's limited subtexel precision
// cannot pick a neighboring block near texel edges.
fn gather_uv(point: MapPoint, size: f32) -> vec2<f32> {
    return (floor(point.texel - vec2<f32>(0.5)) + vec2<f32>(1.0)) / size;
}

// Visibility of `point` from the four texel depths around it, gathered at
// `gather_uv`, in gather order (least u and greatest v, both greatest,
// greatest u and least v, both least), each compared with the receiver's
// plane at the texel's center and weighted bilinearly.
fn plane_visibility(depths: vec4<f32>, point: MapPoint) -> f32 {
    let corner = point.texel - vec2<f32>(0.5);
    let f = corner - floor(corner);
    let du = vec4<f32>(-f.x, 1.0 - f.x, 1.0 - f.x, -f.x);
    let dv = vec4<f32>(1.0 - f.y, 1.0 - f.y, -f.y, -f.y);
    let plane = point.depth + point.slope.x * du + point.slope.y * dv - RECEIVER_MARGIN;
    let lit = select(vec4<f32>(0.0), vec4<f32>(1.0), plane <= depths);
    return mix(mix(lit.w, lit.z, f.x), mix(lit.x, lit.y, f.x), f.y);
}

// Visibility of the surface at `world` in near cascade `cascade` (decision
// 028), looked up `normal` times its offset away; -1 where the cascade does
// not cover the lookup.
fn cascade_visibility(cascade: u32, world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>,
        dy: vec3<f32>) -> f32 {
    let size = f32(textureDimensions(near_shadows).x);
    let point = map_point(shadow.near[cascade], world + normal * shadow.near_offsets[cascade],
        dx, dy, size, shadow.near_limits[cascade]);
    let uv = point.texel / size;
    if any(abs(uv * 2.0 - vec2<f32>(1.0)) >= vec2<f32>(0.98)) || point.depth >= 1.0 {
        return -1.0;
    }
    let corner = gather_uv(point, size);
    return plane_visibility(textureGather(near_shadows, shadow_texels, corner, cascade), point);
}

// Visibility of the surface at `world` in the far maps; 1 outside them.
fn far_surface_visibility(world: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>) -> f32 {
    let size = f32(textureDimensions(moving_shadows).x);
    let point = map_point(shadow.far, world, dx, dy, size, shadow.offsets.z);
    let uv = point.texel / size;
    if any(abs(uv * 2.0 - vec2<f32>(1.0)) >= vec2<f32>(1.0)) || point.depth >= 1.0 {
        return 1.0;
    }
    let corner = gather_uv(point, size);
    var visible = plane_visibility(textureGather(moving_shadows, shadow_texels, corner), point);
    if shadow.state.y != 0.0 {
        let kept = plane_visibility(textureGather(kept_shadows, shadow_texels, corner), point);
        visible = min(visible, kept);
    }
    return visible;
}

// Visibility of the sun at the surface point `world` (decision 020), whose
// position changes by `dx` and `dy` per pixel, looked up a little along the
// surface normal. Near cascades cover the first stretch of the view, each
// three times sharper than the next (decision 028), and hand over to the next
// one across the last tenth of their reach; the far maps, the kept one for
// still casters and the one redrawn every frame, cover the rest and fade out
// at the shadow distance. Lookups compare four texels with the surface's
// plane.
fn sun_visibility(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>,
        distance: f32) -> f32 {
    if shadow.state.x == 0.0 {
        return 1.0;
    }
    // The first cascade reaching the fragment. Reflection probe captures
    // (decision 043) look from elsewhere than the cascades cover and select
    // for: they read the far maps alone.
    let beyond = select(vec4<f32>(0.0), vec4<f32>(1.0), vec4<f32>(distance) >= shadow.reach);
    let cascade = u32(dot(beyond, vec4<f32>(1.0)));
    let cascades = select(u32(shadow.state.z), 0u, camera.options.x != 0.0);
    if cascade < cascades {
        let visible = cascade_visibility(cascade, world, normal, dx, dy);
        if visible >= 0.0 {
            let reach = shadow.reach[cascade];
            let handover = saturate((distance - reach * 0.9) / (reach * 0.1));
            if handover > 0.0 && cascade + 1u < cascades {
                let next = cascade_visibility(cascade + 1u, world, normal, dx, dy);
                if next >= 0.0 {
                    return mix(visible, next, handover);
                }
            }
            return visible;
        }
    }
    let visible = far_surface_visibility(world + normal * shadow.offsets.y, dx, dy);
    let fade = saturate((distance - shadow.state.w) / (shadow.offsets.x - shadow.state.w));
    return mix(visible, 1.0, fade);
}

// Trowbridge-Reitz (GGX) distribution of microfacet normals.
fn distribution(n_h: f32, alpha: f32) -> f32 {
    let alpha2 = alpha * alpha;
    let d = n_h * n_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * d * d);
}

// Smith-GGX visibility (Hammon's approximation of the correlated form).
fn visibility(n_v: f32, n_l: f32, alpha: f32) -> f32 {
    return 0.5 / mix(2.0 * n_l * n_v, n_l + n_v, alpha);
}

fn fresnel(f0: vec3<f32>, cosine: f32) -> vec3<f32> {
    let m = 1.0 - cosine;
    let m2 = m * m;
    return f0 + (vec3<f32>(1.0) - f0) * (m2 * m2 * m);
}

// Karis's analytic fit of the split-sum environment BRDF.
fn environment_brdf(f0: vec3<f32>, roughness: f32, n_v: f32) -> vec3<f32> {
    let r = roughness * vec4<f32>(-1.0, -0.0275, -0.572, 0.022)
        + vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let a004 = min(r.x * r.x, exp2(-9.28 * n_v)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

// Interleaved gradient noise (Jimenez) at pixel `position`: in [0, 1),
// fixed per pixel.
fn pixel_noise(position: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(position, vec2<f32>(0.06711056, 0.00583715))));
}

// Weighted blended order-independent transparency (McGuire and Bavoil;
// decision 026): premultiplied color and coverage times a depth weight,
// added; and the product of what each surface lets through.
struct TransparentTargets {
    @location(0) accumulated: vec4<f32>,
    @location(1) revealed: f32,
}

// A transparent contribution of graded `color` covering `opacity` of what
// is behind it, `distance` from the eye: nearer surfaces weigh more.
fn weighted(color: vec3<f32>, opacity: f32, distance: f32) -> TransparentTargets {
    let near = distance / 5.0;
    let far = distance / 200.0;
    let far3 = far * far * far;
    let weight = opacity * clamp(10.0 / (1.0e-5 + near * near + far3 * far3), 0.01, 3000.0);
    var targets: TransparentTargets;
    targets.accumulated = vec4<f32>(color * opacity, opacity) * weight;
    targets.revealed = opacity;
    return targets;
}
