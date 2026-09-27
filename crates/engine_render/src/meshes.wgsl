// Uniform: three column-major mat4 and two vec4, 16-byte alignment, exactly
// 224 bytes. `view_projection` is the rasterized projection; temporally
// antialiased frames (decision 017) jitter it by `jitter.xy` clip units per w.
// Their motion uses the previous frame's unjittered `previous_view_projection`
// and `jitter.zw`, two over the viewport size in pixels. The unjittered
// inverse turns pixels into sky rays; `position.xyz` is the eye and
// `position.w` presentation time in seconds (decision 019).
struct Camera {
    view_projection: mat4x4<f32>,
    previous_view_projection: mat4x4<f32>,
    inverse_view_projection: mat4x4<f32>,
    position: vec4<f32>,
    jitter: vec4<f32>,
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
    // Emission scale (decision 019) and the exposed brightness above which
    // light glows (decision 021).
    effects: vec4<f32>,
}
// Sun shadows (decision 020): world to the near map and to both far maps
// (x and y in [-1, 1] across a map, z in [0, 1]).
struct Shadow {
    near: mat4x4<f32>,
    far: mat4x4<f32>,
    // x: shadows on; y: the kept far map holds casters; z: view distance the
    // near map covers; w: distance where shadows begin to fade.
    state: vec4<f32>,
    // x: distance where they are gone; y and z: normal offsets (world units)
    // of lookups in the near and far maps.
    offsets: vec4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lighting: Lighting;
@group(0) @binding(2) var<uniform> shadow: Shadow;
@group(0) @binding(3) var near_shadows: texture_depth_2d;
// The far map of still casters, and the one redrawn every frame.
@group(0) @binding(4) var kept_shadows: texture_depth_2d;
@group(0) @binding(5) var moving_shadows: texture_depth_2d;
@group(0) @binding(6) var shadow_sampler: sampler_comparison;
@group(1) @binding(0) var base_color_texture: texture_2d<f32>;
@group(1) @binding(1) var base_color_sampler: sampler;

// Stable engine-owned slots; three aligned rows give an exact 48-byte affine pose.
struct AffinePose {
    row_0: vec4<f32>,
    row_1: vec4<f32>,
    row_2: vec4<f32>,
}
// A slot's linear color and packed surface (decision 019).
struct SlotColor {
    color: vec3<f32>,
    surface: u32,
}
@group(2) @binding(0) var<storage, read> poses: array<AffinePose>;
@group(2) @binding(1) var<storage, read> colors: array<SlotColor>;
// Temporally antialiased frames: each slot's pose in the previous frame.
@group(2) @binding(2) var<storage, read> previous_poses: array<AffinePose>;
// Temporal visible IDs mark moving slots; others kept their previous pose.
const MOVED_SLOT: u32 = 0x80000000u;

// sRGB targets apply output encoding themselves; UNORM canvas targets do not.
override SURFACE_IS_SRGB: bool = false;

const PI: f32 = 3.14159265;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
    @location(6) color: vec3<f32>,
    @location(7) uv: vec2<f32>,
    @location(8) surface: u32,
}

// Visibility and batching reorder only this four-byte slot ID stream.
struct AffineVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) slot: u32,
    @location(7) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) @interpolate(flat) surface: u32,
}

// `VertexOutput` and the unjittered previous clip position: x, y and w.
struct TemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) @interpolate(flat) surface: u32,
    @location(5) previous: vec3<f32>,
}

// Frames with screen effects (decision 021): graded linear color with the
// ambient share of its light in alpha, and the glow weight of the pixel.
struct SceneTargets {
    @location(0) color: vec4<f32>,
    @location(1) glow: f32,
}

// The scene targets, and the offset from this pixel to the same surface
// point in the previous frame, in texture coordinates.
struct TemporalTargets {
    @location(0) color: vec4<f32>,
    @location(1) motion: vec2<f32>,
    @location(2) glow: f32,
}

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

fn pose_matrix(pose: AffinePose) -> mat4x4<f32> {
    return transpose(mat4x4<f32>(pose.row_0, pose.row_1, pose.row_2,
        vec4<f32>(0.0, 0.0, 0.0, 1.0)));
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let model = mat4x4<f32>(input.model_0, input.model_1, input.model_2, input.model_3);
    return transform_vertex(input.position, input.normal, input.color, input.surface,
        input.uv, model);
}

@vertex
fn vs_affine(input: AffineVertexInput) -> VertexOutput {
    let model = pose_matrix(poses[input.slot]);
    let slot = colors[input.slot];
    return transform_vertex(input.position, input.normal, slot.color, slot.surface,
        input.uv, model);
}

@vertex
fn vs_affine_temporal(input: AffineVertexInput) -> TemporalOutput {
    let slot = input.slot & ~MOVED_SLOT;
    let model = pose_matrix(poses[slot]);
    let material = colors[slot];
    let vertex = transform_vertex(input.position, input.normal, material.color,
        material.surface, input.uv, model);
    let point = vec4<f32>(input.position, 1.0);
    var world = model * point;
    // Reading previous poses costs more than the rest of the motion: only
    // moving slots do.
    if (input.slot & MOVED_SLOT) != 0u {
        let pose = previous_poses[slot];
        world = vec4<f32>(dot(pose.row_0, point), dot(pose.row_1, point),
            dot(pose.row_2, point), 1.0);
    }
    let previous = camera.previous_view_projection * world;
    var output: TemporalOutput;
    output.clip_position = vertex.clip_position;
    output.color = vertex.color;
    output.normal = vertex.normal;
    output.world = vertex.world;
    output.uv = vertex.uv;
    output.surface = vertex.surface;
    output.previous = previous.xyw;
    return output;
}

fn transform_vertex(position: vec3<f32>, normal: vec3<f32>, color: vec3<f32>, surface: u32,
        uv: vec2<f32>, model: mat4x4<f32>) -> VertexOutput {
    let world_position = model * vec4<f32>(position, 1.0);
    // A common positive basis scale cancels out of normalized normals. Rescale
    // before cross products and again before multiplication to avoid squared
    // length overflow for large or small admitted world transforms. CPU
    // admission separately bounds normal-matrix conditioning.
    let basis_scale = max(maximum_abs(model[0].xyz),
        max(maximum_abs(model[1].xyz), maximum_abs(model[2].xyz)));
    let x = model[0].xyz / basis_scale;
    let y = model[1].xyz / basis_scale;
    let z = model[2].xyz / basis_scale;
    let cofactors = mat3x3<f32>(
        cross(y, z), cross(z, x), cross(x, y),
    );
    let determinant = dot(x, cofactors[0]);
    let cofactor_scale = max(maximum_abs(cofactors[0]),
        max(maximum_abs(cofactors[1]), maximum_abs(cofactors[2])));
    let normal_matrix = mat3x3<f32>(cofactors[0] / cofactor_scale,
        cofactors[1] / cofactor_scale, cofactors[2] / cofactor_scale);
    var output: VertexOutput;
    output.clip_position = camera.view_projection * world_position;
    output.normal = stable_normalize(normal_matrix * normal) * sign(determinant);
    output.color = color;
    output.world = world_position.xyz;
    output.uv = uv;
    output.surface = surface;
    return output;
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let low = color * 12.92;
    let high = 1.055 * pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, color <= vec3<f32>(0.0031308));
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

// Exposure, a filmic curve (Narkowicz's ACES fit) and saturation.
fn grade(color: vec3<f32>) -> vec3<f32> {
    let exposed = max(color, vec3<f32>(0.0)) * lighting.horizon.w;
    let mapped = saturate((exposed * (2.51 * exposed + 0.03))
        / (exposed * (2.43 * exposed + 0.59) + 0.14));
    let luma = dot(mapped, vec3<f32>(0.2126, 0.7152, 0.0722));
    return saturate(mix(vec3<f32>(luma), mapped, lighting.ground.w));
}

// Glow weight of light `color`: its exposed brightness above the bloom
// threshold, as x / (1 + x) (decision 021).
fn glow_weight(color: vec3<f32>) -> f32 {
    let bright = max(luminance(color) * lighting.horizon.w - lighting.effects.y, 0.0);
    return bright / (1.0 + bright);
}

// Visibility of the sun at `world` (decision 020), looked up a little along
// the surface normal so surfaces do not shadow themselves. The near map
// covers the first stretch of the view; the far maps, the kept one for
// still casters and the one redrawn every frame, cover the rest and fade out
// at the shadow distance. Comparison lookups filter four texels.
fn sun_visibility(world: vec3<f32>, normal: vec3<f32>, distance: f32) -> f32 {
    if shadow.state.x == 0.0 {
        return 1.0;
    }
    if distance < shadow.state.z {
        let clip = shadow.near * vec4<f32>(world + normal * shadow.offsets.y, 1.0);
        if all(abs(clip.xy) < vec2<f32>(0.98)) && clip.z < 1.0 {
            let uv = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
            return textureSampleCompareLevel(near_shadows, shadow_sampler, uv, clip.z);
        }
    }
    let clip = shadow.far * vec4<f32>(world + normal * shadow.offsets.z, 1.0);
    if any(abs(clip.xy) >= vec2<f32>(1.0)) || clip.z >= 1.0 {
        return 1.0;
    }
    let uv = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
    var visible = textureSampleCompareLevel(moving_shadows, shadow_sampler, uv, clip.z);
    if shadow.state.y != 0.0 {
        visible = min(visible, textureSampleCompareLevel(kept_shadows, shadow_sampler, uv, clip.z));
    }
    let fade = saturate((distance - shadow.state.w) / (shadow.offsets.x - shadow.state.w));
    return mix(visible, 1.0, fade);
}

// Wave normals of liquid surfaces (decision 019): three directional waves
// scrolling with presentation time tilt the normal by their slopes.
fn rippled(normal: vec3<f32>, world: vec3<f32>, strength: f32) -> vec3<f32> {
    let time = camera.position.w;
    let p = world.xz;
    let a = dot(p, vec2<f32>(0.8, 0.6)) * 0.7 + time * 1.1;
    let b = dot(p, vec2<f32>(-0.6, 0.8)) * 1.6 + time * 1.7;
    let c = dot(p, vec2<f32>(0.1, -1.0)) * 3.3 + time * 2.3;
    let slope = vec2<f32>(0.8, 0.6) * (0.126 * cos(a))
        + vec2<f32>(-0.6, 0.8) * (0.112 * cos(b))
        + vec2<f32>(0.1, -1.0) * (0.083 * cos(c));
    return normalize(normal - vec3<f32>(slope.x, 0.0, slope.y) * strength);
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

// Shaded light of a surface: graded color, the ambient share of its light
// (for ambient occlusion) and its glow weight.
struct Shaded {
    color: vec3<f32>,
    ambient: f32,
    glow: f32,
}

// Lit, fogged and graded color. Sampling uses implicit derivatives, so
// callers shade in uniform control flow; the rest of this function may not.
fn shade(color: vec3<f32>, surface_normal: vec3<f32>, world: vec3<f32>, uv: vec2<f32>,
        packed: u32) -> Shaded {
    // RGBA8UnormSrgb sampling decodes source RGB; alpha masks emission.
    let texel = textureSample(base_color_texture, base_color_sampler, uv);
    let albedo = texel.rgb * color;
    // Roughness, metalness, emission over 16 and ripples.
    let surface = unpack4x8unorm(packed);
    let roughness = max(surface.x, 0.03);
    let metalness = surface.y;
    let geometric = stable_normalize(surface_normal);
    var normal = geometric;
    if surface.w > 0.0 {
        normal = rippled(normal, world, surface.w);
    }
    let eye = camera.position.xyz;
    let to_eye = eye - world;
    let distance = length(to_eye);
    let view = stable_normalize(to_eye);
    let sun = lighting.sun.xyz;
    let n_l = dot(normal, sun);
    let n_v = max(dot(normal, view), 1.0e-4);
    let alpha = roughness * roughness;
    let f0 = mix(vec3<f32>(0.04), albedo, metalness);
    let diffuse_color = albedo * (1.0 - metalness);
    let specular = lighting.sun_color.w;
    // Wrapped diffuse softens the terminator of flat low-poly faces.
    let wrapped = max((n_l + 0.25) / 1.25, 0.0);
    var highlight = vec3<f32>(0.0);
    if n_l > 0.0 {
        let half_vector = normalize(view + sun);
        let n_h = max(dot(normal, half_vector), 0.0);
        highlight = fresnel(f0, max(dot(view, half_vector), 0.0))
            * (distribution(n_h, alpha) * visibility(n_v, n_l, alpha) * n_l);
    }
    // The sky mirrored in the surface, blurring toward its hemisphere
    // average as the surface roughens, weighted by the environment BRDF.
    let reflected = reflect(-view, normal);
    let mirrored = mix(haze(reflected), sky_average(reflected), roughness);
    let reflection = mirrored * environment_brdf(f0, roughness, n_v) * specular;
    // Rim light outlines walls against the sky; level surfaces seen at
    // grazing angles (distant ground and roofs) take none.
    let grazing = 1.0 - max(dot(geometric, view), 0.0);
    let rim = (grazing * grazing) * (grazing * grazing) * lighting.sun.w
        * (1.0 - abs(geometric.y));
    let sky_light = diffuse_color * (sky_average(normal) * lighting.zenith.w
        + lighting.horizon.rgb * rim) + reflection;
    let sun_light = (diffuse_color * wrapped + highlight * specular) * lighting.sun_color.rgb
        * sun_visibility(world, geometric, distance);
    let emitted = albedo * (surface.z * 16.0 * texel.a * lighting.effects.x);
    let lit = sky_light + sun_light + emitted;
    let covered = fog(eye, world);
    let fogged = mix(lit, haze(-view), covered);
    var shaded: Shaded;
    shaded.color = grade(fogged);
    shaded.ambient = saturate(luminance(sky_light) / max(luminance(lit), 1.0e-6)) * (1.0 - covered);
    shaded.glow = glow_weight(fogged);
    return shaded;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = shade(input.color, input.normal, input.world, input.uv, input.surface).color;
    if SURFACE_IS_SRGB {
        return vec4<f32>(color, 1.0);
    }
    return vec4<f32>(linear_to_srgb(color), 1.0);
}

// The scene target is sRGB: it stores linear color with output encoding.
@fragment
fn fs_post(input: VertexOutput) -> SceneTargets {
    let shaded = shade(input.color, input.normal, input.world, input.uv, input.surface);
    var targets: SceneTargets;
    targets.color = vec4<f32>(shaded.color, shaded.ambient);
    targets.glow = shaded.glow;
    return targets;
}

@fragment
fn fs_temporal(input: TemporalOutput) -> TemporalTargets {
    let shaded = shade(input.color, input.normal, input.world, input.uv, input.surface);
    var targets: TemporalTargets;
    targets.color = vec4<f32>(shaded.color, shaded.ambient);
    targets.glow = shaded.glow;
    // This pixel center without the jitter, and the same point last frame.
    let current = vec2<f32>(input.clip_position.x * camera.jitter.z - 1.0,
        1.0 - input.clip_position.y * camera.jitter.w) - camera.jitter.xy;
    let previous = input.previous.xy / input.previous.z;
    targets.motion = (previous - current) * vec2<f32>(0.5, -0.5);
    // A point behind the previous camera has no position there: send its
    // history lookup out of the viewport.
    if !(input.previous.z > 0.0) {
        targets.motion = vec2<f32>(2.0);
    }
    return targets;
}

// The sky behind all geometry: one triangle over the viewport at the far
// plane, drawn after the meshes so depth testing leaves only uncovered pixels.
struct SkyOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_sky(@builtin(vertex_index) index: u32) -> SkyOutput {
    let ndc = vec2<f32>(f32((index << 1u) & 2u) * 2.0 - 1.0, f32(index & 2u) * 2.0 - 1.0);
    var output: SkyOutput;
    output.clip_position = vec4<f32>(ndc, 1.0, 1.0);
    output.ndc = ndc;
    return output;
}

// The view ray through unjittered `ndc`.
fn sky_ray(ndc: vec2<f32>) -> vec3<f32> {
    let far = camera.inverse_view_projection * vec4<f32>(ndc, 1.0, 1.0);
    return stable_normalize(far.xyz / far.w - camera.position.xyz);
}

@fragment
fn fs_sky(input: SkyOutput) -> @location(0) vec4<f32> {
    let color = grade(sky(sky_ray(input.ndc)));
    if SURFACE_IS_SRGB {
        return vec4<f32>(color, 1.0);
    }
    return vec4<f32>(linear_to_srgb(color), 1.0);
}

// Screen-effect frames: the sky has no ambient occlusion; its sun glows.
@fragment
fn fs_sky_post(input: SkyOutput) -> SceneTargets {
    let light = sky(sky_ray(input.ndc));
    var targets: SceneTargets;
    targets.color = vec4<f32>(grade(light), 0.0);
    targets.glow = glow_weight(light);
    return targets;
}

// Temporal frames: the unjittered sky reprojects by the camera's turn alone.
@fragment
fn fs_sky_temporal(input: SkyOutput) -> TemporalTargets {
    let ray = sky_ray(input.ndc);
    let light = sky(ray);
    var targets: TemporalTargets;
    targets.color = vec4<f32>(grade(light), 0.0);
    targets.glow = glow_weight(light);
    let previous = camera.previous_view_projection * vec4<f32>(ray, 0.0);
    targets.motion = (previous.xy / previous.w - input.ndc) * vec2<f32>(0.5, -0.5);
    if !(previous.w > 0.0) {
        targets.motion = vec2<f32>(2.0);
    }
    return targets;
}
