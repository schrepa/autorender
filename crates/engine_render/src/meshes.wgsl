// Mesh drawing, prepended with scene.wgsl. Opaque entry points write the
// canvas, the screen effects' scene targets (decision 021) or temporal
// targets (decision 017); `_lit` variants add clustered local lights
// (decision 023); transparent ones accumulate weighted blended transparency
// (decision 026). Unlit entry points never touch the lights' bindings.

@group(1) @binding(0) var base_color_texture: texture_2d<f32>;
@group(1) @binding(1) var base_color_sampler: sampler;

// Stable engine-owned slots; three aligned rows give an exact 48-byte affine pose.
struct AffinePose {
    row_0: vec4<f32>,
    row_1: vec4<f32>,
    row_2: vec4<f32>,
}
// A slot's linear color and opacity as four 16-bit fractions (red and green,
// then blue and opacity; decision 026), its packed surface (decision 019)
// and unused flags.
struct SlotColor {
    color: vec2<u32>,
    surface: u32,
    flags: u32,
}
@group(2) @binding(0) var<storage, read> poses: array<AffinePose>;
@group(2) @binding(1) var<storage, read> colors: array<SlotColor>;
// Temporally antialiased frames: each slot's pose in the previous frame.
@group(2) @binding(2) var<storage, read> previous_poses: array<AffinePose>;
// Temporal visible IDs mark moving slots; others kept their previous pose.
const MOVED_SLOT: u32 = 0x80000000u;

// Local lights (decision 023), binned into view clusters by lights.wgsl.
struct Clusters {
    view_projection: mat4x4<f32>,
    // Depth along the view of point p: dot(xyz, p) + w.
    depth: vec4<f32>,
    // x and y projection scales, clustered depth range start and end.
    projection: vec4<f32>,
    // Tile width and height in pixels; slice = log2(depth) * z + w.
    grid: vec4<f32>,
    // Tiles across and down, depth slices, lights per cluster.
    counts: vec4<u32>,
    // Lights, and 1 when slot poses are bound.
    lights: vec4<u32>,
}
struct Light {
    position: vec3<f32>,
    range: f32,
    color: vec3<f32>,
    cos_outer: f32,
    direction: vec3<f32>,
    cos_inner: f32,
}
@group(3) @binding(0) var<uniform> clusters: Clusters;
@group(3) @binding(1) var<storage, read> lights: array<Light>;
@group(3) @binding(2) var<storage, read> cluster_counts: array<u32>;
@group(3) @binding(3) var<storage, read> cluster_lights: array<u32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
    @location(6) color: vec4<f32>,
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
    // Linear color and opacity.
    @location(0) color: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) @interpolate(flat) surface: u32,
}

// `VertexOutput` and the unjittered previous clip position: x, y and w.
struct TemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) @interpolate(flat) surface: u32,
    @location(5) previous: vec3<f32>,
}

// Frames with screen effects (decision 021): graded linear color with the
// ambient share of its light in alpha, and the pixel's glow weight and
// reflectivity (decision 025).
struct SceneTargets {
    @location(0) color: vec4<f32>,
    @location(1) glow: vec2<f32>,
}

// The scene targets, and the offset from this pixel to the same surface
// point in the previous frame, in texture coordinates.
struct TemporalTargets {
    @location(0) color: vec4<f32>,
    @location(1) motion: vec2<f32>,
    @location(2) glow: vec2<f32>,
}

fn pose_matrix(pose: AffinePose) -> mat4x4<f32> {
    return transpose(mat4x4<f32>(pose.row_0, pose.row_1, pose.row_2,
        vec4<f32>(0.0, 0.0, 0.0, 1.0)));
}

fn slot_color(slot: SlotColor) -> vec4<f32> {
    return vec4<f32>(unpack2x16unorm(slot.color.x), unpack2x16unorm(slot.color.y));
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let model = mat4x4<f32>(input.model_0, input.model_1, input.model_2, input.model_3);
    return transform_vertex(input.position, input.normal, input.color, input.surface,
        input.uv, model);
}

@vertex
fn vs_affine(input: AffineVertexInput) -> VertexOutput {
    // Transparent draws of temporal frames see moving slots' marks.
    let index = input.slot & ~MOVED_SLOT;
    let model = pose_matrix(poses[index]);
    let slot = colors[index];
    return transform_vertex(input.position, input.normal, slot_color(slot), slot.surface,
        input.uv, model);
}

@vertex
fn vs_affine_temporal(input: AffineVertexInput) -> TemporalOutput {
    let slot = input.slot & ~MOVED_SLOT;
    let model = pose_matrix(poses[slot]);
    let material = colors[slot];
    let vertex = transform_vertex(input.position, input.normal, slot_color(material),
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

fn transform_vertex(position: vec3<f32>, normal: vec3<f32>, color: vec4<f32>, surface: u32,
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

// What a shaded surface point needs: its material, normals and view.
struct SurfacePoint {
    albedo: vec3<f32>,
    // Emitted light as a multiple of the albedo, masked by texture alpha.
    emission: f32,
    roughness: f32,
    alpha: f32,
    f0: vec3<f32>,
    diffuse_color: vec3<f32>,
    normal: vec3<f32>,
    geometric: vec3<f32>,
    view: vec3<f32>,
    n_v: f32,
    world: vec3<f32>,
    // How `world` changes per pixel, for shadow receiver planes.
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    distance: f32,
}

// Shaded light of a surface: graded color, the ambient share of its light
// (for ambient occlusion), its glow weight and its reflectivity for
// screen-space reflections.
struct Shaded {
    color: vec3<f32>,
    ambient: f32,
    glow: f32,
    reflect: f32,
}

// Sampling uses implicit derivatives, so callers build the point in uniform
// control flow; what follows may not be.
fn surface_point(color: vec3<f32>, surface_normal: vec3<f32>, world: vec3<f32>,
        uv: vec2<f32>, packed: u32) -> SurfacePoint {
    // RGBA8UnormSrgb sampling decodes source RGB; alpha masks emission.
    let texel = textureSample(base_color_texture, base_color_sampler, uv);
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    // Roughness, metalness, emission over 16 and ripples.
    let surface = unpack4x8unorm(packed);
    var point: SurfacePoint;
    point.albedo = texel.rgb * color;
    point.emission = surface.z * 16.0 * texel.a;
    point.roughness = max(surface.x, 0.03);
    point.alpha = point.roughness * point.roughness;
    let metalness = surface.y;
    point.f0 = mix(vec3<f32>(0.04), point.albedo, metalness);
    point.diffuse_color = point.albedo * (1.0 - metalness);
    point.geometric = stable_normalize(surface_normal);
    point.normal = point.geometric;
    if surface.w > 0.0 {
        point.normal = rippled(point.normal, world, surface.w);
    }
    let to_eye = camera.position.xyz - world;
    point.distance = length(to_eye);
    point.view = stable_normalize(to_eye);
    point.n_v = max(dot(point.normal, point.view), 1.0e-4);
    point.world = world;
    point.world_dx = world_dx;
    point.world_dy = world_dy;
    return point;
}

// Lit, fogged and graded color of `point` with `local` light added.
fn finish(point: SurfacePoint, local: vec3<f32>) -> Shaded {
    let normal = point.normal;
    let view = point.view;
    let sun = lighting.sun.xyz;
    let n_l = dot(normal, sun);
    let specular = lighting.sun_color.w;
    // Wrapped diffuse softens the terminator of flat low-poly faces.
    let wrapped = max((n_l + 0.25) / 1.25, 0.0);
    var highlight = vec3<f32>(0.0);
    if n_l > 0.0 {
        let half_vector = normalize(view + sun);
        let n_h = max(dot(normal, half_vector), 0.0);
        highlight = fresnel(point.f0, max(dot(view, half_vector), 0.0))
            * (distribution(n_h, point.alpha) * visibility(point.n_v, n_l, point.alpha) * n_l);
    }
    // The sky mirrored in the surface, blurring toward its hemisphere
    // average as the surface roughens, weighted by the environment BRDF.
    let reflected = reflect(-view, normal);
    let mirrored = mix(haze(reflected), sky_average(reflected), point.roughness);
    let brdf = environment_brdf(point.f0, point.roughness, point.n_v);
    let reflection = mirrored * brdf * specular;
    // Rim light outlines walls against the sky; level surfaces seen at
    // grazing angles (distant ground and roofs) take none.
    let grazing = 1.0 - max(dot(point.geometric, view), 0.0);
    let rim = (grazing * grazing) * (grazing * grazing) * lighting.sun.w
        * (1.0 - abs(point.geometric.y));
    let sky_light = point.diffuse_color * (sky_average(normal) * lighting.zenith.w
        + lighting.horizon.rgb * rim) + reflection;
    let sun_light = (point.diffuse_color * wrapped + highlight * specular) * lighting.sun_color.rgb
        * sun_visibility(point.world, point.geometric, point.world_dx, point.world_dy,
            point.distance);
    let emitted = point.albedo * (point.emission * lighting.effects.x);
    let lit = sky_light + sun_light + emitted + local;
    let covered = fog(camera.position.xyz, point.world);
    let fogged = mix(lit, haze(-view), covered);
    var shaded: Shaded;
    shaded.color = grade(fogged);
    shaded.ambient = saturate(luminance(sky_light) / max(luminance(lit), 1.0e-6)) * (1.0 - covered);
    shaded.glow = glow_weight(fogged);
    // Reflectivity for screen-space reflections: the environment BRDF's
    // share, fading out toward the roughness limit and through fog.
    if lighting.effects.w != 0.0 {
        let smooth_enough = saturate(2.0 * (1.0 - point.roughness / lighting.effects.z));
        shaded.reflect = saturate(luminance(brdf) * specular) * smooth_enough * (1.0 - covered);
    }
    return shaded;
}

fn shade(color: vec3<f32>, normal: vec3<f32>, world: vec3<f32>, uv: vec2<f32>,
        packed: u32) -> Shaded {
    return finish(surface_point(color, normal, world, uv, packed), vec3<f32>(0.0));
}

// Light of the local lights of this pixel's cluster at `point`: windowed
// inverse-square falloff, spot cones and the same GGX highlights as the sun,
// fading out over the last tenth of the clustered depth range.
fn local_light(point: SurfacePoint, pixel: vec2<f32>) -> vec3<f32> {
    let depth = dot(clusters.depth.xyz, point.world) + clusters.depth.w;
    let end = clusters.projection.w;
    if depth >= end {
        return vec3<f32>(0.0);
    }
    let counts = clusters.counts;
    let tile = min(vec2<u32>(max(pixel, vec2<f32>(0.0)) / clusters.grid.xy),
        counts.xy - vec2<u32>(1u));
    let slice = u32(clamp(floor(log2(max(depth, 1.0e-6)) * clusters.grid.z + clusters.grid.w),
        0.0, f32(counts.z - 1u)));
    let cluster = (slice * counts.y + tile.y) * counts.x + tile.x;
    let count = min(cluster_counts[cluster], counts.w);
    let specular = lighting.sun_color.w;
    var sum = vec3<f32>(0.0);
    for (var n = 0u; n < count; n++) {
        let light = lights[cluster_lights[cluster * counts.w + n]];
        let to_light = light.position - point.world;
        let d2 = dot(to_light, to_light);
        let r2 = light.range * light.range;
        if d2 >= r2 {
            continue;
        }
        let l = to_light * inverseSqrt(max(d2, 1.0e-8));
        let n_l = dot(point.normal, l);
        if n_l <= 0.0 {
            continue;
        }
        let ratio = d2 / r2;
        let window = saturate(1.0 - ratio * ratio);
        var falloff = window * window / max(d2, 0.01);
        if light.cos_outer > -1.5 {
            falloff *= smoothstep(light.cos_outer, light.cos_inner, dot(-l, light.direction));
        }
        let half_vector = normalize(point.view + l);
        let n_h = max(dot(point.normal, half_vector), 0.0);
        let highlight = fresnel(point.f0, max(dot(point.view, half_vector), 0.0))
            * (distribution(n_h, point.alpha) * visibility(point.n_v, n_l, point.alpha));
        sum += (point.diffuse_color + highlight * specular) * light.color * (falloff * n_l);
    }
    return sum * saturate((end - depth) / (0.1 * end));
}

fn shade_lit(color: vec3<f32>, normal: vec3<f32>, world: vec3<f32>, uv: vec2<f32>,
        packed: u32, pixel: vec2<f32>) -> Shaded {
    let point = surface_point(color, normal, world, uv, packed);
    return finish(point, local_light(point, pixel));
}

fn scene_targets(shaded: Shaded) -> SceneTargets {
    var targets: SceneTargets;
    targets.color = vec4<f32>(shaded.color, shaded.ambient);
    targets.glow = vec2<f32>(shaded.glow, shaded.reflect);
    return targets;
}

fn temporal_targets(shaded: Shaded, input: TemporalOutput) -> TemporalTargets {
    var targets: TemporalTargets;
    targets.color = vec4<f32>(shaded.color, shaded.ambient);
    targets.glow = vec2<f32>(shaded.glow, shaded.reflect);
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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let shaded = shade(input.color.rgb, input.normal, input.world, input.uv, input.surface);
    return vec4<f32>(encoded(shaded.color), 1.0);
}

@fragment
fn fs_main_lit(input: VertexOutput) -> @location(0) vec4<f32> {
    let shaded = shade_lit(input.color.rgb, input.normal, input.world, input.uv,
        input.surface, input.clip_position.xy);
    return vec4<f32>(encoded(shaded.color), 1.0);
}

// The scene target is sRGB: it stores linear color with output encoding.
@fragment
fn fs_post(input: VertexOutput) -> SceneTargets {
    return scene_targets(shade(input.color.rgb, input.normal, input.world, input.uv,
        input.surface));
}

@fragment
fn fs_post_lit(input: VertexOutput) -> SceneTargets {
    return scene_targets(shade_lit(input.color.rgb, input.normal, input.world, input.uv,
        input.surface, input.clip_position.xy));
}

@fragment
fn fs_temporal(input: TemporalOutput) -> TemporalTargets {
    return temporal_targets(shade(input.color.rgb, input.normal, input.world, input.uv,
        input.surface), input);
}

@fragment
fn fs_temporal_lit(input: TemporalOutput) -> TemporalTargets {
    return temporal_targets(shade_lit(input.color.rgb, input.normal, input.world, input.uv,
        input.surface, input.clip_position.xy), input);
}

@fragment
fn fs_transparent(input: VertexOutput) -> TransparentTargets {
    let shaded = shade(input.color.rgb, input.normal, input.world, input.uv, input.surface);
    return weighted(shaded.color, input.color.a, length(camera.position.xyz - input.world));
}

@fragment
fn fs_transparent_lit(input: VertexOutput) -> TransparentTargets {
    let shaded = shade_lit(input.color.rgb, input.normal, input.world, input.uv,
        input.surface, input.clip_position.xy);
    return weighted(shaded.color, input.color.a, length(camera.position.xyz - input.world));
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
    return vec4<f32>(encoded(grade(sky(sky_ray(input.ndc)))), 1.0);
}

// Screen-effect frames: the sky has no ambient occlusion; its sun glows.
@fragment
fn fs_sky_post(input: SkyOutput) -> SceneTargets {
    let light = sky(sky_ray(input.ndc));
    var targets: SceneTargets;
    targets.color = vec4<f32>(grade(light), 0.0);
    targets.glow = vec2<f32>(glow_weight(light), 0.0);
    return targets;
}

// Temporal frames: the unjittered sky reprojects by the camera's turn alone.
@fragment
fn fs_sky_temporal(input: SkyOutput) -> TemporalTargets {
    let ray = sky_ray(input.ndc);
    let light = sky(ray);
    var targets: TemporalTargets;
    targets.color = vec4<f32>(grade(light), 0.0);
    targets.glow = vec2<f32>(glow_weight(light), 0.0);
    let previous = camera.previous_view_projection * vec4<f32>(ray, 0.0);
    targets.motion = (previous.xy / previous.w - input.ndc) * vec2<f32>(0.5, -0.5);
    if !(previous.w > 0.0) {
        targets.motion = vec2<f32>(2.0);
    }
    return targets;
}
