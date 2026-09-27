// Uniform: three column-major mat4 and two vec4, 16-byte alignment, exactly
// 224 bytes. `view_projection` is the rasterized projection; temporally
// antialiased frames (decision 017) jitter it by `jitter.xy` clip units per w.
// Their motion uses the previous frame's unjittered `previous_view_projection`
// and `jitter.zw`, two over the viewport size in pixels. The unjittered
// inverse turns pixels into sky rays; `position.xyz` is the eye.
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
    // w: specular strength.
    sun_color: vec4<f32>,
    // w: ambient scale.
    zenith: vec4<f32>,
    // w: exposure.
    horizon: vec4<f32>,
    // w: saturation.
    ground: vec4<f32>,
    // Start, density, height falloff and maximum.
    fog: vec4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lighting: Lighting;
@group(1) @binding(0) var base_color_texture: texture_2d<f32>;
@group(1) @binding(1) var base_color_sampler: sampler;

// Stable engine-owned slots; three aligned rows give an exact 48-byte affine pose.
struct AffinePose {
    row_0: vec4<f32>,
    row_1: vec4<f32>,
    row_2: vec4<f32>,
}
@group(2) @binding(0) var<storage, read> poses: array<AffinePose>;
@group(2) @binding(1) var<storage, read> colors: array<vec4<f32>>;
// Temporally antialiased frames: each slot's pose in the previous frame.
@group(2) @binding(2) var<storage, read> previous_poses: array<AffinePose>;
// Temporal visible IDs mark moving slots; others kept their previous pose.
const MOVED_SLOT: u32 = 0x80000000u;

// sRGB targets apply output encoding themselves; UNORM canvas targets do not.
override SURFACE_IS_SRGB: bool = false;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
    @location(6) color: vec4<f32>,
    @location(7) uv: vec2<f32>,
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
    @location(0) color: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world: vec3<f32>,
    @location(3) uv: vec2<f32>,
}

// `VertexOutput` and the unjittered previous clip position: x, y and w.
struct TemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) previous: vec3<f32>,
}

// Linear color, and the offset from this pixel to the same surface point in
// the previous frame, in texture coordinates.
struct TemporalTargets {
    @location(0) color: vec4<f32>,
    @location(1) motion: vec2<f32>,
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
    return transform_vertex(input.position, input.normal, input.color, input.uv, model);
}

@vertex
fn vs_affine(input: AffineVertexInput) -> VertexOutput {
    let model = pose_matrix(poses[input.slot]);
    return transform_vertex(input.position, input.normal, colors[input.slot], input.uv, model);
}

@vertex
fn vs_affine_temporal(input: AffineVertexInput) -> TemporalOutput {
    let slot = input.slot & ~MOVED_SLOT;
    let model = pose_matrix(poses[slot]);
    let vertex = transform_vertex(input.position, input.normal, colors[slot], input.uv, model);
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
    output.previous = previous.xyw;
    return output;
}

fn transform_vertex(position: vec3<f32>, normal: vec3<f32>, color: vec4<f32>,
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
    return output;
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let low = color * 12.92;
    let high = 1.055 * pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, color <= vec3<f32>(0.0031308));
}

// The sky's light in `direction` without the sun's disc: horizon to zenith
// above; below, haze darkening only partway toward the ground, as distant
// ground seen through air does; and a wide halo toward the sun. Fog takes
// this color.
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

// Sky light reaching a surface facing `normal`: the zenith from above, half
// horizon and half zenith from the side (cooler than the sun, so shade
// contrasts with light) and the ground from below.
fn ambient(normal: vec3<f32>) -> vec3<f32> {
    let side = mix(lighting.horizon.rgb, lighting.zenith.rgb, 0.5);
    let above = mix(side, lighting.zenith.rgb, max(normal.y, 0.0));
    return mix(above, lighting.ground.rgb, max(-normal.y, 0.0)) * lighting.zenith.w;
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

// Lit, fogged and graded color. Sampling uses implicit derivatives, so
// callers shade in uniform control flow.
fn shade(color: vec4<f32>, surface_normal: vec3<f32>, world: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let normal = stable_normalize(surface_normal);
    // RGBA8UnormSrgb sampling decodes source RGB; material factors are linear.
    let albedo = textureSample(base_color_texture, base_color_sampler, uv).rgb * color.rgb;
    let eye = camera.position.xyz;
    let view = stable_normalize(eye - world);
    let sun = lighting.sun.xyz;
    // Wrapped diffuse softens the terminator of flat low-poly faces.
    let facing = dot(normal, sun);
    let diffuse = max((facing + 0.25) / 1.25, 0.0);
    let highlight = pow(max(dot(normal, normalize(view + sun)), 0.0), 48.0)
        * lighting.sun_color.w * step(0.0, facing);
    // Rim light outlines walls against the sky; level surfaces seen at
    // grazing angles (distant ground and roofs) take none.
    let grazing = 1.0 - max(dot(normal, view), 0.0);
    let rim = (grazing * grazing) * (grazing * grazing) * lighting.sun.w
        * (1.0 - abs(normal.y));
    let lit = albedo * (ambient(normal) + lighting.sun_color.rgb * diffuse
        + lighting.horizon.rgb * rim) + lighting.sun_color.rgb * highlight;
    return grade(mix(lit, haze(-view), fog(eye, world)));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = shade(input.color, input.normal, input.world, input.uv);
    if SURFACE_IS_SRGB {
        return vec4<f32>(color, 1.0);
    }
    return vec4<f32>(linear_to_srgb(color), 1.0);
}

// The scene target is sRGB: it stores linear color with output encoding.
@fragment
fn fs_temporal(input: TemporalOutput) -> TemporalTargets {
    var targets: TemporalTargets;
    targets.color = vec4<f32>(shade(input.color, input.normal, input.world, input.uv), 1.0);
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

// Temporal frames: the unjittered sky reprojects by the camera's turn alone.
@fragment
fn fs_sky_temporal(input: SkyOutput) -> TemporalTargets {
    let ray = sky_ray(input.ndc);
    var targets: TemporalTargets;
    targets.color = vec4<f32>(grade(sky(ray)), 1.0);
    let previous = camera.previous_view_projection * vec4<f32>(ray, 0.0);
    targets.motion = (previous.xy / previous.w - input.ndc) * vec2<f32>(0.5, -0.5);
    if !(previous.w > 0.0) {
        targets.motion = vec2<f32>(2.0);
    }
    return targets;
}
