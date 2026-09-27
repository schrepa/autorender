// Uniform: two column-major mat4 and a vec4, 16-byte alignment, exactly 144
// bytes. `view_projection` is the rasterized projection; temporally
// antialiased frames (decision 017) jitter it by `jitter.xy` clip units per w.
// Their motion uses the previous frame's unjittered `previous_view_projection`
// and `jitter.zw`, two over the viewport size in pixels.
struct Camera {
    view_projection: mat4x4<f32>,
    previous_view_projection: mat4x4<f32>,
    jitter: vec4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;
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
    @location(2) distance: f32,
    @location(3) uv: vec2<f32>,
}

// `VertexOutput` and the unjittered previous clip position: x, y and w.
struct TemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) distance: f32,
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
    output.distance = vertex.distance;
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
    output.distance = output.clip_position.w;
    output.uv = uv;
    return output;
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let low = color * 12.92;
    let high = 1.055 * pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, color <= vec3<f32>(0.0031308));
}

// Lit, fogged linear color. Sampling uses implicit derivatives, so callers
// shade in uniform control flow.
fn shade(color: vec4<f32>, surface_normal: vec3<f32>, distance: f32, uv: vec2<f32>) -> vec3<f32> {
    let normal = stable_normalize(surface_normal);
    let sun = max(dot(normal, normalize(vec3<f32>(-0.5, 0.85, 0.35))), 0.0);
    let sky = normal.y * 0.5 + 0.5;
    let ambient = mix(vec3<f32>(0.12, 0.14, 0.18), vec3<f32>(0.30, 0.38, 0.46), sky);
    // RGBA8UnormSrgb sampling decodes source RGB; material factors are linear.
    let albedo = textureSample(base_color_texture, base_color_sampler, uv).rgb * color.rgb;
    let lit = albedo * (ambient + vec3<f32>(1.0, 0.91, 0.76) * sun * 0.85);
    let fog = 1.0 - exp(-max(distance - 35.0, 0.0) * 0.006);
    return mix(lit, vec3<f32>(0.009, 0.015, 0.025), min(fog, 0.85));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = shade(input.color, input.normal, input.distance, input.uv);
    if SURFACE_IS_SRGB {
        return vec4<f32>(color, 1.0);
    }
    return vec4<f32>(linear_to_srgb(color), 1.0);
}

// The scene target is sRGB: it stores linear color with output encoding.
@fragment
fn fs_temporal(input: TemporalOutput) -> TemporalTargets {
    var targets: TemporalTargets;
    targets.color = vec4<f32>(shade(input.color, input.normal, input.distance, input.uv), 1.0);
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
