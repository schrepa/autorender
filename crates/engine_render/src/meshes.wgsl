// Uniform: column-major mat4, 16-byte alignment, exactly 64 bytes.
struct Camera {
    view_projection: mat4x4<f32>,
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

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let model = mat4x4<f32>(input.model_0, input.model_1, input.model_2, input.model_3);
    return transform_vertex(input.position, input.normal, input.color, input.uv, model);
}

@vertex
fn vs_affine(input: AffineVertexInput) -> VertexOutput {
    let pose = poses[input.slot];
    let model = transpose(mat4x4<f32>(pose.row_0, pose.row_1, pose.row_2,
        vec4<f32>(0.0, 0.0, 0.0, 1.0)));
    return transform_vertex(input.position, input.normal, colors[input.slot], input.uv, model);
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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let normal = stable_normalize(input.normal);
    let sun = max(dot(normal, normalize(vec3<f32>(-0.5, 0.85, 0.35))), 0.0);
    let sky = normal.y * 0.5 + 0.5;
    let ambient = mix(vec3<f32>(0.12, 0.14, 0.18), vec3<f32>(0.30, 0.38, 0.46), sky);
    // RGBA8UnormSrgb sampling decodes source RGB; material factors are linear.
    let albedo = textureSample(base_color_texture, base_color_sampler, input.uv).rgb * input.color.rgb;
    let lit = albedo * (ambient + vec3<f32>(1.0, 0.91, 0.76) * sun * 0.85);
    let fog = 1.0 - exp(-max(input.distance - 35.0, 0.0) * 0.006);
    let color = mix(lit, vec3<f32>(0.009, 0.015, 0.025), min(fog, 0.85));
    if SURFACE_IS_SRGB {
        return vec4<f32>(color, 1.0);
    }
    return vec4<f32>(linear_to_srgb(color), 1.0);
}
