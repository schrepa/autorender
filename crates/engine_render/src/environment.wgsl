import package::frame::view::PI;
import package::lighting::brdf::distribution;
import package::sky::light::{sky_along, sky_domes};

@group(0) @binding(20) var level_source: texture_cube<f32>;
@group(0) @binding(21) var radiance: texture_cube<f32>;
@group(0) @binding(22) var cube_sampler: sampler;

const PREFILTER_SAMPLES: u32 = 64u;

struct FaceOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
    @location(1) @interpolate(flat) item: u32,
}

struct Faces {
    @location(0) first: vec4<f32>,
    @location(1) second: vec4<f32>,
    @location(2) third: vec4<f32>,
}

@vertex
fn vs_environment(@builtin(vertex_index) index: u32, @builtin(instance_index) item: u32)
        -> FaceOutput {
    let ndc = vec2<f32>(f32((index << 1u) & 2u) * 2.0 - 1.0, f32(index & 2u) * 2.0 - 1.0);
    var output: FaceOutput;
    output.clip_position = vec4<f32>(ndc, 0.0, 1.0);
    output.ndc = ndc;
    output.item = item;
    return output;
}

fn face_direction(face: u32, ndc: vec2<f32>) -> vec3<f32> {
    let u = ndc.x;
    let v = -ndc.y;
    var direction: vec3<f32>;
    switch face {
        case 0u: { direction = vec3<f32>(1.0, -v, -u); }
        case 1u: { direction = vec3<f32>(-1.0, -v, u); }
        case 2u: { direction = vec3<f32>(u, 1.0, v); }
        case 3u: { direction = vec3<f32>(u, -1.0, -v); }
        case 4u: { direction = vec3<f32>(u, -v, 1.0); }
        default: { direction = vec3<f32>(-u, -v, -1.0); }
    }
    return normalize(direction);
}

fn face_sky(face: u32, ndc: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(sky_along(sky_domes.eye.xyz, face_direction(face, ndc)), 1.0);
}

@fragment
fn fs_environment_sky(input: FaceOutput) -> Faces {
    let face = input.item % 6u;
    return Faces(
        face_sky(face, input.ndc),
        face_sky(face + 1u, input.ndc),
        face_sky(face + 2u, input.ndc),
    );
}

fn down(face: u32, ndc: vec2<f32>) -> vec4<f32> {
    let direction = face_direction(face, ndc);
    return vec4<f32>(textureSampleLevel(level_source, cube_sampler, direction, 0.0).rgb, 1.0);
}

@fragment
fn fs_environment_down(input: FaceOutput) -> Faces {
    let face = input.item % 6u;
    return Faces(down(face, input.ndc), down(face + 1u, input.ndc), down(face + 2u, input.ndc));
}

fn hammersley(index: u32, count: u32) -> vec2<f32> {
    return vec2<f32>(f32(index) / f32(count), f32(reverseBits(index)) * 2.3283064365386963e-10);
}

fn ggx_normal(xi: vec2<f32>, normal: vec3<f32>, alpha: f32) -> vec3<f32> {
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.z) < 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return normalize(tangent * (sin_theta * cos(phi)) + bitangent * (sin_theta * sin(phi))
        + normal * cos_theta);
}

fn prefiltered(face: u32, level: u32, ndc: vec2<f32>) -> vec4<f32> {
    let normal = face_direction(face, ndc);
    if level == 0u {
        return vec4<f32>(textureSampleLevel(radiance, cube_sampler, normal, 0.0).rgb, 1.0);
    }
    let levels = textureNumLevels(radiance);
    let roughness = f32(level) / f32(levels - 1u);
    let alpha = roughness * roughness;
    let size = f32(textureDimensions(radiance).x);
    let texel_angle = 4.0 * PI / (6.0 * size * size);
    let deepest = f32(levels - 1u);
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var index = 0u; index < PREFILTER_SAMPLES; index++) {
        let half_vector = ggx_normal(hammersley(index, PREFILTER_SAMPLES), normal, alpha);
        let light = 2.0 * dot(normal, half_vector) * half_vector - normal;
        let n_l = dot(normal, light);
        if n_l > 0.0 {
            let density = distribution(max(dot(normal, half_vector), 0.0), alpha) * 0.25;
            let sample_angle = 1.0 / (f32(PREFILTER_SAMPLES) * density + 1.0e-6);
            let lod = clamp(0.5 * log2(sample_angle / texel_angle) + 1.0, 0.0, deepest);
            sum += textureSampleLevel(radiance, cube_sampler, light, lod).rgb * n_l;
            weight += n_l;
        }
    }
    return vec4<f32>(sum / max(weight, 1.0e-6), 1.0);
}

@fragment
fn fs_environment_prefilter(input: FaceOutput) -> Faces {
    let face = input.item % 6u;
    let level = input.item / 6u;
    return Faces(
        prefiltered(face, level, input.ndc),
        prefiltered(face + 1u, level, input.ndc),
        prefiltered(face + 2u, level, input.ndc),
    );
}
