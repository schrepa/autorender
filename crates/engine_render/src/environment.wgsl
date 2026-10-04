// Environment maps (decision 043), prepended with scene.wgsl: the sky drawn
// into the faces of a radiance cube, its levels averaged down, and each
// level of an environment map blurred from them by the GGX lobe of its
// roughness, which surfaces read their reflections from.

// The level of the radiance cube a downsampling pass reads.
@group(0) @binding(20) var level_source: texture_cube<f32>;
// Every level of the radiance cube, which the blur reads.
@group(0) @binding(21) var radiance: texture_cube<f32>;
@group(0) @binding(22) var cube_sampler: sampler;

// GGX samples per texel of a blurred level.
const PREFILTER_SAMPLES: u32 = 64u;

// Three faces of one level: a triangle over them, and the first face (the
// instance index modulo 6; +X, -X, +Y, -Y, +Z, -Z) of which level (divided
// by 6).
struct FaceOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
    @location(1) @interpolate(flat) item: u32,
}

// The three faces a pass draws, from its first: three 8-byte targets fit the
// 32 bytes a sample every device renders.
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

// The direction through `ndc` of cube face `face`, in the face orientation
// WebGPU samples cubes with (texture rows run down, against NDC y).
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

// The sky's light (without the sun's disc, whose highlights surfaces
// shade themselves) in each direction of the radiance cube's first level.
fn face_sky(face: u32, ndc: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(haze(face_direction(face, ndc)), 1.0);
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

// A radiance level from the one above: each texel's direction meets the
// corner of the four texels it covers there, which filtering averages.
fn down(face: u32, ndc: vec2<f32>) -> vec4<f32> {
    let direction = face_direction(face, ndc);
    return vec4<f32>(textureSampleLevel(level_source, cube_sampler, direction, 0.0).rgb, 1.0);
}

@fragment
fn fs_environment_down(input: FaceOutput) -> Faces {
    let face = input.item % 6u;
    return Faces(down(face, input.ndc), down(face + 1u, input.ndc), down(face + 2u, input.ndc));
}

// A point of the Hammersley set of `count`.
fn hammersley(index: u32, count: u32) -> vec2<f32> {
    return vec2<f32>(f32(index) / f32(count), f32(reverseBits(index)) * 2.3283064365386963e-10);
}

// A microfacet normal around `normal` drawn from the GGX distribution of
// `alpha` at `xi`.
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

// A level of an environment map: the radiance around each direction over
// the GGX lobe of the level's roughness (level over the last level), seen
// along that direction (the split sum's assumption), importance sampled and
// read from the radiance level whose texels match each sample's share of
// the lobe (Karis's filtered importance sampling); the first level copies.
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
            // With the view along the normal, the sample's density is D / 4.
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
