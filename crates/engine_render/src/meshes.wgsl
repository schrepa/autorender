// Mesh drawing, prepended with scene.wgsl. Opaque entry points write the
// canvas, the screen effects' scene targets (decision 021) or temporal
// targets (decision 017); `_lit` variants add clustered local lights
// (decision 023); transparent ones accumulate weighted blended transparency
// (decision 026). Unlit entry points never touch the lights' bindings.

// Material maps (decision 031): glTF base color, metallic-roughness, normal,
// occlusion and emissive textures, each with its sampler, and the material's
// map parameters. Materials without maps bind white textures.
@group(1) @binding(0) var base_color_texture: texture_2d<f32>;
@group(1) @binding(1) var base_color_sampler: sampler;
@group(1) @binding(2) var metallic_roughness_texture: texture_2d<f32>;
@group(1) @binding(3) var metallic_roughness_sampler: sampler;
@group(1) @binding(4) var normal_texture: texture_2d<f32>;
@group(1) @binding(5) var normal_sampler: sampler;
@group(1) @binding(6) var occlusion_texture: texture_2d<f32>;
@group(1) @binding(7) var occlusion_sampler: sampler;
@group(1) @binding(8) var emissive_texture: texture_2d<f32>;
@group(1) @binding(9) var emissive_sampler: sampler;
struct MaterialMaps {
    // Rows of each map's texture coordinate transform, in map order.
    transforms: array<vec4<f32>, 10>,
    // Linear emissive color.
    emissive: vec4<f32>,
    // Normal scale, occlusion strength and alpha cutoff.
    params: vec4<f32>,
}
@group(1) @binding(10) var<uniform> material_maps: MaterialMaps;

// The material's pipeline variant (decision 031). Without MATERIAL_MAPS only
// the base color texture is read, untransformed, and its alpha masks
// emission alone; the other flags need it except DOUBLE_SIDED.
override MATERIAL_MAPS: bool = false;
override NORMAL_MAP: bool = false;
override ALPHA_MASK: bool = false;
override DOUBLE_SIDED: bool = false;

// Stable engine-owned slots; three aligned rows give an exact 48-byte affine pose.
struct AffinePose {
    row_0: vec4<f32>,
    row_1: vec4<f32>,
    row_2: vec4<f32>,
}
// A slot's linear color and opacity as four 16-bit fractions (red and green,
// then blue and opacity; decision 026), its packed surface (decision 019)
// and flags: one more than the offset of a skinned slot's joint palette
// (decision 036), 0 for rigid slots.
struct SlotColor {
    color: vec2<u32>,
    surface: u32,
    flags: u32,
}
@group(2) @binding(0) var<storage, read> poses: array<AffinePose>;
@group(2) @binding(1) var<storage, read> colors: array<SlotColor>;
// Temporally antialiased frames: each slot's pose in the previous frame.
@group(2) @binding(2) var<storage, read> previous_poses: array<AffinePose>;
// Joint palettes (decision 036): each joint's displayed world matrix times
// its inverse bind matrix; temporally antialiased frames also bind the
// previous frame's.
@group(2) @binding(3) var<storage, read> palette: array<AffinePose>;
@group(2) @binding(4) var<storage, read> previous_palette: array<AffinePose>;
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

// Mesh vertices carry octahedral normals (`octahedral_normal`) in 24 bytes.
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec2<f32>,
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
    // Four 16-bit color fractions, as slots store them.
    @location(6) color: vec2<u32>,
    @location(7) uv: vec2<f32>,
    @location(8) surface: u32,
}

// Visibility and batching reorder only this four-byte slot ID stream.
struct AffineVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec2<f32>,
    @location(2) slot: u32,
    @location(7) uv: vec2<f32>,
}

// Four output slots: vertex outputs bound geometry throughput, so texture
// coordinates ride in the w lanes and the material stays packed and flat.
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // World normal and texture u.
    @location(0) normal_u: vec4<f32>,
    // World position and texture v.
    @location(1) world_v: vec4<f32>,
    // Linear color and opacity as four 16-bit fractions, and the packed
    // surface (decision 019).
    @location(2) @interpolate(flat) material: vec3<u32>,
}

// `VertexOutput` and the unjittered previous clip position: x, y and w.
struct TemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec3<u32>,
    @location(3) previous: vec3<f32>,
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

// Linear color and opacity of a vertex output's packed material.
fn output_color(material: vec3<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16unorm(material.x), unpack2x16unorm(material.y));
}

// A normal of `gpu_data::octahedral`: unfolded from the octahedron, not unit
// length (the transform normalizes).
fn octahedral_normal(encoded: vec2<f32>) -> vec3<f32> {
    let z = 1.0 - abs(encoded.x) - abs(encoded.y);
    let fold = max(-z, 0.0);
    let shift = vec2<f32>(select(-fold, fold, encoded.x >= 0.0),
        select(-fold, fold, encoded.y >= 0.0));
    return vec3<f32>(encoded - shift, z);
}

fn vertex_main(input: VertexInput) -> VertexOutput {
    let model = mat4x4<f32>(input.model_0, input.model_1, input.model_2, input.model_3);
    return transform_vertex(input.position, input.normal, input.color, input.surface,
        input.uv, model);
}

fn vertex_affine(input: AffineVertexInput) -> VertexOutput {
    // Transparent draws of temporal frames see moving slots' marks.
    let index = input.slot & ~MOVED_SLOT;
    let model = pose_matrix(poses[index]);
    let slot = colors[index];
    return transform_vertex(input.position, input.normal, slot.color, slot.surface,
        input.uv, model);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    return vertex_main(input);
}

@vertex
fn vs_affine(input: AffineVertexInput) -> VertexOutput {
    return vertex_affine(input);
}

@vertex
fn vs_affine_temporal(input: AffineVertexInput) -> TemporalOutput {
    return vertex_affine_temporal(input);
}

// Meshes with vertex colors or second texture coordinates (decision 034)
// read them from a second vertex stream, and their outputs carry both.
struct ExtendedOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec3<u32>,
    // Linear vertex color and opacity, and second texture coordinates.
    @location(3) color: vec4<f32>,
    @location(4) uv1: vec2<f32>,
}

struct ExtendedTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec3<u32>,
    @location(3) previous: vec3<f32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
}

fn extended_output(vertex: VertexOutput, color: vec4<f32>, uv1: vec2<f32>) -> ExtendedOutput {
    var output: ExtendedOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.world_v = vertex.world_v;
    output.material = vertex.material;
    output.color = color;
    output.uv1 = uv1;
    return output;
}

@vertex
fn vs_main_extended(input: VertexInput, @location(9) color: vec4<f32>,
        @location(10) uv1: vec2<f32>) -> ExtendedOutput {
    return extended_output(vertex_main(input), color, uv1);
}

@vertex
fn vs_affine_extended(input: AffineVertexInput, @location(9) color: vec4<f32>,
        @location(10) uv1: vec2<f32>) -> ExtendedOutput {
    return extended_output(vertex_affine(input), color, uv1);
}

@vertex
fn vs_affine_temporal_extended(input: AffineVertexInput, @location(9) color: vec4<f32>,
        @location(10) uv1: vec2<f32>) -> ExtendedTemporalOutput {
    let vertex = vertex_affine_temporal(input);
    var output: ExtendedTemporalOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.world_v = vertex.world_v;
    output.material = vertex.material;
    output.previous = vertex.previous;
    output.color = color;
    output.uv1 = uv1;
    return output;
}

// Linear blend skinning (decision 036): the joints' palette matrices
// weighted by the vertex's renormalized 16-bit weights.
fn blend_poses(a: AffinePose, b: AffinePose, c: AffinePose, d: AffinePose,
        weights: vec4<f32>) -> mat4x4<f32> {
    let w = weights / max(dot(weights, vec4<f32>(1.0)), 1.0e-6);
    return pose_matrix(AffinePose(
        a.row_0 * w.x + b.row_0 * w.y + c.row_0 * w.z + d.row_0 * w.w,
        a.row_1 * w.x + b.row_1 * w.y + c.row_1 * w.z + d.row_1 * w.w,
        a.row_2 * w.x + b.row_2 * w.y + c.row_2 * w.z + d.row_2 * w.w,
    ));
}

// A skinned slot's matrix for a vertex: `flags` is one more than the slot's
// palette offset.
fn skin_matrix(flags: u32, joints: vec4<u32>, weights: vec4<f32>) -> mat4x4<f32> {
    let base = flags - 1u;
    return blend_poses(palette[base + joints.x], palette[base + joints.y],
        palette[base + joints.z], palette[base + joints.w], weights);
}

fn previous_skin_matrix(flags: u32, joints: vec4<u32>, weights: vec4<f32>) -> mat4x4<f32> {
    let base = flags - 1u;
    return blend_poses(previous_palette[base + joints.x], previous_palette[base + joints.y],
        previous_palette[base + joints.z], previous_palette[base + joints.w], weights);
}

// Meshes with joints drawn through retained slots: skinned slots move each
// vertex by their palette, rigid slots drawing the same mesh by their pose.
fn vertex_affine_skinned(input: AffineVertexInput, joints: vec4<u32>,
        weights: vec4<f32>) -> VertexOutput {
    let index = input.slot & ~MOVED_SLOT;
    let slot = colors[index];
    var model = pose_matrix(poses[index]);
    if slot.flags != 0u {
        model = skin_matrix(slot.flags, joints, weights);
    }
    return transform_vertex(input.position, input.normal, slot.color, slot.surface,
        input.uv, model);
}

@vertex
fn vs_affine_skinned(input: AffineVertexInput, @location(9) color: vec4<f32>,
        @location(10) uv1: vec2<f32>, @location(11) joints: vec4<u32>,
        @location(12) weights: vec4<f32>) -> ExtendedOutput {
    return extended_output(vertex_affine_skinned(input, joints, weights), color, uv1);
}

// Skinned slots' previous positions come from the previous palette, which
// is this frame's when motion is invalid.
@vertex
fn vs_affine_temporal_skinned(input: AffineVertexInput, @location(9) color: vec4<f32>,
        @location(10) uv1: vec2<f32>, @location(11) joints: vec4<u32>,
        @location(12) weights: vec4<f32>) -> ExtendedTemporalOutput {
    let slot = input.slot & ~MOVED_SLOT;
    let material = colors[slot];
    var model = pose_matrix(poses[slot]);
    var previous_model = model;
    if material.flags != 0u {
        model = skin_matrix(material.flags, joints, weights);
        previous_model = previous_skin_matrix(material.flags, joints, weights);
    } else if (input.slot & MOVED_SLOT) != 0u {
        previous_model = pose_matrix(previous_poses[slot]);
    }
    let vertex = transform_vertex(input.position, input.normal, material.color,
        material.surface, input.uv, model);
    let previous = camera.previous_view_projection
        * (previous_model * vec4<f32>(input.position, 1.0));
    var output: ExtendedTemporalOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.world_v = vertex.world_v;
    output.material = vertex.material;
    output.previous = previous.xyw;
    output.color = color;
    output.uv1 = uv1;
    return output;
}

fn vertex_affine_temporal(input: AffineVertexInput) -> TemporalOutput {
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
    output.normal_u = vertex.normal_u;
    output.world_v = vertex.world_v;
    output.material = vertex.material;
    output.previous = previous.xyw;
    return output;
}

fn transform_vertex(position: vec3<f32>, encoded_normal: vec2<f32>, color: vec2<u32>,
        surface: u32, uv: vec2<f32>, model: mat4x4<f32>) -> VertexOutput {
    let world_position = model * vec4<f32>(position, 1.0);
    // A common positive basis scale cancels out of normalized normals.
    // Rescaling to a largest component of 1 keeps the cofactors within
    // [-2, 2] for large or small admitted world transforms, and the final
    // normalization handles their magnitude. CPU admission separately bounds
    // normal-matrix conditioning.
    let inverse_scale = 1.0 / max(maximum_abs(model[0].xyz),
        max(maximum_abs(model[1].xyz), maximum_abs(model[2].xyz)));
    let x = model[0].xyz * inverse_scale;
    let y = model[1].xyz * inverse_scale;
    let z = model[2].xyz * inverse_scale;
    let normal = octahedral_normal(encoded_normal);
    // Cofactor columns times the normal: its inverse-transpose direction.
    let yz = cross(y, z);
    let transformed = yz * normal.x + cross(z, x) * normal.y + cross(x, y) * normal.z;
    var output: VertexOutput;
    output.clip_position = camera.view_projection * world_position;
    output.normal_u = vec4<f32>(stable_normalize(transformed) * sign(dot(x, yz)), uv.x);
    output.world_v = vec4<f32>(world_position.xyz, uv.y);
    output.material = vec3<u32>(color, surface);
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
    // Coverage: color alpha, times base color texture alpha with maps.
    opacity: f32,
    // Emitted light as a multiple of the albedo, masked by texture alpha.
    emission: f32,
    // glTF emission, and how much indirect light reaches the point.
    emissive: vec3<f32>,
    occlusion: f32,
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
// (for ambient occlusion), its glow weight, its reflectivity for
// screen-space reflections and its coverage.
struct Shaded {
    color: vec3<f32>,
    ambient: f32,
    glow: f32,
    reflect: f32,
    opacity: f32,
}

// Texture coordinates of map `index` (base color, metallic-roughness,
// normal, occlusion, emissive) through its transform, from the first or
// second set as the map reads.
fn map_uv(index: u32, uv: vec2<f32>, uv1: vec2<f32>) -> vec2<f32> {
    let second = ((u32(material_maps.params.w) >> index) & 1u) == 1u;
    let point = vec3<f32>(select(uv, uv1, second), 1.0);
    return vec2<f32>(dot(material_maps.transforms[index * 2u].xyz, point),
        dot(material_maps.transforms[index * 2u + 1u].xyz, point));
}

// Normal mapping without stored tangents (decision 031): the tangent frame
// from screen derivatives of the position and of the normal map's texture
// coordinates. glTF normal maps point green up the image, toward decreasing v.
fn mapped_normal(normal: vec3<f32>, dp_dx: vec3<f32>, dp_dy: vec3<f32>, uv_dx: vec2<f32>,
        uv_dy: vec2<f32>, sampled: vec3<f32>, scale: f32) -> vec3<f32> {
    let dp_dy_perp = cross(dp_dy, normal);
    let dp_dx_perp = cross(normal, dp_dx);
    let tangent = dp_dy_perp * uv_dx.x + dp_dx_perp * uv_dy.x;
    let bitangent = dp_dy_perp * uv_dx.y + dp_dx_perp * uv_dy.y;
    let frame = inverseSqrt(max(max(dot(tangent, tangent), dot(bitangent, bitangent)), 1.0e-30));
    let deflection = sampled.xy * (scale * frame);
    return stable_normalize(tangent * deflection.x - bitangent * deflection.y + normal * sampled.z);
}

// Sampling uses implicit derivatives, so callers build the point in uniform
// control flow; what follows may not be. `color` is linear color and
// opacity; `front` whether the surface faces the camera.
fn surface_point(color: vec4<f32>, surface_normal: vec3<f32>, world: vec3<f32>,
        uv: vec2<f32>, uv1: vec2<f32>, packed: u32, front: bool) -> SurfacePoint {
    // RGBA8UnormSrgb sampling decodes source RGB; alpha masks emission and,
    // with maps, is coverage.
    var base_uv = uv;
    if MATERIAL_MAPS {
        base_uv = map_uv(0u, uv, uv1);
    }
    let texel = textureSample(base_color_texture, base_color_sampler, base_uv);
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    // Roughness, metalness, emission over 16 and ripples.
    let surface = unpack4x8unorm(packed);
    var point: SurfacePoint;
    point.albedo = texel.rgb * color.rgb;
    point.opacity = color.a;
    point.emission = surface.z * 16.0 * texel.a;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = 1.0;
    var roughness = surface.x;
    var metalness = surface.y;
    if MATERIAL_MAPS {
        point.opacity *= texel.a;
        // glTF: green is roughness and blue metalness.
        let packed_map = textureSample(metallic_roughness_texture, metallic_roughness_sampler,
            map_uv(1u, uv, uv1));
        roughness *= packed_map.g;
        metalness *= packed_map.b;
        let occluded = textureSample(occlusion_texture, occlusion_sampler,
            map_uv(3u, uv, uv1)).r;
        point.occlusion = 1.0 + material_maps.params.y * (occluded - 1.0);
        point.emissive = material_maps.emissive.rgb
            * textureSample(emissive_texture, emissive_sampler, map_uv(4u, uv, uv1)).rgb;
    }
    var sampled = vec3<f32>(0.0, 0.0, 1.0);
    var normal_dx = vec2<f32>(0.0);
    var normal_dy = vec2<f32>(0.0);
    if NORMAL_MAP {
        let normal_uv = map_uv(2u, uv, uv1);
        sampled = textureSample(normal_texture, normal_sampler, normal_uv).xyz * 2.0 - 1.0;
        normal_dx = dpdx(normal_uv);
        normal_dy = dpdy(normal_uv);
    }
    point.roughness = max(roughness, 0.03);
    point.alpha = point.roughness * point.roughness;
    point.f0 = mix(vec3<f32>(0.04), point.albedo, metalness);
    point.diffuse_color = point.albedo * (1.0 - metalness);
    point.geometric = stable_normalize(surface_normal);
    // Back faces of double-sided materials are lit from their own side.
    if DOUBLE_SIDED && !front {
        point.geometric = -point.geometric;
    }
    point.normal = point.geometric;
    if NORMAL_MAP {
        point.normal = mapped_normal(point.geometric, world_dx, world_dy, normal_dx, normal_dy,
            sampled, material_maps.params.x);
    }
    if surface.w > 0.0 {
        point.normal = rippled(point.normal, world, surface.w);
    }
    if ALPHA_MASK && point.opacity < material_maps.params.z {
        discard;
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
    // Occlusion maps darken indirect light only (glTF).
    let sky_light = (point.diffuse_color * (sky_average(normal) * lighting.zenith.w
        + lighting.horizon.rgb * rim) + reflection) * point.occlusion;
    let sun_light = (point.diffuse_color * wrapped + highlight * specular) * lighting.sun_color.rgb
        * sun_visibility(point.world, point.geometric, point.world_dx, point.world_dy,
            point.distance);
    let emitted = (point.albedo * point.emission + point.emissive) * lighting.effects.x;
    let lit = sky_light + sun_light + emitted + local;
    let covered = fog(camera.position.xyz, point.world);
    let fogged = mix(lit, haze(-view), covered);
    var shaded: Shaded;
    shaded.opacity = point.opacity;
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

// What a fragment shades: linear color and opacity, world normal and
// position, both texture coordinate sets, the packed surface and the pixel.
struct Fragment {
    color: vec4<f32>,
    normal: vec3<f32>,
    world: vec3<f32>,
    uv: vec2<f32>,
    uv1: vec2<f32>,
    surface: u32,
    pixel: vec2<f32>,
}

// Meshes without a second stream read their first coordinates as second.
fn fragment(input: VertexOutput) -> Fragment {
    var values: Fragment;
    values.color = output_color(input.material);
    values.normal = input.normal_u.xyz;
    values.world = input.world_v.xyz;
    values.uv = vec2<f32>(input.normal_u.w, input.world_v.w);
    values.uv1 = values.uv;
    values.surface = input.material.z;
    values.pixel = input.clip_position.xy;
    return values;
}

// Vertex colors multiply the color and opacity (glTF `COLOR_0`).
fn extended_fragment(input: ExtendedOutput) -> Fragment {
    var values: Fragment;
    values.color = output_color(input.material) * input.color;
    values.normal = input.normal_u.xyz;
    values.world = input.world_v.xyz;
    values.uv = vec2<f32>(input.normal_u.w, input.world_v.w);
    values.uv1 = input.uv1;
    values.surface = input.material.z;
    values.pixel = input.clip_position.xy;
    return values;
}

fn shade(input: Fragment, front: bool) -> Shaded {
    return finish(surface_point(input.color, input.normal, input.world, input.uv, input.uv1,
        input.surface, front), vec3<f32>(0.0));
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

fn shade_lit(input: Fragment, front: bool) -> Shaded {
    let point = surface_point(input.color, input.normal, input.world, input.uv, input.uv1,
        input.surface, front);
    return finish(point, local_light(point, input.pixel));
}

fn temporal_output(input: TemporalOutput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = input.clip_position;
    output.normal_u = input.normal_u;
    output.world_v = input.world_v;
    output.material = input.material;
    return output;
}

fn extended_temporal_output(input: ExtendedTemporalOutput) -> ExtendedOutput {
    var output: ExtendedOutput;
    output.clip_position = input.clip_position;
    output.normal_u = input.normal_u;
    output.world_v = input.world_v;
    output.material = input.material;
    output.color = input.color;
    output.uv1 = input.uv1;
    return output;
}

fn scene_targets(shaded: Shaded) -> SceneTargets {
    var targets: SceneTargets;
    targets.color = vec4<f32>(shaded.color, shaded.ambient);
    targets.glow = vec2<f32>(shaded.glow, shaded.reflect);
    return targets;
}

// `clip_position` is the fragment's, `previous` the same point's previous
// clip position: x, y and w.
fn temporal_targets(shaded: Shaded, clip_position: vec4<f32>, previous: vec3<f32>)
        -> TemporalTargets {
    var targets: TemporalTargets;
    targets.color = vec4<f32>(shaded.color, shaded.ambient);
    targets.glow = vec2<f32>(shaded.glow, shaded.reflect);
    // This pixel center without the jitter, and the same point last frame.
    let current = vec2<f32>(clip_position.x * camera.jitter.z - 1.0,
        1.0 - clip_position.y * camera.jitter.w) - camera.jitter.xy;
    targets.motion = (previous.xy / previous.z - current) * vec2<f32>(0.5, -0.5);
    // A point behind the previous camera has no position there: send its
    // history lookup out of the viewport.
    if !(previous.z > 0.0) {
        targets.motion = vec2<f32>(2.0);
    }
    return targets;
}

fn transparent_targets(shaded: Shaded, input: Fragment) -> TransparentTargets {
    return weighted(shaded.color, shaded.opacity, length(camera.position.xyz - input.world));
}

@fragment
fn fs_main(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade(fragment(input), front).color), 1.0);
}

@fragment
fn fs_main_lit(input: VertexOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_lit(fragment(input), front).color), 1.0);
}

// The scene target is sRGB: it stores linear color with output encoding.
@fragment
fn fs_post(input: VertexOutput, @builtin(front_facing) front: bool) -> SceneTargets {
    return scene_targets(shade(fragment(input), front));
}

@fragment
fn fs_post_lit(input: VertexOutput, @builtin(front_facing) front: bool) -> SceneTargets {
    return scene_targets(shade_lit(fragment(input), front));
}

@fragment
fn fs_temporal(input: TemporalOutput, @builtin(front_facing) front: bool) -> TemporalTargets {
    return temporal_targets(shade(fragment(temporal_output(input)), front),
        input.clip_position, input.previous);
}

@fragment
fn fs_temporal_lit(input: TemporalOutput, @builtin(front_facing) front: bool)
        -> TemporalTargets {
    return temporal_targets(shade_lit(fragment(temporal_output(input)), front),
        input.clip_position, input.previous);
}

@fragment
fn fs_transparent(input: VertexOutput, @builtin(front_facing) front: bool)
        -> TransparentTargets {
    let values = fragment(input);
    return transparent_targets(shade(values, front), values);
}

@fragment
fn fs_transparent_lit(input: VertexOutput, @builtin(front_facing) front: bool)
        -> TransparentTargets {
    let values = fragment(input);
    return transparent_targets(shade_lit(values, front), values);
}

// The same passes for meshes with a second vertex stream (decision 034).
@fragment
fn fs_main_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade(extended_fragment(input), front).color), 1.0);
}

@fragment
fn fs_main_lit_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_lit(extended_fragment(input), front).color), 1.0);
}

@fragment
fn fs_post_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> SceneTargets {
    return scene_targets(shade(extended_fragment(input), front));
}

@fragment
fn fs_post_lit_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> SceneTargets {
    return scene_targets(shade_lit(extended_fragment(input), front));
}

@fragment
fn fs_temporal_extended(input: ExtendedTemporalOutput, @builtin(front_facing) front: bool)
        -> TemporalTargets {
    let values = extended_fragment(extended_temporal_output(input));
    return temporal_targets(shade(values, front), input.clip_position, input.previous);
}

@fragment
fn fs_temporal_lit_extended(input: ExtendedTemporalOutput,
        @builtin(front_facing) front: bool) -> TemporalTargets {
    let values = extended_fragment(extended_temporal_output(input));
    return temporal_targets(shade_lit(values, front), input.clip_position, input.previous);
}

@fragment
fn fs_transparent_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> TransparentTargets {
    let values = extended_fragment(input);
    return transparent_targets(shade(values, front), values);
}

@fragment
fn fs_transparent_lit_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> TransparentTargets {
    let values = extended_fragment(input);
    return transparent_targets(shade_lit(values, front), values);
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
