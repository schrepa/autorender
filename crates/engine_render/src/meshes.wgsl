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
    // Transmission (decision 038): its share, index of refraction and
    // thickness.
    volume: vec4<f32>,
}
@group(1) @binding(10) var<uniform> material_maps: MaterialMaps;

// The material's pipeline variant (decision 031). Without MATERIAL_MAPS only
// the base color texture is read, untransformed, and its alpha masks
// emission alone; the other flags need it except DOUBLE_SIDED.
override MATERIAL_MAPS: bool = false;
override NORMAL_MAP: bool = false;
override ALPHA_MASK: bool = false;
override DOUBLE_SIDED: bool = false;
// Maps the material has: absent ones read as the white texel they would
// sample. A packed occlusion map is the metallic-roughness texture, read in
// the same sample's red.
override METALLIC_ROUGHNESS_MAP: bool = true;
override OCCLUSION_MAP: bool = true;
override EMISSIVE_MAP: bool = true;
override PACKED_OCCLUSION: bool = false;
// Transmissive materials shade as glass in the transparent pass (decision
// 038).
override TRANSMISSION: bool = false;

// What glass refracts (decision 038): the frame's opaque color while
// transparent surfaces draw with refraction, a white texel otherwise.
@group(0) @binding(8) var refraction_source: texture_2d<f32>;
@group(0) @binding(9) var refraction_sampler: sampler;
// Environment maps (decision 043): cubes of the light around, blurred down
// their levels by roughness; the first holds the sky.
@group(0) @binding(10) var environment_maps: texture_cube_array<f32>;
@group(0) @binding(11) var environment_sampler: sampler;
// Reflection probes (decision 043) whose environment maps surfaces reflect,
// found through a grid of cells over the ground plane listing the probes
// whose boxes reach each cell. In vec4 words: [0] the grid's first corner
// (world x and z) and cell size (bits of f32) and the probe count; [1] the
// cells across x and z and where the cells and their lists start; then a
// word per cell (the first entry and the length of its list); then the
// lists, smallest boxes first, two words an entry: the probe's center and
// blend distance (bits of f32), and its extent (bits of f32) and cube in the
// environment maps.
@group(0) @binding(12) var<storage, read> probe_words: array<vec4<u32>>;

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

// Virtual geometry (decision 044): the arena `virtual_geometry.wgsl` selects
// clusters from. Each mesh's bases (the last where the pool's page slots
// start) and quantization; three words per cluster (the second ends with
// its mesh, the third holds its vertex and triangle words in its page and
// ends with its global page); the page pool: each page's slot, then the
// slots of VIRTUAL_PAGE_WORDS words.
struct VirtualMeshRecord {
    bases: vec4<u32>,
    info: vec4<u32>,
    position_min: vec4<f32>,
    position_step: vec4<f32>,
    uv: vec4<f32>,
}
@group(2) @binding(5) var<storage, read> virtual_meshes: array<VirtualMeshRecord>;
@group(2) @binding(6) var<storage, read> virtual_clusters: array<vec4<u32>>;
@group(2) @binding(7) var<storage, read> virtual_pages: array<u32>;
const VIRTUAL_PAGE_WORDS: u32 = 32768u;

// The visibility buffer (decision 045): each pixel of virtual geometry
// holds its selected cluster's entry in `visible_clusters` plus one (0: no
// cluster) and its triangle | bin << 7; each material bin's resolve then
// shades the pixels of its 8x8 tiles. A draw's uniform: the bin's first
// entry this phase, the bin, where its tile list starts and the tiles
// across the viewport.
struct VisibilityDraw {
    visible_base: u32,
    bin: u32,
    tile_base: u32,
    tiles_across: u32,
}
@group(2) @binding(8) var<uniform> visibility_draw: VisibilityDraw;
// Visible ID and cluster of each selected cluster, by bin.
@group(2) @binding(9) var<storage, read> visible_clusters: array<vec2<u32>>;
@group(2) @binding(10) var visibility_ids: texture_2d<u32>;
@group(2) @binding(11) var<storage, read> resolve_tiles: array<u32>;

// Local lights (decision 023), binned into view clusters by lights.wgsl.
struct Clusters {
    view_projection: mat4x4<f32>,
    // Depth along the view of point p: dot(xyz, p) + w.
    depth: vec4<f32>,
    // x and y projection scales, clustered depth range start and end.
    projection: vec4<f32>,
    // Tile width and height in pixels; slice = log2(depth) * z + w.
    grid: vec4<f32>,
    // Tiles across and down, depth slices, and the per-cluster budget.
    counts: vec4<u32>,
    // Lights, 1 when slot poses are bound, and the listed entries' capacity.
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
// Per cluster: the first entry of its list in `cluster_lights` and its
// length.
@group(3) @binding(2) var<storage, read> cluster_ranges: array<vec2<u32>>;
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
// ambient share of its light in alpha, the pixel's glow weight and
// reflectivity (decision 025), and the surface reflections follow: its
// shading normal, octahedral in [0, 1], and roughness.
struct SceneTargets {
    @location(0) color: vec4<f32>,
    @location(1) glow: vec2<f32>,
    @location(2) surface: vec4<f32>,
}

// The scene targets, and the offset from this pixel to the same surface
// point in the previous frame, in texture coordinates.
struct TemporalTargets {
    @location(0) color: vec4<f32>,
    @location(1) motion: vec2<f32>,
    @location(2) glow: vec2<f32>,
    @location(3) surface: vec4<f32>,
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

// A selected cluster (decision 044) of a slot: 384 vertices, three per
// triangle of the largest cluster; those past its triangles collapse.
struct VirtualInput {
    @builtin(vertex_index) vertex: u32,
    @location(2) slot: u32,
    @location(13) cluster: u32,
}

// A cluster corner in mesh space; `valid` is false past its triangles.
struct VirtualCorner {
    position: vec3<f32>,
    normal: vec2<f32>,
    uv: vec2<f32>,
    valid: bool,
}

// The low 16 bits as a signed normalized value.
fn snorm16(bits: u32) -> f32 {
    return max(f32(bitcast<i32>(bits << 16u) >> 16u) / 32767.0, -1.0);
}

// Corner `vertex` of cluster `cluster`: its local index, then the stored
// vertex of 16-bit position steps, octahedral normal and texture
// coordinates, dequantized with its mesh's steps.
fn virtual_corner(cluster: u32, vertex: u32) -> VirtualCorner {
    let record = virtual_clusters[cluster * 3u + 1u];
    let words = virtual_clusters[cluster * 3u + 2u];
    var corner: VirtualCorner;
    corner.valid = vertex / 3u < (words.z >> 16u);
    if !corner.valid {
        return corner;
    }
    let mesh = virtual_meshes[record.w];
    let page = mesh.bases.w + virtual_pages[words.w] * VIRTUAL_PAGE_WORDS;
    let packed = virtual_pages[page + words.y + vertex / 4u];
    let base = page + words.x + ((packed >> ((vertex % 4u) * 8u)) & 0xffu) * 4u;
    let w0 = virtual_pages[base];
    let w1 = virtual_pages[base + 1u];
    let w2 = virtual_pages[base + 2u];
    let w3 = virtual_pages[base + 3u];
    let steps = vec3<f32>(f32(w0 & 0xffffu), f32(w0 >> 16u), f32(w1 & 0xffffu));
    corner.position = mesh.position_min.xyz + steps * mesh.position_step.xyz;
    corner.normal = vec2<f32>(snorm16(w1 >> 16u), snorm16(w2));
    corner.uv = mesh.uv.xy + vec2<f32>(f32(w2 >> 16u), f32(w3 & 0xffffu)) * mesh.uv.zw;
    return corner;
}

// Where collapsed corners go: one point outside the view.
const COLLAPSED = vec4<f32>(2.0, 2.0, 0.5, 1.0);

@vertex
fn vs_virtual(input: VirtualInput) -> VertexOutput {
    let corner = virtual_corner(input.cluster, input.vertex);
    let index = input.slot & ~MOVED_SLOT;
    let slot = colors[index];
    var output = transform_vertex(corner.position, corner.normal, slot.color, slot.surface,
        corner.uv, pose_matrix(poses[index]));
    if !corner.valid {
        output.clip_position = COLLAPSED;
    }
    return output;
}

@vertex
fn vs_virtual_temporal(input: VirtualInput) -> TemporalOutput {
    let corner = virtual_corner(input.cluster, input.vertex);
    let slot = input.slot & ~MOVED_SLOT;
    let model = pose_matrix(poses[slot]);
    let material = colors[slot];
    let vertex = transform_vertex(corner.position, corner.normal, material.color,
        material.surface, corner.uv, model);
    let point = vec4<f32>(corner.position, 1.0);
    var world = model * point;
    // Only moving slots read their previous pose, as `vertex_affine_temporal`.
    if (input.slot & MOVED_SLOT) != 0u {
        let pose = previous_poses[slot];
        world = vec4<f32>(dot(pose.row_0, point), dot(pose.row_1, point),
            dot(pose.row_2, point), 1.0);
    }
    let previous = camera.previous_view_projection * world;
    var output: TemporalOutput;
    output.clip_position = select(COLLAPSED, vertex.clip_position, corner.valid);
    output.normal_u = vertex.normal_u;
    output.world_v = vertex.world_v;
    output.material = vertex.material;
    output.previous = previous.xyw;
    return output;
}

// The visibility pass (decision 045): a selected cluster's triangles, posed
// and projected as `vs_virtual` does, write which triangle covers each
// pixel. Its entry is the bin's first this phase plus the instance.
struct VisibilityOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) ids: vec2<u32>,
}

@vertex
fn vs_visibility(input: VirtualInput, @builtin(instance_index) instance: u32)
        -> VisibilityOutput {
    let corner = virtual_corner(input.cluster, input.vertex);
    let model = pose_matrix(poses[input.slot & ~MOVED_SLOT]);
    var output: VisibilityOutput;
    output.clip_position = select(COLLAPSED,
        camera.view_projection * (model * vec4<f32>(corner.position, 1.0)), corner.valid);
    output.ids = vec2<u32>(visibility_draw.visible_base + instance + 1u,
        (input.vertex / 3u) | (visibility_draw.bin << 7u));
    return output;
}

@fragment
fn fs_visibility(input: VisibilityOutput) -> @location(0) vec2<u32> {
    return input.ids;
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
    // Where the sun's shadow is looked up from `world`: zero but for
    // impostors (decision 048), whose rebuilt surfaces look from outside
    // their own meshes' shadows.
    shadow_offset: vec3<f32>,
}

// Shaded light of a surface: graded color (and its linear light before the
// grade, which reflection probes capture), the ambient share of its light
// (for ambient occlusion), its glow weight, its reflectivity, shading normal
// and roughness for screen-space reflections, and its coverage.
struct Shaded {
    color: vec3<f32>,
    linear: vec3<f32>,
    ambient: f32,
    glow: f32,
    reflect: f32,
    normal: vec3<f32>,
    roughness: f32,
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

// Screen variance of the pixel footprint and the largest squared roughness
// specular antialiasing adds.
const SPECULAR_VARIANCE: f32 = 0.15;
const SPECULAR_LIMIT: f32 = 0.2;

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

// What a surface point reads from its material's maps: the base color
// texel, the roughness and metalness factors, the occlusion and emissive
// texels, and the normal map's normal with its coordinates' screen
// derivatives. Materials without maps read the base color alone.
struct MaterialTexels {
    base: vec4<f32>,
    factors: vec2<f32>,
    occlusion: f32,
    emissive: vec3<f32>,
    normal: vec3<f32>,
    normal_dx: vec2<f32>,
    normal_dy: vec2<f32>,
}

fn untextured() -> MaterialTexels {
    var texels: MaterialTexels;
    texels.factors = vec2<f32>(1.0);
    texels.occlusion = 1.0;
    texels.normal = vec3<f32>(0.0, 0.0, 1.0);
    return texels;
}

// The texels of texture coordinates `uv` and `uv1` with implicit
// derivatives, so callers sample in uniform control flow.
fn material_texels(uv: vec2<f32>, uv1: vec2<f32>) -> MaterialTexels {
    var texels = untextured();
    // RGBA8UnormSrgb sampling decodes source RGB; alpha masks emission and,
    // with maps, is coverage.
    var base_uv = uv;
    if MATERIAL_MAPS {
        base_uv = map_uv(0u, uv, uv1);
    }
    texels.base = textureSample(base_color_texture, base_color_sampler, base_uv);
    if MATERIAL_MAPS {
        // glTF: green is roughness and blue metalness, red packed occlusion.
        var metallic_roughness = vec4<f32>(1.0);
        if METALLIC_ROUGHNESS_MAP {
            metallic_roughness = textureSample(metallic_roughness_texture,
                metallic_roughness_sampler, map_uv(1u, uv, uv1));
        }
        texels.factors = metallic_roughness.gb;
        if PACKED_OCCLUSION {
            texels.occlusion = metallic_roughness.r;
        } else if OCCLUSION_MAP {
            texels.occlusion = textureSample(occlusion_texture, occlusion_sampler,
                map_uv(3u, uv, uv1)).r;
        }
        texels.emissive = vec3<f32>(1.0);
        if EMISSIVE_MAP {
            texels.emissive = textureSample(emissive_texture, emissive_sampler,
                map_uv(4u, uv, uv1)).rgb;
        }
    }
    if NORMAL_MAP {
        let normal_uv = map_uv(2u, uv, uv1);
        texels.normal = textureSample(normal_texture, normal_sampler, normal_uv).xyz * 2.0 - 1.0;
        texels.normal_dx = dpdx(normal_uv);
        texels.normal_dy = dpdy(normal_uv);
    }
    return texels;
}

// How the texture coordinates of both sets change per pixel across and
// down.
struct UvGradients {
    dx: vec2<f32>,
    dy: vec2<f32>,
    dx1: vec2<f32>,
    dy1: vec2<f32>,
}

// How map `index`'s coordinates change for changes `d` and `d1` of the first
// and second sets: its transform's linear part.
fn map_gradient(index: u32, d: vec2<f32>, d1: vec2<f32>) -> vec2<f32> {
    let second = ((u32(material_maps.params.w) >> index) & 1u) == 1u;
    let change = vec3<f32>(select(d, d1, second), 0.0);
    return vec2<f32>(dot(material_maps.transforms[index * 2u].xyz, change),
        dot(material_maps.transforms[index * 2u + 1u].xyz, change));
}

// `material_texels` with explicit derivatives, for pixels shading triangles
// their neighbors may not share (decision 045).
fn material_texels_grad(uv: vec2<f32>, uv1: vec2<f32>, gradients: UvGradients)
        -> MaterialTexels {
    var texels = untextured();
    var base_uv = uv;
    var base_dx = gradients.dx;
    var base_dy = gradients.dy;
    if MATERIAL_MAPS {
        base_uv = map_uv(0u, uv, uv1);
        base_dx = map_gradient(0u, gradients.dx, gradients.dx1);
        base_dy = map_gradient(0u, gradients.dy, gradients.dy1);
    }
    texels.base = textureSampleGrad(base_color_texture, base_color_sampler, base_uv, base_dx,
        base_dy);
    if MATERIAL_MAPS {
        var metallic_roughness = vec4<f32>(1.0);
        if METALLIC_ROUGHNESS_MAP {
            metallic_roughness = textureSampleGrad(metallic_roughness_texture,
                metallic_roughness_sampler, map_uv(1u, uv, uv1),
                map_gradient(1u, gradients.dx, gradients.dx1),
                map_gradient(1u, gradients.dy, gradients.dy1));
        }
        texels.factors = metallic_roughness.gb;
        if PACKED_OCCLUSION {
            texels.occlusion = metallic_roughness.r;
        } else if OCCLUSION_MAP {
            texels.occlusion = textureSampleGrad(occlusion_texture, occlusion_sampler,
                map_uv(3u, uv, uv1), map_gradient(3u, gradients.dx, gradients.dx1),
                map_gradient(3u, gradients.dy, gradients.dy1)).r;
        }
        texels.emissive = vec3<f32>(1.0);
        if EMISSIVE_MAP {
            texels.emissive = textureSampleGrad(emissive_texture, emissive_sampler,
                map_uv(4u, uv, uv1), map_gradient(4u, gradients.dx, gradients.dx1),
                map_gradient(4u, gradients.dy, gradients.dy1)).rgb;
        }
    }
    if NORMAL_MAP {
        texels.normal_dx = map_gradient(2u, gradients.dx, gradients.dx1);
        texels.normal_dy = map_gradient(2u, gradients.dy, gradients.dy1);
        texels.normal = textureSampleGrad(normal_texture, normal_sampler, map_uv(2u, uv, uv1),
            texels.normal_dx, texels.normal_dy).xyz * 2.0 - 1.0;
    }
    return texels;
}

// The shading normal of a point whose geometric normal is `geometric`: bent
// by the normal map's `sampled` normal along the tangent frame of the world
// position's and the map's coordinates' screen derivatives, then by
// `ripples`.
fn shading_normal(geometric: vec3<f32>, world: vec3<f32>, world_dx: vec3<f32>,
        world_dy: vec3<f32>, sampled: vec3<f32>, normal_dx: vec2<f32>, normal_dy: vec2<f32>,
        ripples: f32) -> vec3<f32> {
    var normal = geometric;
    if NORMAL_MAP {
        normal = mapped_normal(geometric, world_dx, world_dy, normal_dx, normal_dy, sampled,
            material_maps.params.x);
    }
    if ripples > 0.0 {
        normal = rippled(normal, world, ripples);
    }
    return normal;
}

// The geometric normal of interpolated normal `surface_normal`: back faces
// of double-sided materials are lit from their own side.
fn geometric_normal(surface_normal: vec3<f32>, front: bool) -> vec3<f32> {
    let geometric = stable_normalize(surface_normal);
    if DOUBLE_SIDED && !front {
        return -geometric;
    }
    return geometric;
}

// The point of `texels` sampled there, before specular antialiasing. `color`
// is linear color and opacity; `front` whether the surface faces the camera.
fn textured_point(color: vec4<f32>, surface_normal: vec3<f32>, world: vec3<f32>,
        world_dx: vec3<f32>, world_dy: vec3<f32>, texels: MaterialTexels, packed: u32,
        front: bool) -> SurfacePoint {
    // Roughness, metalness, emission over 16 and ripples.
    let surface = unpack4x8unorm(packed);
    var point: SurfacePoint;
    point.albedo = texels.base.rgb * color.rgb;
    point.opacity = color.a;
    point.emission = surface.z * 16.0 * texels.base.a;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = 1.0;
    var roughness = surface.x;
    var metalness = surface.y;
    if MATERIAL_MAPS {
        point.opacity *= texels.base.a;
        roughness *= texels.factors.x;
        metalness *= texels.factors.y;
        point.occlusion = 1.0 + material_maps.params.y * (texels.occlusion - 1.0);
        point.emissive = material_maps.emissive.rgb * texels.emissive;
    }
    point.roughness = max(roughness, 0.03);
    point.alpha = point.roughness * point.roughness;
    point.f0 = mix(vec3<f32>(0.04), point.albedo, metalness);
    point.diffuse_color = point.albedo * (1.0 - metalness);
    point.geometric = geometric_normal(surface_normal, front);
    point.normal = shading_normal(point.geometric, world, world_dx, world_dy, texels.normal,
        texels.normal_dx, texels.normal_dy, surface.w);
    let to_eye = camera.position.xyz - world;
    point.distance = length(to_eye);
    point.view = stable_normalize(to_eye);
    point.n_v = max(dot(point.normal, point.view), 1.0e-4);
    point.world = world;
    point.world_dx = world_dx;
    point.world_dy = world_dy;
    return point;
}

// Specular antialiasing (Kaplanyan et al.'s kernel, with Filament's variance
// and limit): normals turning within a pixel (by `turn_dx` across and
// `turn_dy` down), over fine bumps or across facet edges, widen the
// highlight lobe, so they shade as the rougher surface they average to
// instead of sparkling or streaking.
fn widened(point: SurfacePoint, turn_dx: vec3<f32>, turn_dy: vec3<f32>) -> SurfacePoint {
    var result = point;
    let widening = min(2.0 * SPECULAR_VARIANCE * (dot(turn_dx, turn_dx) + dot(turn_dy, turn_dy)),
        SPECULAR_LIMIT);
    result.alpha = sqrt(saturate(point.alpha * point.alpha + widening));
    result.roughness = sqrt(result.alpha);
    return result;
}

// Sampling uses implicit derivatives, so callers build the point in uniform
// control flow; what follows may not be. `color` is linear color and
// opacity; `front` whether the surface faces the camera.
fn surface_point(color: vec4<f32>, surface_normal: vec3<f32>, world: vec3<f32>,
        uv: vec2<f32>, uv1: vec2<f32>, packed: u32, front: bool) -> SurfacePoint {
    let texels = material_texels(uv, uv1);
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    let point = textured_point(color, surface_normal, world, world_dx, world_dy, texels, packed,
        front);
    let result = widened(point, dpdx(point.normal), dpdy(point.normal));
    if ALPHA_MASK && result.opacity < material_maps.params.z {
        discard;
    }
    return result;
}

// Where the ray from `position` along `direction` leaves the box `extent`
// around `center`, seen from `center`: the direction to read a probe's map
// along so reflections land on its walls (parallax correction).
fn box_projected(position: vec3<f32>, direction: vec3<f32>, center: vec3<f32>,
        extent: vec3<f32>) -> vec3<f32> {
    let safe = select(direction, vec3<f32>(1.0e-6), abs(direction) < vec3<f32>(1.0e-6));
    let exits = max((center - extent - position) / safe, (center + extent - position) / safe);
    return position + direction * min(exits.x, min(exits.y, exits.z)) - center;
}

// The environment's light reflected toward `reflected` off a surface of
// `normal` and `roughness` at `world`: the reflection probes whose boxes
// hold it (decision 043), smallest first, each fading out toward its faces
// over its blend distance, and the sky's map for what they leave; each map
// at the level blurred by that roughness, read along the lobe's dominant
// direction, which leans toward the normal as the surface roughens (Lagarde
// and de Rousiers).
fn environment_light(reflected: vec3<f32>, normal: vec3<f32>, roughness: f32,
        world: vec3<f32>) -> vec3<f32> {
    let smoothness = 1.0 - roughness;
    let lean = smoothness * (sqrt(smoothness) + roughness);
    let direction = normalize(mix(normal, reflected, lean));
    let level = roughness * f32(textureNumLevels(environment_maps) - 1u);
    var light = vec3<f32>(0.0);
    var covered = 0.0;
    let header = probe_words[0];
    if header.w != 0u {
        let grid = probe_words[1];
        let cell = vec2<i32>(floor((world.xz - bitcast<vec2<f32>>(header.xy))
            / bitcast<f32>(header.z)));
        if all(cell >= vec2<i32>(0)) && all(vec2<u32>(cell) < grid.xy) {
            let list = probe_words[grid.z + u32(cell.y) * grid.x + u32(cell.x)];
            for (var n = 0u; n < list.y && covered < 0.999; n++) {
                let entry = grid.w + (list.x + n) * 2u;
                let first = probe_words[entry];
                let second = probe_words[entry + 1u];
                let center = bitcast<vec3<f32>>(first.xyz);
                let blend = bitcast<f32>(first.w);
                let extent = bitcast<vec3<f32>>(second.xyz);
                let inside = extent - abs(world - center);
                let depth = min(inside.x, min(inside.y, inside.z));
                if depth <= 0.0 {
                    continue;
                }
                let weight = select(1.0, saturate(depth / blend), blend > 0.0) * (1.0 - covered);
                let along = box_projected(world, direction, center, extent);
                light += weight * textureSampleLevel(environment_maps, environment_sampler, along,
                    second.w, level).rgb;
                covered += weight;
            }
        }
    }
    if covered < 0.999 {
        light += (1.0 - covered)
            * textureSampleLevel(environment_maps, environment_sampler, direction, 0, level).rgb;
    }
    return light;
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
    // The environment mirrored in the surface, blurred by its roughness,
    // weighted by the environment BRDF.
    let mirrored = environment_light(reflect(-view, normal), normal, point.roughness,
        point.world);
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
        * sun_visibility(point.world + point.shadow_offset, point.geometric, point.world_dx,
            point.world_dy, point.distance);
    let emitted = (point.albedo * point.emission + point.emissive) * lighting.effects.x;
    let lit = sky_light + sun_light + emitted + local;
    let covered = fog(camera.position.xyz, point.world);
    let fogged = mix(lit, haze(-view), covered);
    var shaded: Shaded;
    shaded.opacity = point.opacity;
    shaded.normal = normal;
    shaded.roughness = point.roughness;
    shaded.linear = fogged;
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
    let range = cluster_ranges[cluster];
    let specular = lighting.sun_color.w;
    var sum = vec3<f32>(0.0);
    for (var n = 0u; n < range.y; n++) {
        let light = lights[cluster_lights[range.x + n]];
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

// Octahedral coordinates of unit `normal` (as `gpu_data::octahedral`, which
// `octahedral_normal` unfolds).
fn octahedral_coordinates(normal: vec3<f32>) -> vec2<f32> {
    let n = normal / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    let sign = select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0));
    return select(n.xy, (vec2<f32>(1.0) - abs(n.yx)) * sign, n.z < 0.0);
}

// The surface target's texel: the shading normal, octahedral in [0, 1], and
// the roughness.
fn surface_texel(shaded: Shaded) -> vec4<f32> {
    return vec4<f32>(octahedral_coordinates(shaded.normal) * 0.5 + 0.5, shaded.roughness, 1.0);
}

fn scene_targets(shaded: Shaded) -> SceneTargets {
    var targets: SceneTargets;
    targets.color = vec4<f32>(shaded.color, shaded.ambient);
    targets.glow = vec2<f32>(shaded.glow, shaded.reflect);
    targets.surface = surface_texel(shaded);
    return targets;
}

// `clip_position` is the fragment's, `previous` the same point's previous
// clip position: x, y and w.
fn temporal_targets(shaded: Shaded, clip_position: vec4<f32>, previous: vec3<f32>)
        -> TemporalTargets {
    var targets: TemporalTargets;
    targets.color = vec4<f32>(shaded.color, shaded.ambient);
    targets.glow = vec2<f32>(shaded.glow, shaded.reflect);
    targets.surface = surface_texel(shaded);
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

// The share of light glass lets through (decision 038); 0 for materials
// without transmission.
fn transmitted_share() -> f32 {
    if TRANSMISSION {
        return material_maps.volume.x;
    }
    return 0.0;
}

// A transparent surface's point: glass diffuses only the light it does not
// let through.
fn transparent_point(input: Fragment, front: bool) -> SurfacePoint {
    var point = surface_point(input.color, input.normal, input.world, input.uv, input.uv1,
        input.surface, front);
    point.diffuse_color *= 1.0 - transmitted_share();
    return point;
}

// Directions of the blur taps of rough glass, alternating full and half
// radius.
const GLASS_TAPS = array<vec2<f32>, 8>(
    vec2<f32>(1.0, 0.0), vec2<f32>(0.3535534, 0.3535534),
    vec2<f32>(0.0, 1.0), vec2<f32>(-0.3535534, 0.3535534),
    vec2<f32>(-1.0, 0.0), vec2<f32>(-0.3535534, -0.3535534),
    vec2<f32>(0.0, -1.0), vec2<f32>(0.3535534, -0.3535534),
);

// Weighted blending of a transparent surface (decision 026). Glass (decision
// 038) with refraction covers what is behind it with the opaque scene seen
// through it: along the ray its index of refraction bends through its
// thickness, blurred by its roughness and tinted by its color. Without
// refraction its transmitted share lets what is behind show through.
fn transparent_targets(shaded: Shaded, point: SurfacePoint, input: Fragment)
        -> TransparentTargets {
    let distance = length(camera.position.xyz - input.world);
    let share = transmitted_share();
    if share <= 0.0 {
        return weighted(shaded.color, shaded.opacity, distance);
    }
    // Light glass reflects instead of letting through.
    let reflectance = saturate(luminance(environment_brdf(point.f0, point.roughness,
        point.n_v)));
    let through = share * (1.0 - reflectance);
    if lighting.refraction.x <= 0.0 {
        let coverage = max(shaded.opacity * (1.0 - through), 1.0e-3);
        return weighted(min(shaded.color / coverage, vec3<f32>(1.0)), coverage, distance);
    }
    let bent = refract(-point.view, point.normal, 1.0 / material_maps.volume.y);
    let exit = point.world + bent * (material_maps.volume.z * lighting.refraction.x);
    let clip = camera.view_projection * vec4<f32>(exit, 1.0);
    let size = vec2<f32>(textureDimensions(refraction_source));
    // A ray leaving behind the eye shows what is straight behind the pixel.
    var uv = input.pixel / size;
    if clip.w > 1.0e-6 {
        uv = vec2<f32>(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
    }
    var behind = textureSampleLevel(refraction_source, refraction_sampler, uv, 0.0).rgb;
    let radius = point.roughness * lighting.refraction.y / size;
    if radius.x * size.x >= 0.5 {
        var sum = behind;
        for (var tap = 0u; tap < 8u; tap++) {
            sum += textureSampleLevel(refraction_source, refraction_sampler,
                uv + GLASS_TAPS[tap] * radius, 0.0).rgb;
        }
        behind = sum / 9.0;
    }
    return weighted(shaded.color + behind * point.albedo * through, shaded.opacity, distance);
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

// The resolve of the visibility buffer (decision 045): each material bin
// draws its tiles, and each of their pixels that bin's triangle covers
// shades as a forward-drawn fragment of that triangle would. The triangle's
// corners are posed as `vs_virtual` poses them, and its attributes
// interpolated at the pixel center with perspective-correct barycentric
// weights. Neighboring pixels may show other triangles, so derivatives come
// from the same triangle's weights at the pixel's partner in its 2x2 quad,
// across and down, as fine derivatives of forward fragments take them.

struct ResolveOutput {
    @builtin(position) clip_position: vec4<f32>,
}

// Corner `vertex` of two triangles over tile `instance` of the bin.
@vertex
fn vs_resolve(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> ResolveOutput {
    let tile = resolve_tiles[visibility_draw.tile_base + instance];
    let size = vec2<f32>(textureDimensions(visibility_ids));
    let corner = vec2<f32>(f32((0x32u >> vertex) & 1u), f32((0x2cu >> vertex) & 1u));
    let origin = vec2<f32>(f32(tile % visibility_draw.tiles_across),
        f32(tile / visibility_draw.tiles_across));
    let pixel = min((origin + corner) * 8.0, size);
    var output: ResolveOutput;
    output.clip_position = vec4<f32>(pixel.x / size.x * 2.0 - 1.0, 1.0 - pixel.y / size.y * 2.0,
        0.0, 1.0);
    return output;
}

// Perspective-correct barycentric weights at normalized device point
// `point` (x, y, 1) of the triangle whose edge functions are `a`, `b` and
// `c` (Olano and Greer's homogeneous rasterization: corners stay in clip
// space, so triangles crossing the eye plane resolve too).
fn barycentric(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, point: vec3<f32>) -> vec3<f32> {
    let edges = vec3<f32>(dot(a, point), dot(b, point), dot(c, point));
    return edges / (edges.x + edges.y + edges.z);
}

// A pixel's triangle: its fragment, how its world position, texture
// coordinates and interpolated normal change per pixel across and down,
// which way its quad partners lie (1 or -1, across and down), whether it
// faces the camera, and the point's previous clip x, y and w.
struct Resolved {
    fragment: Fragment,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    uv_dx: vec2<f32>,
    uv_dy: vec2<f32>,
    normal_dx: vec3<f32>,
    normal_dy: vec3<f32>,
    partner: vec2<f32>,
    front: bool,
    previous: vec3<f32>,
}

// The triangle of this draw's bin covering the pixel at `position`;
// discards pixels of other bins and of nothing.
fn resolve_pixel(position: vec4<f32>) -> Resolved {
    let ids = textureLoad(visibility_ids, vec2<i32>(position.xy), 0).xy;
    if ids.x == 0u || (ids.y >> 7u) != visibility_draw.bin {
        discard;
    }
    let entry = visible_clusters[ids.x - 1u];
    let slot = entry.x & ~MOVED_SLOT;
    let model = pose_matrix(poses[slot]);
    let material = colors[slot];
    let first = (ids.y & 127u) * 3u;
    let a = virtual_corner(entry.y, first);
    let b = virtual_corner(entry.y, first + 1u);
    let c = virtual_corner(entry.y, first + 2u);
    let va = transform_vertex(a.position, a.normal, material.color, material.surface, a.uv,
        model);
    let vb = transform_vertex(b.position, b.normal, material.color, material.surface, b.uv,
        model);
    let vc = transform_vertex(c.position, c.normal, material.color, material.surface, c.uv,
        model);
    let pa = va.clip_position.xyw;
    let pb = vb.clip_position.xyw;
    let pc = vc.clip_position.xyw;
    let ea = cross(pb, pc);
    let eb = cross(pc, pa);
    let ec = cross(pa, pb);
    let size = vec2<f32>(textureDimensions(visibility_ids));
    let point = vec3<f32>(position.x / size.x * 2.0 - 1.0, 1.0 - position.y / size.y * 2.0, 1.0);
    let weights = barycentric(ea, eb, ec, point);
    // Even pixels' partners follow them, odd ones' precede them.
    let partner = select(vec2<f32>(1.0), vec2<f32>(-1.0),
        (vec2<u32>(position.xy) & vec2<u32>(1u)) == vec2<u32>(1u));
    let across = (barycentric(ea, eb, ec, point + vec3<f32>(partner.x * 2.0 / size.x, 0.0, 0.0))
        - weights) * partner.x;
    let down = (barycentric(ea, eb, ec, point - vec3<f32>(0.0, partner.y * 2.0 / size.y, 0.0))
        - weights) * partner.y;
    let normal_u = weights.x * va.normal_u + weights.y * vb.normal_u + weights.z * vc.normal_u;
    let world_v = weights.x * va.world_v + weights.y * vb.world_v + weights.z * vc.world_v;
    var resolved: Resolved;
    resolved.fragment.color = output_color(va.material);
    resolved.fragment.normal = normal_u.xyz;
    resolved.fragment.world = world_v.xyz;
    resolved.fragment.uv = vec2<f32>(normal_u.w, world_v.w);
    resolved.fragment.uv1 = resolved.fragment.uv;
    resolved.fragment.surface = va.material.z;
    resolved.fragment.pixel = position.xy;
    let normal_across = across.x * va.normal_u + across.y * vb.normal_u + across.z * vc.normal_u;
    let normal_down = down.x * va.normal_u + down.y * vb.normal_u + down.z * vc.normal_u;
    let world_across = across.x * va.world_v + across.y * vb.world_v + across.z * vc.world_v;
    let world_down = down.x * va.world_v + down.y * vb.world_v + down.z * vc.world_v;
    resolved.world_dx = world_across.xyz;
    resolved.world_dy = world_down.xyz;
    resolved.uv_dx = vec2<f32>(normal_across.w, world_across.w);
    resolved.uv_dy = vec2<f32>(normal_down.w, world_down.w);
    resolved.normal_dx = normal_across.xyz;
    resolved.normal_dy = normal_down.xyz;
    resolved.partner = partner;
    // Counterclockwise in normalized device coordinates faces the camera.
    resolved.front = dot(pa, ea) > 0.0;
    // Only moving slots read their previous pose, as `vs_virtual_temporal`.
    var previous_world = resolved.fragment.world;
    if (entry.x & MOVED_SLOT) != 0u {
        let pose = previous_poses[slot];
        let mesh_point = vec4<f32>(weights.x * a.position + weights.y * b.position
            + weights.z * c.position, 1.0);
        previous_world = vec3<f32>(dot(pose.row_0, mesh_point), dot(pose.row_1, mesh_point),
            dot(pose.row_2, mesh_point));
    }
    resolved.previous = (camera.previous_view_projection * vec4<f32>(previous_world, 1.0)).xyw;
    return resolved;
}

// The shading normal at a quad partner: the interpolated normal changed by
// `normal_step`, the world position by `world_step` and the texture
// coordinates by `uv_step`, sampling the normal map there.
fn stepped_normal(resolved: Resolved, texels: MaterialTexels, normal_step: vec3<f32>,
        world_step: vec3<f32>, uv_step: vec2<f32>, ripples: f32) -> vec3<f32> {
    let input = resolved.fragment;
    var sampled = texels.normal;
    if NORMAL_MAP {
        let step = map_gradient(2u, uv_step, uv_step);
        sampled = textureSampleGrad(normal_texture, normal_sampler,
            map_uv(2u, input.uv, input.uv1) + step, texels.normal_dx,
            texels.normal_dy).xyz * 2.0 - 1.0;
    }
    return shading_normal(geometric_normal(input.normal + normal_step, resolved.front),
        input.world + world_step, resolved.world_dx, resolved.world_dy, sampled,
        texels.normal_dx, texels.normal_dy, ripples);
}

// `surface_point` of a resolved pixel: explicit derivatives, and the shading
// normal's turn toward its quad partners across and down, as forward
// fragments' derivatives take it.
fn resolved_point(resolved: Resolved) -> SurfacePoint {
    let input = resolved.fragment;
    let gradients = UvGradients(resolved.uv_dx, resolved.uv_dy, resolved.uv_dx, resolved.uv_dy);
    let texels = material_texels_grad(input.uv, input.uv1, gradients);
    let point = textured_point(input.color, input.normal, input.world, resolved.world_dx,
        resolved.world_dy, texels, input.surface, resolved.front);
    let ripples = unpack4x8unorm(input.surface).w;
    let x = resolved.partner.x;
    let y = resolved.partner.y;
    let across = stepped_normal(resolved, texels, resolved.normal_dx * x, resolved.world_dx * x,
        resolved.uv_dx * x, ripples);
    let down = stepped_normal(resolved, texels, resolved.normal_dy * y, resolved.world_dy * y,
        resolved.uv_dy * y, ripples);
    return widened(point, across - point.normal, down - point.normal);
}

fn shade_resolved(resolved: Resolved) -> Shaded {
    return finish(resolved_point(resolved), vec3<f32>(0.0));
}

fn shade_resolved_lit(resolved: Resolved) -> Shaded {
    let point = resolved_point(resolved);
    return finish(point, local_light(point, resolved.fragment.pixel));
}

@fragment
fn fs_resolve(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_resolved(resolve_pixel(position)).color), 1.0);
}

@fragment
fn fs_resolve_lit(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_resolved_lit(resolve_pixel(position)).color), 1.0);
}

@fragment
fn fs_resolve_post(@builtin(position) position: vec4<f32>) -> SceneTargets {
    return scene_targets(shade_resolved(resolve_pixel(position)));
}

@fragment
fn fs_resolve_post_lit(@builtin(position) position: vec4<f32>) -> SceneTargets {
    return scene_targets(shade_resolved_lit(resolve_pixel(position)));
}

@fragment
fn fs_resolve_temporal(@builtin(position) position: vec4<f32>) -> TemporalTargets {
    let resolved = resolve_pixel(position);
    return temporal_targets(shade_resolved(resolved), position, resolved.previous);
}

@fragment
fn fs_resolve_temporal_lit(@builtin(position) position: vec4<f32>) -> TemporalTargets {
    let resolved = resolve_pixel(position);
    return temporal_targets(shade_resolved_lit(resolved), position, resolved.previous);
}

@fragment
fn fs_transparent(input: VertexOutput, @builtin(front_facing) front: bool)
        -> TransparentTargets {
    let values = fragment(input);
    let point = transparent_point(values, front);
    return transparent_targets(finish(point, vec3<f32>(0.0)), point, values);
}

@fragment
fn fs_transparent_lit(input: VertexOutput, @builtin(front_facing) front: bool)
        -> TransparentTargets {
    let values = fragment(input);
    let point = transparent_point(values, front);
    return transparent_targets(finish(point, local_light(point, values.pixel)), point, values);
}

// Reflection probe captures (decision 043): the linear light before the
// grade, which the environment maps hold.
@fragment
fn fs_probe(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return vec4<f32>(shade(fragment(input), front).linear, 1.0);
}

@fragment
fn fs_probe_lit(input: VertexOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    return vec4<f32>(shade_lit(fragment(input), front).linear, 1.0);
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
fn fs_probe_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    return vec4<f32>(shade(extended_fragment(input), front).linear, 1.0);
}

@fragment
fn fs_probe_lit_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    return vec4<f32>(shade_lit(extended_fragment(input), front).linear, 1.0);
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
    let point = transparent_point(values, front);
    return transparent_targets(finish(point, vec3<f32>(0.0)), point, values);
}

@fragment
fn fs_transparent_lit_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> TransparentTargets {
    let values = extended_fragment(input);
    let point = transparent_point(values, front);
    return transparent_targets(finish(point, local_light(point, values.pixel)), point, values);
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
    // The far plane: depth 0 with reversed depth (decision 042).
    output.clip_position = vec4<f32>(ndc, 0.0, 1.0);
    output.ndc = ndc;
    return output;
}

// The view ray through unjittered `ndc`.
fn sky_ray(ndc: vec2<f32>) -> vec3<f32> {
    let far = camera.inverse_view_projection * vec4<f32>(ndc, 0.0, 1.0);
    return stable_normalize(far.xyz / far.w - camera.position.xyz);
}

@fragment
fn fs_sky(input: SkyOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(grade(sky(sky_ray(input.ndc)))), 1.0);
}

// Reflection probe captures (decision 043): the sky's haze without the sun's
// disc, as the sky's environment map holds it (surfaces shade the sun's
// highlights themselves).
@fragment
fn fs_probe_sky(input: SkyOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(haze(sky_ray(input.ndc)), 1.0);
}

// The sky's surface texel: facing up, fully rough (it reflects nothing).
const SKY_SURFACE = vec4<f32>(0.5, 0.5, 1.0, 0.0);

// Screen-effect frames: the sky has no ambient occlusion; its sun glows.
@fragment
fn fs_sky_post(input: SkyOutput) -> SceneTargets {
    let light = sky(sky_ray(input.ndc));
    var targets: SceneTargets;
    targets.color = vec4<f32>(grade(light), 0.0);
    targets.glow = vec2<f32>(glow_weight(light), 0.0);
    targets.surface = SKY_SURFACE;
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
    targets.surface = SKY_SURFACE;
    let previous = camera.previous_view_projection * vec4<f32>(ray, 0.0);
    targets.motion = (previous.xy / previous.w - input.ndc) * vec2<f32>(0.5, -0.5);
    if !(previous.w > 0.0) {
        targets.motion = vec2<f32>(2.0);
    }
    return targets;
}

// Decals (decision 039): a box per decal, drawn after the opaque pass. Its
// fragments rebuild the opaque surface under them from the depth buffer and
// cover it with the decal's material where it lies inside the box.
struct DecalRecord {
    // Transposed top rows of the box's world matrix, unless `pose.x` names
    // its displayed pose in the joint palette.
    model: array<vec4<f32>, 3>,
    // Linear color and alpha.
    color: vec4<f32>,
    // Fade cosine, roughness, metalness and emission.
    params: vec4<f32>,
    // x: palette entry, or NO_DECAL_POSE.
    pose: vec4<u32>,
}
@group(2) @binding(8) var<storage, read> decal_records: array<DecalRecord>;
@group(2) @binding(9) var decal_depth: texture_depth_2d;

const NO_DECAL_POSE: u32 = 0xffffffffu;

// Multiplying decals scale the surface's color instead of covering it;
// canvas pipelines encode what they write.
override DECAL_MULTIPLY: bool = false;
override DECAL_CANVAS: bool = false;

// The unit box's twelve triangles, wound outward: culling front faces keeps
// one layer covering the box's pixels, wherever the eye is.
const DECAL_BOX = array<vec3<f32>, 36>(
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0),
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0),
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 1.0),
);

struct DecalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) decal: u32,
    // Rows of the inverse of the box's world matrix.
    @location(1) @interpolate(flat) inverse_0: vec4<f32>,
    @location(2) @interpolate(flat) inverse_1: vec4<f32>,
    @location(3) @interpolate(flat) inverse_2: vec4<f32>,
    // The box's +y in world space.
    @location(4) @interpolate(flat) up: vec3<f32>,
}

@vertex
fn vs_decal(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> DecalOutput {
    let record = decal_records[instance];
    var rows = record.model;
    if record.pose.x != NO_DECAL_POSE {
        let pose = palette[record.pose.x];
        rows = array<vec4<f32>, 3>(pose.row_0, pose.row_1, pose.row_2);
    }
    let r0 = rows[0].xyz;
    let r1 = rows[1].xyz;
    let r2 = rows[2].xyz;
    let determinant = dot(r0, cross(r1, r2));
    var output: DecalOutput;
    output.decal = instance;
    // A flat box covers nothing.
    if !(abs(determinant) > 1.0e-12) {
        output.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return output;
    }
    // Mirrored boxes turn their triangles inside out: wind them back.
    var corner = vertex;
    if determinant < 0.0 {
        corner = vertex + 2u - 2u * (vertex % 3u);
    }
    let local = vec4<f32>(DECAL_BOX[corner] - 0.5, 1.0);
    let world = vec3<f32>(dot(rows[0], local), dot(rows[1], local), dot(rows[2], local));
    output.clip_position = camera.view_projection * vec4<f32>(world, 1.0);
    // The inverse's columns are the rows' cross products over the
    // determinant; its translation moves back.
    let inverse = mat3x3<f32>(cross(r1, r2), cross(r2, r0), cross(r0, r1)) * (1.0 / determinant);
    let translation = vec3<f32>(rows[0].w, rows[1].w, rows[2].w);
    let inverse_rows = transpose(inverse);
    output.inverse_0 = vec4<f32>(inverse_rows[0], -dot(inverse_rows[0], translation));
    output.inverse_1 = vec4<f32>(inverse_rows[1], -dot(inverse_rows[1], translation));
    output.inverse_2 = vec4<f32>(inverse_rows[2], -dot(inverse_rows[2], translation));
    output.up = stable_normalize(vec3<f32>(rows[0].y, rows[1].y, rows[2].y));
    return output;
}

// The opaque surface's world position at `pixel` of depth `depth`, without
// the jitter of temporally antialiased frames (decision 017).
fn decal_surface(pixel: vec2<i32>, depth: f32) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(decal_depth));
    let center = vec2<f32>(pixel) + 0.5;
    let ndc = vec2<f32>(center.x / size.x * 2.0 - 1.0, 1.0 - center.y / size.y * 2.0)
        - camera.jitter.xy;
    let point = camera.inverse_view_projection * vec4<f32>(ndc, depth, 1.0);
    return point.xyz / point.w;
}

// How the surface at `pixel` moves one pixel along `axis`: toward the
// neighbor whose depth is nearer its own, so edges of other surfaces do not
// tilt it.
fn decal_step(pixel: vec2<i32>, depth: f32, world: vec3<f32>, axis: vec2<i32>) -> vec3<f32> {
    let last = vec2<i32>(textureDimensions(decal_depth)) - 1;
    let after = min(pixel + axis, last);
    let before = max(pixel - axis, vec2<i32>(0));
    let after_depth = textureLoad(decal_depth, after, 0);
    let before_depth = textureLoad(decal_depth, before, 0);
    let at_end = all(after == pixel);
    let at_start = all(before == pixel);
    if !at_end && (at_start || abs(after_depth - depth) <= abs(before_depth - depth)) {
        return decal_surface(after, after_depth) - world;
    }
    return world - decal_surface(before, before_depth);
}

// The decal's point on the opaque surface under this fragment: its albedo,
// coverage (`opacity`), normal and position. Discards pixels outside the
// box, of the sky or of no coverage.
fn decal_point(input: DecalOutput) -> SurfacePoint {
    let pixel = vec2<i32>(input.clip_position.xy);
    let depth = textureLoad(decal_depth, pixel, 0);
    // The sky has no surface to cover: it keeps the cleared depth, 0 with
    // reversed depth (decision 042).
    if !(depth > 0.0) {
        discard;
    }
    let world = decal_surface(pixel, depth);
    let point4 = vec4<f32>(world, 1.0);
    let local = vec3<f32>(dot(input.inverse_0, point4), dot(input.inverse_1, point4),
        dot(input.inverse_2, point4));
    if any(abs(local) > vec3<f32>(0.5)) {
        discard;
    }
    let record = decal_records[input.decal];
    let world_dx = decal_step(pixel, depth, world, vec2<i32>(1, 0));
    let world_dy = decal_step(pixel, depth, world, vec2<i32>(0, 1));
    let to_eye = camera.position.xyz - world;
    var normal = stable_normalize(cross(world_dx, world_dy));
    if dot(normal, to_eye) < 0.0 {
        normal = -normal;
    }
    // Surfaces turning away from the box's +y fade out toward a right angle.
    let facing = saturate(dot(normal, input.up) / max(record.params.x, 1.0e-4));
    // The material spans the box's x and z; its map's footprint follows the
    // surface's steps.
    let uv = local.xz + 0.5;
    let uv_dx = vec2<f32>(dot(input.inverse_0.xyz, world_dx), dot(input.inverse_2.xyz, world_dx));
    let uv_dy = vec2<f32>(dot(input.inverse_0.xyz, world_dy), dot(input.inverse_2.xyz, world_dy));
    let transform = mat2x2<f32>(
        vec2<f32>(material_maps.transforms[0].x, material_maps.transforms[1].x),
        vec2<f32>(material_maps.transforms[0].y, material_maps.transforms[1].y));
    let texel = textureSampleGrad(base_color_texture, base_color_sampler, map_uv(0u, uv, uv),
        transform * uv_dx, transform * uv_dy);
    let alpha = record.color.a * texel.a * facing;
    if !(alpha > 0.0) {
        discard;
    }
    let albedo = record.color.rgb * texel.rgb;
    let metalness = record.params.z;
    var point: SurfacePoint;
    point.albedo = albedo;
    point.opacity = alpha;
    point.emission = record.params.w * texel.a;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = 1.0;
    point.roughness = max(record.params.y, 0.03);
    point.alpha = point.roughness * point.roughness;
    point.f0 = mix(vec3<f32>(0.04), albedo, metalness);
    point.diffuse_color = albedo * (1.0 - metalness);
    point.normal = normal;
    point.geometric = normal;
    point.distance = length(to_eye);
    point.view = stable_normalize(to_eye);
    point.n_v = max(dot(normal, point.view), 1.0e-4);
    point.world = world;
    point.world_dx = world_dx;
    point.world_dy = world_dy;
    return point;
}

// Graded color and alpha as the target stores them.
fn decal_output(color: vec3<f32>, alpha: f32) -> vec4<f32> {
    if DECAL_CANVAS {
        return vec4<f32>(encoded(color), alpha);
    }
    return vec4<f32>(color, alpha);
}

@fragment
fn fs_decal(input: DecalOutput) -> @location(0) vec4<f32> {
    let point = decal_point(input);
    if DECAL_MULTIPLY {
        // Fog hides what it covers.
        let visible = point.opacity * (1.0 - fog(camera.position.xyz, point.world));
        return decal_output(mix(vec3<f32>(1.0), point.albedo, visible), 1.0);
    }
    // Covering decals are lit like any surface.
    return decal_output(finish(point, vec3<f32>(0.0)).color, point.opacity);
}

// Covering decals of frames with local lights (decision 023).
@fragment
fn fs_decal_lit(input: DecalOutput) -> @location(0) vec4<f32> {
    let point = decal_point(input);
    return decal_output(finish(point, local_light(point, input.clip_position.xy)).color,
        point.opacity);
}

// Octahedral impostors (decision 048): a draw group's mesh baked from
// IMPOSTOR_FRAMES x IMPOSTOR_FRAMES directions over the whole sphere into
// one layer of four texture arrays, drawn for instances too small for its
// texels to show as a card facing the viewer that blends the three frames
// around the view. Texels hold premultiplied coverage: albedo and coverage;
// emissive color and emission mask; normal and depth; roughness and
// metalness factors and occlusion. Normals are plain vectors, which filter
// and average across levels as octahedral ones, folded across the lower
// hemisphere's seams, cannot.
const IMPOSTOR_FRAMES: u32 = 8u;
// Texels across a frame of the first level, and the levels.
const IMPOSTOR_TEXELS: u32 = 16u;
const IMPOSTOR_LEVELS: u32 = 4u;

struct ImpostorRecord {
    // Mesh-space center and radius of the sphere every frame frames.
    sphere: vec4<f32>,
    // Mesh-space half extents of the mesh's bounds, and the layer.
    extent: vec3<f32>,
    layer: u32,
    // The material's emissive color (glTF), linear.
    emissive: vec4<f32>,
}
// One record per draw group; groups without an impostor never draw one.
@group(1) @binding(11) var<storage, read> impostor_records: array<ImpostorRecord>;
@group(1) @binding(12) var impostor_albedo: texture_2d_array<f32>;
@group(1) @binding(13) var impostor_emissive: texture_2d_array<f32>;
@group(1) @binding(14) var impostor_geometry: texture_2d_array<f32>;
@group(1) @binding(15) var impostor_surface: texture_2d_array<f32>;
@group(1) @binding(16) var impostor_sampler: sampler;

// Octahedral coordinates in [-1, 1] of unit direction `d` about +y: the
// upper hemisphere fills the central diamond, the lower folds into the
// corners.
fn impostor_octahedral(d: vec3<f32>) -> vec2<f32> {
    let n = d / (abs(d.x) + abs(d.y) + abs(d.z));
    let sign = select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xz >= vec2<f32>(0.0));
    return select(n.xz, (vec2<f32>(1.0) - abs(n.zx)) * sign, n.y < 0.0);
}

// The unit direction of octahedral coordinates `p`.
fn impostor_direction(p: vec2<f32>) -> vec3<f32> {
    var d = vec3<f32>(p.x, 1.0 - abs(p.x) - abs(p.y), p.y);
    let fold = max(-d.y, 0.0);
    d.x += select(fold, -fold, d.x >= 0.0);
    d.z += select(fold, -fold, d.z >= 0.0);
    return normalize(d);
}

// Frame `frame`'s direction toward its viewer: the frames sit on the
// octahedral map's grid points, its edges and corners included.
fn impostor_frame_direction(frame: u32) -> vec3<f32> {
    let cell = vec2<f32>(f32(frame % IMPOSTOR_FRAMES), f32(frame / IMPOSTOR_FRAMES));
    return impostor_direction(cell / f32(IMPOSTOR_FRAMES - 1u) * 2.0 - 1.0);
}

// Right and up of an image seen from direction `d`: up follows +y unless `d`
// nearly does.
struct ImpostorBasis {
    right: vec3<f32>,
    up: vec3<f32>,
}

fn impostor_basis(d: vec3<f32>) -> ImpostorBasis {
    var hint = vec3<f32>(0.0, 1.0, 0.0);
    if abs(d.y) > 0.999 {
        hint = vec3<f32>(0.0, 0.0, -sign(d.y));
    }
    let right = normalize(cross(hint, d));
    return ImpostorBasis(right, cross(d, right));
}

// Baking: every frame renders the mesh orthographically into its cell of the
// layer, instance `frame` for frame `frame`, the sphere filling the cell.
struct ImpostorBake {
    // Mesh-space center and radius of the frames' sphere.
    sphere: vec4<f32>,
}
@group(2) @binding(20) var<uniform> impostor_bake: ImpostorBake;

struct BakeOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Mesh-space normal and texture u; mesh-space position and texture v.
    @location(0) normal_u: vec4<f32>,
    @location(1) local_v: vec4<f32>,
    // Depth toward the frame's viewer, in sphere radii from the center.
    @location(2) depth: f32,
}

struct BakeExtendedOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) local_v: vec4<f32>,
    @location(2) depth: f32,
    @location(3) color: vec4<f32>,
    @location(4) uv1: vec2<f32>,
}

fn bake_vertex(position: vec3<f32>, encoded_normal: vec2<f32>, uv: vec2<f32>, frame: u32)
        -> BakeOutput {
    let d = impostor_frame_direction(frame);
    let basis = impostor_basis(d);
    let local = (position - impostor_bake.sphere.xyz) / impostor_bake.sphere.w;
    let image = vec2<f32>(dot(local, basis.right), -dot(local, basis.up));
    let depth = clamp(dot(local, d), -1.0, 1.0);
    // The frame's cell, from the top left.
    let cell = vec2<f32>(f32(frame % IMPOSTOR_FRAMES), f32(frame / IMPOSTOR_FRAMES));
    let atlas = (cell + 0.5 + image * 0.5) / f32(IMPOSTOR_FRAMES);
    var output: BakeOutput;
    // Nearer the frame's viewer is greater, as reversed depth keeps it.
    output.clip_position = vec4<f32>(atlas.x * 2.0 - 1.0, 1.0 - atlas.y * 2.0,
        0.5 + 0.5 * depth, 1.0);
    output.normal_u = vec4<f32>(octahedral_normal(encoded_normal), uv.x);
    output.local_v = vec4<f32>(position, uv.y);
    output.depth = depth;
    return output;
}

@vertex
fn vs_impostor_bake(@location(0) position: vec3<f32>, @location(1) normal: vec2<f32>,
        @location(7) uv: vec2<f32>, @builtin(instance_index) frame: u32) -> BakeOutput {
    return bake_vertex(position, normal, uv, frame);
}

@vertex
fn vs_impostor_bake_extended(@location(0) position: vec3<f32>, @location(1) normal: vec2<f32>,
        @location(7) uv: vec2<f32>, @location(9) color: vec4<f32>, @location(10) uv1: vec2<f32>,
        @builtin(instance_index) frame: u32) -> BakeExtendedOutput {
    let vertex = bake_vertex(position, normal, uv, frame);
    var output: BakeExtendedOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.local_v = vertex.local_v;
    output.depth = vertex.depth;
    output.color = color;
    output.uv1 = uv1;
    return output;
}

// The albedo, emission, geometry and surface factors of the material at a
// baked point, as its fragments would read them; the targets' resolve
// premultiplies them by coverage.
struct BakeTargets {
    @location(0) albedo: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) geometry: vec4<f32>,
    @location(3) surface: vec4<f32>,
}

fn bake_targets(input: BakeOutput, color: vec4<f32>, uv1: vec2<f32>, front: bool)
        -> BakeTargets {
    let uv = vec2<f32>(input.normal_u.w, input.local_v.w);
    let texels = material_texels(uv, uv1);
    let local = input.local_v.xyz;
    let local_dx = dpdx(local);
    let local_dy = dpdy(local);
    let geometric = geometric_normal(input.normal_u.xyz, front);
    var normal = geometric;
    if NORMAL_MAP {
        normal = mapped_normal(geometric, local_dx, local_dy, texels.normal_dx,
            texels.normal_dy, texels.normal, material_maps.params.x);
    }
    var emissive = vec3<f32>(0.0);
    var occlusion = 1.0;
    if MATERIAL_MAPS {
        emissive = texels.emissive;
        occlusion = 1.0 + material_maps.params.y * (texels.occlusion - 1.0);
    }
    if ALPHA_MASK && texels.base.a * color.a < material_maps.params.z {
        discard;
    }
    var targets: BakeTargets;
    targets.albedo = vec4<f32>(texels.base.rgb * color.rgb, 1.0);
    // Texture alpha masks emission (decision 019).
    targets.emissive = vec4<f32>(emissive, texels.base.a);
    targets.geometry = vec4<f32>(stable_normalize(normal) * 0.5 + 0.5, 0.5 + 0.5 * input.depth);
    targets.surface = vec4<f32>(texels.factors, occlusion, 1.0);
    return targets;
}

@fragment
fn fs_impostor_bake(input: BakeOutput, @builtin(front_facing) front: bool) -> BakeTargets {
    let uv = vec2<f32>(input.normal_u.w, input.local_v.w);
    return bake_targets(input, vec4<f32>(1.0), uv, front);
}

@fragment
fn fs_impostor_bake_extended(input: BakeExtendedOutput, @builtin(front_facing) front: bool)
        -> BakeTargets {
    var vertex: BakeOutput;
    vertex.clip_position = input.clip_position;
    vertex.normal_u = input.normal_u;
    vertex.local_v = input.local_v;
    vertex.depth = input.depth;
    return bake_targets(vertex, input.color, input.uv1, front);
}

// Drawing: one card per impostor instance, six vertices, from the visible
// slot and its draw group.
struct ImpostorInput {
    @builtin(vertex_index) vertex: u32,
    @location(2) slot: u32,
    @location(3) batch: u32,
}

struct ImpostorOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Layer coordinates of the card point in the three blended frames.
    @location(0) frames_ab: vec4<f32>,
    @location(1) frame_c: vec2<f32>,
    // The card point in world space.
    @location(2) world: vec3<f32>,
    // Columns of the normal transform (cofactors, signed by the pose's
    // orientation); w: the first two frames' blend weights.
    @location(3) @interpolate(flat) normal_x: vec4<f32>,
    @location(4) @interpolate(flat) normal_y: vec4<f32>,
    @location(5) @interpolate(flat) normal_z: vec3<f32>,
    // World offset per unit of baked depth toward the viewer, and the
    // card's own depth in those units.
    @location(6) @interpolate(flat) depth_axis: vec4<f32>,
    // Linear color and opacity as four 16-bit fractions, the packed surface
    // (decision 019), and the three frames (six bits each).
    @location(7) @interpolate(flat) material: vec4<u32>,
    @location(8) @interpolate(flat) batch: u32,
}

struct ImpostorTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) frames_ab: vec4<f32>,
    @location(1) frame_c: vec2<f32>,
    @location(2) world: vec3<f32>,
    @location(3) @interpolate(flat) normal_x: vec4<f32>,
    @location(4) @interpolate(flat) normal_y: vec4<f32>,
    @location(5) @interpolate(flat) normal_z: vec3<f32>,
    @location(6) @interpolate(flat) depth_axis: vec4<f32>,
    @location(7) @interpolate(flat) material: vec4<u32>,
    @location(8) @interpolate(flat) batch: u32,
    // The unjittered previous clip position: x, y and w.
    @location(9) previous: vec3<f32>,
}

// Two triangles of a card, corners in its right and up.
const IMPOSTOR_CARD = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

// Unit direction from `point` toward the viewer: toward the eye with
// perspective, against the view with orthographic projections.
fn toward_viewer(point: vec3<f32>) -> vec3<f32> {
    let projection = camera.view_projection;
    let w_row = vec3<f32>(projection[0].w, projection[1].w, projection[2].w);
    if length(w_row) > 0.5 {
        return stable_normalize(camera.position.xyz - point);
    }
    // Reversed depth (decision 042): the near plane is 1, the far plane 0.
    let near = camera.inverse_view_projection * vec4<f32>(0.0, 0.0, 1.0, 1.0);
    let far = camera.inverse_view_projection * vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return stable_normalize(near.xyz / near.w - far.xyz / far.w);
}

// The card of an impostor instance in mesh space, and its outputs besides
// the previous position.
struct ImpostorCard {
    local: vec3<f32>,
    output: ImpostorOutput,
}

fn impostor_card(input: ImpostorInput) -> ImpostorCard {
    let slot = input.slot & ~MOVED_SLOT;
    let model = pose_matrix(poses[slot]);
    let record = impostor_records[input.batch];
    let center = record.sphere.xyz;
    let radius = record.sphere.w;
    // Cofactor columns: the inverse's rows times the determinant, rescaled
    // as `transform_vertex` does.
    let inverse_scale = 1.0 / max(maximum_abs(model[0].xyz),
        max(maximum_abs(model[1].xyz), maximum_abs(model[2].xyz)));
    let x = model[0].xyz * inverse_scale;
    let y = model[1].xyz * inverse_scale;
    let z = model[2].xyz * inverse_scale;
    let yz = cross(y, z);
    let zx = cross(z, x);
    let xy = cross(x, y);
    let orientation = sign(dot(x, yz));
    // The mesh-space direction toward the viewer.
    let toward = toward_viewer((model * vec4<f32>(center, 1.0)).xyz);
    let dir = stable_normalize(vec3<f32>(dot(yz, toward), dot(zx, toward), dot(xy, toward))
        * orientation);
    // The three frames around it: the grid triangle holding it.
    let grid = (impostor_octahedral(dir) * 0.5 + 0.5) * f32(IMPOSTOR_FRAMES - 1u);
    let cell = min(floor(grid), vec2<f32>(f32(IMPOSTOR_FRAMES - 2u)));
    let f = grid - cell;
    let base = vec2<u32>(cell);
    var corners = array<vec2<u32>, 3>(base, base + vec2<u32>(1u, 0u), base + vec2<u32>(0u, 1u));
    var weights = vec3<f32>(1.0 - f.x - f.y, f.x, f.y);
    if f.x + f.y > 1.0 {
        corners = array<vec2<u32>, 3>(base + vec2<u32>(1u), base + vec2<u32>(0u, 1u),
            base + vec2<u32>(1u, 0u));
        weights = vec3<f32>(f.x + f.y - 1.0, 1.0 - f.x, 1.0 - f.y);
    }
    // The card: the bounds' outline seen along `dir`, in front of them.
    let basis = impostor_basis(dir);
    let half = record.extent;
    let support = dot(abs(dir), half);
    let outline = vec2<f32>(dot(abs(basis.right), half), dot(abs(basis.up), half));
    let corner = IMPOSTOR_CARD[input.vertex];
    let local = center + dir * support + basis.right * (corner.x * outline.x)
        + basis.up * (corner.y * outline.y);
    let world = model * vec4<f32>(local, 1.0);
    // Each frame shows the card point where the view projects it onto the
    // frame's plane through the center.
    let offset = local - center;
    var uvs: array<vec2<f32>, 3>;
    var frames = 0u;
    for (var k = 0u; k < 3u; k++) {
        let frame = corners[k].y * IMPOSTOR_FRAMES + corners[k].x;
        let d = impostor_frame_direction(frame);
        let frame_basis = impostor_basis(d);
        let projected = offset - dir * (dot(offset, d) / max(dot(dir, d), 1.0e-3));
        let image = vec2<f32>(dot(projected, frame_basis.right), -dot(projected, frame_basis.up))
            / radius;
        uvs[k] = (vec2<f32>(corners[k]) + 0.5 + image * 0.5) / f32(IMPOSTOR_FRAMES);
        frames |= frame << (6u * k);
    }
    let color = colors[slot];
    var output: ImpostorOutput;
    output.clip_position = camera.view_projection * world;
    output.frames_ab = vec4<f32>(uvs[0], uvs[1]);
    output.frame_c = uvs[2];
    output.world = world.xyz;
    output.normal_x = vec4<f32>(yz * orientation, weights.x);
    output.normal_y = vec4<f32>(zx * orientation, weights.y);
    output.normal_z = xy * orientation;
    let axis = model[0].xyz * dir.x + model[1].xyz * dir.y + model[2].xyz * dir.z;
    output.depth_axis = vec4<f32>(axis * radius, support / radius);
    output.material = vec4<u32>(color.color, color.surface, frames);
    output.batch = input.batch;
    return ImpostorCard(local, output);
}

@vertex
fn vs_impostor(input: ImpostorInput) -> ImpostorOutput {
    return impostor_card(input).output;
}

@vertex
fn vs_impostor_temporal(input: ImpostorInput) -> ImpostorTemporalOutput {
    let card = impostor_card(input);
    let slot = input.slot & ~MOVED_SLOT;
    let point = vec4<f32>(card.local, 1.0);
    // Only moving slots read their previous poses (decision 017).
    var pose = poses[slot];
    if (input.slot & MOVED_SLOT) != 0u {
        pose = previous_poses[slot];
    }
    let world = vec4<f32>(dot(pose.row_0, point), dot(pose.row_1, point),
        dot(pose.row_2, point), 1.0);
    let vertex = card.output;
    var output: ImpostorTemporalOutput;
    output.clip_position = vertex.clip_position;
    output.frames_ab = vertex.frames_ab;
    output.frame_c = vertex.frame_c;
    output.world = vertex.world;
    output.normal_x = vertex.normal_x;
    output.normal_y = vertex.normal_y;
    output.normal_z = vertex.normal_z;
    output.depth_axis = vertex.depth_axis;
    output.material = vertex.material;
    output.batch = vertex.batch;
    output.previous = (camera.previous_view_projection * world).xyw;
    return output;
}

// The surface an impostor fragment shows: the three frames blended by
// weight and coverage, unpremultiplied, on the slot's color and surface;
// discards fragments the blend leaves under half covered.
fn impostor_point(input: ImpostorOutput) -> SurfacePoint {
    let record = impostor_records[input.batch];
    let weights = vec3<f32>(input.normal_x.w, input.normal_y.w,
        1.0 - input.normal_x.w - input.normal_y.w);
    let packed = input.material.w;
    let frames = vec3<u32>(packed & 63u, (packed >> 6u) & 63u, (packed >> 12u) & 63u);
    // The level whose texels match the pixels, and half a texel of the
    // coarser level sampled, which keeps each frame's samples in its cell.
    let size = f32(IMPOSTOR_FRAMES * IMPOSTOR_TEXELS);
    let footprint = max(length(dpdx(input.frames_ab.xy)), length(dpdy(input.frames_ab.xy)))
        * size;
    let level = clamp(log2(max(footprint, 1.0e-6)), 0.0, f32(IMPOSTOR_LEVELS - 1u));
    let inset = 0.5 * exp2(ceil(level)) / size;
    let span = 1.0 / f32(IMPOSTOR_FRAMES);
    let unclamped = array<vec2<f32>, 3>(input.frames_ab.xy, input.frames_ab.zw, input.frame_c);
    var uvs: array<vec2<f32>, 3>;
    var albedo = vec4<f32>(0.0);
    for (var k = 0u; k < 3u; k++) {
        let cell = vec2<f32>(f32(frames[k] % IMPOSTOR_FRAMES), f32(frames[k] / IMPOSTOR_FRAMES))
            * span;
        uvs[k] = clamp(unclamped[k], cell + inset, cell + span - inset);
        albedo += weights[k] * textureSampleLevel(impostor_albedo, impostor_sampler, uvs[k],
            record.layer, level);
    }
    // Most of a card around its mesh is empty: those fragments stop before
    // reading the rest.
    let coverage = albedo.a;
    if coverage < 0.5 {
        discard;
    }
    var emissive = vec4<f32>(0.0);
    var geometry = vec4<f32>(0.0);
    var surface = vec4<f32>(0.0);
    for (var k = 0u; k < 3u; k++) {
        emissive += weights[k] * textureSampleLevel(impostor_emissive, impostor_sampler, uvs[k],
            record.layer, level);
        geometry += weights[k] * textureSampleLevel(impostor_geometry, impostor_sampler, uvs[k],
            record.layer, level);
        surface += weights[k] * textureSampleLevel(impostor_surface, impostor_sampler, uvs[k],
            record.layer, level);
    }
    let unpremultiply = 1.0 / max(coverage, 1.0e-4);
    let mesh_normal = geometry.xyz * unpremultiply * 2.0 - 1.0;
    let geometric = stable_normalize(input.normal_x.xyz * mesh_normal.x
        + input.normal_y.xyz * mesh_normal.y + input.normal_z * mesh_normal.z);
    // The surface behind the card point, at the blended depth.
    let depth = geometry.w * unpremultiply * 2.0 - 1.0;
    let world = input.world - input.depth_axis.xyz * (input.depth_axis.w - depth);
    // Roughness, metalness, emission over 16 and ripples.
    let slot_surface = unpack4x8unorm(input.material.z);
    let color = vec4<f32>(unpack2x16unorm(input.material.x), unpack2x16unorm(input.material.y));
    let albedo_color = albedo.rgb * unpremultiply * color.rgb;
    let metalness = slot_surface.y * surface.y * unpremultiply;
    var point: SurfacePoint;
    point.albedo = albedo_color;
    point.opacity = color.a;
    point.emission = slot_surface.z * 16.0 * emissive.a * unpremultiply;
    point.emissive = record.emissive.rgb * emissive.rgb * unpremultiply;
    point.occlusion = surface.z * unpremultiply;
    point.roughness = max(slot_surface.x * surface.x * unpremultiply, 0.03);
    point.alpha = point.roughness * point.roughness;
    point.f0 = mix(vec3<f32>(0.04), albedo_color, metalness);
    point.diffuse_color = albedo_color * (1.0 - metalness);
    point.geometric = geometric;
    point.normal = geometric;
    if slot_surface.w > 0.0 {
        point.normal = rippled(geometric, world, slot_surface.w);
    }
    let to_eye = camera.position.xyz - world;
    point.distance = length(to_eye);
    point.view = stable_normalize(to_eye);
    point.n_v = max(dot(point.normal, point.view), 1.0e-4);
    point.world = world;
    // Each frame measures depth along its own view, so blended frames
    // rebuild the surface up to a few tenths of the radius off the mesh
    // that casts its shadow: the sun is looked up a radius toward it, from
    // outside the mesh's own shadow, with a flat receiver plane
    // (`world_dx` and `world_dy` stay zero).
    point.shadow_offset = lighting.sun.xyz * length(input.depth_axis.xyz);
    return widened(point, dpdx(point.normal), dpdy(point.normal));
}

fn impostor_output(input: ImpostorTemporalOutput) -> ImpostorOutput {
    var output: ImpostorOutput;
    output.clip_position = input.clip_position;
    output.frames_ab = input.frames_ab;
    output.frame_c = input.frame_c;
    output.world = input.world;
    output.normal_x = input.normal_x;
    output.normal_y = input.normal_y;
    output.normal_z = input.normal_z;
    output.depth_axis = input.depth_axis;
    output.material = input.material;
    output.batch = input.batch;
    return output;
}

fn shade_impostor(input: ImpostorOutput) -> Shaded {
    return finish(impostor_point(input), vec3<f32>(0.0));
}

fn shade_impostor_lit(input: ImpostorOutput) -> Shaded {
    let point = impostor_point(input);
    return finish(point, local_light(point, input.clip_position.xy));
}

@fragment
fn fs_impostor(input: ImpostorOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_impostor(input).color), 1.0);
}

@fragment
fn fs_impostor_lit(input: ImpostorOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_impostor_lit(input).color), 1.0);
}

@fragment
fn fs_impostor_post(input: ImpostorOutput) -> SceneTargets {
    return scene_targets(shade_impostor(input));
}

@fragment
fn fs_impostor_post_lit(input: ImpostorOutput) -> SceneTargets {
    return scene_targets(shade_impostor_lit(input));
}

@fragment
fn fs_impostor_temporal(input: ImpostorTemporalOutput) -> TemporalTargets {
    return temporal_targets(shade_impostor(impostor_output(input)), input.clip_position,
        input.previous);
}

@fragment
fn fs_impostor_temporal_lit(input: ImpostorTemporalOutput) -> TemporalTargets {
    return temporal_targets(shade_impostor_lit(impostor_output(input)), input.clip_position,
        input.previous);
}
