// Mesh vertex stages: poses and skinning, virtual geometry's clusters and
// the visibility pass, and the vertex entry
// points of slots and temporal frames.

fn pose_matrix(pose: AffinePose) -> mat4x4<f32> {
    return transpose(mat4x4<f32>(pose.row_0, pose.row_1, pose.row_2,
        vec4<f32>(0.0, 0.0, 0.0, 1.0)));
}

// Linear color and opacity of a vertex output's packed material.
fn output_color(material: vec4<u32>) -> vec4<f32> {
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

fn vertex_affine(input: AffineVertexInput) -> VertexOutput {
    // Transparent draws of temporal frames see moving slots' marks.
    let index = slot_index(input.slot);
    let model = pose_matrix(poses[index]);
    let slot = colors[index];
    var output = transform_vertex(input.position, input.normal, slot.color, slot.surface,
        input.uv, model);
    output.material.w = slot_fade(input.slot);
    let sway = slot_sway(slot.flags);
    if sway > 0.0 {
        let world = output.world_v.xyz
            + slot_wind(model, output.world_v.xyz, sway, camera.position.w);
        output.world_v = vec4<f32>(world, output.world_v.w);
        output.clip_position = camera.view_projection * vec4<f32>(world, 1.0);
    }
    if GROUNDING {
        output.material = vec4<u32>(grounded(slot.color, index, model[3].xyz,
            output.world_v.xyz), output.material.zw);
    }
    return output;
}

// Packed color `color` of slot `index`, whose origin is `origin`, at
// `world`: where its material has grounding, multiplied by
// its ground's stain at the plane of the ground under its base, the stain
// fading out as far up as it reaches.
fn grounded(color: vec2<u32>, index: u32, origin: vec3<f32>, world: vec3<f32>) -> vec2<u32> {
    let ground = grounds[index];
    let reach = bitcast<f32>(ground.z);
    if !(reach > 0.0) {
        return color;
    }
    let tilt = unpack2x16snorm(ground.x);
    let up = sqrt(max(1.0 - dot(tilt, tilt), 0.01));
    // Height above the plane through the ground over the sunk base.
    let above = world.y - origin.y - bitcast<f32>(ground.y)
        + dot(tilt, world.xz - origin.xz) / up;
    let stain = unpack4x8unorm(ground.w).rgb;
    let amount = 1.0 - smoothstep(0.0, reach, above);
    let linear = vec4<f32>(unpack2x16unorm(color.x), unpack2x16unorm(color.y));
    let stained = linear.rgb * mix(vec3<f32>(1.0), stain, amount);
    return vec2<u32>(pack2x16unorm(stained.rg), pack2x16unorm(vec2<f32>(stained.b, linear.a)));
}

// The offset the wind gives the vertex at `world` of a slot posed by
// `model` that sways `sway`, at time `time`.
fn slot_wind(model: mat4x4<f32>, world: vec3<f32>, sway: f32, time: f32) -> vec3<f32> {
    return wind_sway(camera.wind.xy, camera.wind.z, model[3].xyz, world.y - model[3].y, sway,
        time);
}

// A selected cluster of a slot: 384 vertices, three per
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
    let index = slot_index(input.slot);
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
    let slot = slot_index(input.slot);
    let model = pose_matrix(poses[slot]);
    let material = colors[slot];
    let vertex = transform_vertex(corner.position, corner.normal, material.color,
        material.surface, corner.uv, model);
    let point = vec4<f32>(corner.position, 1.0);
    var world = model * point;
    // Only moving slots read their previous pose, as `vertex_affine_temporal`.
    if slot_moved(input.slot) {
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

// The visibility pass: a selected cluster's triangles, posed
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
    let model = pose_matrix(poses[slot_index(input.slot)]);
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
fn vs_affine(input: AffineVertexInput) -> VertexOutput {
    return vertex_affine(input);
}

@vertex
fn vs_affine_temporal(input: AffineVertexInput) -> TemporalOutput {
    return vertex_affine_temporal(input);
}

// Meshes with vertex colors or second texture coordinates
// read them from a second vertex stream, and their outputs carry both.
struct ExtendedOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    // Linear vertex color and opacity, and second texture coordinates.
    @location(3) color: vec4<f32>,
    @location(4) uv1: vec2<f32>,
}

struct ExtendedTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
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

// Linear blend skinning: the joints' palette matrices
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
    let index = slot_index(input.slot);
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
    let slot = slot_index(input.slot);
    let material = colors[slot];
    var model = pose_matrix(poses[slot]);
    var previous_model = model;
    if material.flags != 0u {
        model = skin_matrix(material.flags, joints, weights);
        previous_model = previous_skin_matrix(material.flags, joints, weights);
    } else if slot_moved(input.slot) {
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
    let slot = slot_index(input.slot);
    let model = pose_matrix(poses[slot]);
    let material = colors[slot];
    var vertex = transform_vertex(input.position, input.normal, material.color,
        material.surface, input.uv, model);
    vertex.material.w = slot_fade(input.slot);
    let point = vec4<f32>(input.position, 1.0);
    var world = model * point;
    // Reading previous poses costs more than the rest of the motion: only
    // moving slots do.
    if slot_moved(input.slot) {
        let pose = previous_poses[slot];
        world = vec4<f32>(dot(pose.row_0, point), dot(pose.row_1, point),
            dot(pose.row_2, point), 1.0);
    }
    // Swaying slots move with the wind, last frame's at its time.
    let sway = slot_sway(material.flags);
    if sway > 0.0 {
        let current = vertex.world_v.xyz
            + slot_wind(model, vertex.world_v.xyz, sway, camera.position.w);
        vertex.world_v = vec4<f32>(current, vertex.world_v.w);
        vertex.clip_position = camera.view_projection * vec4<f32>(current, 1.0);
        world = vec4<f32>(world.xyz + slot_wind(model, world.xyz, sway, camera.wind.w), 1.0);
    }
    if GROUNDING {
        vertex.material = vec4<u32>(grounded(material.color, slot, model[3].xyz,
            vertex.world_v.xyz), vertex.material.zw);
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
    output.material = vec4<u32>(color, surface, 0u);
    return output;
}
