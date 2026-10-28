fn pose_matrix(pose: AffinePose) -> mat4x4<f32> {
    return transpose(mat4x4<f32>(pose.row_0, pose.row_1, pose.row_2,
        vec4<f32>(0.0, 0.0, 0.0, 1.0)));
}

fn origin_key(origin: vec3<f32>) -> u32 {
    let cell = bitcast<vec2<u32>>(vec2<i32>(floor(origin.xz * 4.0)));
    var h = (cell.x * 0x8da6b343u) ^ (cell.y * 0xd8163841u);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return (h ^ (h >> 16u)) & 0xffff0000u;
}

fn output_color(material: vec4<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16unorm(material.x), unpack2x16unorm(material.y));
}

fn octahedral_normal(encoded: vec2<f32>) -> vec3<f32> {
    let z = 1.0 - abs(encoded.x) - abs(encoded.y);
    let fold = max(-z, 0.0);
    let shift = vec2<f32>(select(-fold, fold, encoded.x >= 0.0),
        select(-fold, fold, encoded.y >= 0.0));
    return vec3<f32>(encoded - shift, z);
}

fn vertex_affine(input: AffineVertexInput) -> VertexOutput {
    let index = slot_index(input.slot);
    let model = pose_matrix(poses[index]);
    let slot = colors[index];
    var output = transform_vertex(input.position, input.normal, slot.color, slot.surface,
        input.uv, model);
    output.material.w = slot_fade(input.slot) | origin_key(model[3].xyz);
    let sway = slot_sway(slot.flags);
    if sway > 0.0 {
        let world = output.world_v.xyz
            + vertex_wind(model, input.position, output.world_v.xyz, sway, camera.position.w);
        output.world_v = vec4<f32>(world, output.world_v.w);
        output.clip_position = camera.view_projection * vec4<f32>(world, 1.0);
    }
    if GROUNDING {
        output.material = placed_material(output.material, index);
    }
    return output;
}

fn placed_material(material: vec4<u32>, slot: u32) -> vec4<u32> {
    return vec4<u32>(pack4x8unorm(output_color(material)), slot, material.zw);
}

fn vertex_wind(model: mat4x4<f32>, position: vec3<f32>, world: vec3<f32>, sway: f32,
        time: f32) -> vec3<f32> {
    let root = model[3].xyz;
    return wind_sway(vec4<f32>(camera.wind.xyz, time), 1.0 / max(detail_across(root), 1.0e-6),
        root, world.y - root.y, position, sway);
}

fn slot_wind(model: mat4x4<f32>, world: vec3<f32>, sway: f32, time: f32) -> vec3<f32> {
    return vertex_wind(model, vec3<f32>(0.0), world, sway, time);
}

@vertex
fn vs_affine(input: AffineVertexInput) -> VertexOutput {
    return vertex_affine(input);
}

@vertex
fn vs_affine_temporal(input: AffineVertexInput) -> TemporalOutput {
    return vertex_affine_temporal(input);
}

struct ExtendedOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) color: vec4<f32>,
    @location(4) uv1: vec2<f32>,
}

struct ExtendedTemporalOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
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

fn vertex_affine_temporal(input: AffineVertexInput) -> TemporalOutput {
    let slot = slot_index(input.slot);
    let model = pose_matrix(poses[slot]);
    let material = colors[slot];
    var vertex = transform_vertex(input.position, input.normal, material.color,
        material.surface, input.uv, model);
    vertex.material.w = slot_fade(input.slot) | origin_key(model[3].xyz);
    let point = vec4<f32>(input.position, 1.0);
    var world = model * point;
    if slot_moved(input.slot) {
        let pose = previous_poses[slot];
        world = vec4<f32>(dot(pose.row_0, point), dot(pose.row_1, point),
            dot(pose.row_2, point), 1.0);
    }
    let sway = slot_sway(material.flags);
    if sway > 0.0 {
        let current = vertex.world_v.xyz
            + vertex_wind(model, input.position, vertex.world_v.xyz, sway, camera.position.w);
        vertex.world_v = vec4<f32>(current, vertex.world_v.w);
        vertex.clip_position = camera.view_projection * vec4<f32>(current, 1.0);
        world = vec4<f32>(world.xyz
            + vertex_wind(model, input.position, world.xyz, sway, camera.wind.w), 1.0);
    }
    if GROUNDING {
        vertex.material = placed_material(vertex.material, slot);
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
    let inverse_scale = 1.0 / max(maximum_abs(model[0].xyz),
        max(maximum_abs(model[1].xyz), maximum_abs(model[2].xyz)));
    let x = model[0].xyz * inverse_scale;
    let y = model[1].xyz * inverse_scale;
    let z = model[2].xyz * inverse_scale;
    let normal = octahedral_normal(encoded_normal);
    let yz = cross(y, z);
    let transformed = yz * normal.x + cross(z, x) * normal.y + cross(x, y) * normal.z;
    var output: VertexOutput;
    output.clip_position = camera.view_projection * world_position;
    output.normal_u = vec4<f32>(stable_normalize(transformed) * sign(dot(x, yz)), uv.x);
    output.world_v = vec4<f32>(world_position.xyz, uv.y);
    output.material = vec4<u32>(color, surface, 0u);
    return output;
}
