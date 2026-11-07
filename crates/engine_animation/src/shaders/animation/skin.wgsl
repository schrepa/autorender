import package::generated::animation::{JointRows, RestVertex, SKIN_GROUP, SkinFrame, SkinPart};

struct SkinnedVertex {
    x: f32,
    y: f32,
    z: f32,
    normal: u32,
    u: f32,
    v: f32,
    was_x: f32,
    was_y: f32,
    was_z: f32,
    slot: u32,
}

@group(1) @binding(0) var<uniform> frame: SkinFrame;
@group(1) @binding(1) var<storage, read> parts: array<SkinPart>;
@group(1) @binding(2) var<storage, read> owners: array<u32>;
@group(1) @binding(3) var<storage, read> rest: array<RestVertex>;
@group(1) @binding(4) var<storage, read> rows: array<JointRows>;
@group(1) @binding(5) var<storage, read_write> skinned: array<SkinnedVertex>;

fn blended(start: u32, joints: vec4<u32>, weights: vec4<f32>) -> JointRows {
    var mixed = JointRows(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    for (var at = 0u; at < 4u; at++) {
        let held = rows[start + joints[at]];
        mixed.r0 += held.r0 * weights[at];
        mixed.r1 += held.r1 * weights[at];
        mixed.r2 += held.r2 * weights[at];
    }
    return mixed;
}

fn octahedral(normal: vec3<f32>) -> u32 {
    let n = normal / max(abs(normal.x) + abs(normal.y) + abs(normal.z), 1.0e-6);
    let sides = select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0));
    return pack2x16snorm(select(n.xy, (vec2<f32>(1.0) - abs(n.yx)) * sides, n.z < 0.0));
}

fn turned(pose: JointRows, normal: vec3<f32>) -> vec3<f32> {
    let across_x = cross(pose.r1.xyz, pose.r2.xyz);
    let across_y = cross(pose.r2.xyz, pose.r0.xyz);
    let across_z = cross(pose.r0.xyz, pose.r1.xyz);
    let facing = select(-1.0, 1.0, dot(pose.r0.xyz, across_x) >= 0.0);
    return vec3<f32>(dot(across_x, normal), dot(across_y, normal), dot(across_z, normal)) * facing;
}

@compute @workgroup_size(SKIN_GROUP)
fn skin_vertices(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.vertices {
        return;
    }
    let slot = owners[id.x];
    let part = parts[slot];
    let source = rest[part.rest_at + id.x - part.skinned_at];
    let point = vec4<f32>(source.position_u.xyz, 1.0);
    let now = blended(part.rows_at, source.joints, source.weights);
    let was = blended(part.previous_at, source.joints, source.weights);
    skinned[id.x] = SkinnedVertex(
        dot(now.r0, point), dot(now.r1, point), dot(now.r2, point),
        octahedral(turned(now, source.normal_v.xyz)),
        source.position_u.w, source.normal_v.w,
        dot(was.r0, point), dot(was.r1, point), dot(was.r2, point),
        slot);
}
