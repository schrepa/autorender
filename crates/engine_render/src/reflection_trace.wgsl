@group(1) @binding(0) var<uniform> params: QueueParams;
@group(1) @binding(1) var<storage, read> rays: array<vec2<u32>>;
@group(1) @binding(2) var<storage, read> metadata: array<vec4<u32>, 3>;
@group(1) @binding(3) var<storage, read_write> raw: array<vec4<f32>>;

@compute @workgroup_size(64)
fn trace(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let bin = params.extra.x;
    let index = (group.y * metadata[2][bin] + group.x) * 64u + lane;
    if index >= metadata[0][bin] { return; }
    let texel = rays[metadata[1][bin] + index];
    raw[texel.y * params.size.x + texel.x] = reflection_at(vec2<i32>(texel));
}
