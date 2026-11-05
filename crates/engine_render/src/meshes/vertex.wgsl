import package::engine::math::{FIRST_PERSON_DEPTH, maximum_abs};
import package::engine::view::{camera, stable_normalize};
import package::engine::mesh::declarations::{AffinePose, VertexOutput};

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

fn placed_material(material: vec4<u32>, slot: u32) -> vec4<u32> {
    return vec4<u32>(pack4x8unorm(output_color(material)), slot, material.zw);
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

fn first_person(flags: u32) -> bool {
    return bitcast<f32>(flags) < 0.0;
}

fn first_person_scale(rise: vec3<f32>, flags: u32) -> f32 {
    let focal = -bitcast<f32>(flags) - 1.0;
    if focal <= 0.0 {
        return 1.0;
    }
    return focal / length(rise);
}

fn first_person_clip(clip: vec4<f32>, flags: u32) -> vec4<f32> {
    let lens = camera.view_projection;
    let rise = vec3<f32>(lens[0].y, lens[1].y, lens[2].y)
        - camera.jitter.y * vec3<f32>(lens[0].w, lens[1].w, lens[2].w);
    let center = camera.jitter.xy * clip.w;
    let across = (clip.xy - center) * first_person_scale(rise, flags) + center;
    return vec4<f32>(across, mix(clip.w, clip.z, FIRST_PERSON_DEPTH), clip.w);
}

fn first_person_previous(shown: vec4<f32>, flags: u32) -> vec3<f32> {
    let lens = camera.previous_view_projection;
    let rise = vec3<f32>(lens[0].y, lens[1].y, lens[2].y);
    return vec3<f32>(shown.xy * first_person_scale(rise, flags), shown.w);
}
