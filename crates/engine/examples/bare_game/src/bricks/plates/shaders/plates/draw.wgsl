import package::engine::color::encode_srgb;
import package::engine::drawn::{drawn_model, drawn_slot};
import package::engine::view::view_clip;
import package::plates::checker::checker;

struct PlateCorner {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

fn corner_unit(corner: u32) -> vec2<f32> {
    return vec2<f32>(f32((52u >> corner) & 1u), f32((26u >> corner) & 1u));
}

@vertex fn vs_plate(@builtin(vertex_index) corner: u32, @builtin(instance_index) index: u32)
        -> PlateCorner {
    let pose = drawn_model(drawn_slot(index));
    let uv = corner_unit(corner);
    let offset = pose[0].xyz * (uv.x - 0.5) + pose[2].xyz * (uv.y - 0.5);
    var out: PlateCorner;
    out.clip_position = view_clip(pose[3].xyz + offset);
    out.uv = uv;
    return out;
}

@fragment fn fs_plate(corner: PlateCorner) -> @location(0) vec4<f32> {
    return vec4<f32>(encode_srgb(checker(corner.uv)), 1.0);
}
