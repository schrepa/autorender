import package::engine::drawn::{drawn_model, drawn_slot};
import package::engine::view::{stable_normalize, view_clip};

const PLATE_WHITE: u32 = 0xffffffffu;
const PLATE_ROUGH: u32 = 0xffu;

struct PlateCorner {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
}

fn corner_unit(corner: u32) -> vec2<f32> {
    return vec2<f32>(f32((50u >> corner) & 1u), f32((44u >> corner) & 1u));
}

@vertex fn vs_plate(@builtin(vertex_index) corner: u32, @builtin(instance_index) index: u32)
        -> PlateCorner {
    let pose = drawn_model(drawn_slot(index));
    let uv = corner_unit(corner);
    let offset = pose[0].xyz * (uv.x - 0.5) + pose[2].xyz * (uv.y - 0.5);
    let world = pose[3].xyz + offset;
    var out: PlateCorner;
    out.clip_position = view_clip(world);
    out.normal_u = vec4<f32>(stable_normalize(cross(pose[2].xyz, pose[0].xyz)), uv.x);
    out.world_v = vec4<f32>(world, uv.y);
    out.material = vec4<u32>(PLATE_WHITE, PLATE_WHITE, PLATE_ROUGH, 0u);
    return out;
}
