import package::engine::drawn::drawn_shown;
import package::engine::view::{camera, stable_normalize, view_clip};
import package::generated::plates::PlateInstance;
@group(2) @binding(0) var<storage, read> plates: array<PlateInstance>;

const PLATE_WHITE: u32 = 0xffffffffu;
const PLATE_ROUGH: u32 = 0xffu;
const PLATE_HIDDEN: vec4<f32> = vec4<f32>(2.0, 2.0, 2.0, 1.0);

struct PlateCorner {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(6) @interpolate(flat) object: vec4<u32>,
}

fn corner_unit(corner: u32) -> vec2<f32> {
    return vec2<f32>(f32((50u >> corner) & 1u), f32((44u >> corner) & 1u));
}

@vertex fn vs_plate(@builtin(vertex_index) corner: u32, @builtin(instance_index) index: u32)
        -> PlateCorner {
    let plate = plates[index];
    var out: PlateCorner;
    if !drawn_shown(plate.drawn.x) {
        out.clip_position = PLATE_HIDDEN;
        return out;
    }
    let uv = corner_unit(corner);
    let offset = plate.across.xyz * (uv.x - 0.5) + plate.along.xyz * (uv.y - 0.5);
    let world = plate.center.xyz + offset;
    let shown = camera.previous_view_projection * vec4<f32>(plate.previous.xyz + offset, 1.0);
    out.clip_position = view_clip(world);
    out.normal_u = vec4<f32>(stable_normalize(cross(plate.along.xyz, plate.across.xyz)), uv.x);
    out.world_v = vec4<f32>(world, uv.y);
    out.material = vec4<u32>(PLATE_WHITE, PLATE_WHITE, PLATE_ROUGH, 0u);
    out.previous = shown.xyw;
    out.object = vec4<u32>(0u, plate.drawn.x, 0u, 0u);
    return out;
}
