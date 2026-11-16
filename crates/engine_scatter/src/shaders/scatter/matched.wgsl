import package::engine::kinds::{KindFragment, drawn_pixel, extra_values, fragment_values,
    object_fragment, surface_input};
import package::engine::slots::{shade, slot_output};
import package::engine::surface::surface_point;
import package::generated::writes::{Outputs, outputs};
import package::scatter::draw::{PlacedExtendedVertex, PlacedVertex};

fn shaded_outputs(values: KindFragment, object: vec4<u32>, front: bool, clip_position: vec4<f32>,
        previous: vec3<f32>) -> Outputs {
    let placed = object_fragment(values, object);
    let surface = surface_input(placed, front);
    let point = surface_point(surface);
    return outputs(point, slot_output(surface, point, shade(point, placed.pixel)),
        drawn_pixel(clip_position, previous));
}

@fragment
fn fs_object_matched(input: PlacedVertex, @builtin(front_facing) front: bool) -> Outputs {
    let values = fragment_values(input.clip_position, input.normal_u, input.world_v,
        input.material);
    return shaded_outputs(values, input.object, front, input.clip_position, input.previous);
}

@fragment
fn fs_object_matched_extended(input: PlacedExtendedVertex, @builtin(front_facing) front: bool)
        -> Outputs {
    let values = extra_values(input.clip_position, input.normal_u, input.world_v, input.material,
        input.color, input.uv1);
    return shaded_outputs(values, input.object, front, input.clip_position, input.previous);
}
