import package::engine::mesh::declarations::AffineVertexInput;
import package::engine::view::camera;
import package::face::instances::{AffinePose, SlotVertex, instance_color, slot_fragment,
    slot_masked_depth, slot_pixel, slot_shade, slot_vertex};
import package::generated::targets::{ViewTargets, view_targets};

fn unposed() -> AffinePose {
    return AffinePose(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0));
}

@vertex
fn vs_skinned(input: AffineVertexInput, @location(3) previous: vec3<f32>) -> SlotVertex {
    var vertex = slot_vertex(input, unposed(), unposed(), instance_color(input.slot), 0u);
    let shown = camera.previous_view_projection * vec4<f32>(previous, 1.0);
    vertex.previous = shown.xyw;
    return vertex;
}

@fragment
fn fs_skinned(input: SlotVertex, @builtin(front_facing) front: bool) -> ViewTargets {
    let shaded = slot_shade(slot_fragment(input), front);
    return view_targets(shaded.point, shaded.shaded, slot_pixel(input));
}

@fragment
fn fs_skinned_depth(input: SlotVertex) {
    slot_masked_depth(slot_fragment(input));
}
