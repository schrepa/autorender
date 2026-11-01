import package::engine::mesh::declarations::AffineVertexInput;
import package::engine::shading::{Shaded, ViewPixel};
import package::face::debug::DebugTargets;
import package::meshes::cards::{card_debug, placed_card, placed_card_lit, placed_card_point};
import package::meshes::debug::point_targets;
import package::meshes::forward::drawn_pixel;
import package::meshes::shade::{masked_depth, placed_fragment, shade, surface_point};
import package::meshes::slots::{instance_poses, instance_still};
import package::meshes::vertex::{pose_origin, slot_temporal};
import package::place::instances::{AffinePose, CardDebugTargets, CardShaded, CardVertex,
    SlotColor, SlotExtras, SlotFragment, SlotVertex, SlotVertexExtended};

fn instance_pose(slot: u32) -> AffinePose {
    return instance_poses[slot];
}

fn instance_still_frames(slot: u32) -> u32 {
    return instance_still[slot].y;
}

fn slot_vertex(input: AffineVertexInput, pose: AffinePose, previous: AffinePose,
        color: SlotColor, ground: vec4<u32>, fade: u32) -> SlotVertex {
    let vertex = slot_temporal(input, pose, previous, color, ground, fade);
    return SlotVertex(vertex.clip_position, vertex.normal_u, vertex.world_v, vertex.material,
        vertex.previous, ground, pose_origin(pose));
}

fn slot_vertex_extended(input: AffineVertexInput, extras: SlotExtras, pose: AffinePose,
        previous: AffinePose, color: SlotColor, ground: vec4<u32>, fade: u32)
        -> SlotVertexExtended {
    let vertex = slot_temporal(input, pose, previous, color, ground, fade);
    return SlotVertexExtended(vertex.clip_position, vertex.normal_u, vertex.world_v,
        vertex.material, vertex.previous, extras.color, extras.uv1, ground, pose_origin(pose));
}

fn slot_fragment(input: SlotVertex) -> SlotFragment {
    return placed_fragment(input.clip_position, input.normal_u, input.world_v, input.material,
        input.ground, input.origin);
}

fn slot_fragment_extended(input: SlotVertexExtended) -> SlotFragment {
    var values = placed_fragment(input.clip_position, input.normal_u, input.world_v,
        input.material, input.ground, input.origin);
    values.color *= input.color;
    values.uv1 = input.uv1;
    return values;
}

fn slot_shade(fragment: SlotFragment, front: bool, pretested: bool) -> Shaded {
    return shade(fragment, front, pretested);
}

fn slot_masked_depth(fragment: SlotFragment) {
    masked_depth(fragment);
}

fn slot_pixel(input: SlotVertex) -> ViewPixel {
    return drawn_pixel(input.clip_position, input.previous);
}

fn slot_pixel_extended(input: SlotVertexExtended) -> ViewPixel {
    return drawn_pixel(input.clip_position, input.previous);
}

fn slot_debug(fragment: SlotFragment, front: bool, record: u32, object: u32, detail: u32)
        -> DebugTargets {
    return point_targets(record, object, detail, surface_point(fragment, front, true));
}

fn placed_card_vertex(vertex: u32, batch: u32, pose: AffinePose, previous: AffinePose,
        color: SlotColor) -> CardVertex {
    return placed_card(vertex, batch, pose, previous, color);
}

fn placed_card_shade(input: CardVertex) -> CardShaded {
    return placed_card_lit(input);
}

fn placed_card_pixel(input: CardVertex) -> ViewPixel {
    return drawn_pixel(input.clip_position, input.previous);
}

fn placed_card_debug(input: CardVertex, record: u32, object: u32) -> CardDebugTargets {
    return card_debug(input, placed_card_point(input), record, object);
}
