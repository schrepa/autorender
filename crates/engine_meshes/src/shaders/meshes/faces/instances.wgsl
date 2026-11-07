import package::engine::mesh::declarations::AffineVertexInput;
import package::engine::mesh::impostor_frames::ImpostorRecord;
import package::engine::shading::ViewPixel;
import package::meshes::cards::{card_debug, card_depth, placed_card, placed_card_lit,
    placed_card_point};
import package::meshes::forward::drawn_pixel;
import package::meshes::shade::{masked_depth, placed_fragment, shade};
import package::meshes::slots::{instance_colors, instance_poses};
import package::meshes::vertex::slot_motion;
import package::place::instances::{AffinePose, CardDebugTargets, CardShaded, CardVertex,
    SlotColor, SlotFragment, SlotShaded, SlotVertex};

fn instance_pose(slot: u32) -> AffinePose {
    return instance_poses[slot];
}

fn instance_color(slot: u32) -> SlotColor {
    return instance_colors[slot];
}

fn slot_vertex(input: AffineVertexInput, pose: AffinePose, previous: AffinePose,
        color: SlotColor, fade: u32) -> SlotVertex {
    let vertex = slot_motion(input, pose, previous, color, fade);
    return SlotVertex(vertex.clip_position, vertex.normal_u, vertex.world_v, vertex.material,
        vertex.previous);
}

fn slot_fragment(input: SlotVertex) -> SlotFragment {
    return placed_fragment(input.clip_position, input.normal_u, input.world_v, input.material);
}

fn slot_shade(fragment: SlotFragment, front: bool) -> SlotShaded {
    return shade(fragment, front);
}

fn slot_masked_depth(fragment: SlotFragment) {
    masked_depth(fragment);
}

fn slot_pixel(input: SlotVertex) -> ViewPixel {
    return drawn_pixel(input.clip_position, input.previous);
}

fn placed_card_vertex(vertex: u32, batch: u32, pose: AffinePose, previous: AffinePose,
        color: SlotColor, impostor: ImpostorRecord) -> CardVertex {
    return placed_card(vertex, batch, pose, previous, color, impostor);
}

fn placed_card_shade(input: CardVertex, impostor: ImpostorRecord) -> CardShaded {
    return placed_card_lit(input, impostor);
}

fn placed_card_depth(input: CardVertex, impostor: ImpostorRecord) -> f32 {
    return card_depth(input, placed_card_point(input, impostor));
}

fn placed_card_pixel(input: CardVertex) -> ViewPixel {
    return drawn_pixel(input.clip_position, input.previous);
}

fn placed_card_debug(input: CardVertex, record: u32, object: u32, impostor: ImpostorRecord)
        -> CardDebugTargets {
    return card_debug(input, placed_card_point(input, impostor), record, object);
}
