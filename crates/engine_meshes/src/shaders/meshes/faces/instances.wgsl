import package::engine::mesh::declarations::{faded_entry_fade, faded_slot_slot};
import package::face::impostors::ImpostorRecord;
import package::engine::shading::ViewPixel;
import package::meshes::cards::{card_debug, card_depth, placed_card, placed_card_lit,
    placed_card_point};
import package::meshes::forward::drawn_pixel;
import package::meshes::slots::instance_poses;
import package::place::instances::{AffinePose, CardDebugTargets, CardShaded, CardVertex,
    SlotColor};

fn instance_pose(slot: u32) -> AffinePose {
    return instance_poses[slot];
}

fn placed_card_vertex(vertex: u32, batch: u32, pose: AffinePose, previous: AffinePose,
        color: SlotColor, impostor: ImpostorRecord) -> CardVertex {
    return placed_card(vertex, batch, pose, previous, color, impostor);
}

fn placed_card_shade(input: CardVertex, impostor: ImpostorRecord) -> CardShaded {
    return placed_card_lit(input, impostor);
}

fn placed_card_depth(input: CardVertex, impostor: ImpostorRecord) -> f32 {
    return card_depth(input, placed_card_point(input, impostor), false);
}

fn placed_card_pixel(input: CardVertex) -> ViewPixel {
    return drawn_pixel(input.clip_position, input.previous);
}

fn placed_card_debug(input: CardVertex, record: u32, object: u32, impostor: ImpostorRecord)
        -> CardDebugTargets {
    return card_debug(input, placed_card_point(input, impostor), vec2<u32>(record, object),
        faded_slot_slot(input.batch), faded_entry_fade(input.batch), false);
}
