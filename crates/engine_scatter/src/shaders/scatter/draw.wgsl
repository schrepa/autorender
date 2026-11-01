import package::engine::mesh::declarations::AffineVertexInput;
import package::face::instances::{slot_fragment, slot_masked_depth, slot_pixel, slot_shade, slot_vertex};
import package::generated::targets::{ViewTargets, view_targets};
import package::place::instances::{AffinePose, FADE_SLOT, FADE_WORD_ON, SlotColor, SlotVertex};

@group(2) @binding(0) var<storage, read> poses: array<AffinePose>;
@group(2) @binding(1) var<storage, read> colors: array<SlotColor>;
@group(2) @binding(2) var<storage, read> grounds: array<vec4<u32>>;

fn placed_vertex(input: AffineVertexInput, slot: u32, fade: u32) -> SlotVertex {
    let pose = poses[slot];
    return slot_vertex(input, pose, pose, colors[slot], grounds[slot], fade);
}

@vertex
fn vs_object(input: AffineVertexInput) -> SlotVertex {
    return placed_vertex(input, input.slot, 0u);
}

@vertex
fn vs_object_fading(input: AffineVertexInput) -> SlotVertex {
    return placed_vertex(input, input.slot & FADE_SLOT, (input.slot >> 24u) | FADE_WORD_ON);
}

@fragment
fn fs_object(input: SlotVertex, @builtin(front_facing) front: bool) -> ViewTargets {
    return view_targets(slot_shade(slot_fragment(input), front), slot_pixel(input));
}

@fragment
fn fs_masked_depth(input: SlotVertex) {
    slot_masked_depth(slot_fragment(input));
}
