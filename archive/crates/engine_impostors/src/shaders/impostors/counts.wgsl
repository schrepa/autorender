import package::face::instances::{last_count, last_fading_count, last_late_count};
import package::place::impostors::IMPOSTOR_VERTICES;

struct CardArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

@group(1) @binding(0) var<storage, read_write> card_args: array<CardArgs, 2>;
@group(1) @binding(1) var<storage, read_write> late_card_args: CardArgs;

@compute @workgroup_size(1)
fn count_cards() {
    card_args[0] = CardArgs(IMPOSTOR_VERTICES, last_count(), 0u, 0u);
    card_args[1] = CardArgs(IMPOSTOR_VERTICES, last_fading_count(), 0u, 0u);
}

@compute @workgroup_size(1)
fn count_late_cards() {
    late_card_args = CardArgs(IMPOSTOR_VERTICES, last_late_count(), 0u, 0u);
}
