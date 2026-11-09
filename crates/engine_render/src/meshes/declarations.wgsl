import package::engine::view::fade_noise;

struct AffinePose {
    row_0: vec4<f32>,
    row_1: vec4<f32>,
    row_2: vec4<f32>,
}

fn faded_entry_fade(entry: u32) -> u32 {
    return fade_word(faded_slot_level(entry), faded_slot_out(entry), true);
}

fn fade_hidden(fade: u32, position: vec2<f32>) -> bool {
    if !fade_word_on(fade) {
        return false;
    }
    let shows = fade_noise(position) < f32(fade_word_level(fade)) / f32(FADE_LEVEL_MAX);
    return shows == fade_word_out(fade);
}

struct VertexOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
}
