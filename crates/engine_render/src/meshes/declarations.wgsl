import package::engine::materials::{FADE_LIST, MASK_MIP_SCALE};
import package::engine::view::fade_noise;

struct AffinePose {
    row_0: vec4<f32>,
    row_1: vec4<f32>,
    row_2: vec4<f32>,
}

const MOVED_SLOT: u32 = 0x80000000u;

const FADE_SLOT: u32 = 0x00ffffffu;
const FADE_WORD_ON: u32 = 0x100u;
const FADE_WORD_OUT: u32 = 0x80u;

fn slot_index(entry: u32) -> u32 {
    if FADE_LIST {
        return entry & FADE_SLOT;
    }
    return entry & ~MOVED_SLOT;
}

fn slot_moved(entry: u32) -> bool {
    return !FADE_LIST && (entry & MOVED_SLOT) != 0u;
}

fn slot_fade(entry: u32) -> u32 {
    return select(0u, (entry >> 24u) | FADE_WORD_ON, FADE_LIST);
}

fn fade_hidden(fade: u32, position: vec2<f32>) -> bool {
    if (fade & FADE_WORD_ON) == 0u {
        return false;
    }
    let shows = fade_noise(position) < f32(fade & 0x7fu) / 127.0;
    return shows == ((fade & FADE_WORD_OUT) != 0u);
}

struct AffineVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec2<f32>,
    @location(2) slot: u32,
    @location(7) uv: vec2<f32>,
}

struct VertexOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
}

struct TemporalOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
}
