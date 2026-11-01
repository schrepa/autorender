const MOVED_SLOT: u32 = 0x80000000u;
const FADE_SLOT: u32 = 0x00ffffffu;
const FADE_WORD_ON: u32 = 0x100u;
const FADE_WORD_OUT: u32 = 0x80u;
const STILL_FRAMES: u32 = 16u;

struct AffinePose {
    row_0: vec4<f32>,
    row_1: vec4<f32>,
    row_2: vec4<f32>,
}

struct SlotColor {
    color: vec2<u32>,
    surface: u32,
    flags: u32,
}
