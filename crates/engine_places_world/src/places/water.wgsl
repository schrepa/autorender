const WATER_HELD: f32 = 0.5;

struct WaterWords {
    flow: vec2<f32>,
    span: f32,
    lake: bool,
    sheet: bool,
    state: vec3<f32>,
    reach: u32,
}

struct WaterLevel {
    surface: f32,
    slope: vec2<f32>,
    depth: f32,
    held: f32,
    held_rate: vec2<f32>,
    shares: mat4x4<f32>,
}

struct WaterAt {
    surface: f32,
    depth: f32,
    held: f32,
    words: WaterWords,
    slope: vec2<f32>,
}

struct WaterCell {
    level: u32,
    block: u32,
    first: vec2<u32>,
    within: vec2<f32>,
}

fn water_info_at(texel: vec3<i32>) -> vec4<u32> {
    return vec4<u32>(0u);
}

fn water_texel(block: u32, sample: vec2<u32>) -> vec3<i32> {
    return vec3<i32>(0);
}

fn water_words(words: vec4<u32>) -> WaterWords {
    var found: WaterWords;
    return found;
}

fn water_cell(local: vec2<f32>, level: u32) -> WaterCell {
    var cell: WaterCell;
    return cell;
}

fn water_level(cell: WaterCell) -> WaterLevel {
    var level: WaterLevel;
    return level;
}

fn water_smooth(surfaces: array<vec2<f32>, 16>, within: vec2<f32>, spacing: f32) -> WaterLevel {
    var level: WaterLevel;
    return level;
}

fn water_read(cell: WaterCell) -> WaterAt {
    var at: WaterAt;
    return at;
}

fn water_at_level(level: WaterLevel) -> WaterAt {
    var at: WaterAt;
    return at;
}

fn water_words_add(at: WaterAt, words: WaterWords, share: f32, nearest: bool) -> WaterAt {
    return at;
}

fn water_level_at(local: vec2<f32>) -> WaterLevel {
    var level: WaterLevel;
    return level;
}
