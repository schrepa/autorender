import package::generated::probes::PROBE_ENTRY_WORDS;
import package::probes::sample::probe_words;

fn in_interior(world: vec3<f32>) -> bool {
    let header = probe_words[0];
    if header.w == 0u {
        return false;
    }
    let grid = probe_words[1];
    let cell = vec2<i32>(floor((world.xz - bitcast<vec2<f32>>(header.xy))
        / bitcast<f32>(header.z)));
    if any(cell < vec2<i32>(0)) || any(vec2<u32>(cell) >= grid.xy) {
        return false;
    }
    let list = probe_words[grid.z + u32(cell.y) * grid.x + u32(cell.x)];
    let count = select(0u, list.y, list.z != 0u);
    for (var n = 0u; n < count; n++) {
        let entry = grid.w + (list.x + n) * PROBE_ENTRY_WORDS;
        if probe_words[entry + 2u].z == 0u {
            continue;
        }
        let center = bitcast<vec3<f32>>(probe_words[entry].xyz);
        let extent = bitcast<vec3<f32>>(probe_words[entry + 1u].xyz);
        if all(abs(world - center) < extent) {
            return true;
        }
    }
    return false;
}

fn interior(inside: bool, world: vec3<f32>) -> bool {
    return inside || in_interior(world);
}
