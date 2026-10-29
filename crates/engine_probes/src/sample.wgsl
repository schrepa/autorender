const PROBE_ENTRY_WORDS: u32 = 3u;
@group(0) @binding(13) var<storage, read> probe_words: array<vec4<u32>>;

override PROBES: bool = true;

fn box_projected(position: vec3<f32>, direction: vec3<f32>, center: vec3<f32>,
        extent: vec3<f32>) -> vec3<f32> {
    let safe = select(direction, vec3<f32>(1.0e-6), abs(direction) < vec3<f32>(1.0e-6));
    let exits = max((center - extent - position) / safe, (center + extent - position) / safe);
    return position + direction * min(exits.x, min(exits.y, exits.z)) - center;
}
