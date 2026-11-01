@group(0) @binding(0) var<storage, read> probe_words: array<vec4<u32>>;
@group(0) @binding(0) var probe_maps: texture_cube_array<f32>;
@group(0) @binding(0) var probe_sampler: sampler;

fn box_projected(position: vec3<f32>, direction: vec3<f32>, center: vec3<f32>,
        extent: vec3<f32>) -> vec3<f32> {
    let safe = select(direction, vec3<f32>(1.0e-6), abs(direction) < vec3<f32>(1.0e-6));
    let exits = max((center - extent - position) / safe, (center + extent - position) / safe);
    return position + direction * min(exits.x, min(exits.y, exits.z)) - center;
}

fn probe_sample(direction: vec3<f32>, cube: u32, level: f32) -> vec3<f32> {
    return textureSampleLevel(probe_maps, probe_sampler, direction, cube - 1u, level).rgb;
}
