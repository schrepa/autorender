import package::shadows::faces::routes::{caster_cull, caster_routes};

const CARD = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

@vertex
fn vs_clear_moving(@builtin(vertex_index) vertex: u32,
        @builtin(instance_index) tile: u32) -> @builtin(position) vec4<f32> {
    let slot = caster_routes[tile].slot;
    let texel = mix(slot.xy, slot.zw, CARD[vertex] * 0.5 + vec2<f32>(0.5));
    return vec4<f32>(texel.x * caster_cull.atlas.x - 1.0,
        1.0 - texel.y * caster_cull.atlas.y, 1.0, 1.0);
}

@group(1) @binding(0) var shadow_regions: texture_depth_2d_array;

struct ComposeOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) source: vec3<i32>,
}

@vertex
fn vs_compose(@builtin(vertex_index) vertex: u32, @location(3) place: vec4<f32>,
        @location(4) source: vec4<i32>) -> ComposeOutput {
    let texel = mix(place.xy, place.zw, CARD[vertex] * 0.5 + vec2<f32>(0.5));
    var output: ComposeOutput;
    output.clip = vec4<f32>(texel.x * caster_cull.atlas.x - 1.0,
        1.0 - texel.y * caster_cull.atlas.y, 1.0, 1.0);
    output.source = source.xyz;
    return output;
}

@fragment
fn fs_compose(input: ComposeOutput) -> @builtin(frag_depth) f32 {
    let texel = vec2<i32>(input.clip.xy) + input.source.yz;
    return textureLoad(shadow_regions, texel, input.source.x, 0);
}
