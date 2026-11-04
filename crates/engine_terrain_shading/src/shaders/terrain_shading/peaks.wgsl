import package::generated::terrain_shading::{PEAKS_GROUP, PEAKS_SIDE};

@group(1) @binding(0) var<storage, read> peak_layers: array<u32>;
@group(1) @binding(1) var peaks_in: texture_2d_array<f32>;
@group(1) @binding(2) var peaks_out: texture_storage_2d_array<r32float, write>;

@compute @workgroup_size(PEAKS_GROUP, PEAKS_GROUP)
fn tile_peaks_first(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= PEAKS_SIDE || id.y >= PEAKS_SIDE {
        return;
    }
    let layer = peak_layers[id.z];
    let first = 2u * id.xy + vec2<u32>(1u);
    var peak = textureLoad(peaks_in, first, layer, 0).r;
    for (var z = 0u; z < 3u; z++) {
        for (var x = 0u; x < 3u; x++) {
            peak = max(peak, textureLoad(peaks_in, first + vec2<u32>(x, z), layer, 0).r);
        }
    }
    textureStore(peaks_out, id.xy, layer, vec4<f32>(peak, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(PEAKS_GROUP, PEAKS_GROUP)
fn tile_peaks_down(@builtin(global_invocation_id) id: vec3<u32>) {
    let side = textureDimensions(peaks_out).x;
    if id.x >= side || id.y >= side {
        return;
    }
    let layer = peak_layers[id.z];
    let at = 2u * id.xy;
    let peak = max(
        max(textureLoad(peaks_in, at, layer, 0).r,
            textureLoad(peaks_in, at + vec2<u32>(1u, 0u), layer, 0).r),
        max(textureLoad(peaks_in, at + vec2<u32>(0u, 1u), layer, 0).r,
            textureLoad(peaks_in, at + vec2<u32>(1u, 1u), layer, 0).r));
    textureStore(peaks_out, id.xy, layer, vec4<f32>(peak, 0.0, 0.0, 0.0));
}
