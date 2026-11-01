import package::generated::debug_view::ProbedTargets;

@group(1) @binding(0) var<uniform> probe_pixel: vec4<u32>;
@group(1) @binding(1) var<storage, read_write> probed: ProbedTargets;
@group(1) @binding(2) var probe_ids: texture_2d<u32>;
@group(1) @binding(3) var probe_count: texture_2d<f32>;
@group(1) @binding(4) var probe_terrain: texture_2d<u32>;
@group(1) @binding(5) var probe_depth: texture_depth_2d;

@compute @workgroup_size(1)
fn cs_probe_targets() {
    let last = max(textureDimensions(probe_ids), vec2<u32>(1u)) - 1u;
    let pixel = vec2<i32>(min(probe_pixel.xy, last));
    probed.ids = textureLoad(probe_ids, pixel, 0);
    probed.terrain = textureLoad(probe_terrain, pixel, 0).xy;
    probed.count = textureLoad(probe_count, pixel, 0).r;
    probed.depth = textureLoad(probe_depth, pixel, 0);
}
