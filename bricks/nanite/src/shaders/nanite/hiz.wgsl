import package::generated::nanite::HizBuildInfo;

@group(1) @binding(0) var<uniform> hiz_build_info: HizBuildInfo;
@group(1) @binding(1) var<storage, read> sw_depth_ro: array<u32>;
@group(1) @binding(2) var hiz_src: texture_2d<f32>;
@group(1) @binding(3) var hiz_dst: texture_storage_2d<r32float, write>;

fn footprint(coord: u32, dst_dim: u32, src_dim: u32) -> vec2<u32> {
    let raw_start = (coord * src_dim) / dst_dim;
    let raw_end = ((coord + 1u) * src_dim + dst_dim - 1u) / dst_dim;
    let start = min(raw_start, src_dim - 1u);
    let end = min(max(raw_end, start + 1u), src_dim);
    return vec2<u32>(start, end);
}

@compute @workgroup_size(8, 8, 1)
fn cs_hiz_from_sw(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = textureDimensions(hiz_dst);
    if gid.x >= dst_size.x || gid.y >= dst_size.y {
        return;
    }
    let sw_w = hiz_build_info.src_size.x;
    let sw_h = hiz_build_info.src_size.y;
    if sw_w == 0u || sw_h == 0u {
        textureStore(hiz_dst, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }
    let sw_len = arrayLength(&sw_depth_ro);
    let rx = footprint(gid.x, dst_size.x, sw_w);
    let ry = footprint(gid.y, dst_size.y, sw_h);
    var min_u: u32 = 0x3F800000u;
    for (var py = ry.x; py < ry.y; py = py + 1u) {
        let row = py * sw_w;
        for (var px = rx.x; px < rx.y; px = px + 1u) {
            let idx = row + px;
            let zu = select(0u, sw_depth_ro[idx], idx < sw_len);
            min_u = min(min_u, zu);
        }
    }
    textureStore(hiz_dst, vec2<i32>(gid.xy), vec4<f32>(bitcast<f32>(min_u), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_hiz_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = textureDimensions(hiz_dst);
    if gid.x >= dst_size.x || gid.y >= dst_size.y {
        return;
    }
    let src_size = textureDimensions(hiz_src, 0);
    let max_c = max(vec2<i32>(src_size) - vec2<i32>(1, 1), vec2<i32>(0, 0));
    let base = vec2<i32>(gid.xy) * 2;
    let d00 = textureLoad(hiz_src, min(base, max_c), 0).x;
    let d10 = textureLoad(hiz_src, min(base + vec2<i32>(1, 0), max_c), 0).x;
    let d01 = textureLoad(hiz_src, min(base + vec2<i32>(0, 1), max_c), 0).x;
    let d11 = textureLoad(hiz_src, min(base + vec2<i32>(1, 1), max_c), 0).x;
    let min_z = min(min(d00, d10), min(d01, d11));
    textureStore(hiz_dst, vec2<i32>(gid.xy), vec4<f32>(min_z, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_hiz_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = textureDimensions(hiz_dst);
    if gid.x >= dst_size.x || gid.y >= dst_size.y {
        return;
    }
    textureStore(hiz_dst, vec2<i32>(gid.xy), vec4<f32>(0.0));
}
