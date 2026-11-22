import package::generated::nanite::HizBuildInfo;

@group(1) @binding(0) var<uniform> hiz_build_info: HizBuildInfo;
@group(1) @binding(1) var hw_depth_r32: texture_2d<f32>;
@group(1) @binding(2) var hw_depth_tex: texture_depth_2d;
@group(1) @binding(3) var<storage, read> sw_depth_ro: array<u32>;
@group(1) @binding(4) var hiz_src: texture_2d<f32>;
@group(1) @binding(5) var hiz_dst: texture_storage_2d<r32float, write>;

fn footprint(coord: u32, dst_dim: u32, src_dim: u32) -> vec2<u32> {
    let raw_start = (coord * src_dim) / dst_dim;
    let raw_end = ((coord + 1u) * src_dim + dst_dim - 1u) / dst_dim;
    let start = min(raw_start, src_dim - 1u);
    let end = min(max(raw_end, start + 1u), src_dim);
    return vec2<u32>(start, end);
}

@compute @workgroup_size(8, 8, 1)
fn cs_hiz_from_r32_and_sw(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = textureDimensions(hiz_dst);
    if gid.x >= dst_size.x || gid.y >= dst_size.y {
        return;
    }
    let src_size = textureDimensions(hw_depth_r32);
    if src_size.x == 0u || src_size.y == 0u {
        textureStore(hiz_dst, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }
    let sw_w = select(src_size.x, hiz_build_info.src_size.x, hiz_build_info.src_size.x > 0u);
    let sw_h = select(src_size.y, hiz_build_info.src_size.y, hiz_build_info.src_size.y > 0u);
    let sw_len = arrayLength(&sw_depth_ro);
    let rx = footprint(gid.x, dst_size.x, src_size.x);
    let ry = footprint(gid.y, dst_size.y, src_size.y);
    var min_z: f32 = 1.0;
    for (var py = ry.x; py < ry.y; py = py + 1u) {
        let row = py * sw_w;
        for (var px = rx.x; px < rx.y; px = px + 1u) {
            var pz = textureLoad(hw_depth_r32, vec2<i32>(i32(px), i32(py)), 0).x;
            if px < sw_w && py < sw_h {
                let idx = row + px;
                if idx < sw_len {
                    pz = max(pz, bitcast<f32>(sw_depth_ro[idx]));
                }
            }
            min_z = min(min_z, pz);
        }
    }
    textureStore(hiz_dst, vec2<i32>(gid.xy), vec4<f32>(min_z, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_hiz_from_depth_and_sw(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = textureDimensions(hiz_dst);
    if gid.x >= dst_size.x || gid.y >= dst_size.y {
        return;
    }
    let src_size = textureDimensions(hw_depth_tex);
    if src_size.x == 0u || src_size.y == 0u {
        textureStore(hiz_dst, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }
    let sw_w = select(src_size.x, hiz_build_info.src_size.x, hiz_build_info.src_size.x > 0u);
    let sw_h = select(src_size.y, hiz_build_info.src_size.y, hiz_build_info.src_size.y > 0u);
    let sw_len = arrayLength(&sw_depth_ro);
    let rx = footprint(gid.x, dst_size.x, src_size.x);
    let ry = footprint(gid.y, dst_size.y, src_size.y);
    var min_z: f32 = 1.0;
    for (var py = ry.x; py < ry.y; py = py + 1u) {
        let row = py * sw_w;
        for (var px = rx.x; px < rx.y; px = px + 1u) {
            var pz = textureLoad(hw_depth_tex, vec2<i32>(i32(px), i32(py)), 0);
            if px < sw_w && py < sw_h {
                let idx = row + px;
                if idx < sw_len {
                    pz = max(pz, bitcast<f32>(sw_depth_ro[idx]));
                }
            }
            min_z = min(min_z, pz);
        }
    }
    textureStore(hiz_dst, vec2<i32>(gid.xy), vec4<f32>(min_z, 0.0, 0.0, 0.0));
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
