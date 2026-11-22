import package::generated::hiz::HizInfo;

@group(1) @binding(0) var<uniform> hiz_info: HizInfo;
@group(1) @binding(1) var hw_depth: texture_depth_2d;
@group(1) @binding(2) var<storage, read> sw_depth_u32: array<u32>;
@group(1) @binding(3) var hiz_src: texture_2d<f32>;
@group(1) @binding(4) var hiz_dst: texture_storage_2d<r32float, write>;

fn footprint_range(coord: u32, dst_dim: u32, src_dim: u32) -> vec2<u32> {
    let raw_start = (coord * src_dim) / dst_dim;
    let raw_end = ((coord + 1u) * src_dim + dst_dim - 1u) / dst_dim;
    let start = min(raw_start, src_dim - 1u);
    let end = min(max(raw_end, start + 1u), src_dim);
    return vec2<u32>(start, end);
}

@compute @workgroup_size(8, 8, 1)
fn cs_hiz_from_depth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = textureDimensions(hiz_dst);
    if gid.x >= dst_size.x || gid.y >= dst_size.y {
        return;
    }
    let src_size = textureDimensions(hw_depth);
    if src_size.x == 0u || src_size.y == 0u {
        textureStore(hiz_dst, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }
    let rx = footprint_range(gid.x, dst_size.x, src_size.x);
    let ry = footprint_range(gid.y, dst_size.y, src_size.y);
    var min_z: f32 = 1.0;
    for (var py: u32 = ry.x; py < ry.y; py = py + 1u) {
        for (var px: u32 = rx.x; px < rx.y; px = px + 1u) {
            let z = textureLoad(hw_depth, vec2<i32>(i32(px), i32(py)), 0);
            min_z = min(min_z, z);
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
    let src_size = textureDimensions(hw_depth);
    if src_size.x == 0u || src_size.y == 0u {
        textureStore(hiz_dst, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }
    let sw_w = select(src_size.x, hiz_info.src_size.x, hiz_info.src_size.x > 0u);
    let sw_h = select(src_size.y, hiz_info.src_size.y, hiz_info.src_size.y > 0u);
    let sw_len = arrayLength(&sw_depth_u32);
    let rx = footprint_range(gid.x, dst_size.x, src_size.x);
    let ry = footprint_range(gid.y, dst_size.y, src_size.y);
    var min_z: f32 = 1.0;
    for (var py: u32 = ry.x; py < ry.y; py = py + 1u) {
        let row_offset = py * sw_w;
        for (var px: u32 = rx.x; px < rx.y; px = px + 1u) {
            let hw_z = textureLoad(hw_depth, vec2<i32>(i32(px), i32(py)), 0);
            var pixel_z = hw_z;
            if px < sw_w && py < sw_h {
                let sw_idx = row_offset + px;
                if sw_idx < sw_len {
                    let sw_z = bitcast<f32>(sw_depth_u32[sw_idx]);
                    pixel_z = max(hw_z, sw_z);
                }
            }
            min_z = min(min_z, pixel_z);
        }
    }
    textureStore(hiz_dst, vec2<i32>(gid.xy), vec4<f32>(min_z, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_hiz_downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = textureDimensions(hiz_dst);
    if gid.x >= dst_size.x || gid.y >= dst_size.y {
        return;
    }
    let src_size = textureDimensions(hiz_src, 0);
    let max_coord = max(vec2<i32>(src_size) - vec2<i32>(1, 1), vec2<i32>(0, 0));
    let base = vec2<i32>(gid.xy) * 2;
    let c00 = min(base, max_coord);
    let c10 = min(base + vec2<i32>(1, 0), max_coord);
    let c01 = min(base + vec2<i32>(0, 1), max_coord);
    let c11 = min(base + vec2<i32>(1, 1), max_coord);
    let d00 = textureLoad(hiz_src, c00, 0).x;
    let d10 = textureLoad(hiz_src, c10, 0).x;
    let d01 = textureLoad(hiz_src, c01, 0).x;
    let d11 = textureLoad(hiz_src, c11, 0).x;
    let min_z = min(min(d00, d10), min(d01, d11));
    textureStore(hiz_dst, vec2<i32>(gid.xy), vec4<f32>(min_z, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_hiz_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = textureDimensions(hiz_dst);
    if gid.x >= dst_size.x || gid.y >= dst_size.y {
        return;
    }
    textureStore(hiz_dst, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 0.0));
}
