@group(1) @binding(0) var<uniform> params: QueueParams;
@group(1) @binding(1) var<storage, read_write> tags: array<u32>;
@group(1) @binding(2) var<storage, read_write> prefixes: array<vec4<u32>>;
@group(1) @binding(3) var<storage, read_write> rays: array<vec2<u32>>;
@group(1) @binding(4) var<storage, read_write> raw: array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> metadata: array<vec4<u32>, 3>;
@group(1) @binding(6) var<storage, read_write> arguments: array<u32, 12>;

var<workgroup> scan_values: array<vec4<u32>, 64>;
var<workgroup> scan_total: vec4<u32>;

// All 64 lanes enter, including lanes outside the image or scan level.
fn exclusive_scan(value: vec4<u32>, lane: u32) -> vec4<u32> {
    scan_values[lane] = value;
    workgroupBarrier();
    for (var stride = 1u; stride < 64u; stride <<= 1u) {
        let index = (lane + 1u) * (stride << 1u) - 1u;
        if index < 64u {
            scan_values[index] += scan_values[index - stride];
        }
        workgroupBarrier();
    }
    if lane == 0u {
        scan_total = scan_values[63];
        scan_values[63] = vec4<u32>(0u);
    }
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        let index = (lane + 1u) * (stride << 1u) - 1u;
        if index < 64u {
            let left = scan_values[index - stride];
            scan_values[index - stride] = scan_values[index];
            scan_values[index] += left;
        }
        workgroupBarrier();
    }
    return scan_values[lane];
}

@compute @workgroup_size(8, 8)
fn classify(@builtin(workgroup_id) tile: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let texel = tile.xy * 8u + local.xy;
    let valid = all(texel < params.size.xy);
    var eligible = false;
    var bin = 0u;
    var value = vec4<u32>(0u);
    if valid {
        let pixel = traced_pixel(vec2<i32>(texel));
        eligible = !(textureLoad(glow, pixel, 0).g < 0.01)
            && textureLoad(depth_texture, pixel, 0) > 0.0;
        let roughness = textureLoad(surface_texture, pixel, 0).z;
        bin = u32(roughness >= 0.125) + u32(roughness >= 0.25) + u32(roughness >= 0.375);
        if eligible { value[bin] = 1u; }
        raw[texel.y * params.size.x + texel.x] = vec4<f32>(0.0);
    }
    let rank = exclusive_scan(value, lane);
    if valid {
        tags[texel.y * params.size.x + texel.x] = select(0xffffffffu, 4u * rank[bin] + bin, eligible);
    }
    if lane == 0u {
        prefixes[tile.y * params.size.z + tile.x] = scan_total;
    }
}

@compute @workgroup_size(64)
fn scan_up(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let block = group.y * params.scan.w + group.x;
    let index = block * 64u + lane;
    var value = vec4<u32>(0u);
    if index < params.scan.y { value = prefixes[params.scan.x + index]; }
    let prefix = exclusive_scan(value, lane);
    if index < params.scan.y { prefixes[params.scan.x + index] = prefix; }
    if lane == 0u && block * 64u < params.scan.y {
        if params.scan.z == 0xffffffffu {
            metadata[0] = scan_total;
        } else {
            prefixes[params.scan.z + block] = scan_total;
        }
    }
}

@compute @workgroup_size(64)
fn scan_down(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = (group.y * params.scan.w + group.x) * 64u + lane;
    if index < params.scan.y {
        prefixes[params.scan.x + index] += prefixes[params.scan.z + (index >> 6u)];
    }
}

@compute @workgroup_size(1)
fn finalize() {
    let counts = metadata[0];
    metadata[1] = vec4<u32>(0u, counts.x, counts.x + counts.y, counts.x + counts.y + counts.z);
    let row = 1u << params.extra.z;
    for (var bin = 0u; bin < 4u; bin++) {
        let count = counts[bin];
        let groups = (count >> 6u) + u32((count & 63u) != 0u);
        let width = min(groups, row);
        let height = max(1u, (groups >> params.extra.z) + u32((groups & (row - 1u)) != 0u));
        arguments[bin * 3u] = width;
        arguments[bin * 3u + 1u] = height;
        arguments[bin * 3u + 2u] = 1u;
        metadata[2][bin] = width;
    }
}

@compute @workgroup_size(8, 8)
fn scatter(@builtin(workgroup_id) tile: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let texel = tile.xy * 8u + local.xy;
    if any(texel >= params.size.xy) { return; }
    let tag = tags[texel.y * params.size.x + texel.x];
    if tag == 0xffffffffu { return; }
    let bin = tag & 3u;
    let index = metadata[1][bin] + prefixes[tile.y * params.size.z + tile.x][bin] + (tag >> 2u);
    rays[index] = texel;
}
