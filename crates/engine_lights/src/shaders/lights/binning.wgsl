import package::face::instances::{AffinePose, STILL_FRAMES, instance_pose, instance_still_frames};
import package::generated::lights::{GROUP_LIGHTS, LightInput, NO_SLOT, SHAPED_LIGHTS};
import package::lights::clusters::clusters;

@group(1) @binding(0) var<storage, read> inputs: array<LightInput>;
@group(1) @binding(1) var<storage, read_write> arena: array<vec4<u32>>;
@group(1) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> totals: array<atomic<u32>>;

fn transformed(pose: AffinePose, point: vec4<f32>) -> vec3<f32> {
    return vec3<f32>(dot(pose.row_0, point), dot(pose.row_1, point), dot(pose.row_2, point));
}

fn slice_of(view: u32, depth: f32) -> u32 {
    let slice = floor(log2(max(depth, 1.0e-6)) * clusters[view].grid.z + clusters[view].grid.w);
    return u32(clamp(slice, 0.0, f32(clusters[view].counts.z - 1u)));
}

fn ndc_range(center: f32, radius: f32, near: f32, far: f32) -> vec2<f32> {
    let high = center + radius;
    let low = center - radius;
    let high_ndc = high / select(far, near, high > 0.0);
    let low_ndc = low / select(near, far, low > 0.0);
    return vec2<f32>(low_ndc, high_ndc);
}

const WORKGROUP: u32 = 64u;
const TOTALS_WORDS: u32 = 4u;

struct LightBox {
    first: vec2<u32>,
    columns: u32,
    rows: u32,
    first_slice: u32,
    covered: u32,
    index: u32,
    pad: u32,
    view_center: vec3<f32>,
    radius2: f32,
}

var<workgroup> boxes: array<LightBox, GROUP_LIGHTS>;
var<workgroup> box_ends: array<u32, GROUP_LIGHTS>;
var<workgroup> reached_lights: array<atomic<u32>, GROUP_LIGHTS>;

fn place(view: u32, index: u32, fill: bool) -> LightBox {
    var none: LightBox;
    none.index = index;
    let input = inputs[index];
    var position = input.position;
    var direction = input.direction;
    var size = 1.0;
    if input.slot != NO_SLOT {
        let pose = instance_pose(input.slot);
        position = transformed(pose, vec4<f32>(position, 1.0));
        let turned = transformed(pose, vec4<f32>(direction, 0.0));
        let length_squared = dot(turned, turned);
        direction = select(vec3<f32>(0.0), turned * inverseSqrt(length_squared),
            length_squared > 0.0);
        size = length(vec3<f32>(pose.row_0.y, pose.row_1.y, pose.row_2.y));
    }
    let tube = input.length * size;
    if !fill && view == 0u {
        let row = clusters[view].offsets.x + index * 4u;
        arena[row] = vec4<u32>(bitcast<vec3<u32>>(position), bitcast<u32>(input.range));
        arena[row + 1u] = vec4<u32>(bitcast<vec3<u32>>(input.color), bitcast<u32>(input.cos_outer));
        arena[row + 2u] = vec4<u32>(bitcast<vec3<u32>>(direction), bitcast<u32>(input.cos_inner));
        arena[row + 3u] = vec4<u32>(bitcast<u32>(input.radius * size), bitcast<u32>(tube),
            input.region, 0u);
    }
    if clusters[view].lights.w != 0u && input.slot != NO_SLOT
            && instance_still_frames(input.slot) < STILL_FRAMES {
        return none;
    }
    if !all(abs(position) < vec3<f32>(1.0e30)) {
        return none;
    }
    var center = position;
    var radius = input.range;
    if (clusters[view].lights.y & SHAPED_LIGHTS) != 0u {
        center += direction * (0.5 * tube);
        radius += 0.5 * tube;
    }
    if input.cos_outer > -1.5 && dot(direction, direction) > 0.0 {
        let cos_angle = max(input.cos_outer, 0.0);
        if cos_angle < 0.70710678 {
            center = position + direction * (input.range * cos_angle);
            radius = input.range * sqrt(1.0 - cos_angle * cos_angle);
        } else {
            radius = input.range / (2.0 * cos_angle);
            center = position + direction * radius;
        }
    }
    let clip = clusters[view].view_projection * vec4<f32>(center, 1.0);
    let depth = dot(clusters[view].depth.xyz, center) + clusters[view].depth.w;
    if depth + radius <= 0.0 || depth - radius > clusters[view].projection.w {
        return none;
    }
    let tiles = clusters[view].counts.xy;
    var first = vec2<u32>(0u);
    var last = tiles - vec2<u32>(1u);
    let near = depth - radius;
    if near > 1.0e-3 {
        let far = depth + radius;
        let x = ndc_range(clip.x, radius * clusters[view].projection.x, near, far);
        let y = ndc_range(clip.y, radius * clusters[view].projection.y, near, far);
        if x.x > 1.0 || x.y < -1.0 || y.x > 1.0 || y.y < -1.0 {
            return none;
        }
        let size = vec2<f32>(tiles);
        let low = clamp(floor(vec2<f32>(x.x * 0.5 + 0.5, 0.5 - y.y * 0.5) * size),
            vec2<f32>(0.0), size - 1.0);
        let high = clamp(floor(vec2<f32>(x.y * 0.5 + 0.5, 0.5 - y.x * 0.5) * size),
            vec2<f32>(0.0), size - 1.0);
        first = vec2<u32>(low);
        last = vec2<u32>(high);
    }
    let first_slice = slice_of(view, max(near, 1.0e-6));
    let last_slice = slice_of(view, depth + radius);
    var placed: LightBox;
    placed.index = index;
    placed.first = first;
    placed.columns = last.x - first.x + 1u;
    placed.rows = last.y - first.y + 1u;
    placed.first_slice = first_slice;
    placed.covered = placed.columns * placed.rows * (last_slice - first_slice + 1u);
    placed.view_center = vec3<f32>(clip.x / clusters[view].projection.x,
        clip.y / clusters[view].projection.y, depth);
    placed.radius2 = radius * radius;
    return placed;
}

fn visit_cluster(view: u32, light: LightBox, box: u32, fill: bool) -> bool {
    let x = light.first.x + box % light.columns;
    let y = light.first.y + (box / light.columns) % light.rows;
    let z = light.first_slice + box / (light.columns * light.rows);
    let tiles = clusters[view].counts.xy;
    let tile_size = 2.0 / vec2<f32>(tiles);
    let depth = light.view_center.z;
    let slice_near = select(exp2((f32(z) - clusters[view].grid.w) / clusters[view].grid.z), 0.0,
        z == 0u);
    let slice_far = exp2((f32(z + 1u) - clusters[view].grid.w) / clusters[view].grid.z);
    let dz = max(max(slice_near - depth, depth - slice_far), 0.0);
    if dz * dz > light.radius2 {
        return false;
    }
    let top = 1.0 - f32(y) * tile_size.y;
    let bottom = top - tile_size.y;
    let left = -1.0 + f32(x) * tile_size.x;
    let right = left + tile_size.x;
    let x_low = min(left * slice_near, left * slice_far) / clusters[view].projection.x;
    let x_high = max(right * slice_near, right * slice_far) / clusters[view].projection.x;
    let y_low = min(bottom * slice_near, bottom * slice_far) / clusters[view].projection.y;
    let y_high = max(top * slice_near, top * slice_far) / clusters[view].projection.y;
    let dx = max(max(x_low - light.view_center.x, light.view_center.x - x_high), 0.0);
    let dy = max(max(y_low - light.view_center.y, light.view_center.y - y_high), 0.0);
    if dx * dx + dy * dy + dz * dz > light.radius2 {
        return false;
    }
    let cluster = (z * tiles.y + y) * tiles.x + x;
    let counted = view * tiles.x * tiles.y * clusters[view].counts.z + cluster;
    if fill {
        let slot = atomicAdd(&counts[counted], 1u);
        let pair = arena[clusters[view].offsets.y + cluster / 2u];
        let range = select(pair.xy, pair.zw, (cluster & 1u) != 0u);
        if slot < range.y {
            let listed = range.x + slot;
            arena[clusters[view].offsets.z + listed / 4u][listed % 4u] = light.index;
        }
    } else {
        atomicAdd(&counts[counted], 1u);
    }
    return true;
}

fn visit(view: u32, group: u32, fill: bool, lane: u32) {
    if lane < GROUP_LIGHTS {
        let index = group * GROUP_LIGHTS + lane;
        var light: LightBox;
        light.index = index;
        if index < clusters[view].lights.x {
            light = place(view, index, fill);
        }
        boxes[lane] = light;
        box_ends[lane] = light.covered;
        atomicStore(&reached_lights[lane], 0u);
    }
    workgroupBarrier();
    for (var stride = 1u; stride < GROUP_LIGHTS; stride *= 2u) {
        var earlier = 0u;
        if lane < GROUP_LIGHTS && lane >= stride {
            earlier = box_ends[lane - stride];
        }
        workgroupBarrier();
        if lane < GROUP_LIGHTS {
            box_ends[lane] += earlier;
        }
        workgroupBarrier();
    }
    let total = box_ends[GROUP_LIGHTS - 1u];
    for (var item = lane; item < total; item += WORKGROUP) {
        var low = 0u;
        var high = GROUP_LIGHTS - 1u;
        while low < high {
            let middle = (low + high) / 2u;
            if box_ends[middle] > item {
                high = middle;
            } else {
                low = middle + 1u;
            }
        }
        let start = select(0u, box_ends[max(low, 1u) - 1u], low > 0u);
        if visit_cluster(view, boxes[low], item - start, fill) {
            atomicStore(&reached_lights[low], 1u);
        }
    }
    workgroupBarrier();
    if !fill && lane < GROUP_LIGHTS && atomicLoad(&reached_lights[lane]) != 0u {
        atomicAdd(&totals[view * TOTALS_WORDS], 1u);
    }
}

@compute @workgroup_size(64)
fn count_lights(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    visit(group.z, group.y * 65535u + group.x, false, lane);
}

@compute @workgroup_size(64)
fn bin_lights(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    visit(group.z, group.y * 65535u + group.x, true, lane);
}

const SCAN_THREADS: u32 = 256u;
var<workgroup> partial: array<u32, 256>;

@compute @workgroup_size(256)
fn scan_clusters(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) thread: u32) {
    let view = group.z;
    let total = clusters[view].counts.x * clusters[view].counts.y * clusters[view].counts.z;
    let budget = clusters[view].counts.w;
    let capacity = clusters[view].lights.z;
    let counted = view * total;
    let run = (total + SCAN_THREADS - 1u) / SCAN_THREADS;
    let start = min(thread * run, total);
    let end = min(start + run, total);
    var sum = 0u;
    for (var cluster = start; cluster < end; cluster++) {
        sum += min(atomicLoad(&counts[counted + cluster]), budget);
    }
    partial[thread] = sum;
    workgroupBarrier();
    for (var stride = 1u; stride < SCAN_THREADS; stride *= 2u) {
        var earlier = 0u;
        if thread >= stride {
            earlier = partial[thread - stride];
        }
        workgroupBarrier();
        partial[thread] += earlier;
        workgroupBarrier();
    }
    var offset = partial[thread] - sum;
    var lost = 0u;
    for (var cluster = start; cluster < end; cluster++) {
        let wanted = atomicLoad(&counts[counted + cluster]);
        let listed = min(wanted, budget);
        let room = select(0u, capacity - offset, offset < capacity);
        let kept = min(listed, room);
        let at = clusters[view].offsets.y + cluster / 2u;
        let lane = (cluster & 1u) * 2u;
        arena[at][lane] = offset;
        arena[at][lane + 1u] = kept;
        lost += wanted - kept;
        offset += listed;
        atomicStore(&counts[counted + cluster], 0u);
    }
    if lost != 0u {
        atomicAdd(&totals[view * TOTALS_WORDS + 1u], lost);
    }
    if thread == SCAN_THREADS - 1u {
        atomicStore(&totals[view * TOTALS_WORDS + 2u], partial[thread]);
        atomicStore(&totals[view * TOTALS_WORDS + 3u], capacity);
    }
}
