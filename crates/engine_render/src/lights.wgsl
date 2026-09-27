// Local light binning (decision 023). One thread per light places it in
// world space, from the displayed pose of its slot when it follows one, and
// the threads of its workgroup share the view clusters their lights' spheres'
// screen and depth bounds cover: a tile of the screen by an exponential
// depth slice. A first pass counts each
// cluster's lights, one workgroup turns the counts into list ranges, and a
// second pass fills the lists, so every list is exactly as long as its
// lights (an optional budget shortens them). Fragments shade their
// cluster's list. Bounds are conservative, so a cluster may list a light that
// cannot reach it; fragments test the range again.

struct Clusters {
    view_projection: mat4x4<f32>,
    // Depth along the view of point p: dot(xyz, p) + w.
    depth: vec4<f32>,
    // x and y projection scales, clustered depth range start and end.
    projection: vec4<f32>,
    // Tile width and height in pixels; slice = log2(depth) * z + w.
    grid: vec4<f32>,
    // Tiles across and down, depth slices, and the per-cluster budget
    // (0xffffffff without one).
    counts: vec4<u32>,
    // Lights, 1 when slot poses are bound, and the entries `indices` holds.
    lights: vec4<u32>,
}

// A light as extracted: world position or slot model space (decision 023).
struct Input {
    position: vec3<f32>,
    range: f32,
    color: vec3<f32>,
    slot: u32,
    direction: vec3<f32>,
    cos_inner: f32,
    cos_outer: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Light {
    position: vec3<f32>,
    range: f32,
    color: vec3<f32>,
    cos_outer: f32,
    direction: vec3<f32>,
    cos_inner: f32,
}

// Transposed top three rows of an affine slot pose.
struct Rows {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

const NO_SLOT: u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform> clusters: Clusters;
@group(0) @binding(1) var<storage, read> inputs: array<Input>;
@group(0) @binding(2) var<storage, read> poses: array<Rows>;
@group(0) @binding(3) var<storage, read_write> lights: array<Light>;
// Per cluster: its lights while counting, then its write cursor.
@group(0) @binding(4) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> indices: array<u32>;
// Lights that reached a cluster, cluster entries lost to the budget or to
// the capacity of `indices`, and the entries the lists need.
@group(0) @binding(6) var<storage, read_write> totals: array<atomic<u32>, 3>;
// Per cluster: the first entry of its list in `indices` and its length.
@group(0) @binding(7) var<storage, read_write> ranges: array<vec2<u32>>;

fn transformed(rows: Rows, point: vec4<f32>) -> vec3<f32> {
    return vec3<f32>(dot(rows.r0, point), dot(rows.r1, point), dot(rows.r2, point));
}

fn slice_of(depth: f32) -> u32 {
    let slice = floor(log2(max(depth, 1.0e-6)) * clusters.grid.z + clusters.grid.w);
    return u32(clamp(slice, 0.0, f32(clusters.counts.z - 1u)));
}

// Tile column (or row) range of view-space extent [low, high] (scaled by the
// projection) between depths near and far: conservative NDC bounds.
fn ndc_range(center: f32, radius: f32, near: f32, far: f32) -> vec2<f32> {
    let high = center + radius;
    let low = center - radius;
    let high_ndc = high / select(far, near, high > 0.0);
    let low_ndc = low / select(near, far, low > 0.0);
    return vec2<f32>(low_ndc, high_ndc);
}

const WORKGROUP: u32 = 64u;
// Lights per workgroup: few enough that lights near the eye, with thousands
// of clusters, spread over many workgroups.
const LIGHTS: u32 = 8u;

// A placed light's clusters: its first tile, its columns and rows of tiles,
// its first slice, the clusters of that box (0: none), and its sphere in
// view space (x and y scaled like clip coordinates) with its squared radius.
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

// A workgroup's lights, the running total of their clusters, and whether
// each reached one.
var<workgroup> boxes: array<LightBox, LIGHTS>;
var<workgroup> box_ends: array<u32, LIGHTS>;
var<workgroup> reached_lights: array<atomic<u32>, LIGHTS>;

// Places light `index` in world space (writing its binned record while
// counting) and returns the clusters its sphere's screen and depth bounds
// cover.
fn place(index: u32, fill: bool) -> LightBox {
    var none: LightBox;
    none.index = index;
    let input = inputs[index];
    var position = input.position;
    var direction = input.direction;
    if input.slot != NO_SLOT && clusters.lights.y != 0u {
        let rows = poses[input.slot];
        position = transformed(rows, vec4<f32>(position, 1.0));
        let turned = transformed(rows, vec4<f32>(direction, 0.0));
        let length_squared = dot(turned, turned);
        direction = select(vec3<f32>(0.0), turned * inverseSqrt(length_squared),
            length_squared > 0.0);
    }
    if !fill {
        var light: Light;
        light.position = position;
        light.range = input.range;
        light.color = input.color;
        light.cos_outer = input.cos_outer;
        light.direction = direction;
        light.cos_inner = input.cos_inner;
        lights[index] = light;
    }
    if !all(abs(position) < vec3<f32>(1.0e30)) {
        return none;
    }
    // The sphere the light reaches: a spot cone's bounding sphere (Wronski)
    // is smaller and centered along the cone.
    var center = position;
    var radius = input.range;
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
    let clip = clusters.view_projection * vec4<f32>(center, 1.0);
    let depth = dot(clusters.depth.xyz, center) + clusters.depth.w;
    // Behind the eye or beyond the far plane.
    if depth + radius <= 0.0 || depth - radius > clusters.projection.w {
        return none;
    }
    let tiles = clusters.counts.xy;
    var first = vec2<u32>(0u);
    var last = tiles - vec2<u32>(1u);
    let near = depth - radius;
    // A sphere reaching the eye covers the whole screen.
    if near > 1.0e-3 {
        // Clip x and y are the projection scales times view x and y.
        let far = depth + radius;
        let x = ndc_range(clip.x, radius * clusters.projection.x, near, far);
        let y = ndc_range(clip.y, radius * clusters.projection.y, near, far);
        if x.x > 1.0 || x.y < -1.0 || y.x > 1.0 || y.y < -1.0 {
            return none;
        }
        let size = vec2<f32>(tiles);
        // Screen rows run down while NDC y runs up.
        let low = clamp(floor(vec2<f32>(x.x * 0.5 + 0.5, 0.5 - y.y * 0.5) * size),
            vec2<f32>(0.0), size - 1.0);
        let high = clamp(floor(vec2<f32>(x.y * 0.5 + 0.5, 0.5 - y.x * 0.5) * size),
            vec2<f32>(0.0), size - 1.0);
        first = vec2<u32>(low);
        last = vec2<u32>(high);
    }
    let first_slice = slice_of(max(near, 1.0e-6));
    let last_slice = slice_of(depth + radius);
    var placed: LightBox;
    placed.index = index;
    placed.first = first;
    placed.columns = last.x - first.x + 1u;
    placed.rows = last.y - first.y + 1u;
    placed.first_slice = first_slice;
    placed.covered = placed.columns * placed.rows * (last_slice - first_slice + 1u);
    placed.view_center = vec3<f32>(clip.x / clusters.projection.x,
        clip.y / clusters.projection.y, depth);
    placed.radius2 = radius * radius;
    return placed;
}

// Cluster `box` (in slice, row, column order) of `light`: whether its sphere
// reaches the cluster's view-space box, then counting it there or, with
// `fill`, writing the light into its list.
fn visit_cluster(light: LightBox, box: u32, fill: bool) -> bool {
    let x = light.first.x + box % light.columns;
    let y = light.first.y + (box / light.columns) % light.rows;
    let z = light.first_slice + box / (light.columns * light.rows);
    let tiles = clusters.counts.xy;
    let tile_size = 2.0 / vec2<f32>(tiles);
    let depth = light.view_center.z;
    // Depth range of the slice, the first spanning everything nearer.
    let slice_near = select(exp2((f32(z) - clusters.grid.w) / clusters.grid.z), 0.0, z == 0u);
    let slice_far = exp2((f32(z + 1u) - clusters.grid.w) / clusters.grid.z);
    let dz = max(max(slice_near - depth, depth - slice_far), 0.0);
    if dz * dz > light.radius2 {
        return false;
    }
    // NDC y of the row's edges: rows run down the screen.
    let top = 1.0 - f32(y) * tile_size.y;
    let bottom = top - tile_size.y;
    // The cluster's view-space box: its NDC edges at its nearest and
    // farthest depths.
    let left = -1.0 + f32(x) * tile_size.x;
    let right = left + tile_size.x;
    let x_low = min(left * slice_near, left * slice_far) / clusters.projection.x;
    let x_high = max(right * slice_near, right * slice_far) / clusters.projection.x;
    let y_low = min(bottom * slice_near, bottom * slice_far) / clusters.projection.y;
    let y_high = max(top * slice_near, top * slice_far) / clusters.projection.y;
    let dx = max(max(x_low - light.view_center.x, light.view_center.x - x_high), 0.0);
    let dy = max(max(y_low - light.view_center.y, light.view_center.y - y_high), 0.0);
    if dx * dx + dy * dy + dz * dz > light.radius2 {
        return false;
    }
    let cluster = (z * tiles.y + y) * tiles.x + x;
    if fill {
        // The range already excludes entries the budget or the capacity
        // cannot hold.
        let slot = atomicAdd(&counts[cluster], 1u);
        let range = ranges[cluster];
        if slot < range.y {
            indices[range.x + slot] = light.index;
        }
    } else {
        atomicAdd(&counts[cluster], 1u);
    }
    return true;
}

// Workgroup `group`'s lights: a thread places each, then all threads share
// every cluster the lights cover, so a light near the eye with thousands of
// clusters takes the whole workgroup.
fn visit(group: u32, fill: bool, lane: u32) {
    if lane < LIGHTS {
        let index = group * LIGHTS + lane;
        var light: LightBox;
        light.index = index;
        if index < clusters.lights.x {
            light = place(index, fill);
        }
        boxes[lane] = light;
        box_ends[lane] = light.covered;
        atomicStore(&reached_lights[lane], 0u);
    }
    workgroupBarrier();
    // Running totals of the lights' clusters.
    for (var stride = 1u; stride < LIGHTS; stride *= 2u) {
        var earlier = 0u;
        if lane < LIGHTS && lane >= stride {
            earlier = box_ends[lane - stride];
        }
        workgroupBarrier();
        if lane < LIGHTS {
            box_ends[lane] += earlier;
        }
        workgroupBarrier();
    }
    let total = box_ends[LIGHTS - 1u];
    for (var item = lane; item < total; item += WORKGROUP) {
        // The light whose clusters hold `item`: the first total beyond it.
        var low = 0u;
        var high = LIGHTS - 1u;
        while low < high {
            let middle = (low + high) / 2u;
            if box_ends[middle] > item {
                high = middle;
            } else {
                low = middle + 1u;
            }
        }
        let start = select(0u, box_ends[max(low, 1u) - 1u], low > 0u);
        if visit_cluster(boxes[low], item - start, fill) {
            atomicStore(&reached_lights[low], 1u);
        }
    }
    workgroupBarrier();
    if !fill && lane < LIGHTS && atomicLoad(&reached_lights[lane]) != 0u {
        atomicAdd(&totals[0], 1u);
    }
}

// LIGHTS lights per workgroup; dispatches beyond 65,535 workgroups add rows.
@compute @workgroup_size(64)
fn count_lights(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    visit(group.y * 65535u + group.x, false, lane);
}

@compute @workgroup_size(64)
fn bin_lights(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    visit(group.y * 65535u + group.x, true, lane);
}

const SCAN_THREADS: u32 = 256u;
var<workgroup> partial: array<u32, 256>;

// One workgroup: each thread sums a run of clusters' budgeted counts, the
// workgroup scans the sums, and each thread gives its clusters consecutive
// list ranges, cut at the capacity of `indices`, and zeroes their counts
// for use as write cursors.
@compute @workgroup_size(256)
fn scan_clusters(@builtin(local_invocation_index) thread: u32) {
    let total = clusters.counts.x * clusters.counts.y * clusters.counts.z;
    let budget = clusters.counts.w;
    let capacity = clusters.lights.z;
    let run = (total + SCAN_THREADS - 1u) / SCAN_THREADS;
    let start = min(thread * run, total);
    let end = min(start + run, total);
    var sum = 0u;
    for (var cluster = start; cluster < end; cluster++) {
        sum += min(atomicLoad(&counts[cluster]), budget);
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
        let wanted = atomicLoad(&counts[cluster]);
        let listed = min(wanted, budget);
        let room = select(0u, capacity - offset, offset < capacity);
        let kept = min(listed, room);
        ranges[cluster] = vec2<u32>(offset, kept);
        lost += wanted - kept;
        offset += listed;
        atomicStore(&counts[cluster], 0u);
    }
    if lost != 0u {
        atomicAdd(&totals[1], lost);
    }
    if thread == SCAN_THREADS - 1u {
        atomicStore(&totals[2], partial[thread]);
    }
}
