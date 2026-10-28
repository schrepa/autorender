const VIEW_PLANES: u32 = 12u;
const MAX_VIEWS: u32 = 8u;

struct View {
    planes: array<vec4<f32>, VIEW_PLANES>,
    eye: vec4<f32>,
    lod: vec4<f32>,
    cone: vec4<f32>,
}

struct Frame {
    counts: vec4<u32>,
    view_projection: mat4x4<f32>,
    previous_view_projection: mat4x4<f32>,
    viewport: vec4<f32>,
    tests: vec4<u32>,
    feedback: vec4<u32>,
    views: array<View, MAX_VIEWS>,
}

struct Batch {
    start: u32,
    count: u32,
    draw: u32,
    mesh: u32,
    bin: u32,
    flags: u32,
    view: u32,
    pad0: u32,
}

struct Mesh {
    bases: vec4<u32>,
    info: vec4<u32>,
    position_min: vec4<f32>,
    position_step: vec4<f32>,
    uv: vec4<f32>,
}

struct Bin {
    base: u32,
    capacity: u32,
    pad0: u32,
    pad1: u32,
}

struct Rows {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> batches: array<Batch>;
@group(0) @binding(2) var<storage, read> meshes: array<Mesh>;
@group(0) @binding(3) var<storage, read> nodes: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> groups: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> clusters: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> poses: array<Rows>;
@group(0) @binding(7) var<storage, read> ids: array<u32>;
@group(0) @binding(8) var<storage, read> draws: array<u32>;
@group(0) @binding(9) var<storage, read_write> queue_a: array<vec4<u32>>;
@group(0) @binding(10) var<storage, read_write> queue_b: array<vec4<u32>>;
@group(0) @binding(11) var<storage, read_write> work: array<vec4<u32>>;
@group(0) @binding(12) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(13) var<storage, read_write> visible: array<vec2<u32>>;
@group(0) @binding(14) var<storage, read> bins: array<Bin>;
@group(0) @binding(15) var<storage, read_write> dispatch: array<u32>;
@group(0) @binding(16) var<storage, read_write> draw_args: array<u32>;
@group(0) @binding(19) var depth_pyramid: texture_2d<f32>;
@group(0) @binding(20) var<storage, read_write> retest: array<vec4<u32>>;

const NONE: u32 = 0xffffffffu;
const MOVED_SLOT: u32 = 0x80000000u;
const QUEUE_A: u32 = 0u;
const QUEUE_B: u32 = 1u;
const WORK: u32 = 2u;
const DROPPED_QUEUE: u32 = 3u;
const DROPPED_WORK: u32 = 4u;
const DROPPED_CLUSTERS: u32 = 5u;
const SELECTED_CLUSTERS: u32 = 6u;
const SELECTED_TRIANGLES: u32 = 7u;
const SHADOW_CLUSTERS: u32 = 8u;
const SHADOW_TRIANGLES: u32 = 9u;
const RETESTED: u32 = 10u;
const HIDDEN_CLUSTERS: u32 = 11u;
const BINS: u32 = 12u;
const TEST_PREVIOUS: u32 = 1u;
const TEST_SUBPIXEL: u32 = 2u;
const F32_MAX: f32 = 0x1.fffffep+127f;
const DOUBLE_SIDED: u32 = 1u;
const VIEW_SHIFT: u32 = 8u;
const CONE_ORTHOGRAPHIC: f32 = 1.0;
const GROUP: u32 = 64u;
const CLUSTER_VERTICES: u32 = 384u;

fn transformed(pose: Rows, point: vec3<f32>) -> vec3<f32> {
    let p = vec4<f32>(point, 1.0);
    return vec3<f32>(dot(pose.r0, p), dot(pose.r1, p), dot(pose.r2, p));
}

fn largest_scale(pose: Rows) -> f32 {
    let x = vec3<f32>(pose.r0.x, pose.r1.x, pose.r2.x);
    let y = vec3<f32>(pose.r0.y, pose.r1.y, pose.r2.y);
    let z = vec3<f32>(pose.r0.z, pose.r1.z, pose.r2.z);
    return sqrt(max(dot(x, x), max(dot(y, y), dot(z, z))));
}

fn outside(view: u32, center: vec3<f32>, radius: f32) -> bool {
    for (var plane = 0u; plane < VIEW_PLANES; plane++) {
        let normal = frame.views[view].planes[plane];
        if dot(normal.xyz, center) + normal.w < -radius {
            return true;
        }
    }
    return false;
}

fn projected(view: u32, center: vec3<f32>, radius: f32, error: f32) -> f32 {
    let eye = frame.views[view].eye;
    let distance = max(length(center - eye.xyz) - radius, eye.w);
    return error / distance * frame.views[view].lod.x;
}

fn precise_enough(view: u32, center: vec3<f32>, radius: f32, error: f32) -> bool {
    if projected(view, center, radius, error) <= frame.views[view].lod.y {
        return true;
    }
    return error <= frame.views[view].lod.w && outside(0u, center, radius);
}

fn resident(group: u32) -> bool {
    let word = frame.feedback.z + (group >> 5u);
    return ((groups[word >> 2u][word & 3u] >> (group & 31u)) & 1u) != 0u;
}

fn mark(base: u32, group: u32) {
    atomicOr(&counters[base + (group >> 5u)], 1u << (group & 31u));
}

fn push_node(queue: u32, item: vec4<u32>) {
    let index = atomicAdd(&counters[queue], 1u);
    if index >= frame.counts.x {
        atomicAdd(&counters[DROPPED_QUEUE], 1u);
        return;
    }
    if queue == QUEUE_A {
        queue_a[index] = item;
    } else {
        queue_b[index] = item;
    }
}

@compute @workgroup_size(64)
fn seed(@builtin(global_invocation_id) id: vec3<u32>) {
    let batch = batches[id.y];
    var count = batch.count;
    if count == NONE {
        count = draws[batch.draw * 5u + 1u];
    }
    if id.x >= count {
        return;
    }
    let visible_id = ids[batch.start + id.x];
    let mesh = meshes[batch.mesh];
    let flags = batch.flags | (batch.view << VIEW_SHIFT);
    for (var level = 0u; level < mesh.info.x; level++) {
        push_node(QUEUE_A, vec4<u32>(visible_id, mesh.bases.x + level, batch.bin, flags));
    }
}

fn level_args(source: u32, sink: u32) {
    let count = min(atomicLoad(&counters[source]), frame.counts.x);
    dispatch[0] = (count + GROUP - 1u) / GROUP;
    dispatch[1] = 1u;
    dispatch[2] = 1u;
    atomicStore(&counters[sink], 0u);
}

@compute @workgroup_size(1)
fn level_args_even() {
    level_args(QUEUE_A, QUEUE_B);
}

@compute @workgroup_size(1)
fn level_args_odd() {
    level_args(QUEUE_B, QUEUE_A);
}

fn traverse(index: u32, source: u32, sink: u32) {
    if index >= min(atomicLoad(&counters[source]), frame.counts.x) {
        return;
    }
    var item: vec4<u32>;
    if source == QUEUE_A {
        item = queue_a[index];
    } else {
        item = queue_b[index];
    }
    let view = item.w >> VIEW_SHIFT;
    let pose = poses[item.x & ~MOVED_SLOT];
    let scale = largest_scale(pose);
    let sphere = nodes[item.y * 2u];
    let node = nodes[item.y * 2u + 1u];
    let center = transformed(pose, bitcast<vec3<f32>>(sphere.xyz));
    let radius = bitcast<f32>(sphere.w) * scale;
    if outside(view, center, radius)
            || precise_enough(view, center, radius, bitcast<f32>(node.x) * scale) {
        return;
    }
    if node.y != NONE {
        let slot = atomicAdd(&counters[WORK], 1u);
        if slot < frame.counts.y {
            work[slot] = vec4<u32>(item.x, node.y, item.z, item.w);
        } else {
            atomicAdd(&counters[DROPPED_WORK], 1u);
        }
        return;
    }
    for (var child = node.z; child < node.z + node.w; child++) {
        push_node(sink, vec4<u32>(item.x, child, item.z, item.w));
    }
}

@compute @workgroup_size(64)
fn traverse_even(@builtin(global_invocation_id) id: vec3<u32>) {
    traverse(id.x, QUEUE_A, QUEUE_B);
}

@compute @workgroup_size(64)
fn traverse_odd(@builtin(global_invocation_id) id: vec3<u32>) {
    traverse(id.x, QUEUE_B, QUEUE_A);
}

@compute @workgroup_size(1)
fn work_args() {
    let count = min(atomicLoad(&counters[WORK]), frame.counts.y);
    dispatch[3] = (count + GROUP - 1u) / GROUP;
    dispatch[4] = 1u;
    dispatch[5] = 1u;
}

fn backfacing(view: u32, pose: Rows, cone: u32, center: vec3<f32>, radius: f32) -> bool {
    let mode = frame.views[view].lod.z;
    let bytes = vec4<i32>(bitcast<i32>(cone << 24u), bitcast<i32>(cone << 16u),
        bitcast<i32>(cone << 8u), bitcast<i32>(cone)) >> vec4<u32>(24u);
    let cutoff = f32(bytes.w) / 127.0;
    if cutoff >= 1.0 {
        return false;
    }
    let x = vec3<f32>(pose.r0.x, pose.r1.x, pose.r2.x);
    let y = vec3<f32>(pose.r0.y, pose.r1.y, pose.r2.y);
    let z = vec3<f32>(pose.r0.z, pose.r1.z, pose.r2.z);
    let lengths = vec3<f32>(dot(x, x), dot(y, y), dot(z, z));
    if max(lengths.x, max(lengths.y, lengths.z)) > min(lengths.x, min(lengths.y, lengths.z)) * 1.0001 {
        return false;
    }
    let local = vec3<f32>(bytes.xyz) / 127.0;
    let axis = normalize(x * local.x + y * local.y + z * local.z) * sign(dot(x, cross(y, z)));
    if mode == CONE_ORTHOGRAPHIC {
        return dot(frame.views[view].cone.xyz, axis) >= cutoff;
    }
    let to_center = center - frame.views[view].cone.xyz;
    return dot(to_center, axis) >= cutoff * length(to_center) + radius;
}

@compute @workgroup_size(64)
fn select_clusters(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= min(atomicLoad(&counters[WORK]), frame.counts.y) {
        return;
    }
    let item = work[id.x];
    let view = item.w >> VIEW_SHIFT;
    let pose = poses[item.x & ~MOVED_SLOT];
    let scale = largest_scale(pose);
    if !resident(item.y) {
        mark(frame.feedback.y, item.y);
        return;
    }
    mark(frame.feedback.x, item.y);
    let group = groups[item.y * 2u + 1u];
    let bin = bins[item.z];
    let counts = select(SHADOW_CLUSTERS, SELECTED_CLUSTERS, view == 0u);
    let triangles = select(SHADOW_TRIANGLES, SELECTED_TRIANGLES, view == 0u);
    for (var cluster = group.y; cluster < group.y + group.z; cluster++) {
        let sphere = clusters[cluster * 3u];
        let record = clusters[cluster * 3u + 1u];
        if record.y != NONE {
            let source_sphere = groups[record.y * 2u];
            let source = groups[record.y * 2u + 1u];
            let source_center = transformed(pose, bitcast<vec3<f32>>(source_sphere.xyz));
            if !precise_enough(view, source_center, bitcast<f32>(source_sphere.w) * scale,
                    bitcast<f32>(source.x) * scale) {
                if resident(record.y) {
                    continue;
                }
                mark(frame.feedback.y, record.y);
            }
        }
        let center = transformed(pose, bitcast<vec3<f32>>(sphere.xyz));
        let radius = bitcast<f32>(sphere.w) * scale;
        if outside(view, center, radius) {
            continue;
        }
        if (item.w & DOUBLE_SIDED) == 0u && backfacing(view, pose, record.x, center, radius) {
            continue;
        }
        if view == 0u && frame.tests.x != 0u {
            let low = center - vec3<f32>(radius);
            let high = center + vec3<f32>(radius);
            if (frame.tests.x & TEST_SUBPIXEL) != 0u
                    && covers_no_pixel(frame.view_projection, low, high, frame.viewport.xy) {
                continue;
            }
            if (frame.tests.x & TEST_PREVIOUS) != 0u
                    && hidden(frame.previous_view_projection, low, high, depth_pyramid,
                        frame.viewport) {
                let slot = atomicAdd(&counters[RETESTED], 1u);
                if slot < frame.tests.z {
                    retest[slot] = vec4<u32>(item.x, cluster, frame.tests.y + item.z, 0u);
                    continue;
                }
            }
        }
        let index = atomicAdd(&counters[BINS + item.z], 1u);
        if index >= bin.capacity {
            atomicAdd(&counters[DROPPED_CLUSTERS], 1u);
            continue;
        }
        visible[bin.base + index] = vec2<u32>(item.x, cluster);
        atomicAdd(&counters[counts], 1u);
        atomicAdd(&counters[triangles], clusters[cluster * 3u + 2u].z >> 16u);
    }
}

@compute @workgroup_size(1)
fn retest_args() {
    let count = min(atomicLoad(&counters[RETESTED]), frame.tests.z);
    dispatch[3] = (count + GROUP - 1u) / GROUP;
    dispatch[4] = 1u;
    dispatch[5] = 1u;
}

@compute @workgroup_size(64)
fn retest_clusters(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= min(atomicLoad(&counters[RETESTED]), frame.tests.z) {
        return;
    }
    let entry = retest[id.x];
    let pose = poses[entry.x & ~MOVED_SLOT];
    let sphere = clusters[entry.y * 3u];
    let center = transformed(pose, bitcast<vec3<f32>>(sphere.xyz));
    let radius = bitcast<f32>(sphere.w) * largest_scale(pose);
    if hidden(frame.view_projection, center - vec3<f32>(radius), center + vec3<f32>(radius),
            depth_pyramid, frame.viewport) {
        atomicAdd(&counters[HIDDEN_CLUSTERS], 1u);
        return;
    }
    let bin = bins[entry.z];
    let index = atomicAdd(&counters[BINS + entry.z], 1u);
    if index >= bin.capacity {
        atomicAdd(&counters[DROPPED_CLUSTERS], 1u);
        return;
    }
    visible[bin.base + index] = vec2<u32>(entry.x, entry.y);
    atomicAdd(&counters[SELECTED_CLUSTERS], 1u);
    atomicAdd(&counters[SELECTED_TRIANGLES], clusters[entry.y * 3u + 2u].z >> 16u);
}

@compute @workgroup_size(64)
fn bin_args(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.counts.z {
        return;
    }
    let count = min(atomicLoad(&counters[BINS + id.x]), bins[id.x].capacity);
    draw_args[id.x * 4u] = CLUSTER_VERTICES;
    draw_args[id.x * 4u + 1u] = count;
    draw_args[id.x * 4u + 2u] = 0u;
    draw_args[id.x * 4u + 3u] = 0u;
}

@group(0) @binding(17) var<storage, read> pages: array<u32>;
@group(0) @binding(18) var<uniform> shadow_view: mat4x4<f32>;

const PAGE_WORDS: u32 = 32768u;

const COLLAPSED = vec4<f32>(2.0, 2.0, 0.5, 1.0);

@vertex
fn vs_shadow(@builtin(vertex_index) vertex: u32, @location(0) slot: u32,
        @location(1) cluster: u32) -> @builtin(position) vec4<f32> {
    let words = clusters[cluster * 3u + 2u];
    if vertex / 3u >= (words.z >> 16u) {
        return COLLAPSED;
    }
    let mesh = meshes[clusters[cluster * 3u + 1u].w];
    let page = mesh.bases.w + pages[words.w] * PAGE_WORDS;
    let packed = pages[page + words.y + vertex / 4u];
    let base = page + words.x + ((packed >> ((vertex % 4u) * 8u)) & 0xffu) * 4u;
    let w0 = pages[base];
    let w1 = pages[base + 1u];
    let steps = vec3<f32>(f32(w0 & 0xffffu), f32(w0 >> 16u), f32(w1 & 0xffffu));
    let position = mesh.position_min.xyz + steps * mesh.position_step.xyz;
    var clip = shadow_view * vec4<f32>(transformed(poses[slot & ~MOVED_SLOT], position), 1.0);
    clip.z = max(clip.z, 0.0);
    return clip;
}
