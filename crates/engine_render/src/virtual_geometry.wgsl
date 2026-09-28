// Virtual geometry (decision 044): each visible instance of a virtual mesh
// selects the clusters it draws. Its bounding volume hierarchy is walked one
// level per dispatch, from each level's root: a node whose largest error
// projects at or below the threshold holds only clusters too detailed for
// this view, and one outside the frustum holds only unseen ones. A leaf's
// group draws each of its clusters whose own error (the group it came from)
// projects at or below the threshold, unless culled by its sphere or normal
// cone; its group's error projecting above the threshold is what reached the
// leaf. Selected clusters append to their material bin's region, which the
// vertex shaders of `meshes.wgsl` draw 128 triangles a cluster from.
//
// Arena records use absolute indices: nodes name their children and groups,
// groups their clusters, and clusters their group, the group they came from,
// their mesh and their vertex and triangle words in the pages.

struct Frame {
    // Inward frustum planes, normalized.
    planes: array<vec4<f32>, 6>,
    // Eye position and the smallest distance errors project from.
    eye: vec4<f32>,
    // cot(fovy / 2) / 2, the largest projected error drawn (a fraction of
    // the viewport height), 0, 0.
    lod: vec4<f32>,
    // Batches, node queue capacity, group list capacity, bins.
    counts: vec4<u32>,
}

// One batch of instances of a virtual mesh with one material: its visible
// IDs start, its count (NONE: read the draw arguments of batch `draw`), its
// mesh, its bin and flags.
struct Batch {
    start: u32,
    count: u32,
    draw: u32,
    mesh: u32,
    bin: u32,
    flags: u32,
    pad0: u32,
    pad1: u32,
}

struct Mesh {
    // Node, group and cluster bases, and first page word.
    bases: vec4<u32>,
    // Levels (one root node each), hierarchy depth.
    info: vec4<u32>,
    position_min: vec4<f32>,
    position_step: vec4<f32>,
    // Texture coordinates of step zero, and step size.
    uv: vec4<f32>,
}

// A bin's region of `visible` and its capacity.
struct Bin {
    base: u32,
    capacity: u32,
    pad0: u32,
    pad1: u32,
}

// Transposed top three rows of an affine slot pose.
struct Rows {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> batches: array<Batch>;
@group(0) @binding(2) var<storage, read> meshes: array<Mesh>;
// Two words per node: sphere; error, group (or NONE), first child, children.
@group(0) @binding(3) var<storage, read> nodes: array<vec4<u32>>;
// Two words per group: sphere; error, first cluster, clusters, level.
@group(0) @binding(4) var<storage, read> groups: array<vec4<u32>>;
// Three words per cluster: culling sphere; cone, refined group (or NONE),
// group, mesh; vertex word, triangle word, vertices | triangles << 16, 0.
@group(0) @binding(5) var<storage, read> clusters: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> poses: array<Rows>;
// Visible IDs of GPU-presented frames, slot IDs of CPU-prepared ones.
@group(0) @binding(7) var<storage, read> ids: array<u32>;
// Indirect arguments of GPU-presented frames' batches (five words each).
@group(0) @binding(8) var<storage, read> draws: array<u32>;
// Node work: visible ID, node, bin, flags.
@group(0) @binding(9) var<storage, read_write> queue_a: array<vec4<u32>>;
@group(0) @binding(10) var<storage, read_write> queue_b: array<vec4<u32>>;
// Group work: visible ID, group, bin, flags.
@group(0) @binding(11) var<storage, read_write> work: array<vec4<u32>>;
// Queue A and B counts, the group count, entries dropped from the queues,
// the group list and the bins, clusters and triangles selected, then each
// bin's count.
@group(0) @binding(12) var<storage, read_write> counters: array<atomic<u32>>;
// Visible ID and cluster of each selected cluster, by bin.
@group(0) @binding(13) var<storage, read_write> visible: array<vec2<u32>>;
@group(0) @binding(14) var<storage, read> bins: array<Bin>;
// Indirect dispatches: the next traversal level's, then the group list's.
@group(0) @binding(15) var<storage, read_write> dispatch: array<u32>;
// Per bin: vertices, instances, first vertex, first instance.
@group(0) @binding(16) var<storage, read_write> draw_args: array<u32>;

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
const BINS: u32 = 8u;
// Batch flags.
const DOUBLE_SIDED: u32 = 1u;
const GROUP: u32 = 64u;
// Vertices of every cluster draw: three per triangle of the largest.
const CLUSTER_VERTICES: u32 = 384u;

fn transformed(pose: Rows, point: vec3<f32>) -> vec3<f32> {
    let p = vec4<f32>(point, 1.0);
    return vec3<f32>(dot(pose.r0, p), dot(pose.r1, p), dot(pose.r2, p));
}

// The largest length a pose scales by.
fn largest_scale(pose: Rows) -> f32 {
    let x = vec3<f32>(pose.r0.x, pose.r1.x, pose.r2.x);
    let y = vec3<f32>(pose.r0.y, pose.r1.y, pose.r2.y);
    let z = vec3<f32>(pose.r0.z, pose.r1.z, pose.r2.z);
    return sqrt(max(dot(x, x), max(dot(y, y), dot(z, z))));
}

fn outside(center: vec3<f32>, radius: f32) -> bool {
    for (var plane = 0u; plane < 6u; plane++) {
        if dot(frame.planes[plane].xyz, center) + frame.planes[plane].w < -radius {
            return true;
        }
    }
    return false;
}

// `error` seen from the sphere's nearest point, as a fraction of the
// viewport height; infinite errors never pass a threshold.
fn projected(center: vec3<f32>, radius: f32, error: f32) -> f32 {
    let distance = max(length(center - frame.eye.xyz) - radius, frame.eye.w);
    return error / distance * frame.lod.x;
}

fn push_node(queue: u32, item: vec4<u32>) {
    let index = atomicAdd(&counters[queue], 1u);
    if index >= frame.counts.y {
        atomicAdd(&counters[DROPPED_QUEUE], 1u);
        return;
    }
    if queue == QUEUE_A {
        queue_a[index] = item;
    } else {
        queue_b[index] = item;
    }
}

// Each visible instance of each batch starts at every level's root.
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
    for (var level = 0u; level < mesh.info.x; level++) {
        push_node(QUEUE_A, vec4<u32>(visible_id, mesh.bases.x + level, batch.bin, batch.flags));
    }
}

// Dispatch arguments of the level reading queue `source`, whose output
// queue `sink` starts empty.
fn level_args(source: u32, sink: u32) {
    let count = min(atomicLoad(&counters[source]), frame.counts.y);
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
    if index >= min(atomicLoad(&counters[source]), frame.counts.y) {
        return;
    }
    var item: vec4<u32>;
    if source == QUEUE_A {
        item = queue_a[index];
    } else {
        item = queue_b[index];
    }
    let pose = poses[item.x & ~MOVED_SLOT];
    let scale = largest_scale(pose);
    let sphere = nodes[item.y * 2u];
    let node = nodes[item.y * 2u + 1u];
    let center = transformed(pose, bitcast<vec3<f32>>(sphere.xyz));
    let radius = bitcast<f32>(sphere.w) * scale;
    if outside(center, radius)
            || projected(center, radius, bitcast<f32>(node.x) * scale) <= frame.lod.y {
        return;
    }
    if node.y != NONE {
        let slot = atomicAdd(&counters[WORK], 1u);
        if slot < frame.counts.z {
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
    let count = min(atomicLoad(&counters[WORK]), frame.counts.z);
    dispatch[3] = (count + GROUP - 1u) / GROUP;
    dispatch[4] = 1u;
    dispatch[5] = 1u;
}

// Whether the normal cone `cone` (8-bit snorm axis and cutoff) of a cluster
// posed by `pose` faces away from the eye everywhere in its sphere
// (meshoptimizer's sphere test). Mirroring poses flip it; poses scaling
// unevenly skip it.
fn backfacing(pose: Rows, cone: u32, center: vec3<f32>, radius: f32) -> bool {
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
    let to_center = center - frame.eye.xyz;
    return dot(to_center, axis) >= cutoff * length(to_center) + radius;
}

@compute @workgroup_size(64)
fn select_clusters(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= min(atomicLoad(&counters[WORK]), frame.counts.z) {
        return;
    }
    let item = work[id.x];
    let pose = poses[item.x & ~MOVED_SLOT];
    let scale = largest_scale(pose);
    let group = groups[item.y * 2u + 1u];
    let bin = bins[item.z];
    for (var cluster = group.y; cluster < group.y + group.z; cluster++) {
        let sphere = clusters[cluster * 3u];
        let record = clusters[cluster * 3u + 1u];
        // Its own error: the group it came from, precise enough here.
        if record.y != NONE {
            let source_sphere = groups[record.y * 2u];
            let source = groups[record.y * 2u + 1u];
            let source_center = transformed(pose, bitcast<vec3<f32>>(source_sphere.xyz));
            if projected(source_center, bitcast<f32>(source_sphere.w) * scale,
                    bitcast<f32>(source.x) * scale) > frame.lod.y {
                continue;
            }
        }
        let center = transformed(pose, bitcast<vec3<f32>>(sphere.xyz));
        let radius = bitcast<f32>(sphere.w) * scale;
        if outside(center, radius) {
            continue;
        }
        if (item.w & DOUBLE_SIDED) == 0u && backfacing(pose, record.x, center, radius) {
            continue;
        }
        let index = atomicAdd(&counters[BINS + item.z], 1u);
        if index >= bin.capacity {
            atomicAdd(&counters[DROPPED_CLUSTERS], 1u);
            continue;
        }
        visible[bin.base + index] = vec2<u32>(item.x, cluster);
        atomicAdd(&counters[SELECTED_CLUSTERS], 1u);
        atomicAdd(&counters[SELECTED_TRIANGLES], clusters[cluster * 3u + 2u].z >> 16u);
    }
}

// Each bin draws its selected clusters (within its capacity), 128
// triangles' vertices each.
@compute @workgroup_size(64)
fn bin_args(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.counts.w {
        return;
    }
    let count = min(atomicLoad(&counters[BINS + id.x]), bins[id.x].capacity);
    draw_args[id.x * 4u] = CLUSTER_VERTICES;
    draw_args[id.x * 4u + 1u] = count;
    draw_args[id.x * 4u + 2u] = 0u;
    draw_args[id.x * 4u + 3u] = 0u;
}
