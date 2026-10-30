import package::face::heightfield::{TerrainSubdivision, terrain, terrain_level_height};
import package::frame::view::detail_across;
import package::terrain::draw::terrain_displacement;
import package::terrain::field::{SURFACE_MESHLET_INDICES, SURFACE_MESHLET_LEVEL, SurfaceTriangle};
import package::terrain::surface::{SurfaceWater, surface_ground_error, surface_half, surface_water};

@group(2) @binding(27) var<uniform> subdivision: TerrainSubdivision;
@group(2) @binding(28) var<storage, read_write> subdivision_tree: array<atomic<u32>>;
@group(2) @binding(29) var<storage, read_write> subdivision_nodes: array<u32>;
@group(2) @binding(30) var<storage, read_write> subdivision_leaves: array<SurfaceTriangle>;
@group(2) @binding(31) var<storage, read_write> subdivision_args: array<atomic<u32>>;

const SUBDIVISION_GROUP: u32 = 64u;
const SUBDIVISION_ARGS: u32 = 16u;
const SUBDIVISION_LEAVES: u32 = 3u;
const SUBDIVISION_DRAW: u32 = 4u;
const SUBDIVISION_RESERVED: u32 = 9u;
const SUBDIVISION_REFUSED: u32 = 10u;
const SUBDIVISION_COUNTED: u32 = 8u;
const SUBDIVISION_UNSEEN: f32 = 8.0;
const SUBDIVISION_SPLIT: u32 = 1u;
const SUBDIVISION_MERGE: u32 = 2u;

fn subdivision_bit(node: u32) -> u32 {
    let deepest = subdivision.tree.x;
    return (node << (deepest - firstLeadingBit(node))) - (1u << deepest);
}

fn subdivision_count(node: u32) -> u32 {
    let deepest = subdivision.tree.x;
    let depth = firstLeadingBit(node);
    if depth + SUBDIVISION_COUNTED <= deepest {
        return atomicLoad(&subdivision_tree[node]);
    }
    let span = deepest - depth;
    let first = (node << span) - (1u << deepest);
    let word = subdivision.tree.y + first / 32u;
    if span >= 5u {
        var count = 0u;
        for (var index = 0u; index < (1u << (span - 5u)); index++) {
            count += countOneBits(atomicLoad(&subdivision_tree[word + index]));
        }
        return count;
    }
    let mask = ((1u << (1u << span)) - 1u) << (first % 32u);
    return countOneBits(atomicLoad(&subdivision_tree[word]) & mask);
}

fn subdivision_leaf(index: u32) -> u32 {
    var node = 1u;
    var count = atomicLoad(&subdivision_tree[1]);
    var rest = index;
    for (var depth = 0u; depth < subdivision.tree.x && count > 1u; depth++) {
        let first = subdivision_count(2u * node);
        if rest < first {
            node = 2u * node;
            count = first;
        } else {
            rest -= first;
            node = 2u * node + 1u;
            count -= first;
        }
    }
    return node;
}

fn subdivision_triangle(node: u32) -> SurfaceTriangle {
    let side = subdivision.square.x;
    let depth = firstLeadingBit(node);
    var triangle = SurfaceTriangle(vec2<f32>(side), vec2<f32>(0.0), vec2<f32>(0.0, side));
    if ((node >> (depth - 1u)) & 1u) == 1u {
        triangle = SurfaceTriangle(vec2<f32>(0.0), vec2<f32>(side), vec2<f32>(side, 0.0));
    }
    for (var bit = depth - 1u; bit > 0u; bit--) {
        triangle = surface_half(triangle, (node >> (bit - 1u)) & 1u);
    }
    return triangle;
}

fn subdivision_neighbor(node: u32) -> u32 {
    let depth = firstLeadingBit(node);
    var across = 5u - (node >> (depth - 1u));
    var after = 0u;
    var before = 0u;
    for (var bit = depth - 1u; bit > 0u; bit--) {
        let parent = node >> bit;
        if ((node >> (bit - 1u)) & 1u) == 0u {
            let hypotenuse = select(2u * before + 1u, 0u, before == 0u);
            after = select(2u * across + 1u, 0u, across == 0u);
            before = 2u * parent + 1u;
            across = hypotenuse;
        } else {
            let hypotenuse = select(2u * after, 0u, after == 0u);
            before = select(2u * across, 0u, across == 0u);
            after = 2u * parent;
            across = hypotenuse;
        }
    }
    return across;
}

struct SubdivisionPoint {
    world: vec3<f32>,
    lift: vec3<f32>,
    kept: vec3<f32>,
    found: bool,
}

fn subdivision_point(local: vec2<f32>, displaced: bool, pieces: f32) -> SubdivisionPoint {
    let held = terrain_level_height(local, 0u);
    var point: SubdivisionPoint;
    point.found = held.found;
    point.world = terrain.origin.xyz + vec3<f32>(local.x, held.height, local.y);
    if displaced {
        let lift = terrain_displacement(point.world, local, pieces);
        point.lift = lift.moved;
        point.kept = lift.kept;
        point.world += point.lift;
    }
    return point;
}

fn subdivision_seen(low: vec3<f32>, high: vec3<f32>) -> bool {
    for (var index = 0u; index < 6u; index++) {
        let plane = subdivision.planes[index];
        let corner = select(low, high, plane.xyz >= vec3<f32>(0.0));
        if dot(plane.xyz, corner) + plane.w < 0.0 {
            return false;
        }
    }
    return true;
}

struct SubdivisionMeasure {
    drawn: bool,
    splits: bool,
}

fn subdivision_measure(triangle: SurfaceTriangle) -> SubdivisionMeasure {
    var measure: SubdivisionMeasure;
    let low = min(min(triangle.a, triangle.b), triangle.c);
    let high = max(max(triangle.a, triangle.b), triangle.c);
    let extent = subdivision.extent;
    let cutout = subdivision.cutout;
    if any(low >= extent.zw) || any(high <= extent.xy)
        || (all(low >= cutout.xy) && all(high <= cutout.zw)) {
        return measure;
    }
    let water = subdivision.square.z > 0.5;
    var over = SurfaceWater(false, 0.0);
    if water {
        over = surface_water(triangle);
        if !over.reached {
            return measure;
        }
    }
    let pieces = length(triangle.c - triangle.a) / f32(1u << SURFACE_MESHLET_LEVEL);
    let a = subdivision_point(triangle.a, !water, pieces);
    let b = subdivision_point(triangle.b, !water, pieces);
    let c = subdivision_point(triangle.c, !water, pieces);
    let center = subdivision_point((triangle.a + triangle.b + triangle.c) / 3.0, !water, pieces);
    measure.drawn = water || a.found || b.found || c.found || center.found;
    if !measure.drawn {
        return measure;
    }
    if any(low < extent.xy) || any(high > extent.zw)
        || (all(high > cutout.xy) && all(low < cutout.zw)) {
        measure.splits = true;
        return measure;
    }
    let ab = subdivision_point((triangle.a + triangle.b) * 0.5, !water, pieces);
    let bc = subdivision_point((triangle.b + triangle.c) * 0.5, !water, pieces);
    let ca = subdivision_point((triangle.c + triangle.a) * 0.5, !water, pieces);
    var bound = over.error;
    var rise = max(high.x - low.x, high.y - low.y);
    if !water {
        let normal = cross(b.world - a.world, c.world - a.world);
        let unit = normal / max(length(normal), 1.0e-12);
        let bent = abs(vec4<f32>(dot(ab.kept - (a.kept + b.kept) * 0.5, unit),
            dot(bc.kept - (b.kept + c.kept) * 0.5, unit),
            dot(ca.kept - (c.kept + a.kept) * 0.5, unit),
            dot(center.kept - (a.kept + b.kept + c.kept) / 3.0, unit)));
        let finer = max(max(max(length(a.lift - a.kept), length(b.lift - b.kept)),
            max(length(c.lift - c.kept), length(center.lift - center.kept))),
            max(max(length(ab.lift - ab.kept), length(bc.lift - bc.kept)),
            length(ca.lift - ca.kept)));
        let parts = f32(1u << SURFACE_MESHLET_LEVEL);
        let displacement = finer + max(max(bent.x, bent.y), max(bent.z, bent.w)) / (parts * parts);
        bound = surface_ground_error(triangle) + displacement;
        rise = 0.0;
    }
    let up = vec3<f32>(0.0, rise, 0.0);
    let risen = vec3<f32>(0.0, bound + rise, 0.0);
    var points = array<vec3<f32>, 7>(a.world, b.world, c.world, center.world, ab.world,
        bc.world, ca.world);
    var across = detail_across(points[0]);
    var lowest = points[0];
    var highest = points[0];
    for (var index = 0u; index < 7u; index++) {
        let point = points[index];
        across = min(across, detail_across(point));
        if water {
            across = min(across, detail_across(point + risen));
        }
        lowest = min(lowest, point);
        highest = max(highest, point);
    }
    let seen = subdivision_seen(lowest - vec3<f32>(bound), highest + vec3<f32>(bound) + up);
    let allowed = subdivision.square.y * select(SUBDIVISION_UNSEEN, 1.0, seen);
    measure.splits = bound / max(across, 1.0e-6) > allowed;
    return measure;
}

fn subdivision_merges(base: u32) -> bool {
    let top = subdivision_neighbor(base);
    if top != 0u && top < base {
        return false;
    }
    if subdivision_count(base) != 2u || (top != 0u && subdivision_count(top) != 2u) {
        return false;
    }
    if subdivision_measure(subdivision_triangle(base)).splits {
        return false;
    }
    return top == 0u || !subdivision_measure(subdivision_triangle(top)).splits;
}

fn subdivision_tally(bit: u32, gained: bool) {
    let deepest = subdivision.tree.x;
    let heap = (1u << deepest) + bit;
    for (var shift = SUBDIVISION_COUNTED; shift <= deepest; shift++) {
        if gained {
            atomicAdd(&subdivision_tree[heap >> shift], 1u);
        } else {
            atomicSub(&subdivision_tree[heap >> shift], 1u);
        }
    }
}

fn subdivision_mark(node: u32, apply: bool) -> bool {
    let bit = subdivision_bit(node);
    let word = subdivision.tree.y + bit / 32u;
    let mask = 1u << (bit % 32u);
    if !apply {
        return (atomicLoad(&subdivision_tree[word]) & mask) == 0u;
    }
    if (atomicOr(&subdivision_tree[word], mask) & mask) != 0u {
        return false;
    }
    subdivision_tally(bit, true);
    return true;
}

fn subdivision_clear(node: u32) {
    let bit = subdivision_bit(node);
    let mask = 1u << (bit % 32u);
    if (atomicAnd(&subdivision_tree[subdivision.tree.y + bit / 32u], ~mask) & mask) != 0u {
        subdivision_tally(bit, false);
    }
}

fn subdivision_chain(node: u32, apply: bool) -> u32 {
    if !subdivision_mark(2u * node + 1u, apply) {
        return 0u;
    }
    var count = 1u;
    var across = subdivision_neighbor(node);
    loop {
        if across <= 1u || !subdivision_mark(2u * across + 1u, apply) {
            break;
        }
        count += 1u;
        let parent = across >> 1u;
        if parent <= 1u || !subdivision_mark(2u * parent + 1u, apply) {
            break;
        }
        count += 1u;
        across = subdivision_neighbor(parent);
    }
    return count;
}

@compute @workgroup_size(1)
fn subdivide_begin() {
    let args = subdivision.frame.y * SUBDIVISION_ARGS;
    let leaves = atomicLoad(&subdivision_tree[1]);
    atomicStore(&subdivision_args[args], (leaves + SUBDIVISION_GROUP - 1u) / SUBDIVISION_GROUP);
    atomicStore(&subdivision_args[args + 1u], 1u);
    atomicStore(&subdivision_args[args + 2u], 1u);
    atomicStore(&subdivision_args[args + SUBDIVISION_LEAVES], leaves);
    let draw = args + SUBDIVISION_DRAW;
    atomicStore(&subdivision_args[draw], SURFACE_MESHLET_INDICES);
    atomicStore(&subdivision_args[draw + 1u], leaves);
    atomicStore(&subdivision_args[draw + 2u], 0u);
    atomicStore(&subdivision_args[draw + 3u], 0u);
    atomicStore(&subdivision_args[draw + 4u], 0u);
    atomicStore(&subdivision_args[args + SUBDIVISION_RESERVED], 0u);
    atomicStore(&subdivision_args[args + SUBDIVISION_REFUSED], 0u);
}

@compute @workgroup_size(SUBDIVISION_GROUP)
fn subdivide_decide(@builtin(global_invocation_id) id: vec3<u32>) {
    let args = subdivision.frame.y * SUBDIVISION_ARGS;
    if id.x >= atomicLoad(&subdivision_args[args + SUBDIVISION_LEAVES]) {
        return;
    }
    let node = subdivision_leaf(id.x);
    let triangle = subdivision_triangle(node);
    let measure = subdivision_measure(triangle);
    let at = subdivision.tree.z + id.x;
    var drawn = SurfaceTriangle(triangle.a, triangle.a, triangle.a);
    if measure.drawn {
        drawn = triangle;
    }
    subdivision_leaves[at] = drawn;
    let depth = firstLeadingBit(node);
    var decision = 0u;
    if subdivision.frame.x == 0u {
        if measure.splits && depth < subdivision.tree.x {
            decision = SUBDIVISION_SPLIT;
        }
    } else if depth >= 2u && (node & 1u) == 0u && subdivision_merges(node >> 1u) {
        decision = SUBDIVISION_MERGE;
    }
    subdivision_nodes[at] = node | (decision << 30u);
}

@compute @workgroup_size(SUBDIVISION_GROUP)
fn subdivide_apply(@builtin(global_invocation_id) id: vec3<u32>) {
    let args = subdivision.frame.y * SUBDIVISION_ARGS;
    let leaves = atomicLoad(&subdivision_args[args + SUBDIVISION_LEAVES]);
    if id.x >= leaves {
        return;
    }
    let word = subdivision_nodes[subdivision.tree.z + id.x];
    let node = word & 0x3fffffffu;
    let decision = word >> 30u;
    if decision == SUBDIVISION_SPLIT {
        let adds = subdivision_chain(node, false);
        let room = subdivision.tree.w - min(leaves, subdivision.tree.w);
        if adds > 0u {
            if atomicAdd(&subdivision_args[args + SUBDIVISION_RESERVED], adds) + adds > room {
                atomicAdd(&subdivision_args[args + SUBDIVISION_REFUSED], 1u);
            } else {
                subdivision_chain(node, true);
            }
        }
    } else if decision == SUBDIVISION_MERGE {
        let base = node >> 1u;
        subdivision_clear(2u * base + 1u);
        let top = subdivision_neighbor(base);
        if top != 0u {
            subdivision_clear(2u * top + 1u);
        }
    }
}
