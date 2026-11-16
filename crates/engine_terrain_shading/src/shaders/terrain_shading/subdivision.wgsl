import package::engine::drawn::frustum_reach;
import package::engine::view::{camera, detail_across};
import package::place::heightfield::{SurfaceTriangle, TerrainSubdivision};
import package::terrain_shading::draw::terrain_displacement;
import package::terrain_shading::faces::heightfield::{terrain, terrain_level_height, use_terrain};

import package::terrain_shading::surface::{SurfaceWater, surface_cut, surface_ground_error,
    surface_half, surface_kept, surface_marked, surface_seen, surface_water, surface_whole};
import package::generated::terrain_shading::{SUBDIVISION_ARGS, SUBDIVISION_COUNTED,
    SUBDIVISION_DRAW, SUBDIVISION_GROUP, SUBDIVISION_LEAVES, SUBDIVISION_REFUSED,
    SUBDIVISION_RESERVED, SUBDIVISION_WATERED, SUBDIVISION_WHOLE, SUBDIVISION_HOLED,
    SUBDIVISION_LISTED, SUBDIVISION_LISTS, SURFACE_LISTED, SURFACE_LISTS, SURFACE_LIST_SHIFT,
    SURFACE_MESHLET_INDICES, SURFACE_MESHLET_LEVEL, SURFACE_ROLE_SHIFT, SURFACE_VERTEX_STRIDE};

@group(1) @binding(0) var<uniform> subdivision: TerrainSubdivision;
@group(1) @binding(1) var<storage, read_write> subdivision_tree: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> subdivision_nodes: array<u32>;
@group(1) @binding(3) var<storage, read_write> subdivision_leaves: array<SurfaceTriangle>;
@group(1) @binding(4) var<storage, read_write> subdivision_args: array<atomic<u32>>;
@group(1) @binding(5) var<storage, read_write> subdivision_watered: array<u32>;
@group(1) @binding(6) var<storage, read_write> subdivision_lists: array<u32>;

const SUBDIVISION_SPLIT: u32 = 1u;
const SUBDIVISION_MERGE: u32 = 2u;
const SUBDIVISION_RAN: u32 = 1u << 31u;
const SUBDIVISION_CHANGED: u32 = 1u << 30u;
const SUBDIVISION_APPLIED: u32 = 1u << 29u;
const SUBDIVISION_QUIET: u32 = 3u;
const SUBDIVISION_SETTLED: u32 = 2u;

fn subdivision_state() -> u32 {
    return subdivision.tree.y + (1u << (subdivision.tree.x - 5u));
}

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

fn subdivision_order(node: u32) -> u32 {
    let side = subdivision.square.x;
    let depth = firstLeadingBit(node);
    let eye = camera.position.xz - terrain.origin.xz;
    var triangle = SurfaceTriangle(vec2<f32>(side), vec2<f32>(0.0), vec2<f32>(0.0, side));
    var order = 0u;
    for (var bit = depth; bit > 0u; bit--) {
        let child = (node >> (bit - 1u)) & 1u;
        var near = select(0u, 1u, eye.x > eye.y);
        if bit < depth {
            let edge = (triangle.a + triangle.b) * 0.5 - triangle.c;
            let across = vec2<f32>(-edge.y, edge.x);
            near = select(0u, 1u,
                dot(across, eye - triangle.c) * dot(across, triangle.b - triangle.c) > 0.0);
            triangle = surface_half(triangle, child);
        } else if child == 1u {
            triangle = SurfaceTriangle(vec2<f32>(0.0), vec2<f32>(side), vec2<f32>(side, 0.0));
        }
        if child != near {
            order += subdivision_count(2u * (node >> bit) + near);
        }
    }
    return order;
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

fn subdivision_point(local: vec2<f32>, pieces: f32) -> SubdivisionPoint {
    let held = terrain_level_height(local, 0u);
    var point: SubdivisionPoint;
    point.found = held.found;
    point.world = terrain.origin.xyz + vec3<f32>(local.x, held.height, local.y);
    let lift = terrain_displacement(point.world, local, pieces);
    point.lift = lift.moved;
    point.kept = lift.kept;
    point.world += point.lift;
    return point;
}

fn subdivision_corner(triangle: SurfaceTriangle, corner: u32) -> vec2<f32> {
    if corner == 0u {
        return triangle.a;
    }
    if corner == 1u {
        return triangle.b;
    }
    if corner == 2u {
        return triangle.c;
    }
    if corner == 3u {
        return (triangle.a + triangle.b + triangle.c) / 3.0;
    }
    if corner == 4u {
        return (triangle.a + triangle.b) * 0.5;
    }
    if corner == 5u {
        return (triangle.b + triangle.c) * 0.5;
    }
    return (triangle.c + triangle.a) * 0.5;
}

fn subdivision_middle(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, corner: u32) -> vec3<f32> {
    if corner == 4u {
        return (a + b) * 0.5;
    }
    if corner == 5u {
        return (b + c) * 0.5;
    }
    return (c + a) * 0.5;
}

fn subdivision_seen(low: vec3<f32>, high: vec3<f32>) -> bool {
    for (var index = 0u; index < 6u; index++) {
        if frustum_reach(subdivision.planes[index], low, high) < 0.0 {
            return false;
        }
    }
    return true;
}

struct SubdivisionMeasure {
    drawn: bool,
    splits: bool,
    watered: bool,
}

fn subdivision_water(triangle: SurfaceTriangle) -> SurfaceWater {
    if subdivision.frame.w == 0u {
        return SurfaceWater(false, 0.0);
    }
    return surface_water(triangle);
}

fn subdivision_outside(low: vec2<f32>, high: vec2<f32>) -> bool {
    let extent = subdivision.extent;
    let cutout = subdivision.cutout;
    return any(low >= extent.zw) || any(high <= extent.xy)
        || (all(low >= cutout.xy) && all(high <= cutout.zw));
}

fn subdivision_drawn(triangle: SurfaceTriangle) -> SubdivisionMeasure {
    var measure: SubdivisionMeasure;
    let low = min(min(triangle.a, triangle.b), triangle.c);
    let high = max(max(triangle.a, triangle.b), triangle.c);
    if subdivision_outside(low, high) {
        return measure;
    }
    measure.drawn = terrain_level_height(triangle.a, 0u).found
        || terrain_level_height(triangle.b, 0u).found
        || terrain_level_height(triangle.c, 0u).found
        || terrain_level_height((triangle.a + triangle.b + triangle.c) / 3.0, 0u).found;
    measure.watered = measure.drawn && subdivision_water(triangle).reached;
    return measure;
}

fn subdivision_exceeds(bound: f32, across: f32, lowest: vec3<f32>, highest: vec3<f32>,
        rise: f32, pixels: f32) -> bool {
    let up = vec3<f32>(0.0, rise, 0.0);
    let seen = subdivision_seen(lowest - vec3<f32>(bound), highest + vec3<f32>(bound) + up);
    let allowed = pixels * camera.pixel_error * select(subdivision.square.w, 1.0, seen);
    return bound / max(across, 1.0e-6) > allowed;
}

fn subdivision_measure(triangle: SurfaceTriangle) -> SubdivisionMeasure {
    var measure: SubdivisionMeasure;
    let low = min(min(triangle.a, triangle.b), triangle.c);
    let high = max(max(triangle.a, triangle.b), triangle.c);
    let extent = subdivision.extent;
    let cutout = subdivision.cutout;
    if subdivision_outside(low, high) {
        return measure;
    }
    let pieces = length(triangle.c - triangle.a) / f32(1u << SURFACE_MESHLET_LEVEL);
    let rise = max(high.x - low.x, high.y - low.y);
    let over = subdivision_water(triangle);
    let edged = any(low < extent.xy) || any(high > extent.zw)
        || (all(high > cutout.xy) && all(low < cutout.zw));
    let risen = vec3<f32>(0.0, over.error + rise, 0.0);
    var first = vec3<f32>(0.0);
    var second = vec3<f32>(0.0);
    var kept = mat3x3<f32>();
    var wet = vec3<f32>(0.0);
    var unit = vec3<f32>(0.0);
    var finer = 0.0;
    var bent = 0.0;
    var across = 0.0;
    var risen_across = 0.0;
    var lowest = vec3<f32>(0.0);
    var highest = vec3<f32>(0.0);
    for (var corner = 0u; corner < 7u; corner++) {
        let point = subdivision_point(subdivision_corner(triangle, corner), pieces);
        let moved = length(point.lift - point.kept);
        let near = detail_across(point.world);
        let near_wet = detail_across(point.world + risen);
        measure.drawn = measure.drawn || point.found;
        if corner == 0u {
            first = point.world;
            finer = moved;
            across = near;
            lowest = point.world;
            highest = point.world;
        } else {
            finer = max(finer, moved);
            across = min(across, near);
            lowest = min(lowest, point.world);
            highest = max(highest, point.world);
        }
        if corner == 1u {
            second = point.world;
        }
        if corner < 3u {
            kept = mat3x3<f32>(select(kept[0], point.kept, corner == 0u),
                select(kept[1], point.kept, corner == 1u), select(kept[2], point.kept, corner == 2u));
            wet = select(wet, vec3<f32>(near_wet), vec3<u32>(corner) == vec3<u32>(0u, 1u, 2u));
        }
        if corner == 2u {
            let normal = cross(second - first, point.world - first);
            unit = normal / max(length(normal), 1.0e-12);
        }
        if corner == 3u {
            if !measure.drawn {
                return measure;
            }
            measure.watered = over.reached;
            if edged {
                measure.splits = true;
                return measure;
            }
            bent = abs(dot(point.kept - (kept[0] + kept[1] + kept[2]) / 3.0, unit));
            risen_across = min(min(wet.x, wet.y), min(wet.z, near_wet));
        } else if corner > 3u {
            bent = max(bent, abs(dot(point.kept - subdivision_middle(kept[0], kept[1], kept[2],
                corner), unit)));
            risen_across = min(risen_across, near_wet);
        }
    }
    let parts = f32(1u << SURFACE_MESHLET_LEVEL);
    let displacement = finer + bent / (parts * parts);
    let wet_across = select(across, min(across, risen_across), rise > 0.0);
    measure.splits = subdivision_exceeds(surface_ground_error(triangle) + displacement, across,
        lowest, highest, 0.0, subdivision.square.y)
        || (over.reached && subdivision_exceeds(over.error, wet_across, lowest, highest, rise,
        subdivision.square.z));
    return measure;
}

fn subdivision_mergeable(base: u32, top: u32) -> bool {
    if top != 0u && top < base {
        return false;
    }
    return subdivision_count(base) == 2u && (top == 0u || subdivision_count(top) == 2u);
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

fn subdivision_record(draw: u32, count: u32, base: u32) {
    atomicStore(&subdivision_args[draw], SURFACE_MESHLET_INDICES);
    atomicStore(&subdivision_args[draw + 1u], count);
    atomicStore(&subdivision_args[draw + 2u], 0u);
    atomicStore(&subdivision_args[draw + 3u], base);
    atomicStore(&subdivision_args[draw + 4u], 0u);
}

@compute @workgroup_size(1)
fn subdivide_begin() {
    use_terrain(subdivision.frame.z);
    let args = subdivision.frame.y * SUBDIVISION_ARGS;
    let state_at = subdivision_state();
    let state = atomicLoad(&subdivision_tree[state_at]);
    var quiet = state & SUBDIVISION_QUIET;
    if (state & SUBDIVISION_RAN) != 0u {
        quiet = select(min(quiet + 1u, SUBDIVISION_SETTLED), 0u,
            (state & (SUBDIVISION_CHANGED | SUBDIVISION_APPLIED)) != SUBDIVISION_APPLIED);
    }
    atomicStore(&subdivision_tree[state_at], quiet);
    let settled = quiet == SUBDIVISION_SETTLED;
    let leaves = atomicLoad(&subdivision_tree[1]);
    atomicStore(&subdivision_args[args],
        select((leaves + SUBDIVISION_GROUP - 1u) / SUBDIVISION_GROUP, 0u, settled));
    atomicStore(&subdivision_args[args + 1u], 1u);
    atomicStore(&subdivision_args[args + 2u], 1u);
    atomicStore(&subdivision_args[args + SUBDIVISION_LEAVES], leaves);
    let base = subdivision.frame.z * SURFACE_VERTEX_STRIDE;
    subdivision_record(args + SUBDIVISION_DRAW, leaves, base);
    subdivision_record(args + SUBDIVISION_WHOLE, leaves, base + (1u << SURFACE_ROLE_SHIFT));
    subdivision_record(args + SUBDIVISION_HOLED, leaves, base + (2u << SURFACE_ROLE_SHIFT));
    for (var list = 0u; list < SURFACE_LISTS; list++) {
        subdivision_record(args + SUBDIVISION_LISTS + 5u * list, 0u,
            base + (SURFACE_LISTED << SURFACE_ROLE_SHIFT) + (list << SURFACE_LIST_SHIFT));
    }
    atomicStore(&subdivision_args[args + SUBDIVISION_LISTED],
        (leaves + SUBDIVISION_GROUP - 1u) / SUBDIVISION_GROUP);
    atomicStore(&subdivision_args[args + SUBDIVISION_LISTED + 1u], 1u);
    atomicStore(&subdivision_args[args + SUBDIVISION_LISTED + 2u], 1u);
    atomicStore(&subdivision_args[args + SUBDIVISION_RESERVED], 0u);
    atomicStore(&subdivision_args[args + SUBDIVISION_REFUSED], 0u);
    atomicStore(&subdivision_args[args + SUBDIVISION_WATERED],
        select(0u, atomicLoad(&subdivision_tree[state_at + 1u]), settled));
}

@compute @workgroup_size(SUBDIVISION_GROUP)
fn subdivide_decide(@builtin(global_invocation_id) id: vec3<u32>) {
    use_terrain(subdivision.frame.z);
    let args = subdivision.frame.y * SUBDIVISION_ARGS;
    if id.x >= atomicLoad(&subdivision_args[args + SUBDIVISION_LEAVES]) {
        return;
    }
    if id.x == 0u {
        atomicOr(&subdivision_tree[subdivision_state()], SUBDIVISION_RAN);
    }
    let node = subdivision_leaf(id.x);
    let triangle = subdivision_triangle(node);
    let depth = firstLeadingBit(node);
    let splitting = subdivision.frame.x == 0u;
    var asked = triangle;
    var asks = 0u;
    var top = 0u;
    if splitting {
        asks = select(0u, 1u, depth < subdivision.tree.x);
    } else if depth >= 2u && (node & 1u) == 0u {
        let base = node >> 1u;
        top = subdivision_neighbor(base);
        if subdivision_mergeable(base, top) {
            asked = subdivision_triangle(base);
            asks = select(1u, 2u, top != 0u);
        }
    }
    var measure: SubdivisionMeasure;
    if asks == 0u || !splitting {
        measure = subdivision_drawn(triangle);
    }
    var merges = asks > 0u;
    for (var ask = 0u; ask < asks; ask++) {
        let measured = subdivision_measure(asked);
        if splitting {
            measure = measured;
        } else if measured.splits {
            merges = false;
            break;
        }
        if ask + 1u < asks {
            asked = subdivision_triangle(top);
        }
    }
    let at = subdivision.tree.z + subdivision_order(node);
    var drawn = SurfaceTriangle(triangle.a, triangle.a, triangle.a);
    if measure.drawn {
        drawn = surface_marked(triangle, !surface_cut(triangle));
    }
    subdivision_leaves[at] = drawn;
    if measure.watered {
        let place = atomicAdd(&subdivision_args[args + SUBDIVISION_WATERED], 1u);
        subdivision_watered[subdivision.tree.z + place] = at;
    }
    var decision = 0u;
    if splitting {
        if measure.splits && depth < subdivision.tree.x {
            decision = SUBDIVISION_SPLIT;
        }
    } else if merges {
        decision = SUBDIVISION_MERGE;
    }
    if decision != 0u {
        atomicOr(&subdivision_tree[subdivision_state()], SUBDIVISION_CHANGED);
    }
    subdivision_nodes[subdivision.tree.z + id.x] = node | (decision << 30u);
}

@compute @workgroup_size(SUBDIVISION_GROUP)
fn subdivide_apply(@builtin(global_invocation_id) id: vec3<u32>) {
    use_terrain(subdivision.frame.z);
    let args = subdivision.frame.y * SUBDIVISION_ARGS;
    let leaves = atomicLoad(&subdivision_args[args + SUBDIVISION_LEAVES]);
    if id.x >= leaves {
        return;
    }
    if id.x == 0u {
        atomicStore(&subdivision_tree[subdivision_state() + 1u],
            atomicLoad(&subdivision_args[args + SUBDIVISION_WATERED]));
        atomicOr(&subdivision_tree[subdivision_state()], SUBDIVISION_APPLIED);
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

var<workgroup> listed_kinds: array<u32, SUBDIVISION_GROUP>;
var<workgroup> listed_firsts: array<u32, 2>;

fn subdivision_kind(index: u32, leaves: u32) -> u32 {
    if index >= leaves {
        return 0u;
    }
    let stored = subdivision_leaves[subdivision.tree.z + index];
    if !surface_seen(surface_kept(stored)) {
        return 0u;
    }
    return select(2u, 1u, surface_whole(stored));
}

@compute @workgroup_size(SUBDIVISION_GROUP)
fn subdivide_list_drawn(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_index) local: u32) {
    use_terrain(subdivision.frame.z);
    let args = subdivision.frame.y * SUBDIVISION_ARGS;
    let kind = subdivision_kind(id.x, atomicLoad(&subdivision_args[args + SUBDIVISION_LEAVES]));
    listed_kinds[local] = kind;
    workgroupBarrier();
    if local == 0u {
        var whole = 0u;
        var holed = 0u;
        for (var at = 0u; at < SUBDIVISION_GROUP; at++) {
            whole += select(0u, 1u, listed_kinds[at] == 1u);
            holed += select(0u, 1u, listed_kinds[at] == 2u);
        }
        let records = args + SUBDIVISION_LISTS + 1u;
        listed_firsts[0] = atomicAdd(&subdivision_args[records], whole);
        listed_firsts[1] = atomicAdd(&subdivision_args[records + 5u], holed);
    }
    workgroupBarrier();
    if kind == 0u {
        return;
    }
    var before = 0u;
    for (var at = 0u; at < local; at++) {
        before += select(0u, 1u, listed_kinds[at] == kind);
    }
    let place = subdivision.tree.z + listed_firsts[kind - 1u] + before;
    subdivision_lists[place * SURFACE_LISTS + kind - 1u] = id.x;
}
