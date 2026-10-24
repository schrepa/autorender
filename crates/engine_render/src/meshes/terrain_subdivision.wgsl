// Terrain's subdivision, after `meshes/terrain.wgsl`: each frame, every
// terrain the camera draws splits and merges the leaves of a concurrent
// binary tree over its square (Dupuy et al., "Concurrent Binary Trees (with
// application to longest edge bisection)", HPG 2020) by how far each leaf's
// flat triangle strays from the true surface in pixels seen from the detail
// eye, and lists the leaves the frame draws (`terrain_surface.wgsl`).
//
// The tree is the paper's bitfield, a bit for each node of its deepest
// level, set at the first of each leaf's, under the counts of its levels
// down to SUBDIVISION_COUNTED above the deepest, each node's leaves; the
// levels between are counted from the bitfield's words. Its leaves are the
// triangles of longest edge bisection over the square's two halves, and a
// split also splits every triangle across a hypotenuse whose split it needs
// (the paper's neighbor rule), so the triangulation stays conforming,
// without cracks or T-junctions.
//
// A frame takes three dispatches per terrain, every decision read from the
// tree as the frame found it:
// - begin: the frame's arguments from the leaves the tree holds, which the
//   frame copies to the buffer the dispatches after it and the draws take
//   them from (no dispatch takes its arguments from a buffer it writes);
// - decide: each leaf found by the counts, listed for the frame's draws
//   (collapsed where the terrain draws nothing), and marked to split or, on
//   the frames merging (every other one, so splits and merges never meet),
//   its diamond to merge;
// - apply: the marks set into the bitfield, every count over a bit changing
//   by one as it changes, so the counts always hold the bitfield's sums
//   without a pass summing them.
// A leaf thus changes a level every other frame at most, as the paper's do.
// Splits stop where a terrain's leaves would outnumber its room, counted.

// A terrain's subdivision: the side of its square (meters, terrain-local
// from its origin) and the pixels a leaf's flat triangle may stray from the
// true surface seen from the detail eye; its tree's deepest level, where its
// bitfield starts in `subdivision_tree` (words, after the counts), where its
// leaves start in `subdivision_leaves` and `subdivision_nodes` and how many
// it holds at most; 1 on frames merging leaves (0: splitting) and its place
// in `subdivision_args`; the ground it draws (terrain-local, least corner
// then greatest) and the ground surroundings leave to the terrain they
// surround (all 0 without); and the camera's frustum, its planes inward in
// the world.
struct TerrainSubdivision {
    square: vec4<f32>,
    tree: vec4<u32>,
    frame: vec4<u32>,
    extent: vec4<f32>,
    cutout: vec4<f32>,
    planes: array<vec4<f32>, 6>,
}
@group(2) @binding(27) var<uniform> subdivision: TerrainSubdivision;
// The terrain's tree: its counts by node (words 1 on), then its bitfield.
@group(2) @binding(28) var<storage, read_write> subdivision_tree: array<atomic<u32>>;
// Each leaf's node this frame and what it decided (the top two bits).
@group(2) @binding(29) var<storage, read_write> subdivision_nodes: array<u32>;
// The leaves the frame draws (`terrain_surface_leaves`).
@group(2) @binding(30) var<storage, read_write> subdivision_leaves: array<SurfaceTriangle>;
// Every terrain's SUBDIVISION_ARGS words: its dispatches' arguments, its
// leaves as the frame found them and its draw's arguments, as begin writes
// them, then the leaves its splits reserved and the splits its room refused.
@group(2) @binding(31) var<storage, read_write> subdivision_args: array<atomic<u32>>;

// Lanes of the passes over leaves, and the words each terrain takes of
// `subdivision_args`.
const SUBDIVISION_GROUP: u32 = 64u;
const SUBDIVISION_ARGS: u32 = 16u;
// The deepest levels the tree keeps no counts of: their nodes span at most
// four of the bitfield's words, which count them.
const SUBDIVISION_COUNTED: u32 = 8u;
// How many times the target a leaf the camera does not see may stray: it
// still casts into shadow maps and shows in reflection probe faces, whose
// texels span several of the camera's pixels, and stands a few levels short
// of what the view needs (error grows as a leaf's square), which it reaches
// within frames as the view turns to it.
const SUBDIVISION_UNSEEN: f32 = 8.0;
// What a leaf decided, in the top bits of its word in `subdivision_nodes`.
const SUBDIVISION_SPLIT: u32 = 1u;
const SUBDIVISION_MERGE: u32 = 2u;

// The first bit of node `node`'s range at the tree's deepest level.
fn subdivision_bit(node: u32) -> u32 {
    let deepest = subdivision.tree.x;
    return (node << (deepest - firstLeadingBit(node))) - (1u << deepest);
}

// The leaves within node `node`'s range: its count where the tree keeps it,
// or the bits set in its range of the bitfield.
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

// The node of leaf `index`, counting leaves in the tree's order: down from
// the root by the counts to the first node holding a single leaf.
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

// Node `node`'s triangle, terrain-local: the square's half its first bit
// below the root picks (the first from the far corner to the origin, its
// right angle at x 0, the second back, its right angle at z 0), halved by
// each bit after it (`surface_half`).
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

// The node across node `node`'s hypotenuse at its own depth, 0 past the
// square's edge: as every node of that depth splits the square, from each
// half's three neighbors (across its hypotenuse, its edge from `b` to `c`
// and its edge from `c` back to `a`) to its halves': a half's hypotenuse
// meets the half of its parent's neighbor across that side, its side along
// the parent's hypotenuse the half of the neighbor there, and its side
// from the middle its sibling.
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

// A point of a leaf as the ground draws it: on the true surface (world),
// displaced where deep materials displace it, and whether any level holds
// ground there.
struct SubdivisionPoint {
    world: vec3<f32>,
    found: bool,
}

fn subdivision_point(local: vec2<f32>) -> SubdivisionPoint {
    let held = terrain_level_height(local, 0u);
    var point: SubdivisionPoint;
    point.found = held.found;
    point.world = terrain.origin.xyz + vec3<f32>(local.x, held.height, local.y);
    point.world += terrain_displacement(point.world, local);
    return point;
}

// Whether any of the box from `low` to `high` (world) lies inside the
// camera's frustum.
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

// How a leaf stands: whether the terrain draws it (within its edges, not
// wholly within its cutout, over ground some level holds), and whether it
// wants splitting: across those edges, which its leaves split to follow, or
// where the meshlet it draws strays from the true surface by more pixels
// than the target where the camera sees it, SUBDIVISION_UNSEEN times as
// many where it does not.
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
    let a = subdivision_point(triangle.a);
    let b = subdivision_point(triangle.b);
    let c = subdivision_point(triangle.c);
    let center = subdivision_point((triangle.a + triangle.b + triangle.c) / 3.0);
    measure.drawn = a.found || b.found || c.found || center.found;
    if !measure.drawn {
        return measure;
    }
    if any(low < extent.xy) || any(high > extent.zw)
        || (all(high > cutout.xy) && all(low < cutout.zw)) {
        measure.splits = true;
        return measure;
    }
    // How far its sides' middles and its center stray from the plane
    // through its corners: the surface's curvature over it, none over flat
    // ground however near, each in the pixels it spans seen from the detail
    // eye; and the error the leaf draws, its meshlet's pieces a
    // 2^SURFACE_MESHLET_LEVEL-th of its sides, over which a smoothly
    // curving surface strays as the square of that less.
    let ab = subdivision_point((triangle.a + triangle.b) * 0.5);
    let bc = subdivision_point((triangle.b + triangle.c) * 0.5);
    let ca = subdivision_point((triangle.c + triangle.a) * 0.5);
    let normal = cross(b.world - a.world, c.world - a.world);
    let unit = normal / max(length(normal), 1.0e-12);
    let strays = abs(vec4<f32>(dot(ab.world - a.world, unit), dot(bc.world - a.world, unit),
        dot(ca.world - a.world, unit), dot(center.world - a.world, unit)));
    let across = max(vec4<f32>(detail_across(ab.world), detail_across(bc.world),
        detail_across(ca.world), detail_across(center.world)), vec4<f32>(1.0e-6));
    let parts = f32(1u << SURFACE_MESHLET_LEVEL);
    let pixels = strays / across / (parts * parts);
    // Seen where its points' box, widened by how far it strays, reaches
    // into the camera's frustum.
    let stray = vec3<f32>(max(max(strays.x, strays.y), max(strays.z, strays.w)));
    let lowest = min(min(min(a.world, b.world), min(c.world, center.world)),
        min(min(ab.world, bc.world), ca.world)) - stray;
    let highest = max(max(max(a.world, b.world), max(c.world, center.world)),
        max(max(ab.world, bc.world), ca.world)) + stray;
    let allowed = subdivision.square.y
        * select(SUBDIVISION_UNSEEN, 1.0, subdivision_seen(lowest, highest));
    measure.splits = max(max(pixels.x, pixels.y), max(pixels.z, pixels.w)) > allowed;
    return measure;
}

// Whether the diamond of node `base`, both of whose halves are leaves, and
// the node across its hypotenuse merges this frame: decided once, by the
// first half of the lower-numbered of the two, where every half of both is
// a leaf and neither, a leaf itself, would split.
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

// Adds a leaf to (or takes one from) every count over the bitfield's bit
// `bit`.
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

// Sets the bit of node `node`, a second half, which splits its parent into
// its two halves, and adds the leaf that makes to every count over it;
// returns whether the bit was clear. With `apply` false it only says whether
// the bit is clear.
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

// Clears the bit of node `node`, a second half, which merges its parent's
// halves into it, and takes the leaf that loses from every count over it,
// where the bit is set.
fn subdivision_clear(node: u32) {
    let bit = subdivision_bit(node);
    let mask = 1u << (bit % 32u);
    if (atomicAnd(&subdivision_tree[subdivision.tree.y + bit / 32u], ~mask) & mask) != 0u {
        subdivision_tally(bit, false);
    }
}

// Splits leaf `node` and, by longest edge bisection's rule, every node
// across a hypotenuse whose split that needs: the node across its own at
// its depth, which is a leaf or whose parent is (split first), and on across
// that parent's toward the root. It stops at the first node already split:
// in the tree as the frame found it every node past a split one is split
// too, and a node this frame split continues its own chain. Returns how many
// it split, or with `apply` false how many it would.
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

// The frame's indirect arguments of the terrain, from the leaves its tree
// holds as the frame found it: the decisions' and changes' dispatches, a
// lane a leaf; its draws, a meshlet a leaf (`terrain::TerrainPass::surface_draw`);
// and its room's counters cleared.
@compute @workgroup_size(1)
fn subdivide_begin() {
    let args = subdivision.frame.y * SUBDIVISION_ARGS;
    let leaves = atomicLoad(&subdivision_tree[1]);
    atomicStore(&subdivision_args[args], (leaves + SUBDIVISION_GROUP - 1u) / SUBDIVISION_GROUP);
    atomicStore(&subdivision_args[args + 1u], 1u);
    atomicStore(&subdivision_args[args + 2u], 1u);
    atomicStore(&subdivision_args[args + 3u], leaves);
    atomicStore(&subdivision_args[args + 4u], SURFACE_MESHLET_VERTICES);
    atomicStore(&subdivision_args[args + 5u], leaves);
    atomicStore(&subdivision_args[args + 6u], 0u);
    atomicStore(&subdivision_args[args + 7u], 0u);
    atomicStore(&subdivision_args[args + 8u], 0u);
    atomicStore(&subdivision_args[args + 9u], 0u);
}

// Each leaf: listed for the frame's draws, all its corners on one point
// where the terrain draws nothing; and marked to split where it strays too
// far (on frames splitting) or, as the first half of its diamond, to merge
// where neither half's parent would split (on frames merging).
@compute @workgroup_size(SUBDIVISION_GROUP)
fn subdivide_decide(@builtin(global_invocation_id) id: vec3<u32>) {
    let args = subdivision.frame.y * SUBDIVISION_ARGS;
    if id.x >= atomicLoad(&subdivision_args[args + 3u]) {
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

// Each leaf's decision set into the tree: a split with its chain where the
// terrain's room holds every leaf the chain adds (none of a chain splits
// otherwise, since half of one would crack the ground; counted), or its
// diamond merged.
@compute @workgroup_size(SUBDIVISION_GROUP)
fn subdivide_apply(@builtin(global_invocation_id) id: vec3<u32>) {
    let args = subdivision.frame.y * SUBDIVISION_ARGS;
    let leaves = atomicLoad(&subdivision_args[args + 3u]);
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
            if atomicAdd(&subdivision_args[args + 8u], adds) + adds > room {
                atomicAdd(&subdivision_args[args + 9u], 1u);
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
