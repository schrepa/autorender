// Terrain's surface as its subdivision draws it, after
// `terrain_vertex.wgsl`: the leaves each terrain's subdivision lists this
// frame (`meshes/terrain_subdivision.wgsl`), triangles of longest edge
// bisection over the terrain's square, each drawn as a meshlet bisecting it
// evenly further, indexed over the lattice of its pieces' corners; the
// ground's true height, which the leaves draw within about a pixel and
// everything standing on the ground reads; and how far a leaf's meshlet
// strays from the ground or the water's surface, read from the error
// pyramids of the resident tiles under it.

// Each terrain's levels, finest first, from `terrain.directory.x`: four
// words each (where its tiles' entries start, its tiles along x and z, and
// the meters from the eye its tiles load within, as bits); then the entries,
// row by row, each the layer of a resident tile or all ones.
@group(1) @binding(23) var<storage, read> terrain_directory: array<u32>;

// A leaf: its corners in terrain-local meters (x and z from
// `terrain.origin`), its longest edge from `a` to `b` and its right angle at
// `c`, turning clockwise in x and z so that it faces up. Leaves the terrain
// does not draw (past its edges, within its cutout or over ground no level
// holds) have every corner at one point.
struct SurfaceTriangle {
    a: vec2<f32>,
    b: vec2<f32>,
    c: vec2<f32>,
}

// This frame's leaves of every terrain's subdivision, each terrain's from
// `terrain.surface.x`, in the order its tree holds them.
@group(1) @binding(36) var<storage, read> terrain_surface_leaves: array<SurfaceTriangle>;

// The halvings of a leaf's sides its meshlet cuts it by, each two
// bisections; the points of its lattice, where its pieces' corners stand
// (`surface_vertex`); and the indices its draw takes, three for each of its
// four to that power pieces, each naming the point its corner stands at
// (`terrain/surface.rs`'s `meshlet_indices`, which number the points as
// `surface_vertex` does).
const SURFACE_MESHLET_LEVEL: u32 = 2u;
const SURFACE_MESHLET_VERTICES: u32 = 15u;
const SURFACE_MESHLET_INDICES: u32 = 48u;

// Leaf `leaf`, the instance index of the surface's indirect draws, of this
// frame's subdivision of the terrain whose uniform is bound.
fn surface_leaf(leaf: u32) -> SurfaceTriangle {
    return terrain_surface_leaves[terrain.surface.x + leaf];
}

// Half `half` of `triangle`, split at its longest edge's middle: the first
// from its right angle to the edge's first end, the second from the edge's
// last end to its right angle, each with its right angle at the middle and
// turning as `triangle` does. Corners on a grid of halvings stay exact, so
// neighbors meet on the same points to the bit.
fn surface_half(triangle: SurfaceTriangle, half: u32) -> SurfaceTriangle {
    let middle = (triangle.a + triangle.b) * 0.5;
    if half == 0u {
        return SurfaceTriangle(triangle.c, triangle.a, middle);
    }
    return SurfaceTriangle(triangle.b, triangle.c, middle);
}

// The point `steps` of the 2^SURFACE_MESHLET_LEVEL parts of a leaf's side
// from its end `tail` toward its end `head`, halving the side toward it as
// the meshlet's bisections halve it: the same to the bit counted from either
// end, so leaves sharing the side meet on its points whatever their depths.
fn surface_along(tail: vec2<f32>, head: vec2<f32>, steps: u32) -> vec2<f32> {
    var low = tail;
    var high = head;
    for (var bit = SURFACE_MESHLET_LEVEL; bit > 0u; bit--) {
        let middle = (low + high) * 0.5;
        if ((steps >> (bit - 1u)) & 1u) == 0u {
            high = middle;
        } else {
            low = middle;
        }
    }
    return select(low, head, steps == (1u << SURFACE_MESHLET_LEVEL));
}

// The terrain-local x and z of lattice vertex `vertex` of a leaf's meshlet:
// the point `i` of its legs' 2^SURFACE_MESHLET_LEVEL parts from its right
// angle toward its longest edge's first end and `j` toward its last. The
// lattice's points are numbered in rows of `i + j` from the right angle out,
// each a point longer than the one before, and by `j` along each, as the
// meshlet's indices name its pieces' corners (`terrain/surface.rs`'s
// `meshlet_indices`); the longest edge's are the last. Every side of the
// leaf is cut into the same even parts whatever its neighbor's depth, each
// of its points halved out of that side alone (`surface_along`), so
// meshlets of leaves sharing a side share its points; a point inside stands
// where the leaf's corners weigh it.
fn surface_vertex(triangle: SurfaceTriangle, vertex: u32) -> vec2<f32> {
    let parts = 1u << SURFACE_MESHLET_LEVEL;
    // Its row and the row's first point, from the longest edge's in.
    var row = parts;
    var first = SURFACE_MESHLET_VERTICES - parts - 1u;
    while vertex < first {
        first -= row;
        row--;
    }
    let j = vertex - first;
    let i = row - j;
    // The side it lies on, the longest edge from its first end or a leg
    // from the right angle; or inside, where the corners weigh it.
    var tail = triangle.c;
    var head = triangle.b;
    var steps = j;
    if row == parts {
        tail = triangle.a;
    } else if j == 0u {
        head = triangle.a;
        steps = i;
    } else if i != 0u {
        let share = vec2<f32>(f32(i), f32(j)) / f32(parts);
        return triangle.c + (triangle.a - triangle.c) * share.x
            + (triangle.b - triangle.c) * share.y;
    }
    return surface_along(tail, head, steps);
}

// The height above the terrain's origin of its ground at terrain-local
// `local` on the triangles of `level`'s tiles, unmorphed, or of the nearest
// coarser level's where its tile is not resident, as that level's patches
// draw it whole; `found` false where no level holds it. Level 0's is the
// surface the ground's materials lie on, wherever the geometry drawn
// stands: pixels computing their ground read them there.
struct TerrainLevelHeight {
    height: f32,
    found: bool,
}

fn terrain_level_height(local: vec2<f32>, level: u32) -> TerrainLevelHeight {
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
    var held: TerrainLevelHeight;
    for (var index = level; index < levels; index++) {
        let header = base + index * 4u;
        let samples = max(local / (terrain.anchor.w * f32(1u << index)), vec2<f32>(0.0));
        let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
        let tile = min(vec2<u32>(samples / 256.0), tiles - vec2<u32>(1u));
        let layer = terrain_directory[terrain_directory[header] + tile.y * tiles.x + tile.x];
        if layer == 0xffffffffu {
            continue;
        }
        // Samples 0 to 256 within the tile, a texel past its border; the
        // cell's triangles split from its (x, z+1) corner to its (x+1, z)
        // one, as every consumer splits them.
        let within = clamp(samples - vec2<f32>(tile * 256u), vec2<f32>(0.0), vec2<f32>(256.0));
        let corner = min(vec2<u32>(within), vec2<u32>(255u));
        let f = within - vec2<f32>(corner);
        let texel = corner + vec2<u32>(1u);
        let h00 = terrain_load(layer, texel);
        let h10 = terrain_load(layer, texel + vec2<u32>(1u, 0u));
        let h01 = terrain_load(layer, texel + vec2<u32>(0u, 1u));
        let h11 = terrain_load(layer, texel + vec2<u32>(1u, 1u));
        held.found = true;
        if f.x + f.y <= 1.0 {
            held.height = h00 + f.x * (h10 - h00) + f.y * (h01 - h00);
        } else {
            held.height = h11 + (1.0 - f.x) * (h01 - h11) + (1.0 - f.y) * (h10 - h11);
        }
        return held;
    }
    return held;
}

// The ground's height above the terrain's origin at terrain-local `local`:
// on level 0's triangles, or the finest resident coarser level's where its
// tile is not resident. The subdivision draws it within about a pixel, and
// water, blades and placed objects stand on it.
fn terrain_true_height(local: vec2<f32>) -> f32 {
    return terrain_level_height(local, 0u).height;
}

// Terrain-local `local` at `height` above the origin, relative to the
// terrain's eye: counted from its anchor, a sample of the coarsest level's
// grid near the eye, so points near the eye keep their precision however far
// the terrain reaches.
fn surface_relative(local: vec2<f32>, height: f32) -> vec3<f32> {
    let top = terrain.anchor.w * f32(1u << (terrain.directory.y - 1u));
    let from_anchor = local - vec2<f32>(bitcast<vec2<i32>>(terrain.surface.zw)) * top;
    return vec3<f32>(from_anchor.x + terrain.anchor.x, height + terrain.anchor.y,
        from_anchor.y + terrain.anchor.z);
}

// Each resident tile's error pyramids, at its layer of the height array
// (`engine_terrain::ErrorPyramid`): mip k - 1 holds a texel for each cell of
// 2^k of its samples a side (128 x 128 cells of 2 samples down to one of
// 256), the most a triangle of longest edge bisection lying in the cell, its
// corners on the surface, strays from the surface the tile's samples stand
// at (meters, rounded up): its ground's (`terrain_errors`) and its water's
// surface's (`water_errors`), which holds -1 over cells where no sample is
// wet and 65504 over those where one is and another holds no surface (a
// shore wider than the surface continues under its banks). Only the
// subdivision's passes read them.
@group(1) @binding(62) var terrain_errors: texture_2d_array<f32>;
@group(1) @binding(63) var water_errors: texture_2d_array<f32>;
// The pyramids' mips (`gpu_layouts::constants::ERROR_MIPS`).
const SURFACE_ERROR_MIPS: u32 = 8u;

// How a leaf's meshlet stands over a surface, as the error pyramids of the
// tiles under it bound it: whether a wet sample lies under the leaf or
// within one of its pieces of it (`holds`, over the water's surface), and
// the most its pieces stray from the surface's samples (meters).
struct SurfaceWater {
    holds: bool,
    error: f32,
}

// The finest level, from level `first` on, holding a resident tile over
// block `block` (counted along x and z in level `first`'s tiles), and that
// tile's layer; the layer 0xffffffff where none does.
fn surface_tile(block: vec2<u32>, first: u32) -> vec2<u32> {
    let base = terrain.directory.x;
    for (var index = first; index < terrain.directory.y; index++) {
        let header = base + index * 4u;
        let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
        let tile = block >> vec2<u32>(min(index - first, 31u));
        if any(tile >= tiles) {
            continue;
        }
        let layer = terrain_directory[terrain_directory[header] + tile.y * tiles.x + tile.x];
        if layer != 0xffffffffu {
            return vec2<u32>(index, layer);
        }
    }
    return vec2<u32>(0u, 0xffffffffu);
}

// Whether the square `half` from `middle` each way reaches over the inner
// side of the edge from `tail` to `head` of a triangle turning as `turn`'s
// sign says: its corner farthest into the triangle lies past the edge.
fn surface_inside(tail: vec2<f32>, head: vec2<f32>, turn: f32, middle: vec2<f32>,
        half: f32) -> bool {
    let normal = turn * vec2<f32>(tail.y - head.y, head.x - tail.x);
    return dot(normal, middle - tail) + half * (abs(normal.x) + abs(normal.y)) > 0.0;
}

// Whether the square of a piece whose least corner is `corner`, widened by
// `widen` pieces each way, overlaps the triangle from `a` to `b` to `c` over
// an area (all in pieces, so exactly): neither a side of the triangle's
// bounds nor one of its edges leaves the square wholly outside it.
fn surface_overlaps(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, corner: vec2<f32>,
        widen: f32) -> bool {
    let low = corner - vec2<f32>(widen);
    let high = corner + vec2<f32>(1.0 + widen);
    if any(high <= min(min(a, b), c)) || any(low >= max(max(a, b), c)) {
        return false;
    }
    let middle = (low + high) * 0.5;
    let half = 0.5 + widen;
    let turn = sign((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
    return surface_inside(a, b, turn, middle, half) && surface_inside(b, c, turn, middle, half)
        && surface_inside(c, a, turn, middle, half);
}

// How a leaf's meshlet stands over a surface whose tiles' error pyramids are
// `errors`. Its pieces each lie in a dyadic square a 2^SURFACE_MESHLET_LEVEL-th
// of the leaf's own a side (halves of it at odd depths, quarters cut by both
// diagonals at even ones), so the most of the pyramids' cells of that size
// the leaf covers bounds them: each read from the finest resident tile
// holding it, at the mip of its size there, adding nothing where it is no
// wider than that tile's samples, which splitting further draws no better.
// Cells within a piece of the leaf count toward `holds` too where `around`
// says. The directory is walked once for each tile of the first level whose
// tiles hold a cell whole that the cells fall in.
fn surface_cells(triangle: SurfaceTriangle, errors: texture_2d_array<f32>, around: bool)
        -> SurfaceWater {
    var found = SurfaceWater(false, 0.0);
    let low = min(min(triangle.a, triangle.b), triangle.c);
    let high = max(max(triangle.a, triangle.b), triangle.c);
    // The leaf's dyadic square, as wide as the longer side of its bounds (its
    // legs at odd depths, its hypotenuse at even ones), and its pieces'.
    let side = max(high.x - low.x, high.y - low.y);
    let parts = i32(1u << SURFACE_MESHLET_LEVEL);
    let piece = side / f32(parts);
    let origin = floor(low / side) * side;
    // The leaf in pieces from its square's least corner, and its square's
    // first cell counted from the terrain's origin, exact powers of two
    // apart throughout.
    let a = (triangle.a - origin) / piece;
    let b = (triangle.b - origin) / piece;
    let c = (triangle.c - origin) / piece;
    let first = vec2<i32>(origin / piece);
    // Octaves from a level-0 tile's side to a piece's, the first level whose
    // tiles hold a cell whole, and the cells along a side of one of those
    // tiles as a power of two.
    let octaves = i32(round(log2(piece / (256.0 * terrain.anchor.w))));
    let level = u32(max(octaves, 0));
    let block = u32(i32(level) - octaves);
    let reach = select(0, 1, around);
    let cells_low = first - vec2<i32>(reach);
    let cells_high = first + vec2<i32>(parts - 1 + reach);
    let blocks_low = cells_low >> vec2<u32>(block);
    let blocks_high = cells_high >> vec2<u32>(block);
    for (var bz = blocks_low.y; bz <= blocks_high.y; bz++) {
        for (var bx = blocks_low.x; bx <= blocks_high.x; bx++) {
            if bx < 0 || bz < 0 {
                continue;
            }
            let held = surface_tile(vec2<u32>(vec2<i32>(bx, bz)), level);
            if held.y == 0xffffffffu {
                continue;
            }
            // Cells along a side of the tile holding them, as a power of
            // two: the mip holding them, and past its finest the octaves by
            // which they are narrower than its cells, and than the tile's
            // samples.
            let fit = u32(i32(held.x) - octaves);
            let mip = SURFACE_ERROR_MIPS - 1u - min(fit, SURFACE_ERROR_MIPS - 1u);
            let narrow = min(fit - min(fit, SURFACE_ERROR_MIPS - 1u), 31u);
            let span = min(fit, 31u);
            let start = max(cells_low, vec2<i32>(bx, bz) << vec2<u32>(block));
            let end = min(cells_high, ((vec2<i32>(bx, bz) + 1) << vec2<u32>(block)) - 1);
            for (var z = start.y; z <= end.y; z++) {
                for (var x = start.x; x <= end.x; x++) {
                    let corner = vec2<f32>(vec2<i32>(x, z) - first);
                    let over = surface_overlaps(a, b, c, corner, 0.0);
                    if !over && !(around && surface_overlaps(a, b, c, corner, 1.0)) {
                        continue;
                    }
                    let cell = vec2<u32>(vec2<i32>(x, z));
                    let within = cell - ((cell >> vec2<u32>(span)) << vec2<u32>(span));
                    let texel = vec2<i32>(within >> vec2<u32>(narrow));
                    let bound = textureLoad(errors, texel, i32(held.y), i32(mip)).x;
                    found.holds = found.holds || bound >= 0.0;
                    if over && fit < SURFACE_ERROR_MIPS {
                        found.error = max(found.error, bound);
                    }
                }
            }
        }
    }
    return found;
}

// The most the pieces of a leaf's meshlet stray from the ground's samples
// (meters), as its tiles' error pyramids bound it (`surface_cells`).
fn surface_ground_error(triangle: SurfaceTriangle) -> f32 {
    return surface_cells(triangle, terrain_errors, false).error;
}

// Whether a wet sample lies under a leaf or within one of its pieces of it,
// and the most the pieces of its meshlet stray from the water's surface:
// 65504 or more where a cell it covers is a shore wider than the surface
// continues under its banks, which splits it until its pieces lie where
// the surface reaches (`surface_cells`).
fn surface_water(triangle: SurfaceTriangle) -> SurfaceWater {
    return surface_cells(triangle, water_errors, true);
}
