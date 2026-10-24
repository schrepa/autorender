// Terrain's surface as its subdivision draws it, after
// `terrain_vertex.wgsl`: the leaves each terrain's subdivision lists this
// frame (`meshes/terrain_subdivision.wgsl`), triangles of longest edge
// bisection over the terrain's square, each drawn as a meshlet bisecting it
// evenly further; and the ground's true height, which the leaves draw within
// about a pixel and everything standing on the ground reads.

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
// bisections, and the vertices it draws: three for each of its four to that
// power pieces, a non-indexed list.
const SURFACE_MESHLET_LEVEL: u32 = 2u;
const SURFACE_MESHLET_VERTICES: u32 = 48u;

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

// The terrain-local x and z of vertex `vertex` of a leaf's meshlet: corner
// `vertex % 3` of piece `vertex / 3`, the piece its bits pick, the first
// from the leaf's first bisection on. Every side of the leaf is cut into the
// same even parts whatever its neighbor's depth, so meshlets of leaves
// sharing a side share its points.
fn surface_vertex(triangle: SurfaceTriangle, vertex: u32) -> vec2<f32> {
    let piece = vertex / 3u;
    var part = triangle;
    for (var bit = 2u * SURFACE_MESHLET_LEVEL; bit > 0u; bit--) {
        part = surface_half(part, (piece >> (bit - 1u)) & 1u);
    }
    let corner = vertex % 3u;
    return select(select(part.c, part.b, corner == 1u), part.a, corner == 0u);
}

// The height above the terrain's origin of its ground at terrain-local
// `local` on the triangles of `level`'s tiles, unmorphed, or of the nearest
// coarser level's where its tile is not resident, as that level's patches
// draw it whole; `found` false where no level holds it. Level 0's is the
// surface the ground's materials lie on, wherever the geometry drawn
// stands: the virtual texture's pages bake them there, and pixels
// computing their ground read them there too.
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
