// The ground as a terrain draws it (`meshes/terrain_vertex.wgsl`), for
// what stands on it: ground cover's blades (`cover_grow.wgsl`). Prepended
// to each, which declares the terrains' `directory` and `heights`.

const NOT_RESIDENT: u32 = 0xffffffffu;

// A terrain as its ground reads: where its levels start in `directory` and
// how many there are, its level-0 sample spacing, and the meters from the
// eye up to which each of its first eight levels draws, finest first: each
// morphs into the next coarser level over the last quarter of its range; 0
// for levels that never morph.
struct GroundTerrain {
    directory: u32,
    levels: u32,
    spacing: f32,
    ranges: array<vec4<f32>, 2>,
}

// A tile of a level: its layer (`NOT_RESIDENT` when not loaded) and its
// first sample (the level's samples from the terrain's origin).
struct LevelTile {
    layer: u32,
    first: vec2<u32>,
}

// The tile of `terrain`'s `level` holding its sample `samples`
// (terrain-local).
fn level_tile(terrain: GroundTerrain, level: u32, samples: vec2<f32>) -> LevelTile {
    let header = terrain.directory + level * 4u;
    let tiles = vec2<u32>(directory[header + 1u], directory[header + 2u]);
    let tile = min(vec2<u32>(max(samples, vec2<f32>(0.0)) / 256.0), tiles - vec2<u32>(1u));
    var found: LevelTile;
    found.layer = directory[directory[header] + tile.y * tiles.x + tile.x];
    found.first = tile * 256u;
    return found;
}

// The height of `tile`'s sample `sample` (its level's, terrain-local);
// layers keep a border sample before the tile's first.
fn tile_height(tile: LevelTile, sample: vec2<u32>) -> f32 {
    let texel = min(sample - tile.first, vec2<u32>(256u)) + vec2<u32>(1u);
    return textureLoad(heights, texel, tile.layer, 0).r;
}

// Where `terrain`'s `level` stands morphing into the next coarser level
// `away` meters from the eye: over the last quarter of its range.
fn level_morph(terrain: GroundTerrain, level: u32, away: f32) -> f32 {
    if level >= 8u {
        return 0.0;
    }
    let four = select(terrain.ranges[0], terrain.ranges[1], level >= 4u);
    let range = dot(four, vec4<f32>(vec4<u32>(level % 4u) == vec4<u32>(0u, 1u, 2u, 3u)));
    if range <= 0.0 {
        return 0.0;
    }
    return saturate((away - 0.75 * range) / (0.25 * range));
}

// The height within a cell of corners `a` (0, 0), `b` (1, 0), `c` (0, 1)
// and `d` (1, 1) at `f`, on the two triangles patches split cells into,
// from the (0, 1) corner to the (1, 0) one.
fn triangle_height(a: f32, b: f32, c: f32, d: f32, f: vec2<f32>) -> f32 {
    if f.x + f.y <= 1.0 {
        return a + (b - a) * f.x + (c - a) * f.y;
    }
    return d + (c - d) * (1.0 - f.x) + (b - d) * (1.0 - f.y);
}

// The height of `terrain`'s ground at terrain-local `local` as it draws it
// `away` meters from the eye, where its finest resident level is `finest`
// (its tile `finest_tile`): that level, or the coarser one it has morphed
// into, morphing into the next over the last quarter of its range, its odd
// samples sliding onto the even ones before them as their heights blend
// into the coarser level's, read on the triangles its patches draw.
fn drawn_height(terrain: GroundTerrain, local: vec2<f32>, away: f32, finest: u32,
        finest_tile: LevelTile) -> f32 {
    let levels = terrain.levels;
    var level = finest;
    var tile = finest_tile;
    // A level fully morphed draws as the next coarser one.
    var morph = level_morph(terrain, level, away);
    var coarse: LevelTile;
    coarse.layer = NOT_RESIDENT;
    while morph > 0.0 && level + 1u < levels {
        coarse = level_tile(terrain, level + 1u,
            local / (terrain.spacing * f32(2u << level)));
        if coarse.layer == NOT_RESIDENT {
            morph = 0.0;
            break;
        }
        if morph < 1.0 {
            break;
        }
        level += 1u;
        tile = coarse;
        coarse.layer = NOT_RESIDENT;
        morph = level_morph(terrain, level, away);
    }
    if level + 1u >= levels {
        morph = 0.0;
    }
    // The morphed grid: within each pair of cells from an even sample, the
    // odd sample between them stands `1 - morph` along.
    let samples = max(local / (terrain.spacing * f32(1u << level)), vec2<f32>(0.0));
    let block = floor(samples * 0.5) * 2.0;
    let within = samples - block;
    let edge = 1.0 - morph;
    let second = within >= vec2<f32>(edge);
    let start = select(vec2<f32>(0.0), vec2<f32>(edge), second);
    let width = select(vec2<f32>(edge), vec2<f32>(1.0 + morph), second);
    let f = saturate((within - start) / max(width, vec2<f32>(1.0e-6)));
    let cell = vec2<u32>(block) + select(vec2<u32>(0u), vec2<u32>(1u), second);
    var a = tile_height(tile, cell);
    var b = tile_height(tile, cell + vec2<u32>(1u, 0u));
    var c = tile_height(tile, cell + vec2<u32>(0u, 1u));
    var d = tile_height(tile, cell + vec2<u32>(1u, 1u));
    if morph > 0.0 {
        // Each corner blends into the coarser level's sample at the even
        // sample it slides onto: the block's first, or after the block for
        // the second cell's far corners.
        let pair = vec2<u32>(block) / 2u;
        let next = pair + select(vec2<u32>(0u), vec2<u32>(1u), second);
        a = mix(a, tile_height(coarse, pair), morph);
        b = mix(b, tile_height(coarse, vec2<u32>(next.x, pair.y)), morph);
        c = mix(c, tile_height(coarse, vec2<u32>(pair.x, next.y)), morph);
        d = mix(d, tile_height(coarse, next), morph);
    }
    return triangle_height(a, b, c, d, f);
}
