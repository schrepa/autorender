// Terrain horizon maps: one layer per terrain, a grid aligned
// with the sun over the terrain. Each thread sweeps one line of the grid from
// the sun's side: a texel's horizon is the higher of the terrain one texel
// toward the sun and the horizon there, each lowered by the sun's rise over
// that texel, so it ends up as the height above which the sun clears every
// slope toward it. Texels keep that horizon above their own ground, how far
// along the sun the slope setting it lies, and the ground's height, so
// surfaces compare with the same ground whatever detail they draw at.
// Heights come from the terrain's finest pinned level, or from a resident
// ancestor where one of its tiles is missing.

struct Build {
    // World x and z of the grid's center, and the unit direction toward the
    // sun across it (x and z).
    place: vec4<f32>,
    // Half the grid's extent along the sun and across it (meters), the sun's
    // rise per meter toward it, and the terrain's origin height.
    shape: vec4<f32>,
    // The terrain's origin (world x and z) and its source level's sample
    // spacing.
    origin: vec4<f32>,
    // Texels along each side, the source level's tiles along x and z, and
    // the first of its entries in `tile_layers`.
    counts: vec4<u32>,
}

@group(0) @binding(0) var heights: texture_2d_array<f32>;
// Per source tile, row by row: the layer read in the low 16 bits and how
// many levels coarser its tile is in the high 16; all ones where no level
// holds it.
@group(0) @binding(1) var<storage, read> tile_layers: array<u32>;
@group(0) @binding(2) var<uniform> build: Build;
// The terrain's layer: x the horizon above the ground, y its distance along
// the sun, z the ground's world height.
@group(0) @binding(3) var horizon: texture_storage_2d<rgba16float, write>;

const MISSING: u32 = 0xffffffffu;
// Where no terrain is, or nothing toward the sun rises: below any terrain,
// yet finite, so lookups interpolate across it.
const OPEN: f32 = -1.0e6;
// The lowest horizon above the ground maps store, within half floats.
const LOWEST: f32 = -6.0e4;
const TILE_QUADS: f32 = 256.0;

// The terrain's world height at terrain-local `point`, interpolated between
// samples; OPEN outside it or where no level holds its tile.
fn source_height(point: vec2<f32>) -> f32 {
    let samples = point / build.origin.z;
    let tiles = build.counts.yz;
    if any(samples < vec2<f32>(0.0)) || any(samples > vec2<f32>(tiles) * TILE_QUADS) {
        return OPEN;
    }
    let tile = min(vec2<u32>(samples / TILE_QUADS), tiles - vec2<u32>(1u));
    let entry = tile_layers[build.counts.w + tile.y * tiles.x + tile.x];
    if entry == MISSING {
        return OPEN;
    }
    let coarser = entry >> 16u;
    let layer = entry & 0xffffu;
    // Samples of the tile read, which starts at its first sample; layers
    // keep a border sample before it.
    let at = samples / f32(1u << coarser) - vec2<f32>((tile >> vec2<u32>(coarser)) * 256u);
    let base = min(vec2<u32>(at), vec2<u32>(255u));
    let f = at - vec2<f32>(base);
    let texel = base + vec2<u32>(1u);
    let a = textureLoad(heights, texel, layer, 0).r;
    let b = textureLoad(heights, texel + vec2<u32>(1u, 0u), layer, 0).r;
    let c = textureLoad(heights, texel + vec2<u32>(0u, 1u), layer, 0).r;
    let d = textureLoad(heights, texel + vec2<u32>(1u, 1u), layer, 0).r;
    return build.shape.w + mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

@compute @workgroup_size(64)
fn build_horizon(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = build.counts.x;
    let line = id.x;
    if line >= size {
        return;
    }
    let toward = build.place.zw;
    let side = vec2<f32>(-toward.y, toward.x);
    let step = 2.0 * build.shape.x / f32(size);
    let across = ((f32(line) + 0.5) / f32(size) * 2.0 - 1.0) * build.shape.y;
    let rise = build.shape.z;
    // The highest slope toward the sun so far and its distance along the
    // sun, and the terrain one texel toward the sun.
    var crest = OPEN;
    var distance = 0.0;
    var previous = OPEN;
    for (var n = 0u; n < size; n++) {
        let column = size - 1u - n;
        distance += step;
        if previous - rise * step >= crest - rise * distance {
            crest = previous;
            distance = step;
        }
        let along = (f32(column) + 0.5) * step - build.shape.x;
        let world = build.place.xy + toward * along + side * across;
        let ground = source_height(world - build.origin.xy);
        var clearance = LOWEST;
        if crest > OPEN && ground > OPEN {
            clearance = max(crest - rise * distance - ground, LOWEST);
        }
        textureStore(horizon, vec2<u32>(column, line),
            vec4<f32>(clearance, distance, max(ground, LOWEST), 0.0));
        previous = ground;
    }
}
