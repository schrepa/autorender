// Terrain sky visibility (decision 055): one layer per terrain, a grid over
// the whole terrain. Each thread finds, for one texel, the horizon of the
// terrain around its ground in sixteen directions, marching out in steps
// that grow by a third from under a texel to the build's reach, and keeps
// the share of the sky above both those horizons and the ground's own
// plane, cosine-weighted about the ground's normal. Heights come from the
// terrain's finest pinned level, or from a resident ancestor where one of
// its tiles is missing; beyond the terrain its edge carries on. Each texel
// also keeps the ground's large-scale variety (decision 090): how far it
// lies below the ground a couple of texels around (hollows, where water
// gathers) and below or above the ground a couple of hundred meters around
// (ridges).

struct Build {
    // The terrain's extent (meters along x and z), its source level's
    // sample spacing and how far horizons are looked for.
    shape: vec4<f32>,
    // Texels along each side, the source level's tiles along x and z, and
    // the first of its entries in `tile_layers`.
    counts: vec4<u32>,
    // The first row this dispatch builds and how many it builds.
    rows: vec4<u32>,
}

@group(0) @binding(0) var heights: texture_2d_array<f32>;
// Per source tile, row by row: the layer read in the low 16 bits and how
// many levels coarser its tile is in the high 16; all ones where no level
// holds it.
@group(0) @binding(1) var<storage, read> tile_layers: array<u32>;
@group(0) @binding(2) var<uniform> build: Build;
// The terrain's layer: red its ground's sky visibility; green and blue its
// hollows and ridges, 0.5 on level ground.
@group(0) @binding(3) var sky: texture_storage_2d<rgba8unorm, write>;

const MISSING: u32 = 0xffffffffu;
const TILE_SAMPLES: f32 = 256.0;
const DIRECTIONS: u32 = 16u;
const PI: f32 = 3.14159265;
// How much each step outward grows.
const GROWTH: f32 = 1.33;
// Meters around a texel its ridges are measured against, and the meters a
// hollow and a ridge take to reach the ends of their ranges.
const RIDGE_REACH: f32 = 200.0;
const HOLLOW_SCALE: f32 = 8.0;
const RIDGE_SCALE: f32 = 80.0;

// The terrain's height at terrain-local `point`, held within the terrain
// and interpolated between samples; `fallback` where no level holds its
// tile.
fn source_height(point: vec2<f32>, fallback: f32) -> f32 {
    let tiles = build.counts.yz;
    let samples = clamp(point / build.shape.z, vec2<f32>(0.0), vec2<f32>(tiles) * TILE_SAMPLES);
    let tile = min(vec2<u32>(samples / TILE_SAMPLES), tiles - vec2<u32>(1u));
    let entry = tile_layers[build.counts.w + tile.y * tiles.x + tile.x];
    if entry == MISSING {
        return fallback;
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
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// The sky of one direction's slice above elevation `lowest` (radians),
// cosine-weighted about a normal whose part along the direction is
// `along` and whose upward part is `up`: the integral over elevations e
// from `lowest` to the zenith of the normal's cosine times cos e.
fn slice(lowest: f32, along: f32, up: f32) -> f32 {
    let rise = sin(lowest);
    return along * (0.25 * PI - 0.5 * lowest - 0.25 * sin(2.0 * lowest))
        + up * 0.5 * (1.0 - rise * rise);
}

@compute @workgroup_size(8, 8)
fn build_sky(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = build.counts.x;
    if id.x >= size || id.y >= build.rows.y || id.y + build.rows.x >= size {
        return;
    }
    let texel = vec2<u32>(id.x, id.y + build.rows.x);
    let step = build.shape.xy / f32(size);
    let point = (vec2<f32>(texel) + 0.5) * step;
    let ground = source_height(point, 0.0);
    // The ground's normal across a texel either way.
    let across = max(step.x, step.y);
    let normal = normalize(vec3<f32>(
        source_height(point - vec2<f32>(across, 0.0), ground)
            - source_height(point + vec2<f32>(across, 0.0), ground),
        2.0 * across,
        source_height(point - vec2<f32>(0.0, across), ground)
            - source_height(point + vec2<f32>(0.0, across), ground)));
    var seen = 0.0;
    var open = 0.0;
    for (var index = 0u; index < DIRECTIONS; index++) {
        let angle = (f32(index) + 0.5) * (2.0 * PI / f32(DIRECTIONS));
        let direction = vec2<f32>(cos(angle), sin(angle));
        // The ground's own plane toward the direction: where the normal's
        // cosine falls to 0.
        let along = dot(normal.xz, direction);
        let plane = atan(-along / normal.y);
        var highest = -1.0e9;
        var distance = 0.75 * across;
        loop {
            if distance > build.shape.w {
                break;
            }
            let height = source_height(point + direction * distance, ground);
            highest = max(highest, (height - ground) / distance);
            distance *= GROWTH;
        }
        open += slice(plane, along, normal.y);
        seen += slice(max(atan(highest), plane), along, normal.y);
    }
    // The variety: the ground's height against rings around it, two texels
    // and RIDGE_REACH out.
    var near = 0.0;
    var wide = 0.0;
    for (var index = 0u; index < 8u; index++) {
        let angle = (f32(index) + 0.5) * (2.0 * PI / 8.0);
        let direction = vec2<f32>(cos(angle), sin(angle));
        near += source_height(point + direction * (2.0 * across), ground);
        wide += source_height(point + direction * max(RIDGE_REACH, 4.0 * across), ground);
    }
    let hollow = near * 0.125 - ground;
    let ridge = ground - wide * 0.125;
    textureStore(sky, texel, vec4<f32>(saturate(seen / max(open, 1.0e-6)),
        saturate(0.5 + 0.5 * hollow / HOLLOW_SCALE), saturate(0.5 + 0.5 * ridge / RIDGE_SCALE),
        1.0));
}
