// Water: terrains' water as their tiles hold it, drawn on the terrain's own
// patches. Each vertex stands at its water's surface over the ground its
// patch draws and morphs toward the parent level's as the ground does, so
// the depth each pixel shades by is the tiles' own, their surface less
// their ground on the triangles the ground is drawn on, and the shore falls
// where the ground rises through the water. The waves' tiles
// (`water_waves.wgsl`) tilt it where it stands still and run with its flow
// where it moves, carried along in two sets half a cycle apart, and what
// they hide within a pixel roughens it. It mirrors the sky, the sun's disc
// and, through screen-space reflections, the scene; its body scatters the
// light entering it back toward the eye; and what lies under it shows
// through, bent by the waves where the frame copies the scene's color, and
// dimmed channel by channel over the light's path down to it and back, the
// ground darker for being wet. Whitewater gathers where the tiles' water
// carries it and along shallow shores, and runs white down chutes too
// steep for it to flow smoothly. The sea (`meshes/ocean.wgsl`) lights its
// surface the same way. Where rivers leave their beds, their jets fall as
// sheets of their own (`water_falls.rs`), drawn last.

// Tiles' water, a block of samples for each patch holding any
// (`terrain/water_blocks.rs`): their surfaces (terrain-local meters,
// NO_WATER_SURFACE where none reaches), and their flow along x and z as
// shares of WATER_FLOW_RANGE either way, their whitewater, and 1 where a
// lake holds them, else the half width of the river's water as 126ths of
// WATER_WIDTH_RANGE in 127ths (rgba, signed shares).
@group(1) @binding(50) var water_surface: texture_2d_array<f32>;
@group(1) @binding(51) var water_info: texture_2d_array<f32>;

// Water, or the sea: the camera's view and projection without its
// translation (the sea's; terrains' water draws through the terrain's
// view); the color deep water scatters back (rgb) and the meters through
// which half the light passes (w); the ripples' strength (x), how strongly
// the wind gusts (y) and its velocity along x and z (zw); how much foam
// gathers and the meters of
// water shallower than which shores foam, with, for the sea, the
// compression below which crests foam and the waves' summed height; how
// fast the water takes each channel of light, per meter (rgb); and how many
// times a meter each of the waves' tiles repeats (xyz). The sea's parts
// (`meshes/ocean.wgsl`): its wind's heading
// (`origin`, xy: a unit vector), its still level less the eye, its grid's
// finest cell, cells around and rings a cell apart; the rings' growth, the
// meters it reaches, cells around a sector and its waves; each wave's
// vector (xy, radians per meter), height and phase at the eye; and how far
// each wave's points move along it, four to a vector. For terrains' water,
// `ocean` holds the sea's level less the eye and whether it is on, below
// which their water gives way to it.
struct WaterUniform {
    relative_view_projection: mat4x4<f32>,
    origin: vec4<f32>,
    color: vec4<f32>,
    ripples: vec4<f32>,
    foam: vec4<f32>,
    absorption: vec4<f32>,
    waves: vec4<f32>,
    ocean: vec4<f32>,
    ocean_grid: vec4<f32>,
    ocean_waves: array<vec4<f32>, 16>,
    ocean_reach: array<vec4<f32>, 4>,
}
@group(1) @binding(52) var<uniform> water: WaterUniform;
// The waves' moments, a layer a tile, with their levels
// (`water_waves.wgsl`), and the sampler repeating them.
@group(1) @binding(53) var water_waves: texture_2d_array<f32>;
@group(1) @binding(54) var water_wave_sampler: sampler;
// What lies behind the water, copied from the opaque scene before it
// draws: its depth and, in frames drawing the effects' targets, its color.
@group(2) @binding(22) var water_depth: texture_depth_2d;
@group(2) @binding(23) var water_scene: texture_2d<f32>;

// The opaque surface behind the water at `pixel` (its center), less the
// eye, without the jitter of temporally antialiased frames.
fn water_ground(pixel: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(water_depth));
    let depth = textureLoad(water_depth, vec2<i32>(pixel), 0);
    let ndc = vec2<f32>(pixel.x / size.x * 2.0 - 1.0, 1.0 - pixel.y / size.y * 2.0)
        - camera.jitter.xy;
    let point = camera.inverse_view_projection * vec4<f32>(ndc, depth, 1.0);
    return point.xyz / point.w - camera.position.xyz;
}

// Samples along a block's side and blocks along a layer's side, no block,
// and what a sample no water reaches holds (`gpu_layouts::constants`).
const WATER_BLOCK: u32 = 17u;
const WATER_BLOCKS_SIDE: u32 = 16u;
const NO_WATER_BLOCK: u32 = 0xffffffffu;
const NO_WATER_SURFACE: f32 = -1.0e30;
// Meters a second either way the packed flow spans, and the meters of half
// width the packed width of a river's water does
// (`engine_terrain::WATER_FLOW_RANGE`, `WATER_WIDTH_RANGE`).
const WATER_FLOW_RANGE: f32 = 8.0;
const WATER_WIDTH_RANGE: f32 = 63.0;
// How much darker the ground under water looks: of the light it scatters
// back, the surface turns part back down. Water a few centimeters deep
// darkens it as much.
const WATER_WET: f32 = 0.35;
// The waves' tiles (`gpu_layouts::constants::WAVE_TILES`).
const WATER_WAVE_TILES: u32 = 3u;
// Water's index of refraction over air's.
const WATER_IOR: f32 = 1.333;
// Calm water's roughness, and the most its waves' hidden slopes roughen it.
const WATER_SMOOTH: f32 = 0.05;
const WATER_ROUGHEST: f32 = 0.6;
// How much the sun's disc widens its mirrored lobe (alpha): the half
// vectors of its rays span half the disc's angle, about 0.0023 radians.
const WATER_SUN_ALPHA: f32 = 0.0023;
// How strongly water's body scatters light forward (Henyey and
// Greenstein's g), and its phase's mean over the backward half, where light
// entering the surface turns back toward an eye above it.
const WATER_PHASE_G: f32 = 0.4;
const WATER_PHASE_MEAN: f32 = 0.4498;
// How much light foam scatters back, and the share of the sky water in its
// channel sees.
const WATER_FOAM_ALBEDO: f32 = 0.65;
const WATER_FOAM_SKY: f32 = 0.6;
// The lowest the eye's mirrored view dips: slopes turned from the eye would
// mirror what lies below the horizon.
const WATER_HORIZON: f32 = 0.02;
// Meters the bent view runs at most to what it shows: the water hides what
// lies farther.
const WATER_REACH: f32 = 4.0;
// Meters over which water fades in from its shores and around what stands
// in it.
const WATER_SHORE: f32 = 0.1;
// The share of a tile's period between the corners of the grid its random
// offsets hang on (`water_tile`).
const WATER_PATCH: f32 = 0.5;
// Pixels a tile's period spans below which it blurs into its variance.
const WATER_WAVE_PIXELS: f32 = 32.0;
// Meters across the patches of stronger and calmer wind sweeping over
// water.
const WATER_GUST_SIZE: f32 = 40.0;

// A patch holding water: its record as terrain's (`PatchInput`), and its
// water's blocks: its own, its parent tile's over it (NO_WATER_BLOCK
// without) and where it starts in that block (x | z << 16).
struct WaterPatchInput {
    @builtin(vertex_index) vertex: u32,
    @location(0) first: vec2<i32>,
    @location(1) shape: vec4<f32>,
    @location(2) atlas: vec4<u32>,
    @location(3) water: vec4<u32>,
    @location(4) fade: f32,
}

struct WaterOutput {
    @builtin(position) clip_position: vec4<f32>,
    // The point less the eye.
    @location(0) relative: vec3<f32>,
    // Meters of water over the ground, and its flow along x and z.
    @location(1) water: vec3<f32>,
    // The whitewater its flow carries, and the half width of its water
    // (meters; lakes WATER_WIDTH_RANGE).
    @location(2) foam: vec2<f32>,
    // How fast its depth falls along x and z (meters a meter), the ground
    // rising out of it toward its shores.
    @location(3) shore: vec2<f32>,
}

struct WaterTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    @location(1) water: vec3<f32>,
    @location(2) foam: vec2<f32>,
    @location(3) shore: vec2<f32>,
    @location(4) previous: vec3<f32>,
}

// The texel of block `block` holding its sample `sample` (x and y), and its
// layer (z).
fn water_texel(block: u32, sample: vec2<u32>) -> vec3<i32> {
    let blocks = WATER_BLOCKS_SIDE * WATER_BLOCKS_SIDE;
    let index = block % blocks;
    let origin = vec2<u32>(index % WATER_BLOCKS_SIDE, index / WATER_BLOCKS_SIDE) * WATER_BLOCK;
    return vec3<i32>(vec2<i32>(origin + sample), i32(block / blocks));
}

// Meters a sample no water reaches counts as standing dry by: the water
// ends within a few hundredths of the way past the last sample holding it,
// however deep it is there (where it spills over a lip or its field ends).
const WATER_DRY: f32 = 16.0;

// The water of block `block`'s sample `sample` over the ground there,
// `ground`: its surface (x) and depth (y), or where no water reaches it a
// meter under the ground, standing WATER_DRY dry.
fn water_block_sample(block: u32, sample: vec2<u32>, ground: f32) -> vec2<f32> {
    let texel = water_texel(block, sample);
    let surface = textureLoad(water_surface, texel.xy, texel.z, 0).x;
    if surface <= NO_WATER_SURFACE * 0.5 {
        return vec2<f32>(ground - 1.0, -WATER_DRY);
    }
    return vec2<f32>(surface, surface - ground);
}

// The meters of water block `block`'s sample `sample` holds over ground
// `ground` (x), and 1 where any water reaches it, else 0 (y).
fn water_block_depth(block: u32, sample: vec2<u32>, ground: f32) -> vec2<f32> {
    let texel = water_texel(block, sample);
    let surface = textureLoad(water_surface, texel.xy, texel.z, 0).x;
    return vec2<f32>(surface - ground, select(1.0, 0.0, surface <= NO_WATER_SURFACE * 0.5));
}

// How fast the water's depth falls a meter along `step` at block `block`'s
// sample `sample` (`first` or `last` along it), `center` deep, its ground
// at texel `texel` of tile layer `layer` (the tiles hold a border of one)
// and its samples `spacing` meters apart: between the samples either side,
// or between it and the side the block holds and water reaches.
fn water_falling(block: u32, sample: vec2<u32>, step: vec2<u32>, first: bool, last: bool,
        center: f32, layer: u32, texel: vec2<u32>, spacing: f32) -> f32 {
    var before = vec2<f32>(center, 0.0);
    var after = vec2<f32>(center, 0.0);
    if !first {
        before = water_block_depth(block, sample - step, terrain_load(layer, texel - step));
    }
    if !last {
        after = water_block_depth(block, sample + step, terrain_load(layer, texel + step));
    }
    return (select(center, before.x, before.y > 0.5) - select(center, after.x, after.y > 0.5))
        / (spacing * max(before.y + after.y, 1.0));
}

// How fast the water's depth falls along x and z at block `block`'s sample
// `sample`, as `water_falling` gives it.
fn water_shoaling(block: u32, sample: vec2<u32>, center: f32, layer: u32, texel: vec2<u32>,
        spacing: f32) -> vec2<f32> {
    let last = WATER_BLOCK - 1u;
    return vec2<f32>(
        water_falling(block, sample, vec2<u32>(1u, 0u), sample.x == 0u, sample.x == last, center,
            layer, texel, spacing),
        water_falling(block, sample, vec2<u32>(0u, 1u), sample.y == 0u, sample.y == last, center,
            layer, texel, spacing));
}

fn water_output(input: WaterPatchInput) -> WaterOutput {
    var record: PatchInput;
    record.vertex = input.vertex;
    record.first = input.first;
    record.shape = input.shape;
    record.atlas = input.atlas;
    record.holes = 0xffffffffu;
    record.fade = input.fade;
    let ground = terrain_grid_vertex(record);
    // Its water at its sample and, morphing as the ground does, at its even
    // neighbor's in the parent's block, or its own block without one.
    // How fast its depth falls, from its samples' either side, morphing
    // alike: smooth across the triangles, where the depth's own change
    // between pixels is not.
    let spacing = input.shape.x;
    let height_texel = vec2<u32>(input.atlas.z & 0xffffu, input.atlas.z >> 16u) + ground.grid + 1u;
    let fine = water_block_sample(input.water.x, ground.grid, ground.fine);
    let fine_shoaling = water_shoaling(input.water.x, ground.grid, fine.y, input.atlas.x,
        height_texel, spacing);
    let even = ground.grid - ground.odd;
    var coarse = water_block_sample(input.water.x, even, ground.coarse);
    var coarse_shoaling = fine_shoaling;
    if input.water.y != NO_WATER_BLOCK {
        let offset = vec2<u32>(input.water.z & 0xffffu, input.water.z >> 16u);
        coarse = water_block_sample(input.water.y, offset + even / 2u, ground.coarse);
        let parent_texel = vec2<u32>(input.atlas.w & 0xffffu, input.atlas.w >> 16u)
            + even / 2u + 1u;
        coarse_shoaling = water_shoaling(input.water.y, offset + even / 2u, coarse.y,
            input.atlas.y, parent_texel, 2.0 * spacing);
    }
    let water = fine * (1.0 - ground.morph) + coarse * ground.morph;
    let surface = water.x;
    let texel = water_texel(input.water.x, ground.grid);
    let info = textureLoad(water_info, texel.xy, texel.z, 0);
    let relative = vec3<f32>(ground.vertex.relative.x, surface + terrain.anchor.y,
        ground.vertex.relative.z) + terrain_view.eye_offset.xyz;
    var output: WaterOutput;
    output.clip_position = terrain_view.relative_view_projection * vec4<f32>(relative, 1.0);
    output.relative = relative;
    output.water = vec3<f32>(water.y, info.xy * WATER_FLOW_RANGE);
    let width = select(saturate(info.w * 127.0 / 126.0) * WATER_WIDTH_RANGE, WATER_WIDTH_RANGE,
        info.w > 0.996);
    output.foam = vec2<f32>(max(info.z, 0.0), width);
    output.shore = fine_shoaling * (1.0 - ground.morph) + coarse_shoaling * ground.morph;
    return output;
}

@vertex
fn vs_water(input: WaterPatchInput) -> WaterOutput {
    return water_output(input);
}

// Water's motion is the camera's: its ripples move within its surface.
@vertex
fn vs_water_temporal(input: WaterPatchInput) -> WaterTemporalOutput {
    let output = water_output(input);
    var temporal: WaterTemporalOutput;
    temporal.clip_position = output.clip_position;
    temporal.relative = output.relative;
    temporal.water = output.water;
    temporal.foam = output.foam;
    temporal.shore = output.shore;
    temporal.previous = (camera.previous_view_projection
        * vec4<f32>(output.relative + camera.position.xyz, 1.0)).xyw;
    return temporal;
}

// Value noise at `p` and its gradient (x: value, yz: d/dx and d/dz), from
// hashed lattice values blended by quintic steps. The next cell's
// products are the cell's plus each multiplier (wrapping integers), one
// product per axis.
fn water_noise(p: vec2<f32>) -> vec3<f32> {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    let q = vec2<u32>(vec2<i32>(cell));
    let x0 = q.x * 0x8da6b343u;
    let x1 = x0 + 0x8da6b343u;
    let y0 = q.y * 0xd8163841u;
    let y1 = y0 + 0xd8163841u;
    let a = terrain_hash_of(x0 ^ y0);
    let b = terrain_hash_of(x1 ^ y0);
    let c = terrain_hash_of(x0 ^ y1);
    let d = terrain_hash_of(x1 ^ y1);
    let k = vec4<f32>(a, b - a, c - a, a - b - c + d);
    return vec3<f32>(k.x + k.y * u.x + k.z * u.y + k.w * u.x * u.y,
        du * vec2<f32>(k.y + k.w * u.y, k.z + k.w * u.x));
}

// Gradient noise at `p` (Perlin's), 0 to 1: hashed lattice gradients
// blended by quintic steps, without value noise's blocks.
fn water_gradient_noise(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let q = vec2<u32>(vec2<i32>(cell));
    let x0 = q.x * 0x8da6b343u;
    let x1 = x0 + 0x8da6b343u;
    let y0 = q.y * 0xd8163841u;
    let y1 = y0 + 0xd8163841u;
    let a = dot(vec2<f32>(terrain_hash_of(x0 ^ y0), terrain_hash_of(x0 ^ y0 ^ 0x68e31da4u))
        * 2.0 - 1.0, f);
    let b = dot(vec2<f32>(terrain_hash_of(x1 ^ y0), terrain_hash_of(x1 ^ y0 ^ 0x68e31da4u))
        * 2.0 - 1.0, f - vec2<f32>(1.0, 0.0));
    let c = dot(vec2<f32>(terrain_hash_of(x0 ^ y1), terrain_hash_of(x0 ^ y1 ^ 0x68e31da4u))
        * 2.0 - 1.0, f - vec2<f32>(0.0, 1.0));
    let d = dot(vec2<f32>(terrain_hash_of(x1 ^ y1), terrain_hash_of(x1 ^ y1 ^ 0x68e31da4u))
        * 2.0 - 1.0, f - vec2<f32>(1.0));
    return saturate(0.5 + 0.9 * mix(mix(a, b, u.x), mix(c, d, u.x), u.y));
}

// The gradient a corner of `water_gradient_noise3`'s lattice hashed to
// `key` holds, dotted with `offset` from it.
fn water_corner3(key: u32, offset: vec3<f32>) -> f32 {
    let gradient = vec3<f32>(terrain_hash_of(key), terrain_hash_of(key ^ 0x68e31da4u),
        terrain_hash_of(key ^ 0x2c1b3c6du)) * 2.0 - 1.0;
    return dot(gradient, offset);
}

// Gradient noise at `p` in three dimensions, 0 to 1, as
// `water_gradient_noise`'s.
fn water_gradient_noise3(p: vec3<f32>) -> f32 {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let q = vec3<u32>(vec3<i32>(cell));
    let x0 = q.x * 0x8da6b343u;
    let x1 = x0 + 0x8da6b343u;
    let y0 = q.y * 0xd8163841u;
    let y1 = y0 + 0xd8163841u;
    let z0 = q.z * 0xcb1ab31fu;
    let z1 = z0 + 0xcb1ab31fu;
    let low = mix(mix(water_corner3(x0 ^ y0 ^ z0, f),
            water_corner3(x1 ^ y0 ^ z0, f - vec3<f32>(1.0, 0.0, 0.0)), u.x),
        mix(water_corner3(x0 ^ y1 ^ z0, f - vec3<f32>(0.0, 1.0, 0.0)),
            water_corner3(x1 ^ y1 ^ z0, f - vec3<f32>(1.0, 1.0, 0.0)), u.x), u.y);
    let high = mix(mix(water_corner3(x0 ^ y0 ^ z1, f - vec3<f32>(0.0, 0.0, 1.0)),
            water_corner3(x1 ^ y0 ^ z1, f - vec3<f32>(1.0, 0.0, 1.0)), u.x),
        mix(water_corner3(x0 ^ y1 ^ z1, f - vec3<f32>(0.0, 1.0, 1.0)),
            water_corner3(x1 ^ y1 ^ z1, f - vec3<f32>(1.0)), u.x), u.y);
    return saturate(0.5 + 0.8 * mix(low, high, u.z));
}

// Where corner `corner` of the grid `water_tile` hangs tile `tile`'s
// offset: a random point of the tile (in its periods).
fn water_corner_offset(corner: vec2<i32>, tile: u32) -> vec2<f32> {
    let key = (u32(corner.x) * 0x8da6b343u) ^ (u32(corner.y) * 0xd8163841u) ^ (tile * 0xcb1ab31fu);
    return vec2<f32>(terrain_hash_of(key), terrain_hash_of(key ^ 0x68e31da4u));
}

// Tile `tile`'s waves at `uv` (in its periods), which changes by `dx` and
// `dy` per pixel: their slopes (xy) and the variance of those finer than
// the pixel (z), what its levels' averages lose of the slopes' square.
// Each pixel reads the tile at the random offsets of the three corners of
// the triangle of a skewed grid holding it, blended by its weights there
// so that their variance stays the tile's (tiling and blending, Heitz and
// Neyret, Deliot and Heitz): the waves, a Gaussian field, look the same
// everywhere and never repeat. A tile whose period spans fewer than
// WATER_WAVE_PIXELS pixels blurs into its variance, all of it once a pixel
// spans about a tenth of a period. Where its texels span more than a pixel,
// its levels' averages hide nothing of it: what filtering between texels
// loses of the slopes' square (lines along the texels' edges) is no waves'
// variance, so it fades out as texels grow past a pixel.
fn water_tile(tile: u32, uv: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>) -> vec3<f32> {
    let spread = max(length(dx), length(dy));
    let excess = max(spread * WATER_WAVE_PIXELS, 1.0);
    let blur = excess * excess;
    if spread * blur >= 1.0 {
        let mean = textureSampleLevel(water_waves, water_wave_sampler, vec2<f32>(0.5), tile,
            f32(textureNumLevels(water_waves) - 1u));
        return vec3<f32>(0.0, 0.0, mean.z);
    }
    let texels = sqrt(length(dx) * length(dy)) * blur * f32(textureDimensions(water_waves).x);
    let hidden = saturate(log2(max(texels, 1.0e-6)) + 1.0);
    let grid = uv / WATER_PATCH;
    let skewed = vec2<f32>(grid.x, 1.15470054 * grid.y - 0.57735027 * grid.x);
    let cell = floor(skewed);
    let local = skewed - cell;
    let base = vec2<i32>(cell);
    let third = 1.0 - local.x - local.y;
    var weights = vec3<f32>(third, local.y, local.x);
    var corners = array<vec2<i32>, 3>(base, base + vec2<i32>(0, 1), base + vec2<i32>(1, 0));
    if third <= 0.0 {
        weights = vec3<f32>(-third, 1.0 - local.y, 1.0 - local.x);
        corners = array<vec2<i32>, 3>(base + vec2<i32>(1, 1), base + vec2<i32>(1, 0),
            base + vec2<i32>(0, 1));
    }
    var slopes = vec2<f32>(0.0);
    var variance = 0.0;
    for (var corner = 0u; corner < 3u; corner++) {
        let moments = textureSampleGrad(water_waves, water_wave_sampler,
            uv + water_corner_offset(corners[corner], tile), tile, dx * blur, dy * blur);
        let weight = weights[corner];
        slopes += moments.xy * weight;
        variance += max(moments.z - dot(moments.xy, moments.xy), 0.0) * weight * weight;
    }
    let spread_kept = dot(weights, weights);
    return vec3<f32>(slopes * inverseSqrt(spread_kept), variance / spread_kept * hidden);
}

// The waves at `p` (meters along x and z), which changes by `dx` and `dy`
// per pixel, each tile's slopes scaled by its share of `tiles` (none read
// where it is 0): their slopes (xy) and the variance of those finer than
// the pixel (z).
fn water_waves_at(p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, tiles: vec3<f32>) -> vec3<f32> {
    var waves = vec3<f32>(0.0);
    for (var tile = 0u; tile < WATER_WAVE_TILES; tile++) {
        let share = tiles[tile];
        if share <= 0.0 {
            continue;
        }
        let repeats = water.waves[tile];
        let read = water_tile(tile, p * repeats, dx * repeats, dy * repeats);
        waves += vec3<f32>(read.xy * share, read.z * share * share);
    }
    return waves;
}

// Waves `first` blended `toward` waves `second`, their slopes keeping their
// spread (they are independent): as `water_waves_at` gives them.
fn water_blended(first: vec3<f32>, second: vec3<f32>, toward: f32) -> vec3<f32> {
    let keep = inverseSqrt((1.0 - toward) * (1.0 - toward) + toward * toward);
    return vec3<f32>(mix(first.xy, second.xy, toward) * keep, mix(first.z, second.z, toward));
}

// How much the wind's gusts scale the waves at `p` at `time`: patches of
// stronger and calmer wind, WATER_GUST_SIZE meters across and drawn out
// along it, sweep over the water with it (cat's paws), as strongly as it
// gusts.
fn water_gusts(p: vec2<f32>, time: f32) -> f32 {
    let wind = water.ripples.zw;
    let speed = length(wind);
    let heading = select(vec2<f32>(1.0, 0.0), wind / speed, speed > 1.0e-3);
    let moving = p - wind * time;
    let streaked = vec2<f32>(dot(moving, heading) / 2.5,
        dot(moving, vec2<f32>(-heading.y, heading.x))) / WATER_GUST_SIZE;
    let patches = water_noise(streaked).x * 0.65
        + water_noise(streaked * 2.3 + vec2<f32>(5.2, 1.3)).x * 0.35;
    let gusting = smoothstep(0.2, 0.8, patches);
    return max(mix(1.0 - 1.2 * water.ripples.y, 1.0 + water.ripples.y, gusting), 0.15);
}

// The roughness of water whose waves hide `variance` of their slopes'
// square within a pixel: calm water's, widened by that variance as a
// microfacet distribution's (LEAN mapping, Olano and Baker).
fn water_roughness(variance: f32) -> f32 {
    let calm = WATER_SMOOTH * WATER_SMOOTH;
    return min(sqrt(sqrt(calm * calm + variance)), WATER_ROUGHEST);
}

// What light keeps of each channel over `meters` of water.
fn water_transmitted(meters: f32) -> vec3<f32> {
    return exp(-water.absorption.rgb * meters);
}

// The normal that mirrors the eye's view along `view` no lower than
// WATER_HORIZON: `normal`, or, for slopes turned from the eye, the normal
// mirroring the view just over the horizon instead. Seen from under the
// water, `normal`.
fn water_facing(normal: vec3<f32>, view: vec3<f32>) -> vec3<f32> {
    let mirrored = reflect(-view, normal);
    let across = length(mirrored.xz);
    if view.y <= 0.0 || mirrored.y >= WATER_HORIZON || across < 1.0e-6 {
        return normal;
    }
    let level = mirrored.xz / across * sqrt(1.0 - WATER_HORIZON * WATER_HORIZON);
    return normalize(view + vec3<f32>(level.x, WATER_HORIZON, level.y));
}

// Henyey and Greenstein's phase for light scattering through `cosine` over
// its mean across the backward half: 1 on average where light entering the
// water turns back toward an eye above it.
fn water_phase(cosine: f32) -> f32 {
    let g = WATER_PHASE_G;
    let denominator = 1.0 + g * g - 2.0 * g * cosine;
    return (1.0 - g * g) / (denominator * sqrt(denominator)) / WATER_PHASE_MEAN;
}

// Light water's body scatters toward the eye along `view` per unit of its
// color, before the surface lets it out: the sky's light into its level
// surface, and the sun's `sunlight` refracted in and turned toward the eye,
// the more the nearer the eye looks toward the sun. Seen from under the
// water, the sky's alone.
fn water_scattered(view: vec3<f32>, sunlight: vec3<f32>) -> vec3<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let sky = sky_average(up) * lighting.zenith.w;
    let sun = lighting.sun.xyz;
    if sun.y <= 0.0 || view.y <= 0.0 {
        return sky;
    }
    let entering = refract(-sun, up, 1.0 / WATER_IOR);
    let leaving = -refract(-view, up, 1.0 / WATER_IOR);
    let into = 1.0 - fresnel(vec3<f32>(0.02), sun.y).r;
    return sky + sunlight * (sun.y * into * water_phase(dot(entering, leaving)));
}

// What the eye sees through water at its point `relative` (less the eye)
// of normal `normal` along `view`, where the opaque surface behind its
// pixel `pixel` stands at `ground` (less the eye): the opaque scene's color
// where the view, bent into the water, meets it (graded, as the scene holds
// it), and that surface less the eye. The bent view runs on to the depth of
// the surface behind the pixel; what stands nearer than the water, or out
// of it, shows unbent, as does everything seen from under the water.
struct WaterBehind {
    color: vec3<f32>,
    ground: vec3<f32>,
}

fn water_behind(relative: vec3<f32>, normal: vec3<f32>, view: vec3<f32>, pixel: vec2<f32>,
        ground: vec3<f32>) -> WaterBehind {
    let size = vec2<f32>(textureDimensions(water_depth));
    var behind: WaterBehind;
    behind.ground = ground;
    var at = pixel;
    let bent = refract(-view, normal, 1.0 / WATER_IOR);
    if view.y > 0.0 && bent.y < 0.0 {
        let below = max(relative.y - ground.y, 0.0);
        let reach = min(below / max(-bent.y, 0.1), WATER_REACH);
        let clip = camera.view_projection
            * vec4<f32>(relative + bent * reach + camera.position.xyz, 1.0);
        if clip.w > 1.0e-6 {
            let moved = (vec2<f32>(clip.x, -clip.y) / clip.w * 0.5 + 0.5) * size;
            if all(moved >= vec2<f32>(0.0)) && all(moved < size)
                && textureLoad(water_depth, vec2<i32>(moved), 0) > 0.0 {
                let there = water_ground(moved);
                if dot(there, there) > dot(relative, relative) && there.y < relative.y {
                    behind.ground = there;
                    at = moved;
                }
            }
        }
    }
    behind.color = textureSampleLevel(water_scene, refraction_sampler, at / size, 0.0).rgb;
    return behind;
}

// A water surface to light: its point less the eye and how it changes per
// pixel; its waves' normal and the roughness of what a pixel spans of them;
// the share of it foam covers; what its body keeps of each channel of the
// light on its way to what lies below and back; and light it adds of its
// own (the sea's crests lit through).
struct WaterSurface {
    relative: vec3<f32>,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
    foam: f32,
    transmit: vec3<f32>,
    glow: vec3<f32>,
}

// A water surface's light, fogged and graded, and its reflectance.
struct WaterLit {
    shaded: Shaded,
    reflectance: f32,
}

// The light of water surface `surface`: the sky and scene it mirrors and
// the sun's disc, by Fresnel's share; the light its body scatters back
// toward the eye, for as much as it stops on its way to what lies below and
// back, through what the surface lets out; and its foam, a white layer
// lit as diffuse ground. One shadow lookup serves the sun's three.
fn water_light(surface: WaterSurface) -> WaterLit {
    let world = surface.relative + camera.position.xyz;
    let distance = length(surface.relative);
    let view = stable_normalize(-surface.relative);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let normal = water_facing(surface.normal, view);
    let roughness = mix(surface.roughness, 0.7, surface.foam);
    let alpha = roughness * roughness;
    let n_v = max(dot(normal, view), 1.0e-4);
    let f0 = vec3<f32>(0.02);
    let reflectance = saturate(luminance(environment_brdf(f0, roughness, n_v)));
    let sun = lighting.sun.xyz;
    var sunlight = vec3<f32>(0.0);
    if sun.y > 0.0 {
        sunlight = lighting.sun_color.rgb
            * sun_visibility(world, up, surface.world_dx, surface.world_dy, distance, false);
    }
    // The sun's disc mirrored: its lobe widened by the half vectors the
    // disc spans (Karis), which also keeps calm water's glint from
    // flickering in single pixels.
    var glint = vec3<f32>(0.0);
    let n_l = dot(normal, sun);
    if n_l > 0.0 && any(sunlight > vec3<f32>(0.0)) {
        let widened = alpha + WATER_SUN_ALPHA;
        let half_vector = normalize(view + sun);
        glint = fresnel(f0, max(dot(view, half_vector), 0.0)) * sunlight
            * (distribution(max(dot(normal, half_vector), 0.0), widened)
                * visibility(n_v, n_l, widened) * n_l * lighting.sun_color.w);
    }
    let body = water.color.rgb * (vec3<f32>(1.0) - surface.transmit)
        * water_scattered(view, sunlight) * (1.0 - reflectance);
    // Foam scatters the sun's light, as the waves it rides face it, and the
    // light around it: the sky's, of which water in its channel sees
    // WATER_FOAM_SKY, and the ground's around it for the rest, sunlit and
    // skylit as the terrain's hollows take it, so foam in shade stands no
    // bluer than the banks beside it.
    let sky = sky_average(up) * lighting.zenith.w;
    let around = lighting.sun_color.rgb * (0.35 * saturate(sun.y)) + sky * 0.5;
    let foam = WATER_FOAM_ALBEDO
        * (mix(around, sky, WATER_FOAM_SKY) + sunlight * max(dot(surface.normal, sun), 0.0));
    // The mirrored sky and scene come through `finish`; the sun's light is
    // the water's own.
    var point: SurfacePoint;
    point.albedo = water.color.rgb;
    point.opacity = 1.0;
    point.emission = 0.0;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = 1.0;
    point.bounce = vec3<f32>(0.0);
    point.roughness = roughness;
    point.alpha = alpha;
    point.f0 = f0;
    point.diffuse_color = vec3<f32>(0.0);
    point.geometric = up;
    point.normal = normal;
    point.distance = distance;
    point.view = view;
    point.n_v = n_v;
    point.world = world;
    point.world_dx = surface.world_dx;
    point.world_dy = surface.world_dy;
    point.shadow_offset = vec3<f32>(0.0);
    point.on_terrain = false;
    point.sun_shade = 1.0;
    var lit: WaterLit;
    lit.shaded = finish(point, mix(body + glint, foam, surface.foam) + surface.glow);
    lit.reflectance = reflectance;
    return lit;
}

// Water's light, and how it covers what lies below: in frames blending it,
// the share of the ground it hides (`coverage`); in frames refracting the
// scene, what shows through it (graded, as the scene holds it) and the
// share of the scene it covers, all but where its shores fade.
struct WaterShade {
    shaded: Shaded,
    coverage: f32,
    behind: vec3<f32>,
    fade: f32,
}

// Water's light and what shows through it as its targets' color: the scene
// covered but where the shores fade it out.
fn water_covering(shade: WaterShade) -> vec4<f32> {
    return vec4<f32>((shade.shaded.color + shade.behind) * shade.fade, shade.fade);
}

// Flowing water's ripples: two sets, each carried
// along the flow for WATER_FLOW_CYCLE seconds and then starting over, half
// a cycle apart: the seconds each has run, and how much the second shows,
// so each restarts unseen before the flow stretches it (Vlachos's flow
// maps).
const WATER_FLOW_CYCLE: f32 = 1.6;
// Where the second set lies in the noise, so the two never pulse in step.
const WATER_FLOW_SHIFT: vec2<f32> = vec2<f32>(17.3, 9.1);
// Foam's lace: the meters across its larger cells (its finer ones a third
// of that), the width (in cells) of its lines at the waterline, the meters
// from the waterline it reaches out and the meters of water over what
// stands in it within which it rings it, and how much of what lies under it
// foam hides: foam is thin.
const WATER_LACE_CELL: f32 = 0.24;
const WATER_SHORE_LACE: f32 = 0.42;
// The width (in cells) of the lace where water lands from a fall, and the
// whitewater above which it has (`engine_terrain::hydraulics::RAPIDS_WHITE`
// and whiter).
const WATER_PLUNGE_LACE: f32 = 0.45;
const WATER_PLUNGE: f32 = 0.72;
const WATER_FOAM_REACH: f32 = 1.6;
// The share of narrow water's half width its shores' foam reaches over.
const WATER_FOAM_SHARE: f32 = 0.22;
const WATER_FOAM_RING: f32 = 0.2;
const WATER_FOAM_OPACITY: f32 = 0.9;
// Seconds each of foam's two sets of lace lives, half a life apart, on
// still water (a wave lapping the shore every half life) and on running
// water (shorter, so the current's turns and changes draw its lace out
// less); the meters across which the place's clock for them varies, and
// the lives it varies by (more would draw out the lace a current carries);
// and the meters a wave lapping still water's shore washes its lace in.
const WATER_FOAM_LAPPED: f32 = 5.0;
const WATER_FOAM_CARRIED: f32 = 3.0;
const WATER_FOAM_STAGGER: f32 = 29.0;
const WATER_FOAM_STAGGERED: f32 = 0.8;
const WATER_FOAM_WASH: f32 = 0.35;

struct WaterFlow {
    first: f32,
    second: f32,
    blend: f32,
}

fn water_flow(time: f32) -> WaterFlow {
    let phase = fract(time / WATER_FLOW_CYCLE);
    var carried: WaterFlow;
    carried.first = phase * WATER_FLOW_CYCLE;
    carried.second = fract(phase + 0.5) * WATER_FLOW_CYCLE;
    carried.blend = abs(phase * 2.0 - 1.0);
    return carried;
}

// How near `p` (in cells) lies to the edges between the cells of a jittered
// grid: cellular noise's second-nearest distance less its nearest, 0 on an
// edge. Foam gathers along these edges into lace; they hold still, and the
// lace moves only as the water carries it.
fn water_cells(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let local = p - cell;
    let base = vec2<i32>(cell);
    var nearest = 8.0;
    var next = 8.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let at = vec2<u32>(base + vec2<i32>(x, y));
            let key = (at.x * 0x8da6b343u) ^ (at.y * 0xd8163841u);
            let seed = vec2<f32>(terrain_hash_of(key), terrain_hash_of(key ^ 0x68e31da4u));
            let point = vec2<f32>(f32(x), f32(y)) + 0.15 + 0.7 * seed;
            let distance = length(point - local);
            if distance < nearest {
                next = nearest;
                nearest = distance;
            } else if distance < next {
                next = distance;
            }
        }
    }
    return next - nearest;
}

// Lace of lines `width` (in cells, of `water_cells`' edges) where those
// edges read `edge`, as a pixel spanning `blur` cells sees it: sharp
// where it spans a small part of a cell, as bright as its lines' share of
// the pixel as they thin below it.
fn water_lace(edge: f32, width: f32, blur: f32) -> f32 {
    let soft = max(2.0 * blur, 1.0e-4);
    return (1.0 - smoothstep(width - soft, width + soft, edge)) * min(1.0, width / soft + 0.25);
}

// One set of foam's lace at `p` (in its larger cells), its lines `width`
// cells wide for a pixel spanning `blur` cells, `worn` through its life:
// the cells' edges bent by smooth noise at two scales so its threads wander
// and curl, their width varying over clumps a few cells across, where they
// gather and thin out, and parting here and there, the more the older it
// is, and finer cells' thinner threads within them up close.
fn water_lace_set(p: vec2<f32>, width: f32, blur: f32, worn: f32) -> f32 {
    let curl = vec2<f32>(water_noise(p * 0.9).x, water_noise(p * 0.9 + vec2<f32>(5.3, 1.7)).x)
        - 0.5;
    let sway = vec2<f32>(water_noise(p * 0.3 + vec2<f32>(8.1, 2.2)).x,
        water_noise(p * 0.3 + vec2<f32>(1.4, 9.6)).x) - 0.5;
    let bent = p + curl * 0.6 + sway * 1.6;
    let clumps = water_noise(p * 0.2 + vec2<f32>(2.9, 8.3)).x;
    let parted = smoothstep(0.22 + 0.25 * worn, 0.55 + 0.15 * worn,
        water_noise(bent * 1.4 + vec2<f32>(4.4, 0.7)).x);
    let gathered = min(width * (0.2 + 1.6 * clumps * clumps) * (0.3 + 0.7 * parted), 0.4);
    var lace = water_lace(water_cells(bent), gathered, blur);
    if blur < 0.25 {
        let fine = water_lace(water_cells(bent * 3.0 + 7.1), gathered * 0.55, blur * 3.0);
        lace = max(lace, fine * 0.7 * (1.0 - smoothstep(0.12, 0.25, blur)));
    }
    return lace;
}

// One of foam's two sets of lace (`which`), half a life apart, where the
// place's clock reads `clock` (lives): how far through its life it is (0 as
// it forms, 1 once it has dissolved), where its lace lies in the noise
// (cells), and how strongly it forms (on still water, how big the wave
// washing it in is), all anew each life, so no lace comes back where it
// was.
struct WaterFoamSet {
    age: f32,
    offset: vec2<f32>,
    strength: f32,
}

fn water_foam_set(clock: f32, which: u32) -> WaterFoamSet {
    let shifted = clock + 0.5 * f32(which);
    let life = floor(shifted);
    let key = (u32(life) * 2u + which) * 0x9e3779b9u;
    var lived: WaterFoamSet;
    lived.age = shifted - life;
    lived.offset = vec2<f32>(terrain_hash_of(key), terrain_hash_of(key ^ 0x68e31da4u)) * 64.0;
    lived.strength = 0.6 + 0.4 * terrain_hash_of(key ^ 0x2c1b3c6du);
    return lived;
}

// What stands for a set of foam's lace (`which`) on still water or, where
// `running`, on running water, at `p` (meters) where its clock reads
// `clock` (lives): the width (cells) of its lines, and its lace for a pixel
// spanning `blur` cells where `drawn`. Along the waterline its lines are
// `lining` wide, and where water lands from a fall `plunge` of
// WATER_PLUNGE_LACE. On running water each set forms as the current
// (`current`, meters a second) carries it off and dissolves as the other
// forms; on still water each is a wave lapping the shore, washing its lace
// in toward it (against `offshore`), as big as `lapping` says, and drawing
// it back out as it dissolves.
fn water_foam_lace(which: u32, running: bool, p: vec2<f32>, clock: f32, lining: f32,
        plunge: f32, current: vec2<f32>, offshore: vec2<f32>, lapping: f32, blur: f32,
        drawn: bool) -> vec2<f32> {
    let lived = water_foam_set(clock, which);
    let age = lived.age;
    let swept = 1.0 - abs(2.0 * age - 1.0);
    var width = max(lining, WATER_PLUNGE_LACE * plunge) * swept;
    var at = p - current * (age * WATER_FOAM_CARRIED);
    if !running {
        let lapped = smoothstep(0.0, 0.12, age) * (1.0 - smoothstep(0.25, 1.0, age))
            * lived.strength * lapping;
        width = max(lining * lapped, WATER_PLUNGE_LACE * plunge * swept);
        at = p + offshore * (WATER_FOAM_WASH * smoothstep(0.0, 0.18, age)
            * (1.0 - smoothstep(0.18, 1.0, age)));
    }
    var lace = 0.0;
    if drawn && width > 1.0e-3 {
        lace = water_lace_set(at / WATER_LACE_CELL + lived.offset, width, blur, age);
    }
    return vec2<f32>(width, lace);
}

// The waves where water flows along `flow` (meters a second), as
// `water_waves_at` gives them of `tiles`: two sets carried along it half a
// cycle apart as `carried` says, blended so their slopes keep their spread.
fn water_flowing_waves(p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, tiles: vec3<f32>,
        flow: vec2<f32>, carried: WaterFlow) -> vec3<f32> {
    let first = water_waves_at(p - flow * carried.first, dx, dy, tiles);
    let second = water_waves_at(p + WATER_FLOW_SHIFT - flow * carried.second, dx, dy, tiles);
    return water_blended(first, second, carried.blend);
}

// Tiles' water at `input`, refracting the scene's color when `refracts`.
fn shade_water(input: WaterOutput, refracts: bool) -> WaterShade {
    let world = input.relative + camera.position.xyz;
    // Meters of water over the ground: the shore's dry side holds none.
    let depth = input.water.x;
    if depth <= 0.0 {
        discard;
    }
    let flow = input.water.yz;
    let time = camera.position.w;
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    // Still water's waves run with the wind as they evolve; flowing water's
    // run with it, carried along in two sets that cross-fade; between, the
    // two blend, so nothing changes at once where a river meets a lake.
    let speed = length(flow);
    let flowing = smoothstep(0.05, 0.3, speed);
    // Running water raises no swell, and breaks into finer, livelier
    // ripples.
    let running = saturate(speed * 0.5);
    // How far the pixel spans on the water.
    let footprint = max(length(world_dx.xz), length(world_dy.xz));
    let carried = water_flow(time);
    var waves = vec3<f32>(0.0);
    if flowing < 1.0 {
        waves = water_waves_at(world.xz, world_dx.xz, world_dy.xz, vec3<f32>(1.0));
    }
    if flowing > 0.0 {
        let tiles = vec3<f32>(0.0, mix(1.0, 0.6, running), mix(1.0, 1.5, running));
        let moving = water_flowing_waves(world.xz, world_dx.xz, world_dy.xz, tiles, flow,
            carried);
        waves = water_blended(waves, moving, flowing);
    }
    let gusts = water_gusts(world.xz, time);
    let strength = water.ripples.x * mix(1.0, 1.3, running) * gusts;
    let slopes = waves.xy * strength;
    let normal = normalize(vec3<f32>(-slopes.x, 1.0, -slopes.y));
    let roughness = water_roughness(waves.z * strength * strength);
    let view = stable_normalize(-input.relative);
    // Under the sea, the sea's water stands instead.
    if water.ocean.y > 0.5 && input.relative.y < water.ocean.x {
        discard;
    }
    // What lies behind: the opaque surface behind the pixel and, where the
    // frame refracts the scene, what the bent view meets and its color.
    let pixel = input.clip_position.xy;
    let own = water_ground(pixel);
    var behind: WaterBehind;
    behind.ground = own;
    behind.color = vec3<f32>(0.0);
    if refracts {
        behind = water_behind(input.relative, normal, view, pixel, own);
    }
    // Light comes back through the water along the view to what it shows,
    // as far as the depth lets it run, after reaching it down from the
    // surface.
    let through = min(max(length(behind.ground - input.relative), 0.0),
        depth / max(view.y, 0.1));
    let down = clamp(input.relative.y - behind.ground.y, 0.0, through);
    let transmit = water_transmitted(through + down);
    // Foam: lace along the waterline, where shores and what stands in the
    // water meet it, out to WATER_FOAM_REACH meters (as the settings' foam
    // width scales it), its lines thick at the line and thinning to threads
    // farther out; and dense lace where water lands from a fall, thinning as
    // the current carries it off. Open water, however fast it runs, carries
    // none. The water makes and moves it, in two sets half a life apart,
    // each forming anew as the other dissolves: on running water the
    // current carries each off as it forms, slower at the bank, drawing it
    // out along it; on still water each is a wave lapping the shore, washing
    // its lace in, as big as the gusts raise the waves there, and drawing it
    // back out as it dissolves. Where the pixel spans most of its cells, its
    // share of the surface instead. Foam is thin: WATER_FOAM_OPACITY of it
    // hides the water.
    //
    // Meters from the waterline: the depth over how fast it falls toward
    // the shore (water as thin as the ground it lies on stands no nearer a
    // shore for it), and the way out from the shore; and how near the
    // water's surface what stands in it rises behind the pixel, where it
    // stands solid: not where it changes from pixel to pixel, as blades of
    // grass and twigs do.
    let rise = length(input.shore);
    let from_shore = depth / max(rise, 1.0e-3);
    let offshore = select(vec2<f32>(0.0), -input.shore / rise, rise > 1.0e-6);
    // Narrow water holds a narrower band: no more than a share of its half
    // width.
    let standing = max(input.relative.y - own.y, 0.0);
    let solid = 1.0 - smoothstep(0.03, 0.12, abs(dpdx(standing)) + abs(dpdy(standing)));
    let reach = min(WATER_FOAM_REACH * water.foam.y / 0.6, input.foam.y * WATER_FOAM_SHARE);
    let shore = max(1.0 - smoothstep(0.0, reach, from_shore),
        select(0.0, 1.0 - smoothstep(0.0, WATER_FOAM_RING, standing), standing < depth * 0.8)
            * solid);
    let lining = WATER_SHORE_LACE * shore * sqrt(shore) * water.foam.x / 0.6;
    let plunge = smoothstep(WATER_PLUNGE, 0.95, input.foam.x) * smoothstep(0.0, 0.05, depth);
    // Lace lies on the surface as seen from above: where the surface itself
    // falls steeply (water running down a chute), it would stretch down it,
    // so it gives way there.
    let facing = cross(world_dx, world_dy);
    let steep = length(facing.xz) / max(abs(facing.y), 1.0e-6 * dot(facing, facing));
    let flat = 1.0 - smoothstep(0.5, 1.0, steep);
    // Water running down a chute too steep for it to flow smoothly (a
    // cascade between falls) breaks white all across, as falling water
    // does (`fall_pattern`): streaked along the current and carried down
    // with it, in two sets half a cycle apart.
    let cascade = smoothstep(0.35, 0.8, steep) * smoothstep(1.0, 3.0, speed);
    var white = 0.0;
    if cascade > 1.0e-3 {
        let heading = flow / max(speed, 1.0e-4);
        let side = vec2<f32>(-heading.y, heading.x);
        let first = world.xz - flow * carried.first;
        let second = world.xz + WATER_FLOW_SHIFT - flow * carried.second;
        let one = fall_pattern(vec2<f32>(dot(first, side), dot(first, heading)), 1.0,
            vec2<f32>(0.0));
        let other = fall_pattern(vec2<f32>(dot(second, side), dot(second, heading)), 1.0,
            vec2<f32>(5.3, 2.9));
        white = cascade * smoothstep(0.2, 0.45, mix(one.x, other.x, carried.blend));
    }
    var foam = 0.0;
    if max(lining, plunge) * flat > 1.0e-3 {
        let blur = footprint / WATER_LACE_CELL;
        let far = smoothstep(0.35, 0.9, blur);
        let stagger = water_noise(world.xz / WATER_FOAM_STAGGER).x * WATER_FOAM_STAGGERED;
        let current = flow * mix(0.75, 1.0, smoothstep(0.0, 3.0 * reach, from_shore));
        let lapping = clamp(gusts, 0.3, 1.3);
        var lace = 0.0;
        var covered = 0.0;
        for (var which = 0u; which < 2u; which++) {
            var lapped = vec2<f32>(0.0);
            var streamed = vec2<f32>(0.0);
            if flowing < 1.0 {
                lapped = water_foam_lace(which, false, world.xz, time / WATER_FOAM_LAPPED + stagger,
                    lining, plunge, current, offshore, lapping, blur, far < 1.0);
            }
            if flowing > 0.0 {
                streamed = water_foam_lace(which, true, world.xz, time / WATER_FOAM_CARRIED + stagger,
                    lining, plunge, current, offshore, lapping, blur, far < 1.0);
            }
            let both = mix(lapped, streamed, flowing);
            covered += both.x;
            lace = max(lace, both.y);
        }
        foam = mix(lace, min(covered * 2.2, 0.8), far) * flat * WATER_FOAM_OPACITY;
    }
    foam = max(foam, white);
    var surface: WaterSurface;
    surface.relative = input.relative;
    surface.world_dx = world_dx;
    surface.world_dy = world_dy;
    surface.normal = normal;
    surface.roughness = roughness;
    surface.foam = foam;
    surface.transmit = transmit;
    surface.glow = vec3<f32>(0.0);
    let lit = water_light(surface);
    // Of what the surface lets through, the ground shows darker for being
    // wet however thin the water; foam covers it. The water thins out to
    // nothing where it meets the shore and around what stands in it.
    let wet = smoothstep(0.0, 0.03, depth) * WATER_WET;
    let let_through = (1.0 - lit.reflectance) * (1.0 - foam);
    let fade = smoothstep(0.0, WATER_SHORE, min(depth, max(input.relative.y - own.y, 0.0)));
    var result: WaterShade;
    result.shaded = lit.shaded;
    result.coverage = max(max(1.0 - (1.0 - lit.reflectance) * luminance(transmit), wet), foam)
        * fade;
    result.behind = behind.color * transmit
        * (let_through * (1.0 - wet) * (1.0 - atmospheric_cover(camera.position.xyz, world)));
    result.fade = fade;
    return result;
}

fn water_temporal_output(input: WaterTemporalOutput) -> WaterOutput {
    var output: WaterOutput;
    output.clip_position = input.clip_position;
    output.relative = input.relative;
    output.water = input.water;
    output.foam = input.foam;
    output.shore = input.shore;
    return output;
}

// Colors blend premultiplied over the scene: the scene target's alpha
// keeps the ground's ambient share, the other targets take the water's.
// The canvas holds no copy of the scene's color: water blends over the
// ground there.
@fragment
fn fs_water(input: WaterOutput) -> @location(0) vec4<f32> {
    let shade = shade_water(input, false);
    return vec4<f32>(encoded(shade.shaded.color) * shade.coverage, shade.coverage);
}

@fragment
fn fs_water_post(input: WaterOutput) -> SceneTargets {
    let shade = shade_water(input, true);
    var targets = scene_targets(shade.shaded);
    targets.color = water_covering(shade);
    return targets;
}

@fragment
fn fs_water_temporal(input: WaterTemporalOutput) -> TemporalTargets {
    let shade = shade_water(water_temporal_output(input), true);
    var targets = temporal_targets(shade.shaded, input.clip_position, input.previous);
    targets.color = water_covering(shade);
    return targets;
}

// Waterfalls' sheets (`water_falls.rs`): each fall's water from a little up
// its river over its lip, through the air on each jet and down the chutes
// between, into the water it lands in at last. Each vertex holds where
// across the sheet it lies, how long the water passing it has been on its
// way and how fast it left the lip, so everything is read where the water
// passing a point was across the sheet and when it left the lip: its
// strands, streaks and clumps travel with it and stretch out as it speeds
// up. Water leaves the lip clear and smooth and turns white on its way,
// its foam's strands brightest; the air breaks it up as it flies, its
// edges first, into strands with gaps through which its far side (the same
// water, the sheet's thickness behind) and the rock behind show; it boils
// where it lands, and runs whole down the chutes, its edges ragged. White
// water scatters the sun's and sky's light, and thin white water glows
// with the sun behind it. It thins out where it meets the rock.
struct FallInput {
    @location(0) position: vec3<f32>,
    // The sheet's face, turned out from the rock it falls from.
    @location(1) normal: vec3<f32>,
    // The way its water runs, and how fast (w, meters a second).
    @location(2) tangent: vec4<f32>,
    // Where across the sheet it lies (-1 to 1), the seconds since the water
    // passing it left the lip (negative up the river), the meters a second
    // it left at and the sheet's half width at the lip (meters).
    @location(3) jet: vec4<f32>,
    // The seconds since the water passing it last took off where it flies,
    // less those since it last landed down a chute, the sheet's thickness
    // (meters), its fall's number, and 1 where its water flies.
    @location(4) fall: vec4<f32>,
}

struct FallOutput {
    @builtin(position) clip_position: vec4<f32>,
    // The point less the eye.
    @location(0) relative: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) jet: vec4<f32>,
    @location(4) fall: vec4<f32>,
}

struct FallTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) jet: vec4<f32>,
    @location(4) fall: vec4<f32>,
    @location(5) previous: vec3<f32>,
}

fn fall_output(input: FallInput) -> FallOutput {
    var output: FallOutput;
    output.clip_position = camera.view_projection * vec4<f32>(input.position, 1.0);
    output.relative = input.position - camera.position.xyz;
    output.normal = input.normal;
    output.tangent = input.tangent;
    output.jet = input.jet;
    output.fall = input.fall;
    return output;
}

@vertex
fn vs_fall(input: FallInput) -> FallOutput {
    return fall_output(input);
}

// The sheet stands still: its motion is the camera's.
@vertex
fn vs_fall_temporal(input: FallInput) -> FallTemporalOutput {
    let output = fall_output(input);
    var temporal: FallTemporalOutput;
    temporal.clip_position = output.clip_position;
    temporal.relative = output.relative;
    temporal.normal = output.normal;
    temporal.tangent = output.tangent;
    temporal.jet = output.jet;
    temporal.fall = output.fall;
    temporal.previous = (camera.previous_view_projection * vec4<f32>(input.position, 1.0)).xyw;
    return temporal;
}

fn fall_temporal_output(input: FallTemporalOutput) -> FallOutput {
    var output: FallOutput;
    output.clip_position = input.clip_position;
    output.relative = input.relative;
    output.normal = input.normal;
    output.tangent = input.tangent;
    output.jet = input.jet;
    output.fall = input.fall;
    return output;
}

// Falling water: the meters across and along (at the speed it left its
// lip) of the strands it draws out into, and of the finer streaks within
// them; the meters across the clumps the air it takes in gathers it into,
// and how fast they churn (a second); the seconds on its way after which
// it has turned white, and the seconds of flight after which the air has
// broken it up as far as it goes; how much light white water scatters
// back, and how white the foam between its strands stands; the meters its
// strands stand out of the sheet; how much the sheet's far side shows in
// its gaps, and how much darker it stands for the near side's shade; how
// strongly white water scatters the sun forward (Henyey and Greenstein's
// g) and how much of what it scatters a thin sheet lets through; the
// meters over which it thins out where it meets the rock; and the seconds
// water boils for after it lands, churning faster and gathering bigger
// clumps the harder it landed.
const FALL_BOIL: f32 = 0.8;
const FALL_STRANDS: vec2<f32> = vec2<f32>(0.9, 4.0);
const FALL_STREAKS: vec2<f32> = vec2<f32>(0.3, 1.5);
const FALL_CLUMPS: f32 = 0.55;
const FALL_CHURN: f32 = 1.2;
const FALL_WHITE: f32 = 0.7;
const FALL_BROKEN: f32 = 2.5;
const FALL_ALBEDO: f32 = 0.8;
const FALL_BETWEEN: f32 = 0.4;
const FALL_RELIEF: f32 = 0.15;
const FALL_FAR: f32 = 0.9;
const FALL_FAR_SHADE: f32 = 0.6;
const FALL_PHASE_G: f32 = 0.5;
const FALL_THROUGH: f32 = 0.15;
const FALL_SOFT: f32 = 0.2;

// What stands of falling water at `p` (meters across, and along at the
// speed it left its lip, of where the water passing it was as it left),
// `flown` seconds on its way, its pattern `shift` cells along: how dense
// its strands, wandering across it, streaks and churning clumps stand
// there (0 to 1, x), and its clumps alone (y).
fn fall_pattern(p: vec2<f32>, flown: f32, shift: vec2<f32>) -> vec2<f32> {
    let wander = water_gradient_noise(p / (FALL_STRANDS * 3.0) + shift.yx) - 0.5;
    let strands = water_gradient_noise((p + vec2<f32>(wander * 1.5, 0.0)) / FALL_STRANDS + shift);
    let streaks = water_gradient_noise(p / FALL_STREAKS + shift * 1.7 + vec2<f32>(3.1, 7.7));
    let clumps = water_gradient_noise3(vec3<f32>(p / FALL_CLUMPS + shift * 0.6,
        flown * FALL_CHURN));
    return vec2<f32>(0.5 * strands + 0.3 * streaks + 0.2 * clumps, clumps);
}

// The share of a pixel a sheet whose pattern reads `density` covers, where
// it breaks up below `threshold`, as sharply as the pattern's change
// across the pixel (`blur`) allows.
fn fall_cover(density: f32, threshold: f32, blur: f32) -> f32 {
    let soft = 0.04 + blur;
    return smoothstep(threshold - soft, threshold + soft, density);
}

// A fall's light and the share of what lies behind it it hides.
struct FallShade {
    shaded: Shaded,
    coverage: f32,
}

fn shade_fall(input: FallOutput) -> FallShade {
    let world = input.relative + camera.position.xyz;
    let time = camera.position.w;
    let side = input.jet.x;
    let flown = input.jet.y;
    let lip_speed = input.jet.z;
    let half = input.jet.w;
    let flying = max(input.fall.x, 0.0);
    let thickness = input.fall.y;
    let number = u32(input.fall.z + 0.5);
    let aloft = input.fall.w;
    // It boils where it lands, for FALL_BOIL after.
    let boil = (1.0 - smoothstep(0.0, FALL_BOIL, -input.fall.x)) * (1.0 - aloft);
    let speed = max(input.tangent.w, 0.3);
    let falling = max(flown, 0.0);
    let view = stable_normalize(-input.relative);
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    let normal = normalize(input.normal);
    let tangent = normalize(input.tangent.xyz);
    let across = cross(tangent, normal);
    // The face the eye sees, and how squarely.
    let facing = dot(normal, view);
    let seen = select(-normal, normal, facing >= 0.0);
    let squarely = max(abs(facing), 0.05);
    // Where the water passing here was across the sheet and when it left
    // the lip (meters at the lip's speed), each fall's pattern its own; and
    // where the view leaves the sheet through its far side, the sheet's
    // thickness behind, in the same terms: as far across and along it as
    // the view runs within it, the water there having left the lip as much
    // earlier as it has run farther.
    let number_key = number * 0x9e3779b9u;
    let shift = vec2<f32>(terrain_hash_of(number_key), terrain_hash_of(number_key ^ 0x68e31da4u))
        * 97.0;
    let p = vec2<f32>(side * half, (time - flown) * lip_speed);
    let crossing = (seen * dot(view, seen) - view) * (thickness / squarely);
    let far_p = p + vec2<f32>(dot(crossing, across), -dot(crossing, tangent) * lip_speed / speed);
    let near = fall_pattern(p * (1.0 - 0.35 * boil), falling * (1.0 + 2.0 * boil), shift);
    let far = fall_pattern(far_p, falling, shift + vec2<f32>(31.7, 17.3));
    // The air breaks the water up as it flies, its edges first; down a
    // chute it runs whole, its edges ragged where its strands thin out.
    let broken = smoothstep(0.1, FALL_BROKEN, flying) * aloft;
    let edge = smoothstep(0.5, 1.0, abs(side));
    let threshold = mix(-0.1, 0.45, broken) + edge * mix(0.6, 0.7, broken);
    let blur = fwidth(near.x);
    let near_cover = fall_cover(near.x, threshold, blur);
    let far_cover = fall_cover(far.x, threshold, blur) * FALL_FAR * aloft;
    let cover = near_cover + far_cover * (1.0 - near_cover);
    // Clear as it leaves the lip, white on its way, its foam's strands and
    // clumps whitest.
    let foam = smoothstep(0.3, 0.7, near.x + 0.3 * (near.y - 0.5));
    let white = saturate(falling / FALL_WHITE) * mix(FALL_BETWEEN, 1.0, max(foam, 0.7 * boil));
    // Its strands stand out of the sheet: the normal they bend (surface
    // gradients, Mikkelsen).
    let height = near.x * FALL_RELIEF * (1.0 + 2.0 * boil);
    let r1 = cross(world_dy, seen);
    let r2 = cross(seen, world_dx);
    let det = dot(world_dx, r1);
    let gradient = sign(det) * (dpdx(height) * r1 + dpdy(height) * r2);
    let bent = normalize(abs(det) * seen - gradient);
    // Clear water seen edge on crosses more of itself. It fades in over the
    // river up from its lip, and thins out where it meets the rock.
    let clear = 1.0 - pow(0.45, 1.0 / squarely);
    let behind = water_ground(input.clip_position.xy);
    let meeting = smoothstep(0.0, FALL_SOFT, length(behind) - length(input.relative));
    let coverage = cover * mix(clear, 1.0, saturate(falling / FALL_WHITE))
        * smoothstep(-0.3, 0.0, flown) * meeting;
    // White water scatters the sun forward through itself: thin, it glows
    // with the sun behind it.
    let sun = lighting.sun.xyz;
    var through = vec3<f32>(0.0);
    if sun.y > 0.0 && dot(sun, seen) < 0.0 {
        let g = FALL_PHASE_G;
        let denominator = 1.0 + g * g + 2.0 * g * dot(sun, view);
        let phase = (1.0 - g * g) / (denominator * sqrt(denominator));
        through = lighting.sun_color.rgb * (FALL_ALBEDO * white * FALL_THROUGH * phase
            * (1.0 - 0.5 * near_cover)) * sun_visibility(world, -seen, world_dx, world_dy,
            length(input.relative), false);
    }
    // Clear water shows the light its body scatters back.
    let body = water.color.rgb * sky_average(vec3<f32>(0.0, 1.0, 0.0)) * lighting.zenith.w
        * (0.5 * (1.0 - white));
    var point: SurfacePoint;
    point.albedo = vec3<f32>(FALL_ALBEDO);
    point.opacity = 1.0;
    point.emission = 0.0;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = mix(1.0, 0.8, broken);
    point.bounce = vec3<f32>(0.0);
    point.roughness = mix(0.08, 0.8, white);
    point.alpha = point.roughness * point.roughness;
    point.f0 = vec3<f32>(0.02);
    point.diffuse_color = vec3<f32>(FALL_ALBEDO * white);
    point.geometric = seen;
    point.normal = bent;
    point.distance = length(input.relative);
    point.view = view;
    point.n_v = max(dot(bent, view), 1.0e-4);
    point.world = world;
    point.world_dx = world_dx;
    point.world_dy = world_dy;
    point.shadow_offset = vec3<f32>(0.0);
    point.on_terrain = false;
    point.sun_shade = 0.0;
    var shade: FallShade;
    shade.shaded = finish(point, body + through);
    // The far side's water shows darker in the near side's shade.
    let shown = near_cover + far_cover * (1.0 - near_cover) * FALL_FAR_SHADE;
    shade.shaded.color *= shown / max(cover, 1.0e-4);
    shade.coverage = coverage;
    if coverage < 0.01 {
        discard;
    }
    return shade;
}

// Falls blend premultiplied over what they fall past, as water does.
@fragment
fn fs_fall(input: FallOutput) -> @location(0) vec4<f32> {
    let shade = shade_fall(input);
    return vec4<f32>(encoded(shade.shaded.color) * shade.coverage, shade.coverage);
}

@fragment
fn fs_fall_post(input: FallOutput) -> SceneTargets {
    let shade = shade_fall(input);
    var targets = scene_targets(shade.shaded);
    targets.color = vec4<f32>(shade.shaded.color * shade.coverage, shade.coverage);
    return targets;
}

@fragment
fn fs_fall_temporal(input: FallTemporalOutput) -> TemporalTargets {
    let shade = shade_fall(fall_temporal_output(input));
    var targets = temporal_targets(shade.shaded, input.clip_position, input.previous);
    targets.color = vec4<f32>(shade.shaded.color * shade.coverage, shade.coverage);
    return targets;
}
