// Water: terrains' water as their tiles hold it, drawn on the leaves of a
// subdivision of its own over each terrain's square (`terrain/surface.rs`),
// which splits its leaves by how far they stray from the water's mesh and
// collapses those its surface cannot reach (`terrain_subdivision.wgsl`).
// Each vertex stands at the water's mesh where it lies (`water_field`: wet
// samples' surfaces and the surface continued under the banks, carried on
// at the nearest one's level past where they end), lifted by the pixels
// the drawn ground and the water's mesh may each stray from their surfaces
// seen from the detail eye (`water_lift`), so the ground drawn under its
// surface never hides it. Each pixel then reads the water's state where it
// lies (`water_pixel`), at the level of the tiles' pyramid the size a pixel
// resolves on it seen from the detail eye picks (`detail_across`: its span
// across the view, which its span along it never passes DETAIL_ANISOTROPY
// times), handing over to the next coarser one as detail levels do
// (`detail_level`): the smooth surface through that level's samples holding
// one (the uniform cubic B-spline whose control points they are, as the
// ground's shading takes its smooth surface), whose gradient tilts the
// water; its depth over that level's own ground and what its samples' words
// say of the water (`WaterWords`: its flow, the foam it carries, how
// turbulent it runs, where white water is born in it, its shear, how far
// across its water body is and whether a lake holds it, the reach of its
// stream and what stands in it), read through the same weights, so no
// triangle of the level shows in it; and how much of those weights its
// samples holding a surface carry, the water ending where they carry less
// than half (`water_level`). It lies wherever that surface stands over the
// ground the pixel shows, so its waterline is where the ground drawn rises
// through it, pixel by pixel, whatever shape the ground takes
// (`shade_water`). None of it moves as the eye does and no leaf shapes it.
// Where water is simulated around the camera, its cells hold the same state
// (`sim_water_output`), the span of its body and what its bed holds taken
// from the settled water's sample nearest each. Everything drawn of the
// water reads that state and what follows from it: its Froude number and
// its eddies' speed (`water_eddies`); the water's debug views paint each of
// its fields (`meshes/debug.wgsl`).
// Its waves come from that state and the bed it flows over
// (`water_flow.wgsl`): the bed's stones, read at points of the world as the
// ground draws them, raise the surface over them and leave wedges of waves
// behind them, as what stands in it does, by the water's speed and depth
// there, its eddies roughening it unseen. Where none rise, it lies a calm
// mirror. They are drawn through the pixel's own
// footprint (`water_footprint.wgsl`), every computed part while it spans two
// pixels along the way it varies (`water_shown`), so waves running across
// the view stay drawn far along it, while those running along it average
// out where the footprint spans them. What the pixel averages of its
// slopes, and how far those drawn turn across it, roughens the surface
// along the view and across it, as the ellipse of those slopes projects
// into the view, so far water streaks and glints along the view. It
// mirrors the sky, the sun's disc and,
// through screen-space reflections, the scene, drawn out along the view
// where it grazes the water; its body, tinted by the
// sediment it carries, scatters the light entering it back toward the
// eye; and what lies under it shows through, bent by the waves where the
// frame copies the scene's color, brighter where the waves gather the sun's
// light on it, and dimmed channel by channel over the light's path down to
// it and back, the ground darker for being wet. Its foam is its own: what
// its flow churns up and carries, settling or simulated
// (`engine_terrain::shallow_water`) and what it carries off its bed's stones
// downstream, and white water where it breaks, its crests fold or it runs
// fast against those stones and what stands in it, all one bubble texture
// carried as the water moves (`meshes/water_foam.wgsl`). However
// thin, the water fades only at its waterline. The sea (`meshes/ocean.wgsl`)
// lights its surface the same way. Where rivers leave their beds, their
// jets fall as sheets of their own (`water_falls.rs`), drawn last.

// Tiles' water, a block of samples for each patch holding any surface
// (`terrain/water_blocks.rs`), its own and a sample more past each of its
// edges, the tile's beside it, so the smooth surface through them
// (`water_level`) reaches across patches and tiles: their surfaces as
// water's pass draws them (terrain-local meters, NO_WATER_SURFACE where none
// reaches or the water ends before it) beside their tile's ground (x and
// y); and their words
// (`engine_terrain::TileWater::info`), a texel's channels:
// their flow along x and z as shares of WATER_FLOW_RANGE either way, the
// foam they carry as a share of WATER_FOAM_MOST (negative where a fall's
// sheet runs over them), and how far across their water body is, its code
// in 127ths (`water_span`: none for 0), negative where a lake holds them
// (x, four signed bytes); how turbulent they run, how strongly they break
// white and their shear as a share of WATER_SHEAR_RANGE, in 255ths, and 0
// for no reach, else their reach plus one (y, four bytes); the offset to
// their nearest obstacle's middle along x and z in signed steps of
// WATER_OBSTACLE_STEP, its radius in steps of WATER_RADIUS_STEP (0 for none)
// and how strongly a drop or a jump lies just upstream, in 255ths (z, four
// bytes); and the height above the terrain's origin the water's mesh stands
// at over them, a float's bits (w, `water_field`).
@group(1) @binding(50) var water_surface: texture_2d_array<f32>;
@group(1) @binding(51) var water_info: texture_2d_array<u32>;
// Water's directory (`terrain/water_blocks.rs`) lies in `terrain_directory`
// from `terrain.water.x`, after the directory's own words:
// WATER_TILE_WORDS words for each layer of the terrain's height array, the
// block of each patch of the tile in it, row by row (NO_WATER_BLOCK where
// none of its samples holds a surface). The ground's fragments read the
// water's surfaces through it too, where the ground wets at its waterline
// (`meshes/terrain.wgsl`'s `terrain_waterline`).
// The subdivision of the terrain's water (`terrain/surface.rs`), whose
// leaves its draws take, a meshlet an instance, from `tree.z` of the frame's
// (`terrain_surface_leaves`).
@group(1) @binding(60) var<uniform> water_tree: TerrainSubdivision;

// Water, or the sea: the camera's view and projection without its
// translation (the sea's; terrains' water draws through the terrain's
// view); the color deep water scatters back (rgb) and the meters through
// which half the light passes (w); for the sea, the compression below which
// its crests whiten (x), the amplitude of one wave holding all its waves'
// variance, which runs up its shores (y), and its waves' summed height (w),
// none of it for terrains' water (`foam`); and how fast the water takes
// each channel of light, per meter (rgb).
// The sea's parts (`meshes/ocean.wgsl`): its wind's heading
// (`origin`, xy: a unit vector), its still level less the eye, its grid's
// finest cell, cells around and rings a cell apart; the rings' growth, the
// meters it reaches, cells around a sector and its waves; each wave's
// vector (xy, radians per meter), height and phase at the eye; and how far
// each wave's points move along it, four to a vector. For terrains' water,
// `ocean` holds the sea's level less the eye and whether it is on, below
// which their water gives way to it. Where water is simulated around the
// camera (`water_sim.rs`), `sim` holds its square's first corner along x
// and z, the meters across a cell and the cells along a side (none, zero),
// and `sim_datum` the height its heights stand above (x, world meters) and
// the blocks its blocks' slots turn by along x and z (yz).
// `sediment` holds the color the fine sediment terrains' water carries
// scatters back (rgb, linear) and how strongly it scatters light, per meter
// (w); the sea carries none. `lift` holds the pixels the drawn ground (x,
// `TerrainSettings::subdivision_error`) and terrains' water's mesh (y,
// `Water::subdivision_error`) may each stray from their surfaces seen from
// the detail eye, by which that mesh stands over its surface
// (`water_lift`); the sea's are none.
struct WaterUniform {
    relative_view_projection: mat4x4<f32>,
    origin: vec4<f32>,
    color: vec4<f32>,
    foam: vec4<f32>,
    absorption: vec4<f32>,
    ocean: vec4<f32>,
    ocean_grid: vec4<f32>,
    ocean_waves: array<vec4<f32>, 16>,
    ocean_reach: array<vec4<f32>, 4>,
    sim: vec4<f32>,
    sim_datum: vec4<f32>,
    sediment: vec4<f32>,
    lift: vec4<f32>,
}
@group(1) @binding(52) var<uniform> water: WaterUniform;
// The simulated water as its blocks hold it (`water_sim.wgsl`'s, every
// block of the window's square in its slot, SIM_WATER_BLOCK cells along each
// side): each cell's surface (world meters), its flow along x and z (square
// meters a second) and the foam it carries, a block's row by row; and each
// block's ground, SIM_BLOCK_GROUND entries of it: at its cells' corners (a
// row of its cells and one more at a time), then the levels lakes hold its
// cells at, its cells' beds' roughness and what solids cover of them; and
// what the bed holds under
// each cell, in the cells' order, as the settled water's sample nearest it
// holds it (`engine_terrain`'s `Tile::bed_words`): its reach's code (bits 0
// to 7) beside its span's code as its words' fourth byte packs it (bits 8
// to 15), and its nearest obstacle, offset from the cell's middle, as
// tiles' words pack them.
@group(1) @binding(56) var<storage, read> water_sim_cells: array<vec4<f32>>;
@group(1) @binding(57) var<storage, read> water_sim_ground: array<f32>;
@group(1) @binding(58) var<storage, read> water_sim_beds: array<vec2<u32>>;
const SIM_WATER_BLOCK: u32 = 8u;
const SIM_BLOCK_GROUND: u32 = 337u;
// The most foam water carries, as its flow churns it up
// (`engine_terrain::shallow_water::FOAM_MOST`).
const WATER_FOAM_MOST: f32 = 1.5;
// What lies behind the water, copied from the opaque scene before it
// draws: its depth and, in frames drawing the effects' targets, its color.
@group(2) @binding(22) var water_depth: texture_depth_2d;
@group(2) @binding(23) var water_scene: texture_2d<f32>;

// The opaque surface behind the water at `pixel` (its center), less the
// eye, without the jitter of temporally antialiased frames: a pixel of the
// frame's render size, which it draws at.
fn water_ground(pixel: vec2<f32>) -> vec3<f32> {
    let size = camera.region.xy;
    let depth = textureLoad(water_depth, vec2<i32>(pixel), 0);
    let ndc = vec2<f32>(pixel.x / size.x * 2.0 - 1.0, 1.0 - pixel.y / size.y * 2.0)
        - camera.jitter.xy;
    let point = camera.inverse_view_projection * vec4<f32>(ndc, depth, 1.0);
    return point.xyz / point.w - camera.position.xyz;
}

// Samples along a block's side (its patch's grid and a sample more each
// side) and blocks along a layer's side, no block, and what a sample no
// water reaches holds; patches along a side of a tile, the words each layer
// takes in water's directory, and a terrain directory's entry of a tile not
// resident (`gpu_layouts::constants`).
const WATER_BLOCK: u32 = 19u;
const WATER_BLOCKS_SIDE: u32 = 16u;
const NO_WATER_BLOCK: u32 = 0xffffffffu;
const NO_WATER_SURFACE: f32 = -1.0e30;
const WATER_TILE_PATCHES: u32 = 16u;
const WATER_TILE_WORDS: u32 = 256u;
const NOT_RESIDENT: u32 = 0xffffffffu;
// Meters a second either way the packed flow spans, the narrowest and the
// widest span the packed span tells apart (meters across a water body), the
// shear (per second) the packed shear spans, and the meters a step of the
// packed offset to an obstacle and of its packed radius span
// (`engine_terrain::WATER_FLOW_RANGE`, `WATER_SPAN_LEAST`,
// `WATER_SPAN_MOST`, `WATER_SHEAR_RANGE`, `WATER_OBSTACLE_STEP`,
// `WATER_RADIUS_STEP`); and the steepest a bank rises, meters a meter, past
// which it stands as a wall (`gpu_layouts::constants::WATER_GRADIENT_RANGE`).
const WATER_FLOW_RANGE: f32 = 8.0;
const WATER_SPAN_LEAST: f32 = 0.25;
const WATER_SPAN_MOST: f32 = 4096.0;
const WATER_SHEAR_RANGE: f32 = 8.0;
const WATER_OBSTACLE_STEP: f32 = 0.125;
const WATER_RADIUS_STEP: f32 = 0.0625;
const WATER_GRADIENT_RANGE: f32 = 4.0;
// How much darker the ground under water looks: of the light it scatters
// back, the surface turns part back down; and the meters of water, a few
// centimeters, that darken it as much.
const WATER_WET: f32 = 0.35;
const WATER_SOAKED: f32 = 0.03;
// Water's index of refraction over air's.
const WATER_IOR: f32 = 1.333;
// How strongly water's body scatters light forward (Henyey and
// Greenstein's g), and its phase's mean over the backward half, where light
// entering the surface turns back toward an eye above it.
const WATER_PHASE_G: f32 = 0.4;
const WATER_PHASE_MEAN: f32 = 0.4498;
// How much light foam scatters back, and the share of the sky water in its
// channel sees; and foam's roughness where it lies a single layer of wet
// bubbles thin, and where it heaps.
const WATER_FOAM_ALBEDO: f32 = 0.65;
const WATER_FOAM_SKY: f32 = 0.6;
const WATER_FOAM_ROUGH: vec2<f32> = vec2<f32>(0.3, 0.7);
// Meters the bent view runs at most to what it shows: the water hides what
// lies farther.
const WATER_REACH: f32 = 4.0;
// Meters over which the sea fades in from its shores and around what
// stands in it.
const WATER_SHORE: f32 = 0.1;
// The pixels over which terrains' water fades in from its waterline, where
// its surface comes down to what lies behind the pixel or the samples
// holding it end, and the meters of water over what lies behind the pixel
// past which it is all there however fast that changes across a pixel:
// running water, however thin, is water.
const WATER_EDGE_PIXELS: f32 = 1.5;
const WATER_EDGE: f32 = 0.05;
// The share of the smooth surface's weights its samples holding a surface
// carry where the water ends (`water_level`): halfway from the last sample
// holding one to the first holding none, as the rings continuing the
// water's surface end it halfway to a sample they stop at
// (`engine_terrain::TileWater`).
const WATER_HELD: f32 = 0.5;
// The meters a second over which water goes from still to running, from the
// first to the second (`shade_water`'s `flowing`): its eddies roughen only
// running water.
const WATER_FLOWING: vec2<f32> = vec2<f32>(0.05, 0.3);
// The meters a second over which thin water comes to show whole however
// thin, from the first to the second: calm shallows (a river's water
// spread over its banks, a pool's edge) fade in over WATER_SHORE of depth
// as still water's edges do, while water racing thinner than that (a
// torrent's sheet over its bed) is all there.
const WATER_RACING: vec2<f32> = vec2<f32>(0.8, 1.6);
// Meters a second squared of gravity (`engine_terrain::hydraulics::GRAVITY`),
// and the meters of depth Froude numbers read thinner water as.
const WATER_GRAVITY: f32 = 9.81;
const WATER_THINNEST: f32 = 0.01;

// Tiles' water's vertex: where it draws, and its point of the water's mesh
// less the eye, where its pixels read the water.
struct WaterOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
}

// Tiles' water's vertex in the temporal pipelines: also where its point
// stood in the previous frame's view (x, y and w of its clip position).
struct WaterTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    @location(1) previous: vec3<f32>,
}

// The texel of block `block` holding its sample `sample` (x and y, counted
// from the block's first, a sample before its patch's first), and its
// layer (z).
fn water_texel(block: u32, sample: vec2<u32>) -> vec3<i32> {
    let blocks = WATER_BLOCKS_SIDE * WATER_BLOCKS_SIDE;
    let index = block % blocks;
    let origin = vec2<u32>(index % WATER_BLOCKS_SIDE, index / WATER_BLOCKS_SIDE) * WATER_BLOCK;
    return vec3<i32>(vec2<i32>(origin + sample), i32(block / blocks));
}

// What a sample's words say of its water: its flow along x and z (meters a
// second), the foam it carries (0 none, WATER_FOAM_MOST the most), how far
// across its water body is (its span, meters: a river's width, a lake's
// equal-area diameter; none where none is known), whether a lake holds it
// and whether a fall's sheet runs over it; how turbulent it runs, how
// strongly it breaks white and its shear (a second); its reach's code (0
// for none, else 1 plus `engine_terrain::Reach`); and its nearest
// obstacle's word (`water_obstacle`).
struct WaterWords {
    flow: vec2<f32>,
    foam: f32,
    span: f32,
    lake: bool,
    sheet: bool,
    state: vec3<f32>,
    reach: u32,
    obstacle: u32,
}

// What a sample's words `words` (its texel of `water_info`) say of its
// water.
fn water_words(words: vec4<u32>) -> WaterWords {
    let info = unpack4x8snorm(words.x);
    let state = unpack4x8unorm(words.y);
    var read: WaterWords;
    read.flow = info.xy * WATER_FLOW_RANGE;
    read.foam = abs(info.z) * WATER_FOAM_MOST;
    read.span = water_span(info.w);
    read.lake = info.w < 0.0;
    read.sheet = info.z < 0.0;
    read.state = vec3<f32>(state.xy, state.z * WATER_SHEAR_RANGE);
    read.reach = words.y >> 24u;
    read.obstacle = words.z;
    return read;
}

// The meters across a water body the span's code `code` says (its signed
// byte as a share of 127, `engine_terrain`'s `pack`), whatever its sign: a
// step of it WATER_SPAN_LEAST, 127 steps WATER_SPAN_MOST, on a logarithmic
// scale between them; none for none.
fn water_span(code: f32) -> f32 {
    let steps = round(abs(code) * 127.0);
    let octaves = log2(WATER_SPAN_MOST / WATER_SPAN_LEAST);
    return select(WATER_SPAN_LEAST * exp2(octaves * (steps - 1.0) / 126.0), 0.0, steps < 0.5);
}

// The terrain's extent along x and z (meters): its level-0 tiles'.
fn water_extent() -> vec2<f32> {
    let header = terrain.directory.x;
    let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
    return vec2<f32>(tiles * (WATER_TILE_PATCHES * (TERRAIN_GRID - 1u))) * terrain.anchor.w;
}

// Water at a point as shading reads it: the height above the terrain's
// origin of its smooth surface through its samples holding one
// (`water_level`), and its depth over the ground there, its samples' read
// through the same weights (none where they hold none); how much of those
// weights its samples holding a surface carry (`held`: 1 where all of them
// hold one, the water ending where they carry less than WATER_HELD); what
// its samples' words say of it, what changes by degrees (its span among
// it) read through the smooth surface's weights and what names a kind (a
// lake holding it, a fall's sheet over it, its reach, its nearest
// obstacle's word) its nearest sample's; how fast its smooth surface falls
// along x and z (meters a meter); and its nearest obstacle as
// `water_obstacle` gives it, offset from the point, how strongly a drop or
// a jump lies just upstream read through those weights.
struct WaterAt {
    surface: f32,
    depth: f32,
    held: f32,
    words: WaterWords,
    slope: vec2<f32>,
    obstacle: vec4<f32>,
}

// Where terrain-local `local` lies in the tiles' water of the finest level
// at or coarser than `level` whose tile holding it is resident: that level
// (the terrain's levels where none is, or the point lies past the terrain),
// the block of the patch holding the point's cell (NO_WATER_BLOCK where
// none of the patch's samples holds a surface, or no level is resident),
// the cell's first corner among the block's samples (`water_texel`), and
// where in the cell the point lies (0 to 1 along x and z).
struct WaterCell {
    level: u32,
    block: u32,
    first: vec2<u32>,
    within: vec2<f32>,
}

fn water_cell(local: vec2<f32>, level: u32) -> WaterCell {
    let levels = terrain.directory.y;
    var cell: WaterCell;
    cell.level = levels;
    cell.block = NO_WATER_BLOCK;
    if any(local < vec2<f32>(0.0)) || any(local > water_extent()) {
        return cell;
    }
    let base = terrain.directory.x;
    let quads = TERRAIN_GRID - 1u;
    let tile_quads = WATER_TILE_PATCHES * quads;
    for (var index = level; index < levels; index++) {
        let header = base + index * 4u;
        let samples = local / (terrain.anchor.w * f32(1u << index));
        let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
        let tile = min(vec2<u32>(samples / f32(tile_quads)), tiles - vec2<u32>(1u));
        let layer = terrain_directory[terrain_directory[header] + tile.y * tiles.x + tile.x];
        if layer == NOT_RESIDENT {
            continue;
        }
        // Samples 0 to 256 within the tile, the cell holding the point and
        // the patch holding the cell, whose first sample is its block's
        // second.
        let within = clamp(samples - vec2<f32>(tile * tile_quads), vec2<f32>(0.0),
            vec2<f32>(f32(tile_quads)));
        let corner = min(vec2<u32>(within), vec2<u32>(tile_quads - 1u));
        let tile_patch = corner / quads;
        cell.level = index;
        cell.block = terrain_directory[terrain.water.x + layer * WATER_TILE_WORDS
            + tile_patch.y * WATER_TILE_PATCHES + tile_patch.x];
        cell.first = corner - tile_patch * quads + vec2<u32>(1u);
        cell.within = within - vec2<f32>(corner);
        return cell;
    }
    return cell;
}

// Corner `corner` of the triangle of a cell holding a point, its upper one
// (`upper`: past the diagonal from the cell's (x, z+1) corner to its (x+1,
// z) one, as `terrain_level_height` splits cells) or its lower one: its
// steps from the cell's first corner, in the order `water_weights` weighs
// them.
fn water_corner(corner: u32, upper: bool) -> vec2<u32> {
    let lower = vec2<u32>(select(0u, 1u, corner == 1u), select(0u, 1u, corner == 2u));
    return select(lower, vec2<u32>(1u) - lower, upper);
}

// The weights a point `within` its cell (0 to 1 along x and z) takes of the
// corners of the triangle holding it (`water_corner`): the upper one where
// `within.x + within.y` passes 1.
fn water_weights(within: vec2<f32>) -> vec3<f32> {
    if within.x + within.y <= 1.0 {
        return vec3<f32>(1.0 - within.x - within.y, within.x, within.y);
    }
    return vec3<f32>(within.x + within.y - 1.0, 1.0 - within.x, 1.0 - within.y);
}

// The height above the terrain's origin the water's mesh stands at at
// terrain-local `local`: on the triangle holding the point among the finest
// resident level's samples, each at the level the mesh stands at over it
// (`engine_terrain::WaterMesh`: its surface drawn where it holds one, else
// the surface of the sample holding one nearest it), as the error pyramid
// water's subdivision splits by bounds it (`water_errors`); on the ground
// where the patch holding the point takes no block, as the mesh does over
// it. So the mesh stands at the water's level wherever its surface reaches,
// and each pixel decides where water shows. It depends on the point alone,
// so leaves sharing a vertex meet on it.
fn water_field(local: vec2<f32>) -> f32 {
    let cell = water_cell(local, 0u);
    if cell.block == NO_WATER_BLOCK {
        return terrain_level_height(local, cell.level).height;
    }
    let upper = cell.within.x + cell.within.y > 1.0;
    var levels = vec3<f32>(0.0);
    for (var corner = 0u; corner < 3u; corner++) {
        let texel = water_texel(cell.block, cell.first + water_corner(corner, upper));
        levels[corner] = bitcast<f32>(textureLoad(water_info, texel.xy, texel.z, 0).w);
    }
    return dot(levels, water_weights(cell.within));
}

// The weights the uniform cubic B-spline gives four samples a step apart
// along an axis at a point `t` (0 to 1) of the way from the second to the
// third (`weights`, the first's to the fourth's), and how fast each changes
// a step (`rates`). They sum to 1 and change smoothly to their second
// derivative, a sample's weight coming to nothing with its rate and its
// curvature at the end of its reach, two steps from it.
struct WaterSpline {
    weights: vec4<f32>,
    rates: vec4<f32>,
}

fn water_spline(t: f32) -> WaterSpline {
    let s = 1.0 - t;
    var spline: WaterSpline;
    spline.weights = vec4<f32>(s * s * s, (3.0 * t - 6.0) * t * t + 4.0,
        ((3.0 - 3.0 * t) * t + 3.0) * t + 1.0, t * t * t) / 6.0;
    spline.rates = vec4<f32>(-s * s, (3.0 * t - 4.0) * t, (2.0 - 3.0 * t) * t + 1.0, t * t)
        * 0.5;
    return spline;
}

// The smooth surface through the samples around a point of `cell`
// (`water_cell`'s), as water's pass and the ground's waterline read it: the
// uniform cubic B-spline whose control points are the surfaces of the 4 x 4
// samples around the point (`water_spline`), as the ground's shading takes
// its smooth surface (`engine_terrain::SampleShape`), through those holding
// one, wet or continued under the banks at the water's level, their weights
// shared out over them, so it stays level up to where the samples holding
// one end and no sample the water ends before pulls it down
// (`terrain/water_blocks.rs`). Its height above the terrain's origin and how
// fast it falls along x and z (meters a meter); the samples' depths through
// the same weights, each one's surface over its level's own ground (so
// coarse levels' channels, cut to their lowest, stand as deep as they hold
// them), none or more; how much of the weights the samples holding a
// surface carry (`held`, none where the cell holds no block: the water ends
// where it falls under WATER_HELD) and how fast that grows along x and z (a
// meter); and each sample's share of the surface (`shares[row][column]`,
// counted from the first of the 4 x 4 along z and x), none for those
// holding none.
struct WaterLevel {
    surface: f32,
    slope: vec2<f32>,
    depth: f32,
    held: f32,
    held_rate: vec2<f32>,
    shares: mat4x4<f32>,
}

fn water_level(cell: WaterCell) -> WaterLevel {
    var level: WaterLevel;
    if cell.block == NO_WATER_BLOCK {
        return level;
    }
    // The texel of the first of the 4 x 4 samples around the point, a sample
    // before the cell's first corner along x and z; their weights along x and
    // z; and the surface of the first holding one, from which the smooth
    // surface's sums count so they keep their precision.
    let origin = water_texel(cell.block, cell.first - vec2<u32>(1u));
    let along = water_spline(cell.within.x);
    let across = water_spline(cell.within.y);
    var counted = 0.0;
    var counting = false;
    // Sums over the samples holding a surface: their weights, their surfaces
    // over the counted one and their depths weighed, and the rates of each
    // along x and z.
    var total = 0.0;
    var raised = 0.0;
    var deep = 0.0;
    var rates = vec2<f32>(0.0);
    var rises = vec2<f32>(0.0);
    for (var row = 0u; row < 4u; row++) {
        for (var column = 0u; column < 4u; column++) {
            let steps = vec2<i32>(vec2<u32>(column, row));
            let held = textureLoad(water_surface, origin.xy + steps, origin.z, 0).xy;
            if held.x <= NO_WATER_SURFACE * 0.5 {
                continue;
            }
            if !counting {
                counted = held.x;
                counting = true;
            }
            let weight = along.weights[column] * across.weights[row];
            let rate = vec2<f32>(along.rates[column] * across.weights[row],
                along.weights[column] * across.rates[row]);
            total += weight;
            raised += (held.x - counted) * weight;
            deep += (held.x - held.y) * weight;
            rates += rate;
            rises += (held.x - counted) * rate;
            level.shares[row][column] = weight;
        }
    }
    if total <= 0.0 {
        return level;
    }
    // Each sum shared out over the weights the samples holding a surface
    // take; the surface's gradient that of the quotient, per meter.
    let spacing = terrain.anchor.w * f32(1u << cell.level);
    let share = 1.0 / total;
    let lift = raised * share;
    level.surface = counted + lift;
    level.slope = (rates * lift - rises) * (share / spacing);
    level.depth = max(deep * share, 0.0);
    level.held = total;
    level.held_rate = rates / spacing;
    level.shares = level.shares * share;
    return level;
}

// Tiles' water at a point as `cell` (`water_cell`'s) holds it: its smooth
// surface, how fast it falls, its depth and how much of it the samples
// holding a surface hold (`water_level`); through the same shares, what
// their words say that changes by degrees, so nothing changes at a
// triangle's edge; and what names a kind and the nearest obstacle, its
// offset taken from the point, the nearest sample's, none where that one
// holds no surface.
fn water_read(cell: WaterCell) -> WaterAt {
    var at: WaterAt;
    var level = water_level(cell);
    at.surface = level.surface;
    at.slope = level.slope;
    at.depth = level.depth;
    at.held = level.held;
    if level.held <= 0.0 {
        return at;
    }
    let origin = water_texel(cell.block, cell.first - vec2<u32>(1u));
    let nearest = select(vec2<u32>(0u), vec2<u32>(1u), cell.within >= vec2<f32>(0.5));
    let spacing = terrain.anchor.w * f32(1u << cell.level);
    for (var row = 0u; row < 4u; row++) {
        for (var column = 0u; column < 4u; column++) {
            let share = level.shares[row][column];
            if share <= 0.0 {
                continue;
            }
            let steps = vec2<u32>(column, row);
            let words = water_words(textureLoad(water_info, origin.xy + vec2<i32>(steps),
                origin.z, 0));
            at.words.flow += words.flow * share;
            at.words.foam += words.foam * share;
            at.words.span += words.span * share;
            at.words.state += words.state * share;
            at.obstacle.w += f32(words.obstacle >> 24u) / 255.0 * share;
            if all(steps == nearest + vec2<u32>(1u)) {
                // The obstacle's offset from the point: the sample's own,
                // plus the sample's from the point.
                let toward = vec2<f32>(nearest) - cell.within;
                at.words.lake = words.lake;
                at.words.sheet = words.sheet;
                at.words.reach = words.reach;
                at.words.obstacle = words.obstacle;
                at.obstacle = vec4<f32>(water_obstacle(words.obstacle, toward * spacing).xyz,
                    at.obstacle.w);
            }
        }
    }
    return at;
}

// Water `first` blended `toward` water `second`, both as `WaterAt` holds
// them at one point: what changes by degrees mixed, what names a kind and
// the nearest obstacle the one the blend leans to.
fn water_blend(first: WaterAt, second: WaterAt, toward: f32) -> WaterAt {
    var at = first;
    if toward >= 0.5 {
        at = second;
    }
    at.surface = mix(first.surface, second.surface, toward);
    at.depth = mix(first.depth, second.depth, toward);
    at.held = mix(first.held, second.held, toward);
    at.words.flow = mix(first.words.flow, second.words.flow, toward);
    at.words.foam = mix(first.words.foam, second.words.foam, toward);
    at.words.span = mix(first.words.span, second.words.span, toward);
    at.words.state = mix(first.words.state, second.words.state, toward);
    at.slope = mix(first.slope, second.slope, toward);
    at.obstacle.w = mix(first.obstacle.w, second.obstacle.w, toward);
    return at;
}

// Tiles' water at terrain-local `local` as a pixel resolving `seen` meters
// there (`detail_resolved`), seen from the detail eye, reads it: at the
// level of the tiles' pyramid that size picks over level 0's spacing
// (`detail_level`), the next coarser one blended in over the upper half of
// each octave, each from the finest resident level at or coarser than its
// own (`water_cell`), and read once where both fall in the same tile.
fn water_pixel(local: vec2<f32>, seen: f32) -> WaterAt {
    let picked = detail_level(seen, terrain.anchor.w, terrain.directory.y);
    let cell = water_cell(local, picked.level);
    let at = water_read(cell);
    if picked.blend <= 0.0 || cell.level != picked.level {
        return at;
    }
    return water_blend(at, water_read(water_cell(local, picked.level + 1u)), picked.blend);
}

// Tiles' water at the pixel drawing the point `drawn` of its mesh (less the
// eye): read where the point lies as the pixel, seen from the detail eye,
// resolves it there (`water_pixel`): the meters it spans across the view
// (`detail_across`), as its footprint on any surface never stretches past
// DETAIL_ANISOTROPY times that along it (`detail_span`, `detail_resolved`).
fn water_drawn(drawn: vec3<f32>) -> WaterAt {
    let world = eye_world(drawn);
    return water_pixel(world.xz - terrain.origin.xz, detail_across(world));
}

// The point of the water's surface under the point `drawn` of its mesh
// (both less the eye), as high over the terrain's origin as `at`
// (`water_drawn`'s) holds the surface there.
fn water_on_surface(drawn: vec3<f32>, at: WaterAt) -> vec3<f32> {
    let above = eye_world(drawn).y - terrain.origin.y - at.surface;
    return drawn - vec3<f32>(0.0, above, 0.0);
}

// The point at terrain-local `local` standing `height` meters above the
// terrain's origin, less the eye of the view drawing it.
fn water_relative(local: vec2<f32>, height: f32) -> vec3<f32> {
    let from_eye = terrain.origin.xz - terrain.eye.xz;
    return vec3<f32>(local.x + from_eye.x, height + terrain.anchor.y, local.y + from_eye.y)
        + terrain_view.eye_offset.xyz;
}

// The meters terrains' water's mesh stands over its surface at `world`: the
// pixels the drawn ground and the mesh may each stray from their surfaces
// (`water.lift`), as wide as a pixel seen from the detail eye spans there
// (`detail_across`). The ground drawn under the water's surface then never
// hides it, its pixels deciding where the water lies (`shade_water`), and
// the mesh stands no more than those pixels off the water.
fn water_lift(world: vec3<f32>) -> f32 {
    return (water.lift.x + water.lift.y) * detail_across(world);
}

// Lattice vertex `vertex` of the leaf at `instance` of the subdivision of
// the terrain's water (`water_tree`): the point of the leaf's meshlet
// (`surface_vertex`) at the height the water's mesh stands at there
// (`water_field`), lifted over it (`water_lift`).
fn water_output(vertex: u32, instance: u32) -> WaterOutput {
    let local = surface_vertex(terrain_surface_leaves[water_tree.tree.z + instance], vertex);
    var relative = water_relative(local, water_field(local));
    relative.y += water_lift(eye_world(relative));
    var output: WaterOutput;
    output.clip_position = terrain_view.relative_view_projection * vec4<f32>(relative, 1.0);
    output.relative = relative;
    return output;
}

// The nearest obstacle a word packs (`engine_terrain`'s `pack_obstacle`):
// the offset to its middle along x and z from the point, its sample's
// offset plus `slide`, none without one; its radius (0 for none); and how
// strongly a drop or a jump lies just upstream.
fn water_obstacle(word: u32, slide: vec2<f32>) -> vec4<f32> {
    let radius = f32(extractBits(word, 16u, 8u)) * WATER_RADIUS_STEP;
    let toward = vec2<f32>(f32(extractBits(bitcast<i32>(word), 0u, 8u)),
        f32(extractBits(bitcast<i32>(word), 8u, 8u))) * WATER_OBSTACLE_STEP + slide;
    return vec4<f32>(select(vec2<f32>(0.0), toward, radius > 0.0), radius,
        f32(word >> 24u) / 255.0);
}

@vertex
fn vs_water(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> WaterOutput {
    return water_output(vertex, instance);
}

// Water's motion is the camera's: its waves and foam move within its surface.
@vertex
fn vs_water_temporal(@builtin(vertex_index) vertex: u32,
        @builtin(instance_index) instance: u32) -> WaterTemporalOutput {
    let output = water_output(vertex, instance);
    var temporal: WaterTemporalOutput;
    temporal.clip_position = output.clip_position;
    temporal.relative = output.relative;
    temporal.previous = (camera.previous_view_projection
        * vec4<f32>(eye_world(output.relative), 1.0)).xyw;
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

// What scatters light in water's body besides the water. The fine sediment
// it carries (`water.sediment`: silt, or the glacial flour that turns
// mountain water turquoise), its grains microns across, larger than light's
// wavelengths, scatters every channel about alike (Mie), w a meter, and
// sends back the color rgb. Bubbles, where the water churns, are larger
// still, and each takes twice its cross section out of the light (the
// extinction paradox): air making up a share ε of the water in bubbles of
// radius r scatters 3ε/2r a meter. Where water churns the most it scatters
// WATER_BUBBLES, a thousandth of it air in bubbles half a millimeter in
// radius (breaking water's are mostly under a millimeter, Deane and
// Stokes). Bubbled water sends back WATER_BUBBLE_ALBEDO of the light, alike
// in every channel, so it turns milky and paler: about what whitecaps send
// back averaged over their lives, 22 %, where their fresh foam sends back
// 55 % (Koepke).
const WATER_BUBBLES: f32 = 3.0;
const WATER_BUBBLE_ALBEDO: f32 = 0.2;
// The meters of water the sun's light crosses through a crest for each
// meter of its waves' root mean square height: a crest at Stokes's limit, a
// seventh as high as it is long, stands L/π wide on average above the
// still level, 7·2√2/π ≈ 6.3 root mean square heights of a sine.
const WATER_CREST_WALL: f32 = 6.3;

// How fast light dies in water `aerated` (0 to 1) with the bubbles churning
// water holds, each channel per meter: what the water takes, and what its
// sediment and bubbles scatter.
fn water_extinction(aerated: f32) -> vec3<f32> {
    return water.absorption.rgb + water.sediment.w + WATER_BUBBLES * aerated;
}

// What light keeps of each channel over `meters` of water `aerated` (0 to
// 1) with the bubbles churning water holds (`water_extinction`).
fn water_transmitted(meters: f32, aerated: f32) -> vec3<f32> {
    return exp(-water_extinction(aerated) * meters);
}

// The color water `aerated` (0 to 1) with the bubbles churning water holds
// scatters back: its own, its sediment's and its bubbles', each by its share
// of what the water takes and scatters.
fn water_body(aerated: f32) -> vec3<f32> {
    let own = luminance(water.absorption.rgb);
    let sediment = water.sediment.w;
    let bubbles = WATER_BUBBLES * aerated;
    return (water.color.rgb * own + water.sediment.rgb * sediment
        + vec3<f32>(WATER_BUBBLE_ALBEDO * bubbles)) / max(own + sediment + bubbles, 1.0e-6);
}

// The normal that mirrors the eye's view along `view` above the horizon, so
// water however far off mirrors what stands just above it: `normal`, or,
// for slopes turned so far from the eye that they would mirror what lies
// below the horizon, the normal mirroring the view as far above it
// instead: they would see the next wave's face, turned toward them about as
// far, which mirrors the view back up. Seen from under the water, below the
// plane its surface lies in (`plane`, its normal), `normal`.
fn water_facing(normal: vec3<f32>, view: vec3<f32>, plane: vec3<f32>) -> vec3<f32> {
    let mirrored = reflect(-view, normal);
    if dot(view, plane) <= 0.0 || mirrored.y >= 0.0 {
        return normal;
    }
    return normalize(view + vec3<f32>(mirrored.x, -mirrored.y, mirrored.z));
}

// The way the eye looks along water (a unit vector along x and z, away from
// the eye): its view `view` (toward the eye) laid flat; along x where it
// looks straight down.
fn water_heading(view: vec3<f32>) -> vec2<f32> {
    let away = -view.xz;
    let squared = dot(away, away);
    let laid = away * inverseSqrt(max(squared, 1.0e-30));
    return select(vec2<f32>(1.0, 0.0), laid, squared > 1.0e-12); // optics: the view's frame on water
}

// The tangent of water's normal `normal` along the view `view` (toward the
// eye), away from the eye: where the plane of incidence meets the surface,
// along which its slopes tilt the mirrored ray up and down; across it lies
// `cross(normal, tangent)`. Any tangent where the eye looks straight along
// the normal.
fn water_view_tangent(normal: vec3<f32>, view: vec3<f32>) -> vec3<f32> {
    let laid = normal * dot(view, normal) - view;
    let squared = dot(laid, laid);
    let side = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.x) > 0.5);
    let tangent = laid * inverseSqrt(max(squared, 1.0e-30));
    let any_tangent = normalize(side - normal * dot(side, normal));
    return select(any_tangent, tangent, squared > 1.0e-12); // optics: the view's frame on water
}

// Trowbridge-Reitz (GGX) distribution of water's microfacet normals, their
// slopes spreading `alphas` along the view and across it (Burley 2012): the
// half vector's cosines with the view's tangent (`along`), the tangent
// across it (`across`) and the normal (`n_h`).
fn water_distribution(along: f32, across: f32, n_h: f32, alphas: vec2<f32>) -> f32 {
    let a2 = alphas.x * alphas.y;
    let v = vec3<f32>(alphas.y * along, alphas.x * across, a2 * n_h);
    let w2 = a2 / dot(v, v);
    return a2 * w2 * w2 / PI;
}

// Smith-GGX visibility, height-correlated (Heitz 2014), of water's
// microfacets spreading `alphas` along the view and across it: the view's
// and the light's cosines with the normal (`n_v`, `n_l`) and with the
// view's tangents along and across it (`view_tangents`, `light_tangents`).
fn water_visibility(n_v: f32, n_l: f32, view_tangents: vec2<f32>, light_tangents: vec2<f32>,
        alphas: vec2<f32>) -> f32 {
    let lambda_v = n_l * length(vec3<f32>(alphas * view_tangents, n_v));
    let lambda_l = n_v * length(vec3<f32>(alphas * light_tangents, n_l));
    return 0.5 / (lambda_v + lambda_l);
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
// surface, and the sun's `sunlight` refracted in through the plane its
// surface lies in (`plane`, its normal) and turned toward the eye, the more
// the nearer the eye looks toward the sun. Seen from under the water, the
// sky's alone.
fn water_scattered(view: vec3<f32>, sunlight: vec3<f32>, plane: vec3<f32>) -> vec3<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let sky = sky_average(up) * lighting.zenith.w;
    let sun = lighting.sun.xyz;
    let sun_in = dot(sun, plane);
    if sun_in <= 0.0 || dot(view, plane) <= 0.0 {
        return sky;
    }
    let entering = refract(-sun, plane, 1.0 / WATER_IOR);
    let leaving = -refract(-view, plane, 1.0 / WATER_IOR);
    let into = 1.0 - fresnel(vec3<f32>(0.02), sun_in).r;
    return sky + sunlight * (sun_in * into * water_phase(dot(entering, leaving)));
}

// What the eye sees through water at its point `relative` (less the eye)
// of normal `normal` along `view`, where the opaque surface behind its
// pixel `pixel` stands at `ground` (less the eye), its surface lying in the
// plane of normal `plane`: the opaque scene's color where the view, bent
// into the water, meets it (graded, as the scene holds it), and that
// surface less the eye. The bent view runs on as deep under that plane as
// the surface behind the pixel lies; what stands nearer than the water, or
// out of it, shows unbent, as does everything seen from under the water.
struct WaterBehind {
    color: vec3<f32>,
    ground: vec3<f32>,
}

fn water_behind(relative: vec3<f32>, normal: vec3<f32>, view: vec3<f32>, pixel: vec2<f32>,
        ground: vec3<f32>, plane: vec3<f32>) -> WaterBehind {
    // Places in the view among the pixels of the frame's render size, which
    // the copies hold from their top-left corner.
    let size = camera.region.xy;
    var behind: WaterBehind;
    behind.ground = ground;
    var at = pixel;
    let bent = refract(-view, normal, 1.0 / WATER_IOR);
    let sinking = -dot(bent, plane);
    if dot(view, plane) > 0.0 && sinking > 0.0 {
        let below = max(dot(relative - ground, plane), 0.0);
        let reach = min(below / max(sinking, 0.1), WATER_REACH);
        let clip = camera.view_projection
            * vec4<f32>(eye_world(relative + bent * reach), 1.0);
        if clip.w > 1.0e-6 {
            let moved = (vec2<f32>(clip.x, -clip.y) / clip.w * 0.5 + 0.5) * size;
            if all(moved >= vec2<f32>(0.0)) && all(moved < size)
                && textureLoad(water_depth, vec2<i32>(moved), 0) > 0.0 {
                let there = water_ground(moved);
                if dot(there, there) > dot(relative, relative)
                    && dot(relative - there, plane) > 0.0 {
                    behind.ground = there;
                    at = moved;
                }
            }
        }
    }
    behind.color = textureSampleLevel(water_scene, refraction_sampler,
        min(at, size - 0.5) / vec2<f32>(textureDimensions(water_depth)), 0.0).rgb;
    return behind;
}

// A water surface to light: its point less the eye and how it changes per
// pixel; its waves' normal and the roughness of what a pixel spans of them,
// along the view and across it (`water_roughness`); the normal of the plane
// it lies in without them (up for the sea, tilted as terrains' water
// falls), which the eye looks down onto and light enters through; the
// share of it foam covers; the color its body scatters
// back, and what it keeps of each channel of the light on its way to what
// lies below and back; how near the top of a crest of its waves it stands
// (0 to 1); and how many times over each channel of light dies by e
// crossing the walls of its crests (`water_extinction` over
// WATER_CREST_WALL of its waves' root mean square heights), where the sun's
// light crosses on toward the eye.
struct WaterSurface {
    relative: vec3<f32>,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    normal: vec3<f32>,
    roughness: vec2<f32>,
    plane: vec3<f32>,
    // How the relief of the foam on it tilts the normal the foam is lit by
    // off the waves' (a surface gradient, meters a meter), how much its
    // hollows shade it, and how thin it lies, from heaped (0) to a single
    // layer of bubbles (1): none of each for foam drawn without them, as the
    // sea's.
    foam_tilt: vec3<f32>,
    foam_shade: f32,
    foam_thin: f32,
    foam: f32,
    scatter: vec3<f32>,
    transmit: vec3<f32>,
    crest: f32,
    crest_wall: vec3<f32>,
}

// A water surface's light, fogged and graded, its reflectance, and the
// sun's share of the light falling on it as its shadows leave it.
struct WaterLit {
    shaded: Shaded,
    reflectance: f32,
    sunny: f32,
}

// The light of water surface `surface`: the sky and scene it mirrors and
// the sun's disc, by Fresnel's share; the light its body scatters back
// toward the eye, for as much as it stops on its way to what lies below and
// back, and the sun's light its crests scatter on toward the eye as it
// crosses them, through what the surface lets out; and its foam, a white
// layer lit by its own relief, darker in its hollows and glinting where it
// lies wet. One shadow lookup serves all the sun's light on it, and the
// sun's share of the light falling on the water.
fn water_light(surface: WaterSurface) -> WaterLit {
    let world = eye_world(surface.relative);
    let distance = length(surface.relative);
    let view = stable_normalize(-surface.relative);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let normal = water_facing(surface.normal, view, surface.plane);
    // Foam is rough where it heaps, and smoother where it lies thin and wet,
    // alike every way.
    let foam_roughness = mix(WATER_FOAM_ROUGH.y, WATER_FOAM_ROUGH.x, surface.foam_thin);
    let roughness = mix(surface.roughness, vec2<f32>(foam_roughness), surface.foam);
    let alphas = roughness * roughness;
    let n_v = max(dot(normal, view), 1.0e-4);
    let f0 = vec3<f32>(0.02);
    // The waves mask what it mirrors by their slopes along the view, the
    // way the eye looks across them.
    let reflectance = saturate(luminance(environment_brdf(f0, roughness.x, n_v)));
    let sun = lighting.sun.xyz;
    var sunlight = vec3<f32>(0.0);
    if sun.y > 0.0 {
        sunlight = lighting.sun_color.rgb
            * sun_visibility(world, up, surface.world_dx, surface.world_dy, distance, false);
    }
    // The sun's disc mirrored, its lobe the waves' ellipse of slopes along
    // the view and across it (`water_distribution`, `water_visibility`),
    // widened by the half vectors the disc's rays span (Karis), which also
    // keeps calm water's glint from flickering in single pixels. They span
    // half the disc's angular radius (`lighting.highlights.y`, its sine; none
    // for a point sun) within the plane of incidence, along the view, and
    // that over their cosine with the view out of it, across the view, each
    // way's spread adding to the waves' as a variance, so a mirror shows the
    // disc as bright as it is however grazing the view. A rough surface draws
    // a low sun out into a long path of glitter toward the eye.
    var glint = vec3<f32>(0.0);
    let n_l = dot(normal, sun);
    if n_l > 0.0 && any(sunlight > vec3<f32>(0.0)) {
        let half_vector = normalize(view + sun);
        let v_h = max(dot(view, half_vector), 1.0e-4);
        let disc = 0.5 * lighting.highlights.y;
        let widened = sqrt(alphas * alphas + vec2<f32>(disc * disc, disc * disc / (v_h * v_h)));
        let along = water_view_tangent(normal, view);
        let across = cross(normal, along);
        let tangents = mat2x3<f32>(along, across);
        glint = fresnel(f0, v_h) * sunlight
            * (water_distribution(dot(along, half_vector), dot(across, half_vector),
                max(dot(normal, half_vector), 0.0), widened)
                * water_visibility(n_v, n_l, view * tangents, sun * tangents, widened) * n_l
                * lighting.sun_color.w);
    }
    // Its crests lit through (Sea of Thieves' peak mask, Ang 2018, from the
    // body's own coefficients): at a crest the eye looks through a wall of
    // water, and the sun's light entering its far flank, which faces the sun
    // as the near one faces the eye, crosses it unbent between its two
    // faces, the body scattering it toward the eye by its phase about the
    // sun's own heading, most strongly forward, against the sun. Scattered
    // once on its way through a wall where a channel dies by e x times over,
    // x·e^-x of it goes on toward the eye, where the deep body sends back its
    // color over light's two legs through it, down and back (μ₀ + μ about
    // 2): crests of small waves in clear water let the light by, thick ones
    // stop it, and those about one dying by e thick glow most, paler than
    // the deep body for having taken less of the red.
    var crest_light = vec3<f32>(0.0);
    let far = 2.0 * dot(surface.normal, surface.plane) * surface.plane - surface.normal;
    let far_in = dot(far, sun);
    let above = dot(view, surface.plane) > 0.0; // optics: crests are lit through as seen from above the water
    if surface.crest > 0.0 && far_in > 0.0 && above && any(sunlight > vec3<f32>(0.0)) {
        let wall = surface.crest_wall;
        crest_light = surface.scatter * sunlight * wall * exp(-wall)
            * (2.0 * surface.crest * (1.0 - fresnel(f0, far_in).r)
                * water_phase(dot(-sun, view)));
    }
    let body = (surface.scatter * (vec3<f32>(1.0) - surface.transmit)
        * water_scattered(view, sunlight, surface.plane) + crest_light) * (1.0 - reflectance);
    // Foam scatters the sun's light, as the waves it rides face it, and the
    // light around it: the sky's, of which water in its channel sees
    // WATER_FOAM_SKY, and the ground's around it for the rest, sunlit and
    // skylit as the terrain's hollows take it, so foam in shade stands no
    // bluer than the banks beside it. Its own relief tilts the normal it is
    // lit by off the waves', its hollows shade it, and its wet bubbles
    // mirror the sun's disc softly, the more sharply where it lies thin.
    let sky = sky_average(up) * lighting.zenith.w;
    let around = lighting.sun_color.rgb * (0.35 * saturate(sun.y)) + sky * 0.5;
    let foam_normal = normalize(surface.normal - surface.foam_tilt);
    let foam_n_l = dot(foam_normal, sun);
    var foam_glint = vec3<f32>(0.0);
    if foam_n_l > 0.0 && any(sunlight > vec3<f32>(0.0)) {
        let foam_alpha = foam_roughness * foam_roughness;
        let half_vector = normalize(view + sun);
        foam_glint = fresnel(f0, max(dot(view, half_vector), 0.0)) * sunlight
            * (distribution(max(dot(foam_normal, half_vector), 0.0), foam_alpha)
                * visibility(max(dot(foam_normal, view), 1.0e-4), foam_n_l, foam_alpha)
                * foam_n_l * lighting.sun_color.w);
    }
    let foam = WATER_FOAM_ALBEDO * (1.0 - surface.foam_shade)
        * (mix(around, sky, WATER_FOAM_SKY) + sunlight * max(foam_n_l, 0.0)) + foam_glint;
    // The mirrored sky and scene come through `finish`, rough as the waves
    // are along the view and smoother by what they lack of it across: its
    // screen-space reflections replace the share of the water's light its
    // mirrored sky makes up, so its glints, its lit crests and its foam keep
    // theirs. The sun's light is the water's own.
    var point: SurfacePoint;
    point.albedo = surface.scatter;
    point.opacity = 1.0;
    point.emission = 0.0;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = 1.0;
    point.bounce = vec3<f32>(0.0);
    point.roughness = roughness.x;
    point.alpha = alphas.x;
    point.smoother_across = roughness.x - roughness.y;
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
    // Below its surface lie its water and the ground under it, which
    // reflections' world trace does not hold.
    point.body = BODY_WATER;
    var lit: WaterLit;
    lit.shaded = finish(point, mix(body + glint, foam, surface.foam));
    lit.reflectance = reflectance;
    let sunlit = luminance(sunlight) * max(sun.y, 0.0);
    lit.sunny = sunlit / max(sunlit + luminance(sky), 1.0e-6);
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

// Water's light and what shows through it as its targets' color at `pixel`:
// the scene covered but where the shores fade it out.
fn water_covering(shade: WaterShade, pixel: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(stored(shade.shaded.color + shade.behind, pixel) * shade.fade, shade.fade);
}

// Caustics (Guardado and Sánchez-Crespo): how much brighter the sun's
// light falls on the bed `bed` meters under the water's point `world` of
// normal `normal`, where the waves gather it as they
// refract it down: the area a pixel's beam of sunlight lands on through a
// level surface over the area it lands on through the waves (the inverse
// of the Jacobian of where the refracted rays land, from its change across
// the pixel, gathering at most a thousandfold), raised to
// WATER_CAUSTIC_CONTRAST so its mean stays near 1 and its peaks within
// WATER_CAUSTIC_MOST, which the graded scene holds unclipped. Its
// derivatives need it called in uniform control flow.
const WATER_CAUSTIC_CONTRAST: f32 = 0.3;
const WATER_CAUSTIC_MOST: f32 = 1.5;
// The meters of water over which caustics fade, as the light the waves
// gather blurs and scatters on its way down; and how long the waves
// gathering light at a depth are, about a quarter of it, by which caustics
// draw while they span two pixels (`water_shown`) and average to their
// mean of 1 below.
const WATER_CAUSTIC_DEPTH: f32 = 1.5;
const WATER_CAUSTIC_LENGTH: f32 = 0.25;

fn water_caustics(world: vec3<f32>, normal: vec3<f32>, bed: f32) -> f32 {
    let sun = lighting.sun.xyz;
    let into = refract(-sun, normal, 1.0 / WATER_IOR);
    let level = refract(-sun, vec3<f32>(0.0, 1.0, 0.0), 1.0 / WATER_IOR);
    let landing = world.xz + into.xz * (bed / max(-into.y, 0.1));
    let level_landing = world.xz + level.xz * (bed / max(-level.y, 0.1));
    let spread = abs(determinant(mat2x2<f32>(dpdx(landing), dpdy(landing))));
    let level_spread = abs(determinant(mat2x2<f32>(dpdx(level_landing), dpdy(level_landing))));
    if level_spread <= 0.0 {
        return 1.0;
    }
    let gathered = level_spread / max(spread, level_spread * 1.0e-3);
    return min(pow(gathered, WATER_CAUSTIC_CONTRAST), WATER_CAUSTIC_MOST);
}

// How much of the water stands where `meters`, how far its surface stands
// over what lies behind the pixel, comes to nothing at its waterline,
// changing by `change` across a pixel: none at the line, all of it
// WATER_EDGE_PIXELS pixels from it or WATER_EDGE meters over it.
fn water_edge(meters: f32, change: f32) -> f32 {
    return max(saturate(meters / max(change * WATER_EDGE_PIXELS, 1.0e-6)),
        smoothstep(0.0, WATER_EDGE, meters));
}

// How much of the water stands where the samples holding its surface end,
// by the share `held` of its smooth surface's weights they carry
// (`WaterAt`'s), changing by `change` across a pixel: none at WATER_HELD,
// all of it WATER_EDGE_PIXELS pixels within.
fn water_ends(held: f32, change: f32) -> f32 {
    return saturate((held - WATER_HELD) / max(change * WATER_EDGE_PIXELS, 1.0e-6));
}

// The share a fade in over `width` meters of water (a smoothstep from none
// of it to all) shows of water `meters` deep, averaged over the depths the
// pixel spans, `change` across it, so a fade narrower than the pixel covers
// its share of it: over as many either side of its own as lie on the
// water's side of its waterline, where its edge fades it (`water_edge`),
// so a jump in what lies behind the pixel spreads no fade over it.
fn water_ramp(meters: f32, width: f32, change: f32) -> f32 {
    let half = min(0.5 * change, max(meters, 0.0));
    if half < 1.0e-3 * width {
        return smoothstep(0.0, width, meters);
    }
    return (water_ramped(meters + half, width) - water_ramped(meters - half, width))
        / (2.0 * half);
}

// The integral of a fade in over `width` meters of water (a smoothstep)
// from no water to `meters` of it.
fn water_ramped(meters: f32, width: f32) -> f32 {
    let u = saturate(meters / width);
    return width * u * u * u * (1.0 - 0.5 * u) + max(meters - width, 0.0);
}

// Whether water is simulated at `point` (x and z): within the middles of
// the simulation's first and last cells, where its water stands instead of
// the tiles'.
fn water_simulated(point: vec2<f32>) -> bool {
    let cells = water.sim.w;
    if cells < 2.0 {
        return false;
    }
    let low = water.sim.xy + 0.5 * water.sim.z;
    let high = water.sim.xy + (cells - 0.5) * water.sim.z;
    return all(point >= low) && all(point <= high);
}

// Water at the pixel `pixel` (its center) as `at` holds it, drawn at the
// point `point` of its surface (less the eye), refracting the scene's color
// when `refracts`: tiles' water, or the simulated water, drawn on its
// surface, when `simulated`. It lies wherever its surface stands over what
// lies behind the pixel, up to where the samples holding it end.
fn shade_water(point: vec3<f32>, pixel: vec2<f32>, at: WaterAt, refracts: bool,
        simulated: bool) -> WaterShade {
    let world = eye_world(point);
    if !simulated && water_simulated(world.xz) {
        discard;
    }
    // The water ends where the samples holding its surface do, halfway to
    // the first holding none (`WaterAt`'s `held`).
    if at.held < WATER_HELD {
        discard;
    }
    // The plane its surface lies in without its waves, tilted as it falls:
    // the eye looks down onto it, light enters through it and what lies
    // behind the pixel stands under it, so water running down toward the eye
    // from above its height shows as water below it does.
    let plane = normalize(vec3<f32>(at.slope.x, 1.0, at.slope.y));
    // The opaque surface behind the pixel, and how far the water's surface
    // stands over it, the meters it lies under the surface's plane: the
    // water lies wherever it stands over it, so its waterline falls where
    // the ground the pixel shows, or what stands in the water, rises through
    // its surface, whatever shape it takes. And how fast that changes across
    // the pixel, by which the water fades in from its waterline.
    let own = water_ground(pixel);
    let standing = max(dot(point - own, plane) / plane.y, 0.0); // optics: what the view meets behind the pixel
    if standing <= 0.0 {
        discard;
    }
    let standing_change = fwidth(standing);
    let depth = at.depth;
    let flow = at.words.flow;
    let time = camera.position.w;
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    // Simulated water shows no surface stretched sheer between cells far
    // apart in height, down a cliff from the water above to the pool
    // below: the falls' sheets show what falls there.
    var falling = 0.0;
    if simulated {
        let facing = cross(world_dx, world_dy);
        let upright = abs(facing.y) / max(length(facing), 1.0e-12);
        falling = 1.0 - smoothstep(SIM_SHEER.x, SIM_SHEER.y, upright); // filter: a direction, how upright the surface stands
        if falling > 0.99 {
            discard;
        }
    }
    // The water as it settled: how turbulent it runs, where white water is
    // born in it and how fast its current shears (a second).
    let turbulence = at.words.state.x;
    let born = at.words.state.y;
    let shear = at.words.state.z;
    // How fast it runs, its Froude number, and how fast its eddies turn
    // (`water_eddies`): shallow water running fast down a steep surface
    // stirs fast eddies, deep slow water slow ones.
    let speed = length(flow);
    let froude = speed / sqrt(WATER_GRAVITY * max(depth, WATER_THINNEST));
    let eddy = water_eddies(speed, length(at.slope), depth);
    // The pixel's footprint on the water, which filters the patterns it
    // shows of it (`water_footprint`); the meters the pixel resolves there,
    // by which caustics draw; and the meters it resolves on that plane seen
    // from the detail eye (`detail_span`, `detail_resolved`), which any
    // handover by size on screen measures.
    let footprint = water_footprint(world_dx.xz, world_dy.xz);
    let pixel_resolved = detail_resolved(vec2<f32>(length(world_dx.xz), length(world_dy.xz)));
    let seen = detail_resolved(detail_span(world, plane).xy);
    // Each place's patterns start over at their own times, so the flow
    // never pulses all at once (Vlachos's phase noise): the cycles its
    // clock runs ahead by.
    let stagger = water_noise(world.xz / WATER_FLOW_PHASE).x;
    // What stands in the water stands where the bed holds it, about the
    // obstacle nearest the water's samples: only where a pixel's samples
    // agree on one, the obstacle staying put as the point moves across the
    // pixel.
    let anchor = world.xz + at.obstacle.xy;
    let anchoring = (length(dpdx(anchor)) + length(dpdy(anchor)))
        / max(length(world_dx.xz) + length(world_dy.xz), 1.0e-6);
    let anchored = 1.0 - smoothstep(0.1, 0.4, anchoring); // filter: where a pixel's samples disagree on the obstacle, its waves average out
    // How far it is from still to running: what its current raises shows as
    // far as it runs, so nothing changes at once where a river meets a lake.
    let flowing = smoothstep(WATER_FLOWING.x, WATER_FLOWING.y, speed);
    // The bed it flows over, read at points of the world under it and
    // upstream of it as the ground draws it (`water_bed`), on ground lying as
    // its plane does.
    let bed = water_bed(world, depth, plane, flow, froude, footprint, world_dx, world_dy);
    // Its waves, from its state and its bed (`water_flow_waves`): those its
    // bed's stones and what stands in it raise, and its eddies' roughness;
    // none where none rise.
    let waves = water_flow_waves(footprint, at, eddy, froude, flowing, anchored, bed, world_dx,
        world_dy);
    // They ride its surface, tilting as it falls; and the slopes they hide
    // within a pixel roughen it, along the view and across it as their
    // ellipse projects into the view, so where they run rough, it glints
    // rather than mirroring the sky, its glints and what it mirrors drawn out
    // along the view (Bruneton, Neyret and Holzschuch), as does how far the
    // normal of those it draws turns across the pixel (`water_turning`), so
    // their glints widen instead of sparkling (`water_roughness`); where
    // nothing roughens it, it is a calm mirror.
    let slopes = waves.slopes - at.slope;
    let normal = normalize(vec3<f32>(-slopes.x, 1.0, -slopes.y));
    let view = stable_normalize(-point);
    let roughness = water_roughness(waves.hidden + water_turning(slopes), water_heading(view));
    // Under the sea, the sea's water stands instead: its level
    // (`water.ocean.x`) is held relative to the eye.
    if water.ocean.y > 0.5 && world.y < eye_world(vec3<f32>(0.0, water.ocean.x, 0.0)).y {
        discard;
    }
    // What lies behind: where the frame refracts the scene, what the bent
    // view meets and its color.
    var behind: WaterBehind;
    behind.ground = own;
    behind.color = vec3<f32>(0.0);
    if refracts {
        behind = water_behind(point, normal, view, pixel, own, plane);
    }
    // How deep the bed under the pixel lies, where the waves gather the
    // sun's light.
    let bed_depth = min(depth, standing);
    let gathered = water_caustics(world, normal, bed_depth);
    // The water ends at its waterline and where the samples holding its
    // surface end; thin water shows by how fast it runs: calm shallows fade
    // in over WATER_SHORE of it, while racing water, however thin, is all
    // there. Each fade follows how far its surface stands over what lies
    // behind the pixel, as smoothly as the pixel spans it.
    let racing = smoothstep(WATER_RACING.x, WATER_RACING.y, speed);
    let shallows = mix(water_ramp(standing, WATER_SHORE, standing_change), 1.0, racing);
    let fade = min(water_edge(standing, standing_change), water_ends(at.held, fwidth(at.held)))
        * shallows * (1.0 - falling);
    // Light comes back through the water along the view to what it shows,
    // as far as the depth lets it run, after reaching it down from the
    // surface, taken by the water and scattered by its sediment and bubbles.
    let through = min(max(length(behind.ground - point), 0.0), // optics: the view's way through the water
        depth * plane.y / max(dot(view, plane), 0.1)); // optics: the view's way through the water
    let down = clamp(dot(point - behind.ground, plane) / plane.y, 0.0, through); // optics: the light's way down to what the view meets
    // The water holds bubbles where white water is born in it and where it
    // carries the foam its flow churned up.
    let aerated = max(born, saturate(at.words.foam / WATER_FOAM_MOST));
    let transmit = water_transmitted(through + down, aerated);
    // Foam (`meshes/water_foam.wgsl`), the water's own: the foam it carries
    // as its flow churned it up, drained as it ages, and off its bed's stones
    // and what stands in it downstream; and white water, fresh, where it
    // breaks, where its crests fold and where it runs against those stones
    // and what stands in it fast enough to tear air in, its current or its
    // eddies, whichever is faster (`water_bed_foam`). It floats where the
    // waves carry it, toward their crests by their slopes, and its relief
    // tilts the normal it is lit by.
    // Its crests fold where its waves pass the breaking slope, by their
    // statistics (`water_folding`).
    let foam_folding = water_folding(waves.spread, waves.hidden.x + waves.hidden.y, waves.crest);
    let foam_bed = water_bed_foam(bed, at.obstacle, flow, depth, max(speed, eddy), world_dx,
        world_dy, anchored);
    let foam_fresh = 1.0 - (1.0 - born) * (1.0 - foam_folding) * (1.0 - foam_bed.x);
    let foam_carried = 1.0
        - (1.0 - saturate(at.words.foam / WATER_FOAM_MOST)) * (1.0 - foam_bed.y);
    let foam_ridden = world.xz - waves.slopes * WATER_FOAM_RIDE;
    let froth = water_froth(foam_ridden, footprint, vec2<f32>(foam_fresh, foam_carried), flow,
        shear, time, stagger, seen);
    let foam = froth.cover;
    let foam_tilt = water_foam_tilt(froth.height, normal, world_dx, world_dy);
    let foam_shade = froth.shade;
    let foam_thin = froth.thin;
    var surface: WaterSurface;
    surface.relative = point;
    surface.world_dx = world_dx;
    surface.world_dy = world_dy;
    surface.normal = normal;
    surface.roughness = roughness;
    surface.plane = plane;
    surface.foam_tilt = foam_tilt;
    surface.foam_shade = foam_shade;
    surface.foam_thin = foam_thin;
    surface.foam = foam;
    surface.scatter = water_body(aerated);
    surface.transmit = transmit;
    // Its crests where its waves put them, the walls the sun's light crosses
    // there as thick as WATER_CREST_WALL of the root mean square height of
    // all its waves, resolved and hidden.
    surface.crest = waves.crest;
    surface.crest_wall = water_extinction(aerated) * (WATER_CREST_WALL * sqrt(waves.heights));
    let lit = water_light(surface);
    // The waves' caustics brighten the sun's share of the light on the bed,
    // fading as it lies deeper, as the bubbles churning water holds scatter
    // that light on its way down, and drawn while the waves gathering it
    // span two pixels as the pixel resolves them (`water_shown`).
    let caustic = 1.0 + (gathered - 1.0) * lit.sunny
        * exp(-bed_depth * (1.0 / WATER_CAUSTIC_DEPTH + WATER_BUBBLES * aerated))
        * water_shown(WATER_CAUSTIC_LENGTH * bed_depth, pixel_resolved);
    // Of what the surface lets through, the ground shows darker for being
    // wet however thin the water over it, over WATER_SOAKED of it as smoothly
    // as the pixel spans it; foam covers it.
    let wet = water_ramp(standing, WATER_SOAKED, standing_change) * WATER_WET;
    let let_through = (1.0 - lit.reflectance) * (1.0 - foam);
    var result: WaterShade;
    result.shaded = lit.shaded;
    result.coverage = max(max(1.0 - (1.0 - lit.reflectance) * luminance(transmit), wet), foam)
        * fade;
    result.behind = behind.color * transmit * (caustic * let_through * (1.0 - wet)
        * (1.0 - atmospheric_cover(camera.position.xyz, world)));
    result.fade = fade;
    return water_mirroring(result);
}

// Water's shade `shade` with the shares screen-space reflections take of
// what its pixel shows (`finish`'s of its own light): what it mirrors of
// its sky makes up the less of all it shows the more of what lies under it
// shows through, and both what it mirrors and the scene it mirrors in its
// place go as far as it covers the pixel, fading with its shores.
fn water_mirroring(shade: WaterShade) -> WaterShade {
    var result = shade;
    let own = luminance(shade.shaded.color);
    result.shaded.mirrored *= shade.fade * own / max(own + luminance(shade.behind), 1.0e-6);
    result.shaded.reflect *= shade.fade;
    return result;
}

// Tiles' water at the pixel `pixel` drawing the point `mesh` of its mesh
// (less the eye), refracting the scene's color when `refracts`: its water
// read where the point lies (`water_drawn`), shaded on its surface there.
fn shade_tiles(mesh: vec3<f32>, pixel: vec2<f32>, refracts: bool) -> WaterShade {
    let at = water_drawn(mesh);
    return shade_water(water_on_surface(mesh, at), pixel, at, refracts, false);
}

// Colors blend premultiplied over the scene: the scene target's alpha
// keeps the ground's ambient share, the other targets take the water's.
// The canvas holds no copy of the scene's color: water blends over the
// ground there.
@fragment
fn fs_water(input: WaterOutput) -> @location(0) vec4<f32> {
    let shade = shade_tiles(input.relative, input.clip_position.xy, false);
    return vec4<f32>(encoded(shade.shaded.color, input.clip_position.xy) * shade.coverage,
        shade.coverage);
}

@fragment
fn fs_water_post(input: WaterOutput) -> SceneTargets {
    let shade = shade_tiles(input.relative, input.clip_position.xy, true);
    var targets = scene_targets(shade.shaded, input.clip_position.xy);
    targets.color = water_covering(shade, input.clip_position.xy);
    return targets;
}

@fragment
fn fs_water_temporal(input: WaterTemporalOutput) -> TemporalTargets {
    let shade = shade_tiles(input.relative, input.clip_position.xy, true);
    var targets = temporal_targets(shade.shaded, input.clip_position, input.previous);
    targets.color = water_covering(shade, input.clip_position.xy);
    return targets;
}

// Simulated water (`water_sim.rs`): a vertex at each cell's middle, its
// water's surface, depth and flow, the slope of its surface read across
// its neighbors, the foam the water carries, how turbulent it
// runs, where white water is born in it and its shear, read from the water
// around it, and how far across its water body is, whether a lake holds it
// and its bed's reach and nearest obstacle, as the settled water's sample
// nearest it holds them (its bed's words); shaded as tiles' water is. Its
// heights stand over the live window's datum (`water.sim_datum.x` meters),
// which drawing adds back.
//
// Meters of water under which a cell is dry, and the depth over which its
// velocity is desingularized.
const SIM_WET: f32 = 0.01;
const SIM_THIN_WATER: f32 = 0.05;
// The least speed turbulence measures water's swings of speed against, and
// the swings of its vorticity (per second) at which it runs wholly
// turbulent (`engine_terrain::TURBULENT_SPEED`, `TURBULENT_SWIRL`).
const WATER_TURBULENT_SPEED: f32 = 0.5;
const WATER_TURBULENT_SWIRL: f32 = 4.0;
// The turbulent Froude number of running water (the speed of eddies as
// large as it is deep over √(g·h)) from which it breaks white, and from
// which it breaks white all over; and the meters deep thinner water breaks
// as (`engine_terrain::BREAK_ONSET`, `BREAK_FULL`, `BREAK_THINNEST`).
const WATER_BREAK_ONSET: f32 = 0.45;
const WATER_BREAK_FULL: f32 = 0.85;
const WATER_BREAK_THINNEST: f32 = 0.02;
// How upright (the up part of its triangles' normal) the simulated water's
// surface stands where it gives way, from the first to the second: down a
// cliff, between cells far apart in height, the falls' jets show it.
const SIM_SHEER: vec2<f32> = vec2<f32>(0.45, 0.5);

// The speed (meters a second) at which the eddies of water `depth` meters
// deep turn, running `speed` meters a second down a surface falling
// `slope` meters a meter: running down it, the water loses g·U·S watts a
// kilogram (ε), stirring eddies as large as it is deep (h, at least
// WATER_BREAK_THINNEST) that turn at (ε·h)^(1/3) (Kolmogorov), the eddies
// by which the cook breaks water white (`engine_terrain`'s `breaking`).
fn water_eddies(speed: f32, slope: f32, depth: f32) -> f32 {
    let stirring = WATER_GRAVITY * speed * slope * max(depth, WATER_BREAK_THINNEST);
    return select(0.0, pow(stirring, 1.0 / 3.0), stirring > 0.0);
}

// The turbulent Froude number of that water: its eddies' speed
// (`water_eddies`) over √(g·h), the speed gravity smooths its surface at,
// from WATER_BREAK_ONSET of which they break it white and from
// WATER_BREAK_FULL white all over.
fn water_eddy_froude(speed: f32, slope: f32, depth: f32) -> f32 {
    return water_eddies(speed, slope, depth)
        / sqrt(WATER_GRAVITY * max(depth, WATER_BREAK_THINNEST));
}

// The simulated water's vertex: where it draws, its point less the eye,
// and its water there.
struct SimWaterOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    // Meters of water over the ground, and its flow along x and z.
    @location(1) water: vec3<f32>,
    // How far across its water body is (meters; none where its bed's words
    // know none), and 1 where a lake holds it.
    @location(2) body: vec2<f32>,
    // How fast its surface falls along x and z (meters a meter).
    @location(3) slope: vec2<f32>,
    // The foam its water carries (0 none, WATER_FOAM_MOST the most).
    @location(4) churned: f32,
    // How turbulent it runs (0 calm to 1), how strongly it breaks white (0
    // to 1: where short-lived white water is born), its shear (the
    // magnitude of its velocity's gradient, per second), and whether its
    // surface lies at the water's (1) or the water ends before it (0), which
    // the cells between blend.
    @location(5) state: vec4<f32>,
    // The offset from it to the middle of its nearest obstacle within 16
    // meters, rock breaking its surface or a boulder under it (meters along
    // x and z; 0 for none), the obstacle's radius (meters; 0 for none), and
    // how strongly a drop or a jump lies just upstream (0 to 1).
    @location(6) obstacle: vec4<f32>,
    // The reach of the stream it runs in: 0 for none, else 1 plus
    // `engine_terrain::Reach` (pool, plane bed, steps, cascade, falls).
    @location(7) @interpolate(flat) reach: u32,
}

// The simulated water's vertex in the temporal pipelines: also where its
// point stood in the previous frame's view (x, y and w of its clip
// position).
struct SimWaterTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    @location(1) water: vec3<f32>,
    @location(2) body: vec2<f32>,
    @location(3) slope: vec2<f32>,
    @location(4) churned: f32,
    @location(5) state: vec4<f32>,
    @location(6) obstacle: vec4<f32>,
    @location(7) @interpolate(flat) reach: u32,
    @location(8) previous: vec3<f32>,
}

// Where the simulated water holds cell (i, j), the nearest inside its
// square for one past its edge: its block's first cell (its blocks stand
// row by row, each turned along x and z by the blocks `water.sim_datum.yz`
// says, around the square's blocks) and its place in the block.
fn sim_water_at(i: i32, j: i32) -> u32 {
    let n = i32(water.sim.w);
    let at = vec2<u32>(clamp(vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(n - 1)));
    let blocks = (u32(n) + SIM_WATER_BLOCK - 1u) / SIM_WATER_BLOCK;
    let place = (at / SIM_WATER_BLOCK + vec2<u32>(water.sim_datum.yz)) % blocks;
    let block = place.y * blocks + place.x;
    let local = at % SIM_WATER_BLOCK;
    return (block * SIM_WATER_BLOCK + local.y) * SIM_WATER_BLOCK + local.x;
}

// Cell (i, j) of the simulated water, the nearest inside its square for
// one past its edge.
fn sim_water_cell(i: i32, j: i32) -> vec4<f32> {
    return water_sim_cells[sim_water_at(i, j)];
}

// The ground at cell (i, j)'s middle: its corners' mean.
fn sim_water_ground(i: i32, j: i32) -> f32 {
    let at = sim_water_at(i, j);
    let cells = SIM_WATER_BLOCK * SIM_WATER_BLOCK;
    let row = SIM_WATER_BLOCK + 1u;
    let local = at % cells;
    let corner = (at / cells) * SIM_BLOCK_GROUND + (local / SIM_WATER_BLOCK) * row
        + local % SIM_WATER_BLOCK;
    return 0.25 * (water_sim_ground[corner] + water_sim_ground[corner + 1u]
        + water_sim_ground[corner + row] + water_sim_ground[corner + row + 1u]);
}

// Cell (i, j)'s water's surface and depth, and whether its surface lies at
// the water's (1) or the water ends before it (0). A dry cell beside wet
// ones on ground as high as their surface carries that surface over it,
// as deep as it stands over the ground (none or less), so the shore falls
// where the ground drawn rises through the surface; one on lower ground
// lies on it, as far under water as the shallowest of them stands over
// theirs, the water ending halfway to it; one beside none, a meter below
// its ground, the water ending halfway to it too.
fn sim_water_level(i: i32, j: i32) -> vec3<f32> {
    let ground = sim_water_ground(i, j);
    let surface = sim_water_cell(i, j).x;
    if surface - ground > SIM_WET {
        return vec3<f32>(surface, surface - ground, 1.0);
    }
    var sum = 0.0;
    var count = 0.0;
    var shallowest = 1.0e9;
    for (var k = 0; k < 4; k++) {
        let step = select(vec2<i32>(0, 2 * (k - 2) - 1), vec2<i32>(2 * k - 1, 0), k < 2);
        let other = sim_water_cell(i + step.x, j + step.y).x;
        let deep = other - sim_water_ground(i + step.x, j + step.y);
        if deep > SIM_WET {
            sum += other;
            count += 1.0;
            shallowest = min(shallowest, deep);
        }
    }
    if count == 0.0 {
        return vec3<f32>(ground - 1.0, -1.0, 0.0);
    }
    let level = sum / count;
    if ground >= level {
        return vec3<f32>(level, level - ground, 1.0);
    }
    return vec3<f32>(ground, -shallowest, 0.0);
}

// Cell (i, j)'s water's velocity (xy), its flow over its depth,
// desingularized as the drawn water's, and none where it is dry; and 1
// where it holds water, else 0 (z).
fn sim_water_moving(i: i32, j: i32) -> vec3<f32> {
    let cell = sim_water_cell(i, j);
    let depth = cell.x - sim_water_ground(i, j);
    let wet = depth > SIM_WET;
    return vec3<f32>(select(vec2<f32>(0.0), cell.yz / max(depth, SIM_THIN_WATER), wet),
        select(0.0, 1.0, wet));
}

// How fast the velocity `here` of a cell changes a meter between the cells
// either side of it, `before` and `after` (`sim_water_moving`'s): between
// the two where both hold water, between it and the one that does where
// one does, and not at all where neither does.
fn sim_water_change(before: vec3<f32>, here: vec2<f32>, after: vec3<f32>) -> vec2<f32> {
    let low = select(here, before.xy, before.z > 0.5);
    let high = select(here, after.xy, after.z > 0.5);
    return (high - low) / (water.sim.z * max(before.z + after.z, 1.0));
}

// The surface beside a cell standing at `here`, from its neighbor at
// `other` (`sim_water_level`'s): the neighbor's where it is wet, else the
// cell's own, so shores tilt no surface.
fn sim_water_beside(here: vec3<f32>, other: vec3<f32>) -> f32 {
    return select(here.x, other.x, other.y > SIM_WET);
}

fn sim_water_output(index: u32) -> SimWaterOutput {
    let n = u32(water.sim.w);
    let i = i32(index % n);
    let j = i32(index / n);
    let cell = water.sim.z;
    let here = sim_water_level(i, j);
    let depth = here.y;
    let flow = sim_water_cell(i, j).yz;
    let velocity = select(vec2<f32>(0.0), flow / max(depth, SIM_THIN_WATER), depth > SIM_WET);
    let east = sim_water_level(i + 1, j);
    let west = sim_water_level(i - 1, j);
    let north = sim_water_level(i, j + 1);
    let south = sim_water_level(i, j - 1);
    let world = vec3<f32>(water.sim.x + (f32(i) + 0.5) * cell, here.x + water.sim_datum.x,
        water.sim.y + (f32(j) + 0.5) * cell);
    var output: SimWaterOutput;
    output.relative = world - camera.position.xyz;
    output.clip_position = terrain_view.relative_view_projection * vec4<f32>(output.relative, 1.0);
    output.water = vec3<f32>(depth, velocity);
    output.slope = vec2<f32>(sim_water_beside(here, west) - sim_water_beside(here, east),
        sim_water_beside(here, south) - sim_water_beside(here, north)) / (2.0 * cell);
    output.churned = sim_water_cell(i, j).w;
    // How turbulent it runs, from the velocities around it as the cook's
    // turbulence reads its water's swings (`engine_terrain`'s
    // `turbulence`): how far its velocity strays from the mean of its
    // neighbors holding water, against its speed, and how fast the water
    // turns there; where white water is born in it, by the rule the cook
    // breaks its settled water white by (`engine_terrain`'s `breaking`):
    // where its eddies break its surface (`water_eddy_froude`), from
    // WATER_BREAK_ONSET to WATER_BREAK_FULL; its shear; and whether its
    // surface lies at the water's. Its body's span, whether a lake holds it,
    // its reach and its nearest obstacle are its bed's: the settled water's
    // sample nearest it, as tiles' words pack them.
    let to_east = sim_water_moving(i + 1, j);
    let to_west = sim_water_moving(i - 1, j);
    let to_north = sim_water_moving(i, j + 1);
    let to_south = sim_water_moving(i, j - 1);
    let along_x = sim_water_change(to_west, velocity, to_east);
    let along_z = sim_water_change(to_south, velocity, to_north);
    let holding = to_east.z + to_west.z + to_north.z + to_south.z;
    let mean = (to_east.xy + to_west.xy + to_north.xy + to_south.xy) / max(holding, 1.0);
    let strays = select(0.0, length(velocity - mean), holding > 0.5);
    let swirl = abs(along_x.y - along_z.x);
    let turbulence = saturate(strays / max(length(velocity), WATER_TURBULENT_SPEED)
        + swirl / WATER_TURBULENT_SWIRL);
    let shear = sqrt(dot(along_x, along_x) + dot(along_z, along_z));
    let breaks = water_eddy_froude(length(velocity), length(output.slope), depth);
    let born = smoothstep(WATER_BREAK_ONSET, WATER_BREAK_FULL, breaks);
    let wet = depth > SIM_WET;
    output.state = vec4<f32>(select(vec3<f32>(0.0), vec3<f32>(turbulence, born, shear), wet),
        here.z);
    let bed = water_sim_beds[sim_water_at(i, j)];
    let span = unpack4x8snorm(extractBits(bed.x, 8u, 8u)).x;
    output.body = vec2<f32>(water_span(span), select(0.0, 1.0, span < 0.0));
    output.obstacle = water_obstacle(bed.y, vec2<f32>(0.0));
    output.reach = extractBits(bed.x, 0u, 8u);
    return output;
}

@vertex
fn vs_sim_water(@builtin(vertex_index) index: u32) -> SimWaterOutput {
    return sim_water_output(index);
}

@vertex
fn vs_sim_water_temporal(@builtin(vertex_index) index: u32) -> SimWaterTemporalOutput {
    let output = sim_water_output(index);
    var temporal: SimWaterTemporalOutput;
    temporal.clip_position = output.clip_position;
    temporal.relative = output.relative;
    temporal.water = output.water;
    temporal.body = output.body;
    temporal.slope = output.slope;
    temporal.churned = output.churned;
    temporal.state = output.state;
    temporal.obstacle = output.obstacle;
    temporal.reach = output.reach;
    temporal.previous = (camera.previous_view_projection
        * vec4<f32>(eye_world(output.relative), 1.0)).xyw;
    return temporal;
}

// The simulated water at a pixel as its vertices' outputs `input` blend
// there, as shading reads water (`WaterAt`): as deep as they are, none or
// more; held as far as its cells' surfaces lie at the water's, so it ends
// halfway to a cell it ends before; and no fall's sheet runs over it.
fn sim_water_values(input: SimWaterOutput) -> WaterAt {
    var at: WaterAt;
    at.depth = max(input.water.x, 0.0);
    at.held = input.state.w;
    at.words.flow = input.water.yz;
    at.words.foam = input.churned;
    at.words.span = input.body.x;
    at.words.lake = input.body.y > 0.5;
    at.words.state = input.state.xyz;
    at.words.reach = input.reach;
    at.slope = input.slope;
    at.obstacle = input.obstacle;
    return at;
}

// The simulated water at the pixel its vertices' outputs `input` blend at,
// refracting the scene's color when `refracts`: drawn on its surface.
fn shade_sim(input: SimWaterOutput, refracts: bool) -> WaterShade {
    return shade_water(input.relative, input.clip_position.xy, sim_water_values(input),
        refracts, true);
}

// The simulated water's vertices' outputs in the temporal pipelines
// `input`, without where they stood in the previous frame.
fn sim_water_present(input: SimWaterTemporalOutput) -> SimWaterOutput {
    var output: SimWaterOutput;
    output.clip_position = input.clip_position;
    output.relative = input.relative;
    output.water = input.water;
    output.body = input.body;
    output.slope = input.slope;
    output.churned = input.churned;
    output.state = input.state;
    output.obstacle = input.obstacle;
    output.reach = input.reach;
    return output;
}

@fragment
fn fs_sim_water(input: SimWaterOutput) -> @location(0) vec4<f32> {
    let shade = shade_sim(input, false);
    return vec4<f32>(encoded(shade.shaded.color, input.clip_position.xy) * shade.coverage,
        shade.coverage);
}

@fragment
fn fs_sim_water_post(input: SimWaterOutput) -> SceneTargets {
    let shade = shade_sim(input, true);
    var targets = scene_targets(shade.shaded, input.clip_position.xy);
    targets.color = water_covering(shade, input.clip_position.xy);
    return targets;
}

@fragment
fn fs_sim_water_temporal(input: SimWaterTemporalOutput) -> TemporalTargets {
    let shade = shade_sim(sim_water_present(input), true);
    var targets = temporal_targets(shade.shaded, input.clip_position, input.previous);
    targets.color = water_covering(shade, input.clip_position.xy);
    return targets;
}
