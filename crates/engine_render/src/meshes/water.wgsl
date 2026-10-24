// Water: terrains' water as their tiles hold it, drawn on the leaves of the
// terrain's subdivision over which any stands (`cs_water_leaves`). Each
// vertex stands where the ground's leaf has its own, at its water's surface
// there: the finest resident level's samples around the point, read by
// position and blended on the ground's own triangles (`water_at`), so the
// depth each pixel shades by is the tiles' surface over the true ground
// (`terrain_true_height`), and the shore falls where the ground rises
// through the water. What its samples' words say of the water (its flow,
// foam, white water and what stands in it) blends alike, so none of it
// moves as the eye does. Its foam's lace and relief and the waves standing
// about what stands in it hand over to their means as a pixel seen from the
// detail eye comes to span their detail along the view (`detail_span`),
// while the pixel's own derivatives filter them. The waves' tiles
// (`water_waves.wgsl`) tilt it where it stands still and their finest run
// with its flow where it moves, carried along in sets that start over
// before its shear stretches them; running water's chop comes from its
// current, the eddies it carries and short waves standing on its bed where
// it runs near or above the speed of its own waves, and what they
// hide within a pixel roughens it; waves the bed holds still stand about
// what the water flows past. It mirrors the sky, the sun's disc and,
// through screen-space reflections, the scene; its body, tinted by the
// sediment it carries, scatters the light entering it back toward the
// eye; and what lies under it shows through, bent by the waves where the
// frame copies the scene's color, brighter where the waves gather the sun's
// light on it, and dimmed channel by channel over the light's path down to
// it and back, the ground darker for being wet. White water is born where
// the water loses its energy, fraying as it runs; foam lies where the water
// carries it, as its flow churns it up settling or simulated
// (`engine_terrain::shallow_water`), and along shallow shores. However
// thin, the water fades only at its waterline. The sea (`meshes/ocean.wgsl`)
// lights its surface the same way. Where rivers leave their beds, their
// jets fall as sheets of their own (`water_falls.rs`), drawn last.

// Tiles' water, a block of samples for each patch holding any
// (`terrain/water_blocks.rs`): their surfaces (terrain-local meters,
// NO_WATER_SURFACE where none reaches), and their words
// (`engine_terrain::TileWater::info`), a texel's channels: their flow along
// x and z as shares of WATER_FLOW_RANGE either way, the foam they carry as
// a share of WATER_FOAM_MOST (negative where a fall's sheet runs over
// them), and 1 where a lake holds them, else the half width of the river's
// water as 126ths of WATER_WIDTH_RANGE in 127ths (x, four signed bytes);
// how turbulent they run, how strongly they break white and their shear
// as a share of WATER_SHEAR_RANGE, in 255ths, and 0 for no reach, else
// their reach plus one (y, four bytes); the offset to their nearest
// obstacle's middle along x and z in signed steps of WATER_OBSTACLE_STEP,
// its radius in steps of WATER_RADIUS_STEP (0 for none) and how strongly a
// drop or a jump lies just upstream, in 255ths (z, four bytes); and 0 (w).
@group(1) @binding(50) var water_surface: texture_2d_array<f32>;
@group(1) @binding(51) var water_info: texture_2d_array<u32>;
// Water's directory (`terrain/water_blocks.rs`), WATER_TILE_WORDS words for
// each layer of the terrain's height array: the block of each patch of the
// tile in it (NO_WATER_BLOCK where none holds water), row by row, then a bit
// for each patch in the same order, set where water stands over it there or
// in a finer resident tile within it.
@group(1) @binding(59) var<storage, read> water_directory: array<u32>;
// The leaves of each terrain's subdivision over which water stands, in their
// order (`cs_water_list`).
@group(1) @binding(60) var<storage, read> water_leaves: array<u32>;
// Where a terrain's water lists those leaves (`water.rs`): its first listed
// leaf, its first word of their bits, the most leaves it tests and lists,
// and its draw (`WaterDraw`).
struct WaterLeafRecord {
    places: vec4<u32>,
}
@group(1) @binding(61) var<uniform> water_leaf_record: WaterLeafRecord;

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
// which their water gives way to it. Where water is simulated around the
// camera (`water_sim.rs`), `sim` holds its square's first corner along x
// and z, the meters across a cell and the cells along a side (none, zero),
// and `sim_datum` the height its heights stand above (x, world meters).
// `sediment` holds the color the fine sediment terrains' water carries
// scatters back (rgb, linear) and how strongly it scatters light, per meter
// (w); the sea carries none.
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
    sim: vec4<f32>,
    sim_datum: vec4<f32>,
    sediment: vec4<f32>,
}
@group(1) @binding(52) var<uniform> water: WaterUniform;
// The waves' moments, a layer a tile, with their levels
// (`water_waves.wgsl`), and the sampler repeating them.
@group(1) @binding(53) var water_waves: texture_2d_array<f32>;
@group(1) @binding(54) var water_wave_sampler: sampler;
// Falling water's map (`water_fall_map.rs`): its strands (r) and their
// fibers (g) spread about a half, its foam's bubbles' walls (b) and
// bubbles (a), FALL_MAP meters across and along the water as it leaves its
// lip; read with the waves' sampler, repeating.
@group(1) @binding(55) var water_fall_map: texture_2d<f32>;
// The simulated water as its blocks hold it (`water_sim.wgsl`'s, every
// block of the window's square in order, SIM_WATER_BLOCK cells along each
// side): each cell's surface (world meters), its flow along x and z (square
// meters a second) and the foam it carries, a block's row by row; and each
// block's ground, SIM_BLOCK_GROUND entries of it: at its cells' corners (a
// row of its cells and one more at a time), then the levels lakes hold its
// cells at and its cells' beds' roughness; and what the bed holds under
// each cell, in the cells' order, as the settled water's sample nearest it
// holds it (`engine_terrain`'s `Tile::bed_words`): its reach's code and its
// nearest obstacle, offset from the cell's middle, as tiles' words pack them.
@group(1) @binding(56) var<storage, read> water_sim_cells: array<vec4<f32>>;
@group(1) @binding(57) var<storage, read> water_sim_ground: array<f32>;
@group(1) @binding(58) var<storage, read> water_sim_beds: array<vec2<u32>>;
const SIM_WATER_BLOCK: u32 = 8u;
const SIM_BLOCK_GROUND: u32 = 209u;
// The most foam water carries, as its flow churns it up
// (`engine_terrain::shallow_water::FOAM_MOST`); the shares of the most
// over which it shows as lace on calm water, from the first to the second;
// how much of the lace its most shows; and how white its most streaks
// water running down steeply.
const WATER_FOAM_MOST: f32 = 1.5;
const WATER_FOAM_SHOWN: vec2<f32> = vec2<f32>(0.1, 0.8);
const WATER_FOAM_LACE: f32 = 0.7;
const WATER_FOAM_STREAKED: f32 = 0.6;
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

// Samples along a block's side and blocks along a layer's side, no block,
// and what a sample no water reaches holds; patches along a side of a tile,
// the words each layer takes in water's directory, and a terrain
// directory's entry of a tile not resident (`gpu_layouts::constants`).
const WATER_BLOCK: u32 = 17u;
const WATER_BLOCKS_SIDE: u32 = 16u;
const NO_WATER_BLOCK: u32 = 0xffffffffu;
const NO_WATER_SURFACE: f32 = -1.0e30;
const WATER_TILE_PATCHES: u32 = 16u;
const WATER_TILE_WORDS: u32 = 264u;
const NOT_RESIDENT: u32 = 0xffffffffu;
// Meters a second either way the packed flow spans, the meters of half
// width the packed width of a river's water does, the shear (per second)
// the packed shear does, and the meters a step of the packed offset to an
// obstacle and of its packed radius span (`engine_terrain::WATER_FLOW_RANGE`,
// `WATER_WIDTH_RANGE`, `WATER_SHEAR_RANGE`, `WATER_OBSTACLE_STEP`,
// `WATER_RADIUS_STEP`).
const WATER_FLOW_RANGE: f32 = 8.0;
const WATER_WIDTH_RANGE: f32 = 63.0;
const WATER_SHEAR_RANGE: f32 = 8.0;
const WATER_OBSTACLE_STEP: f32 = 0.125;
const WATER_RADIUS_STEP: f32 = 0.0625;
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
// channel sees; and foam's roughness where it lies a single layer of wet
// bubbles thin, and where it heaps.
const WATER_FOAM_ALBEDO: f32 = 0.65;
const WATER_FOAM_SKY: f32 = 0.6;
const WATER_FOAM_ROUGH: vec2<f32> = vec2<f32>(0.3, 0.7);
// The lowest the eye's mirrored view dips: slopes turned from the eye would
// mirror what lies below the horizon.
const WATER_HORIZON: f32 = 0.02;
// Meters the bent view runs at most to what it shows: the water hides what
// lies farther.
const WATER_REACH: f32 = 4.0;
// Meters over which the sea fades in from its shores and around what
// stands in it.
const WATER_SHORE: f32 = 0.1;
// The pixels over which terrains' water fades in from its waterline, where
// its depth or its height over what stands in it comes to nothing, and the
// meters over the line past which it is all there however fast either
// changes across a pixel: running water, however thin, is water.
const WATER_EDGE_PIXELS: f32 = 1.5;
const WATER_EDGE: f32 = 0.05;
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
// The share of the wind's finest ripples water running 2 m/s or faster
// keeps: its current's chop (`water_chop`) takes over from them.
const WATER_RUNNING_RIPPLES: f32 = 0.5;
// The share of a tile's period between the corners of the grid its random
// offsets hang on (`water_tile`).
const WATER_PATCH: f32 = 0.5;
// Pixels a tile's period spans below which it blurs into its variance; and
// those a chop's wavelength spans below which its slopes start to hide in
// its variance, all of them once it spans half as many.
const WATER_WAVE_PIXELS: f32 = 32.0;
// Meters across the patches of stronger and calmer wind sweeping over
// water.
const WATER_GUST_SIZE: f32 = 40.0;

struct WaterOutput {
    @builtin(position) clip_position: vec4<f32>,
    // The point less the eye.
    @location(0) relative: vec3<f32>,
    // Meters of water over the ground, and its flow along x and z.
    @location(1) water: vec3<f32>,
    // How white its foam streaks it where it runs down steeply, the half
    // width of its water (meters; lakes WATER_WIDTH_RANGE), and 1 where a
    // fall's sheet runs over it.
    @location(2) foam: vec3<f32>,
    // How fast its depth falls along x and z (meters a meter), the ground
    // rising out of it toward its shores.
    @location(3) shore: vec2<f32>,
    // How fast its surface falls along x and z (meters a meter).
    @location(4) slope: vec2<f32>,
    // The foam its water carries (0 none, WATER_FOAM_MOST the most).
    @location(5) churned: f32,
    // How turbulent it runs (0 calm to 1), how strongly it breaks white (0
    // to 1: where short-lived white water is born), its shear (the
    // magnitude of its velocity's gradient, per second), and 0.
    @location(6) state: vec4<f32>,
    // The offset from it to the middle of its nearest obstacle within 16
    // meters, rock breaking its surface or a boulder under it (meters along
    // x and z; 0 for none), the obstacle's radius (meters; 0 for none), and
    // how strongly a drop or a jump lies just upstream (0 to 1).
    @location(7) obstacle: vec4<f32>,
    // The reach of the stream it runs in: 0 for none, else 1 plus
    // `engine_terrain::Reach` (pool, plane bed, steps, cascade, falls).
    @location(8) @interpolate(flat) reach: u32,
}

struct WaterTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    @location(1) water: vec3<f32>,
    @location(2) foam: vec3<f32>,
    @location(3) shore: vec2<f32>,
    @location(4) slope: vec2<f32>,
    @location(5) churned: f32,
    @location(6) state: vec4<f32>,
    @location(7) obstacle: vec4<f32>,
    @location(8) @interpolate(flat) reach: u32,
    @location(9) previous: vec3<f32>,
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

// Block `block`'s water's surface at its sample `sample` (x), and 1 where
// any water reaches it, else 0 (y).
fn water_block_level(block: u32, sample: vec2<u32>) -> vec2<f32> {
    let texel = water_texel(block, sample);
    let surface = textureLoad(water_surface, texel.xy, texel.z, 0).x;
    return vec2<f32>(surface, select(1.0, 0.0, surface <= NO_WATER_SURFACE * 0.5));
}

// How fast the water's surface falls a meter along `step` at block
// `block`'s sample `sample` (`first` or `last` along it), standing at
// `center` there, its samples `spacing` meters apart: between the samples
// either side water reaches, or between it and the one that does.
fn water_surface_falling(block: u32, sample: vec2<u32>, step: vec2<u32>, first: bool, last: bool,
        center: f32, spacing: f32) -> f32 {
    var before = vec2<f32>(center, 0.0);
    var after = vec2<f32>(center, 0.0);
    if !first {
        before = water_block_level(block, sample - step);
    }
    if !last {
        after = water_block_level(block, sample + step);
    }
    return (select(center, before.x, before.y > 0.5) - select(center, after.x, after.y > 0.5))
        / (spacing * max(before.y + after.y, 1.0));
}

// How fast the water's surface falls along x and z at block `block`'s
// sample `sample`, as `water_surface_falling` gives it.
fn water_slope(block: u32, sample: vec2<u32>, center: f32, spacing: f32) -> vec2<f32> {
    let last = WATER_BLOCK - 1u;
    return vec2<f32>(
        water_surface_falling(block, sample, vec2<u32>(1u, 0u), sample.x == 0u,
            sample.x == last, center, spacing),
        water_surface_falling(block, sample, vec2<u32>(0u, 1u), sample.y == 0u,
            sample.y == last, center, spacing));
}

// What a sample's words say of its water: its flow along x and z (meters a
// second), the foam it carries (0 none, WATER_FOAM_MOST the most), the half
// width of its water (meters; lakes WATER_WIDTH_RANGE) and whether a fall's
// sheet runs over it; how turbulent it runs, how strongly it breaks white
// and its shear (a second); its reach's code (0 for none, else 1 plus
// `engine_terrain::Reach`); and its nearest obstacle's word
// (`water_obstacle`).
struct WaterWords {
    flow: vec2<f32>,
    foam: f32,
    width: f32,
    sheet: bool,
    state: vec3<f32>,
    reach: u32,
    obstacle: u32,
}

// The words of block `block`'s sample `sample`.
fn water_words(block: u32, sample: vec2<u32>) -> WaterWords {
    let texel = water_texel(block, sample);
    let words = textureLoad(water_info, texel.xy, texel.z, 0);
    let info = unpack4x8snorm(words.x);
    let state = unpack4x8unorm(words.y);
    var read: WaterWords;
    read.flow = info.xy * WATER_FLOW_RANGE;
    read.foam = abs(info.z) * WATER_FOAM_MOST;
    read.width = select(saturate(info.w * 127.0 / 126.0) * WATER_WIDTH_RANGE, WATER_WIDTH_RANGE,
        info.w > 0.996);
    read.sheet = info.z < 0.0;
    read.state = vec3<f32>(state.xy, state.z * WATER_SHEAR_RANGE);
    read.reach = words.y >> 24u;
    read.obstacle = words.z;
    return read;
}

// The terrain's extent along x and z (meters): its level-0 tiles'.
fn water_extent() -> vec2<f32> {
    let header = terrain.directory.x;
    let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
    return vec2<f32>(tiles * (WATER_TILE_PATCHES * (WATER_BLOCK - 1u))) * terrain.anchor.w;
}

// Tiles' water at a point: its surface's height above the terrain's origin
// and its depth over the ground there; what its samples' words say of it,
// what changes by degrees blended and what names a kind (a fall's sheet
// over it, its reach, its nearest obstacle's word) its nearest sample's;
// how fast its depth and its surface fall along x and z (meters a meter);
// and its nearest obstacle as `water_obstacle` gives it, offset from the
// point, how strongly a drop or a jump lies just upstream blended.
struct WaterAt {
    surface: f32,
    depth: f32,
    words: WaterWords,
    shore: vec2<f32>,
    slope: vec2<f32>,
    obstacle: vec4<f32>,
}

// Tiles' water at terrain-local `local`, over ground `ground` meters above
// the terrain's origin there (`terrain_true_height`): read from the finest
// level whose tile holding the point is resident, in the block of the
// patch holding the point's cell, from the samples around the point,
// blended on the triangle holding it as the ground's heights are
// (`terrain_level_height`) and as `engine_terrain`'s `Tile::water_at`
// reads water; how fast its depth and its surface fall, from each sample's
// neighbors in the block (`water_shoaling`, `water_slope`), blended alike.
// A sample no water reaches stands dry as `water_block_sample` treats it,
// so the shore falls where the surface meets the ground; where the patch
// holds none, no level holds the point or it lies past the terrain, the
// water stands a meter under the ground, WATER_DRY dry. Its depth is its
// samples' over their own ground blended, moved by as far as the ground
// stands under their ground's blend: where every sample holds water, its
// surface over the ground.
fn water_at(local: vec2<f32>, ground: f32) -> WaterAt {
    var at: WaterAt;
    at.surface = ground - 1.0;
    at.depth = -WATER_DRY;
    if any(local < vec2<f32>(0.0)) || any(local > water_extent()) {
        return at;
    }
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
    let quads = WATER_BLOCK - 1u;
    let tile_quads = WATER_TILE_PATCHES * quads;
    for (var index = 0u; index < levels; index++) {
        let header = base + index * 4u;
        let spacing = terrain.anchor.w * f32(1u << index);
        let samples = local / spacing;
        let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
        let tile = min(vec2<u32>(samples / f32(tile_quads)), tiles - vec2<u32>(1u));
        let layer = terrain_directory[terrain_directory[header] + tile.y * tiles.x + tile.x];
        if layer == NOT_RESIDENT {
            continue;
        }
        // Samples 0 to 256 within the tile, and the cell holding the point:
        // its patch, its first corner in the patch's block and where in the
        // cell the point lies; the tiles hold a border of one.
        let within = clamp(samples - vec2<f32>(tile * tile_quads), vec2<f32>(0.0),
            vec2<f32>(f32(tile_quads)));
        let corner = min(vec2<u32>(within), vec2<u32>(tile_quads - 1u));
        let f = within - vec2<f32>(corner);
        let tile_patch = corner / quads;
        let block = water_directory[layer * WATER_TILE_WORDS + tile_patch.y * WATER_TILE_PATCHES
            + tile_patch.x];
        if block == NO_WATER_BLOCK {
            return at;
        }
        let first = corner - tile_patch * quads;
        // The cell's corners' weights (its first, the next along x, along z,
        // and the far one) on the triangle holding the point: the cell's
        // diagonal runs from its (x, z+1) corner to its (x+1, z) one.
        let weights = select(vec4<f32>(0.0, 1.0 - f.y, 1.0 - f.x, f.x + f.y - 1.0),
            vec4<f32>(1.0 - f.x - f.y, f.x, f.y, 0.0), f.x + f.y <= 1.0);
        let nearest = select(vec2<u32>(0u), vec2<u32>(1u), f >= vec2<f32>(0.5));
        var surface = 0.0;
        var depth = 0.0;
        var below = 0.0;
        var jump = 0.0;
        var blended: WaterWords;
        for (var around = 0u; around < 4u; around++) {
            let step = vec2<u32>(around & 1u, around >> 1u);
            let sample = first + step;
            let words = water_words(block, sample);
            if all(step == nearest) {
                blended.sheet = words.sheet;
                blended.reach = words.reach;
                blended.obstacle = words.obstacle;
                // Offset from the point: the sample's own, plus the
                // sample's from the point.
                at.obstacle = water_obstacle(words.obstacle, (vec2<f32>(step) - f) * spacing);
            }
            let weight = weights[around];
            if weight <= 0.0 {
                continue;
            }
            let texel = corner + step + 1u;
            let sample_ground = terrain_load(layer, texel);
            let held = water_block_sample(block, sample, sample_ground);
            surface += held.x * weight;
            depth += held.y * weight;
            below += sample_ground * weight;
            blended.flow += words.flow * weight;
            blended.foam += words.foam * weight;
            blended.width += words.width * weight;
            blended.state += words.state * weight;
            jump += f32(words.obstacle >> 24u) / 255.0 * weight;
            at.shore += water_shoaling(block, sample, held.y, layer, texel, spacing) * weight;
            at.slope += water_slope(block, sample, held.x, spacing) * weight;
        }
        at.surface = surface;
        at.depth = depth + below - ground;
        at.words = blended;
        at.obstacle.w = jump;
        return at;
    }
    return at;
}

// The point at terrain-local `local` standing `height` meters above the
// terrain's origin, less the eye of the view drawing it.
fn water_relative(local: vec2<f32>, height: f32) -> vec3<f32> {
    let from_eye = terrain.origin.xz - terrain.eye.xz;
    return vec3<f32>(local.x + from_eye.x, height + terrain.anchor.y, local.y + from_eye.y)
        + terrain_view.eye_offset.xyz;
}

// Vertex `vertex` of the leaf the terrain's water lists at `instance`
// (`cs_water_list`): where the ground's leaf has its own vertex
// (`surface_vertex`), at its water's surface there, with what its water
// is (`water_at`).
fn water_output(vertex: u32, instance: u32) -> WaterOutput {
    let leaf = water_leaves[water_leaf_record.places.x + instance];
    let local = surface_vertex(surface_leaf(leaf), vertex);
    let at = water_at(local, terrain_true_height(local));
    let relative = water_relative(local, at.surface);
    var output: WaterOutput;
    output.clip_position = terrain_view.relative_view_projection * vec4<f32>(relative, 1.0);
    output.relative = relative;
    output.water = vec3<f32>(at.depth, at.words.flow);
    output.foam = vec3<f32>(min(at.words.foam, 1.0) * WATER_FOAM_STREAKED, at.words.width,
        select(0.0, 1.0, at.words.sheet));
    output.shore = at.shore;
    output.slope = at.slope;
    output.churned = at.words.foam;
    output.state = vec4<f32>(at.words.state, 0.0);
    output.obstacle = at.obstacle;
    output.reach = at.words.reach;
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

// The temporal pipelines' outputs of water `output`, `previous` where its
// point stood in the previous frame's view (x, y and w of its clip
// position).
fn water_temporal(output: WaterOutput, previous: vec3<f32>) -> WaterTemporalOutput {
    var temporal: WaterTemporalOutput;
    temporal.clip_position = output.clip_position;
    temporal.relative = output.relative;
    temporal.water = output.water;
    temporal.foam = output.foam;
    temporal.shore = output.shore;
    temporal.slope = output.slope;
    temporal.churned = output.churned;
    temporal.state = output.state;
    temporal.obstacle = output.obstacle;
    temporal.reach = output.reach;
    temporal.previous = previous;
    return temporal;
}

// Water's motion is the camera's: its ripples move within its surface.
@vertex
fn vs_water_temporal(@builtin(vertex_index) vertex: u32,
        @builtin(instance_index) instance: u32) -> WaterTemporalOutput {
    let output = water_output(vertex, instance);
    return water_temporal(output, (camera.previous_view_projection
        * vec4<f32>(eye_world(output.relative), 1.0)).xyw);
}

// The listing of terrains' water's leaves (`water.rs`): each frame one pass
// tests every leaf of each terrain's subdivision for water over it
// (`cs_water_leaves`), setting a bit for each that may hold some, and one
// workgroup lists those leaves in their order (`cs_water_list`), so the
// terrain's water draws the same leaves in the same order whatever order
// the test ran in. What it writes: those bits, each terrain's leaves listed,
// and its draw: its indirect arguments (a leaf's vertices, the leaves
// listed and zeros) and its counts (its subdivision's leaves, as their own
// draw counts them, those holding water, and zeros).
struct WaterDraw {
    args: vec4<u32>,
    counts: vec4<u32>,
}
@group(2) @binding(24) var<storage, read_write> water_leaf_bits: array<atomic<u32>>;
@group(2) @binding(25) var<storage, read_write> water_leaf_list: array<u32>;
@group(2) @binding(26) var<storage, read_write> water_leaf_draws: array<WaterDraw>;
// Leaves a workgroup of the test takes at a time, and words of bits the
// listing takes at once (`gpu_layouts::constants`).
const WATER_LEAF_GROUP: u32 = 64u;
const WATER_LIST_GROUP: u32 = 256u;
// The leaves the listing tests, and its words' counts as it sums them.
var<workgroup> water_list_tested: u32;
var<workgroup> water_list_counts: array<u32, WATER_LIST_GROUP>;

// Whether water may stand over the patch `place` of `level` (counted from
// the terrain's first corner): its bit in the tile of the finest level from
// `level` on resident there, whose bits hold what finer tiles under it hold;
// and wherever no level is.
fn water_patch_holds(place: vec2<u32>, level: u32) -> bool {
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
    for (var index = level; index < levels; index++) {
        let header = base + index * 4u;
        let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
        let at = place >> vec2<u32>(index - level);
        let tile = min(at / WATER_TILE_PATCHES, tiles - vec2<u32>(1u));
        let layer = terrain_directory[terrain_directory[header] + tile.y * tiles.x + tile.x];
        if layer == NOT_RESIDENT {
            continue;
        }
        let within = min(at - tile * WATER_TILE_PATCHES, vec2<u32>(WATER_TILE_PATCHES - 1u));
        let bit = within.y * WATER_TILE_PATCHES + within.x;
        let word = water_directory[layer * WATER_TILE_WORDS
            + WATER_TILE_PATCHES * WATER_TILE_PATCHES + bit / 32u];
        return (word & (1u << (bit % 32u))) != 0u;
    }
    return true;
}

// Whether water may stand over the leaf `triangle` (terrain-local meters):
// over any patch its bounds overlap, at the level whose patches are at
// least as wide as they are, so they overlap two along each side at most
// (`water_patch_holds`).
fn water_leaf_holds(triangle: SurfaceTriangle) -> bool {
    let low = max(min(min(triangle.a, triangle.b), triangle.c), vec2<f32>(0.0));
    let high = min(max(max(triangle.a, triangle.b), triangle.c), water_extent());
    if any(low > high) {
        return false;
    }
    let levels = terrain.directory.y;
    let quads = f32(WATER_BLOCK - 1u);
    let widest = max(high.x - low.x, high.y - low.y);
    let level = u32(clamp(ceil(log2(max(widest / (terrain.anchor.w * quads), 1.0))), 0.0,
        f32(levels - 1u)));
    let width = terrain.anchor.w * quads * f32(1u << level);
    let first = vec2<u32>(low / width);
    let last = vec2<u32>(high / width);
    for (var z = first.y; z <= last.y; z++) {
        for (var x = first.x; x <= last.x; x++) {
            if water_patch_holds(vec2<u32>(x, z), level) {
                return true;
            }
        }
    }
    return false;
}

// Sets the bit of each leaf of the terrain's subdivision over which water
// may stand (`water_leaf_holds`), of as many as its record tests: each
// invocation tests the leaves as many apart as the dispatch runs.
@compute @workgroup_size(WATER_LEAF_GROUP)
fn cs_water_leaves(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
    let places = water_leaf_record.places;
    let tested = min(water_leaf_draws[places.w].counts.x, places.z);
    let stride = groups.x * WATER_LEAF_GROUP;
    for (var leaf = id.x; leaf < tested; leaf += stride) {
        if water_leaf_holds(surface_leaf(leaf)) {
            atomicOr(&water_leaf_bits[places.y + leaf / 32u], 1u << (leaf % 32u));
        }
    }
}

// Lists the leaves of the terrain's subdivision whose bits are set, in their
// order, as many as its record lists, and writes its draw: the workgroup
// takes WATER_LIST_GROUP words of bits at a time, an invocation each, and
// places each word's leaves after those of the words before it, from the
// running sum of their counts (Hillis and Steele's scan, in workgroup
// memory).
@compute @workgroup_size(WATER_LIST_GROUP)
fn cs_water_list(@builtin(local_invocation_index) lane: u32) {
    let places = water_leaf_record.places;
    if lane == 0u {
        water_list_tested = min(water_leaf_draws[places.w].counts.x, places.z);
    }
    let words = (workgroupUniformLoad(&water_list_tested) + 31u) / 32u;
    var listed = 0u;
    for (var first = 0u; first < words; first += WATER_LIST_GROUP) {
        let at = first + lane;
        var word = 0u;
        if at < words {
            word = atomicLoad(&water_leaf_bits[places.y + at]);
        }
        let count = countOneBits(word);
        water_list_counts[lane] = count;
        workgroupBarrier();
        for (var offset = 1u; offset < WATER_LIST_GROUP; offset <<= 1u) {
            var before = 0u;
            if lane >= offset {
                before = water_list_counts[lane - offset];
            }
            workgroupBarrier();
            water_list_counts[lane] += before;
            workgroupBarrier();
        }
        var slot = listed + water_list_counts[lane] - count;
        for (var bit = 0u; bit < 32u; bit++) {
            if (word & (1u << bit)) != 0u {
                if slot < places.z {
                    water_leaf_list[places.x + slot] = at * 32u + bit;
                }
                slot += 1u;
            }
        }
        listed += water_list_counts[WATER_LIST_GROUP - 1u];
        // Every invocation read this round's total before the next
        // overwrites it.
        workgroupBarrier();
    }
    if lane == 0u {
        water_leaf_draws[places.w].args = vec4<u32>(SURFACE_MESHLET_VERTICES,
            min(listed, places.z), 0u, 0u);
        water_leaf_draws[places.w].counts.y = listed;
    }
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

// The triangle of a grid of equilateral triangles, skewed into squares,
// holding `grid` (in its cells' sides): its three corners (in the skewed
// grid) and the weights `grid` takes of each.
struct WaterTriangle {
    corners: array<vec2<i32>, 3>,
    weights: vec3<f32>,
}

fn water_triangle(grid: vec2<f32>) -> WaterTriangle {
    let skewed = vec2<f32>(grid.x, 1.15470054 * grid.y - 0.57735027 * grid.x);
    let cell = floor(skewed);
    let local = skewed - cell;
    let base = vec2<i32>(cell);
    let third = 1.0 - local.x - local.y;
    var triangle: WaterTriangle;
    triangle.weights = vec3<f32>(third, local.y, local.x);
    triangle.corners = array<vec2<i32>, 3>(base, base + vec2<i32>(0, 1), base + vec2<i32>(1, 0));
    if third <= 0.0 {
        triangle.weights = vec3<f32>(-third, 1.0 - local.y, 1.0 - local.x);
        triangle.corners = array<vec2<i32>, 3>(base + vec2<i32>(1, 1), base + vec2<i32>(1, 0),
            base + vec2<i32>(0, 1));
    }
    return triangle;
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
    let triangle = water_triangle(uv / WATER_PATCH);
    var corners = triangle.corners;
    let weights = triangle.weights;
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
    // As finely as the display shows them: the mip bias.
    let detail = material_detail();
    for (var tile = 0u; tile < WATER_WAVE_TILES; tile++) {
        let share = tiles[tile];
        if share <= 0.0 {
            continue;
        }
        let repeats = water.waves[tile];
        let read = water_tile(tile, p * repeats, dx * (repeats * detail),
            dy * (repeats * detail));
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
// How much more of the light its body scatters water lets out at the top of
// a crest of its chop, where that light crosses less water on its way out
// (wave peaks lit through, as Sea of Thieves lights them): half again.
const WATER_CREST_BODY: f32 = 0.5;

// What light keeps of each channel over `meters` of water `aerated` (0 to
// 1) with the bubbles churning water holds: what the water takes, and what
// its sediment and bubbles scatter.
fn water_transmitted(meters: f32, aerated: f32) -> vec3<f32> {
    return exp(-(water.absorption.rgb + water.sediment.w + WATER_BUBBLES * aerated) * meters);
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

// The normal that mirrors the eye's view along `view` no lower than
// WATER_HORIZON: `normal`, or, for slopes turned from the eye, the normal
// mirroring the view just over the horizon instead. Seen from under the
// water, below the plane its surface lies in (`plane`, its normal),
// `normal`.
fn water_facing(normal: vec3<f32>, view: vec3<f32>, plane: vec3<f32>) -> vec3<f32> {
    let mirrored = reflect(-view, normal);
    let across = length(mirrored.xz);
    if dot(view, plane) <= 0.0 || mirrored.y >= WATER_HORIZON || across < 1.0e-6 {
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
// pixel; its waves' normal and the roughness of what a pixel spans of them;
// the normal of the plane it lies in without them (up for the sea, tilted
// as terrains' water falls), which the eye looks down onto and light
// enters through; the share of it foam covers; the color its body scatters
// back, and what it keeps of each channel of the light on its way to what
// lies below and back; light it adds of its own (the sea's crests lit
// through); and how near the top of a crest of its chop it stands (0 to
// 1), where the light its body scatters crosses less water on its way out.
struct WaterSurface {
    relative: vec3<f32>,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
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
    glow: vec3<f32>,
    crest: f32,
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
// back, through what the surface lets out; and its foam, a white layer
// lit by its own relief, darker in its hollows and glinting where it lies
// wet. One shadow lookup serves the sun's three, and the sun's share of the
// light falling on the water.
fn water_light(surface: WaterSurface) -> WaterLit {
    let world = eye_world(surface.relative);
    let distance = length(surface.relative);
    let view = stable_normalize(-surface.relative);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let normal = water_facing(surface.normal, view, surface.plane);
    // Foam is rough where it heaps, and smoother where it lies thin and wet.
    let foam_roughness = mix(WATER_FOAM_ROUGH.y, WATER_FOAM_ROUGH.x, surface.foam_thin);
    let roughness = mix(surface.roughness, foam_roughness, surface.foam);
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
    let body = surface.scatter * (vec3<f32>(1.0) - surface.transmit)
        * water_scattered(view, sunlight, surface.plane) * (1.0 - reflectance)
        * (1.0 + WATER_CREST_BODY * surface.crest);
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
    // The mirrored sky and scene come through `finish`; the sun's light is
    // the water's own.
    var point: SurfacePoint;
    point.albedo = surface.scatter;
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

// Patterns carried along the flow (Vlachos's flow maps): ripples, foam's
// lace and white water, each drawn in two sets half a cycle apart, each
// set carried from half a cycle before its middle to half a cycle after,
// where it starts over unseen and anew in the noise, so it shows clearest
// where the flow has not yet stretched it (Neyret's latency). A cycle lasts
// as long as the flow's shear takes to strain the pattern by WATER_STRAIN,
// within its kind's bounds (seconds), in whole octaves: each place takes
// the two octaves nearest its own, crossfading over the middle share
// WATER_BAND_BLEND of each, so cycles change smoothly across the water.
// Strain fades what the flow has drawn out as it runs (Chentanez and
// Müller: e^(−Ω τ |∇u|), Ω being WATER_STRAIN_FADE).
const WATER_STRAIN: f32 = 0.8;
const WATER_CYCLES: vec2<f32> = vec2<f32>(0.25, 3.2);
const WATER_BAND_BLEND: vec2<f32> = vec2<f32>(0.25, 0.75);
const WATER_STRAIN_FADE: f32 = 1.0;
// The meters across the patches of water whose patterns start over
// together.
const WATER_FLOW_PHASE: f32 = 40.0;
// Foam's lace, which lies on calm water only: the meters across its larger
// cells (its finer ones a third of that), how wide (in cells) its bubbles
// string at the waterline, the meters from the waterline it reaches out
// and the meters of water over what stands in it within which it rings it;
// how wide the lace the foam water carries strings them at its most; and
// the meters a second over which it gives way as the water quickens, from
// the first to the second, a current tearing its bubbles off the shores and
// breaking up their chains.
const WATER_LACE_CELL: f32 = 0.24;
const WATER_SHORE_LACE: f32 = 0.42;
const WATER_FOAM_REACH: f32 = 1.6;
const WATER_FOAM_RING: f32 = 0.2;
const WATER_PLUNGE_LACE: f32 = 0.45;
const WATER_LACE_CALM: vec2<f32> = vec2<f32>(0.15, 0.5);
// Lace's bubbles: the cells apart their places stand along its chains, the
// share of the places holding none, and how much of its rim a bubble's
// middle shows, its film seen face on there and edge on at its rim; the
// share of its chains' width its bubbles cover where a pixel spans many;
// and how much of what lies under it lace hides: it is a single layer of
// bubbles.
const WATER_LACE_BEAD: f32 = 0.25;
const WATER_LACE_BROKEN: f32 = 0.3;
const WATER_LACE_FILM: f32 = 0.45;
const WATER_LACE_FILL: f32 = 0.5;
const WATER_LACE_OPACITY: f32 = 0.6;
// The share of narrow water's half width its shores' foam reaches over.
const WATER_FOAM_SHARE: f32 = 0.22;
// The slopes of a river's surface (meters a meter, from the first to the
// second) over which, under a fall's sheet, it comes to run as a film.
const WATER_FILM: vec2<f32> = vec2<f32>(0.5, 0.9);
// Seconds each of foam's two sets of lace lives on still water, half a
// life apart (a wave lapping the shore every half life), and the bounds of
// the cycles of the foam running water carries, longer than ripples' as
// foam lingers; the meters across which the place's clock for its lace
// varies, and the lives it varies by (more would draw out the lace a
// current carries); and the meters a wave lapping still water's shore
// washes its lace in.
const WATER_FOAM_LAPPED: f32 = 5.0;
const WATER_FOAM_CYCLES: vec2<f32> = vec2<f32>(1.0, 6.4);
const WATER_FOAM_STAGGER: f32 = 29.0;
const WATER_FOAM_STAGGERED: f32 = 0.8;
const WATER_FOAM_WASH: f32 = 0.35;

// The cycles a carried pattern of a kind cycling within `cycles` (seconds,
// the shortest and the longest) takes where the flow strains it by
// `strain` a second: the octave of the shortest at or below its own cycle
// (`first`, seconds, and `index`, octaves above the shortest), and how much
// the octave above it, twice as long, shows.
struct WaterBands {
    first: f32,
    index: u32,
    toward: f32,
}

fn water_bands(strain: f32, cycles: vec2<f32>) -> WaterBands {
    let own = clamp(WATER_STRAIN / max(strain, 1.0e-4), cycles.x, cycles.y);
    let octave = log2(own / cycles.x);
    let below = floor(octave);
    var bands: WaterBands;
    bands.first = cycles.x * exp2(below);
    bands.index = u32(below);
    bands.toward = smoothstep(WATER_BAND_BLEND.x, WATER_BAND_BLEND.y, octave - below);
    return bands;
}

// One of the two sets (`which`) of a pattern carried in cycles of `cycle`
// seconds, at `time`, the place's clock running `stagger` cycles ahead:
// the seconds it has been carried from its middle (negative before it),
// how much it shows, how far through its cycle it is (0 as it starts) and
// where it lies in the noise, anew each cycle and salted by `salt` (up to
// 64 of the noise's units along each axis).
struct WaterSet {
    seconds: f32,
    weight: f32,
    age: f32,
    offset: vec2<f32>,
}

fn water_set(time: f32, stagger: f32, cycle: f32, which: u32, salt: u32) -> WaterSet {
    let shifted = time / cycle + stagger + 0.5 * f32(which);
    let life = floor(shifted);
    var carried: WaterSet;
    carried.age = shifted - life;
    carried.seconds = (carried.age - 0.5) * cycle;
    carried.weight = 1.0 - abs(2.0 * carried.age - 1.0);
    let key = ((u32(life) * 2u + which) * 0x9e3779b9u) ^ salt;
    carried.offset = vec2<f32>(terrain_hash_of(key), terrain_hash_of(key ^ 0x68e31da4u)) * 64.0;
    return carried;
}

// The edge between the cells of a jittered grid nearest `p` (in cells),
// the bisector of the two of their points nearest it (cellular noise,
// Worley): how far `p` lies from it (cells, 0 on it), how far along it
// from the two points' middle (cells, counted the same way from either side
// of it), and a key the two cells it parts share. Foam strings its lace's
// bubbles along these edges; they hold still, and the lace moves only as
// the water carries it.
struct WaterCellEdge {
    apart: f32,
    along: f32,
    key: u32,
}

fn water_cell_edge(p: vec2<f32>) -> WaterCellEdge {
    let cell = floor(p);
    let local = p - cell;
    let base = vec2<i32>(cell);
    var nearest = 8.0;
    var next = 8.0;
    var nearest_point = vec2<f32>(0.0);
    var next_point = vec2<f32>(0.0);
    var nearest_key = 0u;
    var next_key = 0u;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let at = vec2<u32>(base + vec2<i32>(x, y));
            let key = (at.x * 0x8da6b343u) ^ (at.y * 0xd8163841u);
            let seed = vec2<f32>(terrain_hash_of(key), terrain_hash_of(key ^ 0x68e31da4u));
            let point = vec2<f32>(f32(x), f32(y)) + 0.15 + 0.7 * seed;
            let distance = length(point - local);
            if distance < nearest {
                next = nearest;
                next_point = nearest_point;
                next_key = nearest_key;
                nearest = distance;
                nearest_point = point;
                nearest_key = key;
            } else if distance < next {
                next = distance;
                next_point = point;
                next_key = key;
            }
        }
    }
    // The way along the edge, the same from either side of it: the line from
    // the lower key's point to the higher's, turned a quarter.
    let between = next_point - nearest_point;
    let span = max(length(between), 1.0e-6);
    let way = vec2<f32>(-between.y, between.x)
        * (select(-1.0, 1.0, nearest_key < next_key) / span);
    var edge: WaterCellEdge;
    edge.apart = (next * next - nearest * nearest) / (2.0 * span);
    edge.along = dot(local - 0.5 * (nearest_point + next_point), way);
    edge.key = (min(nearest_key, next_key) * 0x9e3779b9u)
        ^ (max(nearest_key, next_key) * 0x85ebca6bu);
    return edge;
}

// Lace's bubbles strung along edge `edge` (`water_cell_edge`'s), the
// largest `width` cells across, for a pixel spanning `blur` cells: a chain
// of round bubbles, one to each of the places WATER_LACE_BEAD cells apart
// along it, each nudged along it and sized at random, the chain broken
// where WATER_LACE_BROKEN of the places hold none; their middles showing
// WATER_LACE_FILM of their rims; sharp where the pixel spans a small part
// of one, as bright as their share of the pixel as they shrink below it.
fn water_lace_beads(edge: WaterCellEdge, width: f32, blur: f32) -> f32 {
    let soft = max(2.0 * blur, 1.0e-4);
    let place = floor(edge.along / WATER_LACE_BEAD);
    var lace = 0.0;
    for (var beside = -1; beside <= 1; beside++) {
        let index = place + f32(beside);
        let key = edge.key ^ (u32(i32(index)) * 0x9e3779b9u);
        if terrain_hash_of(key ^ 0x2c1b3c6du) < WATER_LACE_BROKEN {
            continue;
        }
        let middle = (index + 0.5 + 0.6 * (terrain_hash_of(key) - 0.5)) * WATER_LACE_BEAD;
        let radius = min(0.5 * width * mix(0.4, 1.0, terrain_hash_of(key ^ 0x68e31da4u)),
            0.6 * WATER_LACE_BEAD);
        let distance = length(vec2<f32>(edge.along - middle, edge.apart));
        let toward_rim = saturate(distance / max(radius, 1.0e-6));
        lace = max(lace, (1.0 - smoothstep(radius - soft, radius + soft, distance))
            * min(1.0, radius / soft + 0.25) * mix(WATER_LACE_FILM, 1.0, toward_rim * toward_rim));
    }
    return lace;
}

// One set of foam's lace at `p` (in its larger cells), its bubbles at most
// `width` cells across for a pixel spanning `blur` cells, `worn` through
// its life: chains of bubbles strung along the cells' edges, bent by smooth
// noise at two scales so they wander and curl, their bubbles larger over
// clumps a few cells across, where they gather, and smaller between, the
// chains parting here and there, the more the older it is, and finer
// cells' smaller bubbles within them, as much as `finer` shows them (0 to
// 1: up close).
fn water_lace_set(p: vec2<f32>, width: f32, blur: f32, finer: f32, worn: f32) -> f32 {
    let curl = vec2<f32>(water_noise(p * 0.9).x, water_noise(p * 0.9 + vec2<f32>(5.3, 1.7)).x)
        - 0.5;
    let sway = vec2<f32>(water_noise(p * 0.3 + vec2<f32>(8.1, 2.2)).x,
        water_noise(p * 0.3 + vec2<f32>(1.4, 9.6)).x) - 0.5;
    let bent = p + curl * 0.6 + sway * 1.6;
    let clumps = water_noise(p * 0.2 + vec2<f32>(2.9, 8.3)).x;
    let parted = smoothstep(0.22 + 0.25 * worn, 0.55 + 0.15 * worn,
        water_noise(bent * 1.4 + vec2<f32>(4.4, 0.7)).x);
    let gathered = min(width * (0.2 + 1.6 * clumps * clumps) * (0.3 + 0.7 * parted), 0.4);
    var lace = water_lace_beads(water_cell_edge(bent), gathered, blur);
    if finer > 0.0 {
        let fine = water_lace_beads(water_cell_edge(bent * 3.0 + 7.1), gathered * 0.55,
            blur * 3.0);
        lace = max(lace, fine * 0.7 * finer);
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

// A set of foam's lace (`which`) on still water at `p` (meters), where its
// clock reads `clock` (lives): a wave lapping the shore, washing its lace in
// toward it (against `offshore`), as big as `lapping` says, and drawing it
// back out as it dissolves; its bubbles up to `lining` cells across along
// the waterline and, where the water carries foam, `plunge` of
// WATER_PLUNGE_LACE: how wide (cells) it strings its bubbles, and its lace
// for a pixel spanning `blur` cells, `finer` of its finer cells shown
// (`water_lace_set`), where `drawn`.
fn water_lapped_lace(which: u32, p: vec2<f32>, clock: f32, lining: f32, plunge: f32,
        offshore: vec2<f32>, lapping: f32, blur: f32, finer: f32, drawn: bool) -> vec2<f32> {
    let lived = water_foam_set(clock, which);
    let age = lived.age;
    let swept = 1.0 - abs(2.0 * age - 1.0);
    let lapped = smoothstep(0.0, 0.12, age) * (1.0 - smoothstep(0.25, 1.0, age))
        * lived.strength * lapping;
    let width = max(lining * lapped, WATER_PLUNGE_LACE * plunge * swept);
    let at = p + offshore * (WATER_FOAM_WASH * smoothstep(0.0, 0.18, age)
        * (1.0 - smoothstep(0.18, 1.0, age)));
    var lace = 0.0;
    if drawn && width > 1.0e-3 {
        lace = water_lace_set(at / WATER_LACE_CELL + lived.offset, width, blur, finer, age);
    }
    return vec2<f32>(width, lace);
}

// Foam's lace on slowly moving water at `p` (meters), its bubbles up to
// `width` cells across, carried along `current` (meters a second) shearing
// `shear` a second, at `time`, the place's clock running `stagger` cycles
// ahead: in sets carried as ripples are, cycling within WATER_FOAM_CYCLES,
// each forming as it starts and dissolving as it ends, its bubbles as large
// as it shows, so the sets' bubbles together are as large as the lace's
// and stay sharp; and the smaller the more the flow has strained them. How
// wide (cells) its bubbles string altogether, and its lace for a pixel
// spanning `blur` cells, `finer` of its finer cells shown
// (`water_lace_set`), where `drawn`.
fn water_carried_lace(p: vec2<f32>, current: vec2<f32>, shear: f32, time: f32, stagger: f32,
        width: f32, blur: f32, finer: f32, drawn: bool) -> vec2<f32> {
    let bands = water_bands(shear, WATER_FOAM_CYCLES);
    var covered = 0.0;
    var lace = 0.0;
    for (var band = 0u; band < 2u; band++) {
        let shown = select(1.0 - bands.toward, bands.toward, band == 1u);
        if shown <= 0.0 {
            continue;
        }
        let cycle = bands.first * f32(1u + band);
        let salt = ((bands.index + band) * 0x632be5abu) ^ 0x27d4eb2fu;
        for (var which = 0u; which < 2u; which++) {
            let carried = water_set(time, stagger, cycle, which, salt);
            let strung = width * shown * carried.weight
                * exp(-WATER_STRAIN_FADE * abs(carried.seconds) * shear);
            covered += strung;
            if drawn && strung > 1.0e-3 {
                let at = p - current * carried.seconds;
                lace = max(lace, water_lace_set(at / WATER_LACE_CELL + carried.offset, strung,
                    blur, finer, carried.age));
            }
        }
    }
    return vec2<f32>(covered, lace);
}

// The wind's finest waves where water flows along `flow` (meters a second),
// shearing `shear` a second, `share` of them as `water_waves_at` gives
// them, at `time`, the place's clock running `stagger` cycles ahead: the
// finest ride the current (Gonzalez-Ochoa, Uncharted 4's rapids), so they
// are carried along with it in sets cycling as fast as its strain asks,
// faded as the flow strains them and blended keeping their spread,
// Σwᵢxᵢ/√Σwᵢ² of slopes whose mean is none (Yu, Neyret, Bruneton and
// Holzschuch). What strain fades of their slopes hides in the variance,
// roughening the water instead.
fn water_flowing_waves(p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, share: f32,
        flow: vec2<f32>, shear: f32, time: f32, stagger: f32) -> vec3<f32> {
    let tile = WATER_WAVE_TILES - 1u;
    let repeats = water.waves[tile];
    // As finely as the display shows them: the mip bias.
    let detail = material_detail();
    let tile_dx = dx * (repeats * detail);
    let tile_dy = dy * (repeats * detail);
    let bands = water_bands(shear, WATER_CYCLES);
    var slopes = vec2<f32>(0.0);
    var variance = 0.0;
    var weights = 0.0;
    for (var band = 0u; band < 2u; band++) {
        let shown = select(1.0 - bands.toward, bands.toward, band == 1u);
        if shown <= 0.0 {
            continue;
        }
        let cycle = bands.first * f32(1u + band);
        let salt = (tile * 0xcb1ab31fu) ^ ((bands.index + band) * 0x632be5abu);
        for (var which = 0u; which < 2u; which++) {
            let carried = water_set(time, stagger, cycle, which, salt);
            let weight = shown * carried.weight;
            if weight <= 0.0 {
                continue;
            }
            let at = (p - flow * carried.seconds) * repeats + carried.offset;
            let read = water_tile(tile, at, tile_dx, tile_dy);
            let fade = exp(-WATER_STRAIN_FADE * abs(carried.seconds) * shear);
            slopes += read.xy * (weight * fade);
            variance += (read.z + (1.0 - fade * fade) * dot(read.xy, read.xy)) * weight * weight;
            weights += weight * weight;
        }
    }
    let kept = max(weights, 1.0e-6);
    return vec3<f32>(slopes * (inverseSqrt(kept) * share), variance / kept * share * share);
}

// Standing waves (Neyret and Praizelin; Yu, Neyret and Steed): the least
// radius (meters) of what stands in the water, and the least speed (meters
// a second) of the water flowing past it, that raise any; the pixels a
// standing wave must span to show, finer ones roughening the water
// instead; how far upstream of its obstacle's middle a pillow heaps up, how
// long it runs along the flow and how high it rises at most, in radii (the
// water it heaps up spills round the obstacle); a front's crest's half width,
// in depths and at least (meters), and the radii over which fronts fall off
// downstream; the share of the velocity head the waves standing past
// obstacles in slower water rise by, how much their wake widens a meter
// downstream, and the wavelengths over which trains fall off; and the
// length and height of an undular jump's waves, in depths.
const WATER_STANDING_LEAST: vec2<f32> = vec2<f32>(0.0625, 0.2);
const WATER_STANDING_PIXELS: f32 = 3.0;
const WATER_PILLOW: vec3<f32> = vec3<f32>(0.5, 0.7, 0.5);
const WATER_CREST: vec2<f32> = vec2<f32>(2.0, 0.03);
const WATER_FRONT_REACH: f32 = 8.0;
const WATER_TRAIN: f32 = 0.3;
const WATER_WAKE: f32 = 0.25;
const WATER_TRAIN_REACH: f32 = 6.0;
const WATER_UNDULAR: vec2<f32> = vec2<f32>(5.0, 0.2);
// How much standing waves breathe either way as slow noise
// WATER_BREATH_SIZE meters across passes over them, carried at
// WATER_BREATH_CARRIED of the flow in sets cycling every WATER_BREATH_CYCLE
// seconds.
const WATER_BREATHING: f32 = 0.3;
const WATER_BREATH_SIZE: f32 = 3.0;
const WATER_BREATH_CARRIED: f32 = 0.5;
const WATER_BREATH_CYCLE: f32 = 6.0;

// How high standing waves rise at `p` (meters along x and z), as a share of
// their height about 1, where water flows along `flow` (meters a second),
// at `time`, the place's clock running `stagger` cycles ahead.
fn water_breathing(p: vec2<f32>, flow: vec2<f32>, time: f32, stagger: f32) -> f32 {
    var sum = 0.0;
    var weights = 0.0;
    for (var which = 0u; which < 2u; which++) {
        let carried = water_set(time, stagger, WATER_BREATH_CYCLE, which, 0x2c1b3c6du);
        let at = (p - flow * (WATER_BREATH_CARRIED * carried.seconds)) / WATER_BREATH_SIZE
            + carried.offset;
        sum += (water_noise(at).x - 0.5) * carried.weight;
        weights += carried.weight * carried.weight;
    }
    return 1.0 + 2.0 * WATER_BREATHING * sum * inverseSqrt(max(weights, 1.0e-6));
}

// A standing wave's slopes `gradient` (meters a meter), its features
// `width` meters across, as a pixel spanning `footprint` meters seen from
// the detail eye shows them: whole where they span WATER_STANDING_PIXELS
// pixels, handing over to the roughness of what the pixel hides as they
// narrow below that (xy: the slopes kept, z: the variance hidden).
fn water_resolved(gradient: vec2<f32>, width: f32, footprint: f32) -> vec3<f32> {
    let kept = saturate(width / (WATER_STANDING_PIXELS * max(footprint, 1.0e-6)));
    return vec3<f32>(gradient * kept, dot(gradient, gradient) * (1.0 - kept * kept));
}

// The waves standing in water `depth` meters deep flowing along `flow`
// (meters a second) at Froude number `froude`, about the obstacle nearest
// it, `obstacle` (its offset from the point along x and z and its radius,
// meters, and how strongly a drop or jump lies just upstream), rising
// `breathing` of their height, for a pixel spanning `footprint` meters seen
// from the detail eye (`water_resolved`): slopes of the surface fixed where
// the bed is, which the flow never smears (xy, meters a meter), and the
// variance of those finer than the pixel (z). A pillow heaps up just
// upstream of the obstacle, as wide as it and as high as the water's
// velocity head U²/2g, or WATER_PILLOW.z of its radius; in supercritical
// water, a front trails each side at the half
// angle asin(1/Fr) (30° at Froude 2), a narrow crest falling off
// downstream; in slower water, waves stand in its wake 2πU²/g apart, the
// length of the waves its current holds still; and below a drop or jump,
// where the water runs slower than Froude 1.7 as below an undular jump, a
// train stands of waves about five depths long.
fn water_standing(obstacle: vec4<f32>, flow: vec2<f32>, depth: f32, froude: f32,
        breathing: f32, footprint: f32) -> vec3<f32> {
    let radius = obstacle.z;
    let speed = length(flow);
    if radius < WATER_STANDING_LEAST.x || speed < WATER_STANDING_LEAST.y {
        return vec3<f32>(0.0);
    }
    let heading = flow / speed;
    let side = vec2<f32>(-heading.y, heading.x);
    // Meters downstream of the obstacle's middle, and across the flow.
    let apart = -obstacle.xy;
    let along = dot(apart, heading);
    let across = dot(apart, side);
    let head = speed * speed / (2.0 * WATER_GRAVITY);
    let run = WATER_PILLOW.y * radius;
    let ahead = (along + WATER_PILLOW.x * radius) / run;
    let wide = across / radius;
    let pillow = min(head, WATER_PILLOW.z * radius) * breathing
        * exp(-ahead * ahead - wide * wide);
    var standing = water_resolved(
        (heading * (ahead / run) + side * (wide / radius)) * (-2.0 * pillow), run, footprint);
    let supercritical = smoothstep(1.0, 1.3, froude);
    if supercritical > 0.0 {
        let sine = 1.0 / max(froude, 1.0);
        let cosine = sqrt(max(1.0 - sine * sine, 0.0));
        // Meters out from the front's crest, and the way out.
        let beside = (abs(across) - radius) * cosine - along * sine;
        let outward = side * (cosine * select(-1.0, 1.0, across >= 0.0)) - heading * sine;
        let crest = max(WATER_CREST.x * depth, WATER_CREST.y);
        let trailing = smoothstep(-radius, 0.0, along)
            * exp(-max(along, 0.0) / (WATER_FRONT_REACH * radius));
        let front = min(head, depth) * breathing * supercritical * trailing
            * exp(-(beside / crest) * (beside / crest));
        standing += water_resolved(outward * (-2.0 * front * beside / (crest * crest)), crest,
            footprint);
    }
    let subcritical = 1.0 - smoothstep(0.8, 1.0, froude);
    let past = along - radius;
    if subcritical > 0.0 && past > 0.0 {
        let wavelength = 4.0 * PI * head;
        let wake = radius + WATER_WAKE * past;
        let height = WATER_TRAIN * head * breathing * subcritical * smoothstep(0.0, radius, past)
            * exp(-past / (WATER_TRAIN_REACH * wavelength) - (across / wake) * (across / wake));
        let rate = 2.0 * PI / wavelength;
        standing += water_resolved(heading * (-height * rate * sin(rate * along)),
            0.25 * wavelength, footprint);
    }
    let undular = obstacle.w * (1.0 - smoothstep(1.4, 1.7, froude));
    if undular > 0.0 {
        let wavelength = WATER_UNDULAR.x * max(depth, WATER_THINNEST);
        let height = WATER_UNDULAR.y * depth * undular * breathing
            * exp(-length(apart) / (WATER_TRAIN_REACH * wavelength));
        let rate = 2.0 * PI / wavelength;
        standing += water_resolved(heading * (-height * rate * sin(rate * along)),
            0.25 * wavelength, footprint);
    }
    return standing;
}

// The chop running water's current makes (`water_chop`). The turbulence's
// imprint it carries: eddies and boils span one to two depths, so its
// wavelength is WATER_CHOP_DEPTHS of the depth, within WATER_CHOP_SHORTEST
// and WATER_CHOP_LONGEST meters, with an octave WATER_CHOP_OCTAVE times
// shorter riding it, WATER_CHOP_FINE as steep; and the root mean square of
// its slopes on turbulent water running a meter a second
// (WATER_CHOP_SLOPE): a quarter of that on still water, rising by three
// quarters of it a meter a second up to 2 m/s, as Catlike Coding's flow
// tutorials scale chop with the flow's speed, and half of it where the
// water runs calm.
const WATER_CHOP_DEPTHS: f32 = 2.0;
const WATER_CHOP_SHORTEST: f32 = 0.08;
const WATER_CHOP_LONGEST: f32 = 1.2;
const WATER_CHOP_OCTAVE: f32 = 2.3;
const WATER_CHOP_FINE: f32 = 0.6;
const WATER_CHOP_SLOPE: f32 = 0.35;
// Short waves standing on the bed where water runs near or above the speed
// of its own waves (`water_standing_chop`): waves running upstream as fast
// as the current carries them down stand still, deep-water waves 2πU²/g
// long, within WATER_STAND_SHORTEST and WATER_STAND_LONGEST meters, drawn
// WATER_STAND_STRETCH times as long across the flow as along it so their
// crests lie across it; the Froude numbers over which they rise, from none
// to all; and the root mean square of their slopes once risen on turbulent
// water (WATER_STAND_SLOPE), three fifths of that where the water runs
// calm.
const WATER_STAND_SHORTEST: f32 = 0.15;
const WATER_STAND_LONGEST: f32 = 3.0;
const WATER_STAND_STRETCH: f32 = 2.5;
const WATER_STAND_FROUDE: vec2<f32> = vec2<f32>(0.6, 1.2);
const WATER_STAND_SLOPE: f32 = 0.3;
// Chop's shape (`water_ridge`): value noise about its middle, s, ridged as
// (1 − √(s² + e²))², sharp where the noise crosses its middle and broad in
// its troughs, as steep waves' crests are and their troughs are not
// (Musgrave's ridged noise). The noise's cells a wavelength
// spans: value noise crosses its middle about every second cell. The root
// mean square of a ridge's slopes, its heights a cell: value noise's
// quintic blend of uniform lattice values gives its gradient a mean square
// of 0.37 a cell squared, which the ridge and its square raise about
// eightfold. The least share of the noise its crests round over, e; and
// how far each wavelength's lattice turns from the next shorter's (the
// golden angle), so their axes never line up.
const WATER_CHOP_CELLS: f32 = 2.0;
const WATER_CHOP_RIDGE: f32 = 1.7;
const WATER_CHOP_ROUND: f32 = 0.06;
const WATER_CHOP_TURN: f32 = 2.3999632;
// Chop's crests: the root mean square of its slopes at which they count
// wholly, the steepest a wave stands before it breaks (Stokes's limit, a
// height a seventh of its length: a sine's slopes of 0.45 at most, 0.32 in
// root mean square); the share of their height over which they rise to
// their tops (`water_ridge`'s w, 1 on the crest), from the first to the
// second; and the share of the surface they cover, which a pixel spanning
// many of them shows.
const WATER_CHOP_BREAKING: f32 = 0.32;
const WATER_CHOP_CREST: vec2<f32> = vec2<f32>(0.75, 0.95);
const WATER_CHOP_COVER: f32 = 0.2;

// The chop a current makes: its slopes (xy, meters a meter) and the
// variance of those finer than the pixel (z), as `water_waves_at` gives
// waves; and how near the tops of its crests the water stands (0 to 1).
struct WaterChop {
    waves: vec3<f32>,
    crest: f32,
}

// Chop summed over its parts: their slopes and the variance of those finer
// than the pixel (xyz, as `WaterChop` holds them); and, of their heights
// (meters), the sum over what the pixel resolves of them, each weighted by
// how near the top of its crest the water stands (`tops`) and not
// (`resolved`), and the sum of them all (`heights`).
struct WaterChopSum {
    waves: vec3<f32>,
    tops: f32,
    resolved: f32,
    heights: f32,
}

// The ridge of noise about its middle `centered` (about -1 to 1), which
// changes by `gradient` a meter, for a pixel spanning `pixel` meters:
// (1 − √(s² + e²))², its crest rounded over e, WATER_CHOP_ROUND of the
// noise or what the pixel spans of it if more (at most half), so no crest
// is sharper than the pixel shows. Its slopes for a ridge a meter high (xy,
// meters a meter); the square of those its rounding hides within the
// pixel, where the pixel spans both flanks of its crest (z); and its
// height as a share of its crest's (w, 1 on the crest).
fn water_ridge(centered: f32, gradient: vec2<f32>, pixel: f32) -> vec4<f32> {
    let rate = length(gradient);
    let rounding = min(max(WATER_CHOP_ROUND, rate * pixel), 0.5);
    let rounded = sqrt(centered * centered + rounding * rounding);
    let ridge = max(1.0 - rounded, 0.0);
    let flank = 2.0 * ridge * rate;
    let top = ridge / (1.0 - rounding);
    return vec4<f32>(gradient * (-2.0 * ridge * centered / rounded),
        flank * flank * (rounding * rounding - WATER_CHOP_ROUND * WATER_CHOP_ROUND)
            / (rounded * rounded), top * top);
}

// Chop `sum` with a part of it added, `height` meters high, the root mean
// square of its slopes `slope`, shaped as `ridge` says (`water_ridge`'s)
// and `kept` of it resolved: its slopes as far as the pixel resolves them,
// and in the variance the rest of their spread and what its rounding
// hides.
fn water_chop_part(sum: WaterChopSum, ridge: vec4<f32>, height: f32, slope: f32,
        kept: f32) -> WaterChopSum {
    var added = sum;
    added.waves += vec3<f32>(ridge.xy * (height * kept),
        ridge.z * height * height * kept * kept + slope * slope * (1.0 - kept * kept));
    added.tops += ridge.w * height * kept;
    added.resolved += height * kept;
    added.heights += height;
    return added;
}

// The turbulence's imprint water flowing along `flow` (meters a second),
// `depth` meters deep and shearing `shear` a second, carries at `time`, the
// place's clock running `stagger` cycles ahead, the root mean square of its
// slopes `slope`, for a pixel spanning `pixel` meters: ridged noise
// (`water_ridge`) of the wavelength its depth sets, an octave
// WATER_CHOP_OCTAVE times shorter riding it. Each is drawn from a ladder of
// fixed wavelengths WATER_CHOP_OCTAVE apart, the two rungs nearest it
// crossfading over WATER_BAND_BLEND as cycles' bands do, so the noise never
// stretches where the depth changes; each rung's lattice turns
// WATER_CHOP_TURN from the next shorter's. It is carried along with the
// flow in sets as ripples are (`water_flowing_waves`), the sets' noise
// blended keeping its spread before it is ridged, so its crests stay sharp.
// What strain fades of its slopes, and what the pixel cannot resolve of a
// rung whose wavelength spans fewer than WATER_WAVE_PIXELS pixels, hides in
// the variance.
fn water_carried_chop(p: vec2<f32>, flow: vec2<f32>, depth: f32, shear: f32, time: f32,
        stagger: f32, slope: f32, pixel: f32) -> WaterChopSum {
    let wavelength = clamp(WATER_CHOP_DEPTHS * depth, WATER_CHOP_SHORTEST, WATER_CHOP_LONGEST);
    let octave = log(wavelength / WATER_CHOP_SHORTEST) / log(WATER_CHOP_OCTAVE);
    let below = floor(octave);
    let toward = smoothstep(WATER_BAND_BLEND.x, WATER_BAND_BLEND.y, octave - below);
    // Each rung's share of the slopes' square, the shortest first: the
    // finer octave's, the coarser's or the finer's, and the coarser's.
    let fine = WATER_CHOP_FINE * WATER_CHOP_FINE / (1.0 + WATER_CHOP_FINE * WATER_CHOP_FINE);
    let shares = vec3<f32>(fine * (1.0 - toward), mix(1.0 - fine, fine, toward),
        (1.0 - fine) * toward);
    // Each rung's meters across a cell of its noise, how much of it the
    // pixel resolves, and its lattice's turn (cosine and sine).
    var cells = vec3<f32>(0.0);
    var kept = vec3<f32>(0.0);
    var turns: array<vec2<f32>, 3>;
    var drawn = false;
    for (var rung = 0u; rung < 3u; rung++) {
        let meters = WATER_CHOP_SHORTEST * pow(WATER_CHOP_OCTAVE, below + f32(rung) - 1.0);
        cells[rung] = meters / WATER_CHOP_CELLS;
        if shares[rung] > 0.0 {
            kept[rung] = smoothstep(0.5 * WATER_WAVE_PIXELS, WATER_WAVE_PIXELS, meters / pixel);
            let turn = (below + f32(rung)) * WATER_CHOP_TURN;
            turns[rung] = vec2<f32>(cos(turn), sin(turn));
            drawn = drawn || kept[rung] > 0.0;
        }
    }
    // Each rung's noise about its middle and its gradient (in its lattice's
    // cells) over the sets, each weighted as it shows and as strain leaves
    // it; and the sums of those weights' squares, without strain and with.
    var centered = vec3<f32>(0.0);
    var gradients: array<vec2<f32>, 3>;
    var weights = 0.0;
    var fadings = 0.0;
    if drawn {
        let bands = water_bands(shear, WATER_CYCLES);
        for (var band = 0u; band < 2u; band++) {
            let shown = select(1.0 - bands.toward, bands.toward, band == 1u);
            if shown <= 0.0 {
                continue;
            }
            let cycle = bands.first * f32(1u + band);
            let salt = ((bands.index + band) * 0x632be5abu) ^ 0x3c6ef372u;
            for (var which = 0u; which < 2u; which++) {
                let carried = water_set(time, stagger, cycle, which, salt);
                let weight = shown * carried.weight;
                if weight <= 0.0 {
                    continue;
                }
                let fading = weight * exp(-WATER_STRAIN_FADE * abs(carried.seconds) * shear);
                let at = p - flow * carried.seconds;
                for (var rung = 0u; rung < 3u; rung++) {
                    if kept[rung] <= 0.0 {
                        continue;
                    }
                    let turn = turns[rung];
                    let scaled = at / cells[rung];
                    let noise = water_noise(vec2<f32>(turn.x * scaled.x - turn.y * scaled.y,
                        turn.y * scaled.x + turn.x * scaled.y) + carried.offset);
                    centered[rung] += fading * (noise.x - 0.5);
                    gradients[rung] += fading * noise.yz;
                }
                weights += weight * weight;
                fadings += fading * fading;
            }
        }
    }
    // The sets' noise blended keeping its spread, from -1 to 1 about its
    // middle, and the share of its slopes strain leaves.
    let blend = 2.0 * inverseSqrt(max(fadings, 1.0e-12));
    let strained = sqrt(fadings / max(weights, 1.0e-12));
    var sum = WaterChopSum(vec3<f32>(0.0), 0.0, 0.0, 0.0);
    for (var rung = 0u; rung < 3u; rung++) {
        if shares[rung] <= 0.0 {
            continue;
        }
        let part = slope * sqrt(shares[rung]);
        var ridge = vec4<f32>(0.0);
        if kept[rung] > 0.0 {
            // Its gradient a meter, turned back from its lattice's.
            let turn = turns[rung];
            let turned = gradients[rung] * (blend / cells[rung]);
            ridge = water_ridge(centered[rung] * blend, vec2<f32>(
                turn.x * turned.x + turn.y * turned.y, turn.x * turned.y - turn.y * turned.x),
                pixel);
        }
        sum = water_chop_part(sum, ridge, part * cells[rung] / WATER_CHOP_RIDGE, part,
            kept[rung] * strained);
    }
    return sum;
}

// Short waves standing on the bed where water flows along `flow` (meters a
// second) near or above the speed of its own waves, the root mean square of
// their slopes `slope`, for a pixel spanning `pixel` meters: ridged noise
// (`water_ridge`) 2πU²/g long along the flow and WATER_STAND_STRETCH times
// as long across it, drawn from a ladder of fixed wavelengths an octave
// apart as `water_carried_chop` draws its own. They stand where the bed
// is, never carried. Each rung is drawn in cells a wavelength across, in
// four grids half a cell apart, each read turned to run along the flow
// about its cell's middle, so it shears no more than the flow turns across
// a cell, and the four blended by weights falling from their cells'
// middles to none at their edges, keeping their spread (Catlike Coding's
// directional flow).
fn water_standing_chop(p: vec2<f32>, flow: vec2<f32>, slope: f32, pixel: f32) -> WaterChopSum {
    let speed = length(flow);
    let heading = flow / max(speed, 1.0e-6);
    let side = vec2<f32>(-heading.y, heading.x);
    let wavelength = clamp(2.0 * PI * speed * speed / WATER_GRAVITY, WATER_STAND_SHORTEST,
        WATER_STAND_LONGEST);
    let octave = log2(wavelength / WATER_STAND_SHORTEST);
    let below = floor(octave);
    let toward = smoothstep(WATER_BAND_BLEND.x, WATER_BAND_BLEND.y, octave - below);
    // The root mean square of the stretched ridge's slopes, its heights a
    // cell of its noise along the flow.
    let spread = WATER_CHOP_RIDGE
        * sqrt(0.5 + 0.5 / (WATER_STAND_STRETCH * WATER_STAND_STRETCH));
    var sum = WaterChopSum(vec3<f32>(0.0), 0.0, 0.0, 0.0);
    for (var rung = 0u; rung < 2u; rung++) {
        let share = select(1.0 - toward, toward, rung == 1u);
        if share <= 0.0 {
            continue;
        }
        let meters = WATER_STAND_SHORTEST * exp2(below + f32(rung));
        // Meters across a cell of its noise along the flow.
        let along = meters / WATER_CHOP_CELLS;
        let part = slope * sqrt(share);
        let kept = smoothstep(0.5 * WATER_WAVE_PIXELS, WATER_WAVE_PIXELS, meters / pixel);
        var ridge = vec4<f32>(0.0);
        if kept > 0.0 {
            let salt = 4u * (u32(below) + rung);
            var centered = 0.0;
            var gradient = vec2<f32>(0.0);
            var weights = 0.0;
            for (var grid = 0u; grid < 4u; grid++) {
                let at = p / meters + vec2<f32>(f32(grid & 1u), f32(grid >> 1u)) * 0.5;
                let cell = floor(at);
                let local = at - cell;
                let tent = 1.0 - abs(2.0 * local - 1.0);
                let weight = tent.x * tent.y;
                if weight <= 0.0 {
                    continue;
                }
                let offset = (local - 0.5) * meters;
                let noise = water_noise(vec2<f32>(dot(offset, heading),
                    dot(offset, side) / WATER_STAND_STRETCH) / along
                    + water_corner_offset(vec2<i32>(cell), salt + grid) * 64.0);
                centered += weight * (noise.x - 0.5);
                gradient += weight * (heading * noise.y + side * (noise.z / WATER_STAND_STRETCH));
                weights += weight * weight;
            }
            let blend = 2.0 * inverseSqrt(max(weights, 1.0e-12));
            ridge = water_ridge(centered * blend, gradient * (blend / along), pixel);
        }
        sum = water_chop_part(sum, ridge, part * along / spread, part, kept);
    }
    return sum;
}

// The chop where water `depth` meters deep flows along `flow` (meters a
// second) at Froude number `froude`, as turbulent as `turbulence` says (0
// to 1) and shearing `shear` a second, at `time`, the place's clock running
// `stagger` cycles ahead, its standing waves rising `breathing` of their
// height, for a pixel spanning `footprint` meters, `share` of it: the
// turbulence's imprint the current carries (`water_carried_chop`), as steep
// as WATER_CHOP_SLOPE says, and short waves standing on its bed
// (`water_standing_chop`), as steep as WATER_STAND_SLOPE says, rising as
// its Froude number nears 1. Its crest: where its parts' heights together
// sit near their tops, the more the steeper it is, wholly where it is as
// steep as waves before they break; where the pixel spans many crests, the
// share of the surface they cover.
fn water_chop(p: vec2<f32>, flow: vec2<f32>, depth: f32, froude: f32, turbulence: f32,
        shear: f32, time: f32, stagger: f32, breathing: f32, footprint: f32,
        share: f32) -> WaterChop {
    let speed = length(flow);
    // As finely as the display shows it: the mip bias.
    let pixel = footprint * material_detail();
    let carried = WATER_CHOP_SLOPE * (0.25 + 0.75 * min(speed, 2.0))
        * mix(0.5, 1.0, turbulence);
    var sum = water_carried_chop(p, flow, depth, shear, time, stagger, carried, pixel);
    let standing = WATER_STAND_SLOPE
        * smoothstep(WATER_STAND_FROUDE.x, WATER_STAND_FROUDE.y, froude)
        * mix(0.6, 1.0, turbulence) * breathing;
    if standing > 0.0 {
        let stood = water_standing_chop(p, flow, standing, pixel);
        sum.waves += stood.waves;
        sum.tops += stood.tops;
        sum.resolved += stood.resolved;
        sum.heights += stood.heights;
    }
    let steep = sqrt(carried * carried + standing * standing);
    let top = sum.tops / max(sum.resolved, 1.0e-9);
    let resolved = sum.resolved / max(sum.heights, 1.0e-9);
    let crest = saturate(steep / WATER_CHOP_BREAKING)
        * mix(WATER_CHOP_COVER, smoothstep(WATER_CHOP_CREST.x, WATER_CHOP_CREST.y, top), resolved);
    return WaterChop(vec3<f32>(sum.waves.xy * share, sum.waves.z * share * share), crest * share);
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
// gather blurs and scatters on its way down; and the shares of the bed's
// depth a pixel spans, from the first to the second, over which they fade
// out: the waves gathering light at a depth are about a quarter of it
// long.
const WATER_CAUSTIC_DEPTH: f32 = 1.5;
const WATER_CAUSTIC_PIXELS: vec2<f32> = vec2<f32>(0.03, 0.08);

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

// How much of the water stands where `meters` (its depth, or its height
// over what stands in it) comes to nothing at its waterline, changing by
// `change` across a pixel: none at the line, all of it WATER_EDGE_PIXELS
// pixels from it or WATER_EDGE meters over it.
fn water_edge(meters: f32, change: f32) -> f32 {
    return max(saturate(meters / max(change * WATER_EDGE_PIXELS, 1.0e-6)),
        smoothstep(0.0, WATER_EDGE, meters));
}

// Surface foam (`water_foam`): falling water's map read WATER_FOAM_MAP
// meters across and along the flow, a quarter of a fall's, so its bubbles
// are about WATER_FOAM_BUBBLE meters across, and WATER_FOAM_FINER times
// finer, in patches WATER_FOAM_PATCH meters across, each turned to run
// along the flow about its own middle; the meters across white water's
// clumps and the times a second they churn, and the meters across the
// patches the foam the water carries drifts in.
const WATER_FOAM_MAP: vec2<f32> = vec2<f32>(1.0, FALL_MAP_STRETCH);
const WATER_FOAM_BUBBLE: f32 = 0.06;
const WATER_FOAM_FINER: f32 = 3.0;
const WATER_FOAM_PATCH: f32 = 0.6;
const WATER_FOAM_CLUMPS: f32 = 0.35;
const WATER_FOAM_CHURN: f32 = 1.5;
const WATER_FOAM_PATCHES: f32 = 1.2;
// How much each part of the grain makes of white water's (x: the larger
// bubbles, y: the smaller, z: the strands, w: its clumps) and of the
// foam's the water carries (x: the larger bubbles, y: both sizes'
// bubbles' walls, z: the strands, w: its patches): fresh white water is
// wet foam, round bubbles churning, and carried foam drains into the cells
// of dry foam, drifting in patches (Weta's foam, 2022).
const WATER_FOAM_WHITE: vec4<f32> = vec4<f32>(0.55, 0.45, 0.3, 0.6);
const WATER_FOAM_DRIFT: vec4<f32> = vec4<f32>(0.45, 0.4, 0.35, 0.7);
// How far falling water's map's strands and fibers spread about their
// middle, their standard deviation (`engine_tables::falls::SPREAD`); the
// spreads of the map's channels about their means: its strands' and
// fibers' (r and g), and its bubbles' walls' (b) and bubbles' (a) about as
// its cellular noise leaves them; and gradient noise's about its middle,
// in two and in three dimensions (`water_gradient_noise`,
// `water_gradient_noise3`), about as their lattices' gradients leave it.
const FALL_MAP_SPREAD: f32 = 0.17;
const WATER_FALL_SPREAD: vec4<f32> = vec4<f32>(FALL_MAP_SPREAD, FALL_MAP_SPREAD, 0.34, 0.18);
const WATER_NOISE_SPREAD: vec2<f32> = vec2<f32>(0.15, 0.12);
// The grain, in its spreads about none, covers the water where it passes a
// threshold its amount lowers from WATER_FOAM_NONE, which next to none of
// a normal spread passes, to WATER_FOAM_DENSEST, which 85% of one passes:
// low amounts leave scattered flecks and clusters, the most dense foam
// with dark water in its gaps, never one sheet. Its edge spans
// WATER_FOAM_EDGE spreads where a pixel resolves the grain; amounts below
// WATER_FOAM_LEAST, which would cover about a thousandth of the water, draw
// none.
const WATER_FOAM_NONE: f32 = 3.5;
const WATER_FOAM_DENSEST: f32 = -1.04;
const WATER_FOAM_EDGE: f32 = 0.3;
const WATER_FOAM_LEAST: f32 = 0.1;

// The share of a noise's variance a pixel spanning `footprint` meters
// resolves, its features `size` meters across: about 1 / (1 + (footprint /
// size)^2), the rest averaging out within the pixel as filtering averages
// it.
fn water_resolved_share(footprint: f32, size: f32) -> f32 {
    let ratio = footprint / size;
    return 1.0 / (1.0 + ratio * ratio);
}
// The spreads past its threshold over which foam heaps from a single layer
// of bubbles, of which WATER_FOAM_THINNEST hides what lies under it, to a
// heap that hides all; how high carried foam heaps beside white water, its
// bubbles drained as it ages; the meters it stands for each spread past its
// threshold, the relief the normal it is lit by follows; and how much its
// hollows darken it.
const WATER_FOAM_THICK: f32 = 1.0;
const WATER_FOAM_THINNEST: f32 = 0.5;
const WATER_FOAM_AGED: f32 = 0.6;
const WATER_FOAM_RELIEF: f32 = 0.01;
const WATER_FOAM_HOLLOW: f32 = 0.35;
// The meters foam rides toward the waves' crests for each unit of their
// slope: a wave moves what floats on it by its slope over its wave number
// (Gerstner's), here the chop's, half a meter long, so foam rides the
// ripples (Vlachos, Portal 2).
const WATER_FOAM_RIDE: f32 = 0.08;
// The meters a second at which the foam a current carries shows as dense
// as the water holds it: floating, it spreads along the current as fast as
// that runs, so it gathers up to twice as dense where the current slows
// and thins where it runs faster (the water carries its foam as a share,
// `water_sim.wgsl`, which spreads none); the share of it the most
// turbulent water draws under; and how strongly a calm shore's lining shows
// as foam, flecks at the waterline beside its lace.
const WATER_FOAM_GATHER: f32 = 0.5;
const WATER_FOAM_TORN: f32 = 0.5;
const WATER_FOAM_SHORE: f32 = 0.5;
// The radii white water trails behind what stands in the water, its wake
// narrowing from the obstacle's width to nothing; and the speeds of the
// water past it, from the first to the second, over which it tears air in
// there, about the meter a second past which water draws air in (Ervine,
// McKeogh and Elsawy 1980).
const WATER_FOAM_WAKE: f32 = 4.0;
const WATER_FOAM_TEARING: vec2<f32> = vec2<f32>(0.7, 1.3);

// Surface foam's grain at `p` (meters along x and z) where the water runs
// along `heading`, changing by `dx` and `dy` a pixel: falling water's map
// read WATER_FOAM_MAP meters across and along the flow and WATER_FOAM_FINER
// times finer, at the three corners of the triangle of a grid of
// WATER_FOAM_PATCH-meter patches holding it, each read turned to run along
// the heading about its corner's middle and shifted at random (salted by
// `salt`), and blended by their weights keeping their spread, as
// `water_tile` reads the waves' tiles. Turning each read about a middle
// nearby keeps it from shearing where the heading turns. Each part stands
// as its channels' deviation from the map's means `mean`, in their spreads
// (WATER_FALL_SPREAD): the larger bubbles (x), the smaller (y), both sizes'
// walls (z) and the strands (w), which run along the flow.
fn water_foam_grain(p: vec2<f32>, heading: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>,
        mean: vec4<f32>, salt: u32) -> vec4<f32> {
    let side = vec2<f32>(-heading.y, heading.x);
    let detail = material_detail();
    let turned_dx = vec2<f32>(dot(dx, side), dot(dx, heading)) * detail / WATER_FOAM_MAP;
    let turned_dy = vec2<f32>(dot(dy, side), dot(dy, heading)) * detail / WATER_FOAM_MAP;
    let triangle = water_triangle(p / WATER_FOAM_PATCH);
    var corners = triangle.corners;
    let weights = triangle.weights;
    let centre = vec4<f32>(mean.a, mean.a, 2.0 * mean.b, mean.r);
    var grain = vec4<f32>(0.0);
    for (var corner = 0u; corner < 3u; corner++) {
        let skewed = vec2<f32>(corners[corner]);
        let middle = vec2<f32>(skewed.x, (skewed.y + 0.57735027 * skewed.x) * 0.8660254)
            * WATER_FOAM_PATCH;
        let turned = vec2<f32>(dot(p - middle, side), dot(p - middle, heading))
            / WATER_FOAM_MAP;
        let larger = textureSampleGrad(water_fall_map, water_wave_sampler,
            turned + water_corner_offset(corners[corner], salt), turned_dx, turned_dy);
        let smaller = textureSampleGrad(water_fall_map, water_wave_sampler,
            turned * WATER_FOAM_FINER + water_corner_offset(corners[corner], salt ^ 0x5bd1e995u),
            turned_dx * WATER_FOAM_FINER, turned_dy * WATER_FOAM_FINER);
        grain += (vec4<f32>(larger.a, smaller.a, larger.b + smaller.b, larger.r) - centre)
            * weights[corner];
    }
    let spread = vec4<f32>(WATER_FALL_SPREAD.a, WATER_FALL_SPREAD.a,
        sqrt(2.0) * WATER_FALL_SPREAD.b, WATER_FALL_SPREAD.r);
    return grain * inverseSqrt(dot(weights, weights)) / spread;
}

// Surface foam over a pixel: the share of it foam hides (cover), the
// meters it heaps over the water (height, its relief), how thin it lies
// (thin: from heaped, 0, to a single layer of bubbles, 1), and how deep in
// its own hollows (hollow, 0 to 1).
struct WaterFoam {
    cover: f32,
    height: f32,
    thin: f32,
    hollow: f32,
}

// Surface foam at `p` (meters along x and z, where what floats on the
// waves lies) of white water as strong as `white` and of foam the water
// carries as strong as `carried` (0 to 1 each), where water flows along
// `flow` (meters a second) shearing `shear` a second, changing by `dx` and
// `dy` a pixel spanning `footprint` meters, `seen` meters seen from the
// detail eye, at `time`, the place's clock running `stagger` cycles ahead:
// a smooth amount thresholding a fine grain (Uncharted 3's foam). The grain
// (`water_foam_grain`), with white water's clumps churning in it and the
// patches carried foam drifts in, is carried along with the flow in sets
// as ripples are, cycling within WATER_FOAM_CYCLES, blended keeping its
// spread and faded as the flow strains it, so streaks thin out rather than
// smear. Each kind covers the water where its grain passes the threshold
// its amount sets, across an edge that widens only as far as the pixel
// spans its bubbles, so its edges follow the grain, never the triangles its
// amounts come from; past it, foam heaps, white water thick and bright,
// carried foam thinner, its relief handing over to none as the pixel seen
// from the detail eye comes to span its bubbles.
fn water_foam(p: vec2<f32>, white: f32, carried: f32, flow: vec2<f32>, shear: f32, time: f32,
        stagger: f32, dx: vec2<f32>, dy: vec2<f32>, footprint: f32, seen: f32) -> WaterFoam {
    let speed = length(flow);
    let heading = select(vec2<f32>(1.0, 0.0), flow / max(speed, 1.0e-6), speed > 1.0e-4);
    // The map's channels' means, as its last level holds them.
    let levels = textureNumLevels(water_fall_map);
    let mean = textureSampleLevel(water_fall_map, water_wave_sampler, vec2<f32>(0.5),
        f32(levels - 1u));
    let bands = water_bands(shear, WATER_FOAM_CYCLES);
    // The share of each part's variance the pixel resolves: the map's grain
    // (x, its mipmaps averaging the rest), white water's clumps (y) and the
    // carried foam's patches (z).
    let kept = vec3<f32>(water_resolved_share(footprint, WATER_FOAM_BUBBLE),
        water_resolved_share(footprint, WATER_FOAM_CLUMPS),
        water_resolved_share(footprint, WATER_FOAM_PATCHES));
    var wet = 0.0;
    var dry = 0.0;
    var weights = 0.0;
    for (var band = 0u; band < 2u; band++) {
        let shown = select(1.0 - bands.toward, bands.toward, band == 1u);
        if shown <= 0.0 {
            continue;
        }
        let cycle = bands.first * f32(1u + band);
        let salt = ((bands.index + band) * 0x632be5abu) ^ 0x1656667bu;
        for (var which = 0u; which < 2u; which++) {
            let phase = water_set(time, stagger, cycle, which, salt);
            let weight = shown * phase.weight;
            if weight <= 0.0 {
                continue;
            }
            let at = p - flow * phase.seconds + phase.offset;
            let grain = water_foam_grain(at, heading, dx, dy, mean, salt);
            let faded = weight * exp(-WATER_STRAIN_FADE * abs(phase.seconds) * shear);
            if white > WATER_FOAM_LEAST {
                // Filtered as the pixel spans its clumps, as the map's
                // grain is by its mipmaps.
                let clumps = (water_gradient_noise3(vec3<f32>(at / WATER_FOAM_CLUMPS,
                    time * WATER_FOAM_CHURN)) - 0.5) / WATER_NOISE_SPREAD.y
                    * sqrt(kept.y);
                wet += (dot(grain.xyw, WATER_FOAM_WHITE.xyz) + WATER_FOAM_WHITE.w * clumps)
                    * faded;
            }
            if carried > WATER_FOAM_LEAST {
                let patches = (water_gradient_noise(at / WATER_FOAM_PATCHES) - 0.5)
                    / WATER_NOISE_SPREAD.x * sqrt(kept.z);
                dry += (dot(grain.xzw, WATER_FOAM_DRIFT.xyz) + WATER_FOAM_DRIFT.w * patches)
                    * faded;
            }
            weights += weight * weight;
        }
    }
    // Each kind's grain in its spreads, and how far past its threshold it
    // stands (x white water, y carried foam).
    let grains = vec2<f32>(wet / length(WATER_FOAM_WHITE), dry / length(WATER_FOAM_DRIFT))
        * inverseSqrt(max(weights, 1.0e-6));
    let above = grains - mix(vec2<f32>(WATER_FOAM_NONE), vec2<f32>(WATER_FOAM_DENSEST),
        vec2<f32>(white, carried));
    // What the pixel averages of each grain, in its spreads squared: the
    // threshold's edge widens by it, as a ramp of the same variance (twelve
    // times a uniform spread's), so the foam a pixel shows is the share of
    // its bubbles passing the threshold within it, however far it is, and
    // never thins out as it recedes.
    let unresolved = vec2<f32>(
        dot(WATER_FOAM_WHITE * WATER_FOAM_WHITE, vec4<f32>(vec3<f32>(1.0 - kept.x), 1.0 - kept.y)),
        dot(WATER_FOAM_DRIFT * WATER_FOAM_DRIFT, vec4<f32>(vec3<f32>(1.0 - kept.x), 1.0 - kept.z)))
        / vec2<f32>(dot(WATER_FOAM_WHITE, WATER_FOAM_WHITE), dot(WATER_FOAM_DRIFT, WATER_FOAM_DRIFT));
    let edge = sqrt(WATER_FOAM_EDGE * WATER_FOAM_EDGE + 12.0 * unresolved);
    let covers = saturate(above / edge + 0.5);
    let heaped = saturate(above / WATER_FOAM_THICK);
    let thick = heaped * vec2<f32>(1.0, WATER_FOAM_AGED);
    let hiding = covers * mix(vec2<f32>(WATER_FOAM_THINNEST), vec2<f32>(1.0), thick);
    let both = max(hiding.x + hiding.y, 1.0e-6);
    var foam: WaterFoam;
    foam.cover = 1.0 - (1.0 - hiding.x) * (1.0 - hiding.y);
    foam.thin = 1.0 - dot(hiding, thick) / both;
    foam.hollow = 1.0 - dot(hiding, heaped) / both;
    // Its relief, which hands over to none as a pixel seen from the detail
    // eye comes to span its bubbles.
    foam.height = WATER_FOAM_RELIEF * max(max(above.x, above.y * WATER_FOAM_AGED), 0.0)
        * (1.0 - saturate(seen / WATER_FOAM_BUBBLE));
    return foam;
}

// How strongly white water trails behind the obstacle nearest water
// `depth` meters deep flowing along `flow` (meters a second), `obstacle`
// (its offset from the point along x and z and its radius, meters, as
// `water_standing` reads it): in a wake narrowing from the obstacle's width
// to nothing WATER_FOAM_WAKE radii downstream of it, where the water runs
// past it fast enough to tear air in (WATER_FOAM_TEARING), and where it
// stands near enough the surface to break it: in water a radius deep, and
// none in water three radii deep.
fn water_wake(obstacle: vec4<f32>, flow: vec2<f32>, depth: f32) -> f32 {
    let radius = obstacle.z;
    let speed = length(flow);
    if radius <= 0.0 || speed <= WATER_FOAM_TEARING.x {
        return 0.0;
    }
    let heading = flow / speed;
    // Meters downstream of the obstacle's middle, and across the flow.
    let apart = -obstacle.xy;
    let along = dot(apart, heading);
    let across = dot(apart, vec2<f32>(-heading.y, heading.x));
    let trail = saturate(1.0 - along / (WATER_FOAM_WAKE * radius));
    let half = max(radius * trail, 1.0e-4);
    return smoothstep(WATER_FOAM_TEARING.x, WATER_FOAM_TEARING.y, speed)
        * smoothstep(0.0, radius, along) * trail
        * (1.0 - smoothstep(0.5 * half, half, abs(across)))
        * (1.0 - smoothstep(radius, 3.0 * radius, depth));
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

// Tiles' water at `input`, refracting the scene's color when `refracts`;
// or the simulated water, when `simulated`.
fn shade_water(input: WaterOutput, refracts: bool, simulated: bool) -> WaterShade {
    let world = eye_world(input.relative);
    if !simulated && water_simulated(world.xz) {
        discard;
    }
    // Meters of water over the ground: the shore's dry side holds none.
    let depth = input.water.x;
    if depth <= 0.0 {
        discard;
    }
    let flow = input.water.yz;
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
        falling = 1.0 - smoothstep(SIM_SHEER.x, SIM_SHEER.y, upright);
        if falling > 0.99 {
            discard;
        }
    }
    // The water as it settled: how turbulent it runs, where white water is
    // born in it and how fast its current shears (a second).
    let turbulence = input.state.x;
    let born = input.state.y;
    let shear = input.state.z;
    // Still water's waves run with the wind as they evolve; flowing water
    // keeps the wind's finest ripples, carried along with it in sets that
    // start over before its shear stretches them, and wears the chop its
    // current makes; between, the two blend, so nothing changes at once
    // where a river meets a lake.
    let speed = length(flow);
    let flowing = smoothstep(0.05, 0.3, speed);
    // How fast it runs, still to 2 m/s: running water raises no swell, and
    // its current's chop takes over from the wind's ripples.
    let running = saturate(speed * 0.5);
    // The plane its surface lies in without its waves, tilted as it falls:
    // the eye looks down onto it, light enters through it and what lies
    // behind the pixel stands under it, so water running down toward the eye
    // from above its height shows as water below it does.
    let plane = normalize(vec3<f32>(input.slope.x, 1.0, input.slope.y));
    // How far the pixel spans on the water, which filters the patterns it
    // shows of it; and how far it spans along the view on that plane seen
    // from the detail eye (`detail_span`), which those patterns hand over to
    // their means by before they shimmer.
    let footprint = max(length(world_dx.xz), length(world_dy.xz));
    let seen = detail_span(world, plane).y;
    // Each place's patterns start over at their own times, so the flow
    // never pulses all at once (Vlachos's phase noise): the cycles its
    // clock runs ahead by.
    let stagger = water_noise(world.xz / WATER_FLOW_PHASE).x;
    var waves = vec3<f32>(0.0);
    if flowing < 1.0 {
        waves = water_waves_at(world.xz, world_dx.xz, world_dy.xz, vec3<f32>(1.0));
    }
    if flowing > 0.0 {
        let moving = water_flowing_waves(world.xz, world_dx.xz, world_dy.xz,
            mix(1.0, WATER_RUNNING_RIPPLES, running), flow, shear, time, stagger);
        waves = water_blended(waves, moving, flowing);
    }
    let gusts = water_gusts(world.xz, time);
    let strength = water.ripples.x * gusts;
    // Waves stand where the bed holds them, about the obstacle nearest the
    // water's samples, which the triangles blend their offsets to: only
    // where a pixel's samples agree on one, the obstacle staying put as the
    // point moves across the pixel.
    let anchor = world.xz + input.obstacle.xy;
    let anchoring = (length(dpdx(anchor)) + length(dpdy(anchor)))
        / max(length(world_dx.xz) + length(world_dy.xz), 1.0e-6);
    let anchored = 1.0 - smoothstep(0.1, 0.4, anchoring);
    let froude = speed / sqrt(WATER_GRAVITY * max(depth, WATER_THINNEST));
    // Waves standing about obstacles, and where the water runs near or
    // above the speed of its own waves, breathe as slow noise passes over
    // them.
    var breathing = 1.0;
    if anchored > 0.0 || (flowing > 0.0 && froude > WATER_STAND_FROUDE.x) {
        breathing = water_breathing(world.xz, flow, time, stagger);
    }
    var stationary = vec3<f32>(0.0);
    if anchored > 0.0 {
        stationary = water_standing(input.obstacle, flow, depth, froude, breathing, seen)
            * anchored;
    }
    // Running water's current chops it: the eddies it carries, and short
    // waves standing on its bed where it runs near or above the speed of its
    // own waves; and its crest, how near the tops of the chop's crests it
    // stands (0 to 1).
    var chop = WaterChop(vec3<f32>(0.0), 0.0);
    if flowing > 0.0 {
        chop = water_chop(world.xz, flow, depth, froude, turbulence, shear, time, stagger,
            breathing, footprint, flowing);
    }
    let crest = chop.crest;
    // Its ripples, chop and standing waves ride its surface, tilting as it
    // falls; and the slopes they hide within a pixel roughen it, so where
    // chop churns it, it glints sharply rather than mirroring the sky
    // (Bruneton, Neyret and Holzschuch).
    let slopes = waves.xy * strength + chop.waves.xy + stationary.xy - input.slope;
    let normal = normalize(vec3<f32>(-slopes.x, 1.0, -slopes.y));
    let roughness = water_roughness(waves.z * strength * strength + chop.waves.z
        + stationary.z);
    let view = stable_normalize(-input.relative);
    // Under the sea, the sea's water stands instead: its level
    // (`water.ocean.x`) is held relative to the eye.
    if water.ocean.y > 0.5 && world.y < eye_world(vec3<f32>(0.0, water.ocean.x, 0.0)).y {
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
        behind = water_behind(input.relative, normal, view, pixel, own, plane);
    }
    // How far the water's surface stands over what stands in it behind the
    // pixel, the meters it lies under the surface's plane, and how fast that
    // changes across it; and the bed under the pixel, where the waves gather
    // the sun's light.
    let standing = max(dot(input.relative - own, plane) / plane.y, 0.0); // optics: what the view meets behind the pixel
    let standing_change = abs(dpdx(standing)) + abs(dpdy(standing));
    let bed = min(depth, standing);
    let gathered = water_caustics(world, normal, bed);
    // The water ends at its waterline, where its depth or its height over
    // what stands in it comes to nothing; thin water shows by how fast it
    // runs: calm shallows fade in over WATER_SHORE of depth, while racing
    // water, however thin, is all there.
    let racing = smoothstep(WATER_RACING.x, WATER_RACING.y, speed);
    let shallows = mix(smoothstep(0.0, WATER_SHORE, min(depth, standing)), 1.0, racing);
    let fade = min(water_edge(depth, fwidth(depth)), water_edge(standing, standing_change))
        * shallows * (1.0 - falling);
    // Light comes back through the water along the view to what it shows,
    // as far as the depth lets it run, after reaching it down from the
    // surface, taken by the water and scattered by its sediment and bubbles.
    // Water a fall's sheet runs over (`shade_fall`) is a film over steep
    // rock: the sheet draws its white and its body.
    let steep = length(input.slope);
    let film = smoothstep(WATER_FILM.x, WATER_FILM.y, steep) * input.foam.z;
    let through = min(max(length(behind.ground - input.relative), 0.0), // optics: the view's way through the water
        depth * plane.y / max(dot(view, plane), 0.1)); // optics: the view's way through the water
    let down = clamp(dot(input.relative - behind.ground, plane) / plane.y, 0.0, through); // optics: the light's way down to what the view meets
    // The water holds bubbles where white water is born in it and where it
    // carries the foam its flow churned up.
    let aerated = max(input.state.y, saturate(input.churned / WATER_FOAM_MOST));
    let transmit = water_transmitted((through + down) * (1.0 - film), aerated);
    // Foam: lace along the waterline, where shores and what stands in the
    // water meet it, out to WATER_FOAM_REACH meters (as the settings' foam
    // width scales it), on calm water only, a quickening current tearing it
    // away; white water where the water breaks, heaped on its waves' crests
    // and trailing behind what stands in it; and the foam it leaves,
    // floating on as the water carries it, gathering where the water slows
    // and thinning where it races and churns. Open water carries none but
    // what its flow churned up. All of it floats where the waves carry it,
    // toward their crests by their slopes.
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
    let solid = 1.0 - smoothstep(0.03, 0.12, standing_change);
    // Narrow water holds a narrower band: no more than a share of its half
    // width.
    let shore_reach = min(WATER_FOAM_REACH * water.foam.y / 0.6,
        input.foam.y * WATER_FOAM_SHARE);
    let shore = max(1.0 - smoothstep(0.0, shore_reach, from_shore),
        select(0.0, 1.0 - smoothstep(0.0, WATER_FOAM_RING, standing), standing < depth * 0.8)
            * solid);
    // Where what floats on the waves lies: WATER_FOAM_RIDE toward their
    // crests for each unit of their slope.
    let ridden = world.xz + normal.xz * (WATER_FOAM_RIDE / max(normal.y, 0.1));
    // Lace lies on calm water only (WATER_LACE_CALM): along its shores and,
    // where it carries foam, out over it, where falls land and currents
    // pile up, as its flow churns it up. Under a fall's sheet the sheet
    // draws the water's foam.
    let slack = 1.0 - smoothstep(WATER_LACE_CALM.x, WATER_LACE_CALM.y, speed);
    let lined = shore * sqrt(shore) * water.foam.x / 0.6 * (1.0 - input.foam.z) * slack;
    let lining = WATER_SHORE_LACE * lined;
    let churned_share = input.churned / WATER_FOAM_MOST;
    let plunge = smoothstep(WATER_FOAM_SHOWN.x, WATER_FOAM_SHOWN.y, churned_share)
        * WATER_FOAM_LACE * slack;
    // Lace lies on the surface as seen from above: where the surface itself
    // falls steeply (water running down a chute), it would stretch down it,
    // so it gives way there. The water makes and moves it, in two sets each
    // forming anew as the other dissolves: on slowly moving water the
    // current carries them along as it carries ripples, slower at the bank,
    // drawing them out along it; on still water each is a wave lapping the
    // shore, washing its lace in, as big as the gusts raise the waves there,
    // and drawing it back out as it dissolves. Its bubbles' edges soften by
    // the cells the pixel spans; seen from the detail eye, its finer cells'
    // smaller bubbles show up close only, and where the pixel spans most of
    // its cells, its share of the surface shows instead. Lace is a single
    // layer of bubbles: WATER_LACE_OPACITY of it hides the water.
    let flat = 1.0 - smoothstep(0.5, 1.0, steep);
    var lace = 0.0;
    if max(lining, plunge) * flat > 1.0e-3 {
        let blur = footprint / WATER_LACE_CELL;
        let seen_cells = seen / WATER_LACE_CELL;
        let far = smoothstep(0.35, 0.9, seen_cells);
        let finer = 1.0 - smoothstep(0.12, 0.25, seen_cells);
        let lives = water_noise(world.xz / WATER_FOAM_STAGGER).x * WATER_FOAM_STAGGERED;
        let current = flow * mix(0.75, 1.0, smoothstep(0.0, 3.0 * shore_reach, from_shore));
        let lapping = clamp(gusts, 0.3, 1.3);
        var lapped = vec2<f32>(0.0);
        if flowing < 1.0 {
            for (var which = 0u; which < 2u; which++) {
                let one = water_lapped_lace(which, ridden, time / WATER_FOAM_LAPPED + lives,
                    lining, plunge, offshore, lapping, blur, finer, far < 1.0);
                lapped = vec2<f32>(lapped.x + one.x, max(lapped.y, one.y));
            }
        }
        var drifted = vec2<f32>(0.0);
        if flowing > 0.0 {
            drifted = water_carried_lace(ridden, current, shear, time, lives,
                max(lining, WATER_PLUNGE_LACE * plunge), blur, finer, far < 1.0);
        }
        let laced = mix(lapped, drifted, flowing);
        lace = mix(laced.y, min(laced.x * 2.2, 0.8) * WATER_LACE_FILL, far) * flat
            * WATER_LACE_OPACITY;
    }
    // Surface foam (`water_foam`), on running water the only foam: white
    // water where the water breaks, the more on its waves' crests, and
    // trailing behind what stands in fast water (`water_wake`, where the
    // triangles agree on it as the standing waves do); the foam the water
    // carries, gathering where it slows (WATER_FOAM_GATHER) and drawn under
    // where it churns; and flecks of it along calm shores. Under a fall's
    // sheet the sheet draws white water.
    let in_wake = water_wake(input.obstacle, flow, depth) * anchored;
    let white_amount = max(min(born * (1.0 + crest), 1.0), in_wake) * (1.0 - film);
    let gathering = 2.0 * WATER_FOAM_GATHER / (WATER_FOAM_GATHER + speed)
        * (1.0 - WATER_FOAM_TORN * turbulence);
    let carried_amount = min(max(churned_share * gathering, WATER_FOAM_SHORE * lined), 1.0);
    var drawn: WaterFoam;
    if max(white_amount, carried_amount) > WATER_FOAM_LEAST {
        drawn = water_foam(ridden, white_amount, carried_amount, flow, shear, time, stagger,
            world_dx.xz, world_dy.xz, footprint, seen);
    }
    // The share of the surface foam covers, lace and surface foam over each
    // other.
    let foam = 1.0 - (1.0 - lace) * (1.0 - drawn.cover);
    // Foam's relief tilts the normal it is lit by: a surface gradient from
    // how its height changes across the pixel (Mikkelsen), as falls bend
    // theirs. Its hollows shade it, and lace lies a single layer of bubbles
    // thin.
    let across_dy = cross(world_dy, normal);
    let across_dx = cross(normal, world_dx);
    let spanned = dot(world_dx, across_dy);
    let foam_tilt = (dpdx(drawn.height) * across_dy + dpdy(drawn.height) * across_dx)
        * (sign(spanned) / max(abs(spanned), 1.0e-12));
    let layers = max(lace + drawn.cover, 1.0e-6);
    var surface: WaterSurface;
    surface.relative = input.relative;
    surface.world_dx = world_dx;
    surface.world_dy = world_dy;
    surface.normal = normal;
    surface.roughness = roughness;
    surface.plane = plane;
    surface.foam_tilt = foam_tilt;
    surface.foam_shade = WATER_FOAM_HOLLOW * drawn.hollow * drawn.cover / layers;
    surface.foam_thin = (drawn.thin * drawn.cover + lace) / layers;
    surface.foam = foam;
    surface.scatter = water_body(aerated);
    surface.transmit = transmit;
    surface.glow = vec3<f32>(0.0);
    surface.crest = crest;
    let lit = water_light(surface);
    // The waves' caustics brighten the sun's share of the light on the bed,
    // fading as it lies deeper, as turbulence scatters the light, and as
    // the pixel spans more of the waves gathering it.
    let caustic = 1.0 + (gathered - 1.0) * lit.sunny * exp(-bed / WATER_CAUSTIC_DEPTH)
        * (1.0 - turbulence) * (1.0 - smoothstep(WATER_CAUSTIC_PIXELS.x, WATER_CAUSTIC_PIXELS.y,
            footprint / max(bed, 1.0e-3)));
    // Of what the surface lets through, the ground shows darker for being
    // wet however thin the water; foam covers it.
    let wet = smoothstep(0.0, 0.03, depth) * WATER_WET;
    let let_through = (1.0 - lit.reflectance) * (1.0 - foam);
    var result: WaterShade;
    result.shaded = lit.shaded;
    result.coverage = max(max(1.0 - (1.0 - lit.reflectance) * luminance(transmit), wet), foam)
        * fade;
    result.behind = behind.color * transmit * (caustic * let_through * (1.0 - wet)
        * (1.0 - atmospheric_cover(camera.position.xyz, world)));
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
    output.slope = input.slope;
    output.churned = input.churned;
    output.state = input.state;
    output.obstacle = input.obstacle;
    output.reach = input.reach;
    return output;
}

// Colors blend premultiplied over the scene: the scene target's alpha
// keeps the ground's ambient share, the other targets take the water's.
// The canvas holds no copy of the scene's color: water blends over the
// ground there.
@fragment
fn fs_water(input: WaterOutput) -> @location(0) vec4<f32> {
    let shade = shade_water(input, false, false);
    return vec4<f32>(encoded(shade.shaded.color, input.clip_position.xy) * shade.coverage,
        shade.coverage);
}

@fragment
fn fs_water_post(input: WaterOutput) -> SceneTargets {
    let shade = shade_water(input, true, false);
    var targets = scene_targets(shade.shaded, input.clip_position.xy);
    targets.color = water_covering(shade, input.clip_position.xy);
    return targets;
}

@fragment
fn fs_water_temporal(input: WaterTemporalOutput) -> TemporalTargets {
    let shade = shade_water(water_temporal_output(input), true, false);
    var targets = temporal_targets(shade.shaded, input.clip_position, input.previous);
    targets.color = water_covering(shade, input.clip_position.xy);
    return targets;
}

// Simulated water (`water_sim.rs`): a vertex at each cell's middle, its
// water's surface, depth and flow, the slopes of its surface and depth
// read across its neighbors, the foam the water carries, how turbulent it
// runs, where white water is born in it and its shear, read from the water
// around it, and its bed's reach and nearest obstacle, as tiles' water
// holds them; shaded as tiles' water is. Its heights stand over the live
// window's datum (`water.sim_datum.x` meters), which drawing adds back.
//
// Meters of water under which a cell is dry; the half width its shores'
// foam reads it as (a river's); and the depth over which its velocity is
// desingularized.
const SIM_WET: f32 = 0.01;
const SIM_HALF_WIDTH: f32 = 8.0;
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
// cliff, between cells far apart in height, the falls' sheets show it; and
// the meters over the simulated water below past which those sheets show,
// from the first to the second.
const SIM_SHEER: vec2<f32> = vec2<f32>(0.45, 0.5);
const SIM_SHEETS_ABOVE: vec2<f32> = vec2<f32>(3.0, 6.0);

// Where the simulated water holds cell (i, j), the nearest inside its
// square for one past its edge: its block's first cell (its blocks stand
// in order, row by row) and its place in the block.
fn sim_water_at(i: i32, j: i32) -> u32 {
    let n = i32(water.sim.w);
    let at = vec2<u32>(clamp(vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(n - 1)));
    let blocks = (u32(n) + SIM_WATER_BLOCK - 1u) / SIM_WATER_BLOCK;
    let block = (at.y / SIM_WATER_BLOCK) * blocks + at.x / SIM_WATER_BLOCK;
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

// Cell (i, j)'s water's surface and depth. A dry cell beside wet ones on
// ground as high as their surface carries that surface over it, as deep
// as it stands over the ground (none or less), so the shore falls where
// the surface meets the ground; one on lower ground lies on it, as far
// under water as the shallowest of them stands over theirs, so the water
// ends halfway down to it; one beside none, a meter below its ground.
fn sim_water_level(i: i32, j: i32) -> vec2<f32> {
    let ground = sim_water_ground(i, j);
    let surface = sim_water_cell(i, j).x;
    if surface - ground > SIM_WET {
        return vec2<f32>(surface, surface - ground);
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
        return vec2<f32>(ground - 1.0, -1.0);
    }
    let level = sum / count;
    if ground >= level {
        return vec2<f32>(level, level - ground);
    }
    return vec2<f32>(ground, -shallowest);
}

// The lowest the simulated water stands about `point` (x and z), in world
// meters: over the cells around the one holding it, each wet one's surface
// or each dry one's ground.
fn sim_water_lowest_at(point: vec2<f32>) -> f32 {
    let at = vec2<i32>(floor((point - water.sim.xy) / water.sim.z));
    var lowest = 1.0e9;
    for (var dj = -1; dj <= 1; dj++) {
        for (var di = -1; di <= 1; di++) {
            let ground = sim_water_ground(at.x + di, at.y + dj);
            let surface = sim_water_cell(at.x + di, at.y + dj).x;
            lowest = min(lowest, select(ground, surface, surface - ground > SIM_WET));
        }
    }
    return lowest + water.sim_datum.x;
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
// `other`: the neighbor's where it is wet, else the cell's own, so shores
// tilt no surface.
fn sim_water_beside(here: vec2<f32>, other: vec2<f32>) -> f32 {
    return select(here.x, other.x, other.y > SIM_WET);
}

fn sim_water_output(index: u32) -> WaterOutput {
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
    var output: WaterOutput;
    output.relative = world - camera.position.xyz;
    output.clip_position = terrain_view.relative_view_projection * vec4<f32>(output.relative, 1.0);
    output.water = vec3<f32>(depth, velocity);
    output.slope = vec2<f32>(sim_water_beside(here, west) - sim_water_beside(here, east),
        sim_water_beside(here, south) - sim_water_beside(here, north)) / (2.0 * cell);
    output.shore = vec2<f32>(west.y - east.y, south.y - north.y) / (2.0 * cell);
    // White water: the foam the water carries, streaking it where it runs
    // down steeply.
    let foam = sim_water_cell(i, j).w;
    output.foam = vec3<f32>(min(foam, 1.0) * WATER_FOAM_STREAKED, SIM_HALF_WIDTH, 0.0);
    output.churned = foam;
    // How turbulent it runs, from the velocities around it as the cook's
    // turbulence reads its water's swings (`engine_terrain`'s
    // `turbulence`): how far its velocity strays from the mean of its
    // neighbors holding water, against its speed, and how fast the water
    // turns there; where white water is born in it, by the rule the cook
    // breaks its settled water white by (`engine_terrain`'s `breaking`):
    // running down its surface, it loses g·|q| times the surface's slope
    // over its depth h a kilogram (ε, at least WATER_BREAK_THINNEST deep),
    // which stirs eddies as large as it is deep turning at (ε·h)^(1/3), and
    // those break its surface from WATER_BREAK_ONSET to WATER_BREAK_FULL of
    // √(g·h); and its shear. Its reach and nearest obstacle are its bed's.
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
    let deep = max(depth, WATER_BREAK_THINNEST);
    let losing = WATER_GRAVITY * length(flow) * length(output.slope) / deep;
    let breaks = pow(losing * deep, 1.0 / 3.0) / sqrt(WATER_GRAVITY * deep);
    let born = smoothstep(WATER_BREAK_ONSET, WATER_BREAK_FULL, breaks);
    let wet = depth > SIM_WET;
    output.state = vec4<f32>(select(vec3<f32>(0.0), vec3<f32>(turbulence, born, shear), wet), 0.0);
    let bed = water_sim_beds[sim_water_at(i, j)];
    output.obstacle = water_obstacle(bed.y, vec2<f32>(0.0));
    output.reach = bed.x;
    return output;
}

@vertex
fn vs_sim_water(@builtin(vertex_index) index: u32) -> WaterOutput {
    return sim_water_output(index);
}

@vertex
fn vs_sim_water_temporal(@builtin(vertex_index) index: u32) -> WaterTemporalOutput {
    let output = sim_water_output(index);
    return water_temporal(output, (camera.previous_view_projection
        * vec4<f32>(eye_world(output.relative), 1.0)).xyw);
}

@fragment
fn fs_sim_water(input: WaterOutput) -> @location(0) vec4<f32> {
    let shade = shade_water(input, false, true);
    return vec4<f32>(encoded(shade.shaded.color, input.clip_position.xy) * shade.coverage,
        shade.coverage);
}

@fragment
fn fs_sim_water_post(input: WaterOutput) -> SceneTargets {
    let shade = shade_water(input, true, true);
    var targets = scene_targets(shade.shaded, input.clip_position.xy);
    targets.color = water_covering(shade, input.clip_position.xy);
    return targets;
}

@fragment
fn fs_sim_water_temporal(input: WaterTemporalOutput) -> TemporalTargets {
    let shade = shade_water(water_temporal_output(input), true, true);
    var targets = temporal_targets(shade.shaded, input.clip_position, input.previous);
    targets.color = water_covering(shade, input.clip_position.xy);
    return targets;
}
