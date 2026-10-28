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
// across its water body is and whether a lake holds it, and the reach of its
// stream), read through the same weights, so no triangle of the level shows
// in it; and how much of those weights its
// samples holding a surface carry, the water ending where they carry less
// than half (`water_level`). It lies wherever that surface stands over the
// ground the pixel shows, so its waterline is where the ground drawn rises
// through it, pixel by pixel, whatever shape the ground takes
// (`shade_water`). None of it moves as the eye does and no leaf shapes it.
// Where the water runs live around the detail eye (`water_sim.rs`), the
// same read takes the live water's cells as the samples of a level of their
// own, each holding what a tile's sample holds (`water_sim.wgsl`'s
// `sim_drawn`), and the vertices stand on whichever surface stands higher
// (`water_field`): one water, which hands back to the tiles' level as a
// pixel's span grows past what the live reach was made for and across the
// live square's rim, where the live water is the settled water
// (`water_live_share`). Everything drawn of the water reads that state and
// what follows from it: its eddies' speed (`water_eddies`); the water's debug
// views paint each of its fields (`meshes/debug.wgsl`).
// Nothing else draws it (`shade_water`): its surface is that smooth surface
// itself, tilted by the small waves the turbulence reaching it raises,
// which its particles carry around the detail eye (`water_particles.rs`,
// `meshes/water_foam.wgsl`): wavefronts raised by the boils of turbulence's
// bursts and where the water runs against what stands in it, running at
// their groups' speed plus the current its surface runs at and spreading,
// so its glints and what it mirrors travel downstream; nothing else adds
// waves to it. What neither resolves of it, the surface its eddies lift and
// lower between its samples, roughens it by
// the simulation's own turbulence and eddies there (`water_flow.wgsl`), and
// so does how far its normal turns across a pixel, along the view and
// across it as the ellipse of those slopes projects into the view
// (`water_footprint.wgsl`), so far water streaks and glints along the view.
// Where nothing stirs it, it lies a calm mirror. It mirrors the sky, the
// sun's disc and, through screen-space reflections, the scene, drawn out
// along the view where it grazes the water; its body, tinted by the
// sediment it carries, scatters the light entering it back toward the eye;
// and what lies under it shows through, bent by its surface where the frame
// copies the scene's color, brighter where that surface's own curvature
// gathers the sun's light on it, and dimmed channel by channel over the
// light's path down to it and back, the ground darker for being wet. Its
// foam is what its particles lay: the white water and foam the same bursts
// lay where it breaks and churns (`engine_terrain::shallow_water`), settling
// or live, as particles carried by the current its surface runs at, drawn
// out into streaks with open water between, and past their reach its own
// state's, their mean; and white water where it runs fast against what
// stands in it. However thin, the water fades only at its waterline. The
// sea (`meshes/ocean.wgsl`) lights its surface the same way.
// Where rivers leave their beds, their jets fall as sheets of their own
// (`water_falls.rs`), drawn last.

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
// for no reach, else their reach plus one (y, four bytes); and the height
// above the terrain's origin the water's mesh stands at over them, a float's
// bits (z, `water_field`). The fourth word holds nothing: WebGPU has no
// texel of three words.
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
// and for terrains' water its particles' (`water_splatted`): the meters
// across their texels (x, none where none run), the meters their textures
// span (y), the pixels their texels span where they hand over to the
// water's own foam (z) and the meters across the cells of the water they
// ride (w) (`foam`); and how fast the water takes each channel of light,
// per meter (rgb).
// The sea's parts (`meshes/ocean.wgsl`): its wind's heading
// (`origin`, xy: a unit vector), its still level less the eye, its grid's
// finest cell, cells around and rings a cell apart; the rings' growth, the
// meters it reaches, cells around a sector and its waves; each wave's
// vector (xy, radians per meter), height and phase at the eye; and how far
// each wave's points move along it, four to a vector. For terrains' water,
// `ocean` holds the sea's level less the eye and whether it is on, below
// which their water gives way to it. Where water runs live around the
// detail eye (`water_sim.rs`), `sim` holds its square's first corner along
// x and z, the meters across a cell and the cells along a side (none,
// zero), `sim_datum` the height its heights stand above (x, world meters),
// the slots its blocks turn by along x and z (yz) and its pool's slots
// along each side (w), and `sim_pixels` the pixels across the view a cell
// spans where the live water hands back to the tiles' (x,
// `water_live_share`), then nothing.
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
    sim_pixels: vec4<f32>,
}
@group(1) @binding(52) var<uniform> water: WaterUniform;
// The live water's cells' looks (`water_sim.wgsl`'s, each block of the
// live square in its slot of the pool, WATER_LIVE_BLOCK cells along each
// side, a block's row by row), WATER_LIVE_LOOKS entries a cell: the settled
// water at it, then its water as water's pass draws it, as a tile's water
// sample holds it: its surface (above the square's datum; NO_WATER_SURFACE
// where none) and its ground, then its two words as `water_info`'s first
// two (`water_words`), their bits.
@group(1) @binding(56) var<storage, read> water_live: array<vec4<f32>>;
const WATER_LIVE_BLOCK: u32 = 8u;
const WATER_LIVE_LOOKS: u32 = 2u;
// Cells along each edge of the live square over which the solve pulls its
// water back to the settled water and water's pass hands it back to the
// tiles' (`gpu_layouts::constants::SIM_RIM`).
const WATER_LIVE_RIM: u32 = 8u;
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
// widest span the packed span tells apart (meters across a water body) and
// the shear (per second) the packed shear spans
// (`engine_terrain::WATER_FLOW_RANGE`, `WATER_SPAN_LEAST`,
// `WATER_SPAN_MOST`, `WATER_SHEAR_RANGE`); and the steepest a bank rises,
// meters a meter, past which it stands as a wall
// (`gpu_layouts::constants::WATER_GRADIENT_RANGE`).
const WATER_FLOW_RANGE: f32 = 8.0;
const WATER_SPAN_LEAST: f32 = 0.25;
const WATER_SPAN_MOST: f32 = 4096.0;
const WATER_SHEAR_RANGE: f32 = 8.0;
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
// channel sees; and foam's roughness where it lies a single drained layer
// thin, and where it heaps.
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
// The meters a second over which thin water comes to show whole however
// thin, from the first to the second: calm shallows (a river's water
// spread over its banks, a pool's edge) fade in over WATER_SHORE of depth
// as still water's edges do, while water racing thinner than that (a
// torrent's sheet over its bed) is all there.
const WATER_RACING: vec2<f32> = vec2<f32>(0.8, 1.6);
// Meters a second squared of gravity (`engine_terrain::hydraulics::GRAVITY`).
const WATER_GRAVITY: f32 = 9.81;

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
// strongly it breaks white and its shear (a second); and its reach's code
// (0 for none, else 1 plus `engine_terrain::Reach`).
struct WaterWords {
    flow: vec2<f32>,
    foam: f32,
    span: f32,
    lake: bool,
    sheet: bool,
    state: vec3<f32>,
    reach: u32,
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
// lake holding it, a fall's sheet over it, its reach) its nearest sample's;
// and how fast its smooth surface falls along x and z (meters a meter).
struct WaterAt {
    surface: f32,
    depth: f32,
    held: f32,
    words: WaterWords,
    slope: vec2<f32>,
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
// it; and where the live water's surface stands higher (`water_live_field`),
// at that surface. So the mesh stands at the water's level wherever its
// surface reaches, settled or live, and each pixel decides where water
// shows. It depends on the point alone, so leaves sharing a vertex meet on
// it.
fn water_field(local: vec2<f32>) -> f32 {
    let cell = water_cell(local, 0u);
    var level = 0.0;
    if cell.block == NO_WATER_BLOCK {
        level = terrain_level_height(local, cell.level).height;
    } else {
        let upper = cell.within.x + cell.within.y > 1.0;
        var levels = vec3<f32>(0.0);
        for (var corner = 0u; corner < 3u; corner++) {
            let texel = water_texel(cell.block, cell.first + water_corner(corner, upper));
            levels[corner] = bitcast<f32>(textureLoad(water_info, texel.xy, texel.z, 0).z);
        }
        level = dot(levels, water_weights(cell.within));
    }
    return max(level, water_live_field(local));
}

// The height above the terrain's origin the live water's surface stands at
// at terrain-local `local`, where it runs there: its cells' drawn surfaces
// (`water_live_look`) between their middles, bilinearly, over those of the
// four around the point holding one; far below any ground
// (NO_WATER_SURFACE) where none of them holds one or no water runs live
// there.
fn water_live_field(local: vec2<f32>) -> f32 {
    let n = water.sim.w;
    let cells = (local + terrain.origin.xz - water.sim.xy) / water.sim.z - 0.5;
    if n < 2.0 || any(cells < vec2<f32>(0.0)) || any(cells > vec2<f32>(n - 1.0)) {
        return NO_WATER_SURFACE;
    }
    let first = min(vec2<i32>(floor(cells)), vec2<i32>(i32(n) - 2));
    let within = cells - vec2<f32>(first);
    var sum = 0.0;
    var weight = 0.0;
    for (var k = 0; k < 4; k++) {
        let step = vec2<i32>(k % 2, k / 2);
        let surface = water_live_look(first + step).x;
        if surface > NO_WATER_SURFACE * 0.5 {
            let shares = mix(vec2<f32>(1.0) - within, within, vec2<f32>(step));
            sum += surface * shares.x * shares.y;
            weight += shares.x * shares.y;
        }
    }
    if weight <= 0.0 {
        return NO_WATER_SURFACE;
    }
    return sum / weight + water.sim_datum.x - terrain.origin.y;
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
    if cell.block == NO_WATER_BLOCK {
        var level: WaterLevel;
        return level;
    }
    // The texel of the first of the 4 x 4 samples around the point, a sample
    // before the cell's first corner along x and z.
    let origin = water_texel(cell.block, cell.first - vec2<u32>(1u));
    var held: array<vec2<f32>, 16>;
    for (var k = 0u; k < 16u; k++) {
        let steps = vec2<i32>(vec2<u32>(k % 4u, k / 4u));
        held[k] = textureLoad(water_surface, origin.xy + steps, origin.z, 0).xy;
    }
    return water_smooth(held, cell.within, terrain.anchor.w * f32(1u << cell.level));
}

// The smooth surface through 4 x 4 samples `spacing` meters apart, as
// `water_level` takes it, their surfaces (above the terrain's origin;
// NO_WATER_SURFACE where a sample holds none) beside their ground in
// `surfaces`, row by row from the first along z and x, at a point `within`
// the square of the middle four (0 to 1 along x and z).
fn water_smooth(surfaces: array<vec2<f32>, 16>, within: vec2<f32>, spacing: f32)
        -> WaterLevel {
    var samples = surfaces;
    var level: WaterLevel;
    // Their weights along x and z; and the surface of the first holding one,
    // from which the smooth surface's sums count so they keep their
    // precision.
    let along = water_spline(within.x);
    let across = water_spline(within.y);
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
            let held = samples[row * 4u + column];
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
// triangle's edge; and what names a kind the nearest sample's, none where
// that one holds no surface.
fn water_read(cell: WaterCell) -> WaterAt {
    var level = water_level(cell);
    var at = water_at_level(level);
    if level.held <= 0.0 {
        return at;
    }
    let origin = water_texel(cell.block, cell.first - vec2<u32>(1u));
    let nearest = select(vec2<u32>(1u), vec2<u32>(2u), cell.within >= vec2<f32>(0.5));
    for (var row = 0u; row < 4u; row++) {
        for (var column = 0u; column < 4u; column++) {
            let share = level.shares[row][column];
            if share <= 0.0 {
                continue;
            }
            let steps = vec2<u32>(column, row);
            let words = water_words(textureLoad(water_info, origin.xy + vec2<i32>(steps),
                origin.z, 0));
            at = water_words_add(at, words, share, all(steps == nearest));
        }
    }
    return at;
}

// Water at a point as its smooth surface `level` (`water_smooth`'s) holds
// it: its surface, how fast it falls, its depth and how much of it the
// samples holding a surface hold, before their words.
fn water_at_level(level: WaterLevel) -> WaterAt {
    var at: WaterAt;
    at.surface = level.surface;
    at.slope = level.slope;
    at.depth = level.depth;
    at.held = level.held;
    return at;
}

// Water `at` with what a sample's words `words` (`water_words`') say of it
// added: what changes by degrees through the sample's share `share` of the
// smooth surface, and what names a kind where it is the sample `nearest`
// the point.
fn water_words_add(at: WaterAt, words: WaterWords, share: f32, nearest: bool) -> WaterAt {
    var sum = at;
    sum.words.flow += words.flow * share;
    sum.words.foam += words.foam * share;
    sum.words.span += words.span * share;
    sum.words.state += words.state * share;
    if nearest {
        sum.words.lake = words.lake;
        sum.words.sheet = words.sheet;
        sum.words.reach = words.reach;
    }
    return sum;
}

// The look of the live water's cell `at` (along x and z), the nearest cell
// inside the live square for one past its edges, as water's pass draws it
// (`water_sim.wgsl`'s `sim_drawn`): its surface and ground above the
// square's datum and its words' bits. Its block's slot is its place in the
// square turned by `water.sim_datum.yz` around the pool's
// `water.sim_datum.w` slots along each side, as the solve stores it.
fn water_live_look(at: vec2<i32>) -> vec4<f32> {
    let last = i32(water.sim.w) - 1;
    let cell = vec2<u32>(clamp(at, vec2<i32>(0), vec2<i32>(last)));
    let pool = u32(water.sim_datum.w);
    let slot = (cell / WATER_LIVE_BLOCK + vec2<u32>(water.sim_datum.yz)) % pool;
    let local = cell % WATER_LIVE_BLOCK;
    let index = ((slot.y * pool + slot.x) * WATER_LIVE_BLOCK + local.y) * WATER_LIVE_BLOCK
        + local.x;
    return water_live[index * WATER_LIVE_LOOKS + 1u];
}

// The live water at terrain-local `local`, read as tiles' water is
// (`water_read`), its cells' middles its samples (`water_live_look`): the
// smooth surface through the 4 x 4 cells around the point, among those two
// or more inside the live square's edges, their heights lifted from the
// square's datum to the terrain's origin; and their words through its
// shares.
fn water_live_read(local: vec2<f32>) -> WaterAt {
    let n = water.sim.w;
    let cells = (local + terrain.origin.xz - water.sim.xy) / water.sim.z - 0.5;
    let corner = clamp(floor(cells), vec2<f32>(1.0), vec2<f32>(n - 3.0));
    let within = saturate(cells - corner);
    let first = vec2<i32>(corner) - vec2<i32>(1);
    let lift = water.sim_datum.x - terrain.origin.y;
    var surfaces: array<vec2<f32>, 16>;
    var words: array<vec2<u32>, 16>;
    for (var k = 0u; k < 16u; k++) {
        let look = water_live_look(first + vec2<i32>(vec2<u32>(k % 4u, k / 4u)));
        surfaces[k] = look.xy + lift;
        words[k] = bitcast<vec2<u32>>(look.zw);
    }
    var level = water_smooth(surfaces, within, water.sim.z);
    var at = water_at_level(level);
    if level.held <= 0.0 {
        return at;
    }
    let nearest = select(vec2<u32>(1u), vec2<u32>(2u), within >= vec2<f32>(0.5));
    for (var row = 0u; row < 4u; row++) {
        for (var column = 0u; column < 4u; column++) {
            let share = level.shares[row][column];
            if share <= 0.0 {
                continue;
            }
            let word = words[row * 4u + column];
            at = water_words_add(at, water_words(vec4<u32>(word, 0u, 0u)), share,
                all(vec2<u32>(column, row) == nearest));
        }
    }
    return at;
}

// How much of the water a pixel resolving `seen` meters at terrain-local
// `local` (`detail_resolved`, seen from the detail eye) reads from the live
// water (`water_live_read`) rather than the tiles': all of it inside the
// live square while a live cell spans √2 times `water.sim_pixels.x` pixels
// or more there, handing back to the tiles' water over the upper half of
// the octave down to that, as detail levels hand over (`detail_level`), so
// it is gone where the live square's reach ends: where its cells span
// `engine_core::LIVE_PIXELS`, or nearer where the budget of their steps
// cuts the square (`engine_core::WaterWindow::pixels`); and across the
// WATER_LIVE_RIM cells along the square's edges, from where all the samples
// its smooth surface reads lie inside it, two cells in, to its edges,
// where the solve has pulled the live water back to the settled water.
// None where no water runs live.
fn water_live_share(local: vec2<f32>, seen: f32) -> f32 {
    let n = water.sim.w;
    if n < f32(2u * WATER_LIVE_RIM) {
        return 0.0;
    }
    let cells = (local + terrain.origin.xz - water.sim.xy) / water.sim.z;
    let inside = min(min(cells.x, cells.y), min(n - cells.x, n - cells.y));
    let rim = smoothstep(2.0, f32(WATER_LIVE_RIM), inside);
    let handed = detail_level(seen, water.sim.z / (2.0 * water.sim_pixels.x), 2u);
    return rim * select(0.0, 1.0 - handed.blend, handed.level == 0u);
}

// Water `first` blended `toward` water `second`, both as `WaterAt` holds
// them at one point: what changes by degrees mixed, what names a kind the
// one the blend leans to.
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
    return at;
}

// The water at terrain-local `local` as a pixel resolving `seen` meters
// there (`detail_resolved`), seen from the detail eye, reads it, what every
// part of water's shading reads of it: the live water where it runs there
// (`water_live_read`) as far as the pixel takes it (`water_live_share`), and
// the tiles' water elsewhere, at the level of the tiles' pyramid that size
// picks over level 0's spacing (`detail_level`), the next coarser one
// blended in over the upper half of each octave, each from the finest
// resident level at or coarser than its own (`water_cell`), and read once
// where both fall in the same tile; the two blended where the pixel hands
// one over to the other.
fn water_pixel(local: vec2<f32>, seen: f32) -> WaterAt {
    let live = water_live_share(local, seen);
    if live >= 1.0 {
        return water_live_read(local);
    }
    let picked = detail_level(seen, terrain.anchor.w, terrain.directory.y);
    let cell = water_cell(local, picked.level);
    var at = water_read(cell);
    if picked.blend > 0.0 && cell.level == picked.level {
        at = water_blend(at, water_read(water_cell(local, picked.level + 1u)), picked.blend);
    }
    if live <= 0.0 {
        return at;
    }
    return water_blend(at, water_live_read(local), live);
}

// The water at the pixel drawing the point `drawn` of its mesh (less the
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

@vertex
fn vs_water(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> WaterOutput {
    return water_output(vertex, instance);
}

// Water's motion is the camera's: its surface moves only as its state does.
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
// pixel; its normal and the roughness of what a pixel spans of it, along
// the view and across it (`water_roughness`); the normal of the plane it
// lies in without its waves (up for the sea, tilted as terrains' water
// falls, its own normal, which no waves tilt), which the eye looks down onto
// and light enters through; the share of it foam covers; the color its body
// scatters back, and what it keeps of each channel of the light on its way
// to what lies below and back; how near the top of a crest of its waves it
// stands (0 to 1, none for terrains' water); and how many times over each
// channel of light dies by e crossing the walls of its crests
// (`water_extinction` over WATER_CREST_WALL of its waves' root mean square
// heights), where the sun's light crosses on toward the eye.
struct WaterSurface {
    relative: vec3<f32>,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    normal: vec3<f32>,
    roughness: vec2<f32>,
    plane: vec3<f32>,
    // How much the hollows of the foam on it shade it, and how thin it
    // lies, from heaped (0) to a single drained layer (1) (`water_foam`).
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
// layer lit as the surface it covers faces the light, darker in the hollows
// of its heaps and glinting where it lies thin. One shadow lookup serves all
// the sun's light on it, and the sun's share of the light falling on the
// water.
fn water_light(surface: WaterSurface) -> WaterLit {
    let world = eye_world(surface.relative);
    let distance = length(surface.relative);
    let view = stable_normalize(-surface.relative);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let normal = water_facing(surface.normal, view, surface.plane);
    // Foam is rough where it heaps, and smoother where it lies a thin layer,
    // alike every way.
    let foam_roughness = mix(WATER_FOAM_ROUGH.y, WATER_FOAM_ROUGH.x, surface.foam_thin);
    let roughness = mix(surface.roughness, vec2<f32>(foam_roughness), surface.foam);
    let alphas = roughness * roughness;
    let n_v = max(dot(normal, view), 1.0e-4);
    let f0 = vec3<f32>(0.02);
    // Its rough surface masks what it mirrors by its slopes along the view,
    // the way the eye looks across them.
    let reflectance = saturate(luminance(environment_brdf(f0, roughness.x, n_v)));
    let sun = lighting.sun.xyz;
    var sunlight = vec3<f32>(0.0);
    if sun.y > 0.0 {
        sunlight = lighting.sun_color.rgb
            * sun_visibility(world, up, surface.world_dx, surface.world_dy, distance, false);
    }
    // The sun's disc mirrored, its lobe the ellipse of the slopes a pixel
    // hides along the view and across it (`water_distribution`,
    // `water_visibility`), widened by the half vectors the disc's rays span
    // (Karis), which also keeps calm water's glint from flickering in single
    // pixels. They span half the disc's angular radius
    // (`lighting.highlights.y`, its sine; none for a point sun) within the
    // plane of incidence, along the view, and that over their cosine with
    // the view out of it, across the view, each way's spread adding to the
    // surface's as a variance, so a mirror shows the disc as bright as it is
    // however grazing the view. A rough surface draws a low sun out into a
    // long path of glitter toward the eye.
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
    // Foam scatters the sun's light, as the surface it covers faces it, and
    // the light around it: the sky's, of which water in its channel sees
    // WATER_FOAM_SKY, and the ground's around it for the rest, sunlit and
    // skylit as the terrain's hollows take it, so foam in shade stands no
    // bluer than the banks beside it. The hollows of its heaps shade it, and
    // its wet bubbles mirror the sun's disc softly, the more sharply where it
    // lies thin.
    let sky = sky_average(up) * lighting.zenith.w;
    let around = lighting.sun_color.rgb * (0.35 * saturate(sun.y)) + sky * 0.5;
    let foam_n_l = dot(surface.normal, sun);
    var foam_glint = vec3<f32>(0.0);
    if foam_n_l > 0.0 && any(sunlight > vec3<f32>(0.0)) {
        let foam_alpha = foam_roughness * foam_roughness;
        let half_vector = normalize(view + sun);
        foam_glint = fresnel(f0, max(dot(view, half_vector), 0.0)) * sunlight
            * (distribution(max(dot(surface.normal, half_vector), 0.0), foam_alpha)
                * visibility(max(dot(surface.normal, view), 1.0e-4), foam_n_l, foam_alpha)
                * foam_n_l * lighting.sun_color.w);
    }
    let foam = WATER_FOAM_ALBEDO * (1.0 - surface.foam_shade)
        * (mix(around, sky, WATER_FOAM_SKY) + sunlight * max(foam_n_l, 0.0)) + foam_glint;
    // The mirrored sky and scene come through `finish`, rough as its surface
    // is along the view and smoother by what it lacks of it across: its
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
// normal `normal`, where the surface gathers it as it refracts it down:
// the area a pixel's beam of sunlight lands on through a level surface
// over the area it lands on through the surface (the inverse of the
// Jacobian of where the refracted rays land, from its change across the
// pixel, gathering at most a thousandfold), raised to
// WATER_CAUSTIC_CONTRAST so its mean stays near 1 and its peaks within
// WATER_CAUSTIC_MOST, which the graded scene holds unclipped. Terrains'
// water's surface is the simulated surface, smooth through its samples (the
// cubic B-spline through the tiles' level a pixel resolves or the live
// water's cells, `water_pixel`), tilted by the waves its particles raise on
// it (`water_splatted`), so its normal turns across a pixel by
// the surface's own curvature: a surface bending κ a meter gathers the light
// landing d meters under it about 1/(1 − d·κ·(1 − 1/n)) times, n water's
// index of refraction. Caustics come from that curvature alone. Where the
// surface bends little, a stream standing a few centimeters over a meter or
// a level pool, they gather a percent or less and go of themselves; they
// show where the simulation bends its surface sharply, over its jumps and
// drops, and travel with the waves turbulence raises on it.
// Its derivatives need it called in uniform control flow.
const WATER_CAUSTIC_CONTRAST: f32 = 0.3;
const WATER_CAUSTIC_MOST: f32 = 1.5;
// The meters of water over which caustics fade, as the light the surface
// gathers blurs and scatters on its way down.
const WATER_CAUSTIC_DEPTH: f32 = 1.5;

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

// Water at the pixel `pixel` (its center) as `at` holds it, drawn at the
// point `point` of its surface (less the eye), refracting the scene's color
// when `refracts`. It lies wherever its surface stands over what lies
// behind the pixel, up to where the samples holding it end, and shows what
// its state holds there and nothing else.
fn shade_water(point: vec3<f32>, pixel: vec2<f32>, at: WaterAt, refracts: bool)
        -> WaterShade {
    let world = eye_world(point);
    // The water ends where the samples holding its surface do, halfway to
    // the first holding none (`WaterAt`'s `held`).
    if at.held < WATER_HELD {
        discard;
    }
    // The plane its surface lies in, tilted as it falls: the simulated
    // surface's own slope. The eye looks down onto it, light enters through
    // it and what lies behind the pixel stands under it, so water running
    // down toward the eye from above its height shows as water below it does.
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
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    // The water as its state holds it: how turbulent it runs and where white
    // water is born in it.
    let turbulence = at.words.state.x;
    let born = at.words.state.y;
    // How fast it runs, and how fast its eddies turn (`water_stirring`):
    // shallow water running fast down a steep surface stirs fast eddies,
    // deep slow water slow ones (`water_eddies`), and turbulent water as fast
    // as its speed swings.
    let speed = length(flow);
    let stirring = water_stirring(speed, water_eddies(speed, length(at.slope), depth),
        turbulence);
    // What the water's particles splat over the pixel's footprint
    // (`water_splatted`): the shares of it the white water and the foam it
    // carries cover, and the waves the turbulence reaching its surface raises
    // on it, as much of them as the particles show; past their reach the
    // white water born where it breaks and the foam its flow churned up as
    // its own state holds them, the particles' mean, on a surface they draw
    // nothing on.
    let footprint = water_footprint(world_dx.xz, world_dy.xz);
    let splatted = water_splatted(world, footprint,
        vec2<f32>(water_white(born), saturate(at.words.foam / WATER_FOAM_MOST)));
    // Its normal: the simulated surface's own, tilted by those waves, so its
    // glints and what it mirrors travel down the current with them.
    let fall = at.slope + splatted.fall;
    let normal = normalize(vec3<f32>(fall.x, 1.0, fall.y));
    // What neither resolves of its surface roughens it by the simulation's
    // own turbulence and eddies there (`water_unresolved`): all of it past the
    // particles' reach, and where they show, what the eddies smaller than
    // their waves' lift, with the slopes of the waves the pixel's footprint
    // hides; and so does how far its normal turns across the pixel
    // (`water_turning`), so its glints widen instead of sparkling: along the
    // view and across it as the ellipse of those slopes projects into the
    // view, so where it runs rough it glints rather than mirroring the sky,
    // its glints and what it mirrors drawn out along the view (Bruneton,
    // Neyret and Holzschuch, `water_roughness`); where nothing stirs it, it is
    // a calm mirror.
    let view = stable_normalize(-point);
    let resolved = water_eddy_size(depth, water.foam.x, water.foam.w);
    let unresolved = mix(water_unresolved(depth, stirring, 0.0),
        water_unresolved(depth, stirring, resolved) + splatted.hidden, splatted.shown);
    let roughness = water_roughness(unresolved + water_turning(fall), water_heading(view));
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
    // How deep the bed under the pixel lies, where the surface's curvature
    // gathers the sun's light (`water_caustics`).
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
        * shallows;
    // Light comes back through the water along the view to what it shows,
    // as far as the depth lets it run, after reaching it down from the
    // surface, taken by the water and scattered by its sediment and bubbles.
    let through = min(max(length(behind.ground - point), 0.0), // optics: the view's way through the water
        depth * plane.y / max(dot(view, plane), 0.1)); // optics: the view's way through the water
    let down = clamp(dot(point - behind.ground, plane) / plane.y, 0.0, through); // optics: the light's way down to what the view meets
    // The share of it white water covers; the water holds bubbles where it is
    // white and where it carries foam.
    let white = splatted.foam.x;
    let aerated = max(white, splatted.foam.y);
    let transmit = water_transmitted(through + down, aerated);
    // Foam (`meshes/water_foam.wgsl`): the white water, and fresh white water
    // where the water runs against what stands in it behind the pixel fast
    // enough to tear air in, by its speed, how far its surface stands over
    // what it meets and how much of its depth that blocks (`water_contact`,
    // `water_blocking`); and the foam it carries, drained as it ages
    // (`water_foam`).
    let contact = water_contact(standing, standing_change, footprint, speed,
        water_blocking(standing, depth));
    let foam = water_foam(1.0 - (1.0 - white) * (1.0 - contact), splatted.foam.y);
    var surface: WaterSurface;
    surface.relative = point;
    surface.world_dx = world_dx;
    surface.world_dy = world_dy;
    surface.normal = normal;
    surface.roughness = roughness;
    surface.plane = plane;
    surface.foam_shade = foam.shade;
    surface.foam_thin = foam.thin;
    surface.foam = foam.cover;
    surface.scatter = water_body(aerated);
    surface.transmit = transmit;
    // Its surface raises no crests for the sun's light to cross.
    surface.crest = 0.0;
    surface.crest_wall = vec3<f32>(0.0);
    let lit = water_light(surface);
    // The surface's caustics brighten the sun's share of the light on the
    // bed, fading as it lies deeper, as the bubbles churning water holds
    // scatter that light on its way down.
    let caustic = 1.0 + (gathered - 1.0) * lit.sunny
        * exp(-bed_depth * (1.0 / WATER_CAUSTIC_DEPTH + WATER_BUBBLES * aerated));
    // Of what the surface lets through, the ground shows darker for being
    // wet however thin the water over it, over WATER_SOAKED of it as smoothly
    // as the pixel spans it; foam covers it.
    let wet = water_ramp(standing, WATER_SOAKED, standing_change) * WATER_WET;
    let let_through = (1.0 - lit.reflectance) * (1.0 - foam.cover);
    var result: WaterShade;
    result.shaded = lit.shaded;
    result.coverage = max(max(1.0 - (1.0 - lit.reflectance) * luminance(transmit), wet),
        foam.cover) * fade;
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

// The water at the pixel `pixel` drawing the point `mesh` of its mesh (less
// the eye), refracting the scene's color when `refracts`: its water read
// where the point lies (`water_drawn`), live or settled, shaded on its
// surface there.
fn shade_mesh(mesh: vec3<f32>, pixel: vec2<f32>, refracts: bool) -> WaterShade {
    let at = water_drawn(mesh);
    return shade_water(water_on_surface(mesh, at), pixel, at, refracts);
}

// Colors blend premultiplied over the scene: the scene target's alpha
// keeps the ground's ambient share, the other targets take the water's.
// The canvas holds no copy of the scene's color: water blends over the
// ground there.
@fragment
fn fs_water(input: WaterOutput) -> @location(0) vec4<f32> {
    let shade = shade_mesh(input.relative, input.clip_position.xy, false);
    return vec4<f32>(encoded(shade.shaded.color, input.clip_position.xy) * shade.coverage,
        shade.coverage);
}

@fragment
fn fs_water_post(input: WaterOutput) -> SceneTargets {
    let shade = shade_mesh(input.relative, input.clip_position.xy, true);
    var targets = scene_targets(shade.shaded, input.clip_position.xy);
    targets.color = water_covering(shade, input.clip_position.xy);
    return targets;
}

@fragment
fn fs_water_temporal(input: WaterTemporalOutput) -> TemporalTargets {
    let shade = shade_mesh(input.relative, input.clip_position.xy, true);
    var targets = temporal_targets(shade.shaded, input.clip_position, input.previous);
    targets.color = water_covering(shade, input.clip_position.xy);
    return targets;
}

// The least speed turbulence measures water's swings of speed against, and
// the meters deep thinner water breaks as (`engine_terrain::TURBULENT_SPEED`,
// `BREAK_THINNEST`).
const WATER_TURBULENT_SPEED: f32 = 0.5;
const WATER_BREAK_THINNEST: f32 = 0.02;
const WATER_BREAK_ONSET: f32 = 0.6;
const WATER_BREAK_FULL: f32 = 1.2;

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
// from `engine_terrain::BREAK_ONSET` of which they break it white and from
// `BREAK_FULL` white all over.
fn water_eddy_froude(speed: f32, slope: f32, depth: f32) -> f32 {
    return water_eddies(speed, slope, depth)
        / sqrt(WATER_GRAVITY * max(depth, WATER_BREAK_THINNEST));
}
