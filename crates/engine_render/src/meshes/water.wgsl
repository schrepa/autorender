// Water: terrains' water as their tiles hold it, drawn on the leaves of a
// subdivision of its own over each terrain's square (`terrain/surface.rs`),
// which splits its leaves by how far they stray from the water's surface and
// collapses those over no water (`terrain_subdivision.wgsl`). Each vertex
// stands at the water's surface field where it lies (`water_field`: wet
// samples' surfaces and the surface continued under the banks, from the
// finest resident level holding it around the point), lifted by the pixels
// the drawn ground and the water's mesh may each stray from their surfaces
// seen from the detail eye (`water_lift`), so the ground drawn under the
// water never hides it. Each pixel then reads the water's state where it
// lies (`water_pixel`), at the level of the tiles' pyramid a pixel's span
// along the view seen from the detail eye picks, handing over to the next
// coarser one as detail levels do (`detail_level`): its depth over that
// level's own ground, whose zero crossing is the shore, and what its
// samples' words say of the water (`WaterWords`: its flow, the foam it
// carries, how turbulent it runs, where white water is born in it, its
// shear, how far across its water body is and whether a lake holds it, the
// reach of its stream and what stands in it), blended on the level's
// triangles, so none of it moves as the eye does and no leaf shapes it.
// Where water is simulated around the camera, its cells hold the same state
// (`sim_water_output`), the span of its body and what its bed holds taken
// from the settled water's sample nearest each. Everything drawn of the
// water reads that state and what follows from it: its Froude number, its
// eddies' speed (`water_eddies`) and its span, the fetch the wind has over
// it; the water's debug views paint each of its fields (`meshes/debug.wgsl`).
// Its waves come from that state alone (`water_flow.wgsl`): the
// wind's, as its spectrum raises them over the water's span and depth
// (`water_spectrum.wgsl`), standing where the water is still and riding
// its current where it runs, with the waves its eddies raise, one to two
// depths long and as high as their speed squared over gravity; short waves
// its bed holds still where it runs near the speed of its own waves; and
// waves standing about what it flows past. Each is drawn as far as it spans
// the steady size (STEADY_PIXELS), and what the pixel averages of its
// slopes, and how far those drawn turn across it, roughens the surface. It
// mirrors the sky, the sun's disc and,
// through screen-space reflections, the scene; its body, tinted by the
// sediment it carries, scatters the light entering it back toward the
// eye; and what lies under it shows through, bent by the waves where the
// frame copies the scene's color, brighter where the waves gather the sun's
// light on it, and dimmed channel by channel over the light's path down to
// it and back, the ground darker for being wet. Its foam is its own: what
// its flow churns up and carries, settling or simulated
// (`engine_terrain::shallow_water`), and white water where it breaks, its
// crests fold or it runs fast against what stands in it, all one bubble
// texture carried as the water moves (`meshes/water_foam.wgsl`). However
// thin, the water fades only at its waterline. The sea (`meshes/ocean.wgsl`)
// lights its surface the same way. Where rivers leave their beds, their
// jets fall as sheets of their own (`water_falls.rs`), drawn last.

// Tiles' water, a block of samples for each patch holding any surface
// (`terrain/water_blocks.rs`): their surfaces (terrain-local meters,
// NO_WATER_SURFACE where none reaches) beside their tile's ground (x and y);
// and their words (`engine_terrain::TileWater::info`), a texel's channels:
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
// bytes); and how fast their surface and their depth fall along x and z,
// between their tile's samples either side holding a surface
// (`water_gradients`: w, four signed bytes).
@group(1) @binding(50) var water_surface: texture_2d_array<f32>;
@group(1) @binding(51) var water_info: texture_2d_array<u32>;
// Water's directory (`terrain/water_blocks.rs`), WATER_TILE_WORDS words for
// each layer of the terrain's height array: the block of each patch of the
// tile in it, row by row (NO_WATER_BLOCK where none of its samples holds a
// surface).
@group(1) @binding(59) var<storage, read> water_directory: array<u32>;
// The subdivision of the terrain's water (`terrain/surface.rs`), whose
// leaves its draws take, a meshlet an instance, from `tree.z` of the frame's
// (`terrain_surface_leaves`).
@group(1) @binding(60) var<uniform> water_tree: TerrainSubdivision;

// Water, or the sea: the camera's view and projection without its
// translation (the sea's; terrains' water draws through the terrain's
// view); the color deep water scatters back (rgb) and the meters through
// which half the light passes (w); the wind's velocity along x and z (xy,
// the sea's own for the sea) and how strongly it gusts (z); for the sea,
// the compression below which its crests whiten (x), the amplitude of one
// wave holding all its waves' variance, which runs up its shores (y), and
// its waves' summed height (w), none of it for terrains' water (`foam`);
// how fast the water takes each channel of light, per meter (rgb); and each
// of the waves' tiles (`water_spectrum.wgsl`): how many times a meter it
// repeats, the longest and the shortest waves it holds (meters) and the
// root mean square height of its waves at unit slope variance (meters).
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
    wind: vec4<f32>,
    foam: vec4<f32>,
    absorption: vec4<f32>,
    waves: array<vec4<f32>, WATER_TILES>,
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
// block of the window's square in its slot, SIM_WATER_BLOCK cells along each
// side): each cell's surface (world meters), its flow along x and z (square
// meters a second) and the foam it carries, a block's row by row; and each
// block's ground, SIM_BLOCK_GROUND entries of it: at its cells' corners (a
// row of its cells and one more at a time), then the levels lakes hold its
// cells at and its cells' beds' roughness; and what the bed holds under
// each cell, in the cells' order, as the settled water's sample nearest it
// holds it (`engine_terrain`'s `Tile::bed_words`): its reach's code (bits 0
// to 7) beside its span's code as its words' fourth byte packs it (bits 8
// to 15), and its nearest obstacle, offset from the cell's middle, as
// tiles' words pack them.
@group(1) @binding(56) var<storage, read> water_sim_cells: array<vec4<f32>>;
@group(1) @binding(57) var<storage, read> water_sim_ground: array<f32>;
@group(1) @binding(58) var<storage, read> water_sim_beds: array<vec2<u32>>;
const SIM_WATER_BLOCK: u32 = 8u;
const SIM_BLOCK_GROUND: u32 = 209u;
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

// Samples along a block's side and blocks along a layer's side, no block,
// and what a sample no water reaches holds; patches along a side of a tile,
// the words each layer takes in water's directory, and a terrain
// directory's entry of a tile not resident (`gpu_layouts::constants`).
const WATER_BLOCK: u32 = 17u;
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
// `WATER_RADIUS_STEP`); and the meters a meter the packed gradients reach
// (`gpu_layouts::constants::WATER_GRADIENT_RANGE`).
const WATER_FLOW_RANGE: f32 = 8.0;
const WATER_SPAN_LEAST: f32 = 0.25;
const WATER_SPAN_MOST: f32 = 4096.0;
const WATER_SHEAR_RANGE: f32 = 8.0;
const WATER_OBSTACLE_STEP: f32 = 0.125;
const WATER_RADIUS_STEP: f32 = 0.0625;
const WATER_GRADIENT_RANGE: f32 = 4.0;
// How much darker the ground under water looks: of the light it scatters
// back, the surface turns part back down. Water a few centimeters deep
// darkens it as much.
const WATER_WET: f32 = 0.35;
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

// What a sample's words say of its water: its flow along x and z (meters a
// second), the foam it carries (0 none, WATER_FOAM_MOST the most), how far
// across its water body is (its span, meters: a river's width, a lake's
// equal-area diameter, the fetch the wind has over it; none where none is
// known), whether a lake holds it and whether a fall's sheet runs over it;
// how turbulent it runs, how strongly it breaks white and its shear (a
// second); its reach's code (0 for none, else 1 plus
// `engine_terrain::Reach`); and its nearest obstacle's word
// (`water_obstacle`).
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

// How fast a sample's surface (xy) and its depth (zw) fall along x and z
// (meters a meter), as the fourth of its words packs them
// (`terrain/water_blocks.rs`): signed bytes, each the square root of its
// share of WATER_GRADIENT_RANGE.
fn water_gradients(word: u32) -> vec4<f32> {
    let root = unpack4x8snorm(word);
    return root * abs(root) * WATER_GRADIENT_RANGE;
}

// The terrain's extent along x and z (meters): its level-0 tiles'.
fn water_extent() -> vec2<f32> {
    let header = terrain.directory.x;
    let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
    return vec2<f32>(tiles * (WATER_TILE_PATCHES * (WATER_BLOCK - 1u))) * terrain.anchor.w;
}

// Water at a point as shading reads it: its surface's height above the
// terrain's origin and its depth over the ground there (none or less where
// it is dry); what its samples' words say of it, what changes by degrees
// (its span among it) blended and what names a kind (a lake holding it, a
// fall's sheet over it, its reach, its nearest obstacle's word) its nearest
// sample's; how fast its depth and its surface fall along x and z (meters a
// meter); and its nearest obstacle as `water_obstacle` gives it, offset
// from the point, how strongly a drop or a jump lies just upstream blended.
struct WaterAt {
    surface: f32,
    depth: f32,
    words: WaterWords,
    shore: vec2<f32>,
    slope: vec2<f32>,
    obstacle: vec4<f32>,
}

// Where terrain-local `local` lies in the tiles' water of the finest level
// at or coarser than `level` whose tile holding it is resident: that level
// (the terrain's levels where none is, or the point lies past the terrain),
// the block of the patch holding the point's cell (NO_WATER_BLOCK where
// none of the patch's samples holds a surface, or no level is resident),
// the cell's first corner in the block, and where in the cell the point
// lies (0 to 1 along x and z).
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
    let quads = WATER_BLOCK - 1u;
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
        // the patch holding the cell.
        let within = clamp(samples - vec2<f32>(tile * tile_quads), vec2<f32>(0.0),
            vec2<f32>(f32(tile_quads)));
        let corner = min(vec2<u32>(within), vec2<u32>(tile_quads - 1u));
        let tile_patch = corner / quads;
        cell.level = index;
        cell.block = water_directory[layer * WATER_TILE_WORDS + tile_patch.y * WATER_TILE_PATCHES
            + tile_patch.x];
        cell.first = corner - tile_patch * quads;
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

// The water's surface field at terrain-local `local`: its height above the
// terrain's origin where wet samples' surfaces and the surface continued
// under the banks (`engine_terrain::TileWater`) hold it, from the finest
// resident level whose samples of the triangle holding the point all hold
// one, blended by the triangle's weights; `found` false where no level's
// do. It depends on the point alone, so leaves sharing a vertex meet on it.
struct WaterField {
    surface: f32,
    found: bool,
}

fn water_field(local: vec2<f32>) -> WaterField {
    let levels = terrain.directory.y;
    var field: WaterField;
    var level = 0u;
    while level < levels {
        let cell = water_cell(local, level);
        level = cell.level + 1u;
        if cell.block == NO_WATER_BLOCK {
            continue;
        }
        let weights = water_weights(cell.within);
        let upper = cell.within.x + cell.within.y > 1.0;
        var surface = 0.0;
        var held = true;
        for (var corner = 0u; corner < 3u; corner++) {
            let texel = water_texel(cell.block, cell.first + water_corner(corner, upper));
            let sample = textureLoad(water_surface, texel.xy, texel.z, 0).x;
            held = held && sample > NO_WATER_SURFACE * 0.5;
            surface += sample * weights[corner];
        }
        if held {
            field.surface = surface;
            field.found = true;
            return field;
        }
    }
    return field;
}

// Tiles' water at a point as `cell` (`water_cell`'s) holds it: from its
// samples of the triangle holding the point, each one's surface over its
// level's own ground (so coarse levels' channels, cut to their lowest,
// stand as deep as they hold them), a sample no water reaches standing a
// meter under its ground, WATER_DRY dry; its words, slopes and shoaling
// blended by the triangle's weights; and what names a kind and the nearest
// obstacle, its offset taken from the point, the nearest sample's. Where
// the cell holds no block, the water stands WATER_DRY dry.
fn water_read(cell: WaterCell) -> WaterAt {
    var at: WaterAt;
    at.depth = -WATER_DRY;
    if cell.block == NO_WATER_BLOCK {
        return at;
    }
    at.depth = 0.0;
    let spacing = terrain.anchor.w * f32(1u << cell.level);
    let weights = water_weights(cell.within);
    let upper = cell.within.x + cell.within.y > 1.0;
    var nearest = 4.0;
    for (var corner = 0u; corner < 3u; corner++) {
        let steps = water_corner(corner, upper);
        let texel = water_texel(cell.block, cell.first + steps);
        let held = textureLoad(water_surface, texel.xy, texel.z, 0).xy;
        let info = textureLoad(water_info, texel.xy, texel.z, 0);
        let words = water_words(info);
        let gradients = water_gradients(info.w);
        let weight = weights[corner];
        let reached = held.x > NO_WATER_SURFACE * 0.5;
        at.surface += select(held.y - 1.0, held.x, reached) * weight;
        at.depth += select(-WATER_DRY, held.x - held.y, reached) * weight;
        at.words.flow += words.flow * weight;
        at.words.foam += words.foam * weight;
        at.words.span += words.span * weight;
        at.words.state += words.state * weight;
        at.slope += gradients.xy * weight;
        at.shore += gradients.zw * weight;
        at.obstacle.w += f32(words.obstacle >> 24u) / 255.0 * weight;
        // The obstacle's offset from the point: the sample's own, plus the
        // sample's from the point.
        let toward = vec2<f32>(steps) - cell.within;
        if dot(toward, toward) < nearest {
            nearest = dot(toward, toward);
            at.words.lake = words.lake;
            at.words.sheet = words.sheet;
            at.words.reach = words.reach;
            at.words.obstacle = words.obstacle;
            at.obstacle = vec4<f32>(water_obstacle(words.obstacle, toward * spacing).xyz,
                at.obstacle.w);
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
    at.words.flow = mix(first.words.flow, second.words.flow, toward);
    at.words.foam = mix(first.words.foam, second.words.foam, toward);
    at.words.span = mix(first.words.span, second.words.span, toward);
    at.words.state = mix(first.words.state, second.words.state, toward);
    at.shore = mix(first.shore, second.shore, toward);
    at.slope = mix(first.slope, second.slope, toward);
    at.obstacle.w = mix(first.obstacle.w, second.obstacle.w, toward);
    return at;
}

// Tiles' water at terrain-local `local` as a pixel spanning `seen` meters
// along the view there, seen from the detail eye, reads it: at the level of
// the tiles' pyramid that span picks over level 0's spacing
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
// spans the plane of the mesh's triangle along the view (`water_pixel`,
// `detail_span`). Its derivatives need it called in uniform control flow.
fn water_drawn(drawn: vec3<f32>) -> WaterAt {
    let world = eye_world(drawn);
    // The triangle's plane, facing up.
    let facing = cross(dpdx(world), dpdy(world));
    let up = select(-facing, facing, facing.y >= 0.0);
    let squared = dot(up, up);
    let plane = select(vec3<f32>(0.0, 1.0, 0.0), up * inverseSqrt(max(squared, 1.0e-30)),
        squared > 1.0e-24);
    return water_pixel(world.xz - terrain.origin.xz, detail_span(world, plane).y);
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
// (`detail_across`). The ground drawn under water then never hides it, and
// the mesh stands no more than those pixels off the water.
fn water_lift(world: vec3<f32>) -> f32 {
    return (water.lift.x + water.lift.y) * detail_across(world);
}

// Lattice vertex `vertex` of the leaf at `instance` of the subdivision of
// the terrain's water (`water_tree`): the point of the leaf's meshlet
// (`surface_vertex`) at the water's surface field there (`water_field`),
// lifted over it (`water_lift`); a meter under the ground where no level
// holds the field there, as samples no water reaches stand.
fn water_output(vertex: u32, instance: u32) -> WaterOutput {
    let local = surface_vertex(terrain_surface_leaves[water_tree.tree.z + instance], vertex);
    let field = water_field(local);
    var height = field.surface;
    if !field.found {
        height = terrain_true_height(local) - 1.0;
    }
    var relative = water_relative(local, height);
    if field.found {
        relative.y += water_lift(eye_world(relative));
    }
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

// Water's motion is the camera's: its ripples move within its surface.
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
    // disc's rays span (Karis), which also keeps calm water's glint from
    // flickering in single pixels. They span half the disc's angular radius
    // (`lighting.highlights.y`, its sine; none for a point sun) within the
    // plane of incidence and that over their cosine with the view out of it,
    // as many as a round lobe holds whose radius is the two's geometric
    // mean, so a mirror shows the disc as bright as it is however grazing
    // the view; the waves' spread of normals adds to it as a variance. A
    // rough surface draws a low sun out into a long path of glitter toward
    // the eye.
    var glint = vec3<f32>(0.0);
    let n_l = dot(normal, sun);
    if n_l > 0.0 && any(sunlight > vec3<f32>(0.0)) {
        let half_vector = normalize(view + sun);
        let v_h = max(dot(view, half_vector), 1.0e-4);
        let disc = 0.5 * lighting.highlights.y;
        let widened = sqrt(alpha * alpha + disc * disc / v_h);
        glint = fresnel(f0, v_h) * sunlight
            * (distribution(max(dot(normal, half_vector), 0.0), widened)
                * visibility(n_v, n_l, widened) * n_l * lighting.sun_color.w);
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
    lit.shaded = finish(point, mix(body + glint, foam, surface.foam));
    lit.reflectance = reflectance;
    // Screen-space reflections stand in for what it mirrors of what lies
    // around, at most that share of its light (`ambient`: its body takes no
    // diffuse light of the sky), so its glints, its lit crests and its foam
    // keep theirs where they outshine what it mirrors.
    lit.shaded.reflect = min(lit.shaded.reflect, lit.shaded.ambient);
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

// The slopes of a river's surface (meters a meter, from the first to the
// second) over which, under a fall's sheet, it comes to run as a film.
const WATER_FILM: vec2<f32> = vec2<f32>(0.5, 0.9);

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
// point `mesh` of its mesh over the point `point` of its surface (both less
// the eye), refracting the scene's color when `refracts`: tiles' water, or
// the simulated water, drawn on its surface, when `simulated`.
fn shade_water(mesh: vec3<f32>, point: vec3<f32>, pixel: vec2<f32>, at: WaterAt,
        refracts: bool, simulated: bool) -> WaterShade {
    let world = eye_world(point);
    if !simulated && water_simulated(world.xz) {
        discard;
    }
    // Meters of water over the ground: the shore's dry side holds none.
    let depth = at.depth;
    if depth <= 0.0 {
        discard;
    }
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
    // The plane its surface lies in without its waves, tilted as it falls:
    // the eye looks down onto it, light enters through it and what lies
    // behind the pixel stands under it, so water running down toward the eye
    // from above its height shows as water below it does.
    let plane = normalize(vec3<f32>(at.slope.x, 1.0, at.slope.y));
    // How far the pixel spans on the water, which filters the patterns it
    // shows of it; and how far it spans along the view on that plane seen
    // from the detail eye (`detail_span`), which any handover by size on
    // screen measures.
    let footprint = max(length(world_dx.xz), length(world_dy.xz));
    let seen = detail_span(world, plane).y;
    // Each place's patterns start over at their own times, so the flow
    // never pulses all at once (Vlachos's phase noise): the cycles its
    // clock runs ahead by.
    let stagger = water_noise(world.xz / WATER_FLOW_PHASE).x;
    // Waves stand where the bed holds them, about the obstacle nearest the
    // water's samples: only where a pixel's samples agree on one, the
    // obstacle staying put as the point moves across the pixel.
    let anchor = world.xz + at.obstacle.xy;
    let anchoring = (length(dpdx(anchor)) + length(dpdy(anchor)))
        / max(length(world_dx.xz) + length(world_dy.xz), 1.0e-6);
    let anchored = 1.0 - smoothstep(0.1, 0.4, anchoring); // filter: where a pixel's samples disagree on the obstacle, its waves average out
    // How far it is from still to running: still water's waves stand where
    // they evolve, running water's ride its current, and between, the two
    // blend, so nothing changes at once where a river meets a lake.
    let flowing = smoothstep(0.05, 0.3, speed);
    // The wind's waves: each of its tiles as strong as its spectrum gives
    // that tile's band where the wind, gusting, blows over as much water as
    // the water's body spans, as deep as it is here.
    let gains = water_wind_gains(water_wind_speed(world.xz, time), at.words.span, depth);
    // Its waves, from its state alone (`water_flow_waves`): the wind's and
    // its eddies', the short waves its bed holds and those standing about
    // what it flows past.
    let waves = water_flow_waves(world.xz, world_dx.xz, world_dy.xz, at, gains, eddy, froude,
        flowing, anchored, time, stagger);
    // They ride its surface, tilting as it falls; and the slopes they hide
    // within a pixel roughen it, so where they run rough, it glints rather
    // than mirroring the sky (Bruneton, Neyret and Holzschuch), as does how
    // far the normal of those it draws turns across the pixel
    // (`water_turning`), so their glints widen instead of sparkling.
    let slopes = waves.slopes - at.slope;
    let normal = normalize(vec3<f32>(-slopes.x, 1.0, -slopes.y));
    let roughness = water_roughness(waves.hidden + water_turning(normal));
    let view = stable_normalize(-point);
    // Under the sea, the sea's water stands instead: its level
    // (`water.ocean.x`) is held relative to the eye.
    if water.ocean.y > 0.5 && world.y < eye_world(vec3<f32>(0.0, water.ocean.x, 0.0)).y {
        discard;
    }
    // What lies behind: the opaque surface behind the pixel and, where the
    // frame refracts the scene, what the bent view meets and its color.
    let own = water_ground(pixel);
    var behind: WaterBehind;
    behind.ground = own;
    behind.color = vec3<f32>(0.0);
    if refracts {
        behind = water_behind(point, normal, view, pixel, own, plane);
    }
    // How far the water's surface stands over what stands in it behind the
    // pixel, the meters it lies under the surface's plane: the bed under the
    // pixel, where the waves gather the sun's light. And how far the water's
    // mesh stands over it, where what stands in the water rises through the
    // water as drawn, and how fast that changes across the pixel: the ground
    // drawn under tiles' water may stand over its surface by as much as the
    // mesh stands over it (`water_lift`), and the water lies on it there as
    // shallow water.
    let standing = max(dot(point - own, plane) / plane.y, 0.0); // optics: what the view meets behind the pixel
    let mesh_standing = max(dot(mesh - own, plane) / plane.y, 0.0); // optics: what the view meets behind the pixel
    let mesh_change = abs(dpdx(mesh_standing)) + abs(dpdy(mesh_standing));
    let bed = min(depth, standing);
    let gathered = water_caustics(world, normal, bed);
    // The water ends at its waterline, where its depth comes to nothing or
    // what stands in it rises through its mesh; thin water shows by how fast
    // it runs: calm shallows fade in over WATER_SHORE of depth, while racing
    // water, however thin, is all there.
    let racing = smoothstep(WATER_RACING.x, WATER_RACING.y, speed);
    let shallows = mix(smoothstep(0.0, WATER_SHORE, min(depth, mesh_standing)), 1.0, racing);
    let fade = min(water_edge(depth, fwidth(depth)), water_edge(mesh_standing, mesh_change))
        * shallows * (1.0 - falling);
    // Light comes back through the water along the view to what it shows,
    // as far as the depth lets it run, after reaching it down from the
    // surface, taken by the water and scattered by its sediment and bubbles.
    // Water a fall's sheet runs over (`shade_fall`) is a film over steep
    // rock: the sheet draws its white and its body.
    let steep = length(at.slope);
    let sheet = select(0.0, 1.0, at.words.sheet);
    let film = smoothstep(WATER_FILM.x, WATER_FILM.y, steep) * sheet;
    let through = min(max(length(behind.ground - point), 0.0), // optics: the view's way through the water
        depth * plane.y / max(dot(view, plane), 0.1)); // optics: the view's way through the water
    let down = clamp(dot(point - behind.ground, plane) / plane.y, 0.0, through); // optics: the light's way down to what the view meets
    // The water holds bubbles where white water is born in it and where it
    // carries the foam its flow churned up.
    let aerated = max(born, saturate(at.words.foam / WATER_FOAM_MOST));
    let transmit = water_transmitted((through + down) * (1.0 - film), aerated);
    // Foam (`meshes/water_foam.wgsl`), the water's own: the foam it carries
    // as its flow churned it up, drained as it ages; and white water, fresh,
    // where it breaks, where its crests fold and where it runs against what
    // stands in it behind the pixel fast enough to tear air in, its current
    // or its eddies, whichever is faster. Under a fall's sheet, where the
    // water runs as a film, the sheet draws its white. It floats where the
    // waves carry it, toward their crests by their slopes, and its relief
    // tilts the normal it is lit by.
    let foam_folding = water_folding(waves.spread, waves.hidden, waves.crest);
    let foam_contact = water_contact(mesh_standing, mesh_change, footprint, max(speed, eddy));
    let foam_fresh = 1.0 - (1.0 - born) * (1.0 - foam_folding) * (1.0 - foam_contact);
    let foam_carried = saturate(at.words.foam / WATER_FOAM_MOST);
    let foam_ridden = world.xz - waves.slopes * WATER_FOAM_RIDE;
    let froth = water_froth(foam_ridden, world_dx.xz, world_dy.xz,
        vec2<f32>(foam_fresh, foam_carried) * (1.0 - film), flow, shear, time, stagger, seen);
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
    // that light on its way down, and as the pixel spans more of the waves
    // gathering it.
    let caustic = 1.0 + (gathered - 1.0) * lit.sunny
        * exp(-bed * (1.0 / WATER_CAUSTIC_DEPTH + WATER_BUBBLES * aerated))
        * (1.0 - smoothstep(WATER_CAUSTIC_PIXELS.x, WATER_CAUSTIC_PIXELS.y, // filter: caustics averaging to their mean of 1
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

// Tiles' water at the pixel `pixel` drawing the point `mesh` of its mesh
// (less the eye), refracting the scene's color when `refracts`: its water
// read where the point lies (`water_drawn`), shaded on its surface there.
fn shade_tiles(mesh: vec3<f32>, pixel: vec2<f32>, refracts: bool) -> WaterShade {
    let at = water_drawn(mesh);
    return shade_water(mesh, water_on_surface(mesh, at), pixel, at, refracts, false);
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
// water's surface, depth and flow, the slopes of its surface and depth
// read across its neighbors, the foam the water carries, how turbulent it
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
// cliff, between cells far apart in height, the falls' sheets show it; and
// the meters over the simulated water below past which those sheets show,
// from the first to the second.
const SIM_SHEER: vec2<f32> = vec2<f32>(0.45, 0.5);
const SIM_SHEETS_ABOVE: vec2<f32> = vec2<f32>(3.0, 6.0);

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

// The simulated water's vertex in the temporal pipelines: also where its
// point stood in the previous frame's view (x, y and w of its clip
// position).
struct SimWaterTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    @location(1) water: vec3<f32>,
    @location(2) body: vec2<f32>,
    @location(3) shore: vec2<f32>,
    @location(4) slope: vec2<f32>,
    @location(5) churned: f32,
    @location(6) state: vec4<f32>,
    @location(7) obstacle: vec4<f32>,
    @location(8) @interpolate(flat) reach: u32,
    @location(9) previous: vec3<f32>,
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
    output.shore = vec2<f32>(west.y - east.y, south.y - north.y) / (2.0 * cell);
    output.churned = sim_water_cell(i, j).w;
    // How turbulent it runs, from the velocities around it as the cook's
    // turbulence reads its water's swings (`engine_terrain`'s
    // `turbulence`): how far its velocity strays from the mean of its
    // neighbors holding water, against its speed, and how fast the water
    // turns there; where white water is born in it, by the rule the cook
    // breaks its settled water white by (`engine_terrain`'s `breaking`):
    // where its eddies break its surface (`water_eddy_froude`), from
    // WATER_BREAK_ONSET to WATER_BREAK_FULL; and its shear. Its body's span,
    // whether a lake holds it, its reach and its nearest obstacle are its
    // bed's: the settled water's sample nearest it, as tiles' words pack
    // them.
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
    output.state = vec4<f32>(select(vec3<f32>(0.0), vec3<f32>(turbulence, born, shear), wet), 0.0);
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
    temporal.shore = output.shore;
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
// there, as shading reads water (`WaterAt`): no fall's sheet runs over it.
fn sim_water_values(input: SimWaterOutput) -> WaterAt {
    var at: WaterAt;
    at.depth = input.water.x;
    at.words.flow = input.water.yz;
    at.words.foam = input.churned;
    at.words.span = input.body.x;
    at.words.lake = input.body.y > 0.5;
    at.words.state = input.state.xyz;
    at.words.reach = input.reach;
    at.shore = input.shore;
    at.slope = input.slope;
    at.obstacle = input.obstacle;
    return at;
}

// The simulated water at the pixel its vertices' outputs `input` blend at,
// refracting the scene's color when `refracts`: drawn on its surface.
fn shade_sim(input: SimWaterOutput, refracts: bool) -> WaterShade {
    return shade_water(input.relative, input.relative, input.clip_position.xy,
        sim_water_values(input), refracts, true);
}

// The simulated water's vertices' outputs in the temporal pipelines
// `input`, without where they stood in the previous frame.
fn sim_water_present(input: SimWaterTemporalOutput) -> SimWaterOutput {
    var output: SimWaterOutput;
    output.clip_position = input.clip_position;
    output.relative = input.relative;
    output.water = input.water;
    output.body = input.body;
    output.shore = input.shore;
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
