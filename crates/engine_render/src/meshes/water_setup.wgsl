// The water's beds and the live water's setup: settling water's blocks'
// ground raised as their regions load (`setup_raise`, `water_settle.rs`);
// and the blocks of the live square a frame sets up, a workgroup a block
// (`setup_blocks`), whole once the level-0 tiles under one, their settled
// water and its records' live cells are resident, its ground alone again,
// or standing as the settled water does while they are not, and the cells
// just past the square's edges, an invocation each (`setup_ring`)
// (`water_sim.rs`); each after an empty first group and the terrain group of
// the terrain the water stands on, through which it reads its materials as
// the ground's mesh band reads them and, live, the ground as its triangles
// hold it and the tiles' water as water's shading reads it; nothing comes
// from the CPU but where the water stands, the solids standing in it, what
// its rivers bring and the live cells of its records, and, settling, its
// bare ground and the level-0 samples it rises by.
//
// Both water's beds are one: a block's ground stands at its cells' corners
// on level 0's triangles, raised by the relief of the ground's materials at
// the level whose texels span a cell (`terrain_ground_rise`): a level the
// cells' spacing sets, not the view, so the bed is a function of where it
// stands alone, its materials blended at level 0, as the ground near the
// eye shades. The relief lies along the ground's smooth normal, as the mesh
// band moves the drawn ground, and raises each corner as high as it raises
// the drawn ground's point there, its rise times the normal's upright part:
// the stones stand where the drawn ones do, within the sideways part of
// their rise, and no higher than their depth on any slope. The live water
// reads the samples it rises by from the tiles' layers
// (`terrain_ground_samples`), settling water from those its region's parts
// carry (`setup_window`), the same samples. A lake lets go of each cell its
// raised ground stands at or over (`setup_held`), as a lake holds no cell it
// does not stand over; and each cell's bed's roughness is the solve's from
// the word of the sample nearest its middle by one rule (`setup_rough`):
// friction counts only the roughness its bed does not hold as its shape.
// The solids standing in the settled water cover each cell by the share of
// SETUP_COVER_POINTS along each side of it they stand on, at the mean
// height their tops stand at there (`engine_terrain::cover_cells`), the
// live water's here, settling water's on the CPU.
//
// Each live cell's water starts exactly where the bake's water stood and
// ran on it, as its tile's record of settled water keeps it
// (`setup_live`, `engine_terrain::LiveWater`): the scheme's own state there
// as its region's last steps left it, the mean surface and flow of the time
// it was wet, over the same raised bed; still at its level where a lake
// holds it, the lake holding it but where its raised ground stands at or
// over the level; dry where the record holds no water. It starts off that
// steady state by noise of zero mean and unit variance hashed at its cell:
// its velocity by as far as its turbulence says its speed swings about its
// mean, the turbulence the tiles' water's sample nearest its middle holds,
// shared by x and z, and its surface by as far as a shallow wave carrying
// that swing heaves it, at most half its depth so it stays wet. The steady
// state is the first entry of its look, which the solve pulls its rim back
// to and measures its swings from, and its bed's codes stand in the top
// bytes of the second's words.
//
// The cells just past the square's edges, and those of its blocks still
// waiting for their data, stand as the settled water does at each
// (`setup_ghost`): as its tile's record keeps it on that cell where the
// record's live cells are resident, else as water's shading reads the tiles'
// water at its middle; held at a lake's level and still, dry, far below any
// ground, or flowing at its flow. Each past the edges brings the square what
// of its flow runs inward across the face it borders; the solve reads the
// waiting blocks' cells as it reads those (`water_sim.wgsl`'s
// `sim_standing_at`), stepping none of them, and water's pass draws the
// tiles' water over them, their looks holding no surface and no ground.

// What a frame sets up. The live water's: the blocks it sets up (each with
// what it sets up of it, SETUP_WHOLE, SETUP_RISEN or SETUP_STANDS), then
// nothing, the cells past the square's edges it sets up (four times its
// cells along a side; none where it sets up none) and the solids standing
// in the settled water over the square; the square's first cell on the
// terrain's lattice of its cells along x and z (an i32's bits each), its
// cells along a side and the live cells along each side of a settled cell;
// the meters across a live cell and the height its heights stand above
// (world meters), then nothing; where its ring starts among the
// simulation's inputs, then nothing; and the level-0 tiles whose records'
// live cells `setup_live` holds, the first along x and z and how many along
// each. Settling water's (`setup_raise`): the blocks whose ground it raises,
// then nothing; the level-0 samples along each side of a block's samples
// and the terrain's last sample along x and z, then nothing; the meters
// across its cells and the height its heights stand above (terrain-local
// meters), then nothing; and nothing, twice.
struct SimSetup {
    counts: vec4<u32>,
    square: vec4<u32>,
    meters: vec4<f32>,
    ring: vec4<u32>,
    tiles: vec4<u32>,
}

@group(2) @binding(41) var<uniform> setup_frame: SimSetup;
// The live water's: the solids standing in the settled water over the
// square, SETUP_SOLID entries each (a float's bits): its center
// (terrain-local) and 1 for a ball, else 0; then the inverse of the axes
// mapping the unit cube (or ball) to it, a column each, the first two beside
// how far it reaches from its center along x and along z. Then each block
// the frame sets up: its slot in the live water's pool, the cell of the
// terrain's lattice of live cells its first corner stands at (an i32's bits
// along x and z) and what it sets up of it. Settling water's
// (`setup_raise`): each block whose ground it raises, two entries: its
// block among the ground's (its slot's first block's index plus its own in
// its pool), the cell of the terrain's lattice of cells its first corner
// stands at (an i32's bits along x and z) and the entry its samples start
// at; then the first of them on the terrain's lattice of level-0 samples
// (an i32's bits along x and z), then nothing. And each block's samples,
// `setup_frame.square.x` along each side, row by row: each one's control
// value, its smooth surface's gradient as a normal map's texel holds it and
// the bits of its bed's roughness's word
// (`engine_terrain::shallow_water::roughness`), then nothing.
@group(2) @binding(42) var<storage, read> setup_list: array<vec4<u32>>;
// The simulation's ground (`water_sim.wgsl`'s `sim_ground`), SETUP_GROUND
// entries a block of its pools: the ground at its cells' corners over its
// datum (SETUP_SIDE a row, rows along z), then each of its cells' lake's
// level (SETUP_UNHELD where no lake holds it), bed's roughness, share the
// solids cover and the height their tops stand at over the datum.
@group(2) @binding(43) var<storage, read_write> setup_ground: array<f32>;
// The live water's step's start (`sim_cells`' first state), each block's
// cells row by row: each one's surface and flow along x and z, then
// nothing; a block standing as the settled water does, each one's surface
// (far below any ground where dry), flow along x and z and the level a lake
// holds it at (SETUP_UNHELD where none does).
@group(2) @binding(44) var<storage, read_write> setup_cells: array<vec4<f32>>;
// The live water's cells' looks (`sim_averages`), two a cell: its settled
// water, then its water as water's pass draws it.
@group(2) @binding(45) var<storage, read_write> setup_looks: array<vec4<f32>>;
// The live water's inputs (`sim_inputs`), whose ring it writes from
// `setup_frame.ring.x`, two entries a cell past the edges: its surface,
// flow along x and z and lake's level, then the flow it brings in, then
// nothing.
@group(2) @binding(46) var<storage, read_write> setup_edges: array<vec4<f32>>;
// The live cells of the settled water's records of the level-0 tiles
// `setup_frame.tiles` spans, over the live square's blocks and a block past
// them on every side (`water_sim.rs`'s `live_words`): for each of those
// tiles, row by row, the word its table starts at (SETUP_NO_LIVE where its
// record is not held); each table its first block on the terrain's lattice
// of blocks of live cells along x and z (an i32's bits each) and its blocks
// along each, SETUP_LIVE_HEAD words, then the word each of its blocks' cells
// start at, row by row (SETUP_NO_LIVE where none of them is wet); and each
// block's cells, row by row, SETUP_LIVE_CELL words each, a float's bits: the
// surface its water stands at (terrain-local; NO_WATER_SURFACE where it is
// dry) and its flow along x and z (square meters a second; SETUP_LIVE_HELD
// along x where a lake holds it at that surface)
// (`engine_terrain::LiveWater`).
@group(2) @binding(47) var<storage, read> setup_live: array<u32>;

// Entries of a block of the simulation's ground, what a cell no lake holds
// holds as its lake's level, the meters of water under which a cell is dry
// and below which velocities are desingularized, the roughness of ground
// outside rivers' channels and the points along each side of a cell its
// cover is measured at (`engine_terrain`'s `BLOCK_GROUND`,
// `shallow_water::UNHELD`, `shallow_water::DRY`, `shallow_water::THIN`,
// `shallow_water::GROUND_D84`, `COVER_POINTS`); and the cells past the
// edges a workgroup of the ring's setup takes, the entries of its list a
// solid takes, what it sets up of a block it lists, and the records' live
// cells' words (`gpu_layouts::constants`).
const SETUP_GROUND: u32 = 337u;
const SETUP_UNHELD: f32 = -1.0e30;
const SETUP_DRY: f32 = 1.0e-4;
const SETUP_THIN: f32 = 0.01;
const SETUP_GROUND_D84: f32 = 0.05;
const SETUP_COVER_POINTS: u32 = 4u;
const SETUP_RING_GROUP: u32 = 64u;
const SETUP_SOLID: u32 = 4u;
const SETUP_WHOLE: u32 = 0u;
const SETUP_RISEN: u32 = 1u;
const SETUP_STANDS: u32 = 2u;
const SETUP_NO_LIVE: u32 = 0xffffffffu;
const SETUP_LIVE_HEAD: u32 = 4u;
const SETUP_LIVE_CELL: u32 = 3u;
const SETUP_LIVE_HELD: f32 = 1.0e30;
// Corners along a side of a block, a block's corners and its cells, and
// the points a cell's cover is measured at.
const SETUP_SIDE: u32 = WATER_LIVE_BLOCK + 1u;
const SETUP_CORNERS: u32 = SETUP_SIDE * SETUP_SIDE;
const SETUP_CELLS: u32 = WATER_LIVE_BLOCK * WATER_LIVE_BLOCK;
const SETUP_POINTS: u32 = SETUP_COVER_POINTS * SETUP_COVER_POINTS;
// Uniform noise over this either side of zero has unit variance: the square
// root of 3.
const SETUP_NOISE_REACH: f32 = 1.7320508;
// Meters past any height a solid spans, standing in for a span reaching
// without end.
const SETUP_FARTHEST: f32 = 1.0e30;

// A live cell's water as the setup stands it: the surface it stands at
// over the datum (NO_WATER_SURFACE where it is dry), its flow along x and z
// (square meters a second), the level a lake holds it at over the datum
// (SETUP_UNHELD where none does), and how far its speed swings about its
// mean (meters a second).
struct SetupWater {
    surface: f32,
    flow: vec2<f32>,
    held: f32,
    rush: f32,
}

// A live cell as its tile's record of settled water keeps it
// (`setup_live`'s): whether the record's live cells over it are resident;
// the surface its water stands at over the datum (NO_WATER_SURFACE where it
// is dry, or they are not resident); its flow along x and z (square meters
// a second, none where a lake holds it); and whether a lake holds it, at
// that surface.
struct SetupLive {
    resident: bool,
    surface: f32,
    flow: vec2<f32>,
    held: bool,
}

// The ground over the datum at corner `corner` of the terrain's lattice of
// live cells, on the finest resident level's triangles (the terrain's
// origin where no level holds it).
fn setup_ground_at(corner: vec2<i32>) -> f32 {
    let local = vec2<f32>(corner) * setup_frame.meters.x;
    return (terrain.origin.y + terrain_level_height(local, 0u).height) - setup_frame.meters.y;
}

// How far the relief of the ground's materials raises the ground `height`
// meters above the terrain's origin at terrain-local `local`, at the level
// whose texels span a cell, `samples` holding the ground there
// (`terrain_ground_rise`), and whether its materials raise it.
fn setup_raised(samples: TerrainGroundSamples, local: vec2<f32>, height: f32)
        -> TerrainGroundRise {
    return terrain_ground_rise(samples, local, height, setup_frame.meters.x);
}

// The level a lake holding a cell at `level` (SETUP_UNHELD where none does)
// holds it at, its raised ground standing at `risen`, the mean of its
// corners as the solve reads it (`water_sim.wgsl`'s `sim_bed`): none where
// that ground stands at or over the level, as a lake holds no cell it does
// not stand over.
fn setup_held(level: f32, risen: f32) -> f32 {
    return select(SETUP_UNHELD, level, level > risen);
}

// The roughness the solve reads of a cell `cell` meters across whose bed's
// roughness's word is `rough` (`engine_terrain::shallow_water::roughness`:
// its D84, negated where steps or knobs are carved into it), `relief` where
// its materials raise its ground: friction counts only the roughness its
// bed does not hold as its shape. Where its bed holds relief as its shape,
// carved or raised, its grains finer than a cell alone, its D84 no more
// than a cell, negated so friction counts its grains alone
// (`water_sim.wgsl`'s `sim_resistance`); elsewhere its D84 as it is.
fn setup_rough(rough: f32, relief: bool, cell: f32) -> f32 {
    if rough < 0.0 || relief {
        return -min(abs(rough), cell);
    }
    return rough;
}

// The cell of a lattice `by` cells of another lattice across holding that
// lattice's cell `at`, both from the terrain's first corner.
fn setup_floor(at: vec2<i32>, by: i32) -> vec2<i32> {
    return (at - select(vec2<i32>(0), vec2<i32>(by - 1), at < vec2<i32>(0))) / by;
}

// The settled cell holding live cell `at`, on the terrain's lattice of
// settled cells.
fn setup_holding(at: vec2<i32>) -> vec2<i32> {
    return setup_floor(at, i32(setup_frame.square.w));
}

// Where the middle of live cell `at` lies in the tiles' water of the finest
// resident level (`water_cell`).
fn setup_cell(at: vec2<i32>) -> WaterCell {
    return water_cell((vec2<f32>(at) + 0.5) * setup_frame.meters.x, 0u);
}

// The words of the tiles' water's sample nearest the point `cell`
// (`setup_cell`'s), as `water_read` takes what names a kind: none, beside
// the ground's roughness outside rivers' channels in the fourth (a float's
// bits), where its patch holds no water.
fn setup_nearest(cell: WaterCell) -> vec4<u32> {
    if cell.block == NO_WATER_BLOCK {
        return vec4<u32>(0u, 0u, 0u, bitcast<u32>(SETUP_GROUND_D84));
    }
    let near = select(vec2<u32>(0u), vec2<u32>(1u), cell.within >= vec2<f32>(0.5));
    let texel = water_texel(cell.block, cell.first + near);
    return textureLoad(water_info, texel.xy, texel.z, 0);
}

// Live cell `at` as its tile's record of settled water keeps it, read from
// the records' live cells (`setup_live`): the level-0 tile holding the
// middle of the settled cell holding it, as records keep their cells, then
// that tile's table's block holding the cell, then the cell. Not resident
// where that tile's record is not held; dry where its table holds no wet
// cell there.
fn setup_live(at: vec2<i32>) -> SetupLive {
    var live = SetupLive(false, NO_WATER_SURFACE, vec2<f32>(0.0), false);
    let spacing = setup_frame.meters.x * f32(setup_frame.square.w);
    let middle = (vec2<f32>(setup_holding(at)) + 0.5) * spacing;
    let size = f32(WATER_TILE_PATCHES * (TERRAIN_GRID - 1u)) * terrain.anchor.w;
    let tiles = vec2<i32>(setup_frame.tiles.zw);
    let tile = vec2<i32>(floor(middle / size)) - bitcast<vec2<i32>>(setup_frame.tiles.xy);
    if any(tile < vec2<i32>(0)) || any(tile >= tiles) {
        return live;
    }
    let table = setup_live[u32(tile.y * tiles.x + tile.x)];
    if table == SETUP_NO_LIVE {
        return live;
    }
    live.resident = true;
    let first = bitcast<vec2<i32>>(vec2<u32>(setup_live[table], setup_live[table + 1u]));
    let blocks = vec2<i32>(vec2<u32>(setup_live[table + 2u], setup_live[table + 3u]));
    let side = i32(WATER_LIVE_BLOCK);
    let block = setup_floor(at, side) - first;
    if any(block < vec2<i32>(0)) || any(block >= blocks) {
        return live;
    }
    let cells = setup_live[table + SETUP_LIVE_HEAD + u32(block.y * blocks.x + block.x)];
    if cells == SETUP_NO_LIVE {
        return live;
    }
    let inside = vec2<u32>(at - (first + block) * side);
    let entry = cells + SETUP_LIVE_CELL * (inside.y * WATER_LIVE_BLOCK + inside.x);
    let surface = bitcast<f32>(setup_live[entry]);
    if surface <= 0.5 * NO_WATER_SURFACE {
        return live;
    }
    let flow = vec2<f32>(bitcast<f32>(setup_live[entry + 1u]),
        bitcast<f32>(setup_live[entry + 2u]));
    live.surface = (surface + terrain.origin.y) - setup_frame.meters.y;
    live.held = flow.x >= 0.5 * SETUP_LIVE_HELD;
    live.flow = select(flow, vec2<f32>(0.0), live.held);
    return live;
}

// The water live cell `at` stands as where the live water does not run
// there, past the square's edges or in a block still waiting for its data:
// as its tile's record keeps it on that cell (`setup_live`) where the
// record's live cells are resident; else as water's shading reads the
// tiles' water at its middle (`water_read`), standing where the samples
// holding a surface carry WATER_HELD of its weights or more and it stands
// over their ground, at its smooth surface, its flow its velocity times
// that depth, and still at that surface where its nearest sample's words
// say a lake holds it. Its surface over the datum (NO_WATER_SURFACE where it
// is dry), its flow along x and z (square meters a second) and the level a
// lake holds it at over the datum (SETUP_UNHELD where none does).
fn setup_ghost(at: vec2<i32>) -> vec4<f32> {
    let live = setup_live(at);
    if live.resident {
        if live.held {
            return vec4<f32>(live.surface, 0.0, 0.0, live.surface);
        }
        return vec4<f32>(live.surface, live.flow, SETUP_UNHELD);
    }
    let shown = water_read(setup_cell(at));
    if shown.held < WATER_HELD || shown.depth <= SETUP_DRY {
        return vec4<f32>(NO_WATER_SURFACE, 0.0, 0.0, SETUP_UNHELD);
    }
    let surface = (shown.surface + terrain.origin.y) - setup_frame.meters.y;
    if shown.words.lake {
        return vec4<f32>(surface, 0.0, 0.0, surface);
    }
    return vec4<f32>(surface, shown.words.flow * shown.depth, SETUP_UNHELD);
}

// The speed (meters a second) of `flow` square meters a second of water
// `depth` meters deep, desingularized as the water thins out as the solve
// takes it (`water_sim.wgsl`'s `sim_velocity`).
fn setup_speed(depth: f32, flow: vec2<f32>) -> f32 {
    let h4 = depth * depth * depth * depth;
    let thin = SETUP_THIN * SETUP_THIN * SETUP_THIN * SETUP_THIN;
    return 1.4142135 * depth * length(flow) / sqrt(h4 + max(h4, thin));
}

// Live water `water` as cell `at` over its ground `bed` holds it, its
// surface and flow along x and z (square meters a second), then nothing:
// still at its level where a lake holds it, which lies below its ground
// where the lake stands under it and the solve dries it there; at its
// ground, dry, where it holds no water over it; else at its surface and
// flow, and where `unsteady`, off that steady state by noise hashed at its
// cell: its velocity by its speed's swing shared by x and z, and its depth
// by as far as a shallow wave carrying that swing heaves its surface (the
// swing times √(h/g)), at most half its depth, its flow carried at its
// velocity over that depth.
fn setup_standing(water: SetupWater, bed: f32, at: vec2<i32>, unsteady: bool) -> vec4<f32> {
    if water.held > 0.5 * SETUP_UNHELD {
        return vec4<f32>(water.held, 0.0, 0.0, 0.0);
    }
    let depth = water.surface - bed;
    if depth <= SETUP_DRY {
        return vec4<f32>(bed, 0.0, 0.0, 0.0);
    }
    if !unsteady {
        return vec4<f32>(water.surface, water.flow, 0.0);
    }
    let heave = water.rush * sqrt(depth / WATER_GRAVITY);
    let deep = depth + clamp(heave * setup_noise(at, 0u), -0.5 * depth, 0.5 * depth);
    let swing = water.rush * inverseSqrt(2.0) * vec2<f32>(setup_noise(at, 1u), setup_noise(at, 2u));
    return vec4<f32>(bed + deep, (deep / depth) * water.flow + deep * swing, 0.0);
}

// A value of zero mean and unit variance hashed from the terrain's lattice
// cell `at` and `channel`: uniform over SETUP_NOISE_REACH either side of
// zero.
fn setup_noise(at: vec2<i32>, channel: u32) -> f32 {
    var h = (bitcast<u32>(at.x) * 0x9e3779b9u) ^ (bitcast<u32>(at.y) * 0x85ebca6bu)
        ^ (channel * 0xc2b2ae35u);
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return ((f32(h >> 8u) + 0.5) / 16777216.0 * 2.0 - 1.0) * SETUP_NOISE_REACH;
}

// Whether solid `solid` of the list reaches live cell `at`: the cell lies,
// along x and z, among those its bounds reach into, from its middle less
// its reach to its middle plus it.
fn setup_reaches(solid: u32, at: vec2<i32>) -> bool {
    let base = SETUP_SOLID * solid;
    let middle = bitcast<vec4<f32>>(setup_list[base]).xz;
    let reach = vec2<f32>(bitcast<f32>(setup_list[base + 1u].w),
        bitcast<f32>(setup_list[base + 2u].w));
    let cell = setup_frame.meters.x;
    let low = vec2<i32>(floor((middle - reach) / cell));
    let high = vec2<i32>(floor((middle + reach) / cell));
    return all(at >= low) && all(at <= high);
}

// The lowest and the highest terrain-local heights solid `solid` of the list
// spans on the upright line through terrain-local `point`, where it stands
// on it; its lowest over its highest where it does not
// (`engine_terrain::Footprint::span`): in its unit cube or ball the line runs
// from `at`, at height 0, by `up` a meter; a ball holds it where
// |at + y·up| ≤ 1, a box where each of its parts lies within 1.
fn setup_span(solid: u32, point: vec2<f32>) -> vec2<f32> {
    let base = SETUP_SOLID * solid;
    let center = bitcast<vec4<f32>>(setup_list[base]);
    let along_x = bitcast<vec4<f32>>(setup_list[base + 1u]).xyz;
    let up = bitcast<vec4<f32>>(setup_list[base + 2u]).xyz;
    let along_z = bitcast<vec4<f32>>(setup_list[base + 3u]).xyz;
    let from_center = vec3<f32>(point.x, 0.0, point.y) - center.xyz;
    let at = along_x * from_center.x + up * from_center.y + along_z * from_center.z;
    let crossed = vec2<f32>(1.0, 0.0);
    if center.w > 0.5 {
        let rise = dot(up, up);
        let along = dot(at, up);
        let reach = along * along - rise * (dot(at, at) - 1.0);
        if !(rise > 0.0 && reach >= 0.0) {
            return crossed;
        }
        let root = sqrt(reach);
        return vec2<f32>(-along - root, -along + root) / rise;
    }
    var span = vec2<f32>(-SETUP_FARTHEST, SETUP_FARTHEST);
    for (var k = 0u; k < 3u; k++) {
        if up[k] != 0.0 {
            let a = (-1.0 - at[k]) / up[k];
            let b = (1.0 - at[k]) / up[k];
            span = vec2<f32>(max(span.x, min(a, b)), min(span.y, max(a, b)));
        } else if abs(at[k]) > 1.0 {
            return crossed;
        }
    }
    return span;
}

// The share of live cell `at` the solids of the list cover, and the height
// their tops stand at over the datum (none each where none covers it): the
// share of its SETUP_COVER_POINTS along each side, a fraction of its
// spacing apart from half that in, that any solid reaching it stands on,
// each standing at the highest top over it, and those tops' mean
// (`engine_terrain::cover_cells`).
fn setup_cover(at: vec2<i32>) -> vec2<f32> {
    let cell = setup_frame.meters.x;
    let points = f32(SETUP_COVER_POINTS);
    let corner = vec2<f32>(at) * cell;
    var tops: array<f32, SETUP_POINTS>;
    for (var k = 0u; k < SETUP_POINTS; k++) {
        tops[k] = -SETUP_FARTHEST;
    }
    for (var solid = 0u; solid < setup_frame.counts.w; solid++) {
        if !setup_reaches(solid, at) {
            continue;
        }
        for (var k = 0u; k < SETUP_POINTS; k++) {
            let place = vec2<f32>(f32(k % SETUP_COVER_POINTS), f32(k / SETUP_COVER_POINTS));
            let span = setup_span(solid, corner + (place + 0.5) * (cell / points));
            if span.x <= span.y {
                tops[k] = max(tops[k], span.y);
            }
        }
    }
    var covered = 0u;
    var sum = 0.0;
    for (var k = 0u; k < SETUP_POINTS; k++) {
        if tops[k] > -SETUP_FARTHEST {
            covered++;
            sum += tops[k];
        }
    }
    if covered == 0u {
        return vec2<f32>(0.0);
    }
    return vec2<f32>(f32(covered) / f32(SETUP_POINTS),
        (terrain.origin.y + sum / f32(covered)) - setup_frame.meters.y);
}

// A block's corners as its workgroup sets them up or raises them: that
// ground raised by the relief of the ground's materials, and whether its
// materials raise it (1 where they do).
var<workgroup> setup_risen: array<f32, SETUP_CORNERS>;
var<workgroup> setup_reliefs: array<u32, SETUP_CORNERS>;

// The corner of a block invocation `local` of its workgroup raises along
// with its own, the block's last column and row taking their far corners
// too: its own (k 0), then past it along x, along z and along both,
// whichever it takes (`setup_corner_taken`).
fn setup_corner(local: vec2<u32>, k: u32) -> vec2<u32> {
    return local + vec2<u32>(k & 1u, k >> 1u);
}

fn setup_corner_taken(local: vec2<u32>, k: u32) -> bool {
    let far = vec2<u32>(k & 1u, k >> 1u);
    return !any((far > vec2<u32>(0u)) & (local < vec2<u32>(WATER_LIVE_BLOCK - 1u)));
}

// The ground raised at the middle of cell `local` of a block, the mean of
// its corners as the solve reads it (`water_sim.wgsl`'s `sim_bed`), and
// whether the materials of any of its corners raise it, as the workgroup
// raised them.
fn setup_cell_risen(local: vec2<u32>) -> vec2<f32> {
    let corner = local.y * SETUP_SIDE + local.x;
    let risen = 0.5 * (0.5 * (setup_risen[corner] + setup_risen[corner + SETUP_SIDE])
        + 0.5 * (setup_risen[corner + 1u] + setup_risen[corner + SETUP_SIDE + 1u]));
    let reliefs = setup_reliefs[corner] | setup_reliefs[corner + 1u]
        | setup_reliefs[corner + SETUP_SIDE] | setup_reliefs[corner + SETUP_SIDE + 1u];
    return vec2<f32>(risen, f32(reliefs));
}

// The blocks a frame sets up, a workgroup each, each invocation a cell. A
// block standing as the settled water does while its data is not resident
// stands each cell as `setup_ghost` finds it, in the step's start, its look
// as water's pass draws it holding no surface and no ground; its ground
// stays as it was, as nothing reads it. A block set up whole or its ground
// alone again: each invocation sets up the ground at the corner its cell
// starts at, those along the block's last column and row their far corners
// too, raised (`setup_raised`) by the samples the tiles' layers hold
// (`terrain_ground_samples`); then its cell's ground: the lake its tile's
// record holds it in holds it but where its raised ground stands at or over
// the lake's level (`setup_held`), its bed's roughness from the word of the
// tiles' water's sample nearest its middle (`setup_rough`) and the solids'
// cover. A block set up whole sets up its cells' water too, each as its
// tile's record keeps it (`setup_live`): what it starts from in the step's
// first state, and its looks.
@compute @workgroup_size(WATER_LIVE_BLOCK, WATER_LIVE_BLOCK)
fn setup_blocks(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let block = group.y * count.x + group.x;
    if block >= setup_frame.counts.x {
        return;
    }
    let listed = setup_list[SETUP_SOLID * setup_frame.counts.w + block];
    let slot = listed.x;
    let first = bitcast<vec2<i32>>(listed.yz);
    let stands = listed.w == SETUP_STANDS;
    let base = slot * SETUP_GROUND;
    if !stands {
        for (var k = 0u; k < 4u; k++) {
            if !setup_corner_taken(local.xy, k) {
                continue;
            }
            let corner = setup_corner(local.xy, k);
            let lattice = first + vec2<i32>(corner);
            let point = vec2<f32>(lattice) * setup_frame.meters.x;
            let bare = setup_ground_at(lattice);
            let height = (setup_frame.meters.y + bare) - terrain.origin.y;
            let raised = setup_raised(terrain_ground_samples(point), point, height);
            let at = corner.y * SETUP_SIDE + corner.x;
            setup_ground[base + at] = bare + raised.rise;
            setup_risen[at] = bare + raised.rise;
            setup_reliefs[at] = select(0u, 1u, raised.relief);
        }
    }
    workgroupBarrier();
    let at = first + vec2<i32>(local.xy);
    let index = local.y * WATER_LIVE_BLOCK + local.x;
    let held = slot * SETUP_CELLS + index;
    if stands {
        setup_cells[held] = setup_ghost(at);
        setup_looks[2u * held + 1u] = vec4<f32>(NO_WATER_SURFACE, NO_WATER_SURFACE, 0.0, 0.0);
        return;
    }
    let risen = setup_cell_risen(local.xy);
    let live = setup_live(at);
    let lake = setup_held(select(SETUP_UNHELD, live.surface, live.held), risen.x);
    let nearest = setup_nearest(setup_cell(at));
    let cover = setup_cover(at);
    let ground = base + SETUP_CORNERS + index;
    setup_ground[ground] = lake;
    setup_ground[ground + SETUP_CELLS] = setup_rough(bitcast<f32>(nearest.w), risen.y > 0.0,
        setup_frame.meters.x);
    setup_ground[ground + 2u * SETUP_CELLS] = cover.x;
    setup_ground[ground + 3u * SETUP_CELLS] = cover.y;
    if listed.w != SETUP_WHOLE {
        return;
    }
    // How far its speed swings about its mean: its turbulence's share of
    // its speed, as the cook counts it (`engine_terrain`'s `turbulence`: the
    // speed counting as WATER_TURBULENT_SPEED at least).
    let speed = setup_speed(live.surface - risen.x, live.flow);
    let rush = unpack4x8unorm(nearest.y).x * max(speed, WATER_TURBULENT_SPEED);
    let water = SetupWater(live.surface, live.flow, lake, rush);
    setup_cells[held] = setup_standing(water, risen.x, at, true);
    setup_looks[2u * held] = setup_standing(water, risen.x, at, false);
    setup_looks[2u * held + 1u] = vec4<f32>(0.0, 0.0, bitcast<f32>(nearest.x & 0xff000000u),
        bitcast<f32>(nearest.y & 0xff000000u));
}

// Sample `at` of a settling block's samples (`setup_raise`'s, from the
// list's entry `window`, `setup_frame.square.x` along each side, counted
// from their first), the nearest of them past their edges.
fn setup_sample(window: u32, at: vec2<i32>) -> vec4<u32> {
    let side = i32(setup_frame.square.x);
    let held = clamp(at, vec2<i32>(0), vec2<i32>(side - 1));
    return setup_list[window + u32(held.y * side + held.x)];
}

// The ground's samples around terrain-local `local` (`TerrainGroundSamples`)
// as a settling block's samples hold them (from the list's entry `window`,
// the first of them `start` on the terrain's lattice of level-0 samples):
// the cell of samples holding the point and where in it the point lies, as
// the tiles' layers find them (`terrain_level_own`'s tile and corner), its
// last cell the terrain's last before the sample `setup_frame.square.yz`.
fn setup_window(window: u32, start: vec2<i32>, local: vec2<f32>) -> TerrainGroundSamples {
    let at = max(local / terrain.anchor.w, vec2<f32>(0.0));
    let last = vec2<f32>(setup_frame.square.yz);
    let cell = min(floor(at), last - 1.0);
    let first = vec2<i32>(cell) - start;
    // In a gather's order: x (0, 1), y (1, 1), z (1, 0), w (0, 0).
    let x = setup_sample(window, first + vec2<i32>(0, 1));
    let y = setup_sample(window, first + vec2<i32>(1, 1));
    let z = setup_sample(window, first + vec2<i32>(1, 0));
    let w = setup_sample(window, first);
    return TerrainGroundSamples(vec4<u32>(x.x, y.x, z.x, w.x), vec4<u32>(x.y, y.y, z.y, w.y),
        min(at, last) - cell);
}

// Settling water's blocks as their regions load, a workgroup each, before
// their first steps (`water_settle.rs`): each invocation raises the ground
// at the corner its cell starts at, those along the block's last column and
// row their far corners too, as the live water's setup raises its own
// (`setup_raised`), from the level-0 samples the block's list holds
// (`setup_window`), the samples the tiles' layers hold; then its cell's lake
// lets go of it where its raised ground stands at or over its level
// (`setup_held`), and its bed's roughness is the solve's from the word of
// the sample nearest its middle (`setup_rough`). Its ground stands bare
// before, as the CPU wrote it, and raised after: it is raised once.
@compute @workgroup_size(WATER_LIVE_BLOCK, WATER_LIVE_BLOCK)
fn setup_raise(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let block = group.y * count.x + group.x;
    if block >= setup_frame.counts.x {
        return;
    }
    let head = setup_list[2u * block];
    let base = head.x * SETUP_GROUND;
    let first = bitcast<vec2<i32>>(head.yz);
    let window = head.w;
    let start = bitcast<vec2<i32>>(setup_list[2u * block + 1u].xy);
    let cell = setup_frame.meters.x;
    for (var k = 0u; k < 4u; k++) {
        if !setup_corner_taken(local.xy, k) {
            continue;
        }
        let corner = setup_corner(local.xy, k);
        let point = vec2<f32>(first + vec2<i32>(corner)) * cell;
        let at = corner.y * SETUP_SIDE + corner.x;
        let bare = setup_ground[base + at];
        let height = setup_frame.meters.y + bare;
        let raised = setup_raised(setup_window(window, start, point), point, height);
        setup_ground[base + at] = bare + raised.rise;
        setup_risen[at] = bare + raised.rise;
        setup_reliefs[at] = select(0u, 1u, raised.relief);
    }
    workgroupBarrier();
    let risen = setup_cell_risen(local.xy);
    let ground = base + SETUP_CORNERS + local.y * WATER_LIVE_BLOCK + local.x;
    setup_ground[ground] = setup_held(setup_ground[ground], risen.x);
    let middle = (vec2<f32>(first + vec2<i32>(local.xy)) + 0.5) * cell / terrain.anchor.w;
    let nearest = setup_sample(window, vec2<i32>(floor(middle + 0.5)) - start);
    setup_ground[ground + SETUP_CELLS] = setup_rough(bitcast<f32>(nearest.z), risen.y > 0.0, cell);
}

// The cells just past the live square's edges, an invocation each, in the
// order the solve reads them (`water_sim.wgsl`'s `sim_ring_flux`): south
// then north of each column, then west then east of each row. Each stands
// as the settled water does at it (`setup_ghost`): held at a lake's level,
// and still; dry, far below any ground (NO_WATER_SURFACE); or flowing at
// its flow, what of it runs into the square across the face it borders the
// flow it brings in.
@compute @workgroup_size(SETUP_RING_GROUP)
fn setup_ring(@builtin(global_invocation_id) id: vec3<u32>) {
    let ghost = id.x;
    if ghost >= setup_frame.counts.z {
        return;
    }
    let n = setup_frame.square.z;
    let first = bitcast<vec2<i32>>(setup_frame.square.xy);
    let far = ghost % 2u == 1u;
    let outside = select(-1, i32(n), far);
    let inward = select(1.0, -1.0, far);
    var at = first;
    var toward = vec2<f32>(0.0, inward);
    if ghost < 2u * n {
        at += vec2<i32>(i32(ghost / 2u), outside);
    } else {
        at += vec2<i32>(outside, i32((ghost - 2u * n) / 2u));
        toward = vec2<f32>(inward, 0.0);
    }
    let standing = setup_ghost(at);
    var carried = vec4<f32>(0.0);
    if standing.w <= 0.5 * SETUP_UNHELD && standing.x > 0.5 * NO_WATER_SURFACE {
        carried.x = max(dot(standing.yz, toward), 0.0);
    }
    let entry = setup_frame.ring.x + 2u * ghost;
    setup_edges[entry] = standing;
    setup_edges[entry + 1u] = carried;
}
