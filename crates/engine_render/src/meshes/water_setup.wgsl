// The water's beds and the live water's setup: settling water's blocks'
// ground raised as their regions load (`setup_raise`, `water_settle.rs`),
// and every block of the live square a frame newly holds, or whose tiles
// changed under it, set up from what the GPU holds of the terrain, a
// workgroup a block (`setup_blocks`), and the cells just past the square's
// edges, an invocation each (`setup_ring`) (`water_sim.rs`); each after an
// empty first group and the terrain group of the terrain the water stands
// on, through which it reads its materials as the ground's mesh band reads
// them and, live, the ground as its triangles hold it and the tiles' water
// as water's shading reads it; nothing comes from the CPU but where the
// water stands, the solids standing in it and what its rivers bring, and,
// settling, its bare ground and the level-0 samples it rises by.
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
// Each live cell's water is the settled water as water's shading reads the
// tiles' water at its middle (`water_read`: the smooth surface through the
// samples of the finest resident level holding one, its flow and turbulence
// through the same weights), standing over its own ground, matched to its
// settled cell, the settled water's cell holding it, which a block holds
// whole, as the samples around that cell's middle hold it between them
// (`setup_settled`): where a lake holds that cell, its live cells stand
// still at the lake's level, and the lake holds each but where its raised
// ground stands at or over the level; where it flows, those of its live
// cells holding water are raised or lowered alike and their velocity turned
// alike, so their mean surface and velocity are its own, or, where none
// would hold water, each stands at its surface and velocity; and where it
// is dry, each stands as it reads. So over each settled cell the live water
// is that cell's water, the mean of the water settling ran over its cells
// on the same bed, and the same water as the settled water around the
// square. It starts off that steady state by noise of zero mean and unit
// variance hashed at its cell: its velocity by as far as its turbulence
// says its speed swings about its mean, shared by x and z, and its surface
// by as far as a shallow wave carrying that swing heaves it, at most half
// its depth so it stays wet. The steady state is the first entry of its
// look, which the solve pulls its rim back to and measures its swings from,
// and its bed's codes stand in the top bytes of the second's words. A
// block whose tiles' settled water is not resident yet reads the coarser
// level shading reads there, and is set up again as the finest arrives.
//
// The cells just past the square's edges stand as the settled water does at
// each, matched to its settled cell over its own ground as a cell inside
// would be: held at a lake's level and still; dry, far below any ground;
// or flowing, its flow its depth over the ground at the face it borders,
// the mean of that face's corners, times its velocity, bringing the square
// what of it runs inward across that face.

// What a frame sets up. The live water's: the blocks it sets up (those set
// up whole first, then those whose ground alone is set up again), those set
// up whole, the cells past the square's edges it sets up (four times its
// cells along a side; none where it sets up none) and the solids standing
// in the settled water over the square; the square's first cell on the
// terrain's lattice of its cells along x and z (an i32's bits each), its
// cells along a side and the live cells along each side of a settled cell;
// the meters across a live cell and the height its heights stand above
// (world meters), then nothing; and where its ring starts among the
// simulation's inputs, then nothing. Settling water's (`setup_raise`): the
// blocks whose ground it raises, then nothing; the level-0 samples along
// each side of a block's samples and the terrain's last sample along x and
// z, then nothing; the meters across its cells and the height its heights
// stand above (terrain-local meters), then nothing; and nothing.
struct SimSetup {
    counts: vec4<u32>,
    square: vec4<u32>,
    meters: vec4<f32>,
    ring: vec4<u32>,
}

@group(2) @binding(41) var<uniform> setup_frame: SimSetup;
// The live water's: the solids standing in the settled water over the
// square, SETUP_SOLID entries each (a float's bits): its center
// (terrain-local) and 1 for a ball, else 0; then the inverse of the axes
// mapping the unit cube (or ball) to it, a column each, the first two beside
// how far it reaches from its center along x and along z. Then each block
// the frame sets up: its slot in the live water's pool and the cell of the
// terrain's lattice of live cells its first corner stands at (an i32's bits
// along x and z). Settling water's (`setup_raise`): each block whose ground
// it raises, two entries: its block among the ground's (its slot's first
// block's index plus its own in its pool), the cell of the terrain's
// lattice of cells its first corner stands at (an i32's bits along x and z)
// and the entry its samples start at; then the first of them on the
// terrain's lattice of level-0 samples (an i32's bits along x and z), then
// nothing. And each block's samples, `setup_frame.square.x` along each side,
// row by row: each one's control value, its smooth surface's gradient as a
// normal map's texel holds it and the bits of its bed's roughness's word
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
// nothing.
@group(2) @binding(44) var<storage, read_write> setup_cells: array<vec4<f32>>;
// The live water's cells' looks (`sim_averages`), two a cell: its settled
// water, then its water as water's pass draws it.
@group(2) @binding(45) var<storage, read_write> setup_looks: array<vec4<f32>>;
// The live water's inputs (`sim_inputs`), whose ring it writes from
// `setup_frame.ring.x`, two entries a cell past the edges: its surface,
// flow along x and z and lake's level, then the flow it brings in, then
// nothing.
@group(2) @binding(46) var<storage, read_write> setup_edges: array<vec4<f32>>;

// Entries of a block of the simulation's ground, what a cell no lake holds
// holds as its lake's level, the meters of water under which a cell is dry,
// the roughness of ground outside rivers' channels and the points along
// each side of a cell its cover is measured at (`engine_terrain`'s
// `BLOCK_GROUND`, `shallow_water::UNHELD`, `shallow_water::DRY`,
// `shallow_water::GROUND_D84`, `COVER_POINTS`); and the cells past the
// edges a workgroup of the ring's setup takes, and the entries of its list
// a solid takes (`gpu_layouts::constants`).
const SETUP_GROUND: u32 = 337u;
const SETUP_UNHELD: f32 = -1.0e30;
const SETUP_DRY: f32 = 1.0e-4;
const SETUP_GROUND_D84: f32 = 0.05;
const SETUP_COVER_POINTS: u32 = 4u;
const SETUP_RING_GROUP: u32 = 64u;
const SETUP_SOLID: u32 = 4u;
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
// What a settled cell holds: no water, a lake's water, or flowing water.
const SETUP_DRY_CELL: u32 = 0u;
const SETUP_HELD: u32 = 1u;
const SETUP_FLOWING: u32 = 2u;

// A live cell's settled water as the setup stands it: the surface it stands
// at over the datum (NO_WATER_SURFACE where none), its velocity along x and
// z (meters a second), the level a lake holds it at over the datum
// (SETUP_UNHELD where none does), and how far its speed swings about its
// mean (meters a second).
struct SetupWater {
    surface: f32,
    velocity: vec2<f32>,
    held: f32,
    rush: f32,
}

// A settled cell as the samples around its middle hold it: what it holds
// (SETUP_DRY_CELL, SETUP_HELD or SETUP_FLOWING), the surface its water
// stands at over the datum and its velocity (meters a second).
struct SetupSettled {
    kind: u32,
    surface: f32,
    velocity: vec2<f32>,
}

// The ground over the datum at corner `corner` of the terrain's lattice of
// live cells, on the finest resident level's triangles (the terrain's
// origin where no level holds it).
fn setup_ground_at(corner: vec2<i32>) -> f32 {
    let local = vec2<f32>(corner) * setup_frame.meters.x;
    return (terrain.origin.y + terrain_level_height(local, 0u).height) - setup_frame.meters.y;
}

// The ground at the middle of live cell `at`: its corners' mean.
fn setup_bed(at: vec2<i32>) -> f32 {
    return 0.25 * (setup_ground_at(at) + setup_ground_at(at + vec2<i32>(1, 0))
        + setup_ground_at(at + vec2<i32>(0, 1)) + setup_ground_at(at + vec2<i32>(1, 1)));
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

// The settled cell holding live cell `at`, on the terrain's lattice of
// settled cells.
fn setup_holding(at: vec2<i32>) -> vec2<i32> {
    let split = vec2<i32>(i32(setup_frame.square.w));
    return (at - select(vec2<i32>(0), split - vec2<i32>(1), at < vec2<i32>(0))) / split;
}

// Where the middle of live cell `at` lies in the tiles' water of the finest
// resident level (`water_cell`).
fn setup_cell(at: vec2<i32>) -> WaterCell {
    return water_cell((vec2<f32>(at) + 0.5) * setup_frame.meters.x, 0u);
}

// The settled water at the point `cell` (`setup_cell`'s) as water's shading
// reads it (`water_read`): its smooth surface over the datum, none where its
// samples hold none; its velocity through the same weights; held by no
// lake; and its speed's swing about its mean, its turbulence's share of its
// speed, as the cook counts it (`engine_terrain`'s `turbulence`: the speed
// counting as WATER_TURBULENT_SPEED at least).
fn setup_read(cell: WaterCell) -> SetupWater {
    let at = water_read(cell);
    var water = SetupWater(NO_WATER_SURFACE, vec2<f32>(0.0), SETUP_UNHELD, 0.0);
    if at.held <= 0.0 {
        return water;
    }
    water.surface = (at.surface + terrain.origin.y) - setup_frame.meters.y;
    water.velocity = at.words.flow;
    water.rush = at.words.state.x * max(length(at.words.flow), WATER_TURBULENT_SPEED);
    return water;
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

// Settled cell `holding` of the terrain's lattice of settled cells as the
// tiles' water's four samples of the finest resident level around its
// middle hold it between them, bilinearly: through the weights of those
// whose surface stands over their ground, its surface, and its velocity
// through those no lake holds; held by a lake where lakes hold most of
// those weights, as tiles' water blends settled cells
// (`engine_terrain::Tile::steadied`); dry where none holds water.
fn setup_settled(holding: vec2<i32>) -> SetupSettled {
    let spacing = setup_frame.meters.x * f32(setup_frame.square.w);
    let cell = water_cell((vec2<f32>(holding) + 0.5) * spacing, 0u);
    var settled = SetupSettled(SETUP_DRY_CELL, 0.0, vec2<f32>(0.0));
    if cell.block == NO_WATER_BLOCK {
        return settled;
    }
    var weight = 0.0;
    var still = 0.0;
    var moving = 0.0;
    var surface = 0.0;
    var flow = vec2<f32>(0.0);
    for (var k = 0u; k < 4u; k++) {
        let step = vec2<u32>(k % 2u, k / 2u);
        let shares = mix(vec2<f32>(1.0) - cell.within, cell.within, vec2<f32>(step));
        let share = shares.x * shares.y;
        let texel = water_texel(cell.block, cell.first + step);
        let held = textureLoad(water_surface, texel.xy, texel.z, 0).xy;
        if share <= 0.0 || held.x <= held.y {
            continue;
        }
        let words = water_words(textureLoad(water_info, texel.xy, texel.z, 0));
        weight += share;
        surface += share * held.x;
        if words.lake {
            still += share;
        } else {
            moving += share;
            flow += share * words.flow;
        }
    }
    if weight <= 0.0 {
        return settled;
    }
    settled.surface = (surface / weight + terrain.origin.y) - setup_frame.meters.y;
    if 2.0 * still > weight {
        settled.kind = SETUP_HELD;
        return settled;
    }
    settled.kind = SETUP_FLOWING;
    if moving > 0.0 {
        settled.velocity = flow / moving;
    }
    return settled;
}

// Whether live water `water` stands over its ground `bed`.
fn setup_wet(water: SetupWater, bed: f32) -> bool {
    return water.surface > 0.5 * NO_WATER_SURFACE && water.surface - bed > SETUP_DRY;
}

// Live water `own` over its ground `bed`, matched to its settled cell
// `settled` (`setup_settled`'s), whose live cells holding water sum to
// `sums` (how many, then their surfaces and their velocities along x and
// z): still at the lake's level where a lake holds that cell; raised or
// lowered and its velocity turned as all of them are, so their mean is the
// cell's, where it flows and it holds water, or at the cell's surface and
// velocity where none of them does; as it is elsewhere.
fn setup_matched(own: SetupWater, bed: f32, settled: SetupSettled, sums: vec4<f32>)
        -> SetupWater {
    var water = own;
    if settled.kind == SETUP_HELD {
        return SetupWater(settled.surface, vec2<f32>(0.0), settled.surface, 0.0);
    }
    if settled.kind != SETUP_FLOWING {
        return water;
    }
    if sums.x <= 0.0 {
        water.surface = settled.surface;
        water.velocity = settled.velocity;
        return water;
    }
    if setup_wet(own, bed) {
        let share = 1.0 / sums.x;
        water.surface += settled.surface - sums.y * share;
        water.velocity += settled.velocity - sums.zw * share;
    }
    return water;
}

// Live water `water` as cell `at` over its ground `bed` holds it, its
// surface and flow along x and z (square meters a second), then nothing:
// still at its level where a lake holds it, which lies below its ground
// where the lake stands under it and the solve dries it there; at its
// ground, dry, where it holds no water; else its depth over its ground at
// its velocity, and where `unsteady`, off its steady state by noise hashed
// at its cell: its velocity by its speed's swing shared by x and z, and its
// depth by as far as a shallow wave carrying that swing heaves its surface
// (the swing times √(h/g)), at most half its depth.
fn setup_standing(water: SetupWater, bed: f32, at: vec2<i32>, unsteady: bool) -> vec4<f32> {
    if water.held > 0.5 * SETUP_UNHELD {
        return vec4<f32>(water.held, 0.0, 0.0, 0.0);
    }
    if !setup_wet(water, bed) {
        return vec4<f32>(bed, 0.0, 0.0, 0.0);
    }
    let depth = water.surface - bed;
    var deep = depth;
    var velocity = water.velocity;
    if unsteady {
        let heave = water.rush * sqrt(depth / WATER_GRAVITY);
        deep += clamp(heave * setup_noise(at, 0u), -0.5 * depth, 0.5 * depth);
        velocity += water.rush * inverseSqrt(2.0)
            * vec2<f32>(setup_noise(at, 1u), setup_noise(at, 2u));
    }
    return vec4<f32>(bed + deep, deep * velocity, 0.0);
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

// Live cell `at`'s water standing alone, its settled cell's other live cells
// read again: as it reads (`setup_read`), over its ground, matched to its
// settled cell (`setup_matched`).
fn setup_lone(at: vec2<i32>) -> SetupWater {
    let holding = setup_holding(at);
    let settled = setup_settled(holding);
    let own = setup_read(setup_cell(at));
    var sums = vec4<f32>(0.0);
    if settled.kind == SETUP_FLOWING {
        let split = i32(setup_frame.square.w);
        for (var z = 0; z < split; z++) {
            for (var x = 0; x < split; x++) {
                let cell = holding * split + vec2<i32>(x, z);
                let water = setup_read(setup_cell(cell));
                if setup_wet(water, setup_bed(cell)) {
                    sums += vec4<f32>(1.0, water.surface, water.velocity);
                }
            }
        }
    }
    return setup_matched(own, setup_bed(at), settled, sums);
}

// A block's corners as its workgroup sets them up or raises them, their
// ground, that ground raised by the relief of the ground's materials and
// whether its materials raise it (1 where they do); and each of its cells'
// water as it reads: whether it holds water over its ground, then its
// surface and velocity along x and z.
var<workgroup> setup_bare: array<f32, SETUP_CORNERS>;
var<workgroup> setup_risen: array<f32, SETUP_CORNERS>;
var<workgroup> setup_reliefs: array<u32, SETUP_CORNERS>;
var<workgroup> setup_waters: array<vec4<f32>, SETUP_CELLS>;

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

// The blocks a frame sets up, a workgroup each: each invocation sets up the
// ground at the corner its cell starts at, those along the block's last
// column and row their far corners too, raised (`setup_raised`) by the
// samples the tiles' layers hold (`terrain_ground_samples`); then its
// cell's ground: the lake holding its settled cell holds it but where its
// raised ground stands at or over the lake's level (`setup_held`), its bed's
// roughness from the word of the tiles' water's sample nearest its middle
// (`setup_rough`) and the solids' cover. A block set up whole sets up its
// cells' water too, each matched over its settled cell, which the block
// holds whole: what it starts from in the step's first state, and its looks.
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
    let whole = block < setup_frame.counts.y;
    let listed = setup_list[SETUP_SOLID * setup_frame.counts.w + block];
    let slot = listed.x;
    let first = bitcast<vec2<i32>>(listed.yz);
    let base = slot * SETUP_GROUND;
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
        setup_bare[at] = bare;
        setup_risen[at] = bare + raised.rise;
        setup_reliefs[at] = select(0u, 1u, raised.relief);
    }
    let at = first + vec2<i32>(local.xy);
    let cell = setup_cell(at);
    var own = SetupWater(NO_WATER_SURFACE, vec2<f32>(0.0), SETUP_UNHELD, 0.0);
    if whole {
        own = setup_read(cell);
    }
    workgroupBarrier();
    let corner = local.y * SETUP_SIDE + local.x;
    let bed = 0.25 * (setup_bare[corner] + setup_bare[corner + 1u]
        + setup_bare[corner + SETUP_SIDE] + setup_bare[corner + SETUP_SIDE + 1u]);
    let risen = setup_cell_risen(local.xy);
    let index = local.y * WATER_LIVE_BLOCK + local.x;
    setup_waters[index] = select(vec4<f32>(0.0), vec4<f32>(1.0, own.surface, own.velocity),
        setup_wet(own, bed));
    workgroupBarrier();
    let settled = setup_settled(setup_holding(at));
    let level = select(SETUP_UNHELD, settled.surface, settled.kind == SETUP_HELD);
    let nearest = setup_nearest(cell);
    let cover = setup_cover(at);
    let ground = base + SETUP_CORNERS + index;
    setup_ground[ground] = setup_held(level, risen.x);
    setup_ground[ground + SETUP_CELLS] = setup_rough(bitcast<f32>(nearest.w), risen.y > 0.0,
        setup_frame.meters.x);
    setup_ground[ground + 2u * SETUP_CELLS] = cover.x;
    setup_ground[ground + 3u * SETUP_CELLS] = cover.y;
    if !whole {
        return;
    }
    // Its settled cell's live cells holding water: how many, and their
    // surfaces and velocities summed.
    let split = setup_frame.square.w;
    let lowest = local.xy / split * split;
    var sums = vec4<f32>(0.0);
    for (var z = 0u; z < split; z++) {
        for (var x = 0u; x < split; x++) {
            let within = lowest + vec2<u32>(x, z);
            sums += setup_waters[within.y * WATER_LIVE_BLOCK + within.x];
        }
    }
    let water = setup_matched(own, bed, settled, sums);
    let held = slot * SETUP_CELLS + index;
    setup_cells[held] = setup_standing(water, bed, at, true);
    setup_looks[2u * held] = setup_standing(water, bed, at, false);
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
// as the settled water does at it, matched to its settled cell over its own
// ground (`setup_lone`), beside the face it borders, whose ground is the
// mean of the bare ground at that face's corners: held at a lake's level,
// and still; dry, far below any ground (NO_WATER_SURFACE); or flowing, its
// flow its depth over the face's ground times its velocity, and what of it
// runs into the square across the face, the flow it brings in.
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
    let edge = select(0, i32(n), far);
    let inward = select(1.0, -1.0, far);
    var at = first;
    var face = first;
    var along = vec2<i32>(1, 0);
    var toward = vec2<f32>(0.0, inward);
    if ghost < 2u * n {
        let k = i32(ghost / 2u);
        at += vec2<i32>(k, outside);
        face += vec2<i32>(k, edge);
    } else {
        let k = i32((ghost - 2u * n) / 2u);
        at += vec2<i32>(outside, k);
        face += vec2<i32>(edge, k);
        along = vec2<i32>(0, 1);
        toward = vec2<f32>(inward, 0.0);
    }
    let water = setup_lone(at);
    let ground = 0.5 * (setup_ground_at(face) + setup_ground_at(face + along));
    var standing = vec4<f32>(NO_WATER_SURFACE, 0.0, 0.0, SETUP_UNHELD);
    var carried = vec4<f32>(0.0);
    if water.held > 0.5 * SETUP_UNHELD {
        standing = vec4<f32>(water.held, 0.0, 0.0, water.held);
    } else if water.surface > 0.5 * NO_WATER_SURFACE {
        let flow = water.velocity * max(water.surface - ground, 0.0);
        standing = vec4<f32>(water.surface, flow, SETUP_UNHELD);
        carried.x = max(dot(flow, toward), 0.0);
    }
    let entry = setup_frame.ring.x + 2u * ghost;
    setup_edges[entry] = standing;
    setup_edges[entry + 1u] = carried;
}
