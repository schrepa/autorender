// Water's simulation: rivers running over rectangles of ground as the
// shallow-water equations say, on grids of cells, by Kurganov and Petrova's
// well-balanced, positivity-preserving central-upwind scheme (2007): second
// order in space (each cell's water reconstructed linearly, its slopes
// limited by a generalized minmod and kept from dipping below the ground at
// its faces, a partially wet cell's lying flat against its waterline as
// Bollermann et al. reconstruct it, 2013) and in time (Heun's two stages),
// Kurganov and Lin's anti-diffusion (2007) keeping shear from smearing,
// friction from each cell's bed (Ferguson's variable-power law, 2007, or
// its grains alone where steps or knobs are carved into it) solved exactly
// in each implicit step, gravity along the bed's slope, each cell's outflow
// kept to the water it holds within a stage, and velocities desingularized
// where the water thins out (`engine_terrain::shallow_water` holds the
// constants these mirror). Each cell holds its water's surface, its flow
// along x and z (square meters a second) and its foam. The ground stands at
// the cells' corners, with the level a lake holds each cell at, if one
// does, each cell's bed's roughness, and the share of each covered by the
// solids standing in it with the height their tops stand at: lakes stand
// still, taking the water that runs into them and giving none, and solids
// block the flow through a cell's faces by the share of the cell they
// cover, as deep as they stand in its water (`sim_opening`), so it turns
// around them. A stage finds the flux through every face, each cell
// reconstructing its water at its own faces and at its neighbors' beside
// them (computing them twice costs less than carrying them through memory),
// then the share of its outflow each cell can give, and updates every cell;
// rivers bring their water to their cells after each step. The water
// carries foam: it churns it up where it loses energy fast enough to break
// (as it runs, where it plunges and jumps, along its eddy lines and where
// it piles up), the current its surface runs at carries it off conserved,
// turned outward in bends and dragged by the wind, so it gathers where that
// surface converges, along the banks it is pushed against and on eddy
// lines, and it fades as its bubbles break.
//
// Every dispatch runs rectangles side by side, the third dimension counting
// them, each a slot of the buffers with its own cells along x and z, cell
// size, steps and edges (`SimSlot`). A slot stores its rectangle's cells by
// blocks of a workgroup's (`engine_terrain::StoredBlocks`), only the blocks
// its water can reach, in a pool its table finds each block in, and a
// dispatch runs a workgroup for each block of the pool. The edges of the
// blocks it stores are open, as a settling rectangle's edges are: water
// running over them toward a block it does not store leaves, and none
// comes back.
// - the square around the camera (`water_sim.rs`) stores every block, each
//   in the slot of its place on the terrain's lattice of blocks, and steps
//   every cell by the frame's seconds, and over its edges water flows to
//   and from the cells just past them, which stand as the water settled
//   there (the ring), and comes in as the water settled there brings it:
//   water left alone stays as it settled. Before each step the solids
//   standing in its water push it (`sim_touch`): toward their own velocity
//   where they fill its depth, so it flows around what stands still and is
//   pushed ahead of what moves, its wakes, bow waves and rings coming out
//   of the shallow-water equations themselves, and churning its foam where
//   they pass it fast enough to tear air in, the foam it carries piling
//   against what stands up through its surface; and the scene's wind drags
//   its surface;
// - regions settling tiles' water (`water_settle.rs`) step each cell by as
//   long as its own and its neighbors' fastest waves allow, which reaches
//   the steady state in far fewer steps; their water leaves freely over
//   their edges and none comes in; each cell's water is summed over the
//   steps since its region's last look, each step weighted by its seconds,
//   with what it loses, how it varies and how it moves, whose means and
//   spreads a look keeps and sums, block by block, for its region to tell
//   whether they still change; and only the blocks water can reach this
//   step step: a block steps where it held water no lake holds at the step
//   before's end, or where a block beside it held water along the side they
//   share, as water crosses less than a cell a step, and every other block
//   stands still, dry or a lake's, the faces between them and those
//   stepping carrying nothing; and no wind drags their surface, as their
//   records stand for whatever wind blows.

const WATER_SIM_GROUP: u32 = 8u;
const WATER_SIM_SOURCE_GROUP: u32 = 64u;
// The most rectangles a dispatch runs.
const SIM_SLOTS: u32 = 64u;
// What a slot's table holds for a block its pool does not store; a block's
// cells and corners; and the entries of a block's ground: the ground at its
// cells' corners, then the level a lake holds each of its cells at, each of
// its cells' bed's roughness, and the share of each of its cells covered by
// solids and the height their tops stand at.
const SIM_NO_BLOCK: u32 = 0xffffffffu;
const SIM_BLOCK_CELLS: u32 = WATER_SIM_GROUP * WATER_SIM_GROUP;
const SIM_BLOCK_CORNERS: u32 = (WATER_SIM_GROUP + 1u) * (WATER_SIM_GROUP + 1u);
const SIM_BLOCK_GROUND: u32 = 337u;
// Gravity (meters a second squared) and water's density (kilograms a cubic
// meter).
const SIM_GRAVITY: f32 = 9.81;
const SIM_DENSITY: f32 = 1000.0;
// The generalized minmod's theta; the depth below which velocities are
// desingularized and the depth under which a cell is dry (meters); and the
// fastest water runs (meters a second), a safety net the sums count.
const SIM_THETA: f32 = 1.3;
const SIM_THIN: f32 = 0.01;
const SIM_DRY: f32 = 1.0e-4;
const SIM_FASTEST: f32 = 20.0;
// The share of the time a cell must be wet for its settled water to stand
// in it (`engine_terrain::shallow_water::WET_MOST`).
const SIM_WET_MOST: f32 = 0.5;
// The bed's friction: Ferguson's law's a1 and a2, the least and the most
// the water's depth over its bed's D84 counts as, and what Manning's n of a
// bed's grains alone divides its D84 to the power 1/6 by.
const SIM_FERGUSON_A1: f32 = 6.5;
const SIM_FERGUSON_A2: f32 = 2.5;
const SIM_SHALLOWEST: f32 = 0.1;
const SIM_DEEPEST: f32 = 30.0;
const SIM_STRICKLER: f32 = 20.4;
// The meters a second the water rivers bring runs as it comes, and what a
// cell no lake holds holds as its lake's level.
const SIM_SOURCE_SPEED: f32 = 1.5;
const SIM_UNHELD: f32 = -1.0e30;
// Local steps: the share of a cell the fastest wave crosses in one, the
// share of that they take, and the slowest wave they count (meters a
// second).
const SIM_COURANT: f32 = 0.24;
const SIM_LOCAL_SHARE: f32 = 0.9;
const SIM_SLOWEST: f32 = 1.0;
// The entries a cell's sums take, its means, and a block's sums at a look.
const SIM_CELL_SUMS: u32 = 4u;
const SIM_CELL_MEANS: u32 = 3u;
const SIM_BLOCK_LOOKS: u32 = 3u;
// A rectangle's flags: it steps; each cell by its own seconds; water
// leaves over its edges freely and none comes in (else the ring stands
// past them); it sums each cell's water over its steps; this dispatch
// looks at its sums; and only its blocks water can reach step.
const SIM_STEPPING: u32 = 1u;
const SIM_LOCAL: u32 = 2u;
const SIM_OPEN: u32 = 4u;
const SIM_SUMMING: u32 = 8u;
const SIM_LOOKING: u32 = 16u;
const SIM_SPARSE: u32 = 32u;

// The stage (0: the first, 1: the second), the slots the buffers hold, the
// blocks a slot's pool holds and the blocks a slot's table maps; the most
// cells rivers bring water to a slot holds, and the faces along its west
// and south edges a slot holds (its cells along x and z); and the free wind
// over the water (meters a second along x and z; none for settling
// regions), then nothing.
struct SimFrame {
    stage: vec4<u32>,
    inputs: vec4<u32>,
    wind: vec4<f32>,
}

// A slot's rectangle: its cells along x and z, its flags and the cells
// its rivers bring water to; the meters across a cell and the seconds of a
// step stepping every cell alike, then nothing; and its blocks along x and
// z and those its pool stores.
struct SimSlot {
    shape: vec4<u32>,
    step: vec4<f32>,
    blocks: vec4<u32>,
}

struct SimSlots {
    slot: array<SimSlot, SIM_SLOTS>,
}

@group(0) @binding(0) var<uniform> frame: SimFrame;
// Each slot's pool's blocks' ground, SIM_BLOCK_GROUND entries a block: the
// ground at its cells' corners (meters, WATER_SIM_GROUP + 1 a row, rows
// along z), then the level a lake holds each of its cells at (SIM_UNHELD
// where none does), then each of its cells' bed's roughness (its D84 in
// meters where friction follows Ferguson's law, the same negated where
// steps or knobs are carved into it and friction counts its grains alone),
// then the share of each of its cells covered by the solids standing in it
// (0 to 1), then the height their tops stand at over it (meters), cells
// row by row.
@group(0) @binding(1) var<storage, read> sim_ground: array<f32>;
// Each slot's step's start and then its first stage, each its pool's
// blocks' cells, a block's row by row: the water's surface, its flow along
// x and z, and its foam (0 none, 1 white all over; in the first stage, once
// the step churned it, its next).
@group(0) @binding(2) var<storage, read_write> sim_cells: array<vec4<f32>>;
// Each cell's water's depths at its faces as reconstructed, the mean of
// their squares across x (its west and east faces') and across z (its
// south and north faces'), which the pull of gravity down its ground reads;
// and, from a step's end until the next step's first stage, the current
// its surface runs at (meters a second along x and z, `sim_drift`), which
// the foam it carries rides.
@group(0) @binding(3) var<storage, read_write> sim_depths: array<vec2<f32>>;
// The fluxes through each slot's faces: each of its cells' east face's,
// then each of its cells' north face's, then those through the faces on its
// rectangle's west edge (a row each) and on its south edge (a column each):
// its mass and its momentum across and along the face; and in the step's
// first stage the fastest wave through it, in its second the head its
// water loses across it (meters).
@group(0) @binding(4) var<storage, read_write> sim_fluxes: array<vec4<f32>>;
// What rivers bring and what stands past the edges: each slot's cells
// rivers bring water to (the cell, its index among the slot's cells as
// bits; the cubic meters a second they bring; and the way the water runs as
// it comes, x and z, a share of SIM_SOURCE_SPEED); after every slot's, each
// slot's ring, the cells just past its edges as their water settled, two
// entries each: its surface (far below the ground where dry), its flow
// along x and z and the level a lake holds it at (SIM_UNHELD where none
// does); then its foam and the water it brings in, the discharge across a
// meter of the edge into the square (square meters a second; none where it
// brings none). South then north of each column, then west then east of
// each row.
@group(0) @binding(5) var<storage, read> sim_inputs: array<vec4<f32>>;
// Each slot's table: the block of its pool storing each block of its
// rectangle, row by row (SIM_NO_BLOCK where none does); then each block of
// its pool's place in the rectangle (its column, and its row above 16
// bits).
@group(0) @binding(6) var<storage, read> sim_blocks: array<u32>;
// Each slot's cells' fastest wave through their faces in the step's first
// stage, then the share of its outflow each can give in a stage
// (`sim_drain`), and their seconds; then each of its pool's blocks':
// whether it held water no lake holds at the step before's end, and
// whether it steps this step and the steps it stood still since its
// rectangle's last look or its waking, whichever came later.
@group(0) @binding(7) var<storage, read_write> sim_steps: array<vec2<f32>>;
@group(0) @binding(8) var<uniform> sim_slots: SimSlots;
// Each slot's cells' sums over their steps since its last look,
// SIM_CELL_SUMS entries a cell (`SimSums`, each step weighted by its
// seconds; then the seconds it ran at the fastest water runs, and the
// seconds themselves); then their means at that look, SIM_CELL_MEANS
// entries a cell, each over the seconds it was wet: its surface, flow along
// x and z and foam; the energy it loses (watts a square meter), the
// variance of its depth, the mean square of its vorticity and the variance
// of its velocity; and the share of the time it was wet, the share of the
// time wet its water left its bed and the share of the time it ran at the
// fastest water runs, then nothing. Then each of its pool's blocks' sums at a look,
// SIM_BLOCK_LOOKS entries each: the water its cells flowing held, their
// speeds, those cells and its cells stepping; the energy those lose, their
// heave, their swirl and those that ran at the fastest water runs; and the
// water its cells beside blocks the pool does not store held, and those
// cells.
@group(0) @binding(9) var<storage, read_write> sim_averages: array<vec4<f32>>;

// The most solids that push the square around the camera's water a frame
// (`engine_core::SIM_TOUCHES`).
const SIM_TOUCHES: u32 = 32u;

// A solid standing in the square around the camera's water: its center
// over the square's first corner and its datum (meters) and 1 where it is a
// ball, else 0 (w); the inverse of the axes mapping the unit cube (or ball)
// to it, a column each; its velocity (meters a second) and the meters it
// reaches up and down from its center (w); and the least and greatest
// corners of its footprint over the square's first corner (meters, x and
// z).
struct SimTouch {
    center: vec4<f32>,
    inverse: array<vec4<f32>, 3>,
    velocity: vec4<f32>,
    reach: vec4<f32>,
}

// The solids standing in the square around the camera's water this frame:
// how many (x); the blocks of the square they reach, the first along x and
// z, then how many along each; and each one, nearest the eye first.
struct SimTouches {
    count: vec4<u32>,
    blocks: vec4<u32>,
    touch: array<SimTouch, SIM_TOUCHES>,
}

@group(0) @binding(10) var<uniform> sim_touches: SimTouches;

// A slot's rectangle as a dispatch runs it.
struct Square {
    slot: u32,
    nx: u32,
    nz: u32,
    flags: u32,
    sourced: u32,
    dx: f32,
    live: f32,
    blocks: vec2<u32>,
    stored: u32,
}

fn sim_square(slot: u32) -> Square {
    let held = sim_slots.slot[slot];
    return Square(slot, held.shape.x, held.shape.y, held.shape.z, held.shape.w, held.step.x,
        held.step.y, held.blocks.xy, held.blocks.z);
}

fn sim_has(q: Square, flag: u32) -> bool {
    return (q.flags & flag) != 0u;
}

// The slots the buffers hold; the blocks a slot's pool holds, and their
// cells; the blocks a slot's table maps; and the faces along a slot's west
// and south edges it holds: from which each slot's entries start.
fn sim_slot_count() -> u32 {
    return frame.stage.y;
}

fn sim_pool() -> u32 {
    return frame.stage.z;
}

fn sim_held() -> u32 {
    return sim_pool() * SIM_BLOCK_CELLS;
}

fn sim_mapped() -> u32 {
    return frame.stage.w;
}

fn sim_edges() -> u32 {
    return frame.inputs.y;
}

fn sim_cells_base(q: Square) -> u32 {
    return q.slot * 2u * sim_held();
}

fn sim_ground_base(q: Square) -> u32 {
    return q.slot * sim_pool() * SIM_BLOCK_GROUND;
}

fn sim_depths_base(q: Square) -> u32 {
    return q.slot * sim_held();
}

fn sim_fluxes_base(q: Square) -> u32 {
    return q.slot * (2u * sim_held() + sim_edges());
}

fn sim_sources_base(q: Square) -> u32 {
    return q.slot * frame.inputs.x;
}

fn sim_ring_base(q: Square) -> u32 {
    return sim_slot_count() * frame.inputs.x + q.slot * 4u * sim_edges();
}

fn sim_table_base(q: Square) -> u32 {
    return q.slot * (sim_mapped() + sim_pool());
}

fn sim_steps_base(q: Square) -> u32 {
    return q.slot * (sim_held() + 2u * sim_pool());
}

fn sim_averages_base(q: Square) -> u32 {
    return q.slot * ((SIM_CELL_SUMS + SIM_CELL_MEANS) * sim_held() + SIM_BLOCK_LOOKS * sim_pool());
}

// Where slot `q` keeps the sums of the cell at `at`, its means, and the
// sums at a look of block `block` of its pool.
fn sim_sums_at(q: Square, at: u32) -> u32 {
    return sim_averages_base(q) + SIM_CELL_SUMS * at;
}

fn sim_means_at(q: Square, at: u32) -> u32 {
    return sim_averages_base(q) + SIM_CELL_SUMS * sim_held() + SIM_CELL_MEANS * at;
}

fn sim_looks_at(q: Square, block: u32) -> u32 {
    return sim_averages_base(q) + (SIM_CELL_SUMS + SIM_CELL_MEANS) * sim_held()
        + SIM_BLOCK_LOOKS * block;
}

// The block of slot `q`'s pool storing block `block` (its column and row)
// of its rectangle: SIM_NO_BLOCK where none does or the rectangle has no
// such block.
fn sim_block(q: Square, block: vec2<i32>) -> u32 {
    if any(block < vec2<i32>(0)) || any(block >= vec2<i32>(q.blocks)) {
        return SIM_NO_BLOCK;
    }
    return sim_blocks[sim_table_base(q) + u32(block.y) * q.blocks.x + u32(block.x)];
}

// The place in slot `q`'s rectangle of block `block` of its pool: its
// column and row.
fn sim_place(q: Square, block: u32) -> vec2<u32> {
    let place = sim_blocks[sim_table_base(q) + sim_mapped() + block];
    return vec2<u32>(place & 0xffffu, place >> 16u);
}

// Where slot `q` holds cell (i, j) of its rectangle: its index among the
// slot's cells, its block's first and then its place there; SIM_NO_BLOCK
// where it lies outside the rectangle or the pool does not store its
// block.
fn sim_at(q: Square, i: i32, j: i32) -> u32 {
    if i < 0 || j < 0 || i >= i32(q.nx) || j >= i32(q.nz) {
        return SIM_NO_BLOCK;
    }
    let cell = vec2<u32>(u32(i), u32(j));
    let block = sim_block(q, vec2<i32>(cell / WATER_SIM_GROUP));
    if block == SIM_NO_BLOCK {
        return SIM_NO_BLOCK;
    }
    let local = cell % WATER_SIM_GROUP;
    return block * SIM_BLOCK_CELLS + local.y * WATER_SIM_GROUP + local.x;
}

// The cell an invocation stands on: where its slot holds it, its column and
// row in the rectangle, and whether it is a cell of a rectangle that steps.
struct SimCell {
    at: u32,
    i: u32,
    j: u32,
    runs: bool,
}

// The cell invocation `local` of workgroup `group`, of `count` along x and
// z, stands on in slot `q`: a workgroup for each block of its pool, in
// rows of `count.x` of them.
fn sim_cell_of(q: Square, group: vec3<u32>, count: vec3<u32>, local: vec3<u32>) -> SimCell {
    let block = group.y * count.x + group.x;
    if group.z >= sim_slot_count() || block >= q.stored {
        return SimCell(0u, 0u, 0u, false);
    }
    let place = sim_place(q, block) * WATER_SIM_GROUP + local.xy;
    let at = block * SIM_BLOCK_CELLS + local.y * WATER_SIM_GROUP + local.x;
    let runs = sim_has(q, SIM_STEPPING) && place.x < q.nx && place.y < q.nz;
    return SimCell(at, place.x, place.y, runs);
}

// Where block `block` of the pool has its entries in the steps: whether it
// held water, then whether it steps and the steps it stood still.
fn sim_block_at(q: Square, block: u32) -> u32 {
    return sim_steps_base(q) + sim_held() + 2u * block;
}

// Whether the block of the cell at `at` steps this step: always, in a
// rectangle not sparse.
fn sim_active(q: Square, at: u32) -> bool {
    if !sim_has(q, SIM_SPARSE) {
        return true;
    }
    return sim_steps[sim_block_at(q, at / SIM_BLOCK_CELLS) + 1u].x > 0.5;
}

// The ground at corner (x, z) of block `block` of the pool, counted from
// its first cell's first corner.
fn sim_corner(q: Square, block: u32, x: u32, z: u32) -> f32 {
    return sim_ground[sim_ground_base(q) + block * SIM_BLOCK_GROUND
        + z * (WATER_SIM_GROUP + 1u) + x];
}

// The ground at the corners of the cell at `at`: its first's, then along
// x, along z and along both.
fn sim_corners(q: Square, at: u32) -> vec4<f32> {
    let block = at / SIM_BLOCK_CELLS;
    let x = at % WATER_SIM_GROUP;
    let z = (at % SIM_BLOCK_CELLS) / WATER_SIM_GROUP;
    return vec4<f32>(sim_corner(q, block, x, z), sim_corner(q, block, x + 1u, z),
        sim_corner(q, block, x, z + 1u), sim_corner(q, block, x + 1u, z + 1u));
}

// The ground at the middles of the faces across x of the cell at `at`, its
// west face's and its east face's; and of those across z, its south face's
// and its north face's.
fn sim_ground_x(q: Square, at: u32) -> vec2<f32> {
    let corners = sim_corners(q, at);
    return vec2<f32>(0.5 * (corners.x + corners.z), 0.5 * (corners.y + corners.w));
}

fn sim_ground_z(q: Square, at: u32) -> vec2<f32> {
    let corners = sim_corners(q, at);
    return vec2<f32>(0.5 * (corners.x + corners.y), 0.5 * (corners.z + corners.w));
}

// The ground at the middle of the cell at `at`.
fn sim_bed(q: Square, at: u32) -> f32 {
    let ground = sim_ground_x(q, at);
    return 0.5 * (ground.x + ground.y);
}

// The level a lake holds the cell at `at` at (SIM_UNHELD where none does),
// and whether one does.
fn sim_lake(q: Square, at: u32) -> f32 {
    return sim_ground[sim_ground_base(q) + (at / SIM_BLOCK_CELLS) * SIM_BLOCK_GROUND
        + SIM_BLOCK_CORNERS + at % SIM_BLOCK_CELLS];
}

fn sim_held_at(q: Square, at: u32) -> bool {
    return sim_lake(q, at) > 0.5 * SIM_UNHELD;
}

// The roughness of the bed of the cell at `at`.
fn sim_rough(q: Square, at: u32) -> f32 {
    return sim_ground[sim_ground_base(q) + (at / SIM_BLOCK_CELLS) * SIM_BLOCK_GROUND
        + SIM_BLOCK_CORNERS + SIM_BLOCK_CELLS + at % SIM_BLOCK_CELLS];
}

// The share of the cell at `at` covered by the solids standing in it, and
// the height their tops stand at over it; none where the slot holds no
// cell there (SIM_NO_BLOCK).
fn sim_cover(q: Square, at: u32) -> vec2<f32> {
    if at == SIM_NO_BLOCK {
        return vec2<f32>(0.0);
    }
    let share = sim_ground_base(q) + (at / SIM_BLOCK_CELLS) * SIM_BLOCK_GROUND
        + SIM_BLOCK_CORNERS + 2u * SIM_BLOCK_CELLS + at % SIM_BLOCK_CELLS;
    return vec2<f32>(sim_ground[share], sim_ground[share + SIM_BLOCK_CELLS]);
}

// The share of a face's water the solids covering the cell at `at` (its
// cover `cover`) block, the water at the face standing at `level`: the
// share of the cell they cover, times the share of the water's depth over
// the cell's ground they fill; all of their share where no water stands
// over that ground.
fn sim_blocked(q: Square, at: u32, cover: vec2<f32>, level: f32) -> f32 {
    if cover.x <= 0.0 {
        return 0.0;
    }
    let bed = sim_bed(q, at);
    let depth = level - bed;
    if depth <= SIM_DRY {
        return cover.x;
    }
    return cover.x * clamp((cover.y - bed) / depth, 0.0, 1.0);
}

// The share of the face between the cells at `low` and `high` (either
// SIM_NO_BLOCK where the slot holds none there) its water passes through
// in the state `state`: what neither side's solids block of the water
// standing at the higher of their surfaces (`sim_blocked`). The fluxes
// through a face carry this share of what open water would, and the pull
// of gravity down a cell's ground counts its faces' pressure by theirs, so
// water at rest beside a solid stays at rest; every stage and the cells on
// both sides of a face find it alike.
fn sim_opening(q: Square, state: u32, low: u32, high: u32) -> f32 {
    let near = sim_cover(q, low);
    let far = sim_cover(q, high);
    if near.x <= 0.0 && far.x <= 0.0 {
        return 1.0;
    }
    var level: f32;
    if low == SIM_NO_BLOCK {
        level = sim_cell(q, state, high).x;
    } else if high == SIM_NO_BLOCK {
        level = sim_cell(q, state, low).x;
    } else {
        level = max(sim_cell(q, state, low).x, sim_cell(q, state, high).x);
    }
    return 1.0 - max(sim_blocked(q, low, near, level), sim_blocked(q, high, far, level));
}

// What a face passing `open` of its water carries of the flux `flux`
// through open water: that share of its mass and momentum, its fastest wave
// or the head its water loses as they were.
fn sim_through(flux: vec4<f32>, open: f32) -> vec4<f32> {
    return vec4<f32>(flux.xyz * open, flux.w);
}

// How the ground of the cell at `at` leans: φ² = 1 + |∇b|², its slope
// along x and z between its faces' middles.
fn sim_lean(q: Square, at: u32) -> f32 {
    let x = sim_ground_x(q, at);
    let z = sim_ground_z(q, at);
    let slope = vec2<f32>(x.y - x.x, z.y - z.x) / q.dx;
    return 1.0 + dot(slope, slope);
}

// Gravity along the bed through the face across x between the cells at
// `west` and `east`, either SIM_NO_BLOCK where the slot holds none there:
// g/φ², φ² = 1 + |∇b|², the bed's slope across the face over both cells
// beside it (over the one there is, where the slot holds one alone) and
// along it between its corners. Both cells find it alike, so a lake at
// rest stays at rest.
fn sim_gravity_x(q: Square, west: u32, east: u32) -> f32 {
    var across: f32;
    var along: f32;
    if west == SIM_NO_BLOCK {
        let ground = sim_ground_x(q, east);
        let corners = sim_corners(q, east);
        across = (ground.y - ground.x) / q.dx;
        along = (corners.z - corners.x) / q.dx;
    } else {
        let ground = sim_ground_x(q, west);
        let corners = sim_corners(q, west);
        along = (corners.w - corners.y) / q.dx;
        if east == SIM_NO_BLOCK {
            across = (ground.y - ground.x) / q.dx;
        } else {
            across = (sim_ground_x(q, east).y - ground.x) / (2.0 * q.dx);
        }
    }
    return SIM_GRAVITY / (1.0 + across * across + along * along);
}

// Gravity along the bed through the face across z between the cells at
// `south` and `north`, as `sim_gravity_x` finds it across x.
fn sim_gravity_z(q: Square, south: u32, north: u32) -> f32 {
    var across: f32;
    var along: f32;
    if south == SIM_NO_BLOCK {
        let ground = sim_ground_z(q, north);
        let corners = sim_corners(q, north);
        across = (ground.y - ground.x) / q.dx;
        along = (corners.y - corners.x) / q.dx;
    } else {
        let ground = sim_ground_z(q, south);
        let corners = sim_corners(q, south);
        along = (corners.w - corners.z) / q.dx;
        if north == SIM_NO_BLOCK {
            across = (ground.y - ground.x) / q.dx;
        } else {
            across = (sim_ground_z(q, north).y - ground.x) / (2.0 * q.dx);
        }
    }
    return SIM_GRAVITY / (1.0 + across * across + along * along);
}

// The cell at `at` of the state `state` (0: the step's start, 1: its first
// stage); a cell standing still stands as the step started.
fn sim_cell(q: Square, state: u32, at: u32) -> vec4<f32> {
    let standing = select(0u, state, sim_active(q, at));
    return sim_cells[sim_cells_base(q) + standing * sim_held() + at];
}

// The cell at `at` of the state `state`, or `instead` where the slot holds
// none there (SIM_NO_BLOCK): past the rectangle's edge, or in a block its
// pool does not store.
fn sim_water(q: Square, state: u32, at: u32, instead: vec4<f32>) -> vec4<f32> {
    if at == SIM_NO_BLOCK {
        return instead;
    }
    return sim_cell(q, state, at);
}

// Whether the cell at `at` holds no water in the state `state`: none where
// the slot holds no cell there, as its neighbors stand themselves in for
// one.
fn sim_dry(q: Square, state: u32, at: u32) -> bool {
    if at == SIM_NO_BLOCK {
        return false;
    }
    return sim_cell(q, state, at).x - sim_bed(q, at) <= SIM_DRY;
}

fn sim_minmod(a: f32, b: f32, c: f32) -> f32 {
    if a > 0.0 && b > 0.0 && c > 0.0 {
        return min(a, min(b, c));
    }
    if a < 0.0 && b < 0.0 && c < 0.0 {
        return max(a, max(b, c));
    }
    return 0.0;
}

// Half a cell's change across it, from the values either side of it,
// limited.
fn sim_half_slope(before: f32, middle: f32, after: f32) -> f32 {
    return 0.5 * sim_minmod(SIM_THETA * (middle - before), 0.5 * (after - before),
        SIM_THETA * (after - middle));
}

// The velocity of `q` square meters a second of flow `h` meters deep,
// desingularized as the water thins out.
fn sim_velocity(h: f32, q: f32) -> f32 {
    let h4 = h * h * h * h;
    let thin = SIM_THIN * SIM_THIN * SIM_THIN * SIM_THIN;
    return 1.4142135 * h * q / sqrt(h4 + max(h4, thin));
}

// A face's water from the cell's surface `w` reconstructed there, the
// ground at the face, and its flow across and along the face: no faster
// than SIM_FASTEST, so no wave outruns the steps' length.
fn sim_face(w: f32, ground: f32, across: f32, along: f32) -> vec4<f32> {
    let h = max(w - ground, 0.0);
    let velocity = clamp(vec2<f32>(sim_velocity(h, across), sim_velocity(h, along)),
        vec2<f32>(-SIM_FASTEST), vec2<f32>(SIM_FASTEST));
    return vec4<f32>(w, h, velocity);
}

// A cell's two faces across an axis as reconstructed, the low one (west or
// south) and the high one (east or north): the water's surface and depth at
// each, and its velocity across the face and along it.
struct SimSides {
    low: vec4<f32>,
    high: vec4<f32>,
}

// Whether the cell at `at`, its water's surface standing at `surface` and
// the ground at its faces across an axis at `ground`, lies at a waterline
// along that axis: wet, no lake's, its surface below the ground at its
// higher face, and the cell past that face (at `higher`) dry in the state
// `state`.
fn sim_shore(q: Square, state: u32, at: u32, surface: f32, ground: vec2<f32>, higher: u32)
    -> bool {
    return surface < max(ground.x, ground.y) && surface - 0.5 * (ground.x + ground.y) > SIM_DRY
        && !sim_held_at(q, at) && sim_dry(q, state, higher);
}

// A cell's two faces across an axis as reconstructed from its water `here`
// (its surface, and its flow across and along the axis), its neighbors'
// `before` and `after` along the axis and the ground at its faces
// `ground`: its slopes limited by its neighbors', its water kept above the
// ground at both faces; or, where it lies at a waterline (`shore`), its
// water lying flat against its lower face, as deep there as holds its
// depth over the whole cell, √(2·h·Δb) (Bollermann et al. 2013), running at
// its velocity, and its higher face dry, so shores stand at rest and thin
// edges run no faster than their water.
fn sim_sides(here: vec3<f32>, before: vec3<f32>, after: vec3<f32>, ground: vec2<f32>,
    shore: bool) -> SimSides {
    if shore {
        let low = min(ground.x, ground.y);
        let high = max(ground.x, ground.y);
        let depth = max(here.x - 0.5 * (ground.x + ground.y), 0.0);
        let deep = sqrt(2.0 * depth * (high - low));
        let velocity = vec2<f32>(sim_velocity(depth, here.y), sim_velocity(depth, here.z));
        let wet = sim_face(low + deep, low, deep * velocity.x, deep * velocity.y);
        let dry = vec4<f32>(high, 0.0, 0.0, 0.0);
        if ground.y > ground.x {
            return SimSides(wet, dry);
        }
        return SimSides(dry, wet);
    }
    let slope = vec3<f32>(sim_half_slope(before.x, here.x, after.x),
        sim_half_slope(before.y, here.y, after.y), sim_half_slope(before.z, here.z, after.z));
    var high_w = here.x + slope.x;
    var low_w = here.x - slope.x;
    if high_w < ground.y {
        high_w = ground.y;
        low_w = 2.0 * here.x - ground.y;
    } else if low_w < ground.x {
        low_w = ground.x;
        high_w = 2.0 * here.x - ground.x;
    }
    return SimSides(sim_face(low_w, ground.x, here.y - slope.y, here.z - slope.z),
        sim_face(high_w, ground.y, here.y + slope.y, here.z + slope.z));
}

// The faces across x of the cell at `at`, cell (i, j), from the state
// `state`: its slopes limited by its neighbors' (itself standing for one
// the slot does not hold), its water kept above the ground at both faces,
// or lying against its waterline.
fn sim_across_x(q: Square, state: u32, at: u32, i: u32, j: u32) -> SimSides {
    let here = sim_cell(q, state, at);
    let west_at = sim_at(q, i32(i) - 1, i32(j));
    let east_at = sim_at(q, i32(i) + 1, i32(j));
    let west = sim_water(q, state, west_at, here);
    let east = sim_water(q, state, east_at, here);
    let ground = sim_ground_x(q, at);
    let higher = select(west_at, east_at, ground.y > ground.x);
    let shore = sim_shore(q, state, at, here.x, ground, higher);
    return sim_sides(here.xyz, west.xyz, east.xyz, ground, shore);
}

// The faces across z of the cell at `at`, cell (i, j), the flow along z
// across them.
fn sim_across_z(q: Square, state: u32, at: u32, i: u32, j: u32) -> SimSides {
    let here = sim_cell(q, state, at);
    let south_at = sim_at(q, i32(i), i32(j) - 1);
    let north_at = sim_at(q, i32(i), i32(j) + 1);
    let south = sim_water(q, state, south_at, here);
    let north = sim_water(q, state, north_at, here);
    let ground = sim_ground_z(q, at);
    let higher = select(south_at, north_at, ground.y > ground.x);
    let shore = sim_shore(q, state, at, here.x, ground, higher);
    return sim_sides(here.xzy, south.xzy, north.xzy, ground, shore);
}

// The minmod of each pair of entries of `a` and `b`: the one smaller in
// size where both share a sign, else none.
fn sim_minmod_pair(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return select(vec3<f32>(0.0), sign(a) * min(abs(a), abs(b)), a * b > vec3<f32>(0.0));
}

// The central-upwind flux through a face whose water stands as `l` on its
// near side and `r` on its far one, gravity along the bed there `g`: its
// mass, and its momentum across and along the face; and the fastest wave
// through it. Its diffusion takes back the share of the jump between the
// sides the state between their fastest waves accounts for (Kurganov and
// Lin's anti-diffusion, 2007), so shear stays sharp; sides alike leave
// none of it, a pure pressure as at rest.
fn sim_flux(l: vec4<f32>, r: vec4<f32>, g: f32) -> vec4<f32> {
    let cl = sqrt(g * l.y);
    let cr = sqrt(g * r.y);
    let fastest = max(max(l.z + cl, r.z + cr), 0.0);
    let slowest = min(min(l.z - cl, r.z - cr), 0.0);
    let span = fastest - slowest;
    let wave = max(fastest, -slowest);
    if span < 1.0e-6 {
        return vec4<f32>(0.0, 0.0, 0.0, wave);
    }
    let fl = vec3<f32>(l.y * l.z, l.y * l.z * l.z + 0.5 * g * l.y * l.y, l.y * l.z * l.w);
    let fr = vec3<f32>(r.y * r.z, r.y * r.z * r.z + 0.5 * g * r.y * r.y, r.y * r.z * r.w);
    let ul = vec3<f32>(l.x, l.y * l.z, l.y * l.w);
    let ur = vec3<f32>(r.x, r.y * r.z, r.y * r.w);
    let between = (fastest * ur - slowest * ul - (fr - fl)) / span;
    let anti = sim_minmod_pair(ur - between, between - ul);
    let flux = (fastest * fl - slowest * fr) / span
        + (fastest * slowest / span) * (ur - ul - anti);
    return vec4<f32>(flux, wave);
}

// The head (meters) water loses through a face from its side `l` to its
// side `r` as its mass `mass` crosses it: what the side it leaves holds
// above the side it reaches, its surface and its speed's head (|u|²/2g),
// none where it would gain any.
fn sim_lost(l: vec4<f32>, r: vec4<f32>, mass: f32) -> f32 {
    let near = l.x + 0.5 * (l.z * l.z + l.w * l.w) / SIM_GRAVITY;
    let far = r.x + 0.5 * (r.z * r.z + r.w * r.w) / SIM_GRAVITY;
    return max(select(near - far, far - near, mass < 0.0), 0.0);
}

// What a face keeps of the flux `flux` through it between its sides `l`
// and `r`: its mass and momentum; then, in the step's first stage, the
// fastest wave through it, and in its second the head its water loses.
fn sim_kept(flux: vec4<f32>, l: vec4<f32>, r: vec4<f32>) -> vec4<f32> {
    if frame.stage.x == 0u {
        return flux;
    }
    return vec4<f32>(flux.xyz, sim_lost(l, r, flux.x));
}

// The flux `flux` through the face from the cell at `near` to the cell at
// `far`, none where it would carry water out of a lake.
fn sim_from_lakes(q: Square, flux: vec4<f32>, near: u32, far: u32) -> vec4<f32> {
    if (flux.x > 0.0 && sim_held_at(q, near)) || (flux.x < 0.0 && sim_held_at(q, far)) {
        return vec4<f32>(0.0, 0.0, 0.0, flux.w);
    }
    return flux;
}

// Ring entry `ghost` (its first of two) as the face it shows the square
// over the edge whose ground stands at `ground`, its flow across and along
// that face as `across_z` says (true for faces between rows): its water
// held at its own surface, first order.
fn sim_ghost_face(q: Square, ghost: u32, ground: f32, across_z: bool) -> vec4<f32> {
    let cell = sim_inputs[sim_ring_base(q) + 2u * ghost];
    let across = select(cell.y, cell.z, across_z);
    let along = select(cell.z, cell.y, across_z);
    return sim_face(cell.x, ground, across, along);
}

// Whether ring entry `ghost` is held at a lake's level.
fn sim_ghost_held(q: Square, ghost: u32) -> bool {
    return sim_inputs[sim_ring_base(q) + 2u * ghost].w > 0.5 * SIM_UNHELD;
}

// The discharge ring entry `ghost` brings into the square across a meter of
// its edge (square meters a second; none where it brings none).
fn sim_ghost_inflow(q: Square, ghost: u32) -> f32 {
    return max(sim_inputs[sim_ring_base(q) + 2u * ghost + 1u].y, 0.0);
}

// The flux through a face on the square's edge from what stands as `l` to
// what stands as `r`, one the inside's face and one a ring cell's, whether
// each is held by a lake, gravity along the bed there `g`: none where it
// would carry water out of one.
fn sim_edge(l: vec4<f32>, r: vec4<f32>, near_held: bool, far_held: bool, g: f32) -> vec4<f32> {
    let flux = sim_flux(l, r, g);
    if (flux.x > 0.0 && near_held) || (flux.x < 0.0 && far_held) {
        return vec4<f32>(0.0, 0.0, 0.0, flux.w);
    }
    return flux;
}

// The flux the ring brings in through a face, `inflow` square meters a
// second of its water standing as `face` (the ring's, with its velocity
// across and along the face), `inward` the way into the square across it
// (+1 or -1) and `g` gravity along the bed there: its mass; the momentum
// its water carries in at its own speed, with its pressure; the momentum
// along the face it carries in; and the fastest wave through it.
fn sim_inflow(face: vec4<f32>, inflow: f32, inward: f32, g: f32) -> vec4<f32> {
    let mass = inward * inflow;
    let speed = inward * sim_velocity(face.y, inflow);
    let wave = abs(speed) + sqrt(g * face.y);
    return vec4<f32>(mass, mass * speed + 0.5 * g * face.y * face.y, mass * face.w, wave);
}

// The flux through a face on the edge of a square with a ring, as the face
// keeps it (`sim_kept`), between its inside's face `inside` and ring entry
// `ghost` past it (its first of two), the ground at the face standing at
// `ground`, its flow across and along it as `across_z` says, `inward` the
// way into the square across it (+1 or -1), `held` whether a lake holds the
// inside and `g` gravity along the bed there: where the ring brings water
// in, its discharge with the momentum of its water as it settled there;
// else between the two as they stand, none where it would carry water out
// of a lake.
fn sim_ring_flux(q: Square, ghost: u32, inside: vec4<f32>, ground: f32, across_z: bool,
    inward: f32, held: bool, g: f32) -> vec4<f32> {
    let outside = sim_ghost_face(q, ghost, ground, across_z);
    let outside_held = sim_ghost_held(q, ghost);
    let inflow = sim_ghost_inflow(q, ghost);
    if inflow > 0.0 && !outside_held {
        return sim_kept(sim_inflow(outside, inflow, inward, g), outside, outside);
    }
    if inward > 0.0 {
        return sim_kept(sim_edge(outside, inside, outside_held, held, g), outside, inside);
    }
    return sim_kept(sim_edge(inside, outside, held, outside_held, g), inside, outside);
}

// The flux through a face on an open rectangle's edge whose inside stands
// as `face`, `inward` the way into the rectangle across it (+1 or -1) and
// `g` gravity along the bed there: water leaves as it runs, the inside
// standing on both sides, and none comes in.
fn sim_open(face: vec4<f32>, inward: f32, g: f32) -> vec4<f32> {
    let flux = sim_flux(face, face, g);
    if flux.x * inward > 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, flux.w);
    }
    return flux;
}

// Each cell that steps, and each beside one that does, finds the fluxes
// through its east and north faces (and its west and south ones on the
// rectangle's edges), each by the gravity along the bed through it and as
// much as the solids beside it let through (`sim_opening`): between two
// cells that step, as their faces say; on the edges, open or between the
// inside and the ring cell past it, or as much as the ring brings in;
// toward a cell the slot does not hold, open (`sim_open_west` finds the
// west and south ones there); and between a cell that steps and one
// standing still, nothing, as no water reaches either this step. A cell
// that steps keeps its faces' depths.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_exchange(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let q = sim_square(group.z);
    let cell = sim_cell_of(q, group, count, local);
    if !cell.runs {
        return;
    }
    let nx = q.nx;
    let nz = q.nz;
    let i = cell.i;
    let j = cell.j;
    let at = cell.at;
    let state = frame.stage.x;
    let open = sim_has(q, SIM_OPEN);
    let fluxes = sim_fluxes_base(q);
    let edges = fluxes + 2u * sim_held();
    let steps = sim_active(q, at);
    let east = sim_at(q, i32(i) + 1, i32(j));
    let north = sim_at(q, i32(i), i32(j) + 1);
    let east_steps = east != SIM_NO_BLOCK && sim_active(q, east);
    let north_steps = north != SIM_NO_BLOCK && sim_active(q, north);
    var x = SimSides(vec4<f32>(0.0), vec4<f32>(0.0));
    var z = SimSides(vec4<f32>(0.0), vec4<f32>(0.0));
    if steps {
        x = sim_across_x(q, state, at, i, j);
        z = sim_across_z(q, state, at, i, j);
        sim_depths[sim_depths_base(q) + at] = vec2<f32>(
            0.5 * (x.low.y * x.low.y + x.high.y * x.high.y),
            0.5 * (z.low.y * z.low.y + z.high.y * z.high.y));
    }
    // Its east face: between columns i and i + 1, or the east edge.
    if steps || east_steps {
        var flux = vec4<f32>(0.0);
        if i + 1u == nx {
            let gravity = sim_gravity_x(q, at, SIM_NO_BLOCK);
            if open {
                flux = sim_kept(sim_open(x.high, -1.0, gravity), x.high, x.high);
            } else {
                flux = sim_ring_flux(q, 2u * nx + 2u * j + 1u, x.high, sim_ground_x(q, at).y,
                    false, -1.0, sim_held_at(q, at), gravity);
            }
        } else if steps && east_steps {
            let far = sim_across_x(q, state, east, i + 1u, j).low;
            let gravity = sim_gravity_x(q, at, east);
            flux = sim_kept(sim_from_lakes(q, sim_flux(x.high, far, gravity), at, east), x.high,
                far);
        } else if steps && east == SIM_NO_BLOCK {
            let gravity = sim_gravity_x(q, at, SIM_NO_BLOCK);
            flux = sim_kept(sim_open(x.high, -1.0, gravity), x.high, x.high);
        }
        sim_fluxes[fluxes + at] = sim_through(flux, sim_opening(q, state, at, east));
    }
    // The west edge.
    if i == 0u && steps {
        let gravity = sim_gravity_x(q, SIM_NO_BLOCK, at);
        var flux: vec4<f32>;
        if open {
            flux = sim_kept(sim_open(x.low, 1.0, gravity), x.low, x.low);
        } else {
            flux = sim_ring_flux(q, 2u * nx + 2u * j, x.low, sim_ground_x(q, at).x, false, 1.0,
                sim_held_at(q, at), gravity);
        }
        sim_fluxes[edges + j] = sim_through(flux, sim_opening(q, state, SIM_NO_BLOCK, at));
    }
    // Its north face: between rows j and j + 1, or the north edge.
    if steps || north_steps {
        var flux = vec4<f32>(0.0);
        if j + 1u == nz {
            let gravity = sim_gravity_z(q, at, SIM_NO_BLOCK);
            if open {
                flux = sim_kept(sim_open(z.high, -1.0, gravity), z.high, z.high);
            } else {
                flux = sim_ring_flux(q, 2u * i + 1u, z.high, sim_ground_z(q, at).y, true, -1.0,
                    sim_held_at(q, at), gravity);
            }
        } else if steps && north_steps {
            let far = sim_across_z(q, state, north, i, j + 1u).low;
            let gravity = sim_gravity_z(q, at, north);
            flux = sim_kept(sim_from_lakes(q, sim_flux(z.high, far, gravity), at, north), z.high,
                far);
        } else if steps && north == SIM_NO_BLOCK {
            let gravity = sim_gravity_z(q, at, SIM_NO_BLOCK);
            flux = sim_kept(sim_open(z.high, -1.0, gravity), z.high, z.high);
        }
        sim_fluxes[fluxes + sim_held() + at] = sim_through(flux,
            sim_opening(q, state, at, north));
    }
    // The south edge.
    if j == 0u && steps {
        let gravity = sim_gravity_z(q, SIM_NO_BLOCK, at);
        var flux: vec4<f32>;
        if open {
            flux = sim_kept(sim_open(z.low, 1.0, gravity), z.low, z.low);
        } else {
            flux = sim_ring_flux(q, 2u * i, z.low, sim_ground_z(q, at).x, true, 1.0,
                sim_held_at(q, at), gravity);
        }
        sim_fluxes[edges + nz + i] = sim_through(flux, sim_opening(q, state, SIM_NO_BLOCK, at));
    }
}

// The flux out through the west face (`across_z` false) or the south one
// (true) of the cell at `at`, cell (i, j), toward a cell the slot does not
// hold: open, as the rectangle's edges are, as much as its solids let
// through. The cells beside it keep no face toward it, so it finds that
// face again.
fn sim_open_west(q: Square, at: u32, i: u32, j: u32, across_z: bool) -> vec4<f32> {
    let state = frame.stage.x;
    let open = sim_opening(q, state, SIM_NO_BLOCK, at);
    if across_z {
        return sim_through(sim_open(sim_across_z(q, state, at, i, j).low, 1.0,
            sim_gravity_z(q, SIM_NO_BLOCK, at)), open);
    }
    return sim_through(sim_open(sim_across_x(q, state, at, i, j).low, 1.0,
        sim_gravity_x(q, SIM_NO_BLOCK, at)), open);
}

// The fluxes into and out of the cell at `at`, cell (i, j): through its
// west, east, south and north faces; open through a face toward a cell the
// slot does not hold.
fn sim_west(q: Square, at: u32, i: u32, j: u32) -> vec4<f32> {
    let fluxes = sim_fluxes_base(q);
    if i == 0u {
        return sim_fluxes[fluxes + 2u * sim_held() + j];
    }
    let west = sim_at(q, i32(i) - 1, i32(j));
    if west == SIM_NO_BLOCK {
        return sim_open_west(q, at, i, j, false);
    }
    return sim_fluxes[fluxes + west];
}

fn sim_east(q: Square, at: u32) -> vec4<f32> {
    return sim_fluxes[sim_fluxes_base(q) + at];
}

fn sim_south(q: Square, at: u32, i: u32, j: u32) -> vec4<f32> {
    let fluxes = sim_fluxes_base(q);
    if j == 0u {
        return sim_fluxes[fluxes + 2u * sim_held() + q.nz + i];
    }
    let south = sim_at(q, i32(i), i32(j) - 1);
    if south == SIM_NO_BLOCK {
        return sim_open_west(q, at, i, j, true);
    }
    return sim_fluxes[fluxes + sim_held() + south];
}

fn sim_north(q: Square, at: u32) -> vec4<f32> {
    return sim_fluxes[sim_fluxes_base(q) + sim_held() + at];
}

// The step to each of a cell's four neighbors: west, east, south, north.
fn sim_toward(k: i32) -> vec2<i32> {
    return select(vec2<i32>(0, 2 * (k - 2) - 1), vec2<i32>(2 * k - 1, 0), k < 2);
}

// Settling regions' cells: the fastest wave through each cell's faces in
// the step's first stage.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_waves(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let q = sim_square(group.z);
    let cell = sim_cell_of(q, group, count, local);
    if !cell.runs || !sim_has(q, SIM_LOCAL) || !sim_active(q, cell.at) {
        return;
    }
    let at = cell.at;
    let fastest = max(max(sim_west(q, at, cell.i, cell.j).w, sim_east(q, at).w),
        max(sim_south(q, at, cell.i, cell.j).w, sim_north(q, at).w));
    sim_steps[sim_steps_base(q) + at].x = fastest;
}

// Settling regions' cells: each one's seconds this step, as long as the
// fastest wave through its faces and its neighbors' allows, so neighbors
// step alike.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_seconds(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let q = sim_square(group.z);
    let cell = sim_cell_of(q, group, count, local);
    if !cell.runs || !sim_has(q, SIM_LOCAL) || !sim_active(q, cell.at) {
        return;
    }
    let base = sim_steps_base(q);
    var fastest = sim_steps[base + cell.at].x;
    for (var k = 0; k < 4; k++) {
        let toward = sim_toward(k);
        let beside = sim_at(q, i32(cell.i) + toward.x, i32(cell.j) + toward.y);
        if beside != SIM_NO_BLOCK {
            fastest = max(fastest, sim_steps[base + beside].x);
        }
    }
    sim_steps[base + cell.at].y =
        SIM_LOCAL_SHARE * SIM_COURANT * q.dx / max(fastest, SIM_SLOWEST);
}

// The seconds this step of the cell at `at`: its own where it steps
// locally, else the rectangle's.
fn sim_seconds_at(q: Square, at: u32) -> f32 {
    if sim_has(q, SIM_LOCAL) {
        return sim_steps[sim_steps_base(q) + at].y;
    }
    return q.live;
}

// Each cell that steps, once a stage's fluxes are found: the share of its
// outflow through its faces it can give over its seconds, all of it where
// it holds that much water and what it holds where it holds less
// (Bollermann et al.'s draining time step, 2013), into its first entry of
// the steps; the stage's update carries a face's flux as far as the cell
// its water leaves can give it, so no cell runs dry below its ground,
// whatever its faces' reconstruction and anti-diffusion.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_drain(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let q = sim_square(group.z);
    let cell = sim_cell_of(q, group, count, local);
    if !cell.runs || !sim_active(q, cell.at) {
        return;
    }
    let at = cell.at;
    let leaving = max(-sim_west(q, at, cell.i, cell.j).x, 0.0) + max(sim_east(q, at).x, 0.0)
        + max(-sim_south(q, at, cell.i, cell.j).x, 0.0) + max(sim_north(q, at).x, 0.0);
    let held = max(sim_cell(q, frame.stage.x, at).x - sim_bed(q, at), 0.0) * q.dx;
    let given = leaving * sim_seconds_at(q, at);
    var share = 1.0;
    if given > held {
        share = held / given;
    }
    sim_steps[sim_steps_base(q) + at].x = share;
}

// The share of its outflow the cell at `at` can give this stage, as
// `sim_drain` found it: all of it where the slot holds no cell there.
fn sim_drained(q: Square, at: u32) -> f32 {
    if at == SIM_NO_BLOCK {
        return 1.0;
    }
    return sim_steps[sim_steps_base(q) + at].x;
}

// How hard the bed of the cell at `at` holds back water `h` meters deep
// running over it: k of dq/dt = -k·|q|·q (per square meter). Where its
// roughness is a D84, Darcy and Weisbach's f/(8h²) of Ferguson's f (2007,
// 8/a2²·(D84/h)² + 8/a1²·(D84/h)^(1/3)); where steps or knobs are carved
// into it, g·n²/h^(7/3) of its grains' Manning's n, D84^(1/6)/SIM_STRICKLER;
// the depth over the D84 counting within SIM_SHALLOWEST and SIM_DEEPEST;
// times φ², as water running down sloping ground runs over more of it
// (`engine_terrain::shallow_water::resistance`).
fn sim_resistance(q: Square, at: u32, h: f32) -> f32 {
    let rough = sim_rough(q, at);
    let relative = clamp(h / max(abs(rough), h / SIM_DEEPEST), SIM_SHALLOWEST, SIM_DEEPEST);
    var k: f32;
    if rough >= 0.0 {
        k = (1.0 / (SIM_FERGUSON_A2 * SIM_FERGUSON_A2 * relative * relative)
            + 1.0 / (SIM_FERGUSON_A1 * SIM_FERGUSON_A1 * pow(relative, 1.0 / 3.0))) / (h * h);
    } else {
        let n = pow(h / relative, 1.0 / 6.0) / SIM_STRICKLER;
        k = SIM_GRAVITY * n * n / (h * h * pow(h, 1.0 / 3.0));
    }
    return k * sim_lean(q, at);
}

// A cell's water `water` (surface and flow) settled over `ground` after a
// stage, the cell at `at`: never below the ground, slowed by its bed's
// friction over `rubbing` seconds by the exact root of its implicit step,
// q' = 2q/(1 + √(1 + 4·dt·k·|q|)), no faster than SIM_FASTEST (its last
// entry 1 where that held it back, else 0), reset where it stopped being
// finite, and held at its lake's level `held` where one holds it.
fn sim_settle(q: Square, at: u32, water: vec3<f32>, ground: f32, held: f32, rubbing: f32)
    -> vec4<f32> {
    if held > 0.5 * SIM_UNHELD {
        return vec4<f32>(held, 0.0, 0.0, 0.0);
    }
    var next = water;
    if !(abs(next.x) < 1.0e20 && abs(next.y) < 1.0e20 && abs(next.z) < 1.0e20) {
        next = vec3<f32>(ground, 0.0, 0.0);
    }
    next.x = max(next.x, ground);
    let h = next.x - ground;
    if h <= SIM_DRY {
        return vec4<f32>(next.x, 0.0, 0.0, 0.0);
    }
    let k = sim_resistance(q, at, h);
    var flow = next.yz * (2.0 / (1.0 + sqrt(1.0 + 4.0 * rubbing * k * length(next.yz))));
    let speed = length(flow) / h;
    var clamped = 0.0;
    if speed > SIM_FASTEST {
        flow *= SIM_FASTEST / speed;
        clamped = 1.0;
    }
    return vec4<f32>(next.x, flow, clamped);
}

// The pull of gravity down a cell's ground across an axis, times its
// width: its faces' gravity along the bed `gravity` (its low face's and its
// high one's, each times the share of the face its water passes through,
// `sim_opening`) pulling its water `depth` deep down ground rising by
// `rise` across it, and the push of the pressure its faces' depths bear
// (the mean of their squares, `squares`) under their different gravity, a
// face blocked by solids pushing back as a wall would; water at rest feels
// none, as its faces' pressure balances it exactly, at a waterline and
// beside a solid too.
fn sim_pull(gravity: vec2<f32>, depth: f32, rise: f32, squares: f32) -> f32 {
    return -0.5 * (gravity.x + gravity.y) * depth * rise + 0.5 * (gravity.y - gravity.x) * squares;
}

// Each cell's water after a stage, its foam as the step started, each face
// carrying as much as the cell its water leaves can give: friction slows
// it over the stage's share of the step, all of it in the first stage and
// half in the second, as Heun's average counts the first's half again, so
// water settled stays settled whatever its steps; a summing rectangle
// counts the seconds its cells ran at the fastest water runs; and in the
// first stage, a sparse rectangle's blocks start the step dry, their cells'
// water marking them again at its end.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_update(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let q = sim_square(group.z);
    let cell = sim_cell_of(q, group, count, local);
    if !cell.runs || !sim_active(q, cell.at) {
        return;
    }
    let at = cell.at;
    let i = i32(cell.i);
    let j = i32(cell.j);
    let held = sim_held();
    let stage = frame.stage.x;
    let dx = q.dx;
    let dt = sim_seconds_at(q, at);
    let west_at = sim_at(q, i - 1, j);
    let east_at = sim_at(q, i + 1, j);
    let south_at = sim_at(q, i, j - 1);
    let north_at = sim_at(q, i, j + 1);
    let own = sim_drained(q, at);
    var west = sim_west(q, at, cell.i, cell.j).xyz;
    west *= select(own, sim_drained(q, west_at), west.x > 0.0);
    var east = sim_east(q, at).xyz;
    east *= select(sim_drained(q, east_at), own, east.x > 0.0);
    var south = sim_south(q, at, cell.i, cell.j).xyz;
    south *= select(own, sim_drained(q, south_at), south.x > 0.0);
    var north = sim_north(q, at).xyz;
    north *= select(sim_drained(q, north_at), own, north.x > 0.0);
    let base = sim_cells_base(q);
    let start = sim_cells[base + at];
    var staged = start;
    if stage == 1u {
        staged = sim_cells[base + held + at];
    }
    let ground_x = sim_ground_x(q, at);
    let ground_z = sim_ground_z(q, at);
    let ground = 0.5 * (ground_x.x + ground_x.y);
    // The pull reads the depth over the whole cell, which its faces'
    // reconstruction across each axis holds, at a waterline as elsewhere.
    let depth = max(vec2<f32>(staged.x) - 0.5 * vec2<f32>(ground_x.x + ground_x.y,
        ground_z.x + ground_z.y), vec2<f32>(0.0));
    let squares = sim_depths[sim_depths_base(q) + at];
    let gravity_x = vec2<f32>(sim_gravity_x(q, west_at, at), sim_gravity_x(q, at, east_at))
        * vec2<f32>(sim_opening(q, stage, west_at, at), sim_opening(q, stage, at, east_at));
    let gravity_z = vec2<f32>(sim_gravity_z(q, south_at, at), sim_gravity_z(q, at, north_at))
        * vec2<f32>(sim_opening(q, stage, south_at, at), sim_opening(q, stage, at, north_at));
    let pull = vec2<f32>(sim_pull(gravity_x, depth.x, ground_x.y - ground_x.x, squares.x),
        sim_pull(gravity_z, depth.y, ground_z.y - ground_z.x, squares.y)) / dx;
    // The flux through faces between rows carries the flow along z across
    // them and the flow along x along them.
    let change = vec3<f32>(
        -(east.x - west.x) / dx - (north.x - south.x) / dx,
        -(east.y - west.y) / dx - (north.z - south.z) / dx + pull.x,
        -(east.z - west.z) / dx - (north.y - south.y) / dx + pull.y);
    var next = start.xyz + dt * change;
    if stage == 1u {
        next = 0.5 * (start.xyz + staged.xyz + dt * change);
    }
    let rubbing = select(dt, 0.5 * dt, stage == 1u);
    let settled = sim_settle(q, at, next, ground, sim_lake(q, at), rubbing);
    let written = select(base + held + at, base + at, stage == 1u);
    sim_cells[written] = vec4<f32>(settled.xyz, start.w);
    if stage == 1u && settled.w > 0.0 && sim_has(q, SIM_SUMMING) {
        sim_averages[sim_sums_at(q, at) + 3u].x += dt;
    }
    let first = at % SIM_BLOCK_CELLS == 0u;
    if stage == 0u && sim_has(q, SIM_SPARSE) && first {
        sim_steps[sim_block_at(q, at / SIM_BLOCK_CELLS)].x = 0.0;
    }
}

// The speeds (meters a second) at which water passing a solid starts to
// tear air in, and at which it churns foam as a churn of one does
// (`sim_churn`'s, holding half the most): Ervine, McKeogh and Elsawy (1980)
// find water drawing air in from about 0.7 to 1.3 meters a second.
const SIM_TEARING: vec2<f32> = vec2<f32>(0.7, 1.3);

// Where along the upright line through `point` (meters over the square's
// first corner, x and z) solid `touch` stands: the lowest and the highest
// heights over the square's datum it spans there (meters), the lowest above
// the highest where it stands nowhere on the line. In its unit cube or
// ball the line runs from `at`, at the datum, by `up` a meter: a ball holds
// it where |at + y·up| ≤ 1, a box where each of its parts lies within 1.
fn sim_touch_span(touch: SimTouch, point: vec2<f32>) -> vec2<f32> {
    let inverse = mat3x3<f32>(touch.inverse[0].xyz, touch.inverse[1].xyz,
        touch.inverse[2].xyz);
    let at = inverse * (vec3<f32>(point.x, 0.0, point.y) - touch.center.xyz);
    let up = inverse[1];
    if touch.center.w > 0.5 {
        let rise = dot(up, up);
        let along = dot(at, up);
        let reach = along * along - rise * (dot(at, at) - 1.0);
        if rise <= 0.0 || reach < 0.0 {
            return vec2<f32>(1.0, -1.0);
        }
        return (vec2<f32>(-along) + vec2<f32>(-1.0, 1.0) * sqrt(reach)) / rise;
    }
    var span = touch.center.y + vec2<f32>(-touch.velocity.w, touch.velocity.w);
    for (var k = 0; k < 3; k++) {
        if up[k] != 0.0 {
            let ends = (vec2<f32>(-1.0, 1.0) - at[k]) / up[k];
            span = vec2<f32>(max(span.x, min(ends.x, ends.y)), min(span.y, max(ends.x, ends.y)));
        } else if abs(at[k]) > 1.0 {
            return vec2<f32>(1.0, -1.0);
        }
    }
    return span;
}

// The solids standing in the square around the camera's water push it
// before each step, a workgroup for each block they reach: in each cell
// whose middle lies under one's footprint, where it reaches below the
// water's surface there, the water's flow relaxes over the step toward the
// solid's velocity times its depth (Brinkman penalization of the shallow
// water's momentum, dq/dt = −(χ/τ)·(q − h·u), χ the share of the cell's
// depth the solid fills and τ the step's seconds, solved exactly over the
// step), so water flows around a solid standing still and is pushed ahead
// of one moving; and where it stands up through the surface, the cell's
// foam churns as `sim_churn` churns it, by how much faster than
// SIM_TEARING.x the water passes it, a share of the span to SIM_TEARING.y
// churning one, toward SIM_FOAM_MOST. A lake holding a cell still at its
// level, a solid churns its foam alone. Solids over one cell push it in
// turn, nearest the eye first.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_touch(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let q = sim_square(0u);
    let cell = (sim_touches.blocks.xy + group.xy) * WATER_SIM_GROUP + local.xy;
    if !sim_has(q, SIM_STEPPING) || cell.x >= q.nx || cell.y >= q.nz {
        return;
    }
    let at = sim_at(q, i32(cell.x), i32(cell.y));
    if at == SIM_NO_BLOCK {
        return;
    }
    let still = sim_held_at(q, at);
    let base = sim_cells_base(q);
    var water = sim_cells[base + at];
    let bed = sim_bed(q, at);
    let h = water.x - bed;
    if h <= SIM_DRY {
        return;
    }
    let middle = (vec2<f32>(cell) + 0.5) * q.dx;
    let flowing = vec2<f32>(sim_velocity(h, water.y), sim_velocity(h, water.z));
    for (var k = 0u; k < sim_touches.count.x; k++) {
        let touch = sim_touches.touch[k];
        if any(middle < touch.reach.xy) || any(middle > touch.reach.zw) {
            continue;
        }
        let span = sim_touch_span(touch, middle);
        let filled = min(span.y, water.x) - max(span.x, bed);
        if filled <= 0.0 {
            continue;
        }
        let solid = touch.velocity.xz;
        if span.y > water.x {
            let churn = max(length(flowing - solid) - SIM_TEARING.x, 0.0)
                / (SIM_TEARING.y - SIM_TEARING.x);
            let kept = exp(-churn * q.live / SIM_FOAM_LIFE);
            water.w = SIM_FOAM_MOST + (water.w - SIM_FOAM_MOST) * kept;
        }
        if !still {
            let carried = h * solid;
            let flow = carried + (water.yz - carried) * exp(-min(filled / h, 1.0));
            water = vec4<f32>(water.x, flow, water.w);
        }
    }
    sim_cells[base + at] = water;
}

// The flow of head a face's water loses (square meters a second times
// meters), from its flux `flux` (its mass along the axis, the head lost in
// its last entry), the cell before it along the axis able to give `low` of
// what flows out of it and the cell past it `high`.
fn sim_face_lost(flux: vec4<f32>, low: f32, high: f32) -> f32 {
    return abs(flux.x) * select(high, low, flux.x > 0.0) * flux.w;
}

// One step's rate of energy loss per bed area at cell `cell` of slot `q`,
// watts a square meter: the work its water does against its bed's friction
// (ρ·g times the friction slope times its flow, ρ·k·|q|³/h), and half the
// power the water crossing its faces loses there (ρ·g times the head lost
// times the flow through), each face's shared with the cell past it; none
// where it is dry. It reads the water the step left and the fluxes of its
// second stage, each carried as far as the cell its water left could give
// it, so it holds from the step's end until the next step's first fluxes.
fn sim_dissipation(q: Square, cell: SimCell) -> f32 {
    let at = cell.at;
    let water = sim_cells[sim_cells_base(q) + at];
    let h = water.x - sim_bed(q, at);
    if h <= SIM_DRY {
        return 0.0;
    }
    let flow = length(water.yz);
    let friction = sim_resistance(q, at, h) * flow * flow * flow / h;
    let i = i32(cell.i);
    let j = i32(cell.j);
    let own = sim_drained(q, at);
    let fluxes = sim_fluxes_base(q);
    let edges = fluxes + 2u * sim_held();
    var lost = sim_face_lost(sim_east(q, at), own, sim_drained(q, sim_at(q, i + 1, j)))
        + sim_face_lost(sim_north(q, at), own, sim_drained(q, sim_at(q, i, j + 1)));
    // Faces toward cells the slot does not hold are open, their sides alike:
    // their water loses nothing there.
    if cell.i == 0u {
        lost += sim_face_lost(sim_fluxes[edges + cell.j], 1.0, own);
    } else {
        let west = sim_at(q, i - 1, j);
        if west != SIM_NO_BLOCK {
            lost += sim_face_lost(sim_fluxes[fluxes + west], sim_drained(q, west), own);
        }
    }
    if cell.j == 0u {
        lost += sim_face_lost(sim_fluxes[edges + q.nz + cell.i], 1.0, own);
    } else {
        let south = sim_at(q, i, j - 1);
        if south != SIM_NO_BLOCK {
            lost += sim_face_lost(sim_fluxes[fluxes + sim_held() + south], sim_drained(q, south),
                own);
        }
    }
    return SIM_DENSITY * (friction + 0.5 * SIM_GRAVITY * lost / q.dx);
}

// Foam, the lace white water leaves on the water: churned up where the
// water loses energy fast enough to break, held by its churning toward the
// most it holds as it fades. The seconds it takes to fade to 1/e; the most
// churning holds; the meters of water under which a cell holds none; the
// least rate (square meters a second cubed) the water loses energy at,
// gravity times its surface's slope times its speed, for it to churn any;
// the meters water drops into a cell to plunge, and the Froude number it
// runs at into water slower than its waves to jump; the vorticity (a
// second) past which eddy lines churn it, as much again past it holding
// half the most; the convergence (a second) and the power the water loses
// (watts a square meter of bed) at which each holds half the most.
const SIM_FOAM_LIFE: f32 = 20.0;
const SIM_FOAM_MOST: f32 = 1.5;
const SIM_FOAM_WET: f32 = 0.005;
const SIM_FOAM_LOSS: f32 = 0.02;
const SIM_FOAM_DROP: f32 = 0.05;
const SIM_FOAM_JUMP: f32 = 1.7;
const SIM_FOAM_SWIRL: f32 = 1.0;
const SIM_FOAM_PILING: f32 = 0.5;
const SIM_FOAM_POWER: f32 = 500.0;
// Foam rides the water's surface, which runs as the depth's mean does but
// for two things. In a bend the helical flow carries the surface toward the
// outer bank and the bed toward the inner one (Rozovskii 1957; de Vriend
// 1977): the flow along the bed turns inward by an angle whose tangent is
// 7 h/R, h the depth and R the streamlines' radius of curvature (Engelund
// 1974), and the surface turns outward by as much, SIM_BEND depths over the
// radius. Sharper bends saturate, as their secondary flow flattens the
// main flow's profile that drives it (Blanckaert and de Vriend 2003): the
// turn's tangent stops at SIM_BEND_MOST, 45 degrees, which the linear rule
// reaches where R nears seven depths, so the surface never runs more
// across the stream than along it. And the wind drags the surface at
// SIM_WIND_DRIFT of its speed along its heading (Wu 1975).
const SIM_BEND: f32 = 7.0;
const SIM_BEND_MOST: f32 = 1.0;
const SIM_WIND_DRIFT: f32 = 0.03;

// The first state's cell at `at`: its surface, depth and velocity; or
// `instead` where it is dry.
fn sim_moving_at(q: Square, at: u32, instead: vec4<f32>) -> vec4<f32> {
    let cell = sim_cells[sim_cells_base(q) + at];
    let h = cell.x - sim_bed(q, at);
    if h <= SIM_FOAM_WET {
        return instead;
    }
    return vec4<f32>(cell.x, h, sim_velocity(h, cell.y), sim_velocity(h, cell.z));
}

// The first state's cell (i, j): its surface, depth and velocity; or
// `instead` where it is dry or the slot holds none there.
fn sim_moving(q: Square, i: i32, j: i32, instead: vec4<f32>) -> vec4<f32> {
    let at = sim_at(q, i, j);
    if at == SIM_NO_BLOCK {
        return instead;
    }
    return sim_moving_at(q, at, instead);
}

// The Froude number of water standing as `water` (its surface, depth and
// velocity): how many times faster than its waves it runs.
fn sim_foam_froude(water: vec4<f32>) -> f32 {
    return length(water.zw) / sqrt(SIM_GRAVITY * max(water.y, SIM_FOAM_WET));
}

// The power (watts a square meter of bed) water standing as `upstream`
// (its surface, depth and velocity) spends landing in a cell `dx` meters
// across standing as `here`, the cell past it standing as `beyond` and
// `toward` the way from it into the cell: the head it loses times the
// water it brings, where it plunges, its surface falling into the cell
// SIM_FOAM_DROP or more further than it falls on past it (a step, not the
// steady fall of a slope), or jumps, running at SIM_FOAM_JUMP times its
// waves' speed or faster into water slower than its waves; none elsewhere.
fn sim_foam_landing(upstream: vec4<f32>, here: vec4<f32>, beyond: vec4<f32>, toward: vec2<f32>,
        dx: f32) -> f32 {
    let drop = upstream.x - here.x;
    let step = drop - max(here.x - beyond.x, 0.0);
    let plunges = step >= SIM_FOAM_DROP;
    let jumps = sim_foam_froude(upstream) >= SIM_FOAM_JUMP && sim_foam_froude(here) < 1.0;
    if !plunges && !jumps {
        return 0.0;
    }
    let brought = upstream.y * max(dot(upstream.zw, toward), 0.0);
    let speeds = dot(upstream.zw, upstream.zw) - dot(here.zw, here.zw);
    let head = select(drop, step, plunges) + speeds / (2.0 * SIM_GRAVITY);
    return SIM_DENSITY * SIM_GRAVITY * brought * max(head, 0.0) / dx;
}

// The current the surface of water standing as `here` (its surface, depth
// and velocity) runs at, its neighbors along x and z standing as `west`,
// `east`, `south` and `north` (`sim_moving`'s, itself standing in for one
// dry or missing), over cells `dx` meters across: its velocity turned
// outward in a bend by δ, tan δ = SIM_BEND·h/R, no more than SIM_BEND_MOST.
// R is the streamlines' radius of curvature, 1/R = |u × (u·∇)u| / |u|³, the
// velocity's derivatives taken between its neighbors as `sim_churn` takes
// them; outward is away from the side u × (u·∇)u says the streamlines
// curve toward.
fn sim_surface_current(here: vec4<f32>, west: vec4<f32>, east: vec4<f32>, south: vec4<f32>,
        north: vec4<f32>, dx: f32) -> vec2<f32> {
    let velocity = here.zw;
    let along_x = (east.zw - west.zw) / (2.0 * dx);
    let along_z = (north.zw - south.zw) / (2.0 * dx);
    let turning = velocity.x * along_x + velocity.y * along_z;
    let speed = length(velocity);
    let curving = (velocity.x * turning.y - velocity.y * turning.x)
        / max(speed * speed * speed, 1.0e-12);
    let tangent = clamp(SIM_BEND * here.y * curving, -SIM_BEND_MOST, SIM_BEND_MOST);
    return vec2<f32>(velocity.x + tangent * velocity.y, velocity.y - tangent * velocity.x)
        * inverseSqrt(1.0 + tangent * tangent);
}

// Each cell that steps, once the step's stages have run and rivers have
// brought their water: the current its surface runs at
// (`sim_surface_current`) with the wind's drag, SIM_WIND_DRIFT of the
// frame's wind, into its entry of `sim_depths`, which the stages are done
// with; none where it is dry. The foam it carries rides it (`sim_churn`).
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_drift(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let q = sim_square(group.z);
    let cell = sim_cell_of(q, group, count, local);
    if !cell.runs || !sim_active(q, cell.at) {
        return;
    }
    let here = sim_moving_at(q, cell.at, vec4<f32>(0.0));
    var drift = vec2<f32>(0.0);
    if here.y > SIM_FOAM_WET {
        let i = i32(cell.i);
        let j = i32(cell.j);
        drift = sim_surface_current(here, sim_moving(q, i - 1, j, here),
            sim_moving(q, i + 1, j, here), sim_moving(q, i, j - 1, here),
            sim_moving(q, i, j + 1, here), q.dx) + SIM_WIND_DRIFT * frame.wind.xy;
    }
    sim_depths[sim_depths_base(q) + cell.at] = drift;
}

// Whether a solid stands up through the water's surface, standing at
// `surface`, in cell `place` of the live square, as `sim_touch` finds the
// solids there, reaching from below the surface to above it at the cell's
// middle: none in the blocks no solid reaches, and none in settling
// regions, whose solids' uniform holds none.
fn sim_pierced(q: Square, place: vec2<u32>, surface: f32) -> bool {
    let block = place / WATER_SIM_GROUP;
    let first = sim_touches.blocks.xy;
    if any(block < first) || any(block >= first + sim_touches.blocks.zw) {
        return false;
    }
    let middle = (vec2<f32>(place) + 0.5) * q.dx;
    for (var k = 0u; k < sim_touches.count.x; k++) {
        let touch = sim_touches.touch[k];
        if any(middle < touch.reach.xy) || any(middle > touch.reach.zw) {
            continue;
        }
        let span = sim_touch_span(touch, middle);
        if span.x < surface && span.y > surface {
            return true;
        }
    }
    return false;
}

// The share of a cell solids standing up through its surface cover from
// which they wall it off from the foam carried into it: half of it, where
// they stand over more of it than its water does.
const SIM_WALLED: f32 = 0.5;

// Whether the cell at `at`, cell `place`, its water's surface standing at
// `surface`, takes the foam carried into it: where it is wet and no solid
// stands up through its surface, neither one pushing its water
// (`sim_pierced`) nor those covering SIM_WALLED of it or more, so foam
// carried against dry ground or a solid stays along it.
fn sim_foam_takes(q: Square, at: u32, place: vec2<u32>, surface: f32) -> bool {
    let cover = sim_cover(q, at);
    let walled = cover.x >= SIM_WALLED && cover.y > surface;
    return surface - sim_bed(q, at) > SIM_FOAM_WET && !walled && !sim_pierced(q, place, surface);
}

// Whether the cell at `at`, its water's surface standing at `surface`,
// gives the foam it carries across a face whose middle's ground stands at
// `ground`: where it is wet and its water stands over that ground, as no
// water crosses a wall higher than it stands.
fn sim_foam_gives(q: Square, at: u32, surface: f32, ground: f32) -> bool {
    return min(surface - sim_bed(q, at), surface - ground) > SIM_FOAM_WET;
}

// The foam the surface's current carries a second through the face between
// the cells at `low` and `high`, cell `place` and the cell past it along x
// or, where `across_z`, along z (positive along the axis): the two cells'
// surface currents across it (`sim_drift`) averaged, carrying the foam of
// the cell it leaves where that cell gives it (`sim_foam_gives`) into the
// other where it takes it (`sim_foam_takes`); none between a cell that
// steps and one standing still, as no water reaches either. Both cells
// beside the face find it alike, so what one gives the other takes.
fn sim_foam_across(q: Square, low: u32, high: u32, place: vec2<u32>, across_z: bool) -> f32 {
    if !sim_active(q, low) || !sim_active(q, high) {
        return 0.0;
    }
    let base = sim_cells_base(q);
    let near = sim_cells[base + low];
    let far = sim_cells[base + high];
    let currents = sim_depths_base(q);
    let both = sim_depths[currents + low] + sim_depths[currents + high];
    var current: f32;
    var ground: f32;
    var past: vec2<u32>;
    if across_z {
        current = 0.5 * both.y;
        ground = sim_ground_z(q, low).y;
        past = place + vec2<u32>(0u, 1u);
    } else {
        current = 0.5 * both.x;
        ground = sim_ground_x(q, low).y;
        past = place + vec2<u32>(1u, 0u);
    }
    if current > 0.0 {
        if sim_foam_gives(q, low, near.x, ground) && sim_foam_takes(q, high, past, far.x) {
            return current * near.w;
        }
    } else if sim_foam_gives(q, high, far.x, ground) && sim_foam_takes(q, low, place, near.x) {
        return current * far.w;
    }
    return 0.0;
}

// The foam the surface's current carries a second through the face on side
// `side` (-1 its west or south face, 1 its east or north one) of the cell
// at `at`, cell `place`, across x or, where `across_z`, across z, toward a
// cell its slot does not hold (positive along the axis): its own surface
// current across it (`sim_drift`) carries its foam out where the cell
// gives it (`sim_foam_gives`), and brings in that of ring entry `ghost`
// (its first of two; SIM_NO_BLOCK where no ring stands there) where that
// entry's water stands over the ground at the face's middle and the cell
// takes foam (`sim_foam_takes`). Open edges, and blocks the pool does not
// store, bring none in, as they bring no water.
fn sim_foam_beyond(q: Square, at: u32, place: vec2<u32>, side: f32, across_z: bool, ghost: u32)
    -> f32 {
    let water = sim_cells[sim_cells_base(q) + at];
    let drift = sim_depths[sim_depths_base(q) + at];
    var current: f32;
    var faces: vec2<f32>;
    if across_z {
        current = drift.y;
        faces = sim_ground_z(q, at);
    } else {
        current = drift.x;
        faces = sim_ground_x(q, at);
    }
    let ground = select(faces.x, faces.y, side > 0.0);
    if current * side > 0.0 {
        return select(0.0, current * water.w, sim_foam_gives(q, at, water.x, ground));
    }
    if ghost == SIM_NO_BLOCK || !sim_foam_takes(q, at, place, water.x) {
        return 0.0;
    }
    let ring = sim_ring_base(q) + 2u * ghost;
    return select(0.0, current * sim_inputs[ring + 1u].x,
        sim_inputs[ring].x - ground > SIM_FOAM_WET);
}

// The first state's foam of cell `cell` after `seconds` of its surface's
// current carrying it conserved, ∂c/∂t + ∇·(c·u_s) = 0: what it held less
// what crosses its four faces out of it, by upwind fluxes
// (`sim_foam_across`, or `sim_foam_beyond` toward cells its slot does not
// hold), so it thins where the surface spreads, gathers where it converges
// and stays along what it is carried against. Its steps keep it above
// none: they let the fastest water and waves cross no more than
// SIM_COURANT of a cell, and the surface's current, its water's velocity
// turned and the wind's drift, runs within them, so its four faces carry
// off less than the cell holds. Past SIM_FOAM_MOST it heaps, its cover
// held at the most.
fn sim_foam_carried(q: Square, cell: SimCell, seconds: f32) -> f32 {
    let at = cell.at;
    let place = vec2<u32>(cell.i, cell.j);
    let ringed = !sim_has(q, SIM_OPEN);
    var leaving = 0.0;
    for (var k = 0; k < 4; k++) {
        let toward = sim_toward(k);
        let across_z = k >= 2;
        let side = f32(toward.x + toward.y);
        let beside = vec2<i32>(place) + toward;
        let other = sim_at(q, beside.x, beside.y);
        var flux: f32;
        if other == SIM_NO_BLOCK {
            // The ring stands past a ringed rectangle's edges: its columns'
            // south and north cells, then its rows' west and east ones.
            let outside = any(beside < vec2<i32>(0)) || beside.x >= i32(q.nx)
                || beside.y >= i32(q.nz);
            var ghost = SIM_NO_BLOCK;
            if ringed && outside {
                if k < 2 {
                    ghost = 2u * q.nx + 2u * cell.j + u32(k);
                } else {
                    ghost = 2u * cell.i + u32(k) - 2u;
                }
            }
            flux = sim_foam_beyond(q, at, place, side, across_z, ghost);
        } else if side > 0.0 {
            flux = sim_foam_across(q, at, other, place, across_z);
        } else {
            flux = sim_foam_across(q, other, at, vec2<u32>(beside), across_z);
        }
        leaving += side * flux;
    }
    let foam = sim_cells[sim_cells_base(q) + at].w;
    return clamp(foam - seconds / q.dx * leaving, 0.0, SIM_FOAM_MOST);
}

// Each cell's foam after a step, into the second state's: what its
// surface's current carried there conserved (`sim_foam_carried`), faded,
// and what the water churned up there. Its churn sums the power its water
// loses as it runs and that of the water landing in it from its neighbors,
// each as a share of SIM_FOAM_POWER; the vorticity of its eddy lines past
// SIM_FOAM_SWIRL, as a share of it; and its convergence, as a share of
// SIM_FOAM_PILING; none where its water loses energy slower than
// SIM_FOAM_LOSS. Churning at `churn`, its foam nears churn / (1 + churn) of
// SIM_FOAM_MOST, 1 + churn times as fast as foam fades, so it never passes
// the most.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_churn(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let q = sim_square(group.z);
    let cell = sim_cell_of(q, group, count, local);
    if !cell.runs || !sim_active(q, cell.at) {
        return;
    }
    let next = sim_cells_base(q) + sim_held() + cell.at;
    let i = i32(cell.i);
    let j = i32(cell.j);
    let here = sim_moving_at(q, cell.at, vec4<f32>(0.0));
    if here.y <= SIM_FOAM_WET {
        sim_cells[next].w = 0.0;
        return;
    }
    let dx = q.dx;
    let dt = sim_seconds_at(q, cell.at);
    let carried = sim_foam_carried(q, cell, dt);
    let west = sim_moving(q, i - 1, j, here);
    let east = sim_moving(q, i + 1, j, here);
    let south = sim_moving(q, i, j - 1, here);
    let north = sim_moving(q, i, j + 1, here);
    let landing = sim_foam_landing(west, here, east, vec2<f32>(1.0, 0.0), dx)
        + sim_foam_landing(east, here, west, vec2<f32>(-1.0, 0.0), dx)
        + sim_foam_landing(south, here, north, vec2<f32>(0.0, 1.0), dx)
        + sim_foam_landing(north, here, south, vec2<f32>(0.0, -1.0), dx);
    let slope = length(vec2<f32>(east.x - west.x, north.x - south.x)) / (2.0 * dx);
    let swirl = abs(east.w - west.w - north.z + south.z) / (2.0 * dx);
    let converging = max(-(east.z - west.z + north.w - south.w) / (2.0 * dx), 0.0);
    var churn = (max(sim_dissipation(q, cell), 0.0) + landing) / SIM_FOAM_POWER
        + max(swirl - SIM_FOAM_SWIRL, 0.0) / SIM_FOAM_SWIRL + converging / SIM_FOAM_PILING;
    if SIM_GRAVITY * slope * length(here.zw) < SIM_FOAM_LOSS {
        churn = 0.0;
    }
    let steady = SIM_FOAM_MOST * churn / (1.0 + churn);
    sim_cells[next].w = steady + (carried - steady) * exp(-(1.0 + churn) * dt / SIM_FOAM_LIFE);
}

// Where a block's cells held water no lake holds at a step's end, as its
// first entry of `sim_steps` keeps it: SIM_WET_CELLS where any did, and the side
// bits of the sides along which one did.
const SIM_WET_CELLS: u32 = 1u;
const SIM_WET_WEST: u32 = 2u;
const SIM_WET_EAST: u32 = 4u;
const SIM_WET_SOUTH: u32 = 8u;
const SIM_WET_NORTH: u32 = 16u;

// The sides of its block a workgroup's cells hold water along.
var<workgroup> sim_wet_sides: atomic<u32>;

// The foam after the step is the next one's start; and a sparse
// rectangle's cells holding water no lake holds mark their block, and the
// sides of it they lie along.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_churned(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
) {
    if index == 0u {
        atomicStore(&sim_wet_sides, 0u);
    }
    workgroupBarrier();
    let q = sim_square(group.z);
    let cell = sim_cell_of(q, group, count, local);
    let at = cell.at;
    if cell.runs && sim_active(q, at) {
        let base = sim_cells_base(q);
        var water = sim_cells[base + at];
        water.w = sim_cells[base + sim_held() + at].w;
        sim_cells[base + at] = water;
        if sim_has(q, SIM_SPARSE) && !sim_held_at(q, at) && water.x - sim_bed(q, at) > SIM_DRY {
            let inside = at % SIM_BLOCK_CELLS;
            let x = inside % WATER_SIM_GROUP;
            let z = inside / WATER_SIM_GROUP;
            var sides = SIM_WET_CELLS;
            sides |= select(0u, SIM_WET_WEST, x == 0u);
            sides |= select(0u, SIM_WET_EAST, x + 1u == WATER_SIM_GROUP);
            sides |= select(0u, SIM_WET_SOUTH, z == 0u);
            sides |= select(0u, SIM_WET_NORTH, z + 1u == WATER_SIM_GROUP);
            atomicOr(&sim_wet_sides, sides);
        }
    }
    workgroupBarrier();
    let sides = atomicLoad(&sim_wet_sides);
    if index == 0u && sides != 0u && cell.runs {
        sim_steps[sim_block_at(q, at / SIM_BLOCK_CELLS)].x = f32(sides);
    }
}

// What a cell's water adds to its sums for each of its seconds: while it
// is wet, its depth, flow along x and z and foam; the energy it loses
// (watts a square meter), its depth squared, its vorticity squared and its
// speed squared; and its velocity along x and z; whether it is wet; and
// whether its water leaves its bed. Dry, it adds none of its water, so its
// means are those of the time it was wet.
struct SimSums {
    water: vec4<f32>,
    motion: vec4<f32>,
    shares: vec4<f32>,
}

// Adds `sums` over `seconds` to the sums of the cell at `at`, and the
// seconds themselves.
fn sim_add(q: Square, at: u32, seconds: f32, sums: SimSums) {
    let base = sim_sums_at(q, at);
    sim_averages[base] += seconds * sums.water;
    sim_averages[base + 1u] += seconds * sums.motion;
    sim_averages[base + 2u] += seconds * sums.shares;
    sim_averages[base + 3u].y += seconds;
}

// What the cell at `at` adds to its sums standing still, its water
// standing as `water`: wet, its depth, flow and foam, its depth squared and
// that it is wet; nothing moving.
fn sim_still(q: Square, at: u32, water: vec4<f32>) -> SimSums {
    let h = max(water.x - sim_bed(q, at), 0.0);
    let wet = select(0.0, 1.0, h > SIM_DRY);
    return SimSums(wet * vec4<f32>(h, water.yzw), vec4<f32>(0.0, wet * h * h, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, wet, 0.0));
}

// The seconds each step a block stands still counts for in its cells'
// sums: the longest a local step takes, as a cell whose water and whose
// neighbors' water stand still takes it.
fn sim_standing(q: Square) -> f32 {
    return SIM_LOCAL_SHARE * SIM_COURANT * q.dx / SIM_SLOWEST;
}

// The velocity of the water of cell (i, j) as the step left it, or
// `instead` where the slot holds none there.
fn sim_flow_at(q: Square, i: i32, j: i32, instead: vec2<f32>) -> vec2<f32> {
    let at = sim_at(q, i, j);
    if at == SIM_NO_BLOCK {
        return instead;
    }
    let water = sim_cells[sim_cells_base(q) + at];
    let h = max(water.x - sim_bed(q, at), 0.0);
    return vec2<f32>(sim_velocity(h, water.y), sim_velocity(h, water.z));
}

// How the bed of cell `cell` curves along the water's velocity `velocity`:
// vᵀHv, H its ground's second derivatives, along x and z between its
// neighbors' beds and its own (none along an axis where the slot holds a
// neighbor on one side of it alone), across both between its corners.
fn sim_bend(q: Square, cell: SimCell, velocity: vec2<f32>) -> f32 {
    let at = cell.at;
    let i = i32(cell.i);
    let j = i32(cell.j);
    let bed = sim_bed(q, at);
    let area = q.dx * q.dx;
    var bend = vec3<f32>(0.0);
    let west = sim_at(q, i - 1, j);
    let east = sim_at(q, i + 1, j);
    if west != SIM_NO_BLOCK && east != SIM_NO_BLOCK {
        bend.x = (sim_bed(q, east) + sim_bed(q, west) - 2.0 * bed) / area;
    }
    let south = sim_at(q, i, j - 1);
    let north = sim_at(q, i, j + 1);
    if south != SIM_NO_BLOCK && north != SIM_NO_BLOCK {
        bend.y = (sim_bed(q, north) + sim_bed(q, south) - 2.0 * bed) / area;
    }
    let corners = sim_corners(q, at);
    bend.z = (corners.w - corners.y - corners.z + corners.x) / area;
    return velocity.x * velocity.x * bend.x + velocity.y * velocity.y * bend.y
        + 2.0 * velocity.x * velocity.y * bend.z;
}

// A summing rectangle's cells after each step, their foam churned: each
// adds its water over its seconds to its sums (`SimSums`): the energy it
// loses (`sim_dissipation`), its vorticity from its neighbors' velocities
// (its own standing in for a neighbor the slot does not hold; none where
// it is dry), and its water leaving its bed where the bed curves away
// under it faster than gravity holds it down, g + vᵀHv ≤ 0.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_tally(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let q = sim_square(group.z);
    let cell = sim_cell_of(q, group, count, local);
    if !cell.runs || !sim_has(q, SIM_SUMMING) || !sim_active(q, cell.at) {
        return;
    }
    let at = cell.at;
    let water = sim_cells[sim_cells_base(q) + at];
    let h = max(water.x - sim_bed(q, at), 0.0);
    let velocity = vec2<f32>(sim_velocity(h, water.y), sim_velocity(h, water.z));
    let i = i32(cell.i);
    let j = i32(cell.j);
    let west = sim_flow_at(q, i - 1, j, velocity);
    let east = sim_flow_at(q, i + 1, j, velocity);
    let south = sim_flow_at(q, i, j - 1, velocity);
    let north = sim_flow_at(q, i, j + 1, velocity);
    let wet = h > SIM_DRY;
    let swirl = select(0.0, ((east.y - west.y) - (north.x - south.x)) / (2.0 * q.dx), wet);
    let leaves = wet && SIM_GRAVITY + sim_bend(q, cell, velocity) <= 0.0;
    let moving = vec4<f32>(sim_dissipation(q, cell), h * h, swirl * swirl,
        dot(velocity, velocity));
    let sums = SimSums(select(vec4<f32>(0.0), vec4<f32>(h, water.yzw), wet),
        select(vec4<f32>(0.0), moving, wet),
        vec4<f32>(select(vec2<f32>(0.0), velocity, wet), select(0.0, 1.0, wet),
            select(0.0, 1.0, leaves)));
    sim_add(q, at, sim_seconds_at(q, at), sums);
}

// Rivers bring their water to their cells (each cell once) after a step,
// over each cell's seconds, marking their blocks in a sparse rectangle.
@compute @workgroup_size(WATER_SIM_SOURCE_GROUP)
fn sim_source(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = sim_square(id.z);
    if id.z >= sim_slot_count() || !sim_has(q, SIM_STEPPING) || id.x >= q.sourced {
        return;
    }
    let source = sim_inputs[sim_sources_base(q) + id.x];
    let at = bitcast<u32>(source.x);
    if sim_held_at(q, at) {
        return;
    }
    let cells = sim_cells_base(q);
    var cell = sim_cells[cells + at];
    let added = source.y * sim_seconds_at(q, at) / (q.dx * q.dx);
    cell.x += added;
    cell.y += added * source.z * SIM_SOURCE_SPEED;
    cell.z += added * source.w * SIM_SOURCE_SPEED;
    sim_cells[cells + at] = cell;
    if sim_has(q, SIM_SPARSE) {
        sim_steps[sim_block_at(q, at / SIM_BLOCK_CELLS)].x = 1.0;
    }
}

// Where the cells of block `place` (its column and row in the slot's
// rectangle) held water at the step before's end, as `sim_churned` marks
// it; nothing for a block the slot does not store.
fn sim_wet_at(q: Square, place: vec2<i32>) -> u32 {
    let block = sim_block(q, place);
    if block == SIM_NO_BLOCK {
        return 0u;
    }
    return u32(sim_steps[sim_block_at(q, block)].x);
}

// A sparse rectangle's blocks, before each step, an invocation for each
// block of its pool: a block steps where it held water no lake holds at the
// step before's end, or where a block beside it held water along the side
// they share; else it stands still, counting the steps it stands. A block
// waking adds to its cells' sums their water over the steps it stood still,
// standing as it stood (`sim_still`, each step `sim_standing`'s seconds).
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_activity(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let q = sim_square(group.z);
    let block = (group.y * count.x + group.x) * SIM_BLOCK_CELLS + local;
    let sparse = sim_has(q, SIM_STEPPING) && sim_has(q, SIM_SPARSE);
    if group.z >= sim_slot_count() || !sparse || block >= q.stored {
        return;
    }
    let place = vec2<i32>(sim_place(q, block));
    // Water crosses less than a cell a step: it reaches a block from one
    // beside it along the side they share.
    var wet = (sim_wet_at(q, place) & SIM_WET_CELLS) != 0u;
    wet = wet || (sim_wet_at(q, place - vec2<i32>(1, 0)) & SIM_WET_EAST) != 0u;
    wet = wet || (sim_wet_at(q, place + vec2<i32>(1, 0)) & SIM_WET_WEST) != 0u;
    wet = wet || (sim_wet_at(q, place - vec2<i32>(0, 1)) & SIM_WET_NORTH) != 0u;
    wet = wet || (sim_wet_at(q, place + vec2<i32>(0, 1)) & SIM_WET_SOUTH) != 0u;
    let entry = sim_block_at(q, block) + 1u;
    let state = sim_steps[entry];
    if wet && state.x < 0.5 && state.y > 0.0 && sim_has(q, SIM_SUMMING) {
        let base = sim_cells_base(q);
        let first = vec2<u32>(place) * WATER_SIM_GROUP;
        let seconds = state.y * sim_standing(q);
        for (var k = 0u; k < SIM_BLOCK_CELLS; k++) {
            let cell = first + vec2<u32>(k % WATER_SIM_GROUP, k / WATER_SIM_GROUP);
            if cell.x < q.nx && cell.y < q.nz {
                let at = block * SIM_BLOCK_CELLS + k;
                sim_add(q, at, seconds, sim_still(q, at, sim_cells[base + at]));
            }
        }
    }
    if wet {
        sim_steps[entry] = vec2<f32>(1.0, 0.0);
    } else {
        sim_steps[entry] = vec2<f32>(0.0, state.y + 1.0);
    }
}

// A look's sums of a workgroup's cells, and of the water standing in those
// beside blocks the slot's pool does not store.
var<workgroup> sim_look_water: array<vec4<f32>, SIM_BLOCK_CELLS>;
var<workgroup> sim_look_motion: array<vec4<f32>, SIM_BLOCK_CELLS>;
var<workgroup> sim_look_edges: array<vec2<f32>, SIM_BLOCK_CELLS>;

// Whether cell (i, j) stands beside a cell of the rectangle the slot does
// not hold.
fn sim_at_edge(q: Square, i: i32, j: i32) -> bool {
    var edge = false;
    for (var k = 0; k < 4; k++) {
        let beside = vec2<i32>(i, j) + sim_toward(k);
        let inside = all(beside >= vec2<i32>(0)) && beside.x < i32(q.nx) && beside.y < i32(q.nz);
        edge = edge || (inside && sim_at(q, beside.x, beside.y) == SIM_NO_BLOCK);
    }
    return edge;
}

// A look at a summing rectangle, a workgroup for each block of its pool:
// each cell's means over the seconds it was wet since the look before (its
// sums, with its water over the steps its block stood still since it last
// did, standing as it stood), the variances of its depth and its velocity
// about their means (the means of their squares less their means' squares,
// the sums kept above the ground so they keep their precision), and the
// shares of the time it was wet and ran at the fastest water runs, replace
// the means kept then, and its sums start anew; each block sums, over its
// cells flowing (wet SIM_WET_MOST of the time or more), the water their
// means hold (cubic meters a meter of cell,
// above their ground), their speeds, the energy they lose, their heave
// (their depth's root mean square about its mean) and their swirl (their
// vorticity's root mean square), and counts them, those that ran at the
// fastest water runs and its cells stepping (a sparse rectangle's, those
// whose block steps); and the water held by those of its cells beside a
// block the pool does not store, and those cells.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_look(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
) {
    let q = sim_square(group.z);
    let cell = sim_cell_of(q, group, count, local);
    let block = group.y * count.x + group.x;
    let looking = group.z < sim_slot_count() && sim_has(q, SIM_LOOKING)
        && sim_has(q, SIM_SUMMING) && block < q.stored;
    let sparse = sim_has(q, SIM_SPARSE);
    var water_sum = vec4<f32>(0.0);
    var motion_sum = vec4<f32>(0.0);
    var edges = vec2<f32>(0.0);
    if looking && cell.runs {
        let at = cell.at;
        let sums = sim_sums_at(q, at);
        var water = sim_averages[sums];
        var motion = sim_averages[sums + 1u];
        var shares = sim_averages[sums + 2u];
        var spent = sim_averages[sums + 3u];
        var stepping = 1.0;
        if sparse {
            let state = sim_steps[sim_block_at(q, block) + 1u];
            let still = sim_still(q, at, sim_cells[sim_cells_base(q) + at]);
            let seconds = state.y * sim_standing(q);
            water += seconds * still.water;
            motion += seconds * still.motion;
            shares += seconds * still.shares;
            spent.y += seconds;
            stepping = state.x;
        }
        let seconds = max(spent.y, 1.0e-30);
        let wet = max(shares.z, 1.0e-30);
        let mean = water / wet;
        let moving = motion / wet;
        let portions = shares / wet;
        let heave = max(moving.y - mean.x * mean.x, 0.0);
        let rush = max(moving.w - dot(portions.xy, portions.xy), 0.0);
        let means = sim_means_at(q, at);
        sim_averages[means] = vec4<f32>(sim_bed(q, at) + mean.x, mean.yzw);
        sim_averages[means + 1u] = vec4<f32>(moving.x, heave, moving.z, rush);
        sim_averages[means + 2u] = vec4<f32>(shares.z / seconds, portions.w, spent.x / seconds,
            0.0);
        for (var k = 0u; k < SIM_CELL_SUMS; k++) {
            sim_averages[sums + k] = vec4<f32>(0.0);
        }
        let h = mean.x;
        if !sim_held_at(q, at) && h > SIM_DRY && shares.z >= SIM_WET_MOST * seconds {
            let velocity = vec2<f32>(sim_velocity(h, mean.y), sim_velocity(h, mean.z));
            water_sum = vec4<f32>(h, length(velocity), 1.0, stepping);
            motion_sum = vec4<f32>(moving.x, sqrt(heave), sqrt(moving.z),
                select(0.0, 1.0, spent.x > 0.0));
            if sim_at_edge(q, i32(cell.i), i32(cell.j)) {
                edges = vec2<f32>(h, 1.0);
            }
        } else {
            water_sum = vec4<f32>(0.0, 0.0, 0.0, stepping);
        }
    }
    sim_look_water[index] = water_sum;
    sim_look_motion[index] = motion_sum;
    sim_look_edges[index] = edges;
    workgroupBarrier();
    var stride = SIM_BLOCK_CELLS / 2u;
    loop {
        if stride == 0u {
            break;
        }
        if index < stride {
            sim_look_water[index] += sim_look_water[index + stride];
            sim_look_motion[index] += sim_look_motion[index + stride];
            sim_look_edges[index] += sim_look_edges[index + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if index == 0u && looking {
        let at = sim_looks_at(q, block);
        sim_averages[at] = sim_look_water[0];
        sim_averages[at + 1u] = sim_look_motion[0];
        sim_averages[at + 2u] = vec4<f32>(sim_look_edges[0], 0.0, 0.0);
        // The steps it stood still start anew with the sums.
        if sparse {
            sim_steps[sim_block_at(q, block) + 1u].y = 0.0;
        }
    }
}
