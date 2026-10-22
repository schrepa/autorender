// Water's simulation: rivers running over rectangles of ground as the
// shallow-water equations say, on grids of cells, by Kurganov and Petrova's
// well-balanced, positivity-preserving central-upwind scheme (2007): second
// order in space (each cell's water reconstructed linearly, its slopes
// limited by a generalized minmod and kept from dipping below the ground at
// its faces) and in time (Heun's two stages), with Manning's friction,
// velocities desingularized where the water thins out
// (`engine_terrain::shallow_water` holds the constants these mirror). Each
// cell holds its water's surface, its flow along x and z (square meters a
// second) and its foam. The ground stands at the cells' corners, with the
// level a lake holds each cell at, if one does: lakes stand still, taking
// the water that runs into them and giving none. A stage finds the flux
// through every face, each cell reconstructing its water at its own faces
// and at its neighbors' beside them (computing them twice costs less than
// carrying them through memory), and updates every cell; rivers bring their
// water to their cells after each step. The water carries foam: it churns it
// up where it runs faster than its waves and where it slams into slower
// water, the current carries it off, and it fades as its bubbles break.
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
// - the square around the camera (`water_sim.rs`) stores every block and
//   steps every cell by the frame's seconds, and over its edges water flows
//   to and from the cells just past them, which stand as the water settled
//   there (the ring): water left alone stays as it settled;
// - regions settling tiles' water (`water_settle.rs`) step each cell by as
//   long as its own and its neighbors' fastest waves allow, which reaches
//   the steady state in far fewer steps; their water leaves freely over
//   their edges and none comes in; each cell's water is summed over the
//   steps since its region's last look, whose means a look keeps and sums,
//   block by block, into how far they moved since the look before; and only
//   the blocks water can reach this step step: a block steps where it held
//   water no lake holds at the step before's end, or where a block beside it
//   held water along the side they share, as water crosses less than a cell
//   a step, and every other block stands still, dry or a lake's, the faces
//   between them and those stepping carrying nothing.

const WATER_SIM_GROUP: u32 = 8u;
const WATER_SIM_SOURCE_GROUP: u32 = 64u;
// The most rectangles a dispatch runs.
const SIM_SLOTS: u32 = 64u;
// What a slot's table holds for a block its pool does not store; a block's
// cells and corners; and the entries of a block's ground: the ground at its
// cells' corners, then the level a lake holds each of its cells at.
const SIM_NO_BLOCK: u32 = 0xffffffffu;
const SIM_BLOCK_CELLS: u32 = WATER_SIM_GROUP * WATER_SIM_GROUP;
const SIM_BLOCK_CORNERS: u32 = (WATER_SIM_GROUP + 1u) * (WATER_SIM_GROUP + 1u);
const SIM_BLOCK_GROUND: u32 = SIM_BLOCK_CORNERS + SIM_BLOCK_CELLS;
const SIM_GRAVITY: f32 = 9.81;
// The generalized minmod's theta; the depth below which velocities are
// desingularized and the depth under which a cell is dry (meters); and the
// fastest water runs (meters a second).
const SIM_THETA: f32 = 1.3;
const SIM_THIN: f32 = 0.01;
const SIM_DRY: f32 = 1.0e-4;
const SIM_FASTEST: f32 = 20.0;
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
// blocks a slot's pool holds and the blocks a slot's table maps; as the
// square around the camera follows it, the cells it moved by along x and
// z; and the most cells rivers bring water to a slot holds, and the faces
// along its west and south edges a slot holds (its cells along x and z).
struct SimFrame {
    stage: vec4<u32>,
    shift: vec4<i32>,
    inputs: vec4<u32>,
}

// A slot's rectangle: its cells along x and z, its flags and the cells
// its rivers bring water to; the meters across a cell, the seconds of a
// step stepping every cell alike, Manning's roughness and the steps its
// sums hold at a look; and its blocks along x and z and those its pool
// stores.
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
// where none does), row by row.
@group(0) @binding(1) var<storage, read> sim_ground: array<f32>;
// Each slot's step's start and then its first stage, each its pool's
// blocks' cells, a block's row by row: the water's surface, its flow along
// x and z, and its foam (0 none, 1 white all over; in the first stage, once
// the step churned it, its next).
@group(0) @binding(2) var<storage, read_write> sim_cells: array<vec4<f32>>;
// Each cell's water's depth at its faces as reconstructed, the mean of its
// west and east faces' and of its south and north faces', which the pull of
// gravity down its ground reads.
@group(0) @binding(3) var<storage, read_write> sim_depths: array<vec2<f32>>;
// The fluxes through each slot's faces: each of its cells' east face's,
// then each of its cells' north face's, then those through the faces on its
// rectangle's west edge (a row each) and on its south edge (a column each):
// its mass and its momentum across and along the face, and the fastest wave
// through it.
@group(0) @binding(4) var<storage, read_write> sim_fluxes: array<vec4<f32>>;
// What rivers bring and what stands past the edges: each slot's cells
// rivers bring water to (the cell, its index among the slot's cells as
// bits; the cubic meters a second they bring; and the way the water runs as
// it comes, x and z, a share of SIM_SOURCE_SPEED); after every slot's, each
// slot's ring, the cells just past its edges as their water settled, two
// entries each: its surface (far below the ground where dry), its flow
// along x and z and the level a lake holds it at (SIM_UNHELD where none
// does); then its foam. South then north of each column, then west then
// east of each row.
@group(0) @binding(5) var<storage, read> sim_inputs: array<vec4<f32>>;
// Each slot's table: the block of its pool storing each block of its
// rectangle, row by row (SIM_NO_BLOCK where none does); then each block of
// its pool's place in the rectangle (its column, and its row above 16
// bits).
@group(0) @binding(6) var<storage, read> sim_blocks: array<u32>;
// Each slot's cells' fastest wave through their faces this step and their
// seconds; then each of its pool's blocks': whether it held water no lake
// holds at the step before's end, and whether it steps this step and the
// steps it stood still since its rectangle's last look or its waking,
// whichever came later.
@group(0) @binding(7) var<storage, read_write> sim_steps: array<vec2<f32>>;
@group(0) @binding(8) var<uniform> sim_slots: SimSlots;
// Each slot's cells' water summed over its steps since its last look
// (surface, flow along x and z, foam), then their means at that look; then
// each of its pool's blocks' sums at a look, two entries each: the water it
// held, how far its cells' velocities moved since the look before, its
// cells flowing and its cells stepping; and the water its cells beside
// blocks the pool does not store held, and those cells.
@group(0) @binding(9) var<storage, read_write> sim_averages: array<vec4<f32>>;

// A slot's rectangle as a dispatch runs it.
struct Square {
    slot: u32,
    nx: u32,
    nz: u32,
    flags: u32,
    sourced: u32,
    dx: f32,
    live: f32,
    roughness: f32,
    summed: f32,
    blocks: vec2<u32>,
    stored: u32,
}

fn sim_square(slot: u32) -> Square {
    let held = sim_slots.slot[slot];
    return Square(slot, held.shape.x, held.shape.y, held.shape.z, held.shape.w, held.step.x,
        held.step.y, held.step.z, held.step.w, held.blocks.xy, held.blocks.z);
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
    return q.slot * (2u * sim_held() + 2u * sim_pool());
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

// The ground at the middles of the faces across x of the cell at `at`, its
// west face's and its east face's; and of those across z, its south face's
// and its north face's.
fn sim_ground_x(q: Square, at: u32) -> vec2<f32> {
    let block = at / SIM_BLOCK_CELLS;
    let x = at % WATER_SIM_GROUP;
    let z = (at % SIM_BLOCK_CELLS) / WATER_SIM_GROUP;
    return vec2<f32>(0.5 * (sim_corner(q, block, x, z) + sim_corner(q, block, x, z + 1u)),
        0.5 * (sim_corner(q, block, x + 1u, z) + sim_corner(q, block, x + 1u, z + 1u)));
}

fn sim_ground_z(q: Square, at: u32) -> vec2<f32> {
    let block = at / SIM_BLOCK_CELLS;
    let x = at % WATER_SIM_GROUP;
    let z = (at % SIM_BLOCK_CELLS) / WATER_SIM_GROUP;
    return vec2<f32>(0.5 * (sim_corner(q, block, x, z) + sim_corner(q, block, x + 1u, z)),
        0.5 * (sim_corner(q, block, x, z + 1u) + sim_corner(q, block, x + 1u, z + 1u)));
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

// The cell at `at` of the state `state` (0: the step's start, 1: its first
// stage); a cell standing still stands as the step started.
fn sim_cell(q: Square, state: u32, at: u32) -> vec4<f32> {
    let standing = select(0u, state, sim_active(q, at));
    return sim_cells[sim_cells_base(q) + standing * sim_held() + at];
}

// Cell (i, j) of the state `state`, or `instead` where the slot holds none
// there: past the rectangle's edge, or in a block its pool does not store.
fn sim_beside(q: Square, state: u32, i: i32, j: i32, instead: vec4<f32>) -> vec4<f32> {
    let at = sim_at(q, i, j);
    if at == SIM_NO_BLOCK {
        return instead;
    }
    return sim_cell(q, state, at);
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

// The faces across x of the cell at `at`, cell (i, j), from the state
// `state`: its slopes limited by its neighbors' (itself standing for one
// the slot does not hold), its water kept above the ground at both faces.
fn sim_across_x(q: Square, state: u32, at: u32, i: u32, j: u32) -> SimSides {
    let here = sim_cell(q, state, at);
    let west = sim_beside(q, state, i32(i) - 1, i32(j), here);
    let east = sim_beside(q, state, i32(i) + 1, i32(j), here);
    let ground = sim_ground_x(q, at);
    let slope = vec3<f32>(sim_half_slope(west.x, here.x, east.x),
        sim_half_slope(west.y, here.y, east.y), sim_half_slope(west.z, here.z, east.z));
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

// The faces across z of the cell at `at`, cell (i, j), the flow along z
// across them.
fn sim_across_z(q: Square, state: u32, at: u32, i: u32, j: u32) -> SimSides {
    let here = sim_cell(q, state, at);
    let south = sim_beside(q, state, i32(i), i32(j) - 1, here);
    let north = sim_beside(q, state, i32(i), i32(j) + 1, here);
    let ground = sim_ground_z(q, at);
    let slope = vec3<f32>(sim_half_slope(south.x, here.x, north.x),
        sim_half_slope(south.y, here.y, north.y), sim_half_slope(south.z, here.z, north.z));
    var high_w = here.x + slope.x;
    var low_w = here.x - slope.x;
    if high_w < ground.y {
        high_w = ground.y;
        low_w = 2.0 * here.x - ground.y;
    } else if low_w < ground.x {
        low_w = ground.x;
        high_w = 2.0 * here.x - ground.x;
    }
    return SimSides(sim_face(low_w, ground.x, here.z - slope.z, here.y - slope.y),
        sim_face(high_w, ground.y, here.z + slope.z, here.y + slope.y));
}

// The central-upwind flux through a face whose water stands as `l` on its
// near side and `r` on its far one: its mass, and its momentum across and
// along the face; and the fastest wave through it.
fn sim_flux(l: vec4<f32>, r: vec4<f32>) -> vec4<f32> {
    let cl = sqrt(SIM_GRAVITY * l.y);
    let cr = sqrt(SIM_GRAVITY * r.y);
    let fastest = max(max(l.z + cl, r.z + cr), 0.0);
    let slowest = min(min(l.z - cl, r.z - cr), 0.0);
    let span = fastest - slowest;
    let wave = max(fastest, -slowest);
    if span < 1.0e-6 {
        return vec4<f32>(0.0, 0.0, 0.0, wave);
    }
    let fl = vec3<f32>(l.y * l.z, l.y * l.z * l.z + 0.5 * SIM_GRAVITY * l.y * l.y,
        l.y * l.z * l.w);
    let fr = vec3<f32>(r.y * r.z, r.y * r.z * r.z + 0.5 * SIM_GRAVITY * r.y * r.y,
        r.y * r.z * r.w);
    let ul = vec3<f32>(l.x, l.y * l.z, l.y * l.w);
    let ur = vec3<f32>(r.x, r.y * r.z, r.y * r.w);
    let flux = (fastest * fl - slowest * fr) / span + (fastest * slowest / span) * (ur - ul);
    return vec4<f32>(flux, wave);
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

// The flux through a face on the square's edge from what stands as `l` to
// what stands as `r`, one the inside's face and one a ring cell's, whether
// each is held by a lake: none where it would carry water out of one.
fn sim_edge(l: vec4<f32>, r: vec4<f32>, near_held: bool, far_held: bool) -> vec4<f32> {
    let flux = sim_flux(l, r);
    if (flux.x > 0.0 && near_held) || (flux.x < 0.0 && far_held) {
        return vec4<f32>(0.0, 0.0, 0.0, flux.w);
    }
    return flux;
}

// The flux through a face on an open rectangle's edge whose inside stands
// as `face`, `inward` the way into the rectangle across it (+1 or -1):
// water leaves as it runs, the inside standing on both sides, and none
// comes in.
fn sim_open(face: vec4<f32>, inward: f32) -> vec4<f32> {
    let flux = sim_flux(face, face);
    if flux.x * inward > 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, flux.w);
    }
    return flux;
}

// Each cell that steps, and each beside one that does, finds the fluxes
// through its east and north faces (and its west and south ones on the
// rectangle's edges): between two cells that step, as their faces say; on
// the edges, open or between the inside and the ring cell past it; toward
// a cell the slot does not hold, open (`sim_open_west` finds the west and
// south ones there); and between a cell that steps and one standing still,
// nothing, as no water reaches either this step. A cell that steps keeps
// its faces' depths.
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
        sim_depths[sim_depths_base(q) + at] =
            vec2<f32>(0.5 * (x.low.y + x.high.y), 0.5 * (z.low.y + z.high.y));
    }
    // Its east face: between columns i and i + 1, or the east edge.
    if steps || east_steps {
        var flux = vec4<f32>(0.0);
        if i + 1u == nx {
            if open {
                flux = sim_open(x.high, -1.0);
            } else {
                let ghost = 2u * nx + 2u * j + 1u;
                flux = sim_edge(x.high, sim_ghost_face(q, ghost, sim_ground_x(q, at).y, false),
                    sim_held_at(q, at), sim_ghost_held(q, ghost));
            }
        } else if steps && east_steps {
            let far = sim_across_x(q, state, east, i + 1u, j).low;
            flux = sim_from_lakes(q, sim_flux(x.high, far), at, east);
        } else if steps && east == SIM_NO_BLOCK {
            flux = sim_open(x.high, -1.0);
        }
        sim_fluxes[fluxes + at] = flux;
    }
    // The west edge.
    if i == 0u && steps {
        var flux: vec4<f32>;
        if open {
            flux = sim_open(x.low, 1.0);
        } else {
            let ghost = 2u * nx + 2u * j;
            flux = sim_edge(sim_ghost_face(q, ghost, sim_ground_x(q, at).x, false), x.low,
                sim_ghost_held(q, ghost), sim_held_at(q, at));
        }
        sim_fluxes[edges + j] = flux;
    }
    // Its north face: between rows j and j + 1, or the north edge.
    if steps || north_steps {
        var flux = vec4<f32>(0.0);
        if j + 1u == nz {
            if open {
                flux = sim_open(z.high, -1.0);
            } else {
                let ghost = 2u * i + 1u;
                flux = sim_edge(z.high, sim_ghost_face(q, ghost, sim_ground_z(q, at).y, true),
                    sim_held_at(q, at), sim_ghost_held(q, ghost));
            }
        } else if steps && north_steps {
            let far = sim_across_z(q, state, north, i, j + 1u).low;
            flux = sim_from_lakes(q, sim_flux(z.high, far), at, north);
        } else if steps && north == SIM_NO_BLOCK {
            flux = sim_open(z.high, -1.0);
        }
        sim_fluxes[fluxes + sim_held() + at] = flux;
    }
    // The south edge.
    if j == 0u && steps {
        var flux: vec4<f32>;
        if open {
            flux = sim_open(z.low, 1.0);
        } else {
            let ghost = 2u * i;
            flux = sim_edge(sim_ghost_face(q, ghost, sim_ground_z(q, at).x, true), z.low,
                sim_ghost_held(q, ghost), sim_held_at(q, at));
        }
        sim_fluxes[edges + nz + i] = flux;
    }
}

// The flux out through the west face (`across_z` false) or the south one
// (true) of the cell at `at`, cell (i, j), toward a cell the slot does not
// hold: open, as the rectangle's edges are. The cells beside it keep no
// face toward it, so it finds that face again.
fn sim_open_west(q: Square, at: u32, i: u32, j: u32, across_z: bool) -> vec4<f32> {
    if across_z {
        return sim_open(sim_across_z(q, frame.stage.x, at, i, j).low, 1.0);
    }
    return sim_open(sim_across_x(q, frame.stage.x, at, i, j).low, 1.0);
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

// A cell's water `water` (surface and flow) settled over `ground` after a
// stage of `dt` seconds: never below the ground, slowed by friction, no
// faster than SIM_FASTEST, reset where it stopped being finite, and held
// at its lake's level `held` where one holds it.
fn sim_settle(q: Square, water: vec3<f32>, ground: f32, held: f32, dt: f32) -> vec3<f32> {
    if held > 0.5 * SIM_UNHELD {
        return vec3<f32>(held, 0.0, 0.0);
    }
    var next = water;
    if !(abs(next.x) < 1.0e20 && abs(next.y) < 1.0e20 && abs(next.z) < 1.0e20) {
        next = vec3<f32>(ground, 0.0, 0.0);
    }
    next.x = max(next.x, ground);
    let h = next.x - ground;
    if h <= SIM_DRY {
        return vec3<f32>(next.x, 0.0, 0.0);
    }
    var flow = next.yz;
    let friction = SIM_GRAVITY * q.roughness * q.roughness * (length(flow) / h)
        / pow(h, 4.0 / 3.0);
    flow = flow / (1.0 + dt * friction);
    let speed = length(flow) / h;
    if speed > SIM_FASTEST {
        flow *= SIM_FASTEST / speed;
    }
    return vec3<f32>(next.x, flow);
}

// Each cell's water after a stage, its foam as the step started; in the
// first, a sparse rectangle's blocks start the step dry, their cells'
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
    let held = sim_held();
    let dx = q.dx;
    let dt = sim_seconds_at(q, at);
    let west = sim_west(q, at, cell.i, cell.j).xyz;
    let east = sim_east(q, at).xyz;
    let south = sim_south(q, at, cell.i, cell.j).xyz;
    let north = sim_north(q, at).xyz;
    let depths = sim_depths[sim_depths_base(q) + at];
    let depth_x = depths.x;
    let depth_z = depths.y;
    let ground_x = sim_ground_x(q, at);
    let ground_z = sim_ground_z(q, at);
    let slope_x = (ground_x.y - ground_x.x) / dx;
    let slope_z = (ground_z.y - ground_z.x) / dx;
    // The flux through faces between rows carries the flow along z across
    // them and the flow along x along them.
    let change = vec3<f32>(
        -(east.x - west.x) / dx - (north.x - south.x) / dx,
        -(east.y - west.y) / dx - (north.z - south.z) / dx - SIM_GRAVITY * depth_x * slope_x,
        -(east.z - west.z) / dx - (north.y - south.y) / dx - SIM_GRAVITY * depth_z * slope_z);
    let base = sim_cells_base(q);
    let start = sim_cells[base + at];
    var next = start.xyz + dt * change;
    if frame.stage.x == 1u {
        next = 0.5 * (start.xyz + sim_cells[base + held + at].xyz + dt * change);
    }
    let ground = 0.5 * (ground_x.x + ground_x.y);
    let written = select(base + held + at, base + at, frame.stage.x == 1u);
    sim_cells[written] = vec4<f32>(sim_settle(q, next, ground, sim_lake(q, at), dt), start.w);
    let first = at % SIM_BLOCK_CELLS == 0u;
    if frame.stage.x == 0u && sim_has(q, SIM_SPARSE) && first {
        sim_steps[sim_block_at(q, at / SIM_BLOCK_CELLS)].x = 0.0;
    }
}

// The square followed the camera by `frame.shift` cells: each cell takes
// the water and foam of the cell standing where it does in the square as it
// was, where that one held any, into the second state, which holds the
// water as it settled where it held none.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_carry(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let q = sim_square(group.z);
    let cell = sim_cell_of(q, group, count, local);
    if !cell.runs {
        return;
    }
    let from_cell = sim_at(q, i32(cell.i) + frame.shift.x, i32(cell.j) + frame.shift.y);
    if from_cell == SIM_NO_BLOCK {
        return;
    }
    let base = sim_cells_base(q);
    sim_cells[base + sim_held() + cell.at] = sim_cells[base + from_cell];
}

// The water carried over is what the steps start from.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_restart(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let q = sim_square(group.z);
    let cell = sim_cell_of(q, group, count, local);
    if !cell.runs {
        return;
    }
    let base = sim_cells_base(q);
    sim_cells[base + cell.at] = sim_cells[base + sim_held() + cell.at];
}

// Foam: the Froude number past which rapids churn it up and how fast (a
// second, per unit past it); how fast water converging churns it (per
// unit of convergence, a second); the seconds it takes to fade to 1/e;
// the most it piles up to; and the meters of water under which a cell
// holds none.
const SIM_FOAM_FROUDE: f32 = 2.0;
const SIM_FOAM_RAPIDS: f32 = 0.3;
const SIM_FOAM_PILING: f32 = 0.25;
const SIM_FOAM_LIFE: f32 = 6.0;
const SIM_FOAM_MOST: f32 = 1.5;
const SIM_FOAM_WET: f32 = 0.005;

// The first state's cell at `at`: its depth and velocity; or `instead`'s
// where it is dry.
fn sim_moving_at(q: Square, at: u32, instead: vec3<f32>) -> vec3<f32> {
    let cell = sim_cells[sim_cells_base(q) + at];
    let h = cell.x - sim_bed(q, at);
    if h <= SIM_FOAM_WET {
        return instead;
    }
    return vec3<f32>(h, sim_velocity(h, cell.y), sim_velocity(h, cell.z));
}

// The first state's cell (i, j): its depth and velocity; or `instead`'s
// where it is dry or the slot holds none there.
fn sim_moving(q: Square, i: i32, j: i32, instead: vec3<f32>) -> vec3<f32> {
    let at = sim_at(q, i, j);
    if at == SIM_NO_BLOCK {
        return instead;
    }
    return sim_moving_at(q, at, instead);
}

// The first state's foam at cell (i, j): none where the slot holds none
// there.
fn sim_foam(q: Square, i: u32, j: u32) -> f32 {
    let at = sim_at(q, i32(i), i32(j));
    if at == SIM_NO_BLOCK {
        return 0.0;
    }
    return sim_cells[sim_cells_base(q) + at].w;
}

// The first state's foam at cell `p` (fractional, along x and z),
// bilinearly; past the rectangle's edges, the nearest ring cell's, or none
// for an open rectangle's.
fn sim_foam_at(q: Square, p: vec2<f32>) -> f32 {
    let nx = q.nx;
    let last = vec2<f32>(f32(nx - 1u), f32(q.nz - 1u));
    let near = vec2<u32>(clamp(round(p), vec2<f32>(0.0), last));
    let ring = sim_ring_base(q);
    let outside = p.x < -0.5 || p.x > last.x + 0.5 || p.y < -0.5 || p.y > last.y + 0.5;
    if outside && sim_has(q, SIM_OPEN) {
        return 0.0;
    }
    if p.x < -0.5 {
        return sim_inputs[ring + 2u * (2u * nx + 2u * near.y) + 1u].x;
    }
    if p.x > last.x + 0.5 {
        return sim_inputs[ring + 2u * (2u * nx + 2u * near.y + 1u) + 1u].x;
    }
    if p.y < -0.5 {
        return sim_inputs[ring + 2u * (2u * near.x) + 1u].x;
    }
    if p.y > last.y + 0.5 {
        return sim_inputs[ring + 2u * (2u * near.x + 1u) + 1u].x;
    }
    let c = clamp(p, vec2<f32>(0.0), last);
    let low = vec2<u32>(floor(c));
    let high = min(low + 1u, vec2<u32>(nx - 1u, q.nz - 1u));
    let f = c - floor(c);
    let below = mix(sim_foam(q, low.x, low.y), sim_foam(q, high.x, low.y), f.x);
    let above = mix(sim_foam(q, low.x, high.y), sim_foam(q, high.x, high.y), f.x);
    return mix(below, above, f.y);
}

// Each cell's foam after a step, into the second state's: what the current
// carried there (traced back along it), faded, and what the water churned
// up there.
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
    let here = sim_moving_at(q, cell.at, vec3<f32>(0.0));
    if here.x <= SIM_FOAM_WET {
        sim_cells[next].w = 0.0;
        return;
    }
    let dx = q.dx;
    let dt = sim_seconds_at(q, cell.at);
    let carried = sim_foam_at(q, vec2<f32>(f32(i), f32(j)) - here.yz * dt / dx);
    let froude = length(here.yz) / sqrt(SIM_GRAVITY * here.x);
    let rapids = max(froude - SIM_FOAM_FROUDE, 0.0) * SIM_FOAM_RAPIDS;
    let east = sim_moving(q, i + 1, j, here).y;
    let west = sim_moving(q, i - 1, j, here).y;
    let north = sim_moving(q, i, j + 1, here).z;
    let south = sim_moving(q, i, j - 1, here).z;
    let converging = max(-(east - west + north - south) / (2.0 * dx), 0.0);
    let churned = carried * exp(-dt / SIM_FOAM_LIFE)
        + dt * (rapids + SIM_FOAM_PILING * converging);
    sim_cells[next].w = min(churned, SIM_FOAM_MOST);
}

// Cell `at`'s water as its sums add it: its surface as meters above its
// ground, so sums over many steps keep the precision a level hundreds of
// meters up would lose, the mean adding its ground back.
fn sim_above(q: Square, at: u32, water: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(water.x - sim_bed(q, at), water.yzw);
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

// The foam after the step is the next one's start; a summing rectangle
// adds each cell's water and foam to its sums; and a sparse one's cells
// holding water no lake holds mark their block, and the sides of it they
// lie along.
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
        if sim_has(q, SIM_SUMMING) {
            sim_averages[sim_averages_base(q) + at] += sim_above(q, at, water);
        }
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
// they share; else it stands still, counting the steps it stands. A block waking adds to its cells' sums their water
// over the steps it stood still, standing as it stood.
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
        let sums = sim_averages_base(q);
        let first = vec2<u32>(place) * WATER_SIM_GROUP;
        for (var k = 0u; k < SIM_BLOCK_CELLS; k++) {
            let cell = first + vec2<u32>(k % WATER_SIM_GROUP, k / WATER_SIM_GROUP);
            if cell.x < q.nx && cell.y < q.nz {
                let at = block * SIM_BLOCK_CELLS + k;
                sim_averages[sums + at] += state.y * sim_above(q, at, sim_cells[base + at]);
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
var<workgroup> sim_look_sums: array<vec4<f32>, SIM_BLOCK_CELLS>;
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
// each cell's mean water over the steps since the look before (its sums
// over `summed` steps, with its water over the steps its block stood still
// since it last did, standing as it stood) replaces the mean kept then,
// and its sums start anew; each block sums the water its cells' means hold
// (cubic meters a meter of cell, above their ground), how far the
// velocities of those flowing moved since the mean before, those flowing
// and those stepping (a sparse rectangle's, those whose block steps); and
// the water held by those of its cells beside a block the pool does not
// store, and those cells.
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
    var sums = vec4<f32>(0.0);
    var edges = vec2<f32>(0.0);
    if looking && cell.runs {
        let at = cell.at;
        let held = sim_held();
        let averages = sim_averages_base(q);
        let now = sim_cells[sim_cells_base(q) + at];
        let ground = sim_bed(q, at);
        var summed = sim_averages[averages + at];
        var stepping = 1.0;
        if sparse {
            let state = sim_steps[sim_block_at(q, block) + 1u];
            summed += state.y * sim_above(q, at, now);
            stepping = state.x;
        }
        var mean = summed / max(q.summed, 1.0);
        mean.x += ground;
        let before = sim_averages[averages + held + at];
        sim_averages[averages + held + at] = mean;
        sim_averages[averages + at] = vec4<f32>(0.0);
        let h = mean.x - ground;
        let was = before.x - ground;
        if !sim_held_at(q, at) && h > SIM_DRY {
            let velocity = vec2<f32>(sim_velocity(h, mean.y), sim_velocity(h, mean.z));
            var earlier = vec2<f32>(0.0);
            if was > SIM_DRY {
                earlier = vec2<f32>(sim_velocity(was, before.y), sim_velocity(was, before.z));
            }
            sums = vec4<f32>(h, length(velocity - earlier), 1.0, stepping);
            if sim_at_edge(q, i32(cell.i), i32(cell.j)) {
                edges = vec2<f32>(h, 1.0);
            }
        } else {
            sums = vec4<f32>(0.0, 0.0, 0.0, stepping);
        }
    }
    sim_look_sums[index] = sums;
    sim_look_edges[index] = edges;
    workgroupBarrier();
    var stride = SIM_BLOCK_CELLS / 2u;
    loop {
        if stride == 0u {
            break;
        }
        if index < stride {
            sim_look_sums[index] += sim_look_sums[index + stride];
            sim_look_edges[index] += sim_look_edges[index + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if index == 0u && looking {
        let at = sim_averages_base(q) + 2u * sim_held() + 2u * block;
        sim_averages[at] = sim_look_sums[0];
        sim_averages[at + 1u] = vec4<f32>(sim_look_edges[0], 0.0, 0.0);
        // The steps it stood still start anew with the sums.
        if sparse {
            sim_steps[sim_block_at(q, block) + 1u].y = 0.0;
        }
    }
}
