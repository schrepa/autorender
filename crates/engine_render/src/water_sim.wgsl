// Water's simulation: rivers running over squares of ground as the
// shallow-water equations say, on grids of cells, by Kurganov and Petrova's
// well-balanced, positivity-preserving central-upwind scheme (2007): second
// order in space (each cell's water reconstructed linearly, its slopes
// limited by a generalized minmod and kept from dipping below the ground at
// its faces) and in time (Heun's two stages), with Manning's friction,
// velocities desingularized where the water thins out
// (`engine_terrain::shallow_water` holds the constants these mirror). Each
// cell holds its water's surface, its flow along x and z (square meters a
// second) and the level a lake holds it at, if one does: lakes stand still,
// taking the water that runs into them and giving none. The ground stands at
// the cells' corners. A stage reconstructs each cell's faces, finds the flux
// through every face and updates every cell; rivers bring their water to
// their cells after each step. The water carries foam: it churns it up
// where it runs faster than its waves and where it slams into slower water,
// the current carries it off, and it fades as its bubbles break.
//
// Every dispatch runs squares side by side, the third dimension counting
// them, each a slot of the buffers with its own cells along a side, cell
// size, steps and edges (`SimSlot`):
// - the square around the camera (`water_sim.rs`) steps every cell by the
//   frame's seconds, and over its edges water flows to and from the cells
//   just past them, which stand as the water settled there (the ring):
//   water left alone stays as it settled;
// - squares settling a tile's water (`water_settle.rs`) step each cell by as
//   long as its own and its neighbors' fastest waves allow, which reaches
//   the steady state in far fewer steps; their water leaves freely over
//   their edges and none comes in; and each cell's water is summed over
//   the steps since its square's last look, whose means a look keeps and
//   sums, workgroup by workgroup, into how far they moved since the look
//   before.

const WATER_SIM_GROUP: u32 = 8u;
const WATER_SIM_SOURCE_GROUP: u32 = 64u;
// The most squares a dispatch runs.
const SIM_SLOTS: u32 = 64u;
const SIM_GRAVITY: f32 = 9.81;
// The generalized minmod's theta; the depth below which velocities are
// desingularized and the depth under which a cell is dry (meters); and the
// fastest water runs (meters a second).
const SIM_THETA: f32 = 1.3;
const SIM_THIN: f32 = 0.01;
const SIM_DRY: f32 = 1.0e-4;
const SIM_FASTEST: f32 = 20.0;
// The meters a second the water rivers bring runs as it comes, what a cell
// no lake holds holds as its lake's level, and what a dry cell past the
// edges stands at.
const SIM_SOURCE_SPEED: f32 = 1.5;
const SIM_UNHELD: f32 = -1.0e30;
const SIM_GHOST_DRY: f32 = -1.0e30;
// Local steps: the share of a cell the fastest wave crosses in one, the
// share of that they take, and the slowest wave they count (meters a
// second).
const SIM_COURANT: f32 = 0.24;
const SIM_LOCAL_SHARE: f32 = 0.9;
const SIM_SLOWEST: f32 = 1.0;
// A square's flags: it steps; each cell by its own seconds; water leaves
// over its edges freely and none comes in (else the ring stands past
// them); it sums each cell's water over its steps; and this dispatch looks
// at its sums.
const SIM_STEPPING: u32 = 1u;
const SIM_LOCAL: u32 = 2u;
const SIM_OPEN: u32 = 4u;
const SIM_SUMMING: u32 = 8u;
const SIM_LOOKING: u32 = 16u;

// The stage (0: the first, 1: the second), the most cells along a side a
// slot holds and the slots the buffers hold; and, as the square around the
// camera follows it, the cells it moved by along x and z.
struct SimFrame {
    stage: vec4<u32>,
    shift: vec4<i32>,
}

// A slot's square: its cells along a side, its flags and the cells its
// rivers bring water to; and the meters across a cell, the seconds of a
// step stepping every cell alike, Manning's roughness and the steps its
// sums hold at a look.
struct SimSlot {
    shape: vec4<u32>,
    step: vec4<f32>,
}

struct SimSlots {
    slot: array<SimSlot, SIM_SLOTS>,
}

@group(0) @binding(0) var<uniform> frame: SimFrame;
// The ground at each slot's cells' corners (meters), cells + 1 a row, rows
// along z.
@group(0) @binding(1) var<storage, read> sim_ground: array<f32>;
// Each slot's step's start and then its first stage, each a row of cells at
// a time: the water's surface, its flow along x and z, and the level a lake
// holds it at (SIM_UNHELD where none does).
@group(0) @binding(2) var<storage, read_write> sim_cells: array<vec4<f32>>;
// Each cell's faces as reconstructed, east, west, north and south: the
// water's surface and depth there, and its velocity across the face and
// along it.
@group(0) @binding(3) var<storage, read_write> sim_faces: array<vec4<f32>>;
// The fluxes through each slot's faces between columns (cells + 1 a row),
// then through those between rows (cells a row): its mass and its momentum
// across and along the face, and the fastest wave through it.
@group(0) @binding(4) var<storage, read_write> sim_fluxes: array<vec4<f32>>;
// What rivers bring and what stands past the edges: each slot's cells
// rivers bring water to (the cell, its index's bits; the cubic meters a
// second they bring; and the way the water runs as it comes, x and z, a
// share of SIM_SOURCE_SPEED); after every slot's, each slot's ring, the
// cells just past its edges as their water settled, two entries each: its
// surface (SIM_GHOST_DRY where dry), its flow along x and z and the level a
// lake holds it at (SIM_UNHELD where none does); then its foam. South then
// north of each column, then west then east of each row.
@group(0) @binding(5) var<storage, read> sim_inputs: array<vec4<f32>>;
// The foam the water carries (0 none, 1 white all over), each slot's step's
// start and then its next, each a row of cells at a time.
@group(0) @binding(6) var<storage, read_write> sim_foam: array<f32>;
// Each cell's fastest wave through its faces this step, and its seconds.
@group(0) @binding(7) var<storage, read_write> sim_steps: array<vec2<f32>>;
@group(0) @binding(8) var<uniform> sim_slots: SimSlots;
// Each slot's cells' water summed over its steps since its last look
// (surface, flow along x and z, foam), then their means at that look; then
// each of its workgroups' sums at a look: the water it held, how far its
// cells' velocities moved since the look before, its cells flowing, and
// its cells stepping.
@group(0) @binding(9) var<storage, read_write> sim_averages: array<vec4<f32>>;

// A slot's square as a dispatch runs it.
struct Square {
    slot: u32,
    n: u32,
    flags: u32,
    sourced: u32,
    dx: f32,
    live: f32,
    roughness: f32,
    summed: f32,
}

fn sim_square(slot: u32) -> Square {
    let held = sim_slots.slot[slot];
    return Square(slot, held.shape.x, held.shape.y, held.shape.z, held.step.x, held.step.y,
        held.step.z, held.step.w);
}

fn sim_has(q: Square, flag: u32) -> bool {
    return (q.flags & flag) != 0u;
}

// Whether invocation `id` stands on a cell of a square that steps.
fn sim_runs(q: Square, id: vec3<u32>) -> bool {
    return id.z < frame.stage.z && sim_has(q, SIM_STEPPING) && id.x < q.n && id.y < q.n;
}

// Where each slot's entries start, from the most cells along a side a slot
// holds.
fn sim_capacity() -> u32 {
    return frame.stage.y;
}

fn sim_cells_base(q: Square) -> u32 {
    return q.slot * 2u * sim_capacity() * sim_capacity();
}

fn sim_ground_base(q: Square) -> u32 {
    return q.slot * (sim_capacity() + 1u) * (sim_capacity() + 1u);
}

fn sim_faces_base(q: Square) -> u32 {
    return q.slot * 4u * sim_capacity() * sim_capacity();
}

fn sim_fluxes_base(q: Square) -> u32 {
    return q.slot * 2u * (sim_capacity() + 1u) * sim_capacity();
}

fn sim_sources_base(q: Square) -> u32 {
    return q.slot * sim_capacity() * sim_capacity();
}

fn sim_ring_base(q: Square) -> u32 {
    return frame.stage.z * sim_capacity() * sim_capacity() + q.slot * 8u * sim_capacity();
}

fn sim_steps_base(q: Square) -> u32 {
    return q.slot * sim_capacity() * sim_capacity();
}

// Workgroups along a side of the largest square, and each slot's sums.
fn sim_groups() -> u32 {
    return (sim_capacity() + WATER_SIM_GROUP - 1u) / WATER_SIM_GROUP;
}

fn sim_averages_base(q: Square) -> u32 {
    return q.slot * (2u * sim_capacity() * sim_capacity() + sim_groups() * sim_groups());
}

fn sim_corner(q: Square, i: u32, j: u32) -> f32 {
    return sim_ground[sim_ground_base(q) + j * (q.n + 1u) + i];
}

// The ground at the middle of the face between columns i - 1 and i of row
// j, and of the face between rows j - 1 and j of column i.
fn sim_ground_x(q: Square, i: u32, j: u32) -> f32 {
    return 0.5 * (sim_corner(q, i, j) + sim_corner(q, i, j + 1u));
}

fn sim_ground_z(q: Square, i: u32, j: u32) -> f32 {
    return 0.5 * (sim_corner(q, i, j) + sim_corner(q, i + 1u, j));
}

// Cell (i, j) of the state `state` (0: the step's start, 1: its first
// stage), the nearest inside the square for one past its edge.
fn sim_cell(q: Square, state: u32, i: i32, j: i32) -> vec4<f32> {
    let n = i32(q.n);
    let at = vec2<i32>(clamp(i, 0, n - 1), clamp(j, 0, n - 1));
    return sim_cells[sim_cells_base(q) + state * q.n * q.n + u32(at.y * n + at.x)];
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

@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_reconstruct(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = sim_square(id.z);
    if !sim_runs(q, id) {
        return;
    }
    let state = frame.stage.x;
    let i = i32(id.x);
    let j = i32(id.y);
    let here = sim_cell(q, state, i, j);
    let west = sim_cell(q, state, i - 1, j);
    let east = sim_cell(q, state, i + 1, j);
    let south = sim_cell(q, state, i, j - 1);
    let north = sim_cell(q, state, i, j + 1);
    let ground_x = vec2<f32>(sim_ground_x(q, id.x, id.y), sim_ground_x(q, id.x + 1u, id.y));
    let ground_z = vec2<f32>(sim_ground_z(q, id.x, id.y), sim_ground_z(q, id.x, id.y + 1u));
    let cell = sim_faces_base(q) + (id.y * q.n + id.x) * 4u;
    // Across x: east then west, kept above the ground at both faces.
    let slope_x = vec3<f32>(sim_half_slope(west.x, here.x, east.x),
        sim_half_slope(west.y, here.y, east.y), sim_half_slope(west.z, here.z, east.z));
    var east_w = here.x + slope_x.x;
    var west_w = here.x - slope_x.x;
    if east_w < ground_x.y {
        east_w = ground_x.y;
        west_w = 2.0 * here.x - ground_x.y;
    } else if west_w < ground_x.x {
        west_w = ground_x.x;
        east_w = 2.0 * here.x - ground_x.x;
    }
    sim_faces[cell] = sim_face(east_w, ground_x.y, here.y + slope_x.y, here.z + slope_x.z);
    sim_faces[cell + 1u] = sim_face(west_w, ground_x.x, here.y - slope_x.y, here.z - slope_x.z);
    // Across z: north then south, the flow along z across these faces.
    let slope_z = vec3<f32>(sim_half_slope(south.x, here.x, north.x),
        sim_half_slope(south.y, here.y, north.y), sim_half_slope(south.z, here.z, north.z));
    var north_w = here.x + slope_z.x;
    var south_w = here.x - slope_z.x;
    if north_w < ground_z.y {
        north_w = ground_z.y;
        south_w = 2.0 * here.x - ground_z.y;
    } else if south_w < ground_z.x {
        south_w = ground_z.x;
        north_w = 2.0 * here.x - ground_z.x;
    }
    sim_faces[cell + 2u] = sim_face(north_w, ground_z.y, here.z + slope_z.z, here.y + slope_z.y);
    sim_faces[cell + 3u] = sim_face(south_w, ground_z.x, here.z - slope_z.z, here.y - slope_z.y);
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

// Whether cell `at` of the stage's state is held at a lake's level.
fn sim_held_at(q: Square, at: u32) -> bool {
    return sim_cells[sim_cells_base(q) + frame.stage.x * q.n * q.n + at].w > 0.5 * SIM_UNHELD;
}

// The flux `flux` through the face from cell `near` to cell `far`, none
// where it would carry water out of a lake.
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

// The flux through a face on an open square's edge whose inside stands as
// `face`, `inward` the way into the square across it (+1 or -1): water
// leaves as it runs, the inside standing on both sides, and none comes in.
fn sim_open(face: vec4<f32>, inward: f32) -> vec4<f32> {
    let flux = sim_flux(face, face);
    if flux.x * inward > 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, flux.w);
    }
    return flux;
}

@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_exchange(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = sim_square(id.z);
    let n = q.n;
    let i = id.x;
    let j = id.y;
    if id.z >= frame.stage.z || !sim_has(q, SIM_STEPPING) {
        return;
    }
    let open = sim_has(q, SIM_OPEN);
    let faces = sim_faces_base(q);
    let fluxes = sim_fluxes_base(q);
    // The face between columns i - 1 and i; at the edges, between the
    // inside and the ring cell past it, or open.
    if j < n && i <= n {
        var flux: vec4<f32>;
        if i == 0u {
            let inside = sim_faces[faces + (j * n) * 4u + 1u];
            if open {
                flux = sim_open(inside, 1.0);
            } else {
                let ghost = 2u * n + 2u * j;
                flux = sim_edge(sim_ghost_face(q, ghost, sim_ground_x(q, 0u, j), false), inside,
                    sim_ghost_held(q, ghost), sim_held_at(q, j * n));
            }
        } else if i == n {
            let inside = sim_faces[faces + (j * n + n - 1u) * 4u];
            if open {
                flux = sim_open(inside, -1.0);
            } else {
                let ghost = 2u * n + 2u * j + 1u;
                flux = sim_edge(inside, sim_ghost_face(q, ghost, sim_ground_x(q, n, j), false),
                    sim_held_at(q, j * n + n - 1u), sim_ghost_held(q, ghost));
            }
        } else {
            let l = sim_faces[faces + (j * n + i - 1u) * 4u];
            let r = sim_faces[faces + (j * n + i) * 4u + 1u];
            flux = sim_from_lakes(q, sim_flux(l, r), j * n + i - 1u, j * n + i);
        }
        sim_fluxes[fluxes + j * (n + 1u) + i] = flux;
    }
    // The face between rows j - 1 and j.
    if i < n && j <= n {
        var flux: vec4<f32>;
        if j == 0u {
            let inside = sim_faces[faces + i * 4u + 3u];
            if open {
                flux = sim_open(inside, 1.0);
            } else {
                let ghost = 2u * i;
                flux = sim_edge(sim_ghost_face(q, ghost, sim_ground_z(q, i, 0u), true), inside,
                    sim_ghost_held(q, ghost), sim_held_at(q, i));
            }
        } else if j == n {
            let inside = sim_faces[faces + ((n - 1u) * n + i) * 4u + 2u];
            if open {
                flux = sim_open(inside, -1.0);
            } else {
                let ghost = 2u * i + 1u;
                flux = sim_edge(inside, sim_ghost_face(q, ghost, sim_ground_z(q, i, n), true),
                    sim_held_at(q, (n - 1u) * n + i), sim_ghost_held(q, ghost));
            }
        } else {
            let l = sim_faces[faces + ((j - 1u) * n + i) * 4u + 2u];
            let r = sim_faces[faces + (j * n + i) * 4u + 3u];
            flux = sim_from_lakes(q, sim_flux(l, r), (j - 1u) * n + i, j * n + i);
        }
        sim_fluxes[fluxes + (n + 1u) * n + j * n + i] = flux;
    }
}

// The fluxes into and out of cell (i, j): west, east, south, north.
fn sim_west(q: Square, i: u32, j: u32) -> vec4<f32> {
    return sim_fluxes[sim_fluxes_base(q) + j * (q.n + 1u) + i];
}

fn sim_east(q: Square, i: u32, j: u32) -> vec4<f32> {
    return sim_fluxes[sim_fluxes_base(q) + j * (q.n + 1u) + i + 1u];
}

fn sim_south(q: Square, i: u32, j: u32) -> vec4<f32> {
    return sim_fluxes[sim_fluxes_base(q) + (q.n + 1u) * q.n + j * q.n + i];
}

fn sim_north(q: Square, i: u32, j: u32) -> vec4<f32> {
    return sim_fluxes[sim_fluxes_base(q) + (q.n + 1u) * q.n + (j + 1u) * q.n + i];
}

// Settling squares' cells: the fastest wave through each cell's faces in
// the step's first stage.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_waves(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = sim_square(id.z);
    if !sim_runs(q, id) || !sim_has(q, SIM_LOCAL) {
        return;
    }
    let i = id.x;
    let j = id.y;
    let fastest = max(max(sim_west(q, i, j).w, sim_east(q, i, j).w),
        max(sim_south(q, i, j).w, sim_north(q, i, j).w));
    sim_steps[sim_steps_base(q) + j * q.n + i].x = fastest;
}

// Settling squares' cells: each one's seconds this step, as long as the
// fastest wave through its faces and its neighbors' allows, so neighbors
// step alike.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_seconds(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = sim_square(id.z);
    if !sim_runs(q, id) || !sim_has(q, SIM_LOCAL) {
        return;
    }
    let n = q.n;
    let base = sim_steps_base(q);
    let at = id.y * n + id.x;
    var fastest = sim_steps[base + at].x;
    if id.x > 0u {
        fastest = max(fastest, sim_steps[base + at - 1u].x);
    }
    if id.x + 1u < n {
        fastest = max(fastest, sim_steps[base + at + 1u].x);
    }
    if id.y > 0u {
        fastest = max(fastest, sim_steps[base + at - n].x);
    }
    if id.y + 1u < n {
        fastest = max(fastest, sim_steps[base + at + n].x);
    }
    sim_steps[base + at].y = SIM_LOCAL_SHARE * SIM_COURANT * q.dx / max(fastest, SIM_SLOWEST);
}

// Cell `at`'s seconds this step: its own where it steps locally, else the
// square's.
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
fn sim_settle(q: Square, water: vec3<f32>, ground: f32, held: f32, dt: f32) -> vec4<f32> {
    if held > 0.5 * SIM_UNHELD {
        return vec4<f32>(held, 0.0, 0.0, held);
    }
    var next = water;
    if !(abs(next.x) < 1.0e20 && abs(next.y) < 1.0e20 && abs(next.z) < 1.0e20) {
        next = vec3<f32>(ground, 0.0, 0.0);
    }
    next.x = max(next.x, ground);
    let h = next.x - ground;
    if h <= SIM_DRY {
        return vec4<f32>(next.x, 0.0, 0.0, held);
    }
    var flow = next.yz;
    let friction = SIM_GRAVITY * q.roughness * q.roughness * (length(flow) / h)
        / pow(h, 4.0 / 3.0);
    flow = flow / (1.0 + dt * friction);
    let speed = length(flow) / h;
    if speed > SIM_FASTEST {
        flow *= SIM_FASTEST / speed;
    }
    return vec4<f32>(next.x, flow, held);
}

@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_update(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = sim_square(id.z);
    if !sim_runs(q, id) {
        return;
    }
    let n = q.n;
    let i = id.x;
    let j = id.y;
    let at = j * n + i;
    let dx = q.dx;
    let dt = sim_seconds_at(q, at);
    let west = sim_west(q, i, j).xyz;
    let east = sim_east(q, i, j).xyz;
    let south = sim_south(q, i, j).xyz;
    let north = sim_north(q, i, j).xyz;
    let faces = sim_faces_base(q) + at * 4u;
    let depth_x = 0.5 * (sim_faces[faces].y + sim_faces[faces + 1u].y);
    let depth_z = 0.5 * (sim_faces[faces + 2u].y + sim_faces[faces + 3u].y);
    let slope_x = (sim_ground_x(q, i + 1u, j) - sim_ground_x(q, i, j)) / dx;
    let slope_z = (sim_ground_z(q, i, j + 1u) - sim_ground_z(q, i, j)) / dx;
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
        next = 0.5 * (start.xyz + sim_cells[base + n * n + at].xyz + dt * change);
    }
    let ground = 0.5 * (sim_ground_x(q, i, j) + sim_ground_x(q, i + 1u, j));
    let written = select(base + n * n + at, base + at, frame.stage.x == 1u);
    sim_cells[written] = sim_settle(q, next, ground, start.w, dt);
}

// The square followed the camera by `frame.shift` cells: each cell takes
// the water and foam of the cell standing where it does in the square as it
// was, where that one held any, into the second state, which holds the
// water as it settled where it held none.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_carry(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = sim_square(id.z);
    if !sim_runs(q, id) {
        return;
    }
    let n = q.n;
    let source = vec2<i32>(id.xy) + frame.shift.xy;
    if any(source < vec2<i32>(0)) || any(source >= vec2<i32>(i32(n))) {
        return;
    }
    let base = sim_cells_base(q);
    let from_cell = u32(source.y) * n + u32(source.x);
    let at = n * n + id.y * n + id.x;
    sim_cells[base + at] = sim_cells[base + from_cell];
    sim_foam[base + at] = sim_foam[base + from_cell];
}

// The water carried over is what the steps start from.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_restart(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = sim_square(id.z);
    if !sim_runs(q, id) {
        return;
    }
    let base = sim_cells_base(q);
    let at = id.y * q.n + id.x;
    sim_cells[base + at] = sim_cells[base + q.n * q.n + at];
    sim_foam[base + at] = sim_foam[base + q.n * q.n + at];
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

// The first state's cell (i, j): its depth and velocity; or `instead`'s
// where it is dry or outside the square.
fn sim_moving(q: Square, i: i32, j: i32, instead: vec3<f32>) -> vec3<f32> {
    let n = i32(q.n);
    if i < 0 || j < 0 || i >= n || j >= n {
        return instead;
    }
    let cell = sim_cells[sim_cells_base(q) + u32(j * n + i)];
    let ground = 0.5 * (sim_ground_x(q, u32(i), u32(j)) + sim_ground_x(q, u32(i) + 1u, u32(j)));
    let h = cell.x - ground;
    if h <= SIM_FOAM_WET {
        return instead;
    }
    return vec3<f32>(h, sim_velocity(h, cell.y), sim_velocity(h, cell.z));
}

// The first state's foam at cell `p` (fractional, along x and z),
// bilinearly; past the square's edges, the nearest ring cell's, or none
// for an open square's.
fn sim_foam_at(q: Square, p: vec2<f32>) -> f32 {
    let n = q.n;
    let last = f32(n - 1u);
    let near = vec2<u32>(clamp(round(p), vec2<f32>(0.0), vec2<f32>(last)));
    let ring = sim_ring_base(q);
    let outside = p.x < -0.5 || p.x > last + 0.5 || p.y < -0.5 || p.y > last + 0.5;
    if outside && sim_has(q, SIM_OPEN) {
        return 0.0;
    }
    if p.x < -0.5 {
        return sim_inputs[ring + 2u * (2u * n + 2u * near.y) + 1u].x;
    }
    if p.x > last + 0.5 {
        return sim_inputs[ring + 2u * (2u * n + 2u * near.y + 1u) + 1u].x;
    }
    if p.y < -0.5 {
        return sim_inputs[ring + 2u * (2u * near.x) + 1u].x;
    }
    if p.y > last + 0.5 {
        return sim_inputs[ring + 2u * (2u * near.x + 1u) + 1u].x;
    }
    let base = sim_cells_base(q);
    let c = clamp(p, vec2<f32>(0.0), vec2<f32>(last));
    let low = vec2<u32>(floor(c));
    let high = min(low + 1u, vec2<u32>(n - 1u));
    let f = c - floor(c);
    let below = mix(sim_foam[base + low.y * n + low.x], sim_foam[base + low.y * n + high.x], f.x);
    let above = mix(sim_foam[base + high.y * n + low.x], sim_foam[base + high.y * n + high.x],
        f.x);
    return mix(below, above, f.y);
}

// Each cell's foam after a step, into the second state: what the current
// carried there (traced back along it), faded, and what the water churned
// up there.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_churn(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = sim_square(id.z);
    if !sim_runs(q, id) {
        return;
    }
    let n = q.n;
    let i = i32(id.x);
    let j = i32(id.y);
    let at = sim_cells_base(q) + n * n + id.y * n + id.x;
    let here = sim_moving(q, i, j, vec3<f32>(0.0));
    if here.x <= SIM_FOAM_WET {
        sim_foam[at] = 0.0;
        return;
    }
    let dx = q.dx;
    let dt = sim_seconds_at(q, id.y * n + id.x);
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
    sim_foam[at] = min(churned, SIM_FOAM_MOST);
}

// The foam after the step is the next one's start; a summing square adds
// each cell's water and foam to its sums.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_churned(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = sim_square(id.z);
    if !sim_runs(q, id) {
        return;
    }
    let base = sim_cells_base(q);
    let at = id.y * q.n + id.x;
    let foam = sim_foam[base + q.n * q.n + at];
    sim_foam[base + at] = foam;
    if sim_has(q, SIM_SUMMING) {
        let water = sim_cells[base + at];
        let sums = sim_averages_base(q) + at;
        sim_averages[sums] += vec4<f32>(water.xyz, foam);
    }
}

// Rivers bring their water to their cells (each cell once) after a step,
// over each cell's seconds.
@compute @workgroup_size(WATER_SIM_SOURCE_GROUP)
fn sim_source(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = sim_square(id.z);
    if id.z >= frame.stage.z || !sim_has(q, SIM_STEPPING) || id.x >= q.sourced {
        return;
    }
    let source = sim_inputs[sim_sources_base(q) + id.x];
    let at = bitcast<u32>(source.x);
    let cells = sim_cells_base(q);
    var cell = sim_cells[cells + at];
    if cell.w > 0.5 * SIM_UNHELD {
        return;
    }
    let added = source.y * sim_seconds_at(q, at) / (q.dx * q.dx);
    cell.x += added;
    cell.y += added * source.z * SIM_SOURCE_SPEED;
    cell.z += added * source.w * SIM_SOURCE_SPEED;
    sim_cells[cells + at] = cell;
}

// A look's sums of a workgroup's cells.
var<workgroup> sim_look_sums: array<vec4<f32>, 64>;

// A look at a summing square: each cell's mean water over the steps since
// the look before (its sums over `summed` steps) replaces the mean kept
// then, and its sums start anew; each workgroup sums the water its cells'
// means hold (cubic meters a meter of cell, above their ground), how far
// the velocities of those flowing moved since the mean before, those
// flowing and those stepping.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_look(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let q = sim_square(id.z);
    var sums = vec4<f32>(0.0);
    let looking = id.z < frame.stage.z && sim_has(q, SIM_LOOKING) && sim_has(q, SIM_SUMMING);
    if looking && id.x < q.n && id.y < q.n {
        let n = q.n;
        let at = id.y * n + id.x;
        let averages = sim_averages_base(q);
        let mean = sim_averages[averages + at] / max(q.summed, 1.0);
        let before = sim_averages[averages + n * n + at];
        sim_averages[averages + n * n + at] = mean;
        sim_averages[averages + at] = vec4<f32>(0.0);
        let ground = 0.5 * (sim_ground_x(q, id.x, id.y) + sim_ground_x(q, id.x + 1u, id.y));
        let held = sim_cells[sim_cells_base(q) + at].w > 0.5 * SIM_UNHELD;
        let h = mean.x - ground;
        let was = before.x - ground;
        if !held && h > SIM_DRY {
            let velocity = vec2<f32>(sim_velocity(h, mean.y), sim_velocity(h, mean.z));
            var earlier = vec2<f32>(0.0);
            if was > SIM_DRY {
                earlier = vec2<f32>(sim_velocity(was, before.y), sim_velocity(was, before.z));
            }
            sums = vec4<f32>(h, length(velocity - earlier), 1.0, 1.0);
        } else {
            sums = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    }
    sim_look_sums[local] = sums;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if stride == 0u {
            break;
        }
        if local < stride {
            sim_look_sums[local] += sim_look_sums[local + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if local == 0u && looking {
        let groups = sim_groups();
        let at = sim_averages_base(q) + 2u * sim_capacity() * sim_capacity() + group.y * groups + group.x;
        sim_averages[at] = sim_look_sums[0];
    }
}
