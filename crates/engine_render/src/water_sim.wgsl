// Water's simulation (`water_sim.rs`): rivers running over a square of
// ground around the camera as the shallow-water equations say, on a grid of
// cells, by Kurganov and Petrova's well-balanced, positivity-preserving
// central-upwind scheme (2007): second order in space (each cell's water
// reconstructed linearly, its slopes limited by a generalized minmod and
// kept from dipping below the ground at its faces) and in time (Heun's two
// stages), with Manning's friction, velocities desingularized where the
// water thins out. Each cell holds its water's surface (world meters), its
// flow along x and z (square meters a second) and the level a lake holds
// it at, if one does: lakes stand still, taking the water that runs into
// them and giving none (the rivers leaving them flow in where they start).
// The ground stands at the cells' corners. A stage reconstructs each cell's faces, finds the flux through
// every face and updates every cell; rivers flow in after each step. Water
// leaves freely over the square's edges, and none comes in over them.
// The water carries foam: it churns it up where it runs faster than its
// waves and where it slams into slower water, the current carries it off,
// and it fades as its bubbles break.

const WATER_SIM_GROUP: u32 = 8u;
const WATER_SIM_INFLOW_GROUP: u32 = 64u;
const SIM_GRAVITY: f32 = 9.81;
// The generalized minmod's theta; the depth below which velocities are
// desingularized and the depth under which a cell is dry (meters); and the
// fastest water runs (meters a second).
const SIM_THETA: f32 = 1.3;
const SIM_THIN: f32 = 0.01;
const SIM_DRY: f32 = 1.0e-4;
const SIM_FASTEST: f32 = 20.0;
// The meters a second rivers flow in at, and what a cell no lake holds
// holds as its lake's level.
const SIM_INFLOW: f32 = 1.5;
const SIM_UNHELD: f32 = -1.0e30;

// The cells along a side, the stage (0: the first, 1: the second) and the
// cells rivers flow in at; the meters across a cell, the seconds a step and
// Manning's roughness; and, as the square follows the camera, the cells it
// moved by along x and z.
struct SimFrame {
    shape: vec4<u32>,
    step: vec4<f32>,
    shift: vec4<i32>,
}

@group(0) @binding(0) var<uniform> frame: SimFrame;
// The ground at the cells' corners (world meters), cells + 1 a row, rows
// along z.
@group(0) @binding(1) var<storage, read> sim_ground: array<f32>;
// The step's start and then its first stage, each a row of cells at a time:
// the water's surface, its flow along x and z, and the level a lake holds
// it at (SIM_UNHELD where none does).
@group(0) @binding(2) var<storage, read_write> sim_cells: array<vec4<f32>>;
// Each cell's faces as reconstructed, east, west, north and south: the
// water's surface and depth there, and its velocity across the face and
// along it.
@group(0) @binding(3) var<storage, read_write> sim_faces: array<vec4<f32>>;
// The fluxes through the faces between columns (cells + 1 a row), then
// through those between rows (cells a row): its mass and its momentum
// across and along the face.
@group(0) @binding(4) var<storage, read_write> sim_fluxes: array<vec4<f32>>;
// The cells rivers flow in at: the cell (its index's bits), the cubic meters
// a second it takes in, and the way the water runs (x and z).
@group(0) @binding(5) var<storage, read> sim_inflows: array<vec4<f32>>;
// The foam the water carries (0 none, 1 white all over), the step's start
// and then its next, each a row of cells at a time.
@group(0) @binding(6) var<storage, read_write> sim_foam: array<f32>;

fn sim_cells_along() -> u32 {
    return frame.shape.x;
}

fn sim_corner(i: u32, j: u32) -> f32 {
    return sim_ground[j * (sim_cells_along() + 1u) + i];
}

// The ground at the middle of the face between columns i - 1 and i of row
// j, and of the face between rows j - 1 and j of column i.
fn sim_ground_x(i: u32, j: u32) -> f32 {
    return 0.5 * (sim_corner(i, j) + sim_corner(i, j + 1u));
}

fn sim_ground_z(i: u32, j: u32) -> f32 {
    return 0.5 * (sim_corner(i, j) + sim_corner(i + 1u, j));
}

// Cell (i, j) of the state starting at `base`, the nearest inside the square
// for one past its edge.
fn sim_cell(base: u32, i: i32, j: i32) -> vec4<f32> {
    let n = i32(sim_cells_along());
    let at = vec2<i32>(clamp(i, 0, n - 1), clamp(j, 0, n - 1));
    return sim_cells[base + u32(at.y * n + at.x)];
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
    let n = sim_cells_along();
    if id.x >= n || id.y >= n {
        return;
    }
    let base = frame.shape.y * n * n;
    let i = i32(id.x);
    let j = i32(id.y);
    let here = sim_cell(base, i, j);
    let west = sim_cell(base, i - 1, j);
    let east = sim_cell(base, i + 1, j);
    let south = sim_cell(base, i, j - 1);
    let north = sim_cell(base, i, j + 1);
    let ground_x = vec2<f32>(sim_ground_x(id.x, id.y), sim_ground_x(id.x + 1u, id.y));
    let ground_z = vec2<f32>(sim_ground_z(id.x, id.y), sim_ground_z(id.x, id.y + 1u));
    let cell = (id.y * n + id.x) * 4u;
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
// along the face.
fn sim_flux(l: vec4<f32>, r: vec4<f32>) -> vec3<f32> {
    let cl = sqrt(SIM_GRAVITY * l.y);
    let cr = sqrt(SIM_GRAVITY * r.y);
    let fastest = max(max(l.z + cl, r.z + cr), 0.0);
    let slowest = min(min(l.z - cl, r.z - cr), 0.0);
    let span = fastest - slowest;
    if span < 1.0e-6 {
        return vec3<f32>(0.0);
    }
    let fl = vec3<f32>(l.y * l.z, l.y * l.z * l.z + 0.5 * SIM_GRAVITY * l.y * l.y,
        l.y * l.z * l.w);
    let fr = vec3<f32>(r.y * r.z, r.y * r.z * r.z + 0.5 * SIM_GRAVITY * r.y * r.y,
        r.y * r.z * r.w);
    let ul = vec3<f32>(l.x, l.y * l.z, l.y * l.w);
    let ur = vec3<f32>(r.x, r.y * r.z, r.y * r.w);
    return (fastest * fl - slowest * fr) / span + (fastest * slowest / span) * (ur - ul);
}

// Whether cell `at` of the stage's state is held at a lake's level.
fn sim_held(at: u32) -> bool {
    let n = sim_cells_along();
    return sim_cells[frame.shape.y * n * n + at].w > 0.5 * SIM_UNHELD;
}

// The flux `flux` through the face from cell `near` to cell `far`, none
// where it would carry water out of a lake.
fn sim_from_lakes(flux: vec3<f32>, near: u32, far: u32) -> vec3<f32> {
    if (flux.x > 0.0 && sim_held(near)) || (flux.x < 0.0 && sim_held(far)) {
        return vec3<f32>(0.0);
    }
    return flux;
}

// The flux through a face on the square's edge whose inside stands as
// `face`, `inward` the way into the square across it (+1 or -1): water
// leaves as it runs, the inside standing on both sides, and none comes in
// (rivers flowing in do so where the inflows say).
fn sim_edge(face: vec4<f32>, inward: f32) -> vec3<f32> {
    let flux = sim_flux(face, face);
    return select(flux, vec3<f32>(0.0), flux.x * inward > 0.0);
}

@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_exchange(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = sim_cells_along();
    let i = id.x;
    let j = id.y;
    // The face between columns i - 1 and i; at the edges, the inside
    // stands on both sides.
    if j < n && i <= n {
        var flux: vec3<f32>;
        if i == 0u {
            flux = sim_edge(sim_faces[(j * n) * 4u + 1u], 1.0);
        } else if i == n {
            flux = sim_edge(sim_faces[(j * n + n - 1u) * 4u], -1.0);
        } else {
            let l = sim_faces[(j * n + i - 1u) * 4u];
            let r = sim_faces[(j * n + i) * 4u + 1u];
            flux = sim_from_lakes(sim_flux(l, r), j * n + i - 1u, j * n + i);
        }
        sim_fluxes[j * (n + 1u) + i] = vec4<f32>(flux, 0.0);
    }
    // The face between rows j - 1 and j.
    if i < n && j <= n {
        var flux: vec3<f32>;
        if j == 0u {
            flux = sim_edge(sim_faces[i * 4u + 3u], 1.0);
        } else if j == n {
            flux = sim_edge(sim_faces[((n - 1u) * n + i) * 4u + 2u], -1.0);
        } else {
            let l = sim_faces[((j - 1u) * n + i) * 4u + 2u];
            let r = sim_faces[(j * n + i) * 4u + 3u];
            flux = sim_from_lakes(sim_flux(l, r), (j - 1u) * n + i, j * n + i);
        }
        sim_fluxes[(n + 1u) * n + j * n + i] = vec4<f32>(flux, 0.0);
    }
}

// A cell's water `water` (surface and flow) settled over `ground` after a
// stage of `dt` seconds: never below the ground, slowed by friction, no
// faster than SIM_FASTEST, reset where it stopped being finite, and held
// at its lake's level `held` where one holds it.
fn sim_settle(water: vec3<f32>, ground: f32, held: f32, dt: f32) -> vec4<f32> {
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
    let roughness = frame.step.z;
    let friction = SIM_GRAVITY * roughness * roughness * (length(flow) / h) / pow(h, 4.0 / 3.0);
    flow = flow / (1.0 + dt * friction);
    let speed = length(flow) / h;
    if speed > SIM_FASTEST {
        flow *= SIM_FASTEST / speed;
    }
    return vec4<f32>(next.x, flow, held);
}

@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_update(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = sim_cells_along();
    if id.x >= n || id.y >= n {
        return;
    }
    let i = id.x;
    let j = id.y;
    let at = j * n + i;
    let dx = frame.step.x;
    let dt = frame.step.y;
    let west = sim_fluxes[j * (n + 1u) + i].xyz;
    let east = sim_fluxes[j * (n + 1u) + i + 1u].xyz;
    let south = sim_fluxes[(n + 1u) * n + j * n + i].xyz;
    let north = sim_fluxes[(n + 1u) * n + (j + 1u) * n + i].xyz;
    let depth_x = 0.5 * (sim_faces[at * 4u].y + sim_faces[at * 4u + 1u].y);
    let depth_z = 0.5 * (sim_faces[at * 4u + 2u].y + sim_faces[at * 4u + 3u].y);
    let slope_x = (sim_ground_x(i + 1u, j) - sim_ground_x(i, j)) / dx;
    let slope_z = (sim_ground_z(i, j + 1u) - sim_ground_z(i, j)) / dx;
    // The flux through faces between rows carries the flow along z across
    // them and the flow along x along them.
    let change = vec3<f32>(
        -(east.x - west.x) / dx - (north.x - south.x) / dx,
        -(east.y - west.y) / dx - (north.z - south.z) / dx - SIM_GRAVITY * depth_x * slope_x,
        -(east.z - west.z) / dx - (north.y - south.y) / dx - SIM_GRAVITY * depth_z * slope_z);
    let start = sim_cells[at];
    var next = start.xyz + dt * change;
    if frame.shape.y == 1u {
        next = 0.5 * (start.xyz + sim_cells[n * n + at].xyz + dt * change);
    }
    let ground = 0.5 * (sim_ground_x(i, j) + sim_ground_x(i + 1u, j));
    let written = select(n * n + at, at, frame.shape.y == 1u);
    sim_cells[written] = sim_settle(next, ground, start.w, dt);
}

// The square followed the camera by `frame.shift` cells: each cell takes
// the water of the cell standing where it does in the square as it was,
// where that one held any, into the second state, which holds the water
// the bake stands where it held none.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_carry(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = sim_cells_along();
    if id.x >= n || id.y >= n {
        return;
    }
    let source = vec2<i32>(id.xy) + frame.shift.xy;
    let at = n * n + id.y * n + id.x;
    if any(source < vec2<i32>(0)) || any(source >= vec2<i32>(i32(n))) {
        sim_foam[at] = 0.0;
        return;
    }
    let from_cell = u32(source.y) * n + u32(source.x);
    sim_cells[at] = sim_cells[from_cell];
    sim_foam[at] = sim_foam[from_cell];
}

// The water carried over is what the steps start from.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_restart(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = sim_cells_along();
    if id.x >= n || id.y >= n {
        return;
    }
    let at = id.y * n + id.x;
    sim_cells[at] = sim_cells[n * n + at];
    sim_foam[at] = sim_foam[n * n + at];
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
fn sim_moving(i: i32, j: i32, instead: vec3<f32>) -> vec3<f32> {
    let n = i32(sim_cells_along());
    if i < 0 || j < 0 || i >= n || j >= n {
        return instead;
    }
    let cell = sim_cells[u32(j * n + i)];
    let ground = 0.5 * (sim_ground_x(u32(i), u32(j)) + sim_ground_x(u32(i) + 1u, u32(j)));
    let h = cell.x - ground;
    if h <= SIM_FOAM_WET {
        return instead;
    }
    return vec3<f32>(h, sim_velocity(h, cell.y), sim_velocity(h, cell.z));
}

// The first state's foam at cell `p` (fractional, along x and z),
// bilinearly.
fn sim_foam_at(p: vec2<f32>) -> f32 {
    let n = sim_cells_along();
    let q = clamp(p, vec2<f32>(0.0), vec2<f32>(f32(n - 1u)));
    let low = vec2<u32>(floor(q));
    let high = min(low + 1u, vec2<u32>(n - 1u));
    let f = q - floor(q);
    let below = mix(sim_foam[low.y * n + low.x], sim_foam[low.y * n + high.x], f.x);
    let above = mix(sim_foam[high.y * n + low.x], sim_foam[high.y * n + high.x], f.x);
    return mix(below, above, f.y);
}

// Each cell's foam after a step, into the second state: what the current
// carried there (traced back along it), faded, and what the water churned
// up there.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_churn(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = sim_cells_along();
    if id.x >= n || id.y >= n {
        return;
    }
    let i = i32(id.x);
    let j = i32(id.y);
    let at = n * n + id.y * n + id.x;
    let here = sim_moving(i, j, vec3<f32>(0.0));
    if here.x <= SIM_FOAM_WET {
        sim_foam[at] = 0.0;
        return;
    }
    let dx = frame.step.x;
    let dt = frame.step.y;
    let carried = sim_foam_at(vec2<f32>(f32(i), f32(j)) - here.yz * dt / dx);
    let froude = length(here.yz) / sqrt(SIM_GRAVITY * here.x);
    let rapids = max(froude - SIM_FOAM_FROUDE, 0.0) * SIM_FOAM_RAPIDS;
    let east = sim_moving(i + 1, j, here).y;
    let west = sim_moving(i - 1, j, here).y;
    let north = sim_moving(i, j + 1, here).z;
    let south = sim_moving(i, j - 1, here).z;
    let converging = max(-(east - west + north - south) / (2.0 * dx), 0.0);
    let churned = carried * exp(-dt / SIM_FOAM_LIFE)
        + dt * (rapids + SIM_FOAM_PILING * converging);
    sim_foam[at] = min(churned, SIM_FOAM_MOST);
}

// The foam after the step is the next one's start.
@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_churned(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = sim_cells_along();
    if id.x >= n || id.y >= n {
        return;
    }
    let at = id.y * n + id.x;
    sim_foam[at] = sim_foam[n * n + at];
}

@compute @workgroup_size(WATER_SIM_INFLOW_GROUP)
fn sim_inflow(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.shape.z {
        return;
    }
    let inflow = sim_inflows[id.x];
    let at = bitcast<u32>(inflow.x);
    var cell = sim_cells[at];
    if cell.w > 0.5 * SIM_UNHELD {
        return;
    }
    let added = inflow.y * frame.step.y / (frame.step.x * frame.step.x);
    cell.x += added;
    cell.y += added * inflow.z * SIM_INFLOW;
    cell.z += added * inflow.w * SIM_INFLOW;
    sim_cells[at] = cell;
}
