const WATER_SIM_GROUP: u32 = 8u;
const WATER_SIM_SOURCE_GROUP: u32 = 64u;
const SIM_SLOTS: u32 = 64u;
const SIM_NO_BLOCK: u32 = 0xffffffffu;
const SIM_STANDING: u32 = 0x80000000u;
const SIM_BLOCK_CELLS: u32 = WATER_SIM_GROUP * WATER_SIM_GROUP;
const SIM_BLOCK_CORNERS: u32 = (WATER_SIM_GROUP + 1u) * (WATER_SIM_GROUP + 1u);
const SIM_BLOCK_GROUND: u32 = 337u;
const SIM_GRAVITY: f32 = 9.81;
const SIM_DENSITY: f32 = 1000.0;
const SIM_THETA: f32 = 1.3;
const SIM_THIN: f32 = 0.01;
const SIM_DRY: f32 = 1.0e-4;
const SIM_FASTEST: f32 = 20.0;
const SIM_WET_MOST: f32 = 0.5;
const SIM_FERGUSON_A1: f32 = 6.5;
const SIM_FERGUSON_A2: f32 = 2.5;
const SIM_SHALLOWEST: f32 = 0.1;
const SIM_DEEPEST: f32 = 30.0;
const SIM_STRICKLER: f32 = 20.4;
const SIM_SOURCE_SPEED: f32 = 1.5;
const SIM_UNHELD: f32 = -1.0e30;
const SIM_COURANT: f32 = 0.24;
const SIM_LOCAL_SHARE: f32 = 0.9;
const SIM_SLOWEST: f32 = 1.0;
const SIM_CELL_SUMS: u32 = 4u;
const SIM_CELL_MEANS: u32 = 3u;
const SIM_BLOCK_LOOKS: u32 = 3u;
const SIM_LIVE_LOOKS: u32 = 2u;
const SIM_STEPPING: u32 = 1u;
const SIM_LOCAL: u32 = 2u;
const SIM_OPEN: u32 = 4u;
const SIM_SUMMING: u32 = 8u;
const SIM_LOOKING: u32 = 16u;
const SIM_SPARSE: u32 = 32u;
const SIM_LIVE: u32 = 64u;
const SIM_BURSTING: u32 = 128u;
const SIM_KEEPING: u32 = 256u;
const SIM_RIM: u32 = 8u;
const SIM_NO_SURFACE: f32 = -1.0e30;
const SIM_FLOW_RANGE: f32 = 8.0;
const SIM_SHEAR_RANGE: f32 = 8.0;
const SIM_TURBULENT_SPEED: f32 = 0.5;
const SIM_TURBULENT_SWIRL: f32 = 4.0;
const SIM_BREAK_ONSET: f32 = 0.6;
const SIM_BREAK_FULL: f32 = 1.2;
const SIM_BREAK_THINNEST: f32 = 0.02;

struct SimFrame {
    stage: vec4<u32>,
    inputs: vec4<u32>,
}

struct SimSlot {
    shape: vec4<u32>,
    step: vec4<f32>,
    blocks: vec4<u32>,
    place: vec4<u32>,
}

struct SimSlots {
    slot: array<SimSlot, SIM_SLOTS>,
}

@group(0) @binding(0) var<uniform> frame: SimFrame;
@group(0) @binding(1) var<storage, read> sim_ground: array<f32>;
@group(0) @binding(2) var<storage, read_write> sim_cells: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> sim_depths: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> sim_fluxes: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> sim_inputs: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> sim_blocks: array<u32>;
@group(0) @binding(7) var<storage, read_write> sim_steps: array<vec2<f32>>;
@group(0) @binding(8) var<uniform> sim_slots: SimSlots;
@group(0) @binding(9) var<storage, read_write> sim_averages: array<vec4<f32>>;

const SIM_TOUCHES: u32 = 32u;

struct SimTouch {
    center: vec4<f32>,
    inverse: array<vec4<f32>, 3>,
    velocity: vec4<f32>,
    reach: vec4<f32>,
}

struct SimTouches {
    count: vec4<u32>,
    blocks: vec4<u32>,
    touch: array<SimTouch, SIM_TOUCHES>,
}

@group(0) @binding(10) var<uniform> sim_touches: SimTouches;

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
    return q.slot * (sim_held() + 2u * sim_pool() + 1u);
}

fn sim_averages_base(q: Square) -> u32 {
    return q.slot * ((SIM_CELL_SUMS + SIM_CELL_MEANS) * sim_held() + SIM_BLOCK_LOOKS * sim_pool());
}

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

fn sim_looks(at: u32) -> u32 {
    return SIM_LIVE_LOOKS * at;
}

fn sim_first(q: Square) -> vec2<i32> {
    return bitcast<vec2<i32>>(sim_slots.slot[q.slot].place.xy);
}

fn sim_count_at(q: Square) -> u32 {
    return sim_steps_base(q) + sim_held() + 2u * sim_pool();
}

fn sim_step(q: Square) -> u32 {
    return sim_slots.slot[q.slot].place.z + u32(sim_steps[sim_count_at(q)].x);
}

fn sim_table(q: Square, block: vec2<i32>) -> u32 {
    if any(block < vec2<i32>(0)) || any(block >= vec2<i32>(q.blocks)) {
        return SIM_NO_BLOCK;
    }
    return sim_blocks[sim_table_base(q) + u32(block.y) * q.blocks.x + u32(block.x)];
}

fn sim_stands(entry: u32) -> bool {
    return entry >= SIM_STANDING && entry != SIM_NO_BLOCK;
}

fn sim_block(q: Square, block: vec2<i32>) -> u32 {
    let entry = sim_table(q, block);
    return select(entry, SIM_NO_BLOCK, entry >= SIM_STANDING);
}

fn sim_place(q: Square, block: u32) -> vec2<u32> {
    let place = sim_blocks[sim_table_base(q) + sim_mapped() + block];
    return vec2<u32>(place & 0xffffu, place >> 16u);
}

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

fn sim_standing_at(q: Square, i: i32, j: i32) -> u32 {
    if i < 0 || j < 0 || i >= i32(q.nx) || j >= i32(q.nz) {
        return SIM_NO_BLOCK;
    }
    let cell = vec2<u32>(u32(i), u32(j));
    let entry = sim_table(q, vec2<i32>(cell / WATER_SIM_GROUP));
    if !sim_stands(entry) {
        return SIM_NO_BLOCK;
    }
    let local = cell % WATER_SIM_GROUP;
    return (entry - SIM_STANDING) * SIM_BLOCK_CELLS + local.y * WATER_SIM_GROUP + local.x;
}

struct SimCell {
    at: u32,
    i: u32,
    j: u32,
    runs: bool,
}

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

fn sim_block_at(q: Square, block: u32) -> u32 {
    return sim_steps_base(q) + sim_held() + 2u * block;
}

fn sim_active(q: Square, at: u32) -> bool {
    if !sim_has(q, SIM_SPARSE) {
        return true;
    }
    return sim_steps[sim_block_at(q, at / SIM_BLOCK_CELLS) + 1u].x > 0.5;
}

fn sim_corner(q: Square, block: u32, x: u32, z: u32) -> f32 {
    return sim_ground[sim_ground_base(q) + block * SIM_BLOCK_GROUND
        + z * (WATER_SIM_GROUP + 1u) + x];
}

fn sim_corners(q: Square, at: u32) -> vec4<f32> {
    let block = at / SIM_BLOCK_CELLS;
    let x = at % WATER_SIM_GROUP;
    let z = (at % SIM_BLOCK_CELLS) / WATER_SIM_GROUP;
    return vec4<f32>(sim_corner(q, block, x, z), sim_corner(q, block, x + 1u, z),
        sim_corner(q, block, x, z + 1u), sim_corner(q, block, x + 1u, z + 1u));
}

fn sim_ground_x(q: Square, at: u32) -> vec2<f32> {
    let corners = sim_corners(q, at);
    return vec2<f32>(0.5 * (corners.x + corners.z), 0.5 * (corners.y + corners.w));
}

fn sim_ground_z(q: Square, at: u32) -> vec2<f32> {
    let corners = sim_corners(q, at);
    return vec2<f32>(0.5 * (corners.x + corners.y), 0.5 * (corners.z + corners.w));
}

fn sim_bed(q: Square, at: u32) -> f32 {
    let ground = sim_ground_x(q, at);
    return 0.5 * (ground.x + ground.y);
}

fn sim_lake(q: Square, at: u32) -> f32 {
    return sim_ground[sim_ground_base(q) + (at / SIM_BLOCK_CELLS) * SIM_BLOCK_GROUND
        + SIM_BLOCK_CORNERS + at % SIM_BLOCK_CELLS];
}

fn sim_held_at(q: Square, at: u32) -> bool {
    return sim_lake(q, at) > 0.5 * SIM_UNHELD;
}

fn sim_rough(q: Square, at: u32) -> f32 {
    return sim_ground[sim_ground_base(q) + (at / SIM_BLOCK_CELLS) * SIM_BLOCK_GROUND
        + SIM_BLOCK_CORNERS + SIM_BLOCK_CELLS + at % SIM_BLOCK_CELLS];
}

fn sim_cover(q: Square, at: u32) -> vec2<f32> {
    if at == SIM_NO_BLOCK {
        return vec2<f32>(0.0);
    }
    let share = sim_ground_base(q) + (at / SIM_BLOCK_CELLS) * SIM_BLOCK_GROUND
        + SIM_BLOCK_CORNERS + 2u * SIM_BLOCK_CELLS + at % SIM_BLOCK_CELLS;
    return vec2<f32>(sim_ground[share], sim_ground[share + SIM_BLOCK_CELLS]);
}

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

fn sim_through(flux: vec4<f32>, open: f32) -> vec4<f32> {
    return vec4<f32>(flux.xyz * open, flux.w);
}

fn sim_lean(q: Square, at: u32) -> f32 {
    let x = sim_ground_x(q, at);
    let z = sim_ground_z(q, at);
    let slope = vec2<f32>(x.y - x.x, z.y - z.x) / q.dx;
    return 1.0 + dot(slope, slope);
}

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

fn sim_cell(q: Square, state: u32, at: u32) -> vec4<f32> {
    let standing = select(0u, state, sim_active(q, at));
    return sim_cells[sim_cells_base(q) + standing * sim_held() + at];
}

fn sim_water(q: Square, state: u32, at: u32, instead: vec4<f32>) -> vec4<f32> {
    if at == SIM_NO_BLOCK {
        return instead;
    }
    return sim_cell(q, state, at);
}

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

fn sim_half_slope(before: f32, middle: f32, after: f32) -> f32 {
    return 0.5 * sim_minmod(SIM_THETA * (middle - before), 0.5 * (after - before),
        SIM_THETA * (after - middle));
}

fn sim_velocity(h: f32, q: f32) -> f32 {
    let h4 = h * h * h * h;
    let thin = SIM_THIN * SIM_THIN * SIM_THIN * SIM_THIN;
    return 1.4142135 * h * q / sqrt(h4 + max(h4, thin));
}

fn sim_face(w: f32, ground: f32, across: f32, along: f32) -> vec4<f32> {
    let h = max(w - ground, 0.0);
    let velocity = clamp(vec2<f32>(sim_velocity(h, across), sim_velocity(h, along)),
        vec2<f32>(-SIM_FASTEST), vec2<f32>(SIM_FASTEST));
    return vec4<f32>(w, h, velocity);
}

struct SimSides {
    low: vec4<f32>,
    high: vec4<f32>,
}

fn sim_shore(q: Square, state: u32, at: u32, surface: f32, ground: vec2<f32>, higher: u32)
    -> bool {
    return surface < max(ground.x, ground.y) && surface - 0.5 * (ground.x + ground.y) > SIM_DRY
        && !sim_held_at(q, at) && sim_dry(q, state, higher);
}

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

fn sim_minmod_pair(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return select(vec3<f32>(0.0), sign(a) * min(abs(a), abs(b)), a * b > vec3<f32>(0.0));
}

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

fn sim_lost(l: vec4<f32>, r: vec4<f32>, mass: f32) -> f32 {
    let near = l.x + 0.5 * (l.z * l.z + l.w * l.w) / SIM_GRAVITY;
    let far = r.x + 0.5 * (r.z * r.z + r.w * r.w) / SIM_GRAVITY;
    return max(select(near - far, far - near, mass < 0.0), 0.0);
}

fn sim_kept(flux: vec4<f32>, l: vec4<f32>, r: vec4<f32>) -> vec4<f32> {
    if frame.stage.x == 0u {
        return flux;
    }
    return vec4<f32>(flux.xyz, sim_lost(l, r, flux.x));
}

fn sim_from_lakes(q: Square, flux: vec4<f32>, near: u32, far: u32) -> vec4<f32> {
    if (flux.x > 0.0 && sim_held_at(q, near)) || (flux.x < 0.0 && sim_held_at(q, far)) {
        return vec4<f32>(0.0, 0.0, 0.0, flux.w);
    }
    return flux;
}

fn sim_edge(l: vec4<f32>, r: vec4<f32>, near_held: bool, far_held: bool, g: f32) -> vec4<f32> {
    let flux = sim_flux(l, r, g);
    if (flux.x > 0.0 && near_held) || (flux.x < 0.0 && far_held) {
        return vec4<f32>(0.0, 0.0, 0.0, flux.w);
    }
    return flux;
}

fn sim_inflow(face: vec4<f32>, inflow: f32, inward: f32, g: f32) -> vec4<f32> {
    let mass = inward * inflow;
    let speed = inward * sim_velocity(face.y, inflow);
    let wave = abs(speed) + sqrt(g * face.y);
    return vec4<f32>(mass, mass * speed + 0.5 * g * face.y * face.y, mass * face.w, wave);
}

fn sim_ghost_flux(standing: vec4<f32>, inflow: f32, inside: vec4<f32>, ground: f32,
    across_z: bool, inward: f32, held: bool, g: f32) -> vec4<f32> {
    let across = select(standing.y, standing.z, across_z);
    let along = select(standing.z, standing.y, across_z);
    let outside = sim_face(standing.x, ground, across, along);
    let outside_held = standing.w > 0.5 * SIM_UNHELD;
    if inflow > 0.0 && !outside_held {
        return sim_kept(sim_inflow(outside, inflow, inward, g), outside, outside);
    }
    if inward > 0.0 {
        return sim_kept(sim_edge(outside, inside, outside_held, held, g), outside, inside);
    }
    return sim_kept(sim_edge(inside, outside, held, outside_held, g), inside, outside);
}

fn sim_ring_flux(q: Square, ghost: u32, inside: vec4<f32>, ground: f32, across_z: bool,
    inward: f32, held: bool, g: f32) -> vec4<f32> {
    let entry = sim_ring_base(q) + 2u * ghost;
    return sim_ghost_flux(sim_inputs[entry], max(sim_inputs[entry + 1u].x, 0.0), inside,
        ground, across_z, inward, held, g);
}

fn sim_open(face: vec4<f32>, inward: f32, g: f32) -> vec4<f32> {
    let flux = sim_flux(face, face, g);
    if flux.x * inward > 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, flux.w);
    }
    return flux;
}

fn sim_beside(q: Square, beside: vec2<i32>, face: vec4<f32>, ground: f32, across_z: bool,
    inward: f32, held: bool, g: f32) -> vec4<f32> {
    let standing = sim_standing_at(q, beside.x, beside.y);
    if standing == SIM_NO_BLOCK {
        return sim_kept(sim_open(face, inward, g), face, face);
    }
    let water = sim_cells[sim_cells_base(q) + standing];
    let inflow = max(inward * select(water.y, water.z, across_z), 0.0);
    return sim_ghost_flux(water, inflow, face, ground, across_z, inward, held, g);
}

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
            flux = sim_beside(q, vec2<i32>(i32(i) + 1, i32(j)), x.high, sim_ground_x(q, at).y,
                false, -1.0, sim_held_at(q, at), gravity);
        }
        sim_fluxes[fluxes + at] = sim_through(flux, sim_opening(q, state, at, east));
    }
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
            flux = sim_beside(q, vec2<i32>(i32(i), i32(j) + 1), z.high, sim_ground_z(q, at).y,
                true, -1.0, sim_held_at(q, at), gravity);
        }
        sim_fluxes[fluxes + sim_held() + at] = sim_through(flux,
            sim_opening(q, state, at, north));
    }
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

fn sim_west_beside(q: Square, at: u32, i: u32, j: u32, across_z: bool) -> vec4<f32> {
    let state = frame.stage.x;
    let open = sim_opening(q, state, SIM_NO_BLOCK, at);
    let held = sim_held_at(q, at);
    if across_z {
        return sim_through(sim_beside(q, vec2<i32>(i32(i), i32(j) - 1),
            sim_across_z(q, state, at, i, j).low, sim_ground_z(q, at).x, true, 1.0, held,
            sim_gravity_z(q, SIM_NO_BLOCK, at)), open);
    }
    return sim_through(sim_beside(q, vec2<i32>(i32(i) - 1, i32(j)),
        sim_across_x(q, state, at, i, j).low, sim_ground_x(q, at).x, false, 1.0, held,
        sim_gravity_x(q, SIM_NO_BLOCK, at)), open);
}

fn sim_west(q: Square, at: u32, i: u32, j: u32) -> vec4<f32> {
    let fluxes = sim_fluxes_base(q);
    if i == 0u {
        return sim_fluxes[fluxes + 2u * sim_held() + j];
    }
    let west = sim_at(q, i32(i) - 1, i32(j));
    if west == SIM_NO_BLOCK {
        return sim_west_beside(q, at, i, j, false);
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
        return sim_west_beside(q, at, i, j, true);
    }
    return sim_fluxes[fluxes + sim_held() + south];
}

fn sim_north(q: Square, at: u32) -> vec4<f32> {
    return sim_fluxes[sim_fluxes_base(q) + sim_held() + at];
}

fn sim_toward(k: i32) -> vec2<i32> {
    return select(vec2<i32>(0, 2 * (k - 2) - 1), vec2<i32>(2 * k - 1, 0), k < 2);
}

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

fn sim_seconds_at(q: Square, at: u32) -> f32 {
    if sim_has(q, SIM_LOCAL) {
        return sim_steps[sim_steps_base(q) + at].y;
    }
    return q.live;
}

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

fn sim_drained(q: Square, at: u32) -> f32 {
    if at == SIM_NO_BLOCK {
        return 1.0;
    }
    return sim_steps[sim_steps_base(q) + at].x;
}

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

fn sim_pull(gravity: vec2<f32>, depth: f32, rise: f32, squares: f32) -> f32 {
    return -0.5 * (gravity.x + gravity.y) * depth * rise + 0.5 * (gravity.y - gravity.x) * squares;
}

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
    let depth = max(vec2<f32>(staged.x) - 0.5 * vec2<f32>(ground_x.x + ground_x.y,
        ground_z.x + ground_z.y), vec2<f32>(0.0));
    let squares = sim_depths[sim_depths_base(q) + at];
    let gravity_x = vec2<f32>(sim_gravity_x(q, west_at, at), sim_gravity_x(q, at, east_at))
        * vec2<f32>(sim_opening(q, stage, west_at, at), sim_opening(q, stage, at, east_at));
    let gravity_z = vec2<f32>(sim_gravity_z(q, south_at, at), sim_gravity_z(q, at, north_at))
        * vec2<f32>(sim_opening(q, stage, south_at, at), sim_opening(q, stage, at, north_at));
    let pull = vec2<f32>(sim_pull(gravity_x, depth.x, ground_x.y - ground_x.x, squares.x),
        sim_pull(gravity_z, depth.y, ground_z.y - ground_z.x, squares.y)) / dx;
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
    sim_cells[written] = vec4<f32>(settled.xyz, 0.0);
    if stage == 1u && settled.w > 0.0 && sim_has(q, SIM_SUMMING) {
        sim_averages[sim_sums_at(q, at) + 3u].x += dt;
    }
    let first = at % SIM_BLOCK_CELLS == 0u;
    if stage == 0u && sim_has(q, SIM_SPARSE) && first {
        sim_steps[sim_block_at(q, at / SIM_BLOCK_CELLS)].x = 0.0;
    }
}

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
    let base = sim_cells_base(q);
    var water = sim_cells[base + at];
    let bed = sim_bed(q, at);
    let h = water.x - bed;
    if h <= SIM_DRY || sim_held_at(q, at) {
        return;
    }
    let middle = (vec2<f32>(cell) + 0.5) * q.dx;
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
        let carried = h * touch.velocity.xz;
        let flow = carried + (water.yz - carried) * exp(-min(filled / h, 1.0));
        water = vec4<f32>(water.x, flow, 0.0);
    }
    sim_cells[base + at] = water;
}

const SIM_EDDY_REACH: u32 = 3u;
const SIM_PUSH_SIDE: u32 = WATER_SIM_GROUP + 2u * SIM_EDDY_REACH;
const SIM_PUSH_CELLS: u32 = SIM_PUSH_SIDE * SIM_PUSH_SIDE;

fn sim_eddy_size(h: f32, dx: f32) -> f32 {
    return clamp(h, dx, f32(SIM_EDDY_REACH) * dx);
}

fn sim_eddy_speed(power: f32, h: f32, size: f32) -> f32 {
    return pow(max(power, 0.0) * size / (SIM_DENSITY * h), 1.0 / 3.0);
}

@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_burst(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let q = sim_square(group.z);
    let cell = sim_cell_of(q, group, count, local);
    if !cell.runs || !sim_has(q, SIM_BURSTING) {
        return;
    }
    let at = cell.at;
    let h = sim_cells[sim_cells_base(q) + at].x - sim_bed(q, at);
    var push = vec2<f32>(0.0);
    if h > SIM_THIN && !sim_held_at(q, at) && sim_active(q, at) {
        let size = sim_eddy_size(h, q.dx);
        let speed = sim_eddy_speed(sim_dissipation(q, cell), h, size);
        let mean = speed * q.live * q.dx * q.dx / (size * size * size);
        let place = vec2<i32>(i32(cell.i), i32(cell.j)) + sim_first(q);
        let burst = water_burst(place, sim_step(q), mean);
        if burst.starts {
            let heading = 6.2831853 * burst.turn;
            push = speed * vec2<f32>(cos(heading), sin(heading));
        }
    }
    sim_depths[sim_depths_base(q) + at] = push;
}

var<workgroup> sim_bursts: array<vec3<f32>, SIM_PUSH_CELLS>;

@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_push(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
) {
    let q = sim_square(group.z);
    let cell = sim_cell_of(q, group, count, local);
    let block = group.y * count.x + group.x;
    var gathering = group.z < sim_slot_count() && sim_has(q, SIM_STEPPING)
        && sim_has(q, SIM_BURSTING) && block < q.stored;
    if gathering {
        let place = sim_place(q, block);
        gathering = place.x < q.blocks.x && place.y < q.blocks.y;
    }
    if gathering {
        let first = vec2<i32>(sim_place(q, block) * WATER_SIM_GROUP)
            - vec2<i32>(i32(SIM_EDDY_REACH));
        let base = sim_cells_base(q);
        let pushes = sim_depths_base(q);
        for (var k = index; k < SIM_PUSH_CELLS; k += SIM_BLOCK_CELLS) {
            let beside = first + vec2<i32>(i32(k % SIM_PUSH_SIDE), i32(k / SIM_PUSH_SIDE));
            let at = sim_at(q, beside.x, beside.y);
            var burst = vec3<f32>(0.0);
            if at != SIM_NO_BLOCK {
                let push = sim_depths[pushes + at];
                if any(push != vec2<f32>(0.0)) {
                    let h = sim_cells[base + at].x - sim_bed(q, at);
                    burst = vec3<f32>(push, sim_eddy_size(h, q.dx));
                }
            }
            sim_bursts[k] = burst;
        }
    }
    workgroupBarrier();
    if !gathering || !cell.runs {
        return;
    }
    let at = cell.at;
    let written = sim_cells_base(q) + at;
    let water = sim_cells[written];
    let h = water.x - sim_bed(q, at);
    if h <= SIM_THIN || sim_held_at(q, at) {
        return;
    }
    let reach = i32(SIM_EDDY_REACH);
    let middle = vec2<i32>(local.xy) + vec2<i32>(reach);
    var push = vec2<f32>(0.0);
    for (var z = -reach; z <= reach; z++) {
        for (var x = -reach; x <= reach; x++) {
            let burst = sim_bursts[u32(middle.y + z) * SIM_PUSH_SIDE + u32(middle.x + x)];
            if burst.z <= 0.0 {
                continue;
            }
            let apart = length(vec2<f32>(f32(x), f32(z))) * q.dx / burst.z;
            if apart < 1.0 {
                let fall = 1.0 - apart * apart;
                push += burst.xy * (fall * fall);
            }
        }
    }
    let flow = water.yz + h * push;
    sim_cells[written].y = flow.x;
    sim_cells[written].z = flow.y;
}

fn sim_face_lost(flux: vec4<f32>, low: f32, high: f32) -> f32 {
    return abs(flux.x) * select(high, low, flux.x > 0.0) * flux.w;
}

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

const SIM_WET_CELLS: u32 = 1u;
const SIM_WET_WEST: u32 = 2u;
const SIM_WET_EAST: u32 = 4u;
const SIM_WET_SOUTH: u32 = 8u;
const SIM_WET_NORTH: u32 = 16u;

var<workgroup> sim_wet_sides: atomic<u32>;

const SIM_RIM_BLOCKS: u32 = (SIM_RIM + WATER_SIM_GROUP - 1u) / WATER_SIM_GROUP;
const SIM_RIM_SIDE: u32 = 2u * SIM_RIM_BLOCKS + 1u;
const SIM_RIM_AROUND: u32 = SIM_RIM_SIDE * SIM_RIM_SIDE;

var<workgroup> sim_rim_stands: array<u32, SIM_RIM_AROUND>;

fn sim_rim_inside(q: Square, cell: SimCell) -> u32 {
    var inside = min(min(cell.i, cell.j), min(q.nx - 1u - cell.i, q.nz - 1u - cell.j));
    let side = i32(WATER_SIM_GROUP);
    let local = vec2<i32>(vec2<u32>(cell.i, cell.j) % WATER_SIM_GROUP);
    for (var k = 0u; k < SIM_RIM_AROUND; k++) {
        if sim_rim_stands[k] == 0u {
            continue;
        }
        let step = vec2<i32>(i32(k % SIM_RIM_SIDE), i32(k / SIM_RIM_SIDE))
            - vec2<i32>(i32(SIM_RIM_BLOCKS));
        let low = local + (-step - vec2<i32>(1)) * side;
        let high = vec2<i32>(side - 1) - local + (step - vec2<i32>(1)) * side;
        let apart = select(select(vec2<i32>(0), low, step < vec2<i32>(0)), high,
            step > vec2<i32>(0));
        inside = min(inside, u32(max(max(apart.x, apart.y), 0)));
    }
    return inside;
}

fn sim_rim_pull(q: Square, cell: SimCell) -> f32 {
    let inside = sim_rim_inside(q, cell);
    if inside >= SIM_RIM {
        return 0.0;
    }
    let chi = f32(SIM_RIM - inside) / f32(SIM_RIM);
    return (exp(pow(chi, 3.5)) - 1.0) / (exp(1.0) - 1.0);
}

@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_ended(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
) {
    let q = sim_square(group.z);
    if index == 0u {
        atomicStore(&sim_wet_sides, 0u);
    }
    if index < SIM_RIM_AROUND {
        let block = group.y * count.x + group.x;
        var stands = false;
        if group.z < sim_slot_count() && sim_has(q, SIM_LIVE) && block < q.stored {
            let near = vec2<i32>(sim_place(q, block))
                + vec2<i32>(i32(index % SIM_RIM_SIDE), i32(index / SIM_RIM_SIDE))
                - vec2<i32>(i32(SIM_RIM_BLOCKS));
            stands = sim_stands(sim_table(q, near));
        }
        sim_rim_stands[index] = select(0u, 1u, stands);
    }
    workgroupBarrier();
    let cell = sim_cell_of(q, group, count, local);
    let at = cell.at;
    if cell.runs && sim_active(q, at) {
        let base = sim_cells_base(q);
        var water = sim_cells[base + at];
        if sim_has(q, SIM_LIVE) {
            water = mix(water, sim_averages[sim_looks(at)], sim_rim_pull(q, cell));
            sim_cells[base + at] = water;
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
    let first = index == 0u && all(group.xy == vec2<u32>(0u));
    if first && group.z < sim_slot_count() && sim_has(q, SIM_STEPPING) {
        sim_steps[sim_count_at(q)].x += 1.0;
    }
}

struct SimSums {
    water: vec4<f32>,
    motion: vec4<f32>,
    shares: vec4<f32>,
}

fn sim_add(q: Square, at: u32, seconds: f32, sums: SimSums) {
    let base = sim_sums_at(q, at);
    sim_averages[base] += seconds * sums.water;
    sim_averages[base + 1u] += seconds * sums.motion;
    sim_averages[base + 2u] += seconds * sums.shares;
    sim_averages[base + 3u].y += seconds;
}

fn sim_still(q: Square, at: u32, water: vec4<f32>) -> SimSums {
    let h = max(water.x - sim_bed(q, at), 0.0);
    let wet = select(0.0, 1.0, h > SIM_DRY);
    return SimSums(wet * vec4<f32>(h, water.yz, 0.0), vec4<f32>(0.0, wet * h * h, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, wet, 0.0));
}

fn sim_standing(q: Square) -> f32 {
    if sim_has(q, SIM_LOCAL) {
        return SIM_LOCAL_SHARE * SIM_COURANT * q.dx / SIM_SLOWEST;
    }
    return q.live;
}

fn sim_flow_at(q: Square, i: i32, j: i32, instead: vec2<f32>) -> vec2<f32> {
    let at = sim_at(q, i, j);
    if at == SIM_NO_BLOCK {
        return instead;
    }
    let water = sim_cells[sim_cells_base(q) + at];
    let h = max(water.x - sim_bed(q, at), 0.0);
    return vec2<f32>(sim_velocity(h, water.y), sim_velocity(h, water.z));
}

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
    let sums = SimSums(select(vec4<f32>(0.0), vec4<f32>(h, water.yz, 0.0), wet),
        select(vec4<f32>(0.0), moving, wet),
        vec4<f32>(select(vec2<f32>(0.0), velocity, wet), select(0.0, 1.0, wet),
            select(0.0, 1.0, leaves)));
    sim_add(q, at, sim_seconds_at(q, at), sums);
}

@compute @workgroup_size(WATER_SIM_SOURCE_GROUP)
fn sim_source(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = sim_square(id.z);
    if id.z >= sim_slot_count() || !sim_has(q, SIM_STEPPING) || id.x >= q.sourced {
        return;
    }
    let source = sim_inputs[sim_sources_base(q) + id.x];
    let at = bitcast<u32>(source.x);
    if sim_place(q, at / SIM_BLOCK_CELLS).x >= q.blocks.x || sim_held_at(q, at) {
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

fn sim_wet_at(q: Square, place: vec2<i32>) -> u32 {
    let block = sim_block(q, place);
    if block == SIM_NO_BLOCK {
        return 0u;
    }
    return u32(sim_steps[sim_block_at(q, block)].x);
}

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

var<workgroup> sim_look_water: array<vec4<f32>, SIM_BLOCK_CELLS>;
var<workgroup> sim_look_motion: array<vec4<f32>, SIM_BLOCK_CELLS>;
var<workgroup> sim_look_edges: array<vec2<f32>, SIM_BLOCK_CELLS>;

fn sim_at_edge(q: Square, i: i32, j: i32) -> bool {
    var edge = false;
    for (var k = 0; k < 4; k++) {
        let beside = vec2<i32>(i, j) + sim_toward(k);
        let inside = all(beside >= vec2<i32>(0)) && beside.x < i32(q.nx) && beside.y < i32(q.nz);
        edge = edge || (inside && sim_at(q, beside.x, beside.y) == SIM_NO_BLOCK);
    }
    return edge;
}

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
        sim_averages[means] = vec4<f32>(sim_bed(q, at) + mean.x, mean.yz, 0.0);
        sim_averages[means + 1u] = vec4<f32>(moving.x, heave, moving.z, rush);
        sim_averages[means + 2u] = vec4<f32>(shares.z / seconds, portions.w, spent.x / seconds,
            0.0);
        let kept = select(0.0, 1.0, sim_has(q, SIM_KEEPING));
        sim_averages[sums] = water * kept;
        sim_averages[sums + 1u] = motion * kept;
        sim_averages[sums + 2u] = shares * kept;
        sim_averages[sums + 3u] = spent * kept;
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
        if sparse {
            sim_steps[sim_block_at(q, block) + 1u].y = 0.0;
        }
    }
}

fn sim_shown(q: Square, i: i32, j: i32) -> vec4<f32> {
    let at = sim_at(q, i, j);
    if at == SIM_NO_BLOCK {
        return vec4<f32>(0.0, -1.0, 0.0, 0.0);
    }
    let water = sim_cells[sim_cells_base(q) + at];
    let h = water.x - sim_bed(q, at);
    if h <= SIM_THIN {
        return vec4<f32>(water.x, 0.0, 0.0, 0.0);
    }
    return vec4<f32>(water.x, h, sim_velocity(h, water.y), sim_velocity(h, water.z));
}

fn sim_change(before: vec4<f32>, here: vec2<f32>, after: vec4<f32>, dx: f32) -> vec2<f32> {
    let low = before.y > 0.0;
    let high = after.y > 0.0;
    if low && high {
        return (after.zw - before.zw) / (2.0 * dx);
    }
    if low {
        return (here - before.zw) / dx;
    }
    if high {
        return (after.zw - here) / dx;
    }
    return vec2<f32>(0.0);
}

@compute @workgroup_size(WATER_SIM_GROUP, WATER_SIM_GROUP)
fn sim_drawn(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let q = sim_square(0u);
    let cell = sim_cell_of(q, group, count, local);
    if !cell.runs || !sim_has(q, SIM_LIVE) {
        return;
    }
    let at = cell.at;
    let looks = sim_looks(at);
    let kept = sim_averages[looks + 1u];
    let span = bitcast<u32>(kept.z) & 0xff000000u;
    let reach = bitcast<u32>(kept.w) & 0xff000000u;
    let i = i32(cell.i);
    let j = i32(cell.j);
    let dx = q.dx;
    let ground = sim_bed(q, at);
    let here = sim_shown(q, i, j);
    var sides = array<vec4<f32>, 4>(sim_shown(q, i - 1, j), sim_shown(q, i + 1, j),
        sim_shown(q, i, j - 1), sim_shown(q, i, j + 1));
    var surface = SIM_NO_SURFACE;
    var velocity = vec2<f32>(0.0);
    var state = vec3<f32>(0.0);
    if here.y > 0.0 {
        surface = here.x;
        velocity = here.zw;
        let settled = sim_averages[looks];
        let settled_depth = settled.x - ground;
        var mean = vec2<f32>(0.0);
        if settled_depth > SIM_THIN {
            mean = vec2<f32>(sim_velocity(settled_depth, settled.y),
                sim_velocity(settled_depth, settled.z));
        }
        let along_x = sim_change(sides[0], velocity, sides[1], dx);
        let along_z = sim_change(sides[2], velocity, sides[3], dx);
        let swirl = abs(along_x.y - along_z.x);
        let turbulence = saturate(length(velocity - mean)
            / max(length(mean), SIM_TURBULENT_SPEED) + swirl / SIM_TURBULENT_SWIRL);
        let depth = max(here.y, SIM_BREAK_THINNEST);
        let eddies = sim_eddy_speed(sim_dissipation(q, cell), depth, depth);
        let breaks = eddies / sqrt(SIM_GRAVITY * depth);
        let leaves = SIM_GRAVITY + sim_bend(q, cell, velocity) <= 0.0;
        let born = saturate(smoothstep(SIM_BREAK_ONSET, SIM_BREAK_FULL, breaks)
            + select(0.0, 1.0, leaves));
        state = vec3<f32>(turbulence, born,
            sqrt(dot(along_x, along_x) + dot(along_z, along_z)));
    } else {
        var level = 0.0;
        var wet = 0.0;
        var carried = vec2<f32>(0.0);
        for (var k = 0; k < 4; k++) {
            let side = sides[k];
            if side.y > 0.0 {
                level += side.x;
                carried += side.zw;
                wet += 1.0;
            }
        }
        if wet > 0.0 && ground >= level / wet {
            surface = level / wet;
            velocity = carried / wet;
        }
    }
    let flowing = pack4x8snorm(vec4<f32>(velocity / SIM_FLOW_RANGE, 0.0, 0.0));
    let stirred = pack4x8unorm(vec4<f32>(state.xy, state.z / SIM_SHEAR_RANGE, 0.0));
    let words = vec2<u32>((flowing & 0x00ffffffu) | span, (stirred & 0x00ffffffu) | reach);
    sim_averages[looks + 1u] = vec4<f32>(surface, ground, bitcast<vec2<f32>>(words));
}
