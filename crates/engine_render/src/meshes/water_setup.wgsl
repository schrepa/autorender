import package::terrain::draw::{TerrainGroundRise, TerrainGroundSamples, terrain_ground_rise,
    terrain_ground_samples};
import package::terrain::surface::terrain_level_height;
import package::terrain::vertex::{TERRAIN_GRID, terrain};
import package::water::surface::{NO_WATER_BLOCK, NO_WATER_SURFACE, WATER_GRAVITY, WATER_HELD,
    WATER_LIVE_BLOCK, WATER_TILE_PATCHES, WATER_TURBULENT_SPEED, WaterCell, water_cell, water_info,
    water_read, water_texel};

struct SimSetup {
    counts: vec4<u32>,
    square: vec4<u32>,
    meters: vec4<f32>,
    ring: vec4<u32>,
    tiles: vec4<u32>,
}

@group(2) @binding(41) var<uniform> setup_frame: SimSetup;
@group(2) @binding(42) var<storage, read> setup_list: array<vec4<u32>>;
@group(2) @binding(43) var<storage, read_write> setup_ground: array<f32>;
@group(2) @binding(44) var<storage, read_write> setup_cells: array<vec4<f32>>;
@group(2) @binding(45) var<storage, read_write> setup_looks: array<vec4<f32>>;
@group(2) @binding(46) var<storage, read_write> setup_edges: array<vec4<f32>>;
@group(2) @binding(47) var<storage, read> setup_live: array<u32>;

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
const SETUP_SIDE: u32 = WATER_LIVE_BLOCK + 1u;
const SETUP_CORNERS: u32 = SETUP_SIDE * SETUP_SIDE;
const SETUP_CELLS: u32 = WATER_LIVE_BLOCK * WATER_LIVE_BLOCK;
const SETUP_POINTS: u32 = SETUP_COVER_POINTS * SETUP_COVER_POINTS;
const SETUP_NOISE_REACH: f32 = 1.7320508;
const SETUP_FARTHEST: f32 = 1.0e30;

struct SetupWater {
    surface: f32,
    flow: vec2<f32>,
    held: f32,
    rush: f32,
}

struct SetupLive {
    resident: bool,
    surface: f32,
    flow: vec2<f32>,
    held: bool,
}

fn setup_ground_at(corner: vec2<i32>) -> f32 {
    let local = vec2<f32>(corner) * setup_frame.meters.x;
    return (terrain.origin.y + terrain_level_height(local, 0u).height) - setup_frame.meters.y;
}

fn setup_raised(samples: TerrainGroundSamples, local: vec2<f32>, height: f32)
        -> TerrainGroundRise {
    return terrain_ground_rise(samples, local, height, setup_frame.meters.x);
}

fn setup_held(level: f32, risen: f32) -> f32 {
    return select(SETUP_UNHELD, level, level > risen);
}

fn setup_rough(rough: f32, relief: bool, cell: f32) -> f32 {
    if rough < 0.0 || relief {
        return -min(abs(rough), cell);
    }
    return rough;
}

fn setup_floor(at: vec2<i32>, by: i32) -> vec2<i32> {
    return (at - select(vec2<i32>(0), vec2<i32>(by - 1), at < vec2<i32>(0))) / by;
}

fn setup_holding(at: vec2<i32>) -> vec2<i32> {
    return setup_floor(at, i32(setup_frame.square.w));
}

fn setup_cell(at: vec2<i32>) -> WaterCell {
    return water_cell((vec2<f32>(at) + 0.5) * setup_frame.meters.x, 0u);
}

fn setup_nearest(cell: WaterCell) -> vec4<u32> {
    if cell.block == NO_WATER_BLOCK {
        return vec4<u32>(0u, 0u, 0u, bitcast<u32>(SETUP_GROUND_D84));
    }
    let near = select(vec2<u32>(0u), vec2<u32>(1u), cell.within >= vec2<f32>(0.5));
    let texel = water_texel(cell.block, cell.first + near);
    return textureLoad(water_info, texel.xy, texel.z, 0);
}

fn setup_live_cell(at: vec2<i32>) -> SetupLive {
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

fn setup_ghost(at: vec2<i32>) -> vec4<f32> {
    let live = setup_live_cell(at);
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

fn setup_speed(depth: f32, flow: vec2<f32>) -> f32 {
    let h4 = depth * depth * depth * depth;
    let thin = SETUP_THIN * SETUP_THIN * SETUP_THIN * SETUP_THIN;
    return 1.4142135 * depth * length(flow) / sqrt(h4 + max(h4, thin));
}

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

var<workgroup> setup_risen: array<f32, SETUP_CORNERS>;
var<workgroup> setup_reliefs: array<u32, SETUP_CORNERS>;

fn setup_corner(local: vec2<u32>, k: u32) -> vec2<u32> {
    return local + vec2<u32>(k & 1u, k >> 1u);
}

fn setup_corner_taken(local: vec2<u32>, k: u32) -> bool {
    let far = vec2<u32>(k & 1u, k >> 1u);
    return !any((far > vec2<u32>(0u)) & (local < vec2<u32>(WATER_LIVE_BLOCK - 1u)));
}

fn setup_cell_risen(local: vec2<u32>) -> vec2<f32> {
    let corner = local.y * SETUP_SIDE + local.x;
    let risen = 0.5 * (0.5 * (setup_risen[corner] + setup_risen[corner + SETUP_SIDE])
        + 0.5 * (setup_risen[corner + 1u] + setup_risen[corner + SETUP_SIDE + 1u]));
    let reliefs = setup_reliefs[corner] | setup_reliefs[corner + 1u]
        | setup_reliefs[corner + SETUP_SIDE] | setup_reliefs[corner + SETUP_SIDE + 1u];
    return vec2<f32>(risen, f32(reliefs));
}

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
    if listed.w == SETUP_WHOLE || listed.w == SETUP_RISEN {
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
    let live = setup_live_cell(at);
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
    let speed = setup_speed(live.surface - risen.x, live.flow);
    let rush = unpack4x8unorm(nearest.y).x * max(speed, WATER_TURBULENT_SPEED);
    let water = SetupWater(live.surface, live.flow, lake, rush);
    setup_cells[held] = setup_standing(water, risen.x, at, true);
    setup_looks[2u * held] = setup_standing(water, risen.x, at, false);
    setup_looks[2u * held + 1u] = vec4<f32>(0.0, 0.0, bitcast<f32>(nearest.x & 0xff000000u),
        bitcast<f32>(nearest.y & 0xff000000u));
}

fn setup_sample(window: u32, at: vec2<i32>) -> vec4<u32> {
    let side = i32(setup_frame.square.x);
    let held = clamp(at, vec2<i32>(0), vec2<i32>(side - 1));
    return setup_list[window + u32(held.y * side + held.x)];
}

fn setup_window(window: u32, start: vec2<i32>, local: vec2<f32>) -> TerrainGroundSamples {
    let at = max(local / terrain.anchor.w, vec2<f32>(0.0));
    let last = vec2<f32>(setup_frame.square.yz);
    let cell = min(floor(at), last - 1.0);
    let first = vec2<i32>(cell) - start;
    let x = setup_sample(window, first + vec2<i32>(0, 1));
    let y = setup_sample(window, first + vec2<i32>(1, 1));
    let z = setup_sample(window, first + vec2<i32>(1, 0));
    let w = setup_sample(window, first);
    return TerrainGroundSamples(vec4<u32>(x.x, y.x, z.x, w.x), vec4<u32>(x.y, y.y, z.y, w.y),
        min(at, last) - cell);
}

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
