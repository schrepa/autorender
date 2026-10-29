import package::engine::color::luminance;
import package::frame::encode::{encoded, stored};
import package::frame::view::{PI, camera, detail_across, detail_level, eye_world, lighting,
    stable_normalize};
import package::lighting::brdf::{distribution, environment_brdf, fresnel, visibility};
import package::lighting::sun::sun_visibility;
import package::lighting::surface::{BODY_WATER, Shaded, SurfacePoint, finish};
import package::meshes::declarations::{SceneTargets, TemporalTargets};
import package::meshes::surface::{scene_targets, temporal_targets};
import package::sky::air::{atmospheric_cover, sky_average};
import package::terrain::subdivision::TerrainSubdivision;
import package::terrain::surface::{surface_vertex, terrain_directory, terrain_level_height,
    terrain_surface_leaves};
import package::terrain::vertex::{TERRAIN_GRID, terrain, terrain_view};
import package::water::flow::{water_detail, water_stirring};
import package::water::foam::{water_blocking, water_contact, water_foam, water_foam_layers,
    water_layers, water_splatted};
import package::water::footprint::{water_footprint, water_roughness, water_turning};
import package::water::refraction::refraction_sampler;

@group(1) @binding(50) var water_surface: texture_2d_array<f32>;
@group(1) @binding(51) var water_info: texture_2d_array<u32>;
@group(1) @binding(60) var<uniform> water_tree: TerrainSubdivision;

struct WaterUniform {
    color: vec4<f32>,
    foam: vec4<f32>,
    absorption: vec4<f32>,
    sim: vec4<f32>,
    sim_datum: vec4<f32>,
    sediment: vec4<f32>,
    lift: vec4<f32>,
    sim_pixels: vec4<f32>,
}
@group(1) @binding(52) var<uniform> water: WaterUniform;
@group(1) @binding(56) var<storage, read> water_live: array<vec4<f32>>;
const WATER_LIVE_BLOCK: u32 = 8u;
const WATER_LIVE_LOOKS: u32 = 2u;
const WATER_LIVE_RIM: u32 = 8u;
@group(2) @binding(22) var water_depth: texture_depth_2d;
@group(2) @binding(23) var water_scene: texture_2d<f32>;

fn water_ground(pixel: vec2<f32>) -> vec3<f32> {
    let size = camera.region.xy;
    let depth = textureLoad(water_depth, vec2<i32>(pixel), 0);
    let ndc = vec2<f32>(pixel.x / size.x * 2.0 - 1.0, 1.0 - pixel.y / size.y * 2.0)
        - camera.jitter.xy;
    let point = camera.inverse_view_projection * vec4<f32>(ndc, depth, 1.0);
    return point.xyz / point.w - camera.position.xyz;
}

const WATER_BLOCK: u32 = 19u;
const WATER_BLOCKS_SIDE: u32 = 16u;
const NO_WATER_BLOCK: u32 = 0xffffffffu;
const NO_WATER_SURFACE: f32 = -1.0e30;
const WATER_TILE_PATCHES: u32 = 16u;
const WATER_TILE_WORDS: u32 = 256u;
const NOT_RESIDENT: u32 = 0xffffffffu;
const WATER_FLOW_RANGE: f32 = 8.0;
const WATER_SPAN_LEAST: f32 = 0.25;
const WATER_SPAN_MOST: f32 = 4096.0;
const WATER_SHEAR_RANGE: f32 = 8.0;
const WATER_GRADIENT_RANGE: f32 = 4.0;
const WATER_WET: f32 = 0.35;
const WATER_SOAKED: f32 = 0.03;
const WATER_IOR: f32 = 1.333;
const WATER_PHASE_G: f32 = 0.4;
const WATER_PHASE_MEAN: f32 = 0.4498;
const WATER_FOAM_ALBEDO: f32 = 0.65;
const WATER_FOAM_SKY: f32 = 0.6;
const WATER_FOAM_ROUGH: vec2<f32> = vec2<f32>(0.3, 0.7);
const WATER_REACH: f32 = 4.0;
const WATER_SHORE: f32 = 0.1;
const WATER_EDGE_PIXELS: f32 = 1.5;
const WATER_EDGE: f32 = 0.05;
const WATER_HELD: f32 = 0.5;
const WATER_RACING: vec2<f32> = vec2<f32>(0.8, 1.6);
const WATER_GRAVITY: f32 = 9.81;

struct WaterOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
}

fn water_texel(block: u32, sample: vec2<u32>) -> vec3<i32> {
    let blocks = WATER_BLOCKS_SIDE * WATER_BLOCKS_SIDE;
    let index = block % blocks;
    let origin = vec2<u32>(index % WATER_BLOCKS_SIDE, index / WATER_BLOCKS_SIDE) * WATER_BLOCK;
    return vec3<i32>(vec2<i32>(origin + sample), i32(block / blocks));
}

struct WaterWords {
    flow: vec2<f32>,
    span: f32,
    lake: bool,
    sheet: bool,
    state: vec3<f32>,
    reach: u32,
}

fn water_words(words: vec4<u32>) -> WaterWords {
    let info = unpack4x8snorm(words.x);
    let state = unpack4x8unorm(words.y);
    var read: WaterWords;
    read.flow = info.xy * WATER_FLOW_RANGE;
    read.span = water_span(info.w);
    read.lake = info.w < 0.0;
    read.sheet = info.z < 0.0;
    read.state = vec3<f32>(state.xy, state.z * WATER_SHEAR_RANGE);
    read.reach = words.y >> 24u;
    return read;
}

fn water_span(code: f32) -> f32 {
    let steps = round(abs(code) * 127.0);
    let octaves = log2(WATER_SPAN_MOST / WATER_SPAN_LEAST);
    return select(WATER_SPAN_LEAST * exp2(octaves * (steps - 1.0) / 126.0), 0.0, steps < 0.5);
}

fn water_extent() -> vec2<f32> {
    let header = terrain.directory.x;
    let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
    return vec2<f32>(tiles * (WATER_TILE_PATCHES * (TERRAIN_GRID - 1u))) * terrain.anchor.w;
}

struct WaterAt {
    surface: f32,
    depth: f32,
    held: f32,
    words: WaterWords,
    slope: vec2<f32>,
}

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

fn water_corner(corner: u32, upper: bool) -> vec2<u32> {
    let lower = vec2<u32>(select(0u, 1u, corner == 1u), select(0u, 1u, corner == 2u));
    return select(lower, vec2<u32>(1u) - lower, upper);
}

fn water_weights(within: vec2<f32>) -> vec3<f32> {
    if within.x + within.y <= 1.0 {
        return vec3<f32>(1.0 - within.x - within.y, within.x, within.y);
    }
    return vec3<f32>(within.x + within.y - 1.0, 1.0 - within.x, 1.0 - within.y);
}

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
    let origin = water_texel(cell.block, cell.first - vec2<u32>(1u));
    var held: array<vec2<f32>, 16>;
    for (var k = 0u; k < 16u; k++) {
        let steps = vec2<i32>(vec2<u32>(k % 4u, k / 4u));
        held[k] = textureLoad(water_surface, origin.xy + steps, origin.z, 0).xy;
    }
    return water_smooth(held, cell.within, terrain.anchor.w * f32(1u << cell.level));
}

fn water_smooth(surfaces: array<vec2<f32>, 16>, within: vec2<f32>, spacing: f32)
        -> WaterLevel {
    var samples = surfaces;
    var level: WaterLevel;
    let along = water_spline(within.x);
    let across = water_spline(within.y);
    var counted = 0.0;
    var counting = false;
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

fn water_at_level(level: WaterLevel) -> WaterAt {
    var at: WaterAt;
    at.surface = level.surface;
    at.slope = level.slope;
    at.depth = level.depth;
    at.held = level.held;
    return at;
}

fn water_words_add(at: WaterAt, words: WaterWords, share: f32, nearest: bool) -> WaterAt {
    var sum = at;
    sum.words.flow += words.flow * share;
    sum.words.span += words.span * share;
    sum.words.state += words.state * share;
    if nearest {
        sum.words.lake = words.lake;
        sum.words.sheet = words.sheet;
        sum.words.reach = words.reach;
    }
    return sum;
}

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

fn water_live_stands(block: vec2<i32>) -> bool {
    return water_live_look(block * i32(WATER_LIVE_BLOCK)).y <= 0.5 * NO_WATER_SURFACE;
}

fn water_live_standing(cells: vec2<f32>, n: f32) -> f32 {
    let side = f32(WATER_LIVE_BLOCK);
    let reach = i32((WATER_LIVE_RIM + WATER_LIVE_BLOCK - 1u) / WATER_LIVE_BLOCK);
    let blocks = i32(n) / i32(WATER_LIVE_BLOCK);
    let own = vec2<i32>(floor(cells / side));
    var nearest = f32(WATER_LIVE_RIM);
    for (var z = -reach; z <= reach; z++) {
        for (var x = -reach; x <= reach; x++) {
            let block = own + vec2<i32>(x, z);
            if any(block < vec2<i32>(0)) || any(block >= vec2<i32>(blocks)) {
                continue;
            }
            if !water_live_stands(block) {
                continue;
            }
            let low = vec2<f32>(block) * side;
            let apart = max(max(low - cells, cells - (low + side)), vec2<f32>(0.0));
            nearest = min(nearest, max(apart.x, apart.y));
        }
    }
    return nearest;
}

fn water_live_share(local: vec2<f32>, seen: f32) -> f32 {
    let n = water.sim.w;
    if n < f32(2u * WATER_LIVE_RIM) {
        return 0.0;
    }
    let handed = detail_level(seen, water.sim.z / (2.0 * water.sim_pixels.x), 2u);
    if handed.level != 0u {
        return 0.0;
    }
    let cells = (local + terrain.origin.xz - water.sim.xy) / water.sim.z;
    let edges = min(min(cells.x, cells.y), min(n - cells.x, n - cells.y));
    if edges <= 2.0 {
        return 0.0;
    }
    let inside = min(edges, water_live_standing(cells, n));
    let rim = smoothstep(2.0, f32(WATER_LIVE_RIM), inside);
    return rim * (1.0 - handed.blend);
}

fn water_blend(first: WaterAt, second: WaterAt, toward: f32) -> WaterAt {
    var at = first;
    if toward >= 0.5 {
        at = second;
    }
    at.surface = mix(first.surface, second.surface, toward);
    at.depth = mix(first.depth, second.depth, toward);
    at.held = mix(first.held, second.held, toward);
    at.words.flow = mix(first.words.flow, second.words.flow, toward);
    at.words.span = mix(first.words.span, second.words.span, toward);
    at.words.state = mix(first.words.state, second.words.state, toward);
    at.slope = mix(first.slope, second.slope, toward);
    return at;
}

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

fn water_drawn(drawn: vec3<f32>) -> WaterAt {
    let world = eye_world(drawn);
    return water_pixel(world.xz - terrain.origin.xz, detail_across(world));
}

fn water_on_surface(drawn: vec3<f32>, at: WaterAt) -> vec3<f32> {
    let above = eye_world(drawn).y - terrain.origin.y - at.surface;
    return drawn - vec3<f32>(0.0, above, 0.0);
}

fn water_relative(local: vec2<f32>, height: f32) -> vec3<f32> {
    let from_eye = terrain.origin.xz - terrain.eye.xz;
    return vec3<f32>(local.x + from_eye.x, height + terrain.anchor.y, local.y + from_eye.y)
        + terrain_view.eye_offset.xyz;
}

fn water_lift(world: vec3<f32>) -> f32 {
    return (water.lift.x + water.lift.y) * detail_across(world);
}

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

fn water_previous(relative: vec3<f32>, flow: vec2<f32>, share: f32) -> vec3<f32> {
    let seconds = clamp(camera.position.w - camera.wind.w, 0.0, WATER_LONGEST_FRAME);
    let before = eye_world(relative) - vec3<f32>(flow.x, 0.0, flow.y) * (seconds * share);
    return (camera.previous_view_projection * vec4<f32>(before, 1.0)).xyw;
}

const WATER_LONGEST_FRAME: f32 = 0.1;

const WATER_BUBBLES: f32 = 3.0;
const WATER_BUBBLE_ALBEDO: f32 = 0.2;

fn water_extinction(aerated: f32) -> vec3<f32> {
    return water.absorption.rgb + water.sediment.w + WATER_BUBBLES * aerated;
}

fn water_transmitted(meters: f32, aerated: f32) -> vec3<f32> {
    return exp(-water_extinction(aerated) * meters);
}

fn water_body(aerated: f32) -> vec3<f32> {
    let own = luminance(water.absorption.rgb);
    let sediment = water.sediment.w;
    let bubbles = WATER_BUBBLES * aerated;
    return (water.color.rgb * own + water.sediment.rgb * sediment
        + vec3<f32>(WATER_BUBBLE_ALBEDO * bubbles)) / max(own + sediment + bubbles, 1.0e-6);
}

fn water_facing(normal: vec3<f32>, view: vec3<f32>, plane: vec3<f32>) -> vec3<f32> {
    let mirrored = reflect(-view, normal);
    if dot(view, plane) <= 0.0 || mirrored.y >= 0.0 {
        return normal;
    }
    return normalize(view + vec3<f32>(mirrored.x, -mirrored.y, mirrored.z));
}

fn water_heading(view: vec3<f32>) -> vec2<f32> {
    let away = -view.xz;
    let squared = dot(away, away);
    let laid = away * inverseSqrt(max(squared, 1.0e-30));
    return select(vec2<f32>(1.0, 0.0), laid, squared > 1.0e-12); // optics: the view's frame on water
}

fn water_view_tangent(normal: vec3<f32>, view: vec3<f32>) -> vec3<f32> {
    let laid = normal * dot(view, normal) - view;
    let squared = dot(laid, laid);
    let side = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.x) > 0.5);
    let tangent = laid * inverseSqrt(max(squared, 1.0e-30));
    let any_tangent = normalize(side - normal * dot(side, normal));
    return select(any_tangent, tangent, squared > 1.0e-12); // optics: the view's frame on water
}

fn water_distribution(along: f32, across: f32, n_h: f32, alphas: vec2<f32>) -> f32 {
    let a2 = alphas.x * alphas.y;
    let v = vec3<f32>(alphas.y * along, alphas.x * across, a2 * n_h);
    let w2 = a2 / dot(v, v);
    return a2 * w2 * w2 / PI;
}

fn water_visibility(n_v: f32, n_l: f32, view_tangents: vec2<f32>, light_tangents: vec2<f32>,
        alphas: vec2<f32>) -> f32 {
    let lambda_v = n_l * length(vec3<f32>(alphas * view_tangents, n_v));
    let lambda_l = n_v * length(vec3<f32>(alphas * light_tangents, n_l));
    return 0.5 / (lambda_v + lambda_l);
}

fn water_phase(cosine: f32) -> f32 {
    let g = WATER_PHASE_G;
    let denominator = 1.0 + g * g - 2.0 * g * cosine;
    return (1.0 - g * g) / (denominator * sqrt(denominator)) / WATER_PHASE_MEAN;
}

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

struct WaterBehind {
    color: vec3<f32>,
    ground: vec3<f32>,
}

fn water_behind(relative: vec3<f32>, normal: vec3<f32>, view: vec3<f32>, pixel: vec2<f32>,
        ground: vec3<f32>, plane: vec3<f32>) -> WaterBehind {
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

struct WaterSurface {
    relative: vec3<f32>,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    normal: vec3<f32>,
    roughness: vec2<f32>,
    plane: vec3<f32>,
    foam_shade: f32,
    foam_thin: f32,
    foam: f32,
    scatter: vec3<f32>,
    transmit: vec3<f32>,
}

struct WaterLit {
    shaded: Shaded,
    reflectance: f32,
    sunny: f32,
}

fn water_light(surface: WaterSurface) -> WaterLit {
    let world = eye_world(surface.relative);
    let distance = length(surface.relative);
    let view = stable_normalize(-surface.relative);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let normal = water_facing(surface.normal, view, surface.plane);
    let foam_roughness = mix(WATER_FOAM_ROUGH.y, WATER_FOAM_ROUGH.x, surface.foam_thin);
    let roughness = mix(surface.roughness, vec2<f32>(foam_roughness), surface.foam);
    let alphas = roughness * roughness;
    let n_v = max(dot(normal, view), 1.0e-4);
    let f0 = vec3<f32>(0.02);
    let reflectance = saturate(luminance(environment_brdf(f0, roughness.x, n_v)));
    let sun = lighting.sun.xyz;
    var sunlight = vec3<f32>(0.0);
    if sun.y > 0.0 {
        sunlight = lighting.sun_color.rgb
            * sun_visibility(world, up, surface.world_dx, surface.world_dy, distance, false);
    }
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
    let body = surface.scatter * (vec3<f32>(1.0) - surface.transmit)
        * water_scattered(view, sunlight, surface.plane) * (1.0 - reflectance);
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
    point.body = BODY_WATER;
    var lit: WaterLit;
    lit.shaded = finish(point, mix(body + glint, foam, surface.foam));
    lit.reflectance = reflectance;
    let sunlit = luminance(sunlight) * max(sun.y, 0.0);
    lit.sunny = sunlit / max(sunlit + luminance(sky), 1.0e-6);
    return lit;
}

struct WaterShade {
    shaded: Shaded,
    coverage: f32,
    behind: vec3<f32>,
    fade: f32,
    flow: vec2<f32>,
}

fn water_covering(shade: WaterShade, pixel: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(stored(shade.shaded.color + shade.behind, pixel) * shade.fade, shade.fade);
}

const WATER_CAUSTIC_CONTRAST: f32 = 0.3;
const WATER_CAUSTIC_MOST: f32 = 1.5;
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

fn water_edge(meters: f32, change: f32) -> f32 {
    return max(saturate(meters / max(change * WATER_EDGE_PIXELS, 1.0e-6)),
        smoothstep(0.0, WATER_EDGE, meters));
}

fn water_ends(held: f32, change: f32) -> f32 {
    return saturate((held - WATER_HELD) / max(change * WATER_EDGE_PIXELS, 1.0e-6));
}

fn water_ramp(meters: f32, width: f32, change: f32) -> f32 {
    let half = min(0.5 * change, max(meters, 0.0));
    if half < 1.0e-3 * width {
        return smoothstep(0.0, width, meters);
    }
    return (water_ramped(meters + half, width) - water_ramped(meters - half, width))
        / (2.0 * half);
}

fn water_ramped(meters: f32, width: f32) -> f32 {
    let u = saturate(meters / width);
    return width * u * u * u * (1.0 - 0.5 * u) + max(meters - width, 0.0);
}

fn shade_water(point: vec3<f32>, pixel: vec2<f32>, at: WaterAt, refracts: bool)
        -> WaterShade {
    let world = eye_world(point);
    if at.held < WATER_HELD {
        discard;
    }
    let plane = normalize(vec3<f32>(at.slope.x, 1.0, at.slope.y));
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
    let turbulence = at.words.state.x;
    let speed = length(flow);
    let stirring = water_stirring(speed, water_eddies(speed, length(at.slope), depth),
        turbulence);
    let footprint = water_footprint(world_dx.xz, world_dy.xz);
    let splatted = water_splatted(world, footprint);
    let detail = water_detail(world, flow, depth, stirring, detail_across(world),
        camera.position.w);
    let fall = at.slope + detail.slope + splatted.fall;
    let normal = normalize(vec3<f32>(fall.x, 1.0, fall.y));
    let view = stable_normalize(-point);
    let unresolved = vec3<f32>(0.5 * detail.hidden, 0.5 * detail.hidden, 0.0)
        + splatted.hidden * splatted.shown;
    let roughness = water_roughness(unresolved + water_turning(fall), water_heading(view));
    var behind: WaterBehind;
    behind.ground = own;
    behind.color = vec3<f32>(0.0);
    if refracts {
        behind = water_behind(point, normal, view, pixel, own, plane);
    }
    let bed_depth = min(depth, standing);
    let gathered = water_caustics(world, normal, bed_depth);
    let racing = smoothstep(WATER_RACING.x, WATER_RACING.y, speed);
    let shallows = mix(water_ramp(standing, WATER_SHORE, standing_change), 1.0, racing);
    let fade = min(water_edge(standing, standing_change), water_ends(at.held, fwidth(at.held)))
        * shallows;
    let through = min(max(length(behind.ground - point), 0.0), // optics: the view's way through the water
        depth * plane.y / max(dot(view, plane), 0.1)); // optics: the view's way through the water
    let down = clamp(dot(point - behind.ground, plane) / plane.y, 0.0, through); // optics: the light's way down to what the view meets
    let contact = water_contact(standing, standing_change, footprint, speed,
        water_blocking(standing, depth));
    let layers = water_foam_layers(pixel);
    let white = layers.x + water_layers(contact);
    let foam = water_foam(white, layers.y);
    let cover = foam.cover;
    let aerated = 1.0 - exp(-white);
    let transmit = water_transmitted(through + down, aerated);
    var surface: WaterSurface;
    surface.relative = point;
    surface.world_dx = world_dx;
    surface.world_dy = world_dy;
    surface.normal = normal;
    surface.roughness = roughness;
    surface.plane = plane;
    surface.foam_shade = foam.shade;
    surface.foam_thin = foam.thin;
    surface.foam = cover;
    surface.scatter = water_body(aerated);
    surface.transmit = transmit;
    let lit = water_light(surface);
    let caustic = 1.0 + (gathered - 1.0) * lit.sunny
        * exp(-bed_depth * (1.0 / WATER_CAUSTIC_DEPTH + WATER_BUBBLES * aerated));
    let wet = water_ramp(standing, WATER_SOAKED, standing_change) * WATER_WET;
    let let_through = (1.0 - lit.reflectance) * (1.0 - cover);
    var result: WaterShade;
    result.shaded = lit.shaded;
    result.coverage = max(max(1.0 - (1.0 - lit.reflectance) * luminance(transmit), wet),
        cover) * fade;
    result.behind = behind.color * transmit * (caustic * let_through * (1.0 - wet)
        * (1.0 - atmospheric_cover(camera.position.xyz, world)));
    result.fade = fade;
    result.flow = flow;
    return water_mirroring(result);
}

fn water_mirroring(shade: WaterShade) -> WaterShade {
    var result = shade;
    let own = luminance(shade.shaded.color);
    result.shaded.mirrored *= shade.fade * own / max(own + luminance(shade.behind), 1.0e-6);
    result.shaded.reflect *= shade.fade;
    return result;
}

fn shade_mesh(mesh: vec3<f32>, pixel: vec2<f32>, refracts: bool) -> WaterShade {
    let at = water_drawn(mesh);
    return shade_water(water_on_surface(mesh, at), pixel, at, refracts);
}

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
fn fs_water_temporal(input: WaterOutput) -> TemporalTargets {
    let shade = shade_mesh(input.relative, input.clip_position.xy, true);
    var targets = temporal_targets(shade.shaded, input.clip_position,
        water_previous(input.relative, shade.flow, saturate(shade.coverage)));
    targets.color = water_covering(shade, input.clip_position.xy);
    return targets;
}

const WATER_TURBULENT_SPEED: f32 = 0.5;
const WATER_BREAK_THINNEST: f32 = 0.02;
const WATER_BREAK_ONSET: f32 = 0.6;
const WATER_BREAK_FULL: f32 = 1.2;

fn water_eddies(speed: f32, slope: f32, depth: f32) -> f32 {
    let stirring = WATER_GRAVITY * speed * slope * max(depth, WATER_BREAK_THINNEST);
    return select(0.0, pow(stirring, 1.0 / 3.0), stirring > 0.0);
}

fn water_eddy_froude(speed: f32, slope: f32, depth: f32) -> f32 {
    return water_eddies(speed, slope, depth)
        / sqrt(WATER_GRAVITY * max(depth, WATER_BREAK_THINNEST));
}
