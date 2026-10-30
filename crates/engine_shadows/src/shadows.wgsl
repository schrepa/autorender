import package::engine::wind::{slot_sway, wind_sway};
import package::meshes::declarations::SlotColor;
import package::meshes::impostor_frames::{
    IMPOSTOR_HEMISPHERE, ImpostorRecord, impostor_basis, impostor_frame_direction, impostor_frames,
    impostor_grid, impostor_index,
};
import package::presentation::cull::{Batch, Candidate, Rows, VALID};

struct Cull {
    light: array<vec4<f32>, 3>,
    regions: array<vec4<f32>, REGIONS>,
    region_depths: array<vec4<f32>, REGIONS>,
    windows: array<vec4<f32>, SHADOW_LEVELS>,
    candidates: u32,
    groups: u32,
    capacity: u32,
    mode: u32,
    counts: vec4<u32>,
    words: vec4<u32>,
    moving: vec4<u32>,
    sky: array<vec4<f32>, 3>,
    sky_rect: vec4<f32>,
    sky_depths: vec4<f32>,
}

const REGIONS: u32 = 7u;
const SHADOW_LEVELS: u32 = 20u;

struct DrawArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform> cull: Cull;
@group(0) @binding(1) var<storage, read> candidates: array<Candidate>;
@group(0) @binding(2) var<storage, read> batches: array<Batch>;
@group(0) @binding(3) var<storage, read> poses: array<Rows>;
@group(0) @binding(4) var<storage, read> flags: array<u32>;
@group(0) @binding(5) var<storage, read_write> stillness: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read_write> ids: array<u32>;
@group(0) @binding(7) var<storage, read_write> args: array<DrawArgs>;
struct Counters {
    totals: array<atomic<u32>, TOTALS>,
    words: array<atomic<u32>>,
}
@group(0) @binding(8) var<storage, read_write> counters: Counters;
@group(0) @binding(12) var tile_slots: texture_2d<u32>;

const STILLNESS: u32 = 1u;
const SKY_REFRESH: u32 = 2u;
const KEEP_AFTER: u32 = 8u;
const MOVING_UNTIL: u32 = 64u;
const KEEP_AFTER_RESTLESS: u32 = 180u;
const MOVING_UNTIL_RESTLESS: u32 = 236u;
const LEVELED: u32 = 0x2000000u;
const RESTLESS: u32 = 0x80000000u;
const FOOTPRINT_UNIT: f32 = 4.0;
const MOVING_MAP: u32 = 7u;
const SKY_MAP: u32 = 8u;
const MOVING_PAIRS: u32 = 8u;
const MOVING_TILES: u32 = 256u;
const TOTALS: u32 = 4u;

fn mix_key(key: u32, value: u32) -> u32 {
    var x = (key ^ value) * 0x9e3779b1u;
    return x ^ (x >> 15u);
}

fn pose_key(pose: Rows, group: u32) -> u32 {
    var key = mix_key(0x2545f491u, group);
    for (var row = 0u; row < 3u; row++) {
        var values: vec4<f32>;
        switch row {
            case 0u: { values = pose.r0; }
            case 1u: { values = pose.r1; }
            default: { values = pose.r2; }
        }
        let bits = bitcast<vec4<u32>>(values);
        key = mix_key(mix_key(mix_key(mix_key(key, bits.x), bits.y), bits.z), bits.w);
    }
    return key;
}

fn overlaps(rectangle: vec4<f32>, low: vec3<f32>, high: vec3<f32>, depth: f32) -> bool {
    return all(high.xy >= rectangle.xy) && all(low.xy <= rectangle.zw) && low.z <= depth;
}

fn group_start(group: u32) -> u32 {
    return atomicLoad(&counters.words[cull.words.w + group]);
}

fn group_end(group: u32) -> u32 {
    if group + 1u < cull.groups {
        return group_start(group + 1u);
    }
    return cull.capacity;
}

fn map_start(map: u32) -> u32 {
    if map <= MOVING_MAP {
        return map * cull.capacity;
    }
    return (map + MOVING_PAIRS - 1u) * cull.capacity;
}

fn append(map: u32, group: u32, slot: u32) {
    let offset = atomicAdd(&args[map * cull.groups + group].instance_count, 1u);
    ids[map_start(map) + group_start(group) + offset] = slot;
}

fn append_moving(group: u32, slot: u32, tile: u32) {
    let draw = MOVING_MAP * cull.groups + group;
    let offset = atomicAdd(&args[draw].instance_count, 1u);
    let start = group_start(group);
    if offset >= MOVING_PAIRS * (group_end(group) - start) {
        atomicSub(&args[draw].instance_count, 1u);
        return;
    }
    ids[map_start(MOVING_MAP) + MOVING_PAIRS * start + offset] = slot | (tile << 24u);
}

fn mark_tiles(mask: u32, low: vec2<f32>, high: vec2<f32>, moving: bool, group: u32,
        slot: u32) {
    if !(all(abs(low) < vec2<f32>(1.0e9)) && all(abs(high) < vec2<f32>(1.0e9))) {
        return;
    }
    let side = cull.counts.w;
    let last = i32(side) - 1;
    for (var level = 0u; level < cull.counts.y; level++) {
        let window = cull.windows[level];
        let first = vec2<i32>(round(window.xy / window.z));
        let from_tile = max(vec2<i32>(floor(low / window.z)), first);
        let to_tile = min(vec2<i32>(floor(high / window.z)), first + vec2<i32>(last));
        for (var row = from_tile.y; row <= to_tile.y; row++) {
            for (var column = from_tile.x; column <= to_tile.x; column++) {
                let cell = vec2<u32>(vec2<i32>(column, row) & vec2<i32>(last));
                let entry = textureLoad(tile_slots, vec2<u32>(cell.x, level * side + cell.y),
                    0).r;
                if (entry & 0xffffu) == 0u {
                    continue;
                }
                let bit = (level * side + cell.y) * side + cell.x;
                let word = mask + bit / 32u;
                let flag = 1u << (bit % 32u);
                if (atomicLoad(&counters.words[word]) & flag) == 0u {
                    atomicOr(&counters.words[word], flag);
                }
                let tile = entry >> 16u;
                if moving && tile != 0u {
                    append_moving(group, slot, tile - 1u - cull.moving.x);
                }
            }
        }
    }
}

fn footprint_word(value: vec2<i32>) -> u32 {
    return (u32(value.x) & 0xffffu) | (u32(value.y) << 16u);
}

fn footprint_corner(word: u32) -> vec2<f32> {
    return vec2<f32>(f32(i32(word << 16u) >> 16u), f32(i32(word) >> 16u)) * FOOTPRINT_UNIT;
}

fn light_box(center: vec3<f32>, half: vec3<f32>) -> array<vec3<f32>, 2> {
    let light_center = vec3<f32>(dot(cull.light[0].xyz, center),
        dot(cull.light[1].xyz, center), dot(cull.light[2].xyz, center));
    let light_half = vec3<f32>(dot(abs(cull.light[0].xyz), half),
        dot(abs(cull.light[1].xyz), half), dot(abs(cull.light[2].xyz), half));
    return array<vec3<f32>, 2>(light_center - light_half, light_center + light_half);
}

@compute @workgroup_size(256)
fn select_casters(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= cull.candidates {
        return;
    }
    let candidate = candidates[id.x];
    let classify = (cull.mode & STILLNESS) != 0u;
    if classify && (flags[candidate.position] & VALID) == 0u {
        return;
    }
    let pose = poses[candidate.slot];
    let batch = batches[candidate.batch];
    if batch.group >= cull.groups {
        return;
    }
    let center = (batch.bounds_min.xyz + batch.bounds_max.xyz) * 0.5;
    let half = (batch.bounds_max.xyz - batch.bounds_min.xyz) * 0.5;
    let world_center = vec3<f32>(dot(pose.r0.xyz, center) + pose.r0.w,
        dot(pose.r1.xyz, center) + pose.r1.w, dot(pose.r2.xyz, center) + pose.r2.w);
    let world_half = vec3<f32>(dot(abs(pose.r0.xyz), half), dot(abs(pose.r1.xyz), half),
        dot(abs(pose.r2.xyz), half));
    let bounds = light_box(world_center, world_half);
    let low = bounds[0];
    let high = bounds[1];
    var moving = true;
    var kept = false;
    if classify {
        let key = pose_key(pose, batch.group);
        let state = stillness[candidate.slot];
        var count = state.y & 0xffu;
        var leveled = state.y & LEVELED;
        var footprint = state.zw;
        var restless = state.y & RESTLESS;
        var keep_after = select(KEEP_AFTER, KEEP_AFTER_RESTLESS, restless != 0u);
        if state.x == key {
            count = min(count + 1u, 255u);
        } else {
            if count >= keep_after && leveled != 0u {
                mark_tiles(cull.words.y, footprint_corner(footprint.x),
                    footprint_corner(footprint.y), false, 0u, 0u);
                restless = RESTLESS;
                keep_after = KEEP_AFTER_RESTLESS;
            }
            count = 0u;
            leveled = 0u;
        }
        if count == keep_after {
            mark_tiles(cull.words.y, low.xy, high.xy, false, 0u, 0u);
        }
        moving = count < select(MOVING_UNTIL, MOVING_UNTIL_RESTLESS, restless != 0u);
        kept = count >= keep_after;
        if kept {
            leveled = LEVELED;
            footprint = vec2<u32>(
                footprint_word(vec2<i32>(floor(low.xy / FOOTPRINT_UNIT))),
                footprint_word(vec2<i32>(ceil(high.xy / FOOTPRINT_UNIT))));
        }
        let packed = count | leveled | restless;
        if state.x != key || state.y != packed || any(state.zw != footprint) {
            stillness[candidate.slot] = vec4<u32>(key, packed, footprint);
        }
    }
    if kept {
        for (var region = 0u; region < cull.counts.x; region++) {
            if overlaps(cull.regions[region], low, high, cull.region_depths[region].x) {
                append(region, batch.group, candidate.slot);
            }
        }
    }
    if moving {
        mark_tiles(cull.words.z, low.xy, high.xy, true, batch.group, candidate.slot);
    }
    if (cull.mode & SKY_REFRESH) != 0u && (kept || !classify) {
        let sky_center = vec3<f32>(dot(cull.sky[0].xyz, world_center),
            dot(cull.sky[1].xyz, world_center), dot(cull.sky[2].xyz, world_center));
        let sky_half = vec3<f32>(dot(abs(cull.sky[0].xyz), world_half),
            dot(abs(cull.sky[1].xyz), world_half), dot(abs(cull.sky[2].xyz), world_half));
        if overlaps(cull.sky_rect, sky_center - sky_half, sky_center + sky_half,
            cull.sky_depths.x) {
            append(SKY_MAP, batch.group, candidate.slot);
        }
    }
}

const COUNT_LANES: u32 = 256u;
var<workgroup> cast_totals: array<atomic<u32>, 2>;

@compute @workgroup_size(256)
fn count_casters(@builtin(local_invocation_index) lane: u32) {
    if lane < 2u {
        atomicStore(&cast_totals[lane], 0u);
    }
    workgroupBarrier();
    var totals = vec2<u32>(0u);
    let draws = SKY_MAP * cull.groups;
    for (var draw = lane; draw < draws; draw += COUNT_LANES) {
        let map = draw / cull.groups;
        totals[select(0u, 1u, map == MOVING_MAP)] += atomicLoad(&args[draw].instance_count);
    }
    for (var index = 0u; index < 2u; index++) {
        if totals[index] != 0u {
            atomicAdd(&cast_totals[index], totals[index]);
        }
    }
    workgroupBarrier();
    if lane < 2u {
        atomicStore(&counters.totals[lane], atomicLoad(&cast_totals[lane]));
    }
}

struct View {
    clip: mat4x4<f32>,
    wind: vec4<f32>,
    eye: vec4<f32>,
}

@group(0) @binding(9) var<uniform> view: View;
struct Routes {
    clips: array<mat4x4<f32>, MOVING_TILES>,
    slots: array<vec4<f32>, MOVING_TILES>,
    atlas: vec4<f32>,
}
@group(0) @binding(13) var<storage, read> routes: Routes;
@group(1) @binding(0) var<storage, read> slot_poses: array<Rows>;

fn flattened(clip: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(clip.xy, max(clip.z, 0.0), clip.w);
}

fn project(world: vec3<f32>) -> vec4<f32> {
    return flattened(view.clip * vec4<f32>(world, 1.0));
}

fn route(world: vec3<f32>, tile: u32) -> vec4<f32> {
    return flattened(routes.clips[tile] * vec4<f32>(world, 1.0));
}

fn outside_slot(position: vec2<f32>, tile: u32) -> bool {
    let slot = routes.slots[tile];
    return any(position < slot.xy) || any(position >= slot.zw);
}

fn slot_world(slot: u32, position: vec3<f32>) -> vec3<f32> {
    let pose = slot_poses[slot];
    let point = vec4<f32>(position, 1.0);
    let world = vec3<f32>(dot(pose.r0, point), dot(pose.r1, point), dot(pose.r2, point));
    let sway = slot_sway(slot_colors[slot].flags);
    if sway == 0.0 {
        return world;
    }
    let root = vec3<f32>(pose.r0.w, pose.r1.w, pose.r2.w);
    let pixels = view.eye.w / max(length(root - view.eye.xyz), 1.0e-6);
    return world + wind_sway(view.wind, pixels, root, world.y - root.y, position, sway);
}

@vertex
fn vs_affine(@location(0) position: vec3<f32>, @location(2) slot: u32) -> @builtin(position) vec4<f32> {
    return project(slot_world(slot, position));
}

const MOVING_SLOT: u32 = 0xffffffu;

struct MovingOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) tile: u32,
}

@vertex
fn vs_moving(@location(0) position: vec3<f32>, @location(2) pair: u32) -> MovingOutput {
    var output: MovingOutput;
    output.tile = pair >> 24u;
    output.clip = route(slot_world(pair & MOVING_SLOT, position), output.tile);
    return output;
}

@fragment
fn fs_moving(input: MovingOutput) {
    if outside_slot(input.clip.xy, input.tile) {
        discard;
    }
}

@vertex
fn vs_clear_moving(@builtin(vertex_index) vertex: u32,
        @builtin(instance_index) tile: u32) -> @builtin(position) vec4<f32> {
    let slot = routes.slots[tile];
    let texel = mix(slot.xy, slot.zw, CARD[vertex] * 0.5 + vec2<f32>(0.5));
    return vec4<f32>(texel.x * routes.atlas.x - 1.0, 1.0 - texel.y * routes.atlas.y, 1.0, 1.0);
}

@group(1) @binding(3) var shadow_regions: texture_depth_2d_array;

struct ComposeOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) source: vec3<i32>,
}

@vertex
fn vs_compose(@builtin(vertex_index) vertex: u32, @location(3) place: vec4<f32>,
        @location(4) source: vec4<i32>) -> ComposeOutput {
    let texel = mix(place.xy, place.zw, CARD[vertex] * 0.5 + vec2<f32>(0.5));
    var output: ComposeOutput;
    output.clip = vec4<f32>(texel.x * view.clip[0].x - 1.0, 1.0 - texel.y * view.clip[0].y,
        1.0, 1.0);
    output.source = source.xyz;
    return output;
}

@fragment
fn fs_compose(input: ComposeOutput) -> @builtin(frag_depth) f32 {
    let texel = vec2<i32>(input.clip.xy) + input.source.yz;
    return textureLoad(shadow_regions, texel, input.source.x, 0);
}

@group(1) @binding(1) var<storage, read> slot_colors: array<SlotColor>;
@group(2) @binding(0) var base_color_texture: texture_2d<f32>;
@group(2) @binding(1) var base_color_sampler: sampler;
struct MaterialMaps {
    transforms: array<vec4<f32>, 10>,
    emissive: vec4<f32>,
    params: vec4<f32>,
}
@group(2) @binding(10) var<uniform> material_maps: MaterialMaps;

struct MaskedOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv_alpha: vec3<f32>,
}

fn masked_output(clip: vec4<f32>, uv: vec2<f32>, color: vec2<u32>) -> MaskedOutput {
    var output: MaskedOutput;
    output.clip = clip;
    output.uv_alpha = vec3<f32>(uv, unpack2x16unorm(color.y).y);
    return output;
}

@vertex
fn vs_affine_masked(@location(0) position: vec3<f32>, @location(7) uv: vec2<f32>,
        @location(2) slot: u32) -> MaskedOutput {
    return masked_output(project(slot_world(slot, position)), uv, slot_colors[slot].color);
}

struct MovingMaskedOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv_alpha: vec3<f32>,
    @location(1) @interpolate(flat) tile: u32,
}

fn moving_masked_output(masked: MaskedOutput, tile: u32) -> MovingMaskedOutput {
    var output: MovingMaskedOutput;
    output.clip = masked.clip;
    output.uv_alpha = masked.uv_alpha;
    output.tile = tile;
    return output;
}

@vertex
fn vs_moving_masked(@location(0) position: vec3<f32>, @location(7) uv: vec2<f32>,
        @location(2) pair: u32) -> MovingMaskedOutput {
    let slot = pair & MOVING_SLOT;
    let tile = pair >> 24u;
    return moving_masked_output(masked_output(route(slot_world(slot, position), tile), uv,
        slot_colors[slot].color), tile);
}

@group(2) @binding(11) var<storage, read> impostor_records: array<ImpostorRecord>;
@group(2) @binding(12) var impostor_albedo: texture_2d_array<f32>;
@group(2) @binding(16) var impostor_sampler: sampler;

struct CardOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) frame: vec4<u32>,
}

const CARD = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

@vertex
fn vs_impostor_card(@builtin(vertex_index) vertex: u32, @location(2) slot: u32,
        @location(3) index: u32) -> CardOutput {
    let pose = slot_poses[slot];
    let record = impostor_records[impostor_index(index)];
    let toward = -vec3<f32>(view.clip[0].z, view.clip[1].z, view.clip[2].z);
    let x = vec3<f32>(pose.r0.x, pose.r1.x, pose.r2.x);
    let y = vec3<f32>(pose.r0.y, pose.r1.y, pose.r2.y);
    let z = vec3<f32>(pose.r0.z, pose.r1.z, pose.r2.z);
    let yz = cross(y, z);
    let orientation = sign(dot(x, yz));
    let sun = normalize(vec3<f32>(dot(yz, toward), dot(cross(z, x), toward),
        dot(cross(x, y), toward)) * orientation);
    let hemisphere = (record.flags & IMPOSTOR_HEMISPHERE) != 0u;
    let frames = impostor_frames(record.flags);
    let grid = (impostor_grid(sun, hemisphere) * 0.5 + 0.5) * f32(frames - 1u);
    let cell = vec2<u32>(min(round(grid), vec2<f32>(f32(frames - 1u))));
    let basis = impostor_basis(impostor_frame_direction(cell.y * frames + cell.x, frames,
        hemisphere));
    let corner = CARD[vertex];
    let local = vec4<f32>(record.sphere.xyz
        + (basis.right * corner.x + basis.up * corner.y) * record.sphere.w, 1.0);
    var output: CardOutput;
    output.clip = project(vec3<f32>(dot(pose.r0, local), dot(pose.r1, local),
        dot(pose.r2, local)));
    output.uv = (vec2<f32>(cell) + 0.5 + vec2<f32>(corner.x, -corner.y) * 0.5) / f32(frames);
    output.frame = vec4<u32>(record.layer, cell, frames);
    return output;
}

@fragment
fn fs_impostor_card(input: CardOutput) {
    let size = f32(textureDimensions(impostor_albedo).x);
    let dx = dpdx(input.uv) * size;
    let dy = dpdy(input.uv) * size;
    let coarsest = f32(textureNumLevels(impostor_albedo) - 1u);
    let squared = max(max(dot(dx, dx), dot(dy, dy)), 1.0e-12);
    let level = clamp(0.5 * log2(squared), 0.0, coarsest); // filter: the atlas's level whose texels match the map's
    let span = 1.0 / f32(input.frame.w);
    let cell = vec2<f32>(input.frame.yz) * span;
    let inset = 0.5 * exp2(ceil(level)) / size;
    let uv = clamp(input.uv, cell + inset, cell + span - inset);
    let coverage = textureSampleLevel(impostor_albedo, impostor_sampler, uv, input.frame.x,
        level).a;
    if coverage * (1.0 + 0.25 * level) < 0.5 {
        discard;
    }
}

fn masked_cut(uv_alpha: vec3<f32>) -> bool {
    let point = vec3<f32>(uv_alpha.xy, 1.0);
    let uv = vec2<f32>(dot(material_maps.transforms[0].xyz, point),
        dot(material_maps.transforms[1].xyz, point));
    let texel = textureSample(base_color_texture, base_color_sampler, uv);
    let size = vec2<f32>(textureDimensions(base_color_texture));
    let dx = dpdx(uv) * size;
    let dy = dpdy(uv) * size;
    let coverage = 1.0 + 0.25 * max(0.5 * log2(max(dot(dx, dx), dot(dy, dy))), 0.0);
    return texel.a * uv_alpha.z * coverage < material_maps.params.z;
}

@fragment
fn fs_masked(input: MaskedOutput) {
    if masked_cut(input.uv_alpha) {
        discard;
    }
}

@fragment
fn fs_moving_masked(input: MovingMaskedOutput) {
    let cut = masked_cut(input.uv_alpha);
    if cut || outside_slot(input.clip.xy, input.tile) {
        discard;
    }
}
