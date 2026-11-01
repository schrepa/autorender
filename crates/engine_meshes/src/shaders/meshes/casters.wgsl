import package::engine::materials::{base_color_sampler, base_color_texture, material_maps};
import package::engine::view::camera;
import package::face::caster_routes::{caster_box, caster_in_region, caster_in_sky, caster_levels,
    caster_regions, caster_span, mark_caster, outside_route, route};
import package::face::wind::{slot_sway, wind_sway};
import package::generated::meshes::{BatchData as Batch, CULL_GROUP, CandidateData as Candidate,
    CasterFrame, MOVING_MAP, MOVING_PAIRS, SKY_MAP, STILLNESS};
import package::meshes::presentation::{Rows, VALID};
import package::meshes::slots::{instance_colors, instance_poses};

struct DrawArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(1) @binding(0) var<uniform> selection: CasterFrame;
@group(1) @binding(1) var<storage, read> candidates: array<Candidate>;
@group(1) @binding(2) var<storage, read> batches: array<Batch>;
@group(1) @binding(3) var<storage, read> poses: array<Rows>;
@group(1) @binding(4) var<storage, read> flags: array<u32>;
@group(1) @binding(5) var<storage, read_write> stillness: array<vec4<u32>>;
@group(1) @binding(6) var<storage, read_write> ids: array<u32>;
@group(1) @binding(7) var<storage, read_write> args: array<DrawArgs>;
@group(1) @binding(8) var<storage, read_write> totals: array<atomic<u32>, 4>;

const KEEP_AFTER: u32 = 8u;
const MOVING_UNTIL: u32 = 64u;
const KEEP_AFTER_RESTLESS: u32 = 180u;
const MOVING_UNTIL_RESTLESS: u32 = 236u;
const LEVELED: u32 = 0x2000000u;
const RESTLESS: u32 = 0x80000000u;
const FOOTPRINT_UNIT: f32 = 4.0;

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

fn group_start(group: u32) -> u32 {
    return ids[map_start(SKY_MAP + 1u) + group];
}

fn group_end(group: u32) -> u32 {
    if group + 1u < selection.groups {
        return group_start(group + 1u);
    }
    return selection.capacity;
}

fn map_start(map: u32) -> u32 {
    if map <= MOVING_MAP {
        return map * selection.capacity;
    }
    return (map + MOVING_PAIRS - 1u) * selection.capacity;
}

fn append(map: u32, group: u32, slot: u32) {
    let offset = atomicAdd(&args[map * selection.groups + group].instance_count, 1u);
    ids[map_start(map) + group_start(group) + offset] = slot;
}

fn append_moving(group: u32, slot: u32, tile: u32) {
    let draw = MOVING_MAP * selection.groups + group;
    let offset = atomicAdd(&args[draw].instance_count, 1u);
    let start = group_start(group);
    if offset >= MOVING_PAIRS * (group_end(group) - start) {
        atomicSub(&args[draw].instance_count, 1u);
        return;
    }
    ids[map_start(MOVING_MAP) + MOVING_PAIRS * start + offset] = slot | (tile << 24u);
}

fn mark_tiles(low: vec2<f32>, high: vec2<f32>, moving: bool, group: u32, slot: u32) {
    for (var level = 0u; level < caster_levels(); level++) {
        let span = caster_span(level, low, high);
        for (var row = span.y; row <= span.w; row++) {
            for (var column = span.x; column <= span.z; column++) {
                let routed = mark_caster(level, vec2<i32>(column, row), moving);
                if moving && routed != 0u {
                    append_moving(group, slot, routed - 1u);
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

@compute @workgroup_size(CULL_GROUP)
fn select_casters(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= selection.candidates {
        return;
    }
    let candidate = candidates[id.x];
    let classify = (selection.mode & STILLNESS) != 0u;
    if classify && (flags[candidate.position] & VALID) == 0u {
        return;
    }
    let pose = poses[candidate.slot];
    let batch = batches[candidate.batch];
    if batch.group >= selection.groups {
        return;
    }
    let center = (batch.bounds_min.xyz + batch.bounds_max.xyz) * 0.5;
    let half = (batch.bounds_max.xyz - batch.bounds_min.xyz) * 0.5;
    let world_center = vec3<f32>(dot(pose.r0.xyz, center) + pose.r0.w,
        dot(pose.r1.xyz, center) + pose.r1.w, dot(pose.r2.xyz, center) + pose.r2.w);
    let world_half = vec3<f32>(dot(abs(pose.r0.xyz), half), dot(abs(pose.r1.xyz), half),
        dot(abs(pose.r2.xyz), half));
    let bounds = caster_box(world_center, world_half);
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
                mark_tiles(footprint_corner(footprint.x), footprint_corner(footprint.y), false,
                    0u, 0u);
                restless = RESTLESS;
                keep_after = KEEP_AFTER_RESTLESS;
            }
            count = 0u;
            leveled = 0u;
        }
        if count == keep_after {
            mark_tiles(low.xy, high.xy, false, 0u, 0u);
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
        for (var region = 0u; region < caster_regions(); region++) {
            if caster_in_region(region, low, high) {
                append(region, batch.group, candidate.slot);
            }
        }
    }
    if moving {
        mark_tiles(low.xy, high.xy, true, batch.group, candidate.slot);
    }
    if (kept || !classify) && caster_in_sky(world_center, world_half) {
        append(SKY_MAP, batch.group, candidate.slot);
    }
}

const COUNT_LANES: u32 = 256u;
var<workgroup> cast_totals: array<atomic<u32>, 2>;

@compute @workgroup_size(COUNT_LANES)
fn count_casters(@builtin(local_invocation_index) lane: u32) {
    if lane < 2u {
        atomicStore(&cast_totals[lane], 0u);
    }
    workgroupBarrier();
    var counted = vec2<u32>(0u);
    let draws = SKY_MAP * selection.groups;
    for (var draw = lane; draw < draws; draw += COUNT_LANES) {
        let map = draw / selection.groups;
        counted[select(0u, 1u, map == MOVING_MAP)] += atomicLoad(&args[draw].instance_count);
    }
    for (var index = 0u; index < 2u; index++) {
        if counted[index] != 0u {
            atomicAdd(&cast_totals[index], counted[index]);
        }
    }
    workgroupBarrier();
    if lane < 2u {
        atomicStore(&totals[lane], atomicLoad(&cast_totals[lane]));
    }
}

const MOVING_SLOT: u32 = 0xffffffu;

fn flattened(clip: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(clip.xy, max(clip.z, 0.0), clip.w);
}

fn project(world: vec3<f32>) -> vec4<f32> {
    return flattened(camera.view_projection * vec4<f32>(world, 1.0));
}

fn slot_world(slot: u32, position: vec3<f32>) -> vec3<f32> {
    let pose = instance_poses[slot];
    let point = vec4<f32>(position, 1.0);
    let world = vec3<f32>(dot(pose.row_0, point), dot(pose.row_1, point), dot(pose.row_2, point));
    let sway = slot_sway(instance_colors[slot].flags);
    if sway == 0.0 {
        return world;
    }
    let root = vec3<f32>(pose.row_0.w, pose.row_1.w, pose.row_2.w);
    let pixels = camera.position.w / max(length(root - camera.position.xyz), 1.0e-6);
    return world + wind_sway(camera.wind, pixels, root, world.y - root.y, position, sway);
}

@vertex
fn vs_caster(@location(0) position: vec3<f32>, @location(2) slot: u32)
        -> @builtin(position) vec4<f32> {
    return project(slot_world(slot, position));
}

struct MovingOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) tile: u32,
}

@vertex
fn vs_caster_moving(@location(0) position: vec3<f32>, @location(2) pair: u32) -> MovingOutput {
    var output: MovingOutput;
    output.tile = pair >> 24u;
    output.clip = route(slot_world(pair & MOVING_SLOT, position), output.tile);
    return output;
}

@fragment
fn fs_caster_moving(input: MovingOutput) {
    if outside_route(input.clip.xy, input.tile) {
        discard;
    }
}

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
fn vs_caster_masked(@location(0) position: vec3<f32>, @location(7) uv: vec2<f32>,
        @location(2) slot: u32) -> MaskedOutput {
    return masked_output(project(slot_world(slot, position)), uv, instance_colors[slot].color);
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
fn vs_caster_moving_masked(@location(0) position: vec3<f32>, @location(7) uv: vec2<f32>,
        @location(2) pair: u32) -> MovingMaskedOutput {
    let slot = pair & MOVING_SLOT;
    let tile = pair >> 24u;
    return moving_masked_output(masked_output(route(slot_world(slot, position), tile), uv,
        instance_colors[slot].color), tile);
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
fn fs_caster_masked(input: MaskedOutput) {
    if masked_cut(input.uv_alpha) {
        discard;
    }
}

@fragment
fn fs_caster_moving_masked(input: MovingMaskedOutput) {
    let cut = masked_cut(input.uv_alpha);
    if cut || outside_route(input.clip.xy, input.tile) {
        discard;
    }
}
