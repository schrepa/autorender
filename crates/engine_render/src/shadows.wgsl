// Sun shadows (decision 020): caster selection and depth-only drawing.
//
// `select_casters` visits every candidate once. It transforms the candidate's
// mesh bounds into light space and appends its slot to the instance IDs of
// each shadow map it can darken: the near cascades that follow the camera
// (decision 028), the far map redrawn every frame, and the tiles of the
// kept far map redrawn this frame. GPU-presented frames also track how many frames each slot's
// pose stayed the same: still slots move into the kept map, which is drawn
// only when a region of it is dirty, and leave the per-frame map a little
// later, once the kept map surely holds them. Casters with virtual geometry
// (decision 044) go to the maps drawn every frame, whose views select their
// clusters, and never to the kept map.

struct Cull {
    // World to light space: the sun's right, up and away-from-sun axes.
    light: array<vec4<f32>, 3>,
    // Light-space rectangles (minimum x and y, maximum x and y) of each near
    // cascade, the far maps and the region of the kept map redrawn this
    // frame.
    near: array<vec4<f32>, 4>,
    far: vec4<f32>,
    refresh: vec4<f32>,
    // Where visible receivers of the far maps can be: the moving map
    // selects its casters there.
    moving: vec4<f32>,
    // Light-space depth beyond which casters cannot darken a receiver of
    // each near cascade; then (y and z) of the far maps and the refreshed
    // region, and (w) of the moving map's selection.
    near_depths: vec4<f32>,
    depths: vec4<f32>,
    candidates: u32,
    groups: u32,
    // Instance IDs per map.
    capacity: u32,
    // STILLNESS | REFRESH.
    mode: u32,
    // Kept-map tiles (TILES a side): light-space x and y of the first
    // tile's corner, and the tile width and height.
    tiles: vec4<f32>,
    // Near cascades drawn this frame, and skinned slots of GPU-presented
    // frames.
    cascades: u32,
    skinned: u32,
    pad0: u32,
    pad1: u32,
}

struct Candidate {
    slot: u32,
    node: u32,
    batch: u32,
    position: u32,
}

// A draw group (the presentation record, whose padding holds the group;
// `bounds_min.w` is 1 for meshes with joints, decision 036).
struct Batch {
    start: u32,
    count: u32,
    index_count: u32,
    group: u32,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    // The mesh's first index and base vertex in the mesh arena, and the
    // first instance of its draws (unused here), then its levels of detail
    // (decision 047; unused here).
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
    first_plane: u32,
    lod_ranges: vec4<u32>,
    lod_errors: vec4<f32>,
}

struct Rows {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

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
// GPU-presented frames: presentation's per-candidate flags (VALID marks a
// candidate whose pose this frame's culling wrote).
@group(0) @binding(4) var<storage, read> flags: array<u32>;
// Per slot: its pose key, and its still-frame count with the kept-map
// tiles it was last drawn into.
@group(0) @binding(5) var<storage, read_write> stillness: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> ids: array<u32>;
@group(0) @binding(7) var<storage, read_write> args: array<DrawArgs>;
// Counters read back ([0..8]: kept-map tiles to redraw, two rows of 16
// each; [8..11]: this frame's casters of the near cascades, moving and kept
// maps), then the start of each draw group's ID region within a map.
// Sharing one binding keeps the selection within eight storage buffers.
struct Counters {
    dirty: array<atomic<u32>, 12>,
    starts: array<u32>,
}
@group(0) @binding(8) var<storage, read_write> counters: Counters;
// GPU-presented frames: skinned slots (decision 036), each with its
// candidate position, draw group and joint count, and their world bounds
// from their palettes (minimum, then maximum).
struct Skinned {
    slot: u32,
    position: u32,
    batch: u32,
    joints: u32,
}
@group(0) @binding(10) var<storage, read> skinned: array<Skinned>;
@group(0) @binding(11) var<storage, read> skin_bounds: array<vec4<f32>>;

const STILLNESS: u32 = 1u;
// Exact hex literal: browsers reject decimal spellings above f32::MAX.
const F32_MAX: f32 = 0x1.fffffep+127f;
const REFRESH: u32 = 2u;
// Presentation's candidate flag of a valid pose.
const VALID: u32 = 4u;
// Frames a pose must stay unchanged before its slot is kept, and until it
// leaves the per-frame map: long enough for the dirty tiles to be read
// back and redrawn.
const KEEP_AFTER: u32 = 8u;
const MOVING_UNTIL: u32 = 64u;
// A kept slot that moves again becomes restless: it waits this much longer
// before it is kept again, so props that pause and vehicles that stop stay
// in the per-frame map instead of redrawing kept tiles at every pause.
const KEEP_AFTER_RESTLESS: u32 = 180u;
const MOVING_UNTIL_RESTLESS: u32 = 236u;
// Stillness state bits: the count in the low byte; the kept-map tiles the
// slot was last drawn into (first and last column, first and last row, four
// bits each) and DRAWN when it was; and RESTLESS.
const FOOTPRINT: u32 = 0x1ffff00u;
const DRAWN: u32 = 0x1000000u;
const RESTLESS: u32 = 0x80000000u;
// Tiles across each side of the kept map, and the counter words marking
// them dirty, before the caster totals.
const TILES: u32 = 16u;
const TILE_WORDS: u32 = 8u;
// Near cascades are maps 0 to 3; the far maps follow.
const MOVING_MAP: u32 = 4u;
const KEPT_MAP: u32 = 5u;
const MAPS: u32 = 6u;

fn mix_key(key: u32, value: u32) -> u32 {
    var x = (key ^ value) * 0x9e3779b1u;
    return x ^ (x >> 15u);
}

// A 32-bit fingerprint of a pose and its draw group.
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
    return counters.starts[group];
}

fn append(map: u32, group: u32, slot: u32) {
    let offset = atomicAdd(&args[map * cull.groups + group].instance_count, 1u);
    ids[map * cull.capacity + group_start(group) + offset] = slot;
}

// Skinned slots of GPU-presented frames (decision 036): each casts into the
// near cascades and the moving map its palette bounds overlap, like any
// moving caster, and never into the kept map.
@compute @workgroup_size(64)
fn select_skinned(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= cull.skinned {
        return;
    }
    let record = skinned[id.x];
    if (flags[record.position] & VALID) == 0u {
        return;
    }
    let group = batches[record.batch].group;
    if group >= cull.groups {
        return;
    }
    let world_low = skin_bounds[2u * id.x].xyz;
    let world_high = skin_bounds[2u * id.x + 1u].xyz;
    let world_center = (world_low + world_high) * 0.5;
    let world_half = (world_high - world_low) * 0.5;
    let light_center = vec3<f32>(dot(cull.light[0].xyz, world_center),
        dot(cull.light[1].xyz, world_center), dot(cull.light[2].xyz, world_center));
    let light_half = vec3<f32>(dot(abs(cull.light[0].xyz), world_half),
        dot(abs(cull.light[1].xyz), world_half), dot(abs(cull.light[2].xyz), world_half));
    let low = light_center - light_half;
    let high = light_center + light_half;
    // Bounds that are not finite cast into every map drawn each frame.
    let bounded = all(abs(low) <= vec3<f32>(F32_MAX)) && all(abs(high) <= vec3<f32>(F32_MAX));
    for (var cascade = 0u; cascade < cull.cascades; cascade++) {
        if !bounded || overlaps(cull.near[cascade], low, high, cull.near_depths[cascade]) {
            append(cascade, group, record.slot);
        }
    }
    if !bounded || overlaps(cull.moving, low, high, cull.depths.w) {
        append(MOVING_MAP, group, record.slot);
    }
}

// The kept-map tiles under light-space `low` to `high`, packed as the
// stillness state keeps them, with DRAWN.
fn footprint(low: vec2<f32>, high: vec2<f32>) -> u32 {
    let last = vec2<f32>(f32(TILES - 1u));
    let first_tile = vec2<u32>(clamp(floor((low - cull.tiles.xy) / cull.tiles.zw),
        vec2<f32>(0.0), last));
    let last_tile = vec2<u32>(clamp(floor((high - cull.tiles.xy) / cull.tiles.zw),
        vec2<f32>(0.0), last));
    return DRAWN | (first_tile.x << 8u) | (last_tile.x << 12u) | (first_tile.y << 16u)
        | (last_tile.y << 20u);
}

// Marks the kept-map tiles of `packed` (a footprint) dirty.
fn mark_dirty(packed: u32) {
    let columns = (0xffffu >> (15u - ((packed >> 12u) & 15u)))
        & (0xffffu << ((packed >> 8u) & 15u));
    for (var row = (packed >> 16u) & 15u; row <= ((packed >> 20u) & 15u); row++) {
        atomicOr(&counters.dirty[row / 2u], columns << (row % 2u * 16u));
    }
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
    // Batches without a group (transparent ones) cast no shadow.
    if batch.group >= cull.groups {
        return;
    }
    // Batches drawing virtual geometry (decision 044, flag 2) cast the
    // clusters each map's view selects: never kept, as the camera's detail
    // changes with it.
    let clustered = batch.bounds_max.w == 2.0;
    // Joints place meshes with joints (decision 036), wherever their slot
    // is: GPU-presented frames select them by their palette bounds
    // (`select_skinned`); others cast them into every map drawn each frame.
    // Neither keeps them.
    if batch.bounds_min.w != 0.0 {
        if classify {
            return;
        }
        for (var cascade = 0u; cascade < cull.cascades; cascade++) {
            append(cascade, batch.group, candidate.slot);
        }
        append(MOVING_MAP, batch.group, candidate.slot);
        return;
    }
    // World bounds of the posed mesh bounds, then their light-space box.
    let center = (batch.bounds_min.xyz + batch.bounds_max.xyz) * 0.5;
    let half = (batch.bounds_max.xyz - batch.bounds_min.xyz) * 0.5;
    let world_center = vec3<f32>(dot(pose.r0.xyz, center) + pose.r0.w,
        dot(pose.r1.xyz, center) + pose.r1.w, dot(pose.r2.xyz, center) + pose.r2.w);
    let world_half = vec3<f32>(dot(abs(pose.r0.xyz), half), dot(abs(pose.r1.xyz), half),
        dot(abs(pose.r2.xyz), half));
    let light_center = vec3<f32>(dot(cull.light[0].xyz, world_center),
        dot(cull.light[1].xyz, world_center), dot(cull.light[2].xyz, world_center));
    let light_half = vec3<f32>(dot(abs(cull.light[0].xyz), world_half),
        dot(abs(cull.light[1].xyz), world_half), dot(abs(cull.light[2].xyz), world_half));
    let low = light_center - light_half;
    let high = light_center + light_half;
    // Non-finite bounds fail every comparison and cast nothing.
    var moving = true;
    var kept = false;
    if classify && !clustered {
        let key = pose_key(pose, batch.group);
        let state = stillness[candidate.slot];
        var count = state.y & 0xffu;
        var drawn = state.y & FOOTPRINT;
        var restless = state.y & RESTLESS;
        var keep_after = select(KEEP_AFTER, KEEP_AFTER_RESTLESS, restless != 0u);
        if state.x == key {
            count = min(count + 1u, 255u);
        } else {
            // A kept slot that moved leaves its shadow in the tiles it was
            // drawn into: redraw them, and keep it moving longer next time.
            if count >= keep_after && drawn != 0u {
                mark_dirty(drawn);
                restless = RESTLESS;
                keep_after = KEEP_AFTER_RESTLESS;
            }
            count = 0u;
            drawn = 0u;
        }
        if count == keep_after && overlaps(cull.far, low, high, cull.depths.y) {
            // Newly still: its tiles must be drawn with it.
            mark_dirty(footprint(low.xy, high.xy));
        }
        moving = count < select(MOVING_UNTIL, MOVING_UNTIL_RESTLESS, restless != 0u);
        kept = count >= keep_after;
        if kept && (cull.mode & REFRESH) != 0u && overlaps(cull.refresh, low, high, cull.depths.z) {
            append(KEPT_MAP, batch.group, candidate.slot);
            drawn = footprint(low.xy, high.xy);
        }
        let packed = count | drawn | restless;
        if state.x != key || state.y != packed {
            stillness[candidate.slot] = vec2<u32>(key, packed);
        }
    }
    for (var cascade = 0u; cascade < cull.cascades; cascade++) {
        if overlaps(cull.near[cascade], low, high, cull.near_depths[cascade]) {
            append(cascade, batch.group, candidate.slot);
        }
    }
    if moving {
        if overlaps(cull.moving, low, high, cull.depths.w) {
            append(MOVING_MAP, batch.group, candidate.slot);
        }
    }
}

// Totals of this frame's casters of the near cascades together, the moving
// map and the kept map, read back with the dirty tiles for profiling
// counters.
@compute @workgroup_size(1)
fn count_casters() {
    var totals = vec3<u32>(0u);
    for (var map = 0u; map < MAPS; map++) {
        var total = 0u;
        for (var group = 0u; group < cull.groups; group++) {
            total += atomicLoad(&args[map * cull.groups + group].instance_count);
        }
        totals[select(0u, map + 1u - MOVING_MAP, map >= MOVING_MAP)] += total;
    }
    for (var index = 0u; index < 3u; index++) {
        atomicStore(&counters.dirty[TILE_WORDS + index], totals[index]);
    }
}

// Depth-only drawing into one shadow map.

struct View {
    // World to this map's clip space.
    clip: mat4x4<f32>,
}

// Its own binding: the selection pass above owns binding 0.
@group(0) @binding(9) var<uniform> view: View;
@group(1) @binding(0) var<storage, read> slot_poses: array<Rows>;
// Joint palettes of skinned slots (decision 036), as meshes.wgsl reads them.
@group(1) @binding(2) var<storage, read> slot_palette: array<Rows>;

fn project(world: vec3<f32>) -> vec4<f32> {
    var clip = view.clip * vec4<f32>(world, 1.0);
    // Casters between the sun and the map's near plane still darken its
    // receivers: flatten them onto it.
    clip.z = max(clip.z, 0.0);
    return clip;
}

@vertex
fn vs_affine(@location(0) position: vec3<f32>, @location(2) slot: u32) -> @builtin(position) vec4<f32> {
    let pose = slot_poses[slot];
    let point = vec4<f32>(position, 1.0);
    return project(vec3<f32>(dot(pose.r0, point), dot(pose.r1, point), dot(pose.r2, point)));
}

// A triangle over the whole map at the far depth: cleared regions of the
// kept map, bounded by the scissor.
@vertex
fn vs_clear(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 1.0, 1.0);
}

// Alpha-masked casters (decision 031): the base color texture's alpha times
// the object's color alpha, tested against the material's cutoff.
struct SlotColor {
    color: vec2<u32>,
    surface: u32,
    flags: u32,
}
@group(1) @binding(1) var<storage, read> slot_colors: array<SlotColor>;
// The material group of meshes.wgsl: base color texture and sampler, and
// the maps' transforms and parameters.
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
    // Mesh texture coordinates, and the object's color alpha.
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
    let pose = slot_poses[slot];
    let point = vec4<f32>(position, 1.0);
    let clip = project(vec3<f32>(dot(pose.r0, point), dot(pose.r1, point), dot(pose.r2, point)));
    return masked_output(clip, uv, slot_colors[slot].color);
}

// A slot's world position of a vertex with joints and weights (decision
// 036): its palette blend when skinned (flags one more than the palette
// offset), else its pose.
fn skinned_world(slot: u32, position: vec3<f32>, joints: vec4<u32>,
        weights: vec4<f32>) -> vec3<f32> {
    let point = vec4<f32>(position, 1.0);
    let flags = slot_colors[slot].flags;
    if flags == 0u {
        let pose = slot_poses[slot];
        return vec3<f32>(dot(pose.r0, point), dot(pose.r1, point), dot(pose.r2, point));
    }
    let base = flags - 1u;
    let w = weights / max(dot(weights, vec4<f32>(1.0)), 1.0e-6);
    let a = slot_palette[base + joints.x];
    let b = slot_palette[base + joints.y];
    let c = slot_palette[base + joints.z];
    let d = slot_palette[base + joints.w];
    let r0 = a.r0 * w.x + b.r0 * w.y + c.r0 * w.z + d.r0 * w.w;
    let r1 = a.r1 * w.x + b.r1 * w.y + c.r1 * w.z + d.r1 * w.w;
    let r2 = a.r2 * w.x + b.r2 * w.y + c.r2 * w.z + d.r2 * w.w;
    return vec3<f32>(dot(r0, point), dot(r1, point), dot(r2, point));
}

@vertex
fn vs_affine_skinned(@location(0) position: vec3<f32>, @location(2) slot: u32,
        @location(11) joints: vec4<u32>, @location(12) weights: vec4<f32>)
        -> @builtin(position) vec4<f32> {
    return project(skinned_world(slot, position, joints, weights));
}

@vertex
fn vs_affine_masked_skinned(@location(0) position: vec3<f32>, @location(7) uv: vec2<f32>,
        @location(2) slot: u32, @location(11) joints: vec4<u32>,
        @location(12) weights: vec4<f32>) -> MaskedOutput {
    return masked_output(project(skinned_world(slot, position, joints, weights)), uv,
        slot_colors[slot].color);
}

// Material bindings are fragment-visible: the base color map's transform
// applies here.
@fragment
fn fs_masked(input: MaskedOutput) {
    let point = vec3<f32>(input.uv_alpha.xy, 1.0);
    let uv = vec2<f32>(dot(material_maps.transforms[0].xyz, point),
        dot(material_maps.transforms[1].xyz, point));
    let texel = textureSample(base_color_texture, base_color_sampler, uv);
    if texel.a * input.uv_alpha.z < material_maps.params.z {
        discard;
    }
}
