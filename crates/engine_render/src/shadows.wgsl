// Sun shadows: caster selection and depth-only drawing into the clipmap's
// tiles (`shadows/mod.rs`).
//
// `select_casters` visits every candidate once. It transforms the candidate's
// mesh bounds into light space and appends its slot to the instance IDs of
// each map it casts into: each region of tiles redrawn this frame, which
// takes still casters, and the moving map, which takes moving casters once
// for each tile moving casters draw over this frame that they cross, the
// tile in the ID's top eight bits beside the slot. GPU-presented frames
// track how many frames each slot's pose stayed the same: still slots are
// kept in the tiles they were drawn into, and leave the moving tiles a
// little later, once the regions surely redrew theirs. A kept slot that
// moves marks the drawn tiles it was kept in dirty, and one coming to rest
// the drawn tiles under it, in every level; moving casters mark the drawn
// tiles they cross, which then draw them over the still casters; all read
// back a few frames later. CPU-prepared frames track no stillness: they
// keep the casters with virtual geometry, whose clusters the regions'
// views select and alone draw, and every other caster moves. Still casters
// (on CPU-prepared frames every caster) also go to the sky map redrawn this
// frame, looking down from one of sky occlusion's directions instead of
// from the sun.

struct Cull {
    // World to light space: the sun's right, up and away-from-sun axes.
    light: array<vec4<f32>, 3>,
    // Each region of tiles redrawn this frame: its light-space rectangle
    // (minimum x and y, maximum x and y), and (x) the farthest receiver
    // depth its casters darken.
    regions: array<vec4<f32>, REGIONS>,
    region_depths: array<vec4<f32>, REGIONS>,
    // Each level's window: light-space x and y of its first tile's corner,
    // and its tiles' width.
    windows: array<vec4<f32>, SHADOW_LEVELS>,
    candidates: u32,
    groups: u32,
    // Instance IDs of each region's map and of the sky map; the moving map
    // holds MOVING_PAIRS times as many.
    capacity: u32,
    // STILLNESS | SKY_REFRESH.
    mode: u32,
    // Regions redrawn this frame, levels in use, skinned slots of
    // GPU-presented frames, and tiles along a side of a level's window.
    counts: vec4<u32>,
    // The counters' words: those of a mask of every level's tiles, a bit a
    // cell (cell c of level l at bit l * side * side + c), and where the
    // dirty tiles' mask, the moving tiles' and the draw groups' starts lie.
    words: vec4<u32>,
    // The slot of the first moving tile, past the still casters' slots.
    moving: vec4<u32>,
    // The sky map redrawn this frame (SKY_REFRESH): world to its space (its
    // right, up and away-from-the-sky axes), its rectangle there, and (x)
    // the depth beyond which casters hide no sky from its receivers.
    sky: array<vec4<f32>, 3>,
    sky_rect: vec4<f32>,
    sky_depths: vec4<f32>,
}

// Regions a frame redraws at most, and levels the uniforms hold.
const REGIONS: u32 = 7u;
const SHADOW_LEVELS: u32 = 20u;

struct Candidate {
    slot: u32,
    node: u32,
    batch: u32,
    position: u32,
}

// A draw group (the presentation record, whose padding holds the group;
// `bounds_min.w` is 1 for meshes with joints, `bounds_max.w` 2 for those
// virtual geometry draws).
struct Batch {
    start: u32,
    count: u32,
    index_count: u32,
    group: u32,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    // The mesh's first index and base vertex in the mesh arena, and the
    // first instance of its draws (unused here), then its levels of detail
    // (unused here).
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
// Per slot: its pose key, its still-frame count with LEVELED and RESTLESS,
// and its light-space footprint while kept.
@group(0) @binding(5) var<storage, read_write> stillness: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read_write> ids: array<u32>;
@group(0) @binding(7) var<storage, read_write> args: array<DrawArgs>;
// Counters read back: this frame's casters of the regions, the moving
// map's draws and those it left out (TOTALS); then the words of the dirty
// tiles' and the moving tiles' masks, and the start of each draw group's ID
// region within a map. Sharing one binding keeps the selection within
// eight storage buffers.
struct Counters {
    totals: array<atomic<u32>, TOTALS>,
    words: array<atomic<u32>>,
}
@group(0) @binding(8) var<storage, read_write> counters: Counters;
// GPU-presented frames: skinned slots, each with its
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
// The slots of each level's tiles as lookups read them (`scene.wgsl`'s
// `shadow_tiles`): the drawn tile's in the low half and the moving tile's
// in the high half, each one more than the slot.
@group(0) @binding(12) var tile_slots: texture_2d<u32>;

const STILLNESS: u32 = 1u;
const SKY_REFRESH: u32 = 2u;
// Exact hex literal: browsers reject decimal spellings above f32::MAX.
const F32_MAX: f32 = 0x1.fffffep+127f;
// Presentation's candidate flag of a valid pose.
const VALID: u32 = 4u;
// Frames a pose must stay unchanged before its slot is kept, and until it
// leaves the moving tiles: long enough for the dirty tiles to be read back
// and redrawn.
const KEEP_AFTER: u32 = 8u;
const MOVING_UNTIL: u32 = 64u;
// A kept slot that moves again becomes restless: it waits this much longer
// before it is kept again, so props that pause and vehicles that stop stay
// over the moving tiles instead of redrawing kept tiles at every pause.
const KEEP_AFTER_RESTLESS: u32 = 180u;
const MOVING_UNTIL_RESTLESS: u32 = 236u;
// Stillness state bits: the count in the low byte; LEVELED while the slot
// is kept, whose footprint the state's last two words hold; and RESTLESS.
const LEVELED: u32 = 0x2000000u;
const RESTLESS: u32 = 0x80000000u;
// World units of footprints: light-space corners rounded outward to them,
// as signed 16-bit pairs.
const FOOTPRINT_UNIT: f32 = 4.0;
// The regions' maps come first, then the moving map and the sky map
// redrawn this frame; the moving map holds MOVING_PAIRS IDs for each of a
// group's casters, over at most MOVING_TILES tiles.
const MOVING_MAP: u32 = 7u;
const SKY_MAP: u32 = 8u;
const MAPS: u32 = 9u;
const MOVING_PAIRS: u32 = 8u;
const MOVING_TILES: u32 = 256u;
const TOTALS: u32 = 4u;

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
    return atomicLoad(&counters.words[cull.words.w + group]);
}

// Where draw group `group`'s ID region ends within a map of
// `cull.capacity` IDs.
fn group_end(group: u32) -> u32 {
    if group + 1u < cull.groups {
        return group_start(group + 1u);
    }
    return cull.capacity;
}

// Where map `map`'s instance IDs start: each region's map and the sky map
// hold `cull.capacity`, the moving map MOVING_PAIRS times as many.
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

// Appends caster `slot` of draw group `group` to the moving map for moving
// tile `tile`; past the group's room there, counts it instead.
fn append_moving(group: u32, slot: u32, tile: u32) {
    let draw = MOVING_MAP * cull.groups + group;
    let offset = atomicAdd(&args[draw].instance_count, 1u);
    let start = group_start(group);
    if offset >= MOVING_PAIRS * (group_end(group) - start) {
        atomicSub(&args[draw].instance_count, 1u);
        atomicAdd(&counters.totals[2], 1u);
        return;
    }
    ids[map_start(MOVING_MAP) + MOVING_PAIRS * start + offset] = slot | (tile << 24u);
}

// Marks in the mask of the counters' words from `mask` the drawn tiles of
// every level under light-space `low` to `high`, those of each window; and
// where `moving`, appends caster `slot` of draw group `group` to the moving
// map for each of those tiles moving casters draw over this frame.
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
                let set = 1u << (bit % 32u);
                if (atomicLoad(&counters.words[word]) & set) == 0u {
                    atomicOr(&counters.words[word], set);
                }
                let tile = entry >> 16u;
                if moving && tile != 0u {
                    append_moving(group, slot, tile - 1u - cull.moving.x);
                }
            }
        }
    }
}

// Signed 16-bit halves of a footprint word.
fn footprint_word(value: vec2<i32>) -> u32 {
    return (u32(value.x) & 0xffffu) | (u32(value.y) << 16u);
}

fn footprint_corner(word: u32) -> vec2<f32> {
    return vec2<f32>(f32(i32(word << 16u) >> 16u), f32(i32(word) >> 16u)) * FOOTPRINT_UNIT;
}

// The light-space box of the world box `center` +- `half`: its least
// corner, then its greatest.
fn light_box(center: vec3<f32>, half: vec3<f32>) -> array<vec3<f32>, 2> {
    let light_center = vec3<f32>(dot(cull.light[0].xyz, center),
        dot(cull.light[1].xyz, center), dot(cull.light[2].xyz, center));
    let light_half = vec3<f32>(dot(abs(cull.light[0].xyz), half),
        dot(abs(cull.light[1].xyz), half), dot(abs(cull.light[2].xyz), half));
    return array<vec3<f32>, 2>(light_center - light_half, light_center + light_half);
}

// Skinned slots of GPU-presented frames: each moves, drawn over the moving
// tiles its palette bounds cross, and never kept.
@compute @workgroup_size(64)
fn select_skinned(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= cull.counts.z {
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
    let bounds = light_box((world_low + world_high) * 0.5, (world_high - world_low) * 0.5);
    mark_tiles(cull.words.z, bounds[0].xy, bounds[1].xy, true, group, record.slot);
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
    // Joints place meshes with joints: GPU-presented frames select them by
    // their palette bounds (`select_skinned`), others by their mesh's
    // bounds where their slot stands.
    if batch.bounds_min.w != 0.0 && classify {
        return;
    }
    // World bounds of the posed mesh bounds, then their light-space box.
    let center = (batch.bounds_min.xyz + batch.bounds_max.xyz) * 0.5;
    let half = (batch.bounds_max.xyz - batch.bounds_min.xyz) * 0.5;
    let world_center = vec3<f32>(dot(pose.r0.xyz, center) + pose.r0.w,
        dot(pose.r1.xyz, center) + pose.r1.w, dot(pose.r2.xyz, center) + pose.r2.w);
    let world_half = vec3<f32>(dot(abs(pose.r0.xyz), half), dot(abs(pose.r1.xyz), half),
        dot(abs(pose.r2.xyz), half));
    let bounds = light_box(world_center, world_half);
    let low = bounds[0];
    let high = bounds[1];
    // Non-finite bounds fail every comparison and cast nothing. Frames
    // tracking no stillness keep the casters with virtual geometry, whose
    // clusters the regions alone draw, and no other: each of those moves,
    // drawn over the moving tiles it crosses.
    let clustered = batch.bounds_max.w != 0.0;
    var moving = !clustered;
    var kept = clustered;
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
            // A kept slot that moved leaves its shadow in the tiles it was
            // kept in: redraw them, and keep it moving longer next time.
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
            // Newly still: the tiles under it must be drawn with it.
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
    // Sky occlusion's map, which keeps what stays: its box seen from the sky.
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

// Totals of this frame's casters of the regions together and of the
// moving map, read back with the masks for the frame stats: each of a
// workgroup's lanes sums some draws.
const COUNT_LANES: u32 = 256u;
var<workgroup> cast_totals: array<atomic<u32>, 2>;

@compute @workgroup_size(256)
fn count_casters(@builtin(local_invocation_index) lane: u32) {
    if lane < 2u {
        atomicStore(&cast_totals[lane], 0u);
    }
    workgroupBarrier();
    // Every lane sums every 256th draw of the regions' maps and the moving
    // map, those before the sky map, which shadows do not count.
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

// Depth-only drawing into the clipmap: a region's layer through its view,
// the atlas's slots of the tiles composed from the regions and of the
// moving tiles through their routes, and the sky map.

struct View {
    // World to this view's clip space; the compose view's first column holds
    // two over the atlas's width and height (x and y).
    clip: mat4x4<f32>,
    // The wind: its velocity (x and z), gust strength and the
    // presentation time, which placed objects sway in as `wind.wgsl`'s
    // `wind_sway` has them.
    wind: vec4<f32>,
    // The detail eye, whose distance sizes the wind's shaking on screen as
    // meshes see it, and (w) the pixels a radian spans at the view's center,
    // one over a pixel's angle. Views without one (w 0), the regions', cast
    // placed objects by their trunks' mean bend alone, in their wind without
    // gusts at time zero.
    eye: vec4<f32>,
}

// Its own binding: the selection pass above owns binding 0.
@group(0) @binding(9) var<uniform> view: View;
// The routes of this frame's moving tiles: each tile's world to the atlas's
// clip space, mapping the tile onto its slot, and its slot's texels (first
// x and y, then past the last); and two over the atlas's width and height
// (x and y).
struct Routes {
    clips: array<mat4x4<f32>, MOVING_TILES>,
    slots: array<vec4<f32>, MOVING_TILES>,
    atlas: vec4<f32>,
}
@group(0) @binding(13) var<uniform> routes: Routes;
@group(1) @binding(0) var<storage, read> slot_poses: array<Rows>;
// Joint palettes of skinned slots, as meshes.wgsl reads them.
@group(1) @binding(2) var<storage, read> slot_palette: array<Rows>;

// Casters between the sun and the near plane of `clip`, world to a clip
// space of depths from 0 on, still darken its receivers: flattened onto
// it.
fn flattened(clip: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(clip.xy, max(clip.z, 0.0), clip.w);
}

fn project(world: vec3<f32>) -> vec4<f32> {
    return flattened(view.clip * vec4<f32>(world, 1.0));
}

// `world` through moving tile `tile`'s route onto its slot of the atlas.
fn route(world: vec3<f32>, tile: u32) -> vec4<f32> {
    return flattened(routes.clips[tile] * vec4<f32>(world, 1.0));
}

// Whether the atlas texel at `position` (a fragment's) lies outside moving
// tile `tile`'s slot: what its casters draw past their tile stays out of
// the slots around.
fn outside_slot(position: vec2<f32>, tile: u32) -> bool {
    let slot = routes.slots[tile];
    return any(position < slot.xy) || any(position >= slot.zw);
}

// Slot `slot`'s world position of mesh point `position`, swayed by the
// wind where its flags carry a sway (skinned slots' palette
// offsets read as none), as the meshes' vertices sway.
fn slot_world(slot: u32, position: vec3<f32>) -> vec3<f32> {
    let pose = slot_poses[slot];
    let point = vec4<f32>(position, 1.0);
    let world = vec3<f32>(dot(pose.r0, point), dot(pose.r1, point), dot(pose.r2, point));
    let sway = slot_sway(slot_colors[slot].flags);
    if sway == 0.0 {
        return world;
    }
    let root = vec3<f32>(pose.r0.w, pose.r1.w, pose.r2.w);
    // The pixels a meter at the root spans on screen seen from the detail
    // eye, as `detail_across` measures it for meshes; none without an eye.
    let pixels = view.eye.w / max(length(root - view.eye.xyz), 1.0e-6);
    return world + wind_sway(view.wind, pixels, root, world.y - root.y, position, sway);
}

@vertex
fn vs_affine(@location(0) position: vec3<f32>, @location(2) slot: u32) -> @builtin(position) vec4<f32> {
    return project(slot_world(slot, position));
}

// A moving caster's slot: its instance ID's low 24 bits.
const MOVING_SLOT: u32 = 0xffffffu;

// A moving caster drawn over the moving tile its instance ID's top eight
// bits name.
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

// Its depth within its tile's slot alone.
@fragment
fn fs_moving(input: MovingOutput) {
    if outside_slot(input.clip.xy, input.tile) {
        discard;
    }
}

// The slots of this frame's moving tiles (the instance's) cleared to the
// far depth: two triangles over its texels.
@vertex
fn vs_clear_moving(@builtin(vertex_index) vertex: u32,
        @builtin(instance_index) tile: u32) -> @builtin(position) vec4<f32> {
    let slot = routes.slots[tile];
    let texel = mix(slot.xy, slot.zw, CARD[vertex] * 0.5 + vec2<f32>(0.5));
    return vec4<f32>(texel.x * routes.atlas.x - 1.0, 1.0 - texel.y * routes.atlas.y, 1.0, 1.0);
}

// The regions of tiles redrawn this frame, a layer each (`shadows/mod.rs`),
// which the tiles composed from them read; depth passes drawing into them
// bind a placeholder. A region's layer holds its square's tiles side by
// side from its first, rows running down, and an apron around them.
@group(1) @binding(3) var shadow_regions: texture_depth_2d_array;

struct ComposeOutput {
    @builtin(position) clip: vec4<f32>,
    // The region's layer, and what to add to an atlas texel of the slot for
    // the region's texel holding it (y and z).
    @location(0) @interpolate(flat) source: vec3<i32>,
}

// A tile composed from its region into its slot of the atlas: two
// triangles over the slot's texels (`place`: first x and y, then past the
// last), carrying its region's layer and offset (`source`).
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

// The slot's depth at its texel: the region's at the same light-space
// texel.
@fragment
fn fs_compose(input: ComposeOutput) -> @builtin(frag_depth) f32 {
    let texel = vec2<i32>(input.clip.xy) + input.source.yz;
    return textureLoad(shadow_regions, texel, input.source.x, 0);
}

// Alpha-masked casters: the base color texture's alpha times
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
    return masked_output(project(slot_world(slot, position)), uv, slot_colors[slot].color);
}

// A masked moving caster drawn over the moving tile its instance ID names.
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

// A slot's world position of a vertex with joints and weights:
// its palette blend when skinned (flags one more than the palette
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

@vertex
fn vs_moving_skinned(@location(0) position: vec3<f32>, @location(2) pair: u32,
        @location(11) joints: vec4<u32>, @location(12) weights: vec4<f32>) -> MovingOutput {
    var output: MovingOutput;
    output.tile = pair >> 24u;
    output.clip = route(skinned_world(pair & MOVING_SLOT, position, joints, weights),
        output.tile);
    return output;
}

@vertex
fn vs_moving_masked_skinned(@location(0) position: vec3<f32>, @location(7) uv: vec2<f32>,
        @location(2) pair: u32, @location(11) joints: vec4<u32>,
        @location(12) weights: vec4<f32>) -> MovingMaskedOutput {
    let slot = pair & MOVING_SLOT;
    let tile = pair >> 24u;
    return moving_masked_output(masked_output(
        route(skinned_world(slot, position, joints, weights), tile), uv,
        slot_colors[slot].color), tile);
}

// Placed objects' impostor cards: objects the camera draws
// as impostors cast a card through their sphere's center facing the
// frame nearest the sun, covered as that frame covers it. The impostors'
// group of meshes.wgsl: records, the albedo and coverage array and its
// sampler.
@group(2) @binding(11) var<storage, read> impostor_records: array<ImpostorRecord>;
@group(2) @binding(12) var impostor_albedo: texture_2d_array<f32>;
@group(2) @binding(16) var impostor_sampler: sampler;

struct CardOutput {
    @builtin(position) clip: vec4<f32>,
    // Layer coordinates of the card point in its frame.
    @location(0) uv: vec2<f32>,
    // The layer, the frame's column and row, and the frames a side.
    @location(1) @interpolate(flat) frame: vec4<u32>,
}

// Two triangles of a card, corners in its right and up.
const CARD = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

@vertex
fn vs_impostor_card(@builtin(vertex_index) vertex: u32, @location(2) slot: u32,
        @location(3) index: u32) -> CardOutput {
    let pose = slot_poses[slot];
    let record = impostor_records[impostor_index(index)];
    // The mesh-space direction toward the sun: depth grows away from it.
    // Cofactor columns map it as the inverse would, but for scale.
    let toward = -vec3<f32>(view.clip[0].z, view.clip[1].z, view.clip[2].z);
    let x = vec3<f32>(pose.r0.x, pose.r1.x, pose.r2.x);
    let y = vec3<f32>(pose.r0.y, pose.r1.y, pose.r2.y);
    let z = vec3<f32>(pose.r0.z, pose.r1.z, pose.r2.z);
    let yz = cross(y, z);
    let orientation = sign(dot(x, yz));
    let sun = normalize(vec3<f32>(dot(yz, toward), dot(cross(z, x), toward),
        dot(cross(x, y), toward)) * orientation);
    // The frame nearest it, whose image the card shows.
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

// The frame's coverage where texels match the map's, scaled up on coarser
// levels as the impostors' own tests are; samples stay within the frame.
@fragment
fn fs_impostor_card(input: CardOutput) {
    let size = f32(textureDimensions(impostor_albedo).x);
    let dx = dpdx(input.uv) * size;
    let dy = dpdy(input.uv) * size;
    let level = clamp(0.5 * log2(max(max(dot(dx, dx), dot(dy, dy)), 1.0e-12)), 0.0,
        f32(textureNumLevels(impostor_albedo) - 1u));
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

// Whether a masked caster's texel at mesh coordinates `uv_alpha.xy`, of the
// object's color alpha `uv_alpha.z`, is cut out. Material bindings are
// fragment-visible: the base color map's transform applies here. Alpha
// averaged down the mip chain keeps its coverage, as meshes.wgsl's masked
// tests scale it: a quarter more per level below the base.
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

// A masked moving caster's texel within its tile's slot alone.
@fragment
fn fs_moving_masked(input: MovingMaskedOutput) {
    let cut = masked_cut(input.uv_alpha);
    if cut || outside_slot(input.clip.xy, input.tile) {
        discard;
    }
}
