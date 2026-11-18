use bevy_ecs::system::{Res, SystemParamItem};
use engine_core::{
    CullPlan, Cursor, DeviceLimits, DrawnPoses, DrawnRecord, DrawnTable, Entry, Error, ErrorCode,
    GlobalTransforms, GpuStruct, Layers, MAX_HIERARCHY_DEPTH, NodeInfo, NodePose, NodeStreams,
    PoseRows, ScenePasses, Since, Var, ViewFacts, Views, frustum_bounded, frustum_planes,
    gpu_struct,
};
use glam::Mat4;

use crate::shaders::core_module;
use crate::uploads::{UploadRange, listed_ranges};
use crate::{
    Allocation, Bind, Buffer, ComputePipeline, Count, Encode, GpuPart, Install, Lifetime, Pass,
    Prepare,
};

pub(crate) const DRAWN_MODULE: &str = "engine::drawn";
pub(crate) const DRAWN_CULL_MODULE: &str = "engine::drawn_cull";
const CULL_GROUP: u32 = 64;
const SLOT_BITS: u32 = u32::BITS;
const FIRST_SLOTS: u64 = 64;
const VIEW_BYTES: u64 = 16;
const POSE_BYTES: u64 = 48;
const COMPOSED_BYTES: u64 = 112;
const MIN_STORAGE_BYTES: u64 = 256;
const NODE_LEVELS: usize = MAX_HIERARCHY_DEPTH + 1;
const WRITE_CALL_BYTES: usize = 4_096;

const VARS: &str = "struct DrawnPose {
    row_0: vec4<f32>,
    row_1: vec4<f32>,
    row_2: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> drawn_records: array<DrawnRecord>;
@group(0) @binding(0) var<storage, read> drawn_culled: array<u32>;
@group(0) @binding(0) var<uniform> drawn_view: vec4<u32>;
@group(0) @binding(0) var<storage, read> drawn_views: array<CullView>;
@group(0) @binding(0) var<storage, read> drawn_poses: array<DrawnPose>;

fn drawn_in_view(view: u32, slot: u32) -> bool {
    let words = drawn_culled[0];
    let word = slot / 32u;
    if word >= words {
        return false;
    }
    return (drawn_culled[1u + view * words + word] & (1u << (slot % 32u))) != 0u;
}

fn drawn_shown(slot: u32) -> bool {
    return drawn_in_view(drawn_view.x, slot);
}

fn drawn_pose_matrix(pose: DrawnPose) -> mat4x4<f32> {
    return transpose(mat4x4<f32>(pose.row_0, pose.row_1, pose.row_2,
        vec4<f32>(0.0, 0.0, 0.0, 1.0)));
}

fn drawn_model(slot: u32) -> mat4x4<f32> {
    return drawn_pose_matrix(drawn_poses[slot]);
}

fn frustum_reach(plane: vec4<f32>, low: vec3<f32>, high: vec3<f32>) -> f32 {
    let support = select(low, high, plane.xyz >= vec3<f32>(0.0));
    return dot(plane.xyz, support) + plane.w;
}
";

const CULL_IMPORTS: &str = "import package::engine::drawn::{DrawnPose, DrawnRecord,
    drawn_flags_own_projection, drawn_flags_unbounded, drawn_model, drawn_records, drawn_views,
    frustum_reach};
import package::engine::math::maximum_abs;
import package::engine::presentation::{MAX_COEFFICIENT, MAX_PRODUCT_BOUND, NO_PARENT,
    POSE_FLOATS, Pose, ROUNDING_MARGIN, drawn_local, pose_admitted, pose_trs, pose_word_blend,
    pose_word_composed, pose_word_parent_composed, pose_word_valid};
";

const CULL: &str = "
const F32_MAX: f32 = 3.4028233e38f;
const SUBPIXEL_MARGIN: f32 = 0.0625;

struct Rows {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

struct Composed {
    display: Rows,
    current: Rows,
    status: vec4<u32>,
}

struct Level {
    start: u32,
    end: u32,
    pad0: u32,
    pad1: u32,
}

struct Projected {
    low: vec2<f32>,
    high: vec3<f32>,
    valid: bool,
}

@group(2) @binding(0) var<uniform> cull_frame: DrawnFrame;
@group(2) @binding(1) var<storage, read> node_locals: array<f32>;
@group(2) @binding(2) var<storage, read> node_info: array<vec2<u32>>;
@group(2) @binding(3) var<storage, read> node_authority: array<Rows>;
@group(2) @binding(4) var<storage, read_write> node_composed: array<Composed>;
@group(2) @binding(5) var<storage, read> node_parents: array<u32>;
@group(2) @binding(6) var<uniform> node_level: Level;
@group(2) @binding(7) var<storage, read_write> cull_poses: array<DrawnPose>;
@group(2) @binding(8) var<storage, read_write> cull_bits: array<u32>;

fn rows_matrix(rows: Rows) -> mat4x4<f32> {
    return transpose(mat4x4<f32>(rows.r0, rows.r1, rows.r2, vec4<f32>(0.0, 0.0, 0.0, 1.0)));
}

fn matrix_rows(m: mat4x4<f32>) -> Rows {
    let t = transpose(m);
    return Rows(t[0], t[1], t[2]);
}

fn finite(value: vec4<f32>) -> bool {
    return all(abs(value) <= vec4<f32>(F32_MAX));
}

fn node_pose(at: u32) -> Pose {
    return Pose(
        vec3<f32>(node_locals[at], node_locals[at + 1u], node_locals[at + 2u]),
        vec4<f32>(node_locals[at + 4u], node_locals[at + 5u], node_locals[at + 6u],
            node_locals[at + 7u]),
        vec3<f32>(node_locals[at + 3u], node_locals[at + 8u], node_locals[at + 9u]),
    );
}

fn compose(node: u32) -> Composed {
    let record = node_info[node];
    let parent = record.x;
    let word = record.y;
    var base_display = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
    var base_current = base_display;
    var parent_fallback = false;
    if pose_word_parent_composed(word) {
        let above = node_composed[parent];
        base_display = rows_matrix(above.display);
        base_current = rows_matrix(above.current);
        parent_fallback = above.status.x != 0u;
    } else if parent != NO_PARENT {
        base_display = rows_matrix(node_authority[parent]);
        base_current = base_display;
    }
    let now = node_pose(node * POSE_FLOATS);
    let current = base_current * pose_trs(now);
    var result: Composed;
    result.current = matrix_rows(current);
    if !parent_fallback {
        let was = node_pose((cull_frame.capacity + node) * POSE_FLOATS);
        let display = base_display * drawn_local(was, now, cull_frame.alpha, pose_word_blend(word));
        if pose_admitted(display) {
            result.display = matrix_rows(display);
            result.status = vec4<u32>(0u);
            return result;
        }
    }
    result.display = result.current;
    result.status = vec4<u32>(1u, 0u, 0u, 0u);
    return result;
}

fn placed(node: u32) -> bool {
    return node < cull_frame.nodes && pose_word_valid(node_info[node].y);
}

@compute @workgroup_size(64)
fn drawn_compose(@builtin(global_invocation_id) id: vec3<u32>) {
    let position = node_level.start + id.x;
    if position >= node_level.end {
        return;
    }
    let node = node_parents[position];
    node_composed[node] = compose(node);
}

@compute @workgroup_size(64)
fn drawn_pose(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot = id.x;
    if slot >= cull_frame.slots {
        return;
    }
    let node = drawn_records[slot].node;
    if !placed(node) {
        return;
    }
    var rows: Rows;
    if pose_word_composed(node_info[node].y) {
        rows = compose(node).display;
    } else {
        rows = node_authority[node];
    }
    cull_poses[slot] = DrawnPose(rows.r0, rows.r1, rows.r2);
}

fn outside(view: u32, m: mat4x4<f32>, bounds_min: vec3<f32>, bounds_max: vec3<f32>) -> bool {
    let maximum = max(abs(bounds_min), abs(bounds_max));
    let absolute = mat4x4<f32>(abs(m[0]), abs(m[1]), abs(m[2]), abs(m[3]));
    if maximum_abs(maximum) > MAX_COEFFICIENT
        || max(maximum_abs(absolute[0].xyz), max(maximum_abs(absolute[1].xyz),
            max(maximum_abs(absolute[2].xyz), maximum_abs(absolute[3].xyz)))) > MAX_COEFFICIENT {
        return false;
    }
    let world_bound = absolute * vec4<f32>(maximum, 1.0);
    let clip_bound = drawn_views[view].absolute * world_bound;
    if !finite(world_bound) || !finite(clip_bound)
        || max(maximum_abs(world_bound.xyz), world_bound.w) > MAX_PRODUCT_BOUND
        || max(maximum_abs(clip_bound.xyz), abs(clip_bound.w)) > MAX_PRODUCT_BOUND {
        return false;
    }
    var plane_bounds = array<f32, 6>(
        clip_bound.x + clip_bound.w,
        clip_bound.x + clip_bound.w,
        clip_bound.y + clip_bound.w,
        clip_bound.y + clip_bound.w,
        clip_bound.z + clip_bound.w,
        clip_bound.z,
    );
    let t = transpose(m);
    for (var plane = 0u; plane < 6u; plane++) {
        let local = t * drawn_views[view].planes[plane];
        let distance = frustum_reach(local, bounds_min, bounds_max);
        if !finite(local) || !(abs(distance) <= F32_MAX) {
            return false;
        }
        if distance < -ROUNDING_MARGIN * max(plane_bounds[plane], 1.0) {
            return true;
        }
    }
    return false;
}

fn project_box(clip_from_box: mat4x4<f32>, low: vec3<f32>, high: vec3<f32>) -> Projected {
    var projected: Projected;
    projected.low = vec2<f32>(F32_MAX);
    projected.high = vec3<f32>(-F32_MAX);
    for (var corner = 0u; corner < 8u; corner++) {
        let position = vec3<f32>(select(low.x, high.x, (corner & 1u) != 0u),
            select(low.y, high.y, (corner & 2u) != 0u),
            select(low.z, high.z, (corner & 4u) != 0u));
        let clip = clip_from_box * vec4<f32>(position, 1.0);
        if !(clip.w > 1.0e-6) {
            return projected;
        }
        let ndc = clip.xyz / clip.w;
        projected.low = min(projected.low, ndc.xy);
        projected.high = max(projected.high, ndc);
    }
    projected.valid = all(projected.low >= vec2<f32>(-F32_MAX))
        && all(projected.high.xy <= vec2<f32>(F32_MAX));
    return projected;
}

fn covers_no_pixel(clip_from_box: mat4x4<f32>, low: vec3<f32>, high: vec3<f32>,
        size: vec2<f32>) -> bool {
    let projected = project_box(clip_from_box, low, high);
    if !projected.valid || !(all(projected.low >= vec2<f32>(-4.0))
            && all(projected.high.xy <= vec2<f32>(4.0))) {
        return false;
    }
    let first = vec2<f32>(projected.low.x * 0.5 + 0.5, 0.5 - projected.high.y * 0.5) * size
        - SUBPIXEL_MARGIN;
    let last = vec2<f32>(projected.high.x * 0.5 + 0.5, 0.5 - projected.low.y * 0.5) * size
        + SUBPIXEL_MARGIN;
    let centers = floor(last - 0.5) - ceil(first - 0.5);
    return centers.x < 0.0 || centers.y < 0.0;
}

fn seen(view: u32, slot: u32, record: DrawnRecord) -> bool {
    if drawn_flags_unbounded(record.flags) {
        return true;
    }
    if !placed(record.node) {
        return false;
    }
    let own = drawn_flags_own_projection(record.flags);
    let model = drawn_model(slot);
    let low = record.bounds_min.xyz;
    let high = record.bounds_max.xyz;
    if !own && drawn_views[view].frustum != 0u && outside(view, model, low, high) {
        return false;
    }
    return own || drawn_views[view].subpixel == 0u
        || !covers_no_pixel(drawn_views[view].rasterized * model, low, high,
            drawn_views[view].viewport.xy);
}

@compute @workgroup_size(64)
fn drawn_cull(@builtin(global_invocation_id) id: vec3<u32>) {
    let words = cull_bits[0];
    let word = id.x;
    let view = id.y;
    if word >= words || view >= cull_frame.views {
        return;
    }
    let total = min(arrayLength(&drawn_records), cull_frame.slots);
    let first = word * 32u;
    let count = min(32u, total - min(first, total));
    var bits = 0u;
    for (var lane = 0u; lane < count; lane += 1u) {
        let slot = first + lane;
        let record = drawn_records[slot];
        let member = (record.passes & drawn_views[view].passes) != 0u
            && (record.layers & drawn_views[view].layers) != 0u;
        if member && seen(view, slot, record) {
            bits |= 1u << lane;
        }
    }
    cull_bits[1u + view * words + word] = bits;
}
";

gpu_struct! {
    pub struct CullView {
        pub planes: [[f32; 4]; 6],
        pub absolute: [[f32; 4]; 4],
        pub rasterized: [[f32; 4]; 4],
        pub viewport: [f32; 4],
        pub passes: u32,
        pub layers: u32,
        pub frustum: u32,
        pub subpixel: u32,
    }
}

gpu_struct! {
    pub struct DrawnFrame {
        pub alpha: f32,
        pub capacity: u32,
        pub nodes: u32,
        pub slots: u32,
        pub views: u32,
        pub pad_0: u32,
        pub pad_1: u32,
        pub pad_2: u32,
    }
}

pub(crate) fn source() -> String {
    let mut source = engine_core::drawn_wgsl();
    source.push_str(CullView::WGSL);
    source.push_str(VARS);
    source
}

pub(crate) fn cull_source() -> String {
    let mut source = String::from(CULL_IMPORTS);
    source.push_str(DrawnFrame::WGSL);
    source.push_str(CULL);
    source
}

struct Nodes {
    locals: Buffer<DrawnGpu>,
    info: Buffer<DrawnGpu>,
    authority: Buffer<DrawnGpu>,
    composed: Buffer<DrawnGpu>,
    parents: Buffer<DrawnGpu>,
    levels: Vec<Buffer<DrawnGpu>>,
    capacity: usize,
    composed_capacity: usize,
    parent_capacity: usize,
    parents_sent: Vec<u32>,
    level_ranges: Vec<(u32, u32)>,
    seen: Cursor,
    whole: bool,
    ranges: Vec<UploadRange>,
    bits: Vec<u64>,
}

pub(crate) struct DrawnGpu {
    records: Buffer<Self>,
    culled: Buffer<Self>,
    views: Buffer<Self>,
    view: Buffer<Self>,
    frame: Buffer<Self>,
    poses: Buffer<Self>,
    nodes: Nodes,
    cull: Pass<Self>,
    compose: Vec<ComputePipeline<Self>>,
    pose: ComputePipeline<Self>,
    pipeline: ComputePipeline<Self>,
    seen: Cursor,
    slot_bits: Vec<u64>,
    slot_ranges: Vec<UploadRange>,
    entries: Vec<CullView>,
    views_sent: Vec<CullView>,
    words_sent: u32,
    frame_sent: DrawnFrame,
    slots: u32,
    groups: [u32; 3],
}

fn var(module: &str, name: &'static str) -> Var {
    Var::generated(core_module(module), name)
}

fn storage_bytes(bytes: u64) -> u64 {
    bytes.max(MIN_STORAGE_BYTES)
}

fn grow(needed: usize) -> usize {
    needed.max(1_024).next_power_of_two()
}

fn merged(ranges: &[UploadRange], size: usize) -> impl Iterator<Item = UploadRange> + '_ {
    let mut at = 0;
    std::iter::from_fn(move || {
        let mut range = *ranges.get(at)?;
        at += 1;
        while let Some(next) = ranges.get(at)
            && (next.start - range.end).saturating_mul(size) <= WRITE_CALL_BYTES
        {
            range.end = next.end;
            at += 1;
        }
        Some(range)
    })
}

fn write_ranges<T: bytemuck::Pod>(
    cx: &mut Prepare<'_, DrawnGpu>,
    (buffer, base): (Buffer<DrawnGpu>, u64),
    (data, per): (&[T], usize),
    ranges: &[UploadRange],
) -> (u64, u32) {
    let size = per * size_of::<T>();
    let (mut bytes, mut writes) = (0, 0);
    for range in merged(ranges, size) {
        cx.write(
            buffer,
            base + (range.start * size) as u64,
            &data[range.start * per..range.end * per],
        );
        bytes += ((range.end - range.start) * size) as u64;
        writes += 1;
    }
    (bytes, writes)
}

fn absolute(view_projection: Mat4) -> [[f32; 4]; 4] {
    Mat4::from_cols(
        view_projection.x_axis.abs(),
        view_projection.y_axis.abs(),
        view_projection.z_axis.abs(),
        view_projection.w_axis.abs(),
    )
    .to_cols_array_2d()
}

fn cull_views(
    (views, passes): (&Views, Option<&ScenePasses>),
    plan: &CullPlan,
    entries: &mut Vec<CullView>,
) {
    let listed = passes.map_or(&[][..], ScenePasses::listed);
    let main = listed
        .iter()
        .filter_map(|scene| scene.bit)
        .fold(0, |mask, bit| mask | bit.mask().word());
    let facts = views.listed();
    entries.clear();
    for view in views.drawn() {
        let placed = facts
            .get(view.local as usize)
            .map_or(view.view_projection, |facts| facts.view_projection);
        entries.push(CullView {
            planes: frustum_planes(placed).map(|plane| plane.to_array()),
            absolute: absolute(placed),
            rasterized: placed.to_cols_array_2d(),
            viewport: [view.size.x as f32, view.size.y as f32, 0.0, 0.0],
            passes: view.passes.word(),
            layers: view.layers.0,
            frustum: u32::from(plan.frustum && frustum_bounded(placed)),
            subpixel: u32::from(plan.subpixel),
        });
    }
    if entries.is_empty() {
        entries.push(CullView {
            passes: main,
            layers: Layers::ALL.0,
            ..CullView::default()
        });
    }
}

impl DrawnGpu {
    fn upload(&mut self, table: &DrawnTable, slots: u64, cx: &mut Prepare<'_, Self>) {
        let records = table.records();
        let fresh = matches!(
            cx.allocate_buffer(self.records, slots * size_of::<DrawnRecord>() as u64),
            Allocation::Fresh
        );
        let changes = table.changes_since(&mut self.seen);
        if records.is_empty() {
            return;
        }
        let count = records.len();
        self.slot_ranges.clear();
        match (fresh, changes) {
            (false, Since::Same) => return,
            (false, Since::Listed(changed))
                if changed.iter().all(|&slot| (slot as usize) < count) =>
            {
                listed_ranges(changed, count, &mut self.slot_bits, &mut self.slot_ranges);
            }
            (true, _) | (false, Since::All | Since::Listed(_)) => {
                self.slot_ranges.push(UploadRange {
                    start: 0,
                    end: count,
                });
            }
        }
        write_ranges(cx, (self.records, 0), (records, 1), &self.slot_ranges);
    }

    fn ensure_nodes(&mut self, nodes: usize, parents: usize, cx: &mut Prepare<'_, Self>) {
        let held = &mut self.nodes;
        if nodes > held.capacity {
            held.capacity = grow(nodes);
            held.composed_capacity = 0;
        }
        let capacity = held.capacity as u64;
        let sizes = [
            (
                held.locals,
                storage_bytes(2 * capacity * size_of::<NodePose>() as u64),
            ),
            (
                held.info,
                storage_bytes(capacity * size_of::<NodeInfo>() as u64),
            ),
            (
                held.authority,
                storage_bytes(capacity * size_of::<PoseRows>() as u64),
            ),
        ];
        let mut fresh = false;
        for (buffer, bytes) in sizes {
            fresh |= cx.allocate_buffer(buffer, bytes) == Allocation::Fresh;
        }
        if parents != 0 && held.composed_capacity < held.capacity {
            held.composed_capacity = held.capacity;
        }
        fresh |= cx.allocate_buffer(
            held.composed,
            storage_bytes(held.composed_capacity as u64 * COMPOSED_BYTES),
        ) == Allocation::Fresh;
        if parents > held.parent_capacity {
            held.parent_capacity = grow(parents);
        }
        let parent_bytes = storage_bytes(held.parent_capacity as u64 * 4);
        fresh |= cx.allocate_buffer(held.parents, parent_bytes) == Allocation::Fresh;
        for &level in &held.levels {
            fresh |= cx.allocate_buffer(level, 16) == Allocation::Fresh;
        }
        if fresh {
            held.whole = true;
        }
    }

    fn upload_nodes(
        &mut self,
        poses: &DrawnPoses,
        streams: &NodeStreams<'_>,
        cx: &mut Prepare<'_, Self>,
    ) -> Option<(bool, u64, u32)> {
        let count = streams.info.len();
        self.ensure_nodes(count, streams.parents.len(), cx);
        let held = &mut self.nodes;
        let changes = poses.changes_since(&mut held.seen);
        let whole = std::mem::take(&mut held.whole);
        held.ranges.clear();
        let full = match changes {
            Since::Same if !whole => return None,
            Since::Listed(listed)
                if !whole && listed.iter().all(|&node| (node as usize) < count) =>
            {
                listed_ranges(listed, count, &mut held.bits, &mut held.ranges);
                false
            }
            Since::Same | Since::Listed(_) | Since::All => {
                if count != 0 {
                    held.ranges.push(UploadRange {
                        start: 0,
                        end: count,
                    });
                }
                true
            }
        };
        let previous = (held.capacity * size_of::<NodePose>()) as u64;
        let ranges = &held.ranges;
        let sent = [
            write_ranges(cx, (held.locals, 0), (streams.current, 1), ranges),
            write_ranges(cx, (held.locals, previous), (streams.previous, 1), ranges),
            write_ranges(cx, (held.info, 0), (streams.info, 1), ranges),
            write_ranges(cx, (held.authority, 0), (streams.authority, 1), ranges),
        ];
        let (mut bytes, mut writes) = (0, 0);
        for (more, calls) in sent {
            bytes += more;
            writes += calls;
        }
        if whole {
            held.parents_sent.clear();
            held.level_ranges.clear();
        }
        if !streams.parents.is_empty() && held.parents_sent.as_slice() != streams.parents {
            cx.write(held.parents, 0, streams.parents);
            held.parents_sent.clear();
            held.parents_sent.extend_from_slice(streams.parents);
            writes += 1;
        }
        let mut start = 0;
        for (level, &end) in streams.levels.iter().enumerate() {
            let range = (start, end);
            start = end;
            if held.level_ranges.get(level) == Some(&range) {
                continue;
            }
            if level < NODE_LEVELS {
                cx.write(held.levels[level], 0, &[range.0, range.1, 0, 0]);
                writes += 1;
            }
            match held.level_ranges.get_mut(level) {
                Some(known) => *known = range,
                None => held.level_ranges.push(range),
            }
        }
        held.level_ranges.truncate(streams.levels.len());
        Some((full, bytes, writes))
    }
}

impl GpuPart for DrawnGpu {
    type Reads = (
        Res<'static, DrawnTable>,
        Res<'static, Views>,
        Option<Res<'static, ScenePasses>>,
        Res<'static, DeviceLimits>,
        Res<'static, DrawnPoses>,
        Res<'static, GlobalTransforms>,
        Res<'static, CullPlan>,
    );
    const NAME: &'static str = "drawn";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let drawn_at = |name: &'static str| var(DRAWN_MODULE, name);
        let culled_at = |name: &'static str| var(DRAWN_CULL_MODULE, name);
        let records = cx.buffer("drawn_records", Lifetime::Kept);
        let culled = cx.buffer("drawn_culled", Lifetime::Kept);
        let views = cx.buffer("drawn_views", Lifetime::Kept);
        let view = cx.view_buffer("drawn_view", Lifetime::Kept);
        let frame = cx.buffer("drawn_frame", Lifetime::Kept);
        let poses = cx.buffer("drawn_poses", Lifetime::Kept);
        let nodes = Nodes {
            locals: cx.buffer("drawn_node_locals", Lifetime::Kept),
            info: cx.buffer("drawn_node_info", Lifetime::Kept),
            authority: cx.buffer("drawn_node_authority", Lifetime::Kept),
            composed: cx.buffer("drawn_node_composed", Lifetime::Kept),
            parents: cx.buffer("drawn_node_parents", Lifetime::Kept),
            levels: (0..NODE_LEVELS)
                .map(|_| cx.buffer("drawn_node_level", Lifetime::Kept))
                .collect(),
            capacity: 0,
            composed_capacity: 0,
            parent_capacity: 0,
            parents_sent: Vec::new(),
            level_ranges: Vec::new(),
            seen: Cursor::default(),
            whole: false,
            ranges: Vec::new(),
            bits: Vec::new(),
        };
        cx.frame(drawn_at("drawn_records"), Bind::Buffer(records))?;
        cx.frame(drawn_at("drawn_culled"), Bind::Buffer(culled))?;
        cx.frame(drawn_at("drawn_view"), Bind::Buffer(view))?;
        cx.frame(drawn_at("drawn_views"), Bind::Buffer(views))?;
        cx.frame(drawn_at("drawn_poses"), Bind::Buffer(poses))?;
        let cull = cx.pass("drawn_cull").label("engine/drawn-cull").id();
        let entry = |name: &'static str| Entry::generated(core_module(DRAWN_CULL_MODULE), name);
        let node_binds = |level: Buffer<Self>| {
            [
                (culled_at("cull_frame"), Bind::Buffer(frame)),
                (culled_at("node_locals"), Bind::Buffer(nodes.locals)),
                (culled_at("node_info"), Bind::Buffer(nodes.info)),
                (culled_at("node_authority"), Bind::Buffer(nodes.authority)),
                (culled_at("node_composed"), Bind::Buffer(nodes.composed)),
                (culled_at("node_parents"), Bind::Buffer(nodes.parents)),
                (culled_at("node_level"), Bind::Buffer(level)),
            ]
        };
        let mut compose = Vec::with_capacity(NODE_LEVELS);
        for &level in &nodes.levels {
            compose.push(cx.compute_pipeline(cull, entry("drawn_compose"), &node_binds(level))?);
        }
        let pose = cx.compute_pipeline(
            cull,
            entry("drawn_pose"),
            &[
                (culled_at("cull_frame"), Bind::Buffer(frame)),
                (culled_at("node_locals"), Bind::Buffer(nodes.locals)),
                (culled_at("node_info"), Bind::Buffer(nodes.info)),
                (culled_at("node_authority"), Bind::Buffer(nodes.authority)),
                (culled_at("node_composed"), Bind::Buffer(nodes.composed)),
                (culled_at("cull_poses"), Bind::Buffer(poses)),
            ],
        )?;
        let pipeline = cx.compute_pipeline(
            cull,
            entry("drawn_cull"),
            &[
                (culled_at("cull_frame"), Bind::Buffer(frame)),
                (culled_at("node_info"), Bind::Buffer(nodes.info)),
                (culled_at("cull_bits"), Bind::Buffer(culled)),
            ],
        )?;
        Ok(Self {
            records,
            culled,
            views,
            view,
            frame,
            poses,
            nodes,
            cull,
            compose,
            pose,
            pipeline,
            seen: Cursor::default(),
            slot_bits: Vec::new(),
            slot_ranges: Vec::new(),
            entries: Vec::new(),
            views_sent: Vec::new(),
            words_sent: 0,
            frame_sent: DrawnFrame::default(),
            slots: 0,
            groups: [0; 3],
        })
    }

    fn prepare(
        &mut self,
        (table, views, passes, limits, poses, hierarchy, plan): SystemParamItem<
            '_,
            '_,
            Self::Reads,
        >,
        cx: &mut Prepare<'_, Self>,
    ) {
        let count = table.records().len();
        let slots = (count as u64).next_power_of_two().max(FIRST_SLOTS);
        self.upload(&table, slots, cx);
        let streams = poses.streams(&hierarchy);
        self.upload_nodes(&poses, &streams, cx);
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.poses, slots * POSE_BYTES);
        cull_views((&views, passes.as_deref()), &plan, &mut self.entries);
        let listed = self.entries.len();
        let words = (count as u32).div_ceil(SLOT_BITS).max(1);
        let bytes = (1 + u64::from(words) * listed as u64) * 4;
        if limits.storage_room != 0 && bytes > limits.storage_room {
            cx.refuse(
                Error::new(ErrorCode::BindingLimit, "$.drawn.views")
                    .at_most(limits.storage_room)
                    .got(bytes),
            );
            return;
        }
        let fresh = cx.allocate_buffer(self.views, (listed * size_of::<CullView>()) as u64)
            == Allocation::Fresh;
        if fresh
            || bytemuck::cast_slice::<CullView, u8>(&self.entries)
                != bytemuck::cast_slice::<CullView, u8>(&self.views_sent)
        {
            cx.write(self.views, 0, &self.entries);
            std::mem::swap(&mut self.entries, &mut self.views_sent);
        }
        let fresh = cx.allocate_buffer(self.culled, bytes.next_power_of_two()) == Allocation::Fresh;
        if fresh || self.words_sent != words {
            cx.write(self.culled, 0, &[words]);
            self.words_sent = words;
        }
        self.slots = count as u32;
        let fresh =
            cx.allocate_buffer(self.frame, size_of::<DrawnFrame>() as u64) == Allocation::Fresh;
        let frame = DrawnFrame {
            alpha: streams.alpha,
            capacity: self.nodes.capacity as u32,
            nodes: streams.info.len() as u32,
            slots: self.slots,
            views: listed as u32,
            pad_0: 0,
            pad_1: 0,
            pad_2: 0,
        };
        if fresh || bytemuck::bytes_of(&frame) != bytemuck::bytes_of(&self.frame_sent) {
            cx.write(self.frame, 0, &[frame]);
            self.frame_sent = frame;
        }
        self.groups = [words.div_ceil(CULL_GROUP), listed as u32, 1];
        if self.slots != 0 {
            cx.run(self.cull);
        }
    }

    fn view(&mut self, _: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        let index = cx.view_index();
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(self.view, VIEW_BYTES);
        cx.write(self.view, 0, &[index, 0, 0, 0]);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass != self.cull {
            return 0;
        }
        for (level, &(start, end)) in self.nodes.level_ranges.iter().enumerate() {
            if end > start
                && let Some(pipeline) = self.compose.get(level)
            {
                cx.dispatch(
                    pipeline,
                    Count::Groups([(end - start).div_ceil(CULL_GROUP), 1, 1]),
                );
            }
        }
        cx.dispatch(
            &self.pose,
            Count::Groups([self.slots.div_ceil(CULL_GROUP), 1, 1]),
        );
        cx.dispatch(&self.pipeline, Count::Groups(self.groups))
    }
}
