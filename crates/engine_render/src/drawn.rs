use bevy_ecs::system::{Res, SystemParamItem};
use engine_core::{
    CullPlan, Cursor, DeviceLimits, DrawnPoses, DrawnRecord, DrawnTable, Entry, Error, ErrorCode,
    GlobalTransforms, GpuBits, GpuStruct, Layers, MAX_HIERARCHY_DEPTH, NodeInfo, NodePose,
    NodeStreams, PassViews, PlanState, PoseRows, ScenePasses, Since, Var, ViewFacts, Views,
    frustum_bounded, frustum_planes, gpu_bits, gpu_struct,
};
use glam::{Mat4, Vec4};

use crate::shaders::core_module;
use crate::uploads::{UploadRange, listed_ranges};
use crate::{
    Allocation, AttachPoint, Bind, Buffer, ComputePipeline, Copied, CopyRegion, Count, Encode,
    GpuPart, Install, Lifetime, PYRAMID_GROUP, Pass, Prepare, TargetName, Texture, jitter_clip,
    mip_levels, pyramid_extents, wgpu,
};

pub(crate) const DRAWN_MODULE: &str = "engine::drawn";
pub(crate) const DRAWN_CULL_MODULE: &str = "engine::drawn_cull";
pub(crate) const DRAWN_FADES_MODULE: &str = "engine::drawn_fades";
const CULL_GROUP: u32 = 64;
const SLOT_BITS: u32 = u32::BITS;
const FIRST_SLOTS: u64 = 64;
const VIEW_BYTES: u64 = 16;
const POSE_BYTES: u64 = 48;
const COMPOSED_BYTES: u64 = 112;
const MIN_STORAGE_BYTES: u64 = 256;
const NODE_LEVELS: usize = MAX_HIERARCHY_DEPTH + 1;
const PYRAMID_LEVELS: usize = 16;
const PYRAMID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const NODE_STRIDE: usize = size_of::<PoseRows>();
const NODE_BYTES: usize =
    2 * size_of::<NodePose>() + size_of::<NodeInfo>() + size_of::<PoseRows>();

const VARS: &str = "struct DrawnPose {
    row_0: vec4<f32>,
    row_1: vec4<f32>,
    row_2: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> drawn_records: array<DrawnRecord>;
@group(0) @binding(0) var<storage, read> drawn_values: array<vec4<f32>>;
@group(0) @binding(0) var<storage, read> drawn_culled: array<u32>;
@group(0) @binding(0) var<uniform> drawn_view: vec4<u32>;
@group(0) @binding(0) var<storage, read> drawn_fades: array<u32>;
@group(0) @binding(0) var<storage, read> drawn_views: array<CullView>;
@group(0) @binding(0) var<storage, read> drawn_poses: array<DrawnPose>;
@group(0) @binding(0) var<storage, read> drawn_previous_poses: array<DrawnPose>;
@group(0) @binding(0) var<storage, read> drawn_marks: array<u32>;
@group(0) @binding(0) var<storage, read> drawn_occlusion: array<u32>;

fn drawn_value(record: DrawnRecord, index: u32) -> f32 {
    if index >= record.floats {
        return 0.0;
    }
    return drawn_values[drawn_value_vec4(record, index)][index % 4u];
}

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

fn drawn_fade(slot: u32) -> u32 {
    if slot >= arrayLength(&drawn_fades) {
        return 0u;
    }
    return drawn_fades[slot];
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

fn drawn_view_sees(view: u32, low: vec3<f32>, high: vec3<f32>) -> bool {
    if view >= arrayLength(&drawn_views) || drawn_views[view].frustum == 0u {
        return true;
    }
    for (var plane = 0u; plane < 6u; plane++) {
        if frustum_reach(drawn_views[view].planes[plane], low, high) < 0.0 {
            return false;
        }
    }
    return true;
}
";

const FADES: &str = "@group(0) @binding(0) var<storage, read_write> drawn_fade_words: array<u32>;

fn set_drawn_fade(slot: u32, fade: u32) {
    if slot < arrayLength(&drawn_fade_words) {
        drawn_fade_words[slot] = fade;
    }
}
";

const CULL_IMPORTS: &str = "import package::engine::drawn::{DRAWN_OWN_PROJECTION, DRAWN_UNBOUNDED,
    DrawnPose, DrawnRecord, drawn_in_view, drawn_mark_first, drawn_mark_with_first,
    drawn_mark_with_moving, drawn_mark_with_subpixel, drawn_marks, drawn_model,
    drawn_occlusion, drawn_records, drawn_views, frustum_reach,
    occlusion_mark_hidden, occlusion_mark_with_hidden, occlusion_mark_with_late};
import package::engine::math::maximum_abs;
import package::engine::presentation::{MAX_COEFFICIENT, MAX_PRODUCT_BOUND, NO_PARENT,
    POSE_FLOATS, Pose, ROUNDING_MARGIN, drawn_local, pose_admitted, pose_trs, pose_word_blend,
    pose_word_composed, pose_word_parent_composed, pose_word_valid};
import package::engine::pyramid::{PYRAMID_GROUP, PyramidExtent, farthest_depth, farthest_level};
";

const CULL: &str = "
const F32_MAX: f32 = 3.4028233e38f;
const SUBPIXEL_MARGIN: f32 = 0.0625;
const HIDDEN_SCALE: f32 = 0.9999847412109375;

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

struct Seen {
    shown: bool,
    mark: u32,
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
@group(2) @binding(9) var<storage, read_write> cull_marks: array<u32>;
@group(2) @binding(10) var<storage, read_write> cull_occlusion: array<u32>;
@group(2) @binding(11) var source_depth: texture_depth_2d;
@group(2) @binding(12) var source_level: texture_2d<f32>;
@group(2) @binding(13) var target_level: texture_storage_2d<r32float, write>;
@group(2) @binding(14) var<uniform> pyramid_extent: PyramidExtent;
@group(2) @binding(15) var depth_pyramid: texture_2d<f32>;

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

fn hidden(clip_from_box: mat4x4<f32>, low: vec3<f32>, high: vec3<f32>,
        viewport: vec4<f32>) -> bool {
    let projected = project_box(clip_from_box, low, high);
    if !projected.valid || !(projected.high.z >= 0.0 && projected.high.z <= 1.0) {
        return false;
    }
    let size = viewport.xy;
    let first = clamp(floor(vec2<f32>(projected.low.x * 0.5 + 0.5,
        0.5 - projected.high.y * 0.5) * size), vec2<f32>(0.0), size - 1.0);
    let last = clamp(floor(vec2<f32>(projected.high.x * 0.5 + 0.5,
        0.5 - projected.low.y * 0.5) * size), vec2<f32>(0.0), size - 1.0);
    var a = vec2<u32>(first) / 2u;
    var b = vec2<u32>(last) / 2u;
    let levels = u32(viewport.z);
    var mip = 0u;
    loop {
        if (b.x - a.x <= 1u && b.y - a.y <= 1u) || mip + 1u >= levels {
            break;
        }
        a = a / 2u;
        b = b / 2u;
        mip += 1u;
    }
    if b.x - a.x > 1u || b.y - a.y > 1u {
        return false;
    }
    let farthest = min(
        min(textureLoad(depth_pyramid, a, mip).r,
            textureLoad(depth_pyramid, vec2<u32>(b.x, a.y), mip).r),
        min(textureLoad(depth_pyramid, vec2<u32>(a.x, b.y), mip).r,
            textureLoad(depth_pyramid, b, mip).r));
    return projected.high.z < farthest * HIDDEN_SCALE;
}

fn seen(view: u32, slot: u32, record: DrawnRecord) -> Seen {
    var result = Seen(false, 0u);
    let own = (record.flags & DRAWN_OWN_PROJECTION) != 0u;
    if own && drawn_views[view].owned != 0u {
        return result;
    }
    if (record.flags & DRAWN_UNBOUNDED) != 0u {
        result.shown = true;
        return result;
    }
    let node = record.node;
    if !placed(node) {
        return result;
    }
    let model = drawn_model(slot);
    let low = record.bounds_min.xyz;
    let high = record.bounds_max.xyz;
    if !own && drawn_views[view].frustum != 0u && outside(view, model, low, high) {
        return result;
    }
    if !own && drawn_views[view].subpixel != 0u
            && covers_no_pixel(drawn_views[view].rasterized * model, low, high,
                drawn_views[view].viewport.xy) {
        result.mark = drawn_mark_with_subpixel(0u, true);
        return result;
    }
    let previous = drawn_views[view].occlusion != 0u && !own
        && occlusion_mark_hidden(drawn_occlusion[slot]);
    let moving = cull_frame.motion != 0u && pose_word_composed(node_info[node].y);
    result.shown = true;
    result.mark = drawn_mark_with_first(drawn_mark_with_moving(0u, moving), !previous);
    return result;
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
            && (record.layers & drawn_views[view].layers) != 0u
            && (drawn_views[view].takes == 0u || record.mobility == drawn_views[view].takes);
        var mark = 0u;
        if member {
            let found = seen(view, slot, record);
            if found.shown {
                bits |= 1u << lane;
            }
            mark = found.mark;
        }
        if view == 0u {
            cull_marks[slot] = mark;
        }
    }
    cull_bits[1u + view * words + word] = bits;
}

@compute @workgroup_size(PYRAMID_GROUP, PYRAMID_GROUP)
fn drawn_pyramid_first(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= pyramid_extent.size) {
        return;
    }
    let farthest = farthest_depth(source_depth, id.xy * 2u, pyramid_extent.last);
    textureStore(target_level, id.xy, vec4<f32>(farthest, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(PYRAMID_GROUP, PYRAMID_GROUP)
fn drawn_pyramid_down(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= pyramid_extent.size) {
        return;
    }
    let farthest = farthest_level(source_level, id.xy * 2u, pyramid_extent.last);
    textureStore(target_level, id.xy, vec4<f32>(farthest, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(64)
fn drawn_occlusion_test(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot = id.x;
    if slot >= cull_frame.slots {
        return;
    }
    var hidden_now = occlusion_mark_hidden(cull_occlusion[slot]);
    var late = false;
    let record = drawn_records[slot];
    if drawn_in_view(0u, slot) && (record.flags & (DRAWN_OWN_PROJECTION | DRAWN_UNBOUNDED)) == 0u {
        hidden_now = hidden(drawn_views[0].rasterized * drawn_model(slot), record.bounds_min.xyz,
            record.bounds_max.xyz, drawn_views[0].viewport);
        late = !hidden_now && !drawn_mark_first(drawn_marks[slot]);
    }
    cull_occlusion[slot] = occlusion_mark_with_late(occlusion_mark_with_hidden(0u, hidden_now),
        late);
}
";

gpu_struct! {
    pub struct CullView {
        pub planes: [[f32; 4]; 6],
        pub absolute: [[f32; 4]; 4],
        pub rasterized: [[f32; 4]; 4],
        pub view_projection: [[f32; 4]; 4],
        pub eye: [f32; 4],
        pub viewport: [f32; 4],
        pub passes: u32,
        pub layers: u32,
        pub takes: u32,
        pub owned: u32,
        pub frustum: u32,
        pub subpixel: u32,
        pub occlusion: u32,
        pub pad: u32,
    }
}

gpu_struct! {
    pub struct DrawnFrame {
        pub alpha: f32,
        pub capacity: u32,
        pub nodes: u32,
        pub slots: u32,
        pub views: u32,
        pub motion: u32,
        pub pad_0: u32,
        pub pad_1: u32,
    }
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawnMark {
    pub first: bool,
    pub subpixel: bool,
    pub moving: bool,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OcclusionMark {
    pub hidden: bool,
    pub late: bool,
}

pub(crate) fn source() -> String {
    let mut source = engine_core::drawn_wgsl();
    source.push_str(CullView::WGSL);
    source.push_str(<DrawnMark as GpuBits>::WGSL);
    source.push_str(<OcclusionMark as GpuBits>::WGSL);
    source.push_str(VARS);
    source
}

pub(crate) fn fades_source() -> String {
    String::from(FADES)
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
    level_ranges: Vec<(u32, u32)>,
    seen: Cursor,
    whole: bool,
    ranges: Vec<UploadRange>,
    bits: Vec<u64>,
}

struct Occlusion {
    pass: Pass<DrawnGpu>,
    held: Buffer<DrawnGpu>,
    pyramid: Texture<DrawnGpu>,
    extents: Vec<Buffer<DrawnGpu>>,
    first: ComputePipeline<DrawnGpu>,
    down: Vec<ComputePipeline<DrawnGpu>>,
    test: ComputePipeline<DrawnGpu>,
    written: Vec<(u32, u32)>,
    running: bool,
}

pub(crate) struct DrawnGpu {
    records: Buffer<Self>,
    values: Buffer<Self>,
    fades: Buffer<Self>,
    culled: Buffer<Self>,
    views: Buffer<Self>,
    view: Buffer<Self>,
    frame: Buffer<Self>,
    poses: Buffer<Self>,
    previous: Buffer<Self>,
    marks: Buffer<Self>,
    nodes: Nodes,
    occlusion: Occlusion,
    cull: Pass<Self>,
    keep: Pass<Self>,
    compose: Vec<ComputePipeline<Self>>,
    pose: ComputePipeline<Self>,
    pipeline: ComputePipeline<Self>,
    seen: Cursor,
    stride: usize,
    slots: u32,
    groups: [u32; 3],
    motion_generation: Option<u64>,
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

fn rasterized(facts: &ViewFacts) -> Mat4 {
    let Some(offset) = facts.jitter else {
        return facts.view_projection;
    };
    let [x, y] = jitter_clip(offset.to_array(), facts.render.x, facts.render.y);
    Mat4::from_cols(Vec4::X, Vec4::Y, Vec4::Z, Vec4::new(x, y, 0.0, 1.0)) * facts.view_projection
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
    views: &Views,
    passes: Option<&ScenePasses>,
    plan: &CullPlan,
    levels: u32,
) -> Vec<CullView> {
    let listed = passes.map_or(&[][..], ScenePasses::listed);
    let main = listed
        .iter()
        .filter(|scene| scene.pass.views != PassViews::Owned)
        .filter_map(|scene| scene.bit)
        .fold(0, |mask, bit| mask | bit.mask().word());
    let facts = views.listed();
    let mut entries: Vec<CullView> = views
        .drawn()
        .iter()
        .enumerate()
        .map(|(index, view)| {
            let owned = view.owner.is_some();
            let placed = if owned {
                view.view_projection
            } else {
                facts
                    .get(view.local as usize)
                    .map_or(view.view_projection, rasterized)
            };
            let planes = match view.planes {
                Some(planes) if owned => planes,
                Some(_) | None => frustum_planes(placed).map(Vec4::to_array),
            };
            let occluding = index == 0 && levels != 0;
            CullView {
                planes,
                absolute: absolute(placed),
                rasterized: placed.to_cols_array_2d(),
                view_projection: view.view_projection.to_cols_array_2d(),
                eye: view.eye.extend(0.0).to_array(),
                viewport: [
                    view.size.x as f32,
                    view.size.y as f32,
                    if occluding { levels as f32 } else { 0.0 },
                    0.0,
                ],
                passes: if view.draws { view.passes.word() } else { 0 },
                layers: view.layers.0,
                takes: view.takes.map_or(0, |mobility| mobility as u32 + 1),
                owned: u32::from(owned),
                frustum: u32::from(owned || (plan.frustum && frustum_bounded(placed))),
                subpixel: u32::from(owned || plan.subpixel),
                occlusion: u32::from(occluding),
                pad: 0,
            }
        })
        .collect();
    if entries.is_empty() {
        entries.push(CullView {
            passes: main,
            layers: Layers::ALL.0,
            ..CullView::default()
        });
    }
    entries
}

impl DrawnGpu {
    fn upload(&mut self, table: &DrawnTable, slots: u64, cx: &mut Prepare<'_, Self>) {
        let records = table.records();
        let stride = table.value_vec4s();
        let fresh = matches!(
            cx.allocate_buffer(self.records, slots * size_of::<DrawnRecord>() as u64),
            Allocation::Fresh
        ) | matches!(
            cx.allocate_buffer(self.values, slots * (stride.max(1) * 16) as u64),
            Allocation::Fresh
        ) | (stride != self.stride);
        self.stride = stride;
        let fades = slots * 4;
        match cx.allocate_buffer(self.fades, fades) {
            Allocation::Fresh | Allocation::Kept => cx.clear_buffer_range(self.fades, 0..fades),
        }
        let changes = table.changes_since(&mut self.seen);
        if records.is_empty() {
            return;
        }
        let range = match (fresh, changes) {
            (true, _) | (false, Since::All) => 0..records.len(),
            (false, Since::Listed(changed)) => {
                let (low, high) = changed.iter().fold((usize::MAX, 0), |(low, high), &slot| {
                    (low.min(slot as usize), high.max(slot as usize + 1))
                });
                low..high.min(records.len())
            }
            (false, Since::Same) => return,
        };
        if range.is_empty() {
            return;
        }
        cx.write(
            self.records,
            (range.start * size_of::<DrawnRecord>()) as u64,
            &records[range.clone()],
        );
        if stride > 0 {
            cx.write(
                self.values,
                (range.start * stride * 16) as u64,
                &table.values()[range.start * stride..range.end * stride],
            );
        }
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
        cx.allocate_buffer(
            held.composed,
            storage_bytes(held.composed_capacity as u64 * COMPOSED_BYTES),
        );
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
    ) -> Option<(bool, u64)> {
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
                listed_ranges(listed, count, NODE_STRIDE, &mut held.bits, &mut held.ranges);
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
        let pose = size_of::<NodePose>();
        let previous = (held.capacity * pose) as u64;
        let mut bytes = 0;
        for range in &held.ranges {
            let (start, end) = (range.start, range.end);
            bytes += ((end - start) * NODE_BYTES) as u64;
            cx.write(held.locals, (start * pose) as u64, &streams.current[start..end]);
            cx.write(
                held.locals,
                previous + (start * pose) as u64,
                &streams.previous[start..end],
            );
            cx.write(
                held.info,
                (start * size_of::<NodeInfo>()) as u64,
                &streams.info[start..end],
            );
            cx.write(
                held.authority,
                (start * size_of::<PoseRows>()) as u64,
                &streams.authority[start..end],
            );
        }
        if !streams.parents.is_empty() {
            cx.write(held.parents, 0, streams.parents);
        }
        held.level_ranges.clear();
        let mut start = 0;
        for &end in streams.levels {
            held.level_ranges.push((start, end));
            start = end;
        }
        for (level, &(start, end)) in held.level_ranges.iter().enumerate().take(NODE_LEVELS) {
            cx.write(held.levels[level], 0, &[start, end, 0, 0]);
        }
        Some((full, bytes))
    }

    fn prepare_poses(
        &mut self,
        (table, main): (&DrawnTable, Option<&ViewFacts>),
        slots: u64,
        cx: &mut Prepare<'_, Self>,
    ) -> bool {
        let kept = cx.read_target(TargetName::MOTION);
        let bytes = slots * POSE_BYTES;
        let poses = cx.allocate_buffer(self.poses, bytes);
        let previous =
            cx.allocate_buffer(self.previous, if kept { bytes } else { storage_bytes(0) });
        let fresh = poses == Allocation::Fresh || previous == Allocation::Fresh;
        let generation = match table.plan() {
            PlanState::Built { generation } => Some(generation),
            PlanState::Stale => None,
        };
        let history = main.is_some_and(|main| main.history.valid());
        let motion = kept
            && history
            && !fresh
            && cx.ran(self.cull)
            && generation.is_some()
            && self.motion_generation == generation;
        self.motion_generation = if kept { generation } else { None };
        cx.memory(
            "memory.gpu_previous_pose_bytes",
            if kept { bytes } else { 0 },
        );
        if kept {
            engine_core::profile::counter("render.object_motion", f64::from(u8::from(motion)));
            cx.copy(
                self.keep,
                &[CopyRegion::Bytes {
                    from: 0,
                    to: 0,
                    size: bytes,
                }],
            );
        }
        motion
    }

    fn prepare_occlusion(
        &mut self,
        main: Option<&ViewFacts>,
        (wanted, slots): (bool, u64),
        cx: &mut Prepare<'_, Self>,
    ) -> u32 {
        let occlusion = &mut self.occlusion;
        let bytes = slots * 4;
        let fresh = cx.allocate_buffer(occlusion.held, bytes) == Allocation::Fresh;
        let Some(main) = main.filter(|_| wanted) else {
            if occlusion.running {
                cx.clear_buffer_range(occlusion.held, 0..bytes);
                cx.release(occlusion.pyramid);
                for &extent in &occlusion.extents[..occlusion.written.len()] {
                    cx.release_buffer(extent);
                }
                occlusion.written.clear();
                occlusion.running = false;
            }
            return 0;
        };
        if fresh || !occlusion.running {
            cx.clear_buffer_range(occlusion.held, 0..bytes);
        }
        occlusion.running = true;
        let (display, render) = (main.size(), main.render);
        let base = (
            display.x.div_ceil(2).max(1).next_power_of_two(),
            display.y.div_ceil(2).max(1).next_power_of_two(),
        );
        let count = mip_levels(base.0, base.1).min(PYRAMID_LEVELS as u32);
        cx.allocate(
            occlusion.pyramid,
            wgpu::Extent3d {
                width: base.0,
                height: base.1,
                depth_or_array_layers: 1,
            },
            count,
        );
        occlusion.written.clear();
        for (level, extent) in pyramid_extents((render.x, render.y), count)
            .iter()
            .enumerate()
        {
            let buffer = occlusion.extents[level];
            cx.allocate_buffer(buffer, 16);
            cx.write(buffer, 0, extent);
            occlusion.written.push((extent[0], extent[1]));
        }
        cx.run(occlusion.pass);
        count
    }
}

fn install_occlusion(
    cx: &mut Install<'_, DrawnGpu>,
    cull: Pass<DrawnGpu>,
    frame: Buffer<DrawnGpu>,
) -> Result<Occlusion, Error> {
    let pass = cx
        .pass("drawn_occlusion")
        .at(AttachPoint::Opaque)
        .after(cull)
        .label("engine/drawn-occlusion")
        .id();
    let held = cx.buffer("drawn_occlusion", Lifetime::Kept);
    cx.frame(var(DRAWN_MODULE, "drawn_occlusion"), Bind::Buffer(held))?;
    let pyramid = cx.texture("drawn_depth_pyramid", PYRAMID_FORMAT, Lifetime::Kept);
    let extents: Vec<Buffer<DrawnGpu>> = (0..PYRAMID_LEVELS)
        .map(|_| cx.buffer("drawn_pyramid_extent", Lifetime::Kept))
        .collect();
    let entry = |name: &'static str| Entry::generated(core_module(DRAWN_CULL_MODULE), name);
    let first = cx.compute_pipeline(
        pass,
        entry("drawn_pyramid_first"),
        &[
            (var(DRAWN_CULL_MODULE, "source_depth"), Bind::Target(TargetName::DEPTH)),
            (
                var(DRAWN_CULL_MODULE, "target_level"),
                Bind::Slice {
                    texture: pyramid,
                    mips: 0..1,
                    layers: 0..1,
                },
            ),
            (var(DRAWN_CULL_MODULE, "pyramid_extent"), Bind::Buffer(extents[0])),
        ],
    )?;
    let mut down = Vec::with_capacity(PYRAMID_LEVELS - 1);
    for level in 1..PYRAMID_LEVELS as u32 {
        down.push(cx.compute_pipeline(
            pass,
            entry("drawn_pyramid_down"),
            &[
                (
                    var(DRAWN_CULL_MODULE, "source_level"),
                    Bind::Slice {
                        texture: pyramid,
                        mips: level - 1..level,
                        layers: 0..1,
                    },
                ),
                (
                    var(DRAWN_CULL_MODULE, "target_level"),
                    Bind::Slice {
                        texture: pyramid,
                        mips: level..level + 1,
                        layers: 0..1,
                    },
                ),
                (
                    var(DRAWN_CULL_MODULE, "pyramid_extent"),
                    Bind::Buffer(extents[level as usize]),
                ),
            ],
        )?);
    }
    let test = cx.compute_pipeline(
        pass,
        entry("drawn_occlusion_test"),
        &[
            (var(DRAWN_CULL_MODULE, "cull_frame"), Bind::Buffer(frame)),
            (var(DRAWN_CULL_MODULE, "cull_occlusion"), Bind::Buffer(held)),
            (var(DRAWN_CULL_MODULE, "depth_pyramid"), Bind::Texture(pyramid)),
        ],
    )?;
    Ok(Occlusion {
        pass,
        held,
        pyramid,
        extents,
        first,
        down,
        test,
        written: Vec::new(),
        running: false,
    })
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
        let records = cx.buffer("drawn_records", Lifetime::Kept);
        let values = cx.buffer("drawn_values", Lifetime::Kept);
        let fades = cx.buffer("drawn_fades", Lifetime::Kept);
        let culled = cx.buffer("drawn_culled", Lifetime::Kept);
        let views = cx.buffer("drawn_views", Lifetime::Kept);
        let view = cx.view_buffer("drawn_view", Lifetime::Kept);
        let frame = cx.buffer("drawn_frame", Lifetime::Kept);
        let poses = cx.buffer("drawn_poses", Lifetime::Kept);
        let previous = cx.buffer("drawn_previous_poses", Lifetime::Previous);
        let marks = cx.buffer("drawn_marks", Lifetime::Kept);
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
            level_ranges: Vec::new(),
            seen: Cursor::default(),
            whole: false,
            ranges: Vec::new(),
            bits: Vec::new(),
        };
        cx.frame(var(DRAWN_MODULE, "drawn_records"), Bind::Buffer(records))?;
        cx.frame(var(DRAWN_MODULE, "drawn_values"), Bind::Buffer(values))?;
        cx.frame(var(DRAWN_MODULE, "drawn_culled"), Bind::Buffer(culled))?;
        cx.frame(var(DRAWN_MODULE, "drawn_view"), Bind::Buffer(view))?;
        cx.frame(var(DRAWN_MODULE, "drawn_fades"), Bind::Buffer(fades))?;
        cx.frame(var(DRAWN_MODULE, "drawn_views"), Bind::Buffer(views))?;
        cx.frame(var(DRAWN_MODULE, "drawn_poses"), Bind::Buffer(poses))?;
        cx.frame(var(DRAWN_MODULE, "drawn_previous_poses"), Bind::Buffer(previous))?;
        cx.frame(var(DRAWN_MODULE, "drawn_marks"), Bind::Buffer(marks))?;
        cx.frame(var(DRAWN_FADES_MODULE, "drawn_fade_words"), Bind::Buffer(fades))?;
        let cull = cx.pass("drawn_cull").label("engine/drawn-cull").id();
        let keep = cx.copy(
            "drawn_keep_poses",
            Copied::Buffer(poses),
            Copied::Buffer(previous),
        );
        let entry = |name: &'static str| Entry::generated(core_module(DRAWN_CULL_MODULE), name);
        let node_binds = |level: Buffer<Self>| {
            [
                (var(DRAWN_CULL_MODULE, "cull_frame"), Bind::Buffer(frame)),
                (var(DRAWN_CULL_MODULE, "node_locals"), Bind::Buffer(nodes.locals)),
                (var(DRAWN_CULL_MODULE, "node_info"), Bind::Buffer(nodes.info)),
                (var(DRAWN_CULL_MODULE, "node_authority"), Bind::Buffer(nodes.authority)),
                (var(DRAWN_CULL_MODULE, "node_composed"), Bind::Buffer(nodes.composed)),
                (var(DRAWN_CULL_MODULE, "node_parents"), Bind::Buffer(nodes.parents)),
                (var(DRAWN_CULL_MODULE, "node_level"), Bind::Buffer(level)),
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
                (var(DRAWN_CULL_MODULE, "cull_frame"), Bind::Buffer(frame)),
                (var(DRAWN_CULL_MODULE, "node_locals"), Bind::Buffer(nodes.locals)),
                (var(DRAWN_CULL_MODULE, "node_info"), Bind::Buffer(nodes.info)),
                (var(DRAWN_CULL_MODULE, "node_authority"), Bind::Buffer(nodes.authority)),
                (var(DRAWN_CULL_MODULE, "node_composed"), Bind::Buffer(nodes.composed)),
                (var(DRAWN_CULL_MODULE, "cull_poses"), Bind::Buffer(poses)),
            ],
        )?;
        let pipeline = cx.compute_pipeline(
            cull,
            entry("drawn_cull"),
            &[
                (var(DRAWN_CULL_MODULE, "cull_frame"), Bind::Buffer(frame)),
                (var(DRAWN_CULL_MODULE, "node_info"), Bind::Buffer(nodes.info)),
                (var(DRAWN_CULL_MODULE, "cull_bits"), Bind::Buffer(culled)),
                (var(DRAWN_CULL_MODULE, "cull_marks"), Bind::Buffer(marks)),
            ],
        )?;
        let occlusion = install_occlusion(cx, cull, frame)?;
        Ok(Self {
            records,
            values,
            fades,
            culled,
            views,
            view,
            frame,
            poses,
            previous,
            marks,
            nodes,
            occlusion,
            cull,
            keep,
            compose,
            pose,
            pipeline,
            seen: Cursor::default(),
            stride: 0,
            slots: 0,
            groups: [0; 3],
            motion_generation: None,
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
        let main = views.main();
        let streams = poses.streams(&hierarchy);
        let uploaded = self.upload_nodes(&poses, &streams, cx);
        let motion = self.prepare_poses((&*table, main), slots, cx);
        cx.allocate_buffer(self.marks, slots * 4);
        let levels = self.prepare_occlusion(main, (plan.occlusion, slots), cx);
        let entries = cull_views(&views, passes.as_deref(), &plan, levels);
        let words = (count as u32).div_ceil(SLOT_BITS).max(1);
        let bytes = (1 + u64::from(words) * entries.len() as u64) * 4;
        if limits.storage_room != 0 && bytes > limits.storage_room {
            cx.refuse(
                Error::new(ErrorCode::BindingLimit, "$.drawn.views")
                    .at_most(limits.storage_room)
                    .got(bytes),
            );
            return;
        }
        cx.allocate_buffer(self.views, (entries.len() * size_of::<CullView>()) as u64);
        cx.write(self.views, 0, &entries);
        cx.allocate_buffer(self.culled, bytes.next_power_of_two());
        cx.write(self.culled, 0, &[words]);
        self.slots = count as u32;
        cx.allocate_buffer(self.frame, size_of::<DrawnFrame>() as u64);
        cx.write(
            self.frame,
            0,
            &[DrawnFrame {
                alpha: streams.alpha,
                capacity: self.nodes.capacity as u32,
                nodes: streams.info.len() as u32,
                slots: self.slots,
                views: entries.len() as u32,
                motion: u32::from(motion),
                pad_0: 0,
                pad_1: 0,
            }],
        );
        self.groups = [words.div_ceil(CULL_GROUP), entries.len() as u32, 1];
        cx.run(self.cull);
        if engine_core::profile::active() {
            use engine_core::profile::counter;
            let (full, bytes) = uploaded.unwrap_or((false, 0));
            counter("render.gpu_node_upload_bytes", bytes as f64);
            counter("render.gpu_full_node_upload", f64::from(u8::from(full)));
            counter(
                "render.gpu_parent_levels",
                self.nodes.level_ranges.len() as f64,
            );
        }
    }

    fn view(&mut self, _: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        let index = cx.view_index();
        match cx.allocate_buffer(self.view, VIEW_BYTES) {
            Allocation::Fresh | Allocation::Kept => cx.write(self.view, 0, &[index, 0, 0, 0]),
        }
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.occlusion.pass {
            self.encode_occlusion(cx);
            return 0;
        }
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
        if self.slots != 0 {
            cx.dispatch(
                &self.pose,
                Count::Groups([self.slots.div_ceil(CULL_GROUP), 1, 1]),
            );
        }
        cx.dispatch(&self.pipeline, Count::Groups(self.groups))
    }
}

impl DrawnGpu {
    fn encode_occlusion(&self, cx: &mut Encode<'_, Self>) {
        let occlusion = &self.occlusion;
        for (index, &(width, height)) in occlusion.written.iter().enumerate() {
            let pipeline = match index.checked_sub(1) {
                None => &occlusion.first,
                Some(down) => match occlusion.down.get(down) {
                    Some(pipeline) => pipeline,
                    None => break,
                },
            };
            cx.dispatch(
                pipeline,
                Count::Groups([
                    width.div_ceil(PYRAMID_GROUP),
                    height.div_ceil(PYRAMID_GROUP),
                    1,
                ]),
            );
        }
        if self.slots != 0 {
            cx.dispatch(
                &occlusion.test,
                Count::Groups([self.slots.div_ceil(CULL_GROUP), 1, 1]),
            );
        }
    }
}
