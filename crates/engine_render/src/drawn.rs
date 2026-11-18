use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::system::{Res, SystemParamItem};
use engine_core::{
    DrawnPoses, DrawnRecord, DrawnTable, Entry, Error, GeometryHandle, GlobalTransforms, GpuStruct,
    Layers, MAX_HIERARCHY_DEPTH, MaterialHandle, NodeInfo, NodePose, NodeStreams, PassBit,
    PassMask, PoseRows, Var, ViewFacts, gpu_struct,
};
use engine_place::Changed;

use crate::shaders::core_module;
use crate::uploads::{UploadRange, listed_ranges};
use crate::{
    Allocation, Bind, Buffer, ComputePipeline, Count, Encode, GpuPart, Install, Lifetime, Pass,
    Recorded,
    Prepare,
};

pub(crate) const DRAWN_MODULE: &str = "engine::drawn";
pub(crate) const DRAWN_POSE_MODULE: &str = "engine::drawn_pose";
const POSE_GROUP: u32 = 64;
const FIRST_SLOTS: u64 = 64;
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
@group(0) @binding(0) var<storage, read> drawn_values: array<vec4<f32>>;
@group(0) @binding(0) var<storage, read> drawn_order: array<u32>;
@group(0) @binding(0) var<storage, read> drawn_poses: array<DrawnPose>;

fn drawn_model(slot: u32) -> mat4x4<f32> {
    let pose = drawn_poses[slot];
    return transpose(mat4x4<f32>(pose.row_0, pose.row_1, pose.row_2,
        vec4<f32>(0.0, 0.0, 0.0, 1.0)));
}
";

const POSE_IMPORTS: &str = "import package::engine::drawn::{DrawnPose, drawn_records};
import package::engine::presentation::{NO_PARENT, POSE_FLOATS, Pose, drawn_local,
    pose_admitted, pose_trs, pose_word_blend, pose_word_composed, pose_word_parent_composed,
    pose_word_valid};
";

const POSE: &str = "
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

@group(2) @binding(0) var<uniform> pose_frame: DrawnFrame;
@group(2) @binding(1) var<storage, read> node_locals: array<f32>;
@group(2) @binding(2) var<storage, read> node_info: array<vec2<u32>>;
@group(2) @binding(3) var<storage, read> node_authority: array<Rows>;
@group(2) @binding(4) var<storage, read_write> node_composed: array<Composed>;
@group(2) @binding(5) var<storage, read> node_parents: array<u32>;
@group(2) @binding(6) var<uniform> node_level: Level;
@group(2) @binding(7) var<storage, read_write> posed: array<DrawnPose>;

fn rows_matrix(rows: Rows) -> mat4x4<f32> {
    return transpose(mat4x4<f32>(rows.r0, rows.r1, rows.r2, vec4<f32>(0.0, 0.0, 0.0, 1.0)));
}

fn matrix_rows(m: mat4x4<f32>) -> Rows {
    let t = transpose(m);
    return Rows(t[0], t[1], t[2]);
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
        let was = node_pose((pose_frame.capacity + node) * POSE_FLOATS);
        let display = base_display * drawn_local(was, now, pose_frame.alpha, pose_word_blend(word));
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
    return node < pose_frame.nodes && pose_word_valid(node_info[node].y);
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
    if slot >= pose_frame.slots {
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
    posed[slot] = DrawnPose(rows.r0, rows.r1, rows.r2);
}
";

gpu_struct! {
    pub struct DrawnFrame {
        pub alpha: f32,
        pub capacity: u32,
        pub nodes: u32,
        pub slots: u32,
    }
}

pub(crate) fn source() -> String {
    let mut source = engine_core::drawn_wgsl();
    source.push_str(VARS);
    source
}

pub(crate) fn pose_source() -> String {
    let mut source = String::from(POSE_IMPORTS);
    source.push_str(DrawnFrame::WGSL);
    source.push_str(POSE);
    source
}

pub(crate) struct KindBatch {
    pub(crate) instances: Range<u32>,
    pub(crate) geometry: Option<GeometryHandle>,
    pub(crate) material: Option<MaterialHandle>,
    passes: Option<PassMask>,
    layers: Layers,
    #[cfg(any(feature = "dev", feature = "profiling"))]
    pub(crate) cause: u32,
    #[cfg(any(feature = "dev", feature = "profiling"))]
    pub(crate) drawn: std::sync::atomic::AtomicBool,
}

#[derive(Clone, Default)]
pub(crate) struct Batches {
    listed: Arc<[KindBatch]>,
    layers: Layers,
    filled: bool,
}

impl Batches {
    pub(crate) fn drawn(&self, pass: Option<PassBit>) -> impl Iterator<Item = &KindBatch> + '_ {
        let layers = self.layers;
        self.listed.iter().filter(move |batch| {
            batch.layers.overlaps(layers)
                && batch
                    .passes
                    .is_none_or(|mask| pass.is_some_and(|bit| mask.has(bit)))
        })
    }

    fn fill(&mut self, table: &DrawnTable, kind: engine_core::DrawKind) {
        #[cfg(any(feature = "dev", feature = "profiling"))]
        let held = std::mem::take(&mut self.listed);
        self.listed = table
            .batches(kind)
            .map(|batch| KindBatch {
                instances: batch.instances.clone(),
                geometry: batch.geometry,
                material: batch.material,
                passes: batch.passes,
                layers: batch.layers,
                #[cfg(any(feature = "dev", feature = "profiling"))]
                cause: batch.cause,
                #[cfg(any(feature = "dev", feature = "profiling"))]
                drawn: held
                    .iter()
                    .any(|known| {
                        known.cause == batch.cause
                            && known.geometry == batch.geometry
                            && known.material == batch.material
                            && known.passes == batch.passes
                            && known.layers == batch.layers
                            && known.drawn.load(std::sync::atomic::Ordering::Relaxed)
                    })
                    .into(),
            })
            .collect();
        self.filled = true;
    }
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
    whole: bool,
}

pub(crate) struct DrawnGpu {
    records: Buffer<Self>,
    values: Buffer<Self>,
    order: Buffer<Self>,
    frame: Buffer<Self>,
    poses: Buffer<Self>,
    nodes: Nodes,
    pass: Pass<Self>,
    compose: Vec<ComputePipeline<Self>>,
    pose: ComputePipeline<Self>,
    scratch: Scratch,
    frame_sent: DrawnFrame,
    slots: u32,
    owed: bool,
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

struct Scratch {
    bits: Vec<u64>,
    held: Vec<u32>,
    ranges: Vec<UploadRange>,
}

impl Scratch {
    fn changed(&mut self, changed: &Changed<'_>, whole: bool, count: usize) {
        self.ranges.clear();
        if whole || changed.all {
            if count != 0 {
                self.ranges.push(UploadRange {
                    start: 0,
                    end: count,
                });
            }
            return;
        }
        self.held.clear();
        self.held
            .extend(changed.listed.iter().filter(|&&at| (at as usize) < count));
        listed_ranges(&self.held, count, &mut self.bits, &mut self.ranges);
    }
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
) {
    let size = per * size_of::<T>();
    for range in merged(ranges, size) {
        cx.write(
            buffer,
            base + (range.start * size) as u64,
            &data[range.start * per..range.end * per],
        );
    }
}

impl DrawnGpu {
    fn upload_slots(&mut self, table: &DrawnTable, cx: &mut Prepare<'_, Self>) -> bool {
        let count = table.slots();
        let room = (count as u64).next_power_of_two().max(FIRST_SLOTS);
        let stride = table.value_vec4s();
        let record_bytes = room * size_of::<DrawnRecord>() as u64;
        let value_bytes = room * (stride * size_of::<[f32; 4]>()) as u64;
        let mut whole =
            cx.allocate_buffer(self.records, storage_bytes(record_bytes)) == Allocation::Fresh;
        whole |= cx.allocate_buffer(self.values, storage_bytes(value_bytes)) == Allocation::Fresh;
        let fresh = cx.allocate_buffer(self.poses, room * POSE_BYTES) == Allocation::Fresh;
        let changes = table.slot_changes();
        let read = changes.read();
        self.scratch.changed(&read, whole, count);
        let ranges = &self.scratch.ranges;
        write_ranges(cx, (self.records, 0), (table.records(), 1), ranges);
        write_ranges(cx, (self.values, 0), (table.values(), stride), ranges);
        changes.take(0, read.upto);
        let moved = fresh || !ranges.is_empty() || self.slots != count as u32;
        self.slots = count as u32;
        moved
    }

    fn upload_order(&mut self, table: &DrawnTable, cx: &mut Prepare<'_, Self>) -> bool {
        let order = table.order();
        let bytes = (order.len() as u64).next_power_of_two() * size_of::<u32>() as u64;
        let whole = cx.allocate_buffer(self.order, storage_bytes(bytes)) == Allocation::Fresh;
        let changes = table.order_changes();
        let read = changes.read();
        self.scratch.changed(&read, whole, order.len());
        write_ranges(cx, (self.order, 0), (order, 1), &self.scratch.ranges);
        changes.take(0, read.upto);
        whole || read.all || !read.listed.is_empty()
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
    ) -> bool {
        let count = streams.info.len();
        self.ensure_nodes(count, streams.parents.len(), cx);
        let changes = poses.node_changes();
        let read = changes.read();
        let whole = std::mem::take(&mut self.nodes.whole);
        if !whole && !read.all && read.listed.is_empty() {
            return false;
        }
        self.scratch.changed(&read, whole, count);
        let held = &mut self.nodes;
        let previous = (held.capacity * size_of::<NodePose>()) as u64;
        let ranges = &self.scratch.ranges;
        write_ranges(cx, (held.locals, 0), (streams.current, 1), ranges);
        write_ranges(cx, (held.locals, previous), (streams.previous, 1), ranges);
        write_ranges(cx, (held.info, 0), (streams.info, 1), ranges);
        write_ranges(cx, (held.authority, 0), (streams.authority, 1), ranges);
        if whole {
            held.parents_sent.clear();
            held.level_ranges.clear();
        }
        if !streams.parents.is_empty() && held.parents_sent.as_slice() != streams.parents {
            cx.write(held.parents, 0, streams.parents);
            held.parents_sent.clear();
            held.parents_sent.extend_from_slice(streams.parents);
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
            }
            match held.level_ranges.get_mut(level) {
                Some(known) => *known = range,
                None => held.level_ranges.push(range),
            }
        }
        held.level_ranges.truncate(streams.levels.len());
        changes.take(0, read.upto);
        true
    }
}

impl GpuPart for DrawnGpu {
    type Reads = (
        Res<'static, DrawnTable>,
        Res<'static, DrawnPoses>,
        Res<'static, GlobalTransforms>,
    );
    const NAME: &'static str = "drawn";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let drawn_at = |name: &'static str| var(DRAWN_MODULE, name);
        let posed_at = |name: &'static str| var(DRAWN_POSE_MODULE, name);
        let records = cx.buffer("drawn_records", Lifetime::Kept);
        let values = cx.buffer("drawn_values", Lifetime::Kept);
        let order = cx.buffer("drawn_order", Lifetime::Kept);
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
            whole: false,
        };
        cx.frame(drawn_at("drawn_records"), Bind::Buffer(records))?;
        cx.frame(drawn_at("drawn_values"), Bind::Buffer(values))?;
        cx.frame(drawn_at("drawn_order"), Bind::Buffer(order))?;
        cx.frame(drawn_at("drawn_poses"), Bind::Buffer(poses))?;
        let pass = cx.pass("drawn_pose").label("engine/drawn-pose").id();
        let entry = |name: &'static str| Entry::generated(core_module(DRAWN_POSE_MODULE), name);
        let node_binds = |level: Buffer<Self>| {
            [
                (posed_at("pose_frame"), Bind::Buffer(frame)),
                (posed_at("node_locals"), Bind::Buffer(nodes.locals)),
                (posed_at("node_info"), Bind::Buffer(nodes.info)),
                (posed_at("node_authority"), Bind::Buffer(nodes.authority)),
                (posed_at("node_composed"), Bind::Buffer(nodes.composed)),
                (posed_at("node_parents"), Bind::Buffer(nodes.parents)),
                (posed_at("node_level"), Bind::Buffer(level)),
            ]
        };
        let mut compose = Vec::with_capacity(NODE_LEVELS);
        for &level in &nodes.levels {
            let binds = node_binds(level);
            compose.push(cx.compute_pipeline(pass, entry("drawn_compose"), &[], &binds)?);
        }
        let pose = cx.compute_pipeline(
            pass,
            entry("drawn_pose"),
            &[],
            &[
                (posed_at("pose_frame"), Bind::Buffer(frame)),
                (posed_at("node_locals"), Bind::Buffer(nodes.locals)),
                (posed_at("node_info"), Bind::Buffer(nodes.info)),
                (posed_at("node_authority"), Bind::Buffer(nodes.authority)),
                (posed_at("node_composed"), Bind::Buffer(nodes.composed)),
                (posed_at("posed"), Bind::Buffer(poses)),
            ],
        )?;
        Ok(Self {
            records,
            values,
            order,
            frame,
            poses,
            nodes,
            pass,
            compose,
            pose,
            scratch: Scratch {
                bits: Vec::new(),
                held: Vec::new(),
                ranges: Vec::new(),
            },
            frame_sent: DrawnFrame::default(),
            slots: 0,
            owed: true,
        })
    }

    fn prepare(
        &mut self,
        (table, poses, hierarchy): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.owed |= self.upload_slots(&table, cx);
        let moved = self.upload_order(&table, cx);
        for member in &mut cx.core.installs.draw_entries {
            if moved || !member.batches.filled {
                member.batches.fill(&table, member.decl.kind);
            }
        }
        let streams = poses.streams(&hierarchy);
        self.owed |= self.upload_nodes(&poses, &streams, cx);
        let fresh =
            cx.allocate_buffer(self.frame, size_of::<DrawnFrame>() as u64) == Allocation::Fresh;
        let frame = DrawnFrame {
            alpha: if streams.blended > 0 {
                streams.alpha
            } else {
                self.frame_sent.alpha
            },
            capacity: self.nodes.capacity as u32,
            nodes: streams.info.len() as u32,
            slots: self.slots,
        };
        if fresh || bytemuck::bytes_of(&frame) != bytemuck::bytes_of(&self.frame_sent) {
            cx.write(self.frame, 0, &[frame]);
            self.frame_sent = frame;
            self.owed = true;
        }
        if self.owed && self.slots != 0 {
            cx.run(self.pass);
        }
    }

    fn view(&mut self, facts: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        for member in &mut cx.core.installs.draw_entries {
            member.batches.layers = facts.layers;
        }
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) {
        if pass != self.pass {
            return;
        }
        cx.recorded();
        for (level, &(start, end)) in self.nodes.level_ranges.iter().enumerate() {
            if end > start
                && let Some(pipeline) = self.compose.get(level)
            {
                cx.dispatch(
                    pipeline,
                    Count::Groups([(end - start).div_ceil(POSE_GROUP), 1, 1]),
                );
            }
        }
        cx.dispatch(
            &self.pose,
            Count::Groups([self.slots.div_ceil(POSE_GROUP), 1, 1]),
        );
        self.owed = cx.recorded() != Recorded::Exact;
    }
}