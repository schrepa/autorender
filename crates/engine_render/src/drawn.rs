use std::ops::Range;

use bevy_ecs::query::With;
use bevy_ecs::system::{Query, Res, SystemParamItem};
use engine_core::{
    Asset, Assets, Drawn, DrawnRecord, DrawnTable, Entry, Error, Layers, Material, MeshFile,
    PassBit, PassMask, Var, ViewFacts,
};
use engine_place::{Changed, DrawnPoses, PoseEntry, PoseFrame, PoseRows};

use crate::BatchOf;
use crate::encode::DrawRecord;
use crate::shaders::core_module;
use crate::uploads::{UploadRange, listed_ranges};
use crate::{
    Allocation, Bind, Buffer, ComputePipeline, Encode, GpuPart, Install, Lifetime, Pass, Prepare,
    Recorded,
};

pub(crate) const DRAWN_MODULE: &str = "engine::drawn";
pub(crate) const POSES_MODULE: &str = "engine::poses";
pub(crate) const DRAWN_POSE_MODULE: &str = "engine::drawn_pose";
const POSE_GROUP: u32 = 64;
const MIN_STORAGE_BYTES: u64 = 256;
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

const POSE: &str = "import package::engine::drawn::DrawnPose;
import package::engine::poses::{NO_PARENT, NO_SLOT, PoseEntry, PoseFrame, pose_admitted,
    pose_base, pose_local};

@group(1) @binding(0) var<uniform> pose_frame: PoseFrame;
@group(1) @binding(1) var<storage, read> pose_entries: array<PoseEntry>;
@group(1) @binding(2) var<storage, read_write> posed: array<DrawnPose>;

fn pose_chain(first: u32, alpha: f32) -> mat4x4<f32> {
    var m = pose_local(pose_entries[first], alpha);
    var at = pose_entries[first].parent;
    while at != NO_PARENT {
        m = pose_local(pose_entries[at], alpha) * m;
        at = pose_entries[at].parent;
    }
    return pose_base(pose_entries[first]) * m;
}

@compute @workgroup_size(POSE_GROUP)
fn drawn_pose(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= pose_frame.entries || pose_entries[id.x].slot == NO_SLOT {
        return;
    }
    var m = pose_chain(id.x, pose_frame.alpha);
    if !pose_admitted(m[0], m[1], m[2], m[3]) {
        m = pose_chain(id.x, 1.0);
    }
    let rows = transpose(m);
    posed[pose_entries[id.x].slot] = DrawnPose(rows[0], rows[1], rows[2]);
}
";

pub(crate) fn source() -> String {
    let mut source = engine_core::drawn_wgsl();
    source.push_str(VARS);
    source
}

pub(crate) fn pose_source() -> String {
    format!("{POSE}\nconst POSE_GROUP: u32 = {POSE_GROUP}u;\n")
}

#[derive(Clone)]
pub(crate) struct KindBatch {
    pub(crate) instances: Range<u32>,
    pub(crate) geometry: Option<Asset<MeshFile>>,
    pub(crate) material: Option<Asset<Material>>,
    pub(crate) held: bool,
    passes: Option<PassMask>,
    layers: Layers,
    pub(crate) records: Vec<DrawRecord>,
}

impl KindBatch {
    fn shown(&self, layers: Layers, pass: Option<PassBit>) -> bool {
        self.layers.overlaps(layers)
            && self
                .passes
                .is_none_or(|mask| pass.is_some_and(|bit| mask.has(bit)))
    }
}

#[derive(Clone, Default)]
pub(crate) struct Batches {
    pub(crate) listed: Vec<KindBatch>,
    layers: Layers,
    filled: bool,
    pub(crate) holds: bool,
    pub(crate) judged: Option<(u64, u32)>,
}

#[derive(Default)]
pub(crate) struct Verdicts {
    order: Vec<u32>,
    numbers: Vec<Option<u32>>,
}

impl Verdicts {
    pub(crate) fn judge(
        &mut self,
        batch: &KindBatch,
        verdict: &mut dyn FnMut(BatchOf<'_>, u32) -> Option<u32>,
    ) {
        for at in batch.instances.clone() {
            let Some(&slot) = self.order.get(at as usize) else {
                continue;
            };
            if let Some(number) = self.numbers.get_mut(slot as usize) {
                *number = verdict(BatchOf::of(batch), slot);
            }
        }
    }

    pub(crate) fn runs(&self, instances: &Range<u32>, mut kept: impl FnMut(Range<u32>, u32)) {
        let mut open = None;
        for at in instances.clone() {
            let number = (self.order.get(at as usize))
                .and_then(|&slot| self.numbers.get(slot as usize).copied().flatten());
            match (open, number) {
                (None, Some(number)) => open = Some((at, number)),
                (Some((start, first)), None) => {
                    kept(start..at, first);
                    open = None;
                }
                (Some(_), Some(_)) | (None, None) => {}
            }
        }
        if let Some((start, first)) = open {
            kept(start..instances.end, first);
        }
    }
}

impl Batches {
    pub(crate) fn drawn(&self, pass: Option<PassBit>) -> impl Iterator<Item = &KindBatch> + '_ {
        let layers = self.layers;
        self.listed.iter().filter(move |batch| batch.shown(layers, pass))
    }

    pub(crate) fn drawn_mut(
        &mut self,
        pass: Option<PassBit>,
    ) -> impl Iterator<Item = (usize, &mut KindBatch)> + '_ {
        let layers = self.layers;
        (self.listed.iter_mut().enumerate()).filter(move |(_, batch)| batch.shown(layers, pass))
    }

    pub(crate) fn forget(&mut self, key: Option<Asset<Material>>) {
        for batch in &mut self.listed {
            if batch.material == key {
                batch.records.clear();
            }
        }
    }

    pub(crate) fn forget_all(&mut self) {
        for batch in &mut self.listed {
            batch.records.clear();
        }
    }

    fn fill(&mut self, table: &DrawnTable, kind: engine_core::DrawKind) {
        let mut count = 0;
        for batch in table.batches(kind) {
            match self.listed.get_mut(count) {
                Some(known) => {
                    if known.geometry != batch.geometry || known.material != batch.material {
                        known.records.clear();
                        known.geometry = batch.geometry;
                        known.material = batch.material;
                    }
                    known.instances.clone_from(&batch.instances);
                    known.held = batch.held;
                    known.passes = batch.passes;
                    known.layers = batch.layers;
                }
                None => self.listed.push(KindBatch {
                    instances: batch.instances.clone(),
                    geometry: batch.geometry,
                    material: batch.material,
                    held: batch.held,
                    passes: batch.passes,
                    layers: batch.layers,
                    records: Vec::new(),
                }),
            }
            count += 1;
        }
        self.listed.truncate(count);
        self.filled = true;
        self.holds = self.listed.iter().any(|batch| batch.held);
    }
}

pub(crate) struct DrawnGpu {
    records: Buffer<Self>,
    values: Buffer<Self>,
    order: Buffer<Self>,
    poses: Buffer<Self>,
    frame: Buffer<Self>,
    entries: Buffer<Self>,
    pass: Pass<Self>,
    pose: ComputePipeline<Self>,
    scratch: Scratch,
    frame_sent: PoseFrame,
    entries_sent: Vec<PoseEntry>,
    owed: bool,
    seen: Option<(u64, u64)>,
}

fn var(module: &str, name: &'static str) -> Var {
    Var::generated(core_module(module), name)
}

fn storage_bytes(bytes: u64) -> u64 {
    bytes.max(MIN_STORAGE_BYTES)
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
    fn upload_slots(
        &mut self,
        (table, entities): (&DrawnTable, usize),
        cx: &mut Prepare<'_, Self>,
    ) -> bool {
        let count = table.slots();
        let room = (count.max(entities) as u64).next_power_of_two();
        let stride = table.value_vec4s();
        let record_bytes = room * size_of::<DrawnRecord>() as u64;
        let value_bytes = room * (stride * size_of::<[f32; 4]>()) as u64;
        let pose_bytes = room * size_of::<PoseRows>() as u64;
        let mut whole =
            cx.allocate_buffer(self.records, storage_bytes(record_bytes)) == Allocation::Fresh;
        whole |= cx.allocate_buffer(self.values, storage_bytes(value_bytes)) == Allocation::Fresh;
        whole |= cx.allocate_buffer(self.poses, storage_bytes(pose_bytes)) == Allocation::Fresh;
        let changes = table.slot_changes();
        let read = changes.read();
        self.scratch.changed(&read, whole, count);
        let ranges = &self.scratch.ranges;
        write_ranges(cx, (self.records, 0), (table.records(), 1), ranges);
        write_ranges(cx, (self.values, 0), (table.values(), stride), ranges);
        write_ranges(cx, (self.poses, 0), (table.poses(), 1), ranges);
        changes.take(0, read.upto);
        !ranges.is_empty()
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

    fn upload_entries(&mut self, entries: &[PoseEntry], cx: &mut Prepare<'_, Self>) -> bool {
        let bytes = (entries.len() as u64).next_power_of_two() * size_of::<PoseEntry>() as u64;
        if cx.allocate_buffer(self.entries, storage_bytes(bytes)) == Allocation::Fresh {
            self.entries_sent.clear();
        }
        let sent: &[u8] = bytemuck::cast_slice(&self.entries_sent);
        if sent == bytemuck::cast_slice::<PoseEntry, u8>(entries) {
            return false;
        }
        if !entries.is_empty() {
            cx.write(self.entries, 0, entries);
        }
        self.entries_sent.clear();
        self.entries_sent.extend_from_slice(entries);
        true
    }

    fn upload_frame(&mut self, frame: PoseFrame, cx: &mut Prepare<'_, Self>) -> bool {
        if cx.allocate_buffer(self.frame, size_of::<PoseFrame>() as u64) == Allocation::Fresh {
            self.frame_sent = PoseFrame::default();
        }
        if frame.entries == 0 || bytemuck::bytes_of(&frame) == bytemuck::bytes_of(&self.frame_sent)
        {
            return false;
        }
        cx.write(self.frame, 0, &[frame]);
        self.frame_sent = frame;
        true
    }
}

impl GpuPart for DrawnGpu {
    type Reads = (
        Res<'static, DrawnTable>,
        Res<'static, DrawnPoses>,
        Option<Res<'static, Assets>>,
        Query<'static, 'static, (), With<Drawn>>,
    );
    const NAME: &'static str = "drawn";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let drawn_at = |name: &'static str| var(DRAWN_MODULE, name);
        let posed_at = |name: &'static str| var(DRAWN_POSE_MODULE, name);
        let records = cx.buffer("drawn_records", Lifetime::Kept);
        let values = cx.buffer("drawn_values", Lifetime::Kept);
        let order = cx.buffer("drawn_order", Lifetime::Kept);
        let poses = cx.buffer("drawn_poses", Lifetime::Kept);
        let frame = cx.buffer("drawn_pose_frame", Lifetime::Kept);
        let entries = cx.buffer("drawn_pose_entries", Lifetime::Kept);
        cx.frame(drawn_at("drawn_records"), Bind::Buffer(records))?;
        cx.frame(drawn_at("drawn_values"), Bind::Buffer(values))?;
        cx.frame(drawn_at("drawn_order"), Bind::Buffer(order))?;
        cx.frame(drawn_at("drawn_poses"), Bind::Buffer(poses))?;
        let pass = cx.pass("drawn_pose").label("engine/drawn-pose").id();
        let pose = cx.compute_pipeline(
            pass,
            Entry::generated(core_module(DRAWN_POSE_MODULE), "drawn_pose"),
            &[],
            &[
                (posed_at("pose_frame"), Bind::Buffer(frame)),
                (posed_at("pose_entries"), Bind::Buffer(entries)),
                (posed_at("posed"), Bind::Buffer(poses)),
            ],
        )?;
        Ok(Self {
            records,
            values,
            order,
            poses,
            frame,
            entries,
            pass,
            pose,
            scratch: Scratch {
                bits: Vec::new(),
                held: Vec::new(),
                ranges: Vec::new(),
            },
            frame_sent: PoseFrame::default(),
            entries_sent: Vec::new(),
            owed: true,
            seen: None,
        })
    }

    fn prepare(
        &mut self,
        (table, poses, assets, drawn): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.owed |= self.upload_slots((&table, drawn.iter().len()), cx);
        let moved = self.upload_order(&table, cx);
        let verdicts = &mut cx.core.verdicts;
        if moved {
            verdicts.order.clear();
            verdicts.order.extend_from_slice(table.order());
        }
        verdicts.numbers.resize(table.slots(), None);
        for member in &mut cx.core.installs.draw_entries {
            if moved || !member.batches.filled || member.batches.holds {
                member.batches.fill(&table, member.decl.kind);
            }
        }
        let revision = assets.map_or(0, |assets| assets.revision());
        cx.core.settle_records(&mut self.seen, revision);
        self.owed |= self.upload_entries(poses.entries(), cx);
        self.owed |= self.upload_frame(poses.frame(), cx);
        if self.owed && !self.entries_sent.is_empty() {
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
        let groups = (self.entries_sent.len() as u32).div_ceil(POSE_GROUP);
        cx.dispatch(&self.pose, [groups, 1, 1]);
        self.owed = cx.recorded() != Recorded::Exact;
    }
}
