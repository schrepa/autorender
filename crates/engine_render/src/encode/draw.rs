use std::ops::Range;

use engine_core::{AssetId, DropReason};

use super::compile::{Landed, Resolve};
use super::run::MeshLayouts;
use super::*;
use crate::drawn::{Batches, KindBatch};
use crate::{BatchOf, DrawOrder, Shape};

#[derive(Clone)]
struct MeshRecord {
    buffers: Box<[wgpu::Buffer]>,
    indices: wgpu::Buffer,
    format: wgpu::IndexFormat,
    range: Range<u32>,
    base_vertex: i32,
}

#[derive(Clone)]
pub(crate) struct DrawRecord {
    pipeline: u32,
    layout: u32,
    rank: u64,
    made: wgpu::RenderPipeline,
    exact: bool,
    groups: Box<[(u32, wgpu::BindGroup)]>,
    mesh: Option<MeshRecord>,
}

#[derive(Clone, Copy)]
struct Head {
    pipeline: u32,
    filled: Option<usize>,
    layout: u32,
}

pub(crate) struct Kept {
    key: (u64, u32, u32),
    at: u32,
    record: u32,
    instances: Range<u32>,
}

fn mesh_record(
    (assets, layouts, faults): (&Assets, &mut MeshLayouts, &mut Vec<Error>),
    (machine, pass): (&Machine, u32),
    (pipeline, mesh): (u32, Asset<MeshFile>),
) -> Result<MeshRecord, DropReason> {
    let draw = assets
        .meshes
        .draw(mesh)
        .ok_or(DropReason::GeometryNotResident)?;
    let held = layouts.entry((pipeline, draw.class)).or_insert_with(|| {
        let key = machine.draw_key(pass, pipeline);
        let path = format!(
            "$.parts.{}.pipelines.{}.meshes.{}",
            key.part,
            key.pipeline,
            mesh.slot()
        );
        let declared = &machine.pipelines[pipeline as usize].geometry;
        crate::mesh_arena::layout(path, declared, draw.streams)
            .map_err(|error| faults.push(error))
            .ok()
    });
    let streams = held.as_deref().ok_or(DropReason::MeshLayoutRefused)?;
    Ok(MeshRecord {
        buffers: streams
            .iter()
            .map(|&at| draw.buffers[at as usize].clone())
            .collect(),
        indices: draw.indices.clone(),
        format: draw.index_format,
        range: draw.first_index..draw.first_index + draw.index_count,
        base_vertex: draw.base_vertex,
    })
}

fn handles_of(shape: &Shape) -> (Option<Asset<MeshFile>>, Option<Asset<Material>>) {
    match shape {
        Shape::Mesh { mesh, key, .. } => (Some(*mesh), *key),
        Shape::Pulled { key, .. } => (None, *key),
    }
}

fn dropped_asset(
    reason: DropReason,
    (mesh, key): (Option<Asset<MeshFile>>, Option<Asset<Material>>),
) -> Option<AssetId> {
    if reason == DropReason::GeometryNotResident {
        mesh.map(Asset::id)
    } else if reason == DropReason::MaterialNotResident {
        key.map(Asset::id)
    } else {
        None
    }
}

fn held_group<'g>(
    groups: &'g Groups,
    set: &'g views::ViewSet,
    holder: u32,
    parity: usize,
) -> Option<&'g wgpu::BindGroup> {
    set.group(holder)
        .map_or(groups.made[parity].as_ref(), |made| made[parity].as_ref())
}

fn rank_of(made: &wgpu::RenderPipeline) -> u64 {
    let mut hasher = DefaultHasher::new();
    made.hash(&mut hasher);
    hasher.finish()
}

fn material_rank(key: Option<Asset<Material>>) -> u32 {
    key.map_or(0, |handle| handle.slot() as u32 + 1)
}

impl Run {
    pub(super) fn scoped(&self, part: u32, point: Option<PassBit>, pipeline: u32) -> bool {
        let Some(built) = self.machine.pipelines.get(pipeline as usize) else {
            return false;
        };
        if built.decl.part != part {
            return false;
        }
        match (built.decl.scope, point) {
            (Scoped::Pass(pass), None) => pass == self.pass,
            (Scoped::Scene(known), Some(point)) => known == point,
            (Scoped::Scene(_), None) | (Scoped::Pass(_), Some(_)) | (Scoped::Outside(_), _) => {
                false
            }
        }
    }

    pub(crate) fn draw(
        &mut self,
        part: u32,
        point: Option<PassBit>,
        handle: u32,
        shape: Shape,
        timer: &GpuProfiler,
    ) -> crate::Recorded {
        let handles = handles_of(&shape);
        let (Shape::Pulled { instances, .. } | Shape::Mesh { instances, .. }) = &shape;
        let vertices = match &shape {
            Shape::Pulled { vertices, .. } => vertices.clone(),
            Shape::Mesh { .. } => 0..0,
        };
        let drawn = (vertices, instances.clone());
        let issued = self.head(part, point, handle).and_then(|head| {
            let record = self.build(head, handles)?;
            self.issue((part, timer), head, &record, drawn)
        });
        self.settle_draw(part, issued, handles)
    }

    pub(crate) fn batches(
        &mut self,
        (part, timer): (u32, &GpuProfiler),
        (handle, vertices, order): (u32, Range<u32>, DrawOrder),
        (batches, pass): (&mut Batches, Option<PassBit>),
    ) {
        let head = self.head(part, self.point, handle);
        let judged = batches.judged == Some(self.view_stamp());
        let mut kept = std::mem::take(&mut self.spare.kept);
        for (at, batch) in batches.drawn_mut(pass) {
            if batch.held {
                self.hold(head, batch);
                continue;
            }
            let from = kept.len();
            let mut push = |instances, number| {
                kept.push(Kept {
                    key: (0, 0, number),
                    at: at as u32,
                    record: 0,
                    instances,
                });
            };
            if judged {
                self.verdicts.runs(&batch.instances, &mut push);
            } else {
                push(batch.instances.clone(), 0);
            }
            if kept.len() == from {
                continue;
            }
            match head.and_then(|head| self.cached(head, batch)) {
                Ok(record) => {
                    let (rank, material) = match order {
                        DrawOrder::Grouped => {
                            (batch.records[record].rank, material_rank(batch.material))
                        }
                        DrawOrder::Sorted => (0, 0),
                    };
                    for run in &mut kept[from..] {
                        run.key = (rank, material, run.key.2);
                        run.record = record as u32;
                    }
                }
                Err(reason) => {
                    kept.truncate(from);
                    let handles = (batch.geometry, batch.material);
                    self.settle_draw(part, Err(reason), handles);
                }
            }
        }
        if !kept.is_sorted_by_key(|run| run.key) {
            kept.sort_by_key(|run| run.key);
        }
        if let Ok(head) = head {
            for run in &kept {
                let batch = &batches.listed[run.at as usize];
                let drawn = (vertices.clone(), run.instances.clone());
                let record = &batch.records[run.record as usize];
                let issued = self.issue((part, timer), head, record, drawn);
                self.settle_draw(part, issued, (batch.geometry, batch.material));
            }
        }
        kept.clear();
        self.spare.kept = kept;
    }

    fn view_stamp(&self) -> (u64, u32) {
        (self.frames.current, self.uniforms.camera_slot)
    }

    pub(crate) fn judge(
        &mut self,
        kind: engine_core::DrawKind,
        verdict: &mut dyn FnMut(BatchOf<'_>, u32) -> Option<u32>,
    ) {
        let stamp = self.view_stamp();
        let Self {
            installs,
            verdicts,
            part,
            ..
        } = self;
        let mut judged = false;
        let members = (installs.draw_entries.iter_mut())
            .filter(|member| member.part == *part && member.decl.kind == kind);
        for member in members {
            judged = true;
            member.batches.judged = Some(stamp);
            for batch in &member.batches.listed {
                verdicts.judge(batch, verdict);
            }
        }
        if !judged {
            let error = fault(self.path("judge.kind"), ErrorCode::UnknownVariant).got(kind.name());
            self.fault(error);
        }
    }

    fn settle_draw(
        &mut self,
        part: u32,
        issued: Result<crate::Recorded, DropReason>,
        handles: (Option<Asset<MeshFile>>, Option<Asset<Material>>),
    ) -> crate::Recorded {
        let recorded = match issued {
            Ok(recorded) => {
                self.tallies.kinds.reached();
                recorded
            }
            Err(reason) => {
                let asset = dropped_asset(reason, handles);
                self.tallies
                    .kinds
                    .drop_draws((part, self.pass), (reason, asset), 1);
                if reason == DropReason::PipelineRefused {
                    crate::Recorded::Refused
                } else {
                    crate::Recorded::Waiting
                }
            }
        };
        self.tallies.worst = self.tallies.worst.max(recorded);
        recorded
    }

    fn hold(&mut self, head: Result<Head, DropReason>, batch: &mut KindBatch) {
        let waiting = match head.and_then(|head| self.cached(head, batch)) {
            Ok(record) => !batch.records[record].exact && !self.pipelines.standing.is_empty(),
            Err(reason) => matches!(
                reason,
                DropReason::PipelineCompiling
                    | DropReason::GeometryNotResident
                    | DropReason::MaterialNotResident
            ),
        };
        self.tallies.held = self.tallies.held.map(|held| held || waiting);
        self.tallies.kinds.skip("held until its group is ready");
        self.tallies.worst = self.tallies.worst.max(crate::Recorded::Waiting);
    }

    fn head(&mut self, part: u32, point: Option<PassBit>, handle: u32) -> Result<Head, DropReason> {
        let (pipeline, filled) = self
            .pipeline_of(handle)
            .ok_or(DropReason::UnknownPipeline)?;
        if !self.scoped(part, point, pipeline) {
            let saved = self.part;
            self.part = part;
            let error = fault(self.path("draw.pipeline"), ErrorCode::PassMismatch);
            self.fault(error);
            self.part = saved;
            return Err(DropReason::PassMismatch);
        }
        if self.machine.pipelines[pipeline as usize].holes && filled.is_none() {
            return Err(DropReason::UnfilledHoles);
        }
        if filled.is_some_and(|filled| self.filled[filled].groups.is_empty()) {
            self.fill_groups(pipeline, filled);
        }
        let layout = self
            .recording
            .open()
            .ok_or(DropReason::NoRenderPass)?
            .layout;
        Ok(Head {
            pipeline,
            filled,
            layout,
        })
    }

    fn cached(&mut self, head: Head, batch: &mut KindBatch) -> Result<usize, DropReason> {
        let known = batch
            .records
            .iter()
            .position(|record| record.pipeline == head.pipeline && record.layout == head.layout);
        if let Some(at) = known {
            return Ok(at);
        }
        let record = self.build(head, (batch.geometry, batch.material))?;
        batch.records.push(record);
        Ok(batch.records.len() - 1)
    }

    fn resolve_of(&mut self, pipeline: u32, key: Option<Asset<Material>>) -> &mut Resolve {
        let Self {
            machine, assets, ..
        } = self;
        let built = &mut machine.pipelines[pipeline as usize];
        match assets.batches.resolve_mut(&built.batches, pipeline, key) {
            Some(resolve) => resolve,
            None => &mut built.plain,
        }
    }

    fn landed(&mut self, head: Head, key: Option<Asset<Material>>) -> Result<Landed, DropReason> {
        let mut resolve = std::mem::take(self.resolve_of(head.pipeline, key));
        let found = match resolve
            .landed
            .iter()
            .position(|landed| landed.layout == head.layout)
        {
            Some(at) => Ok(at),
            None => self.resolve_layout(&mut resolve, head.layout),
        };
        let landed = found.map(|at| resolve.landed[at].clone());
        *self.resolve_of(head.pipeline, key) = resolve;
        landed
    }

    fn build(
        &mut self,
        head: Head,
        (mesh, key): (Option<Asset<MeshFile>>, Option<Asset<Material>>),
    ) -> Result<DrawRecord, DropReason> {
        if mesh.is_some_and(|mesh| !self.assets.meshes.contains(mesh)) {
            return Err(DropReason::GeometryNotResident);
        }
        if key.is_some()
            && !self.machine.declared.batch_groups.is_empty()
            && !self.assets.batches.holds(key)
        {
            return Err(DropReason::MaterialNotResident);
        }
        let landed = self.landed(head, key)?;
        let groups = self.machine.pipelines[head.pipeline as usize]
            .batches
            .iter()
            .map(|&(index, batch)| Some((index, self.assets.batches.group(batch, key)?.clone())))
            .collect::<Option<Box<[_]>>>()
            .ok_or(DropReason::MaterialNotResident)?;
        let mesh = match mesh {
            Some(mesh) => {
                let held = (&self.assets, &mut self.layouts, &mut self.installs.faults);
                Some(mesh_record(held, (&self.machine, self.pass), (head.pipeline, mesh))?)
            }
            None => None,
        };
        Ok(DrawRecord {
            pipeline: head.pipeline,
            layout: head.layout,
            rank: rank_of(&landed.made),
            made: landed.made,
            exact: landed.exact,
            groups,
            mesh,
        })
    }

    fn issue(
        &mut self,
        (part, timer): (u32, &GpuProfiler),
        head: Head,
        record: &DrawRecord,
        (vertices, instances): (Range<u32>, Range<u32>),
    ) -> Result<crate::Recorded, DropReason> {
        if self.recording.collecting() {
            return Ok(crate::Recorded::Waiting);
        }
        let at = (self.pass, part);
        let Self {
            machine,
            recording,
            bindings:
                Bindings {
                    holders,
                    frame_groups,
                    own_groups,
                    ..
                },
            filled: fills,
            sets,
            frames,
            bound,
            tallies,
            ..
        } = self;
        let parity = frames.parity;
        let set = &sets.list[sets.at];
        let open = recording.open_mut().ok_or(DropReason::NoRenderPass)?;
        let built = &machine.pipelines[head.pipeline as usize];
        let render = &mut open.pass;
        let frame = frame_groups
            .get(built.variant as usize)
            .and_then(|groups| held_group(groups, set, built.variant, parity))
            .ok_or(DropReason::MissingBindGroup)?;
        let camera = machine
            .variants
            .get(built.variant as usize)
            .is_some_and(|found| found.camera);
        bound.draw_pipeline(render, &record.made);
        bound.draw_group(render, 0, frame, camera.then_some(open.offset));
        for (own, group) in built.groups.iter().enumerate() {
            let filled_group = head.filled.and_then(|filled| {
                fills[filled]
                    .groups
                    .iter()
                    .find(|(index, _)| *index == group.index)
                    .map(|(_, made)| made)
            });
            let holder = holders.index(Holder::Own(head.pipeline, own as u32));
            let owned = held_group(&own_groups[head.pipeline as usize][own], set, holder, parity);
            let made = filled_group.or(owned).ok_or(DropReason::MissingBindGroup)?;
            bound.draw_group(render, group.index, made, None);
        }
        for (index, group) in &record.groups {
            bound.draw_group(render, *index, group, None);
        }
        let counted = instances.end.saturating_sub(instances.start);
        let timed = (timer, &*machine, (at.0, head.pipeline));
        match &record.mesh {
            Some(mesh) => {
                for (slot, buffer) in mesh.buffers.iter().enumerate() {
                    bound.vertex_buffer(render, slot as u32, buffer);
                }
                bound.index_buffer(render, &mesh.indices, mesh.format);
                timed_draw(render, timed, |render| {
                    render.draw_indexed(mesh.range.clone(), mesh.base_vertex, instances);
                });
            }
            None => timed_draw(render, timed, |render| render.draw(vertices, instances)),
        }
        tallies.drawn.add(at, counted);
        Ok(if record.exact {
            crate::Recorded::Exact
        } else {
            crate::Recorded::StandIn
        })
    }

    pub(crate) fn forget_records(&mut self, key: Option<Asset<Material>>) {
        for member in &mut self.installs.draw_entries {
            member.batches.forget(key);
        }
    }

    pub(crate) fn settle_records(&mut self, seen: &mut Option<(u64, u64)>, revision: u64) {
        let now = (self.arrivals(), revision);
        if *seen == Some(now) {
            return;
        }
        if seen.is_none_or(|(arrivals, _)| arrivals != now.0) {
            self.refresh_held();
        }
        for member in &mut self.installs.draw_entries {
            member.batches.forget_all();
        }
        *seen = Some(now);
    }

    pub(super) fn fill_groups(&mut self, pipeline: u32, filled: Option<usize>) {
        let Some(filled) = filled else { return };
        let textures = self.filled[filled].textures.clone();
        let groups = self.machine.pipelines[pipeline as usize].groups.len();
        let mut made = Vec::with_capacity(groups);
        for group in 0..groups {
            let entries = self.machine.pipelines[pipeline as usize].groups[group]
                .entries
                .clone();
            let index = self.machine.pipelines[pipeline as usize].groups[group].index;
            let part = self.machine.pipelines[pipeline as usize].decl.part;
            let mut hole = 0;
            let mut resolved = Vec::with_capacity(entries.len());
            for (binding, bound, ty) in entries {
                let found = if bound == Bound::Asset {
                    let texture = textures.get(hole).copied();
                    hole += 1;
                    match texture.and_then(|texture| self.assets.textures.texture_view(texture)) {
                        Some(view) => Resolved::View(view.clone()),
                        None => self.reported_stand_in(Some(part), ty, (index, binding)),
                    }
                } else {
                    self.resource((&bound, ty), (Some(part), (index, binding)))
                };
                resolved.push((binding, found));
            }
            let layout = self.machine.pipelines[pipeline as usize].groups[group]
                .layout
                .clone();
            if let Some(group) = self.bind_group(&layout, resolved) {
                made.push((index, group));
            }
        }
        self.filled[filled].groups = made;
    }

    pub(crate) fn dispatch(&mut self, part: u32, handle: u32, groups: [u32; 3]) {
        let recorded = self.issue_dispatch(part, handle, groups);
        self.tallies.worst = self.tallies.worst.max(recorded);
    }

    fn issue_dispatch(&mut self, part: u32, handle: u32, [x, y, z]: [u32; 3]) -> crate::Recorded {
        self.end_open();
        let Some(built) = self.machine.pipelines.get(handle as usize) else {
            return crate::Recorded::Waiting;
        };
        if built.decl.part != part || built.decl.scope != Scoped::Pass(self.pass) {
            let error = fault(self.path("dispatch.pipeline"), ErrorCode::PassMismatch);
            self.fault(error);
            return crate::Recorded::Refused;
        }
        if built.compute_ready.is_none() {
            let made = match self.compute_pipeline(handle) {
                Ok(made) => made,
                Err(reason) => {
                    let at = (part, self.pass);
                    self.tallies.kinds.drop_draws(at, (reason, None), 1);
                    return if reason == DropReason::PipelineRefused {
                        crate::Recorded::Refused
                    } else {
                        crate::Recorded::Waiting
                    };
                }
            };
            self.machine.pipelines[handle as usize].compute_ready = Some(made);
        }
        let current = self.pass;
        if self.recording.computing(current).is_none() {
            self.end_render();
            self.group(current);
            let label = self.label_of(current).unwrap_or("engine/part-dispatch");
            let Some(encoder) = self.recording.encoder() else {
                return crate::Recorded::Waiting;
            };
            let begun = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                })
                .forget_lifetime();
            self.recording
                .transition(Step::Compute(current, Box::new(begun)));
            self.bound.forget();
        }
        let offset = self.uniforms.camera_slot * CAMERA_STRIDE as u32;
        let Self {
            machine,
            recording,
            bindings:
                Bindings {
                    holders,
                    frame_groups,
                    own_groups,
                    ..
                },
            sets,
            frames,
            bound,
            ..
        } = self;
        let parity = frames.parity;
        let set = &sets.list[sets.at];
        let Some(pass) = recording.computing(current) else {
            return crate::Recorded::Waiting;
        };
        let built = &machine.pipelines[handle as usize];
        let Some(compute) = built.compute_ready.as_ref() else {
            return crate::Recorded::Waiting;
        };
        bound.dispatch_pipeline(pass, compute);
        if let Some(group) = frame_groups
            .get(built.variant as usize)
            .and_then(|groups| held_group(groups, set, built.variant, parity))
        {
            let camera = machine
                .variants
                .get(built.variant as usize)
                .is_some_and(|found| found.camera);
            bound.dispatch_group(pass, 0, group, camera.then_some(offset));
        }
        for (at, group) in built.groups.iter().enumerate() {
            let holder = holders.index(Holder::Own(handle, at as u32));
            if let Some(made) = held_group(&own_groups[handle as usize][at], set, holder, parity) {
                bound.dispatch_group(pass, group.index, made, None);
            }
        }
        pass.dispatch_workgroups(x, y, z);
        crate::Recorded::Exact
    }
}

fn timed_draw(
    render: &mut wgpu::RenderPass<'static>,
    (timer, machine, (pass, pipeline)): (&GpuProfiler, &Machine, (u32, u32)),
    draw: impl FnOnce(&mut wgpu::RenderPass<'static>),
) {
    let began = timer.begin_draw(render, || machine.draw_key(pass, pipeline));
    draw(render);
    timer.end_draw(render, began);
}
