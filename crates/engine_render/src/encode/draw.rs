use engine_core::{Asset, DropReason, GeometryStream};

use super::*;
use crate::mesh_arena::MeshDraw;

fn instances_of(shape: &RawShape) -> Option<u32> {
    match shape {
        RawShape::Pulled { instances, .. }
        | RawShape::Buffers { instances, .. }
        | RawShape::Mesh { instances, .. } => Some(instances.end.saturating_sub(instances.start)),
        RawShape::Indirect { .. } | RawShape::IndirectAt { .. } | RawShape::MeshIndirect { .. } => {
            None
        }
    }
}

fn mesh_draw<'a>(
    assets: &'a Assets,
    mesh: GeometryHandle,
    geometry: &[GeometryStream],
) -> Result<MeshDraw<'a>, DropReason> {
    let draw = assets
        .meshes
        .draw(mesh)
        .ok_or(DropReason::GeometryNotResident)?;
    if draw.holds(geometry) {
        Ok(draw)
    } else {
        Err(DropReason::GeometryLacksStreams)
    }
}

fn handles_of(shape: &RawShape) -> (Option<GeometryHandle>, Option<MaterialHandle>) {
    match shape {
        RawShape::MeshIndirect { mesh, key, .. } | RawShape::Mesh { mesh, key, .. } => {
            (Some(*mesh), *key)
        }
        RawShape::Pulled { key, .. } | RawShape::Indirect { key, .. } => (None, *key),
        RawShape::Buffers { .. } | RawShape::IndirectAt { .. } => (None, None),
    }
}

fn dropped_asset(
    reason: DropReason,
    (mesh, key): (Option<GeometryHandle>, Option<MaterialHandle>),
) -> Option<Asset> {
    if reason == DropReason::GeometryNotResident {
        mesh.map(Asset::Mesh)
    } else if reason == DropReason::MaterialNotResident {
        key.map(Asset::Material)
    } else {
        None
    }
}

fn bound_input<'b>(
    input: impl Fn(u32) -> Option<&'b wgpu::Buffer>,
    stream: u32,
) -> Result<&'b wgpu::Buffer, DropReason> {
    input(stream).ok_or(DropReason::MissingInput)
}

fn indirect_args<'b>(
    (machine, arena): (&Machine, &'b Arena),
    input: impl Fn(u32) -> Option<&'b wgpu::Buffer>,
    (args, offset, indexed): (u32, u64, bool),
) -> Result<(&'b wgpu::Buffer, u64), DropReason> {
    let direct = bound_input(input, args)?;
    let size = if indexed { INDEXED_ARGS } else { PLAIN_ARGS };
    let laid = machine
        .declared
        .inputs
        .get(args as usize)
        .and_then(|decl| arena.args(decl.buffer, (offset, size)));
    Ok(laid.unwrap_or((direct, offset)))
}

const INDEXED_ARGS: u64 = 20;
const PLAIN_ARGS: u64 = 16;

fn index_input<'b>(
    input: impl Fn(u32) -> Option<&'b wgpu::Buffer>,
    indices: Option<u32>,
) -> Result<Option<&'b wgpu::Buffer>, DropReason> {
    indices.map(|stream| bound_input(input, stream)).transpose()
}

fn input_of<'b>(
    machine: &Machine,
    (shared, set): (&'b views::Tables, &'b views::ViewSet),
    part: u32,
    input: u32,
) -> Option<&'b wgpu::Buffer> {
    let decl = machine.declared.inputs.get(input as usize)?;
    if decl.part != part {
        return None;
    }
    let tables = if machine.declared.buffers.get(decl.buffer as usize)?.per_view {
        &set.tables
    } else {
        shared
    };
    tables
        .buffers
        .get(decl.buffer as usize)?
        .as_ref()
        .map(|(buffer, _)| buffer)
}

fn held_group<'g>(
    groups: &'g Groups,
    set: &'g views::ViewSet,
    holder: Holder,
    parity: usize,
) -> Option<&'g wgpu::BindGroup> {
    set.groups
        .get(&holder)
        .map_or(groups.made[parity].as_ref(), |made| made[parity].as_ref())
}

fn offsets_of<'o>(
    windows: usize,
    given: &mut &[u32],
    held: &'o mut [u32; MOST_OFFSETS],
) -> &'o [u32] {
    let count = windows.min(MOST_OFFSETS);
    for slot in &mut held[..count] {
        let (first, rest) = given
            .split_first()
            .map_or((0, &[][..]), |(first, rest)| (*first, rest));
        *slot = first;
        *given = rest;
    }
    &held[..count]
}

impl Run {
    #[cfg(any(feature = "dev", feature = "profiling"))]
    pub(crate) fn first_drawn(&self, handle: u32, cause: u32) {
        let kinds = &self.tallies.kinds;
        let kind = kinds
            .drawing
            .and_then(|at| kinds.listed.get(at))
            .map_or("", |tally| tally.kind.name());
        let pipeline = self.pipeline_of(handle).map_or("", |(pipeline, _)| {
            self.machine.draw_key(self.pass, pipeline).pipeline
        });
        engine_profile::caused(engine_profile::Cause(cause), || {
            engine_profile::event(
                "drawn",
                kind,
                &[("pipeline", engine_profile::Fact::Text(pipeline))],
            );
        });
    }

    pub(super) fn scoped(&self, part: u32, point: Option<&'static str>, pipeline: u32) -> bool {
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

    pub(crate) fn speedup_made(
        &mut self,
        part: u32,
        point: Option<&'static str>,
        handle: u32,
    ) -> bool {
        let Some((pipeline, None)) = self.pipeline_of(handle) else {
            return true;
        };
        let Some(open) = self.recording.open().filter(|_| self.scoped(part, point, pipeline))
        else {
            return true;
        };
        let layout = open.layout;
        let Some(built) = self.machine.pipelines.get(pipeline as usize) else {
            return true;
        };
        if built
            .ready
            .iter()
            .any(|ready| ready.layout == layout && ready.variant.is_none())
        {
            return true;
        }
        let targets = (layout.formats(), layout.depth);
        let shown = (layout.plain, layout.main);
        self.split()
            .0
            .speedup_unmade(pipeline, targets, shown, None)
            .is_ok()
    }

    pub(crate) fn draw(
        &mut self,
        part: u32,
        point: Option<&'static str>,
        handle: u32,
        shape: RawShape,
        timer: &GpuProfiler,
    ) -> crate::Recorded {
        let handles = handles_of(&shape);
        let recorded = match self.issue_draw(part, point, handle, shape, timer) {
            Ok(recorded) => {
                self.tallies.kinds.reached();
                recorded
            }
            Err(reason) => {
                let pass = self
                    .machine
                    .declared
                    .passes
                    .get(self.pass as usize)
                    .map_or("", |declared| declared.name);
                let asset = dropped_asset(reason, handles);
                self.tallies
                    .kinds
                    .drop_draws((part, pass), (reason, asset), 1);
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

    fn batch_variant(
        &self,
        (mesh, key): (Option<GeometryHandle>, Option<MaterialHandle>),
    ) -> Result<Option<std::sync::Arc<crate::Variant>>, DropReason> {
        if mesh.is_some_and(|mesh| self.assets.meshes.get(mesh).is_none()) {
            return Err(DropReason::GeometryNotResident);
        }
        match self.assets.batches.held(key) {
            Some(variant) => Ok(variant),
            None if key.is_some() && !self.machine.declared.batch_groups.is_empty() => {
                Err(DropReason::MaterialNotResident)
            }
            None => Ok(None),
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn issue_draw(
        &mut self,
        part: u32,
        point: Option<&'static str>,
        handle: u32,
        shape: RawShape,
        timer: &GpuProfiler,
    ) -> Result<crate::Recorded, DropReason> {
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
        let (mesh, key) = handles_of(&shape);
        let variant = self.batch_variant((mesh, key))?;
        let ready = self.render_pipeline(pipeline, variant.as_deref())?;
        if self.recording.collecting() {
            return Ok(crate::Recorded::Waiting);
        }
        let at = (self.pass, part);
        let counted = instances_of(&shape);
        let Self {
            machine,
            recording,
            bindings:
                Bindings {
                    frame_groups,
                    own_groups,
                    ..
                },
            filled: fills,
            tables,
            sets,
            assets,
            frames,
            bound,
            arena,
            tallies,
            ..
        } = self;
        let parity = frames.parity;
        let set = &sets.list[sets.at];
        let open = recording.open_mut().ok_or(DropReason::NoRenderPass)?;
        let timed = (timer, &*machine, (at.0, pipeline));
        let built = &machine.pipelines[pipeline as usize];
        let chosen = &built.ready[ready];
        let recorded = if chosen.exact {
            crate::Recorded::Exact
        } else {
            crate::Recorded::StandIn
        };
        let render = &mut open.pass;
        let frame = frame_groups
            .get(built.variant as usize)
            .and_then(|groups| held_group(groups, set, Holder::Frame(built.variant), parity))
            .ok_or(DropReason::MissingBindGroup)?;
        let camera = usize::from(
            machine
                .variants
                .get(built.variant as usize)
                .is_some_and(|found| found.camera),
        );
        bound.draw_pipeline(render, &chosen.made);
        bound.draw_group(render, 0, frame, &[open.offset][..camera]);
        let held = [0_u32; MOST_OFFSETS];
        for (at, group) in built.groups.iter().enumerate() {
            let dynamic = &held[..group.windows.min(MOST_OFFSETS)];
            let filled_group = filled.and_then(|filled| {
                fills[filled]
                    .groups
                    .iter()
                    .find(|(index, _)| *index == group.index)
                    .map(|(_, made)| made)
            });
            let owned = held_group(
                &own_groups[pipeline as usize][at],
                set,
                Holder::Own(pipeline, at as u32),
                parity,
            );
            let made = filled_group.or(owned).ok_or(DropReason::MissingBindGroup)?;
            bound.draw_group(render, group.index, made, dynamic);
        }
        for &(index, batch) in &built.batches {
            let group = assets
                .batches
                .group(batch, key)
                .ok_or(DropReason::MaterialNotResident)?;
            bound.draw_group(render, index, group, &[]);
        }
        let input = |input: u32| input_of(machine, (&*tables, set), part, input);
        let geometry = &built.geometry;
        let geometry_slots = geometry.len() as u32;
        match shape {
            RawShape::Pulled {
                vertices,
                instances,
                ..
            } => timed_draw(render, timed, |render| render.draw(vertices, instances)),
            RawShape::Buffers {
                vertices,
                indices,
                elements,
                instances,
            } => {
                let indexed = index_input(input, indices)?;
                for (slot, (stream, _)) in vertices.listed().iter().enumerate() {
                    let buffer = bound_input(input, *stream)?;
                    bound.vertex_buffer(render, slot as u32, buffer, 0);
                }
                match indexed {
                    Some(indices) => {
                        bound.index_buffer(render, indices);
                        timed_draw(render, timed, |render| {
                            render.draw_indexed(elements, 0, instances)
                        });
                    }
                    None => timed_draw(render, timed, |render| render.draw(elements, instances)),
                }
            }
            RawShape::Indirect {
                vertices,
                indices,
                args,
                offset,
                ..
            } => {
                let (args, offset) =
                    indirect_args((machine, arena), input, (args, offset, indices.is_some()))?;
                let indexed = index_input(input, indices)?;
                for (slot, (stream, _)) in vertices.listed().iter().enumerate() {
                    let buffer = bound_input(input, *stream)?;
                    bound.vertex_buffer(render, slot as u32, buffer, 0);
                }
                match indexed {
                    Some(indices) => {
                        bound.index_buffer(render, indices);
                        timed_draw(render, timed, |render| {
                            render.draw_indexed_indirect(args, offset)
                        });
                    }
                    None => timed_draw(render, timed, |render| render.draw_indirect(args, offset)),
                }
            }
            RawShape::IndirectAt {
                vertices,
                indices,
                args,
                offset,
            } => {
                let (args, offset) =
                    indirect_args((machine, arena), input, (args, offset, indices.is_some()))?;
                let indexed = index_input(input, indices)?;
                for (slot, (stream, start)) in vertices.listed().iter().enumerate() {
                    let buffer = bound_input(input, *stream)?;
                    bound.vertex_buffer(render, slot as u32, buffer, *start);
                }
                match indexed {
                    Some(indices) => {
                        bound.index_buffer(render, indices);
                        timed_draw(render, timed, |render| {
                            render.draw_indexed_indirect(args, offset)
                        });
                    }
                    None => timed_draw(render, timed, |render| render.draw_indirect(args, offset)),
                }
            }
            RawShape::MeshIndirect {
                mesh,
                args,
                offset,
                streams,
                ..
            } => {
                let draw = mesh_draw(assets, mesh, geometry)?;
                let (args, offset) = indirect_args((machine, arena), input, (args, offset, true))?;
                let buffers = geometry.iter().filter_map(|&stream| draw.stream(stream));
                for (slot, buffer) in buffers.enumerate() {
                    bound.vertex_buffer(render, slot as u32, buffer, 0);
                }
                for (at, (stream, start)) in streams.listed().iter().enumerate() {
                    let buffer = bound_input(input, *stream)?;
                    let slot = geometry_slots + at as u32;
                    bound.vertex_buffer(render, slot, buffer, *start);
                }
                bound.index_buffer(render, draw.indices);
                timed_draw(render, timed, |render| {
                    render.draw_indexed_indirect(args, offset)
                });
            }
            RawShape::Mesh { mesh, instances, .. } => {
                let draw = mesh_draw(assets, mesh, geometry)?;
                let buffers = geometry.iter().filter_map(|&stream| draw.stream(stream));
                for (slot, buffer) in buffers.enumerate() {
                    bound.vertex_buffer(render, slot as u32, buffer, 0);
                }
                bound.index_buffer(render, draw.indices);
                let indices = draw.first_index..draw.first_index + draw.index_count;
                timed_draw(render, timed, |render| {
                    render.draw_indexed(indices, draw.base_vertex, instances)
                });
            }
        }
        tallies.drawn.add(at, counted);
        Ok(recorded)
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
            let mut hole = 0;
            let mut resolved = Vec::with_capacity(entries.len());
            for (binding, bound, ty) in entries {
                let found = if bound == Bound::Asset {
                    let texture = textures.get(hole).copied();
                    hole += 1;
                    match texture.and_then(|texture| self.assets.textures.texture_view(texture)) {
                        Some(view) => Resolved::View(view.clone()),
                        None => Resolved::Placeholder(ty),
                    }
                } else {
                    self.resource(&bound, ty)
                };
                resolved.push((binding, found));
            }
            let layout = self.machine.pipelines[pipeline as usize].groups[group]
                .layout
                .clone();
            made.push((
                index,
                self.bind_group(&layout, (Some(pipeline as usize), index), resolved),
            ));
        }
        self.filled[filled].groups = made;
    }

    pub(crate) fn scissor(&mut self, rect: [u32; 4]) {
        if let Some(open) = self.recording.open_mut() {
            self.bound.scissor(&mut open.pass, rect);
        }
    }

    pub(crate) fn within(&mut self, size: (u32, u32)) {
        if let Some(open) = self.recording.open_mut() {
            self.bound.within(&mut open.pass, size);
        }
    }

    pub(crate) fn reset_scissor(&mut self) {
        let size = self
            .set()
            .targets
            .first()
            .map(|(_, made)| (made.size.width, made.size.height));
        if let (Some(open), Some((width, height))) = (
            self.recording.open_mut().filter(|open| open.layered),
            size,
        ) {
            self.bound.scissor(&mut open.pass, [0, 0, width, height]);
        }
    }

    pub(crate) fn dispatch(
        &mut self,
        part: u32,
        handle: u32,
        count: CountRaw,
        offsets: &[u32],
    ) {
        self.end_open();
        let Some(built) = self.machine.pipelines.get(handle as usize) else {
            return;
        };
        if built.decl.part != part || built.decl.scope != Scoped::Pass(self.pass) {
            let error = fault(self.path("dispatch.pipeline"), ErrorCode::PassMismatch);
            self.fault(error);
            return;
        }
        if built.compute_ready.is_none() {
            let Some(made) = self.compute_pipeline(handle, true) else {
                return;
            };
            self.machine.pipelines[handle as usize].compute_ready = Some(made);
        }
        if let CountRaw::Indirect(input, _) = count
            && input_of(&self.machine, (&self.tables, self.set()), part, input).is_none()
        {
            return;
        }
        let current = self.pass;
        if self.recording.computing(current).is_none() {
            self.end_render();
            self.group(current);
            let label = self.label_of(current).unwrap_or("engine/part-dispatch");
            let Some(encoder) = self.recording.encoder() else {
                return;
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
                    frame_groups,
                    own_groups,
                    ..
                },
            tables,
            sets,
            frames,
            bound,
            ..
        } = self;
        let parity = frames.parity;
        let set = &sets.list[sets.at];
        let Some(pass) = recording.computing(current) else {
            return;
        };
        let built = &machine.pipelines[handle as usize];
        let Some(compute) = built.compute_ready.as_ref() else {
            return;
        };
        bound.dispatch_pipeline(pass, compute);
        if let Some(group) = frame_groups
            .get(built.variant as usize)
            .and_then(|groups| held_group(groups, set, Holder::Frame(built.variant), parity))
        {
            let camera = usize::from(
                machine
                    .variants
                    .get(built.variant as usize)
                    .is_some_and(|found| found.camera),
            );
            bound.dispatch_group(pass, 0, group, &[offset][..camera]);
        }
        let mut given = offsets;
        let mut held = [0_u32; MOST_OFFSETS];
        for (at, group) in built.groups.iter().enumerate() {
            let dynamic = offsets_of(group.windows, &mut given, &mut held);
            let holder = Holder::Own(handle, at as u32);
            if let Some(made) = held_group(&own_groups[handle as usize][at], set, holder, parity) {
                bound.dispatch_group(pass, group.index, made, dynamic);
            }
        }
        match count {
            CountRaw::Indirect(input, offset) => {
                if let Some(buffer) = input_of(machine, (&*tables, set), part, input) {
                    pass.dispatch_workgroups_indirect(buffer, offset);
                }
            }
            CountRaw::Groups([x, y, z]) => pass.dispatch_workgroups(x, y, z),
        }
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
