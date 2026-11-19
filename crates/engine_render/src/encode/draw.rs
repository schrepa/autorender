use engine_core::{Asset, DropReason, GeometryStream};

use super::*;
use crate::Shape;
use crate::mesh_arena::MeshDraw;

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

fn handles_of(shape: &Shape) -> (Option<GeometryHandle>, Option<MaterialHandle>) {
    match shape {
        Shape::Mesh { mesh, key, .. } => (Some(*mesh), *key),
        Shape::Pulled { key, .. } => (None, *key),
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

fn held_group<'g>(
    groups: &'g Groups,
    set: &'g views::ViewSet,
    holder: u32,
    parity: usize,
) -> Option<&'g wgpu::BindGroup> {
    set.group(holder)
        .map_or(groups.made[parity].as_ref(), |made| made[parity].as_ref())
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
        let recorded = match self.issue_draw(part, point, handle, shape, timer) {
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

    fn batch_variant(
        &self,
        (mesh, key): (Option<GeometryHandle>, Option<MaterialHandle>),
    ) -> Result<Option<resources::Drawn>, DropReason> {
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

    fn issue_draw(
        &mut self,
        part: u32,
        point: Option<PassBit>,
        handle: u32,
        shape: Shape,
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
        let drawn = variant
            .as_ref()
            .map(|(variant, identity)| (&**variant, *identity));
        let ready = self.render_pipeline(pipeline, drawn)?;
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
            assets,
            frames,
            bound,
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
            .and_then(|groups| held_group(groups, set, built.variant, parity))
            .ok_or(DropReason::MissingBindGroup)?;
        let camera = machine
            .variants
            .get(built.variant as usize)
            .is_some_and(|found| found.camera);
        bound.draw_pipeline(render, &chosen.made);
        bound.draw_group(render, 0, frame, camera.then_some(open.offset));
        for (at, group) in built.groups.iter().enumerate() {
            let filled_group = filled.and_then(|filled| {
                fills[filled]
                    .groups
                    .iter()
                    .find(|(index, _)| *index == group.index)
                    .map(|(_, made)| made)
            });
            let holder = holders.index(Holder::Own(pipeline, at as u32));
            let owned = held_group(&own_groups[pipeline as usize][at], set, holder, parity);
            let made = filled_group.or(owned).ok_or(DropReason::MissingBindGroup)?;
            bound.draw_group(render, group.index, made, None);
        }
        for &(index, batch) in &built.batches {
            let group = assets
                .batches
                .group(batch, key)
                .ok_or(DropReason::MaterialNotResident)?;
            bound.draw_group(render, index, group, None);
        }
        let geometry = &built.geometry;
        let (Shape::Pulled { instances, .. } | Shape::Mesh { instances, .. }) = &shape;
        let counted = instances.end.saturating_sub(instances.start);
        match shape {
            Shape::Pulled {
                vertices,
                instances,
                ..
            } => timed_draw(render, timed, |render| render.draw(vertices, instances)),
            Shape::Mesh { mesh, instances, .. } => {
                let draw = mesh_draw(assets, mesh, geometry)?;
                let buffers = geometry.iter().filter_map(|&stream| draw.stream(stream));
                for (slot, buffer) in buffers.enumerate() {
                    bound.vertex_buffer(render, slot as u32, buffer);
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
            let Some(made) = self.compute_pipeline(handle, true) else {
                return crate::Recorded::Waiting;
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
