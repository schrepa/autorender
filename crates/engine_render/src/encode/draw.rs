use engine_core::{Asset, DropReason, GeometryStream};

use super::*;
use crate::mesh_arena::MeshDraw;

type Painted = (
    Option<crate::surface::MaterialVariant>,
    Option<Option<MaterialHandle>>,
);

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
        RawShape::MeshIndirect { mesh, material, .. } | RawShape::Mesh { mesh, material, .. } => {
            (Some(*mesh), *material)
        }
        RawShape::Pulled { material, .. } | RawShape::Indirect { material, .. } => {
            (None, *material)
        }
        RawShape::Buffers { .. } | RawShape::IndirectAt { .. } => (None, None),
    }
}

fn dropped_asset(
    reason: DropReason,
    (mesh, material): (Option<GeometryHandle>, Option<MaterialHandle>),
) -> Option<Asset> {
    if reason == DropReason::GeometryNotResident {
        mesh.map(Asset::Mesh)
    } else if reason == DropReason::MaterialNotResident {
        material.map(Asset::Material)
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

    pub(crate) fn draw(
        &mut self,
        part: u32,
        point: Option<&'static str>,
        handle: u32,
        shape: RawShape,
        timed: TimedDraw<'_>,
    ) {
        let _row = engine_profile::timed(engine_profile::row("render.draw"));
        let handles = handles_of(&shape);
        match self.issue_draw(part, point, handle, shape, timed) {
            Ok(()) => self.tallies.kinds.reached(),
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
            }
        }
    }

    fn drawn_variant(&self, shape: &RawShape) -> Result<Painted, DropReason> {
        match shape {
            RawShape::MeshIndirect { mesh, material, .. }
            | RawShape::Mesh { mesh, material, .. } => {
                if self.assets.meshes.get(*mesh).is_none() {
                    return Err(DropReason::GeometryNotResident);
                }
                let variant = self
                    .assets
                    .materials
                    .variant(*material)
                    .ok_or(DropReason::MaterialNotResident)?;
                Ok((Some(variant), None))
            }
            RawShape::Pulled { material, .. } | RawShape::Indirect { material, .. } => {
                match material {
                    Some(handle) => {
                        let variant = self
                            .assets
                            .materials
                            .variant(Some(*handle))
                            .ok_or(DropReason::MaterialNotResident)?;
                        Ok((Some(variant), Some(*material)))
                    }
                    None => Ok((None, Some(None))),
                }
            }
            RawShape::Buffers { .. } | RawShape::IndirectAt { .. } => Ok((None, Some(None))),
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
        timed: TimedDraw<'_>,
    ) -> Result<(), DropReason> {
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
        let (variant, painted) = self.drawn_variant(&shape)?;
        let ready = self.render_pipeline(pipeline, variant)?;
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
            ..
        } = self;
        let parity = frames.parity;
        let set = &sets.list[sets.at];
        let open = recording.open_mut().ok_or(DropReason::NoRenderPass)?;
        let built = &machine.pipelines[pipeline as usize];
        let chosen = &built.ready[ready];
        let render = &mut open.pass;
        if open.set.pipeline.is_none() {
            bound.vertex.clear();
            bound.index = None;
        }
        if open.set.pipeline != Some((pipeline, chosen.key)) {
            render.set_pipeline(&chosen.made);
            open.set.pipeline = Some((pipeline, chosen.key));
            open.set.frame = None;
        }
        if open.set.frame != Some((built.variant, open.offset)) {
            let group = frame_groups
                .get(built.variant as usize)
                .and_then(|groups| held_group(groups, set, Holder::Frame(built.variant), parity))
                .ok_or(DropReason::MissingBindGroup)?;
            let camera = usize::from(
                machine
                    .variants
                    .get(built.variant as usize)
                    .is_some_and(|found| found.camera),
            );
            render.set_bind_group(0, group, &[open.offset][..camera]);
            open.set.frame = Some((built.variant, open.offset));
        }
        let held = [0_u32; MOST_OFFSETS];
        for (at, group) in built.groups.iter().enumerate() {
            let dynamic = &held[..group.windows.min(MOST_OFFSETS)];
            let holding = (pipeline, filled, at);
            let slot = group.index as usize;
            if group.windows == 0 && open.set.own.get(slot).copied().flatten() == Some(holding) {
                continue;
            }
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
            render.set_bind_group(group.index, made, dynamic);
            if let Some(known) = open.set.own.get_mut(slot) {
                *known = Some(holding);
            }
            if group.index == 1 {
                open.set.material = None;
            }
        }
        if let Some(material) = painted
            && built.material
        {
            let group = assets
                .materials
                .group(material)
                .ok_or(DropReason::MaterialNotResident)?;
            if open.set.material != Some(material) {
                render.set_bind_group(1, group, &[]);
                open.set.material = Some(material);
                if let Some(known) = open.set.own.get_mut(1) {
                    *known = None;
                }
            }
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
                let args = bound_input(input, args)?;
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
                let args = bound_input(input, args)?;
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
                material,
                args,
                offset,
                streams,
            } => {
                let draw = mesh_draw(assets, mesh, geometry)?;
                let group = assets
                    .materials
                    .group(material)
                    .ok_or(DropReason::MaterialNotResident)?;
                let args = bound_input(input, args)?;
                if open.set.material != Some(material) {
                    render.set_bind_group(1, group, &[]);
                    open.set.material = Some(material);
                    open.set.own[1] = None;
                }
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
            RawShape::Mesh {
                mesh,
                material,
                instances,
            } => {
                let draw = mesh_draw(assets, mesh, geometry)?;
                let group = assets
                    .materials
                    .group(material)
                    .ok_or(DropReason::MaterialNotResident)?;
                if open.set.material != Some(material) {
                    render.set_bind_group(1, group, &[]);
                    open.set.material = Some(material);
                    open.set.own[1] = None;
                }
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
        Ok(())
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
                    match texture.and_then(|texture| self.assets.materials.texture_view(texture)) {
                        Some(view) => Resolved::View(view.clone()),
                        None => Resolved::Placeholder(ty),
                    }
                } else {
                    self.resource(&bound, ty, !ty.writes())
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
            open.pass
                .set_scissor_rect(rect[0], rect[1], rect[2], rect[3]);
        }
    }

    pub(crate) fn view_rect(&mut self, [x, y, width, height]: [u32; 4]) {
        if let Some(open) = self.recording.open_mut() {
            open.pass
                .set_viewport(x as f32, y as f32, width as f32, height as f32, 0.0, 1.0);
            open.pass.set_scissor_rect(x, y, width, height);
        }
    }

    pub(crate) fn within(&mut self, size: (u32, u32)) {
        if let Some(open) = self.recording.open_mut() {
            crate::targets::draw_within(&mut open.pass, size);
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
            self.region.or(size),
        ) {
            open.pass.set_scissor_rect(0, 0, width, height);
        }
    }

    pub(crate) fn dispatch(
        &mut self,
        part: u32,
        handle: u32,
        count: CountRaw,
        offsets: &[u32],
    ) -> u32 {
        self.end_open();
        let Some(built) = self.machine.pipelines.get(handle as usize) else {
            return 0;
        };
        if built.decl.part != part || built.decl.scope != Scoped::Pass(self.pass) {
            let error = fault(self.path("dispatch.pipeline"), ErrorCode::PassMismatch);
            self.fault(error);
            return 0;
        }
        if built.compute_ready.is_none() {
            let Some(made) = self.compute_pipeline(handle, true) else {
                return 0;
            };
            self.machine.pipelines[handle as usize].compute_ready = Some(made);
        }
        if let CountRaw::Indirect(input, _) = count
            && input_of(&self.machine, (&self.tables, self.set()), part, input).is_none()
        {
            return 0;
        }
        let current = self.pass;
        if self.recording.computing(current).is_none() {
            self.end_render();
            self.group(current);
            let label = self.label_of(current).unwrap_or("engine/part-dispatch");
            let Some(encoder) = self.recording.encoder() else {
                return 0;
            };
            let begun = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                })
                .forget_lifetime();
            self.recording
                .transition(Step::Compute(current, Box::new(begun)));
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
            ..
        } = self;
        let parity = frames.parity;
        let set = &sets.list[sets.at];
        let Some(pass) = recording.computing(current) else {
            return 0;
        };
        let built = &machine.pipelines[handle as usize];
        let Some(compute) = built.compute_ready.as_ref() else {
            return 0;
        };
        pass.set_pipeline(compute);
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
            pass.set_bind_group(0, group, &[offset][..camera]);
        }
        let mut given = offsets;
        let mut held = [0_u32; MOST_OFFSETS];
        for (at, group) in built.groups.iter().enumerate() {
            let dynamic = offsets_of(group.windows, &mut given, &mut held);
            let holder = Holder::Own(handle, at as u32);
            if let Some(made) = held_group(&own_groups[handle as usize][at], set, holder, parity) {
                pass.set_bind_group(group.index, made, dynamic);
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
        1
    }
}

fn timed_draw(
    render: &mut wgpu::RenderPass<'static>,
    timed: TimedDraw<'_>,
    draw: impl FnOnce(&mut wgpu::RenderPass<'static>),
) {
    let Some((timer, label)) = timed else {
        draw(render);
        return;
    };
    let began = timer.begin_draw(render, label);
    draw(render);
    timer.end_draw(render, began);
}
