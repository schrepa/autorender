use super::*;

fn input_of<'b>(
    machine: &Machine,
    buffers: &'b [Option<(wgpu::Buffer, u64)>],
    part: u32,
    input: u32,
) -> Option<&'b wgpu::Buffer> {
    let decl = machine.declared.inputs.get(input as usize)?;
    if decl.part != part {
        return None;
    }
    buffers
        .get(decl.buffer as usize)?
        .as_ref()
        .map(|(buffer, _)| buffer)
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
    pub(super) fn scoped(&self, part: u32, point: Option<TypeId>, pipeline: u32) -> bool {
        let Some(built) = self.machine.pipelines.get(pipeline as usize) else {
            return false;
        };
        if built.decl.part != part {
            return false;
        }
        match (built.decl.scope, point) {
            (Scoped::Pass(pass), None) => pass == self.pass,
            (Scoped::Point(known), Some(point)) => known == point,
            (Scoped::Point(known), None) => {
                known == TypeId::of::<crate::MainView>()
                    && self
                        .machine
                        .declared
                        .passes
                        .get(self.pass as usize)
                        .is_some_and(shared)
            }
            (Scoped::Pass(_), Some(_)) => false,
        }
    }

    fn mesh_variant(
        &self,
        point: Option<TypeId>,
        mesh: MeshHandle,
        material: Option<MaterialHandle>,
        pass: crate::MeshPass,
    ) -> Option<crate::materials::Variant> {
        let extended = self.assets.meshes.get(mesh)?.extended;
        if material.is_some_and(|handle| self.assets.materials.group(handle).is_none()) {
            return None;
        }
        let uber = point.is_some_and(|point| {
            self.machine
                .declared
                .claims
                .iter()
                .any(|claim| claim.point == point && claim.uber)
        });
        let variant = self
            .assets
            .materials
            .binding(material)
            .1
            .with_extended(extended)
            .with_fading(pass.fading)
            .with_grounding(pass.grounding);
        Some(if uber {
            variant.uber()
        } else if pass.prepass {
            variant.prepass()
        } else if pass.opaque {
            variant.opaque()
        } else {
            variant
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(crate) fn draw(
        &mut self,
        part: u32,
        point: Option<TypeId>,
        handle: u32,
        (shape, offsets, pass): (RawShape, &[u32], crate::MeshPass),
        timed: TimedDraw<'_>,
    ) {
        let _row = engine_profile::timed(engine_profile::row("render.draw"));
        let Some((pipeline, filled)) = self.pipeline_of(handle) else {
            return;
        };
        if !self.scoped(part, point, pipeline) {
            let saved = self.part;
            self.part = part;
            let error = fault(self.path("draw.pipeline"), ErrorCode::PassMismatch);
            self.fault(error);
            self.part = saved;
            return;
        }
        if self.machine.pipelines[pipeline as usize].holes && filled.is_none() {
            return;
        }
        if filled.is_some_and(|filled| self.filled[filled].groups.is_empty()) {
            self.fill_groups(pipeline, filled);
        }
        let variant = match &shape {
            RawShape::MeshIndirect { mesh, material, .. } => {
                let Some(variant) = self.mesh_variant(point, *mesh, *material, pass) else {
                    return;
                };
                Some(variant)
            }
            RawShape::Pulled { .. }
            | RawShape::Buffers { .. }
            | RawShape::Indirect { .. }
            | RawShape::IndirectAt { .. } => None,
        };
        let Some(ready) = self.render_pipeline(pipeline, variant) else {
            return;
        };
        let Self {
            machine,
            open,
            frame_groups,
            own_groups,
            filled: fills,
            buffers,
            assets,
            parity,
            ..
        } = self;
        let parity = *parity;
        let Some(open) = open.as_mut() else {
            return;
        };
        let built = &machine.pipelines[pipeline as usize];
        let chosen = &built.ready[ready];
        let render = &mut open.pass;
        if open.set.pipeline != Some((pipeline, chosen.key)) {
            render.set_pipeline(&chosen.made);
            open.set.pipeline = Some((pipeline, chosen.key));
            open.set.frame = None;
        }
        if open.set.frame != Some((built.variant, open.offset))
            && let Some(group) = frame_groups
                .get(built.variant as usize)
                .and_then(|groups| groups.made[parity].as_ref())
        {
            let camera = usize::from(
                machine
                    .variants
                    .get(built.variant as usize)
                    .is_some_and(|found| found.camera),
            );
            render.set_bind_group(0, group, &[open.offset][..camera]);
            open.set.frame = Some((built.variant, open.offset));
        }
        let mut given = offsets;
        let mut held = [0_u32; MOST_OFFSETS];
        for (at, group) in built.groups.iter().enumerate() {
            let dynamic = offsets_of(group.windows, &mut given, &mut held);
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
            if let Some(made) =
                filled_group.or(own_groups[pipeline as usize][at].made[parity].as_ref())
            {
                render.set_bind_group(group.index, made, dynamic);
                if let Some(known) = open.set.own.get_mut(slot) {
                    *known = Some(holding);
                }
                if group.index == 1 {
                    open.set.material = None;
                }
            }
        }
        let input = |input: u32| input_of(machine, buffers, part, input);
        let mesh_slots = built.mesh_slots;
        match shape {
            RawShape::Pulled {
                vertices,
                instances,
            } => timed_draw(render, timed, |render| render.draw(vertices, instances)),
            RawShape::Buffers {
                vertices,
                indices,
                elements,
                instances,
            } => {
                for (slot, (stream, _)) in vertices.listed().iter().enumerate() {
                    if let Some(buffer) = input(*stream) {
                        render.set_vertex_buffer(slot as u32, buffer.slice(..));
                    }
                }
                match indices.and_then(input) {
                    Some(indices) => {
                        render.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
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
            } => {
                for (slot, (stream, _)) in vertices.listed().iter().enumerate() {
                    if let Some(buffer) = input(*stream) {
                        render.set_vertex_buffer(slot as u32, buffer.slice(..));
                    }
                }
                let Some(args) = input(args) else { return };
                match indices.and_then(input) {
                    Some(indices) => {
                        render.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
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
                for (slot, (stream, start)) in vertices.listed().iter().enumerate() {
                    if let Some(buffer) = input(*stream) {
                        render.set_vertex_buffer(slot as u32, buffer.slice(*start..));
                    }
                }
                let Some(args) = input(args) else { return };
                match indices.and_then(input) {
                    Some(indices) => {
                        render.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
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
                let Some(draw) = assets.meshes.draw(mesh) else {
                    return;
                };
                let group = match material {
                    Some(handle) => assets.materials.group(handle),
                    None => Some(assets.materials.binding(None).0),
                };
                let Some(group) = group else { return };
                let Some(args) = input(args) else { return };
                if open.set.material != Some(material) {
                    render.set_bind_group(1, group, &[]);
                    open.set.material = Some(material);
                    open.set.own[1] = None;
                }
                render.set_vertex_buffer(0, draw.vertices.slice(..));
                if mesh_slots > 1
                    && let Some(extras) = draw.extras
                {
                    render.set_vertex_buffer(1, extras.slice(..));
                }
                for (at, (stream, start)) in streams.listed().iter().enumerate() {
                    if let Some(buffer) = input(*stream) {
                        render.set_vertex_buffer(mesh_slots + at as u32, buffer.slice(*start..));
                    }
                }
                render.set_index_buffer(draw.indices.slice(..), wgpu::IndexFormat::Uint32);
                timed_draw(render, timed, |render| {
                    render.draw_indexed_indirect(args, offset)
                });
            }
        }
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
                    match texture
                        .and_then(|texture| self.assets.materials.texture_view(texture, true))
                    {
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
        if let Some(open) = self.open.as_mut() {
            open.pass
                .set_scissor_rect(rect[0], rect[1], rect[2], rect[3]);
        }
    }

    pub(crate) fn within(&mut self, size: (u32, u32)) {
        if let Some(open) = self.open.as_mut() {
            crate::targets::draw_within(&mut open.pass, size);
        }
    }

    pub(crate) fn reset_scissor(&mut self) {
        let size = self
            .targets
            .first()
            .map(|(_, made)| (made.size.width, made.size.height));
        if let (Some(open), Some((width, height))) = (
            self.open.as_mut().filter(|open| open.layered),
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
            let Some(made) = self.compute_pipeline(handle) else {
                return 0;
            };
            self.machine.pipelines[handle as usize].compute_ready = Some(made);
        }
        if let CountRaw::Indirect(input, _) = count
            && input_of(&self.machine, &self.buffers, part, input).is_none()
        {
            return 0;
        }
        let current = self.pass;
        if self
            .computing
            .as_ref()
            .is_none_or(|(known, _)| *known != current)
        {
            self.computing = None;
            self.group(current);
            let label = self.label_of(current).unwrap_or("engine/part-dispatch");
            let Some(encoder) = self.encoder.as_mut() else {
                return 0;
            };
            let begun = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                })
                .forget_lifetime();
            self.computing = Some((current, begun));
        }
        let Self {
            machine,
            computing,
            frame_groups,
            own_groups,
            buffers,
            parity,
            ..
        } = self;
        let parity = *parity;
        let Some((_, pass)) = computing.as_mut() else {
            return 0;
        };
        let built = &machine.pipelines[handle as usize];
        let Some(compute) = built.compute_ready.as_ref() else {
            return 0;
        };
        pass.set_pipeline(compute);
        if let Some(group) = frame_groups
            .get(built.variant as usize)
            .and_then(|groups| groups.made[parity].as_ref())
        {
            let camera = usize::from(
                machine
                    .variants
                    .get(built.variant as usize)
                    .is_some_and(|found| found.camera),
            );
            pass.set_bind_group(0, group, &[0][..camera]);
        }
        let mut given = offsets;
        let mut held = [0_u32; MOST_OFFSETS];
        for (at, group) in built.groups.iter().enumerate() {
            let dynamic = offsets_of(group.windows, &mut given, &mut held);
            if let Some(made) = own_groups[handle as usize][at].made[parity].as_ref() {
                pass.set_bind_group(group.index, made, dynamic);
            }
        }
        match count {
            CountRaw::Indirect(input, offset) => {
                if let Some(buffer) = input_of(machine, buffers, part, input) {
                    pass.dispatch_workgroups_indirect(buffer, offset);
                }
            }
            CountRaw::Groups([x, y, z]) => pass.dispatch_workgroups(x, y, z),
        }
        1
    }
}
