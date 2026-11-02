use super::*;

type OwnGroups = Vec<(u32, Option<wgpu::BindGroup>, Vec<u32>)>;

fn with_offsets(mut own: OwnGroups, groups: &[crate::machine::OwnGroup], offsets: &[u32]) -> OwnGroups {
    let mut given = offsets.iter().copied();
    for ((_, _, dynamic), group) in own.iter_mut().zip(groups) {
        let mut windows: Vec<u32> = group
            .entries
            .iter()
            .filter(|(_, bound, _)| bound.window())
            .map(|(binding, _, _)| *binding)
            .collect();
        windows.sort_unstable();
        dynamic.extend(windows.iter().map(|_| given.next().unwrap_or(0)));
    }
    own
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
                    && self.machine.declared.passes.get(self.pass as usize).is_some_and(shared)
            }
            (Scoped::Pass(_), Some(_)) => false,
        }
    }

    pub(super) fn input(&self, part: u32, input: u32) -> Option<wgpu::Buffer> {
        let decl = self.machine.declared.inputs.get(input as usize)?;
        if decl.part != part {
            return None;
        }
        self.buffers.get(decl.buffer as usize)?.as_ref().map(|(buffer, _)| buffer.clone())
    }

    #[expect(clippy::too_many_lines, reason = "longer than the line ceiling; it may only shrink")]
    pub(crate) fn draw(&mut self, part: u32, point: Option<TypeId>, handle: u32, (shape, offsets, pass): (RawShape, &[u32], crate::MeshPass), timed: TimedDraw<'_>) {
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
        if self.machine.pipelines[pipeline as usize].decl.holes() > 0 && filled.is_none() {
            return;
        }
        if filled.is_some_and(|filled| self.filled[filled].groups.is_empty()) {
            self.fill_groups(pipeline, filled);
        }
        let uber = point.is_some_and(|point| self.machine.declared.claims.iter().any(|claim| claim.point == point && claim.uber));
        let variant = match &shape {
            RawShape::Mesh { mesh, material, .. } | RawShape::MeshIndirect { mesh, material, .. } => {
                let extended = self.assets.meshes.get(*mesh).is_some_and(|gpu| gpu.extended);
                let known = material.is_none_or(|handle| self.assets.materials.group(handle).is_some());
                known.then(|| {
                    let variant = self
                        .assets
                        .materials
                        .binding(*material)
                        .1
                        .with_extended(extended)
                        .with_fading(pass.fading)
                        .with_grounding(pass.grounding);
                    let variant = if uber {
                        variant.uber()
                    } else if pass.prepass {
                        variant.prepass()
                    } else if pass.opaque {
                        variant.opaque()
                    } else {
                        variant
                    };
                    (variant.bits(), variant.constants())
                })
            }
            RawShape::Pulled { .. } | RawShape::Buffers { .. } | RawShape::Indirect { .. } | RawShape::IndirectAt { .. } => None,
        };
        let Some((key, made)) = self.render_pipeline(pipeline, variant) else {
            return;
        };
        let parity = self.parity;
        let built = &self.machine.pipelines[pipeline as usize];
        let variant = built.variant;
        let mesh_slots = match &built.decl.program {
            Program::Render(desc) => desc.vertices.iter().filter(|layout| matches!(layout, VertexLayout::Mesh)).count() as u32,
            Program::Compute(_) => 0,
        };
        let frame_group = self.frame_groups.get(variant as usize).and_then(|groups| groups.0[parity].clone());
        let camera = usize::from(self.camera_bound(variant));
        let own: Vec<(u32, Option<wgpu::BindGroup>, Vec<u32>)> = built
            .groups
            .iter()
            .enumerate()
            .map(|(at, group)| {
                let filled_group = filled.and_then(|filled| {
                    self.filled[filled].groups.iter().find(|(index, _)| *index == group.index).map(|(_, made)| made.clone())
                });
                (group.index, filled_group.or_else(|| self.own_groups[pipeline as usize].0[at][parity].clone()), Vec::new())
            })
            .collect();
        let own = with_offsets(own, &built.groups, offsets);
        let vertex_inputs: Vec<Option<wgpu::Buffer>> = match &shape {
            RawShape::Buffers { vertices, .. } | RawShape::Indirect { vertices, .. } => {
                vertices.iter().map(|input| self.input(part, *input)).collect()
            }
            RawShape::Pulled { .. } | RawShape::IndirectAt { .. } | RawShape::Mesh { .. } | RawShape::MeshIndirect { .. } => Vec::new(),
        };
        let index_input = match &shape {
            RawShape::Buffers { indices, .. } | RawShape::Indirect { indices, .. } | RawShape::IndirectAt { indices, .. } => {
                indices.and_then(|input| self.input(part, input))
            }
            RawShape::Pulled { .. } | RawShape::Mesh { .. } | RawShape::MeshIndirect { .. } => None,
        };
        let args_input = match &shape {
            RawShape::Indirect { args, .. } | RawShape::IndirectAt { args, .. } | RawShape::MeshIndirect { args, .. } => self.input(part, *args),
            RawShape::Pulled { .. } | RawShape::Buffers { .. } | RawShape::Mesh { .. } => None,
        };
        let stream_inputs: Vec<(Option<wgpu::Buffer>, u64)> = match &shape {
            RawShape::Mesh { streams, .. } | RawShape::MeshIndirect { streams, .. } | RawShape::IndirectAt { vertices: streams, .. } => {
                streams.iter().map(|(input, offset)| (self.input(part, *input), *offset)).collect()
            }
            RawShape::Pulled { .. } | RawShape::Buffers { .. } | RawShape::Indirect { .. } => Vec::new(),
        };
        let mesh = match &shape {
            RawShape::Mesh { mesh, material, .. } | RawShape::MeshIndirect { mesh, material, .. } => {
                let Some(draw) = self.assets.meshes.draw(*mesh) else {
                    return;
                };
                let group = match material {
                    Some(handle) => match self.assets.materials.group(*handle) {
                        Some(group) => group.clone(),
                        None => return,
                    },
                    None => self.assets.materials.binding(None).0.clone(),
                };
                Some((draw.vertices.clone(), draw.extras.cloned(), draw.indices.clone(), draw.first_index, draw.base_vertex, draw.index_count, group))
            }
            RawShape::Pulled { .. } | RawShape::Buffers { .. } | RawShape::Indirect { .. } | RawShape::IndirectAt { .. } => None,
        };
        let Some(open) = self.open.as_mut() else {
            return;
        };
        let render = &mut open.pass;
        if open.set.pipeline != Some((pipeline, key)) {
            render.set_pipeline(&made);
            open.set.pipeline = Some((pipeline, key));
            open.set.frame = None;
        }
        if open.set.frame != Some((variant, open.offset))
            && let Some(group) = &frame_group
        {
            render.set_bind_group(0, group, &[open.offset][..camera]);
            open.set.frame = Some((variant, open.offset));
        }
        for (index, group, dynamic) in &own {
            if let Some(group) = group {
                render.set_bind_group(*index, group, dynamic);
            }
        }
        match shape {
            RawShape::Pulled { vertices, instances } => timed_draw(render, timed, |render| render.draw(vertices, instances)),
            RawShape::Buffers { elements, instances, .. } => {
                for (slot, buffer) in vertex_inputs.iter().enumerate() {
                    if let Some(buffer) = buffer {
                        render.set_vertex_buffer(slot as u32, buffer.slice(..));
                    }
                }
                match index_input {
                    Some(indices) => {
                        render.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                        timed_draw(render, timed, |render| render.draw_indexed(elements, 0, instances));
                    }
                    None => timed_draw(render, timed, |render| render.draw(elements, instances)),
                }
            }
            RawShape::Indirect { offset, .. } => {
                for (slot, buffer) in vertex_inputs.iter().enumerate() {
                    if let Some(buffer) = buffer {
                        render.set_vertex_buffer(slot as u32, buffer.slice(..));
                    }
                }
                let Some(args) = args_input else { return };
                match index_input {
                    Some(indices) => {
                        render.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                        timed_draw(render, timed, |render| render.draw_indexed_indirect(&args, offset));
                    }
                    None => timed_draw(render, timed, |render| render.draw_indirect(&args, offset)),
                }
            }
            RawShape::IndirectAt { offset, .. } => {
                for (slot, (buffer, start)) in stream_inputs.iter().enumerate() {
                    if let Some(buffer) = buffer {
                        render.set_vertex_buffer(slot as u32, buffer.slice(*start..));
                    }
                }
                let Some(args) = args_input else { return };
                match index_input {
                    Some(indices) => {
                        render.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                        timed_draw(render, timed, |render| render.draw_indexed_indirect(&args, offset));
                    }
                    None => timed_draw(render, timed, |render| render.draw_indirect(&args, offset)),
                }
            }
            RawShape::Mesh { instances, .. } => {
                let Some((vertices, extras, indices, first, base, count, group)) = mesh else { return };
                render.set_bind_group(1, &group, &[]);
                render.set_vertex_buffer(0, vertices.slice(..));
                if mesh_slots > 1
                    && let Some(extras) = &extras
                {
                    render.set_vertex_buffer(1, extras.slice(..));
                }
                for (at, (buffer, start)) in stream_inputs.iter().enumerate() {
                    if let Some(buffer) = buffer {
                        render.set_vertex_buffer(mesh_slots + at as u32, buffer.slice(*start..));
                    }
                }
                render.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                timed_draw(render, timed, |render| render.draw_indexed(first..first + count, base, instances));
            }
            RawShape::MeshIndirect { offset, .. } => {
                let Some((vertices, extras, indices, _, _, _, group)) = mesh else { return };
                let Some(args) = args_input else { return };
                render.set_bind_group(1, &group, &[]);
                render.set_vertex_buffer(0, vertices.slice(..));
                if mesh_slots > 1
                    && let Some(extras) = &extras
                {
                    render.set_vertex_buffer(1, extras.slice(..));
                }
                for (at, (buffer, start)) in stream_inputs.iter().enumerate() {
                    if let Some(buffer) = buffer {
                        render.set_vertex_buffer(mesh_slots + at as u32, buffer.slice(*start..));
                    }
                }
                render.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                timed_draw(render, timed, |render| render.draw_indexed_indirect(&args, offset));
            }
        }
    }

    pub(super) fn fill_groups(&mut self, pipeline: u32, filled: Option<usize>) {
        let Some(filled) = filled else { return };
        let textures = self.filled[filled].textures.clone();
        let groups = self.machine.pipelines[pipeline as usize].groups.len();
        let mut made = Vec::with_capacity(groups);
        for group in 0..groups {
            let entries = self.machine.pipelines[pipeline as usize].groups[group].entries.clone();
            let index = self.machine.pipelines[pipeline as usize].groups[group].index;
            let mut hole = 0;
            let mut resolved = Vec::with_capacity(entries.len());
            for (binding, bound, ty) in entries {
                let found = if bound == Bound::Asset {
                    let texture = textures.get(hole).copied();
                    hole += 1;
                    match texture.and_then(|texture| self.assets.materials.texture_view(texture, true)) {
                        Some(view) => Resolved::View(view.clone()),
                        None => Resolved::Placeholder(ty),
                    }
                } else {
                    self.resource(&bound, ty, !ty.writes())
                };
                resolved.push((binding, found));
            }
            let layout = self.machine.pipelines[pipeline as usize].groups[group].layout.clone();
            made.push((index, self.bind_group(&layout, (Some(pipeline as usize), index), resolved)));
        }
        self.filled[filled].groups = made;
    }

    pub(crate) fn scissor(&mut self, rect: [u32; 4]) {
        if let Some(open) = self.open.as_mut() {
            open.pass.set_scissor_rect(rect[0], rect[1], rect[2], rect[3]);
        }
    }

    pub(crate) fn within(&mut self, size: (u32, u32)) {
        if let Some(open) = self.open.as_mut() {
            crate::targets::draw_within(&mut open.pass, size);
        }
    }

    pub(crate) fn reset_scissor(&mut self) {
        let size = self.targets.first().map(|(_, made)| (made.size.width, made.size.height));
        if let (Some(open), Some((width, height))) = (self.open.as_mut().filter(|open| open.layered), self.region.or(size)) {
            open.pass.set_scissor_rect(0, 0, width, height);
        }
    }

    pub(crate) fn dispatch(&mut self, part: u32, handle: u32, count: CountRaw, offsets: &[u32]) -> u32 {
        self.end_open();
        let Some(built) = self.machine.pipelines.get(handle as usize) else {
            return 0;
        };
        if built.decl.part != part || built.decl.scope != Scoped::Pass(self.pass) {
            let error = fault(self.path("dispatch.pipeline"), ErrorCode::PassMismatch);
            self.fault(error);
            return 0;
        }
        let Some(compute) = self.compute_pipeline(handle) else {
            return 0;
        };
        let built = &self.machine.pipelines[handle as usize];
        let label = self.label_of(self.pass).unwrap_or("engine/part-dispatch");
        let parity = self.parity;
        let frame_group = self.frame_groups.get(built.variant as usize).and_then(|groups| groups.0[parity].clone());
        let camera = usize::from(self.camera_bound(built.variant));
        let own: Vec<(u32, Option<wgpu::BindGroup>, Vec<u32>)> = built
            .groups
            .iter()
            .enumerate()
            .map(|(at, group)| (group.index, self.own_groups[handle as usize].0[at][parity].clone(), Vec::new()))
            .collect();
        let own = with_offsets(own, &built.groups, offsets);
        let indirect = match count {
            CountRaw::Indirect(input, offset) => match self.input(part, input) {
                Some(buffer) => Some((buffer, offset)),
                None => return 0,
            },
            CountRaw::Groups(_) => None,
        };
        let current = self.pass;
        if self.computing.as_ref().is_none_or(|(known, _)| *known != current) {
            self.computing = None;
            self.group(current);
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
        let Some((_, pass)) = self.computing.as_mut() else {
            return 0;
        };
        pass.set_pipeline(&compute);
        if let Some(group) = &frame_group {
            pass.set_bind_group(0, group, &[0][..camera]);
        }
        for (index, group, dynamic) in &own {
            if let Some(group) = group {
                pass.set_bind_group(*index, group, dynamic);
            }
        }
        match (count, indirect) {
            (_, Some((buffer, offset))) => pass.dispatch_workgroups_indirect(&buffer, offset),
            (CountRaw::Groups([x, y, z]), None) => pass.dispatch_workgroups(x, y, z),
            (CountRaw::Indirect(..), None) => {}
        }
        1
    }
}
