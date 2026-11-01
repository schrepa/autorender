use super::*;

impl Run {
    pub(crate) fn cameras(&mut self, views: &[CameraUniform]) {
        let part = self.part;
        match self.views.iter().position(|(known, _, _)| *known == part) {
            Some(at) if self.views[at].2 as usize == views.len() => {
                let base = self.views[at].1 as usize;
                self.cameras[base..base + views.len()].copy_from_slice(views);
            }
            Some(at) => {
                self.views.remove(at);
                self.repack();
                self.push_views(part, views);
            }
            None => self.push_views(part, views),
        }
    }

    pub(super) fn push_views(&mut self, part: u32, views: &[CameraUniform]) {
        let base = self.cameras.len() as u32;
        self.cameras.extend_from_slice(views);
        self.views.push((part, base, views.len() as u32));
    }

    pub(super) fn repack(&mut self) {
        let mut packed = vec![self.cameras[0]];
        for (_, base, count) in &mut self.views {
            let from = *base as usize;
            let kept = &self.cameras[from..from + *count as usize];
            *base = packed.len() as u32;
            packed.extend_from_slice(kept);
        }
        self.cameras = packed;
    }

    pub(crate) fn view_offset(&self, part: u32, view: Option<u32>) -> Option<u32> {
        let Some(view) = view else {
            return Some(0);
        };
        let (_, base, count) = self.views.iter().find(|(known, _, _)| *known == part)?;
        (view < *count).then(|| (base + view) * CAMERA_STRIDE as u32)
    }

    pub(crate) fn write_cameras(&mut self) -> u64 {
        let bytes = self.cameras.len() as u64 * CAMERA_STRIDE;
        if self.camera.size() < bytes {
            self.camera = crate::memory::buffer(
                &self.device,
                &wgpu::BufferDescriptor {
                    label: Some("engine/cameras"),
                    size: bytes.next_power_of_two(),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            );
            self.generation += 1;
            self.uploads.cameras.clear();
        }
        let mut staged = vec![0_u8; bytes as usize];
        for (index, camera) in self.cameras.iter().enumerate() {
            let at = index * CAMERA_STRIDE as usize;
            let raw = bytemuck::bytes_of(camera);
            staged[at..at + raw.len()].copy_from_slice(raw);
        }
        if staged == self.uploads.cameras {
            return 0;
        }
        let main = size_of::<CameraUniform>();
        let changed = self.uploads.cameras.get(..main) != staged.get(..main);
        crate::gpu_calls::upload_buffer(&self.queue, &self.camera, 0, &staged);
        self.uploads.cameras = staged;
        if changed { main as u64 } else { 0 }
    }

    pub(crate) fn write_post(&mut self) {
        if self.post_written == Some(self.post) {
            return;
        }
        crate::gpu_calls::upload_buffer(&self.queue, &self.post_buffer, 0, bytemuck::bytes_of(&self.post));
        self.post_written = Some(self.post);
    }

    pub(crate) fn write_pieces(&mut self, bytes: Vec<u8>) -> u64 {
        if self.uploads.pieces.as_ref() == Some(&bytes) {
            return 0;
        }
        let size = (bytes.len() as u64).max(crate::pieces::PIECE_ALIGN).next_multiple_of(crate::pieces::PIECE_ALIGN);
        if self.pieces.size() != size {
            self.pieces = crate::memory::buffer(
                &self.device,
                &wgpu::BufferDescriptor {
                    label: Some("engine/pieces"),
                    size,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            );
            self.generation += 1;
        }
        let mut staged = bytes.clone();
        staged.resize(size as usize, 0);
        crate::gpu_calls::upload_buffer(&self.queue, &self.pieces, 0, &staged);
        self.uploads.pieces = Some(bytes);
        size
    }

    pub(crate) fn read_back(&mut self, asked: Asked) -> bool {
        let slots = self.readback_slots;
        let free = asked.once || match self.rings.iter().find(|ring| ring.part == asked.part && ring.tag == asked.tag) {
            Some(ring) => {
                ring.kept() < slots
                    || ring.slots.iter().any(|slot| !slot.once && slot.state.load(Ordering::Acquire) == 0)
            }
            None => true,
        };
        if free {
            self.asked.push(asked);
        }
        free
    }

    pub(crate) fn fill(&mut self, pipeline: u32, textures: &[TextureHandle]) -> u32 {
        for texture in textures {
            if !self.wanted_textures.contains(texture) {
                self.wanted_textures.push(*texture);
            }
        }
        let generation = self.assets.materials.generation();
        let base = self.machine.pipelines.len() as u32;
        if let Some(at) = self.filled.iter().position(|filled| filled.base == pipeline && filled.textures == textures && filled.generation == generation) {
            self.filled[at].used = self.frame;
            return base + at as u32;
        }
        self.filled.push(Filled {
            base: pipeline,
            textures: textures.to_vec(),
            generation,
            used: self.frame,
            groups: Vec::new(),
        });
        base + (self.filled.len() - 1) as u32
    }

    pub(crate) fn drop_unfilled(&mut self) {
        let frame = self.frame;
        self.filled.retain(|filled| filled.used + 1 >= frame);
    }

    pub(super) fn reads_direct(&self) -> bool {
        let index = self.pass as usize;
        self.machine.declared.passes.get(index).is_some_and(|pass| {
            reads_depth_direct(pass, self.machine.depth_writes.get(index).copied().unwrap_or(true))
        })
    }

    pub(super) fn output_view(&self, index: u32, writing: bool) -> Option<&wgpu::TextureView> {
        let target = Target::Output(index);
        let doubled = self
            .machine
            .outputs
            .get(index as usize)
            .is_some_and(|output| output.layers.is_empty());
        let second = doubled && (self.parity == 1) == writing;
        let list = if second { &self.previous_targets } else { &self.targets };
        list.iter().find(|(known, _)| *known == target).map(|(_, made)| &made.view)
    }

    pub(super) fn view_of(&self, target: Target, writing: bool) -> Option<&wgpu::TextureView> {
        match target {
            Target::Canvas => self.canvas.as_ref(),
            Target::Debug(index) => self.debug_targets.get(index as usize).map(|made| &made.view),
            Target::Output(index) => self.output_view(index, writing),
            Target::Scene | Target::Depth | Target::Copy(..) => {
                self.targets.iter().find(|(known, _)| *known == target).map(|(_, made)| &made.view)
            }
        }
    }

    pub(crate) fn texture_view(&mut self, bound: &Bound, ty: Ty, reading: bool) -> Option<wgpu::TextureView> {
        let parity = self.parity;
        let dimension = match ty {
            Ty::Texture { view, .. } | Ty::StorageTexture { view, .. } => Some(view),
            Ty::Uniform | Ty::Storage { .. } | Ty::Sampler | Ty::Comparison => None,
        };
        match bound {
            Bound::Texture(texture) | Bound::Slice { texture, .. } => {
                let index = if self.doubled(*texture) && reading { 1 - parity } else { parity };
                let made = self.textures.get_mut(*texture as usize)?;
                let made = match index {
                    1 if made[1].is_some() => made[1].as_mut()?,
                    _ => made[0].as_mut()?,
                };
                let (mips, layers) = match bound {
                    Bound::Slice { mips, layers, .. } => (mips.clone(), layers.clone()),
                    Bound::Texture(_) | Bound::Buffer(_) | Bound::Window { .. } | Bound::Sampler(_) | Bound::Target(_) | Bound::Asset => {
                        (0..made.mips, 0..made.layers())
                    }
                };
                made.slice(&mips, &layers, dimension).cloned()
            }
            Bound::Target(view) => {
                let wanted = target_read(*view, self.pass_layer(), self.reads_direct());
                self.view_of(wanted, false).cloned()
            }
            Bound::Buffer(_) | Bound::Window { .. } | Bound::Sampler(_) | Bound::Asset => None,
        }
    }

    pub(crate) fn pass_layer(&self) -> Option<crate::Layer> {
        pass_layer(self.machine.declared.passes.get(self.pass as usize)?)
    }

    pub(super) fn placeholder(&mut self, key: (Ty, u64)) {
        if !self.placeholders.iter().any(|(known, _)| *known == key) {
            let made = placeholder(&self.device, key);
            self.placeholders.push((key, made));
        }
    }

    pub(super) fn binding_bytes(&mut self, pipeline: Option<usize>, group: u32, binding: u32) -> u64 {
        let key = (pipeline, group, binding);
        if let Some((_, bytes)) = self.sized.iter().find(|(known, _)| *known == key) {
            return *bytes;
        }
        let marker = format!("@binding({binding})");
        let reach = match pipeline {
            Some(at) => at..at + 1,
            None => 0..self.machine.pipelines.len(),
        };
        let needed = reach
            .filter_map(|at| self.machine.pipelines.get(at))
            .flat_map(|built| built.stages.iter())
            .filter(|stage| stage.composed.text.contains(&marker))
            .find_map(|stage| crate::sizes::binding_size(&stage.composed.text, group, binding));
        let bytes = needed.unwrap_or(0).max(PLACEHOLDER_BYTES).next_multiple_of(16);
        self.sized.push((key, bytes));
        bytes
    }

    pub(super) fn resource(&mut self, bound: &Bound, ty: Ty, reading: bool) -> Resolved {
        match bound {
            Bound::Buffer(buffer) => match self.buffers.get(*buffer as usize).and_then(Option::as_ref) {
                Some((made, _)) => Resolved::Buffer(made.clone()),
                None => Resolved::Placeholder(ty),
            },
            Bound::Window { buffer, size } => match self.buffers.get(*buffer as usize).and_then(Option::as_ref) {
                Some((made, bytes)) if *bytes >= *size => Resolved::Dynamic(made.clone(), *size),
                Some(_) | None => Resolved::Placeholder(ty),
            },
            Bound::Sampler(sampler) => match self.samplers.get(*sampler as usize) {
                Some(made) => Resolved::Sampler(made.clone()),
                None => Resolved::Placeholder(ty),
            },
            Bound::Texture(_) | Bound::Slice { .. } | Bound::Target(_) => match self.texture_view(bound, ty, reading) {
                Some(view) => Resolved::View(view),
                None => Resolved::Placeholder(ty),
            },
            Bound::Asset => Resolved::Placeholder(ty),
        }
    }

    #[expect(clippy::disallowed_methods, reason = "the machine binds groups by the linked declarations")]
    pub(super) fn bind_group(
        &mut self,
        layout: &wgpu::BindGroupLayout,
        (pipeline, group): (Option<usize>, u32),
        entries: Vec<(u32, Resolved)>,
    ) -> wgpu::BindGroup {
        let mut keys: Vec<Option<(Ty, u64)>> = Vec::with_capacity(entries.len());
        for (binding, resolved) in &entries {
            let key = match resolved {
                Resolved::Placeholder(ty @ (Ty::Uniform | Ty::Storage { .. })) => Some((*ty, self.binding_bytes(pipeline, group, *binding))),
                Resolved::Placeholder(ty) => Some((*ty, 0)),
                Resolved::Buffer(_) | Resolved::Dynamic(..) | Resolved::View(_) | Resolved::Sampler(_) => None,
            };
            if let Some(key) = key {
                self.placeholder(key);
            }
            keys.push(key);
        }
        let resources: Vec<wgpu::BindGroupEntry<'_>> = entries
            .iter()
            .zip(&keys)
            .map(|((binding, resolved), key)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: match resolved {
                    Resolved::Buffer(buffer) => buffer.as_entire_binding(),
                    Resolved::Dynamic(buffer, size) => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(*size),
                    }),
                    Resolved::View(view) => wgpu::BindingResource::TextureView(view),
                    Resolved::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
                    Resolved::Placeholder(_) => match key.and_then(|key| self.placeholders.iter().find(|(known, _)| *known == key)) {
                        Some((_, Placeholder::Buffer(buffer))) => buffer.as_entire_binding(),
                        Some((_, Placeholder::Texture(view))) => wgpu::BindingResource::TextureView(view),
                        Some((_, Placeholder::Sampler(sampler))) => wgpu::BindingResource::Sampler(sampler),
                        None => self.camera.as_entire_binding(),
                    },
                },
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine/bound-group"),
            layout,
            entries: &resources,
        })
    }

    pub(crate) fn refresh_groups(&mut self) {
        let atlas = self.assets.impostors.generation();
        if self.atlas_generation != Some(atlas) {
            self.atlas_generation = Some(atlas);
            self.generation += 1;
        }
        let generation = self.generation;
        let saved_pass = self.pass;
        self.pass = u32::MAX;
        for variant in 0..self.machine.variants.len() {
            if self.frame_groups[variant].1 == generation {
                continue;
            }
            for parity in 0..2 {
                let saved = self.parity;
                self.parity = parity;
                let entries: Vec<(u32, crate::bindings::Stages)> = self.machine.variants[variant].entries.clone();
                let mut resolved = Vec::with_capacity(entries.len());
                for (binding, _) in entries {
                    let ty = self.frame_ty(binding);
                    let found = match binding {
                        0 => Resolved::Dynamic(self.camera.clone(), size_of::<CameraUniform>() as u64),
                        1 => Resolved::Buffer(self.pieces.clone()),
                        _ => match self.machine.frame_bound.get(&binding).cloned() {
                            Some(bound) => self.resource(&bound, ty, true),
                            None => self.core_frame(binding, ty),
                        },
                    };
                    resolved.push((binding, found));
                }
                let layout = self.machine.variants[variant].layout.clone();
                let group = self.bind_group(&layout, (None, 0), resolved);
                self.frame_groups[variant].0[parity] = Some(group);
                self.parity = saved;
            }
            self.frame_groups[variant].1 = generation;
        }
        for pipeline in 0..self.machine.pipelines.len() {
            if self.own_groups[pipeline].1 == generation {
                continue;
            }
            let groups = self.machine.pipelines[pipeline].groups.len();
            self.pass = self.home_pass(pipeline);
            for group in 0..groups {
                for parity in 0..2 {
                    let saved = self.parity;
                    self.parity = parity;
                    let entries = self.machine.pipelines[pipeline].groups[group].entries.clone();
                    let mut resolved = Vec::with_capacity(entries.len());
                    for (binding, bound, ty) in entries {
                        let found = self.resource(&bound, ty, !ty.writes());
                        resolved.push((binding, found));
                    }
                    let layout = self.machine.pipelines[pipeline].groups[group].layout.clone();
                    let index = self.machine.pipelines[pipeline].groups[group].index;
                    let made = self.bind_group(&layout, (Some(pipeline), index), resolved);
                    self.own_groups[pipeline].0[group][parity] = Some(made);
                    self.parity = saved;
                }
            }
            self.own_groups[pipeline].1 = generation;
        }
        self.pass = saved_pass;
    }

    pub(super) fn scope_passes(&self, pipeline: usize) -> Vec<u32> {
        let Some(built) = self.machine.pipelines.get(pipeline) else {
            return Vec::new();
        };
        self.machine
            .declared
            .passes
            .iter()
            .enumerate()
            .filter(|(index, pass)| crate::machine::in_pass(built, *index as u32, pass))
            .map(|(index, _)| index as u32)
            .collect()
    }

    pub(super) fn home_pass(&self, pipeline: usize) -> u32 {
        self.scope_passes(pipeline).first().copied().unwrap_or(u32::MAX)
    }

    pub(super) fn frame_ty(&self, binding: u32) -> Ty {
        self.machine
            .frame
            .iter()
            .find(|(path, name, _)| self.machine.linker.binding_of(path, name) == Some(binding))
            .map_or(Ty::Uniform, |(_, _, ty)| *ty)
    }

    pub(super) fn core_frame(&mut self, binding: u32, ty: Ty) -> Resolved {
        let found = self
            .machine
            .frame
            .iter()
            .find(|(path, name, _)| self.machine.linker.binding_of(path, name) == Some(binding))
            .filter(|(path, _, _)| self.machine.linker.is_core(path));
        if let Some((_, name, _)) = found.filter(|(path, _, _)| path == crate::shaders::IMPOSTORS_MODULE) {
            return match self.assets.impostors.frame_resource(name) {
                Some(crate::impostors::FrameResource::View(view)) => Resolved::View(view.clone()),
                Some(crate::impostors::FrameResource::Sampler(sampler)) => Resolved::Sampler(sampler.clone()),
                None => Resolved::Placeholder(ty),
            };
        }
        let named = found.map(|(_, name, _)| name.clone());
        let target = match named.as_deref() {
            Some(crate::shared::POST) => return Resolved::Buffer(self.post_buffer.clone()),
            Some(crate::shared::LINEAR) => return Resolved::Sampler(self.linear.clone()),
            Some(name) => crate::machine::core_target(name, &self.machine.outputs),
            None => None,
        };
        match target.and_then(|target| self.view_of(target, false).cloned()) {
            Some(view) => Resolved::View(view),
            None => Resolved::Placeholder(ty),
        }
    }

    pub(crate) fn size_targets(&mut self, size: (u32, u32), canvas: Option<wgpu::TextureView>) {
        self.canvas = canvas;
        let extent = wgpu::Extent3d {
            width: size.0.max(1),
            height: size.1.max(1),
            depth_or_array_layers: 1,
        };
        if self.targets.first().is_some_and(|(_, made)| made.size == extent) {
            return;
        }
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
        let device = self.device.clone();
        let made = |label: &'static str, format| {
            let made = Made {
                label,
                format,
                usage,
                dimension: wgpu::TextureDimension::D2,
            };
            Allocated::new(&device, &made, extent, 1)
        };
        let mut targets = vec![
            (Target::Scene, made("engine/scene", crate::machine::SCENE_FORMAT)),
            (Target::Depth, made("engine/depth", DEPTH_FORMAT)),
        ];
        let mut previous = Vec::new();
        for (index, output) in self.machine.outputs.iter().enumerate() {
            targets.push((Target::Output(index as u32), made(output.name, output.format)));
            if output.layers.is_empty() {
                previous.push((Target::Output(index as u32), made(output.name, output.format)));
            }
        }
        for (target, _) in &self.machine.ids.targets {
            if matches!(target, Target::Copy(..))
                && let Some(format) = target_format(*target, &self.machine.outputs)
            {
                targets.push((*target, made("engine/view-copy", format)));
            }
        }
        self.targets = targets;
        self.previous_targets = previous;
        self.debug_targets = DEBUG_FORMATS.iter().map(|format| made("engine/debug-target", *format)).collect();
        self.generation += 1;
    }
}
