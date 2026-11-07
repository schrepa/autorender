use super::*;

impl Run {
    pub(crate) fn cameras(&mut self, views: &[CameraUniform]) {
        let base = self.cameras.len() as u32;
        self.cameras.extend_from_slice(views);
        self.views.push((self.part, base, views.len() as u32));
    }

    pub(crate) fn drop_cameras(&mut self, main: &[CameraUniform]) {
        self.cameras.clear();
        self.cameras.extend_from_slice(main);
        self.views.clear();
    }

    pub(crate) fn view_offset(&self, part: u32, view: Option<u32>) -> Option<u32> {
        let Some(view) = view else {
            return Some(self.camera_slot * CAMERA_STRIDE as u32);
        };
        let (_, base, count) = self.views.iter().rev().find(|(known, _, _)| *known == part)?;
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
            self.rebuilt.buffers += 1;
            self.touch(Held::Cameras);
        }
        let mut staged = std::mem::take(&mut self.spare.cameras);
        staged.clear();
        staged.resize(bytes as usize, 0);
        for (index, camera) in self.cameras.iter().enumerate() {
            let at = index * CAMERA_STRIDE as usize;
            let raw = bytemuck::bytes_of(camera);
            staged[at..at + raw.len()].copy_from_slice(raw);
        }
        self.queue.write_buffer(&self.camera, 0, &staged);
        self.spare.cameras = staged;
        size_of::<CameraUniform>() as u64
    }

    pub(crate) fn write_post(&mut self) {
        self.queue
            .write_buffer(&self.post_buffer, 0, bytemuck::bytes_of(&self.post));
    }

    pub(crate) fn write_pieces(&mut self, staged: &mut Vec<u8>) -> u64 {
        let size = (staged.len() as u64)
            .max(crate::pieces::PIECE_ALIGN)
            .next_multiple_of(crate::pieces::PIECE_ALIGN);
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
            self.rebuilt.buffers += 1;
            self.touch(Held::Pieces);
        }
        staged.resize(size as usize, 0);
        self.queue.write_buffer(&self.pieces, 0, staged);
        size
    }

    pub(crate) fn read_back(&mut self, asked: Asked) -> bool {
        let reader = Reader::Part {
            part: asked.part,
            tag: asked.tag,
        };
        let free = asked.once || self.readbacks.room(reader, self.readback_slots);
        if free {
            self.queued.asked.push(asked);
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
        if let Some(at) = self.filled.iter().position(|filled| {
            filled.base == pipeline
                && filled.textures == textures
                && filled.generation == generation
        }) {
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
            reads_depth_direct(
                pass,
                self.machine
                    .depth_writes
                    .get(index)
                    .copied()
                    .unwrap_or(true),
            )
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
        let list = if second {
            &self.previous_targets
        } else {
            &self.targets
        };
        list.iter()
            .find(|(known, _)| *known == target)
            .map(|(_, made)| &made.view)
    }

    pub(super) fn view_of(&self, target: Target, writing: bool) -> Option<&wgpu::TextureView> {
        match target {
            Target::Canvas => self.canvas.as_ref(),
            Target::Debug(index) => self
                .debug_targets
                .get(index as usize)
                .map(|made| &made.view),
            Target::Output(index) => self.output_view(index, writing),
            Target::Scene | Target::Depth | Target::Copy(..) => self
                .targets
                .iter()
                .find(|(known, _)| *known == target)
                .map(|(_, made)| &made.view),
        }
    }

    pub(crate) fn texture_view(
        &mut self,
        bound: &Bound,
        ty: Ty,
        reading: bool,
    ) -> Option<wgpu::TextureView> {
        let parity = self.parity;
        let dimension = match ty {
            Ty::Texture { view, .. } | Ty::StorageTexture { view, .. } => Some(view),
            Ty::Uniform | Ty::Storage { .. } | Ty::Sampler | Ty::Comparison => None,
        };
        match bound {
            Bound::Texture(texture) | Bound::Slice { texture, .. } => {
                let index = if self.doubled(*texture) && reading {
                    1 - parity
                } else {
                    parity
                };
                let made = self.textures.get_mut(*texture as usize)?;
                let made = match index {
                    1 if made[1].is_some() => made[1].as_mut()?,
                    _ => made[0].as_mut()?,
                };
                let (mips, layers) = match bound {
                    Bound::Slice { mips, layers, .. } => (mips.clone(), layers.clone()),
                    Bound::Texture(_)
                    | Bound::Buffer(_)
                    | Bound::Window { .. }
                    | Bound::Sampler(_)
                    | Bound::Target(_)
                    | Bound::Asset => (0..made.mips, 0..made.layers()),
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

    pub(crate) fn pass_layer(&self) -> Option<engine_core::AttachPoint> {
        pass_layer(self.machine.declared.passes.get(self.pass as usize)?)
    }

    pub(super) fn placeholder(&mut self, key: (Ty, u64)) {
        if !self.placeholders.iter().any(|(known, _)| *known == key) {
            let made = placeholder(&self.device, key);
            self.placeholders.push((key, made));
        }
    }

    pub(super) fn binding_bytes(&self, pipeline: Option<usize>, group: u32, binding: u32) -> u64 {
        let needed = self.machine.binding_bytes.get(&(pipeline, group, binding));
        needed
            .copied()
            .unwrap_or(0)
            .max(PLACEHOLDER_BYTES)
            .next_multiple_of(16)
    }

    pub(super) fn resource(&mut self, bound: &Bound, ty: Ty, reading: bool) -> Resolved {
        match bound {
            Bound::Buffer(buffer) => {
                match self.buffers.get(*buffer as usize).and_then(Option::as_ref) {
                    Some((made, _)) => Resolved::Buffer(made.clone()),
                    None => Resolved::Placeholder(ty),
                }
            }
            Bound::Window { buffer, size } => {
                match self.buffers.get(*buffer as usize).and_then(Option::as_ref) {
                    Some((made, bytes)) if *bytes >= *size => {
                        Resolved::Dynamic(made.clone(), *size)
                    }
                    Some(_) | None => Resolved::Placeholder(ty),
                }
            }
            Bound::Sampler(sampler) => match self.samplers.get(*sampler as usize) {
                Some(made) => Resolved::Sampler(made.clone()),
                None => Resolved::Placeholder(ty),
            },
            Bound::Texture(_) | Bound::Slice { .. } | Bound::Target(_) => {
                match self.texture_view(bound, ty, reading) {
                    Some(view) => Resolved::View(view),
                    None => Resolved::Placeholder(ty),
                }
            }
            Bound::Asset => Resolved::Placeholder(ty),
        }
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "the machine binds groups by the linked declarations"
    )]
    pub(super) fn bind_group(
        &mut self,
        layout: &wgpu::BindGroupLayout,
        (pipeline, group): (Option<usize>, u32),
        entries: Vec<(u32, Resolved)>,
    ) -> wgpu::BindGroup {
        let mut keys: Vec<Option<(Ty, u64)>> = Vec::with_capacity(entries.len());
        for (binding, resolved) in &entries {
            let key = match resolved {
                Resolved::Placeholder(ty @ (Ty::Uniform | Ty::Storage { .. })) => {
                    Some((*ty, self.binding_bytes(pipeline, group, *binding)))
                }
                Resolved::Placeholder(ty) => Some((*ty, 0)),
                Resolved::Buffer(_)
                | Resolved::Dynamic(..)
                | Resolved::View(_)
                | Resolved::Sampler(_) => None,
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
                    Resolved::Dynamic(buffer, size) => {
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: 0,
                            size: wgpu::BufferSize::new(*size),
                        })
                    }
                    Resolved::View(view) => wgpu::BindingResource::TextureView(view),
                    Resolved::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
                    Resolved::Placeholder(_) => match key
                        .and_then(|key| self.placeholders.iter().find(|(known, _)| *known == key))
                    {
                        Some((_, Placeholder::Buffer(buffer))) => buffer.as_entire_binding(),
                        Some((_, Placeholder::Texture(view))) => {
                            wgpu::BindingResource::TextureView(view)
                        }
                        Some((_, Placeholder::Sampler(sampler))) => {
                            wgpu::BindingResource::Sampler(sampler)
                        }
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
            self.touch(Held::Atlas);
        }
        if self.stale.is_empty() {
            return;
        }
        let saved_pass = self.pass;
        let saved_parity = self.parity;
        let stale = std::mem::take(&mut self.stale);
        self.rebuilt.groups += stale.len() as u32;
        for holder in stale {
            match holder {
                Holder::Frame(variant) => self.rebuild_frame(variant as usize),
                Holder::Own(pipeline, group) => self.rebuild_own(pipeline as usize, group as usize),
            }
        }
        self.parity = saved_parity;
        self.pass = saved_pass;
    }

    fn rebuild_frame(&mut self, variant: usize) {
        self.pass = u32::MAX;
        for parity in 0..2 {
            self.parity = parity;
            let entries: Vec<(u32, crate::bindings::Stages)> =
                self.machine.variants[variant].entries.clone();
            let mut resolved = Vec::with_capacity(entries.len());
            for (binding, _) in entries {
                let ty = self
                    .machine
                    .frame_slots
                    .get(&binding)
                    .map_or(Ty::Uniform, |slot| slot.ty);
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
            self.frame_groups[variant].made[parity] = Some(group);
        }
    }

    fn rebuild_own(&mut self, pipeline: usize, group: usize) {
        self.pass = self.machine.pipelines[pipeline].home;
        for parity in 0..2 {
            self.parity = parity;
            let entries = self.machine.pipelines[pipeline].groups[group]
                .entries
                .clone();
            let mut resolved = Vec::with_capacity(entries.len());
            for (binding, bound, ty) in entries {
                let found = self.resource(&bound, ty, !ty.writes());
                resolved.push((binding, found));
            }
            let layout = self.machine.pipelines[pipeline].groups[group]
                .layout
                .clone();
            let index = self.machine.pipelines[pipeline].groups[group].index;
            let made = self.bind_group(&layout, (Some(pipeline), index), resolved);
            self.own_groups[pipeline][group].made[parity] = Some(made);
        }
        for filled in &mut self.filled {
            if filled.base as usize == pipeline {
                filled.groups.clear();
            }
        }
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

    pub(super) fn core_frame(&mut self, binding: u32, ty: Ty) -> Resolved {
        let core = self
            .machine
            .frame_slots
            .get(&binding)
            .and_then(|slot| slot.core.as_ref());
        match core {
            Some(CoreFrame::Impostor(name)) => match self.assets.impostors.frame_resource(name) {
                Some(crate::impostors::FrameResource::View(view)) => Resolved::View(view.clone()),
                Some(crate::impostors::FrameResource::Sampler(sampler)) => {
                    Resolved::Sampler(sampler.clone())
                }
                None => Resolved::Placeholder(ty),
            },
            Some(CoreFrame::Post) => Resolved::Buffer(self.post_buffer.clone()),
            Some(CoreFrame::Linear) => Resolved::Sampler(self.linear.clone()),
            Some(CoreFrame::Target(target)) => match self.view_of(*target, false) {
                Some(view) => Resolved::View(view.clone()),
                None => Resolved::Placeholder(ty),
            },
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
        let split = self.canvas_rect.is_some();
        let held = self
            .targets
            .iter()
            .any(|(target, made)| *target == Target::Canvas && made.texture.format() == self.canvas_format);
        if self.debug != self.debug_targets.is_empty()
            && split == held
            && self
                .targets
                .first()
                .is_some_and(|(_, made)| made.size == extent)
        {
            return;
        }
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST;
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
            (
                Target::Scene,
                made("engine/scene", crate::machine::SCENE_FORMAT),
            ),
            (Target::Depth, made("engine/depth", DEPTH_FORMAT)),
        ];
        let mut previous = Vec::new();
        for (index, output) in self.machine.outputs.iter().enumerate() {
            targets.push((
                Target::Output(index as u32),
                made(output.name, output.format),
            ));
            if output.layers.is_empty() {
                previous.push((
                    Target::Output(index as u32),
                    made(output.name, output.format),
                ));
            }
        }
        for (target, _) in &self.machine.ids.targets {
            if matches!(target, Target::Copy(..))
                && let Some(format) = target_format(*target, &self.machine.outputs)
            {
                targets.push((*target, made("engine/view-copy", format)));
            }
        }
        if split {
            targets.push((Target::Canvas, made("engine/view-canvas", self.canvas_format)));
        }
        self.targets = targets;
        self.previous_targets = previous;
        self.debug_targets = DEBUG_FORMATS
            .iter()
            .filter(|_| self.debug)
            .map(|format| made("engine/debug-target", *format))
            .collect();
        self.rebuilt.textures += 1;
        self.touch(Held::Targets);
    }

    pub(crate) fn park(&mut self, view: bevy_ecs::entity::Entity) -> Option<usize> {
        if self.resident.is_none_or(|resident| resident == view) {
            self.resident = Some(view);
            return None;
        }
        if let Some(at) = self.parked.iter().position(|state| state.view == view) {
            return Some(at);
        }
        let (textures, buffers) = view_resources(&self.machine);
        self.parked.push(ViewState {
            view,
            targets: Vec::new(),
            previous_targets: Vec::new(),
            debug_targets: Vec::new(),
            textures: textures.iter().map(|_| [None, None]).collect(),
            sizes: textures.iter().map(|_| None).collect(),
            buffers: buffers.iter().map(|_| None).collect(),
            post_buffer: uniform_buffer(
                &self.device,
                "engine/post-uniform",
                size_of::<crate::shared::PostUniform>() as u64,
            ),
            groups: self
                .view_holders
                .iter()
                .map(|holder| (*holder, [None, None]))
                .collect(),
            stale: self.view_holders.clone(),
            ran: Bits::of(self.machine.graph.passes()),
        });
        Some(self.parked.len() - 1)
    }

    pub(crate) fn settle_views(&mut self, main: bevy_ecs::entity::Entity, listed: &[bevy_ecs::entity::Entity]) {
        if self.resident.is_some_and(|resident| resident != main)
            && let Some(at) = self.parked.iter().position(|state| state.view == main)
        {
            self.swap_view(at);
            if let (Some(state), Some(resident)) = (self.parked.get_mut(at), self.resident) {
                state.view = resident;
            }
        }
        self.resident = Some(main);
        let machine = &self.machine;
        self.parked.retain_mut(|state| {
            let kept = listed.contains(&state.view);
            if !kept {
                state.release(machine);
            }
            kept
        });
    }

    pub(crate) fn swap_view(&mut self, at: usize) {
        let mut parked = std::mem::take(&mut self.parked);
        if let Some(state) = parked.get_mut(at) {
            self.exchange(state);
        }
        self.parked = parked;
    }

    fn exchange(&mut self, state: &mut ViewState) {
        std::mem::swap(&mut self.targets, &mut state.targets);
        std::mem::swap(&mut self.previous_targets, &mut state.previous_targets);
        std::mem::swap(&mut self.debug_targets, &mut state.debug_targets);
        std::mem::swap(&mut self.post_buffer, &mut state.post_buffer);
        std::mem::swap(&mut self.ran, &mut state.ran);
        let (textures, buffers) = view_resources(&self.machine);
        for (slot, texture) in textures.into_iter().enumerate() {
            std::mem::swap(&mut self.textures[texture as usize], &mut state.textures[slot]);
            std::mem::swap(&mut self.sizes[texture as usize], &mut state.sizes[slot]);
        }
        for (slot, buffer) in buffers.into_iter().enumerate() {
            std::mem::swap(&mut self.buffers[buffer as usize], &mut state.buffers[slot]);
        }
        for (holder, made) in &mut state.groups {
            let held = match *holder {
                Holder::Frame(variant) => &mut self.frame_groups[variant as usize].made,
                Holder::Own(pipeline, group) => {
                    &mut self.own_groups[pipeline as usize][group as usize].made
                }
            };
            std::mem::swap(held, made);
        }
        let incoming = std::mem::take(&mut state.stale);
        for holder in &self.view_holders {
            if self.stale.remove(holder) {
                state.stale.insert(*holder);
            }
        }
        self.stale.extend(incoming);
        let view_holders = &self.view_holders;
        for filled in &mut self.filled {
            let held = view_holders
                .iter()
                .any(|holder| matches!(holder, Holder::Own(pipeline, _) if *pipeline == filled.base));
            if held {
                filled.groups.clear();
            }
        }
    }

    pub(crate) fn carry(&mut self, from: Option<usize>, current: Option<usize>) -> bool {
        let source = match (from, current) {
            (Some(at), _) | (None, Some(at)) => match self.parked.get(at) {
                Some(state) => &state.targets,
                None => return false,
            },
            (None, None) => return false,
        };
        let pairs: Vec<(&Allocated, &Allocated)> = [Target::Scene, Target::Depth]
            .iter()
            .filter_map(|target| {
                let from = source.iter().find(|(known, _)| known == target)?;
                let to = self.targets.iter().find(|(known, _)| known == target)?;
                (from.1.size == to.1.size).then_some((&from.1, &to.1))
            })
            .collect();
        if pairs.len() != 2 {
            return false;
        }
        let Some(encoder) = self.recording.encoder() else {
            return false;
        };
        for (from, to) in pairs {
            encoder.copy_texture_to_texture(
                from.texture.as_image_copy(),
                to.texture.as_image_copy(),
                to.size,
            );
        }
        true
    }

    pub(crate) fn canvas_in(&mut self) {
        let (Some(rect), Some(canvas)) = (self.canvas_rect, self.canvas_texture.as_ref()) else {
            return;
        };
        let Some((_, held)) = self.targets.iter().find(|(target, _)| *target == Target::Canvas)
        else {
            return;
        };
        let Some(encoder) = self.recording.encoder() else {
            return;
        };
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: canvas,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: rect[0],
                    y: rect[1],
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            held.texture.as_image_copy(),
            held.size,
        );
    }

    pub(crate) fn canvas_out(&mut self) {
        let (Some(rect), Some(canvas)) = (self.canvas_rect, self.canvas_texture.as_ref()) else {
            return;
        };
        let Some((_, held)) = self.targets.iter().find(|(target, _)| *target == Target::Canvas)
        else {
            return;
        };
        let Some(encoder) = self.recording.encoder() else {
            return;
        };
        encoder.copy_texture_to_texture(
            held.texture.as_image_copy(),
            wgpu::TexelCopyTextureInfo {
                texture: canvas,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: rect[0],
                    y: rect[1],
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            held.size,
        );
    }

    pub(crate) fn view_canvas(&self) -> Option<wgpu::TextureView> {
        self.targets
            .iter()
            .find(|(target, _)| *target == Target::Canvas)
            .map(|(_, made)| made.view.clone())
            .or_else(|| self.canvas.clone())
    }
}

pub(crate) struct ViewState {
    pub(crate) view: bevy_ecs::entity::Entity,
    targets: Vec<(Target, Allocated)>,
    previous_targets: Vec<(Target, Allocated)>,
    debug_targets: Vec<Allocated>,
    textures: Vec<[Option<Allocated>; 2]>,
    sizes: Vec<Option<(wgpu::Extent3d, u32)>>,
    buffers: Vec<Option<(wgpu::Buffer, u64)>>,
    post_buffer: wgpu::Buffer,
    groups: Vec<(Holder, [Option<wgpu::BindGroup>; 2])>,
    pub(crate) stale: std::collections::BTreeSet<Holder>,
    pub(crate) ran: Bits,
}

impl ViewState {
    pub(crate) fn release_part(&mut self, machine: &Machine, part: u32) {
        let (textures, buffers) = view_resources(machine);
        for (slot, texture) in textures.into_iter().enumerate() {
            if machine.declared.textures[texture as usize].part == part {
                for made in self.textures[slot].iter().flatten() {
                    crate::memory::release_any(&made.texture);
                }
                self.textures[slot] = [None, None];
                self.sizes[slot] = None;
            }
        }
        for (slot, buffer) in buffers.into_iter().enumerate() {
            if machine.declared.buffers[buffer as usize].part == part
                && let Some((made, _)) = self.buffers[slot].take()
            {
                crate::memory::release_buffer(&made);
            }
        }
        self.stale
            .extend(self.groups.iter().map(|(holder, _)| *holder));
    }

    fn release(&mut self, machine: &Machine) {
        for part in 0..machine.declared.parts.len() as u32 {
            self.release_part(machine, part);
        }
    }
}

fn view_resources(machine: &Machine) -> (Vec<u32>, Vec<u32>) {
    let textures = machine
        .declared
        .textures
        .iter()
        .enumerate()
        .filter(|(_, texture)| texture.per_view)
        .map(|(index, _)| index as u32)
        .collect();
    let buffers = machine
        .declared
        .buffers
        .iter()
        .enumerate()
        .filter(|(_, buffer)| buffer.per_view)
        .map(|(index, _)| index as u32)
        .collect();
    (textures, buffers)
}

pub(super) fn view_passes(machine: &Machine) -> (Bits, Bits) {
    let graph = &machine.graph;
    let declared = &machine.declared.passes;
    let mut passes = Bits::of(graph.passes());
    let mut held = Bits::of(graph.passes());
    for (index, pass) in declared.iter().enumerate() {
        if pass.views == engine_core::PassViews::Owned {
            held.insert(index);
        } else if pass.per_view() {
            passes.insert(index);
        }
    }
    for index in declared.len()..graph.passes() {
        passes.insert(index);
    }
    let mut varying = graph.view_resources();
    let (textures, buffers) = view_resources(machine);
    for texture in textures {
        if let Some(id) = machine.ids.textures.get(texture as usize) {
            varying.insert(id.index());
        }
    }
    for buffer in buffers {
        if let Some(id) = machine.ids.buffers.get(buffer as usize) {
            varying.insert(id.index());
        }
    }
    let spread = graph.spread(passes.clone(), varying, &held);
    let mut main = Bits::of(graph.passes());
    for pass in spread.iter().filter(|&pass| !passes.contains(pass)) {
        main.insert(pass);
    }
    (passes, main)
}
