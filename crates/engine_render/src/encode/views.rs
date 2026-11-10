use super::*;

impl Run {
    pub(crate) fn view_offset(&self, view: Option<&OwnedDraw>) -> u32 {
        view.map_or(self.uniforms.camera_slot, |view| view.slot) * CAMERA_STRIDE as u32
    }

    pub(crate) fn write_cameras(&mut self) -> u64 {
        let bytes = self.uniforms.cameras.len() as u64 * CAMERA_STRIDE;
        if self.uniforms.camera.size() < bytes {
            self.uniforms.camera = crate::memory::buffer(
                &self.device,
                &wgpu::BufferDescriptor {
                    label: Some("engine/cameras"),
                    size: bytes.next_power_of_two(),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            );
            self.tallies.rebuilt.buffers += 1;
            self.touch(Held::Cameras);
        }
        let mut staged = std::mem::take(&mut self.spare.cameras);
        staged.clear();
        staged.resize(bytes as usize, 0);
        for (index, camera) in self.uniforms.cameras.iter().enumerate() {
            let at = index * CAMERA_STRIDE as usize;
            let raw = bytemuck::bytes_of(camera);
            staged[at..at + raw.len()].copy_from_slice(raw);
        }
        self.queue.write_buffer(&self.uniforms.camera, 0, &staged);
        self.spare.cameras = staged;
        size_of::<CameraUniform>() as u64
    }

    pub(crate) fn write_post(&mut self) {
        self.queue.write_buffer(
            &self.set().post_buffer,
            0,
            bytemuck::bytes_of(&self.uniforms.post),
        );
    }

    pub(crate) fn write_pieces(&mut self, staged: &mut Vec<u8>) -> u64 {
        let size = (staged.len() as u64)
            .max(crate::pieces::PIECE_ALIGN)
            .next_multiple_of(crate::pieces::PIECE_ALIGN);
        if self.uniforms.pieces.size() != size {
            self.uniforms.pieces = crate::memory::buffer(
                &self.device,
                &wgpu::BufferDescriptor {
                    label: Some("engine/pieces"),
                    size,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            );
            self.tallies.rebuilt.buffers += 1;
            self.touch(Held::Pieces);
        }
        staged.resize(size as usize, 0);
        self.queue.write_buffer(&self.uniforms.pieces, 0, staged);
        size
    }

    pub(crate) fn read_back(&mut self, asked: Asked) -> bool {
        let reader = Reader::Part {
            part: asked.part,
            tag: asked.tag,
        };
        let free = asked.once || self.reading.list.room(reader, self.reading.slots);
        if free {
            self.queued.asked.push(asked);
        }
        free
    }

    pub(crate) fn fill(&mut self, pipeline: u32, textures: &[TextureHandle]) -> u32 {
        for texture in textures {
            if !self.bindings.wanted_textures.contains(texture) {
                self.bindings.wanted_textures.push(*texture);
            }
        }
        let generation = self.assets.materials.generation();
        let base = self.machine.pipelines.len() as u32;
        if let Some(at) = self.filled.iter().position(|filled| {
            filled.base == pipeline
                && filled.textures == textures
                && filled.generation == generation
        }) {
            self.filled[at].used = self.frames.current;
            return base + at as u32;
        }
        self.filled.push(Filled {
            base: pipeline,
            textures: textures.to_vec(),
            generation,
            used: self.frames.current,
            groups: Vec::new(),
        });
        base + (self.filled.len() - 1) as u32
    }

    pub(crate) fn drop_unfilled(&mut self) {
        let frame = self.frames.current;
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

    pub(super) fn view_of(&self, target: Target) -> Option<&wgpu::TextureView> {
        match target {
            Target::Canvas => self.canvas.view.as_ref(),
            Target::Output(_) | Target::Scene | Target::Depth | Target::Copy(..) => {
                self.set().target(target)
            }
        }
    }

    pub(crate) fn texture_view(
        &mut self,
        bound: &Bound,
        ty: Ty,
        reading: bool,
    ) -> Option<wgpu::TextureView> {
        let parity = self.frames.parity;
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
                let made = self
                    .texture_tables_mut(*texture)
                    .textures
                    .get_mut(*texture as usize)?;
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
                self.view_of(wanted).cloned()
            }
            Bound::Buffer(_) | Bound::Window { .. } | Bound::Sampler(_) | Bound::Asset => None,
        }
    }

    pub(crate) fn pass_layer(&self) -> Option<engine_core::AttachPoint> {
        pass_layer(self.machine.declared.passes.get(self.pass as usize)?)
    }

    pub(super) fn placeholder(&mut self, key: (Ty, u64)) {
        if !self
            .bindings
            .placeholders
            .iter()
            .any(|(known, _)| *known == key)
        {
            let made = placeholder(&self.device, key);
            self.bindings.placeholders.push((key, made));
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
            Bound::Buffer(buffer) => match self.buffer_held(*buffer) {
                Some((made, _)) => Resolved::Buffer(made.clone()),
                None => Resolved::Placeholder(ty),
            },
            Bound::Window { buffer, size } => match self.buffer_held(*buffer) {
                Some((made, bytes)) if *bytes >= *size => Resolved::Dynamic(made.clone(), *size),
                Some(_) | None => Resolved::Placeholder(ty),
            },
            Bound::Sampler(sampler) => match self.bindings.samplers.get(*sampler as usize) {
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
                    Resolved::Placeholder(_) => match key.and_then(|key| {
                        self.bindings
                            .placeholders
                            .iter()
                            .find(|(known, _)| *known == key)
                    }) {
                        Some((_, Placeholder::Buffer(buffer))) => buffer.as_entire_binding(),
                        Some((_, Placeholder::Texture(view))) => {
                            wgpu::BindingResource::TextureView(view)
                        }
                        Some((_, Placeholder::Sampler(sampler))) => {
                            wgpu::BindingResource::Sampler(sampler)
                        }
                        None => self.uniforms.camera.as_entire_binding(),
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
        if self.bindings.stale.is_empty() && self.set().stale.is_empty() {
            return;
        }
        let saved_pass = self.pass;
        let saved_parity = self.frames.parity;
        let mut stale = std::mem::take(&mut self.bindings.stale);
        stale.append(&mut self.set_mut().stale);
        self.tallies.rebuilt.groups += stale.len() as u32;
        for holder in stale {
            match holder {
                Holder::Frame(variant) => self.rebuild_frame(variant as usize),
                Holder::Own(pipeline, group) => self.rebuild_own(pipeline as usize, group as usize),
            }
        }
        self.frames.parity = saved_parity;
        self.pass = saved_pass;
    }

    fn rebuild_frame(&mut self, variant: usize) {
        self.pass = u32::MAX;
        for parity in 0..2 {
            self.frames.parity = parity;
            let entries: Vec<(u32, wgpu::ShaderStages)> =
                self.machine.variants[variant].entries.clone();
            let mut resolved = Vec::with_capacity(entries.len());
            for (binding, _) in entries {
                let ty = self
                    .machine
                    .frame_slots
                    .get(&binding)
                    .map_or(Ty::Uniform, |slot| slot.ty);
                let found = match binding {
                    0 => Resolved::Dynamic(
                        self.uniforms.camera.clone(),
                        size_of::<CameraUniform>() as u64,
                    ),
                    1 => Resolved::Buffer(self.uniforms.pieces.clone()),
                    _ => match self.machine.frame_bound.get(&binding).cloned() {
                        Some(bound) => self.resource(&bound, ty, true),
                        None => self.core_frame(binding, ty),
                    },
                };
                resolved.push((binding, found));
            }
            let layout = self.machine.variants[variant].layout.clone();
            let group = self.bind_group(&layout, (None, 0), resolved);
            self.keep_group(Holder::Frame(variant as u32), parity, group);
        }
    }

    fn keep_group(&mut self, holder: Holder, parity: usize, group: wgpu::BindGroup) {
        let made = if self.sets.holders.contains(&holder) {
            self.set_mut().groups.entry(holder).or_default()
        } else {
            match holder {
                Holder::Frame(variant) => &mut self.bindings.frame_groups[variant as usize].made,
                Holder::Own(pipeline, at) => {
                    &mut self.bindings.own_groups[pipeline as usize][at as usize].made
                }
            }
        };
        made[parity] = Some(group);
    }

    fn rebuild_own(&mut self, pipeline: usize, group: usize) {
        self.pass = self.machine.pipelines[pipeline].home;
        for parity in 0..2 {
            self.frames.parity = parity;
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
            self.keep_group(Holder::Own(pipeline as u32, group as u32), parity, made);
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
            Some(CoreFrame::Post) => Resolved::Buffer(self.set().post_buffer.clone()),
            Some(CoreFrame::Linear) => Resolved::Sampler(self.uniforms.linear.clone()),
            Some(CoreFrame::Target(target)) => match self.view_of(*target) {
                Some(view) => Resolved::View(view.clone()),
                None => Resolved::Placeholder(ty),
            },
            None => Resolved::Placeholder(ty),
        }
    }

    pub(crate) fn size_targets(&mut self, size: (u32, u32), canvas: Option<wgpu::TextureView>) {
        self.canvas.view = canvas;
        let extent = wgpu::Extent3d {
            width: size.0.max(1),
            height: size.1.max(1),
            depth_or_array_layers: 1,
        };
        let split = self.canvas.rect.is_some();
        let set = self.set();
        let held = set.targets.iter().any(|(target, made)| {
            *target == Target::Canvas && made.texture.format() == self.canvas.format
        });
        let live: Vec<Target> = (0..self.machine.outputs.len())
            .filter(|&index| self.planning.live.get(index).copied().unwrap_or(true))
            .map(|index| Target::Output(index as u32))
            .collect();
        let kept: Vec<Target> = set
            .targets
            .iter()
            .map(|(target, _)| *target)
            .filter(|target| matches!(target, Target::Output(_)))
            .collect();
        if live == kept
            && split == held
            && set
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
        for target in live {
            if let Target::Output(index) = target
                && let Some(output) = self.machine.outputs.get(index as usize)
            {
                targets.push((target, made(output.name, output.format)));
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
            targets.push((
                Target::Canvas,
                made("engine/view-canvas", self.canvas.format),
            ));
        }
        let set = self.set_mut();
        set.targets = targets;
        self.tallies.rebuilt.textures += 1;
        self.touch(Held::Targets);
    }

    pub(crate) fn settle_sets(&mut self, listed: &[bevy_ecs::entity::Entity]) {
        let main = listed
            .first()
            .copied()
            .unwrap_or(bevy_ecs::entity::Entity::PLACEHOLDER);
        if !self.sets.list.iter().any(|set| set.view == main)
            && let Some(set) = self
                .sets
                .list
                .iter_mut()
                .find(|set| !listed.contains(&set.view))
        {
            set.view = main;
        }
        let machine = &self.machine;
        self.sets.list.retain_mut(|set| {
            let kept = listed.contains(&set.view);
            if !kept {
                set.release(machine, None);
            }
            kept
        });
        self.sets.at = usize::MAX;
        self.select(main);
    }

    pub(crate) fn select(&mut self, view: bevy_ecs::entity::Entity) -> usize {
        let at = match self.sets.list.iter().position(|set| set.view == view) {
            Some(at) => at,
            None => {
                let set = ViewSet::new(&self.device, &self.machine, view, &self.sets.holders);
                self.sets.list.push(set);
                self.sets.list.len() - 1
            }
        };
        if at != self.sets.at {
            let holders = &self.sets.holders;
            for filled in &mut self.filled {
                let held = holders.iter().any(
                    |holder| matches!(holder, Holder::Own(pipeline, _) if *pipeline == filled.base),
                );
                if held {
                    filled.groups.clear();
                }
            }
            self.sets.at = at;
        }
        at
    }

    pub(crate) fn carry(&mut self, from: usize) -> bool {
        let at = self.sets.at;
        let (Some(source), Some(current)) = (self.sets.list.get(from), self.sets.list.get(at))
        else {
            return false;
        };
        if from == at {
            return false;
        }
        let pairs: Vec<(wgpu::Texture, wgpu::Texture, wgpu::Extent3d)> =
            [Target::Scene, Target::Depth]
                .iter()
                .filter_map(|target| {
                    let from = source.targets.iter().find(|(known, _)| known == target)?;
                    let to = current.targets.iter().find(|(known, _)| known == target)?;
                    (from.1.size == to.1.size)
                        .then(|| (from.1.texture.clone(), to.1.texture.clone(), to.1.size))
                })
                .collect();
        if pairs.len() != 2 {
            return false;
        }
        let Some(encoder) = self.recording.encoder() else {
            return false;
        };
        for (from, to, size) in pairs {
            encoder.copy_texture_to_texture(from.as_image_copy(), to.as_image_copy(), size);
        }
        true
    }

    pub(crate) fn copy_canvas(&mut self, inward: bool) {
        let (Some(rect), Some(canvas)) = (self.canvas.rect, self.canvas.texture.as_ref()) else {
            return;
        };
        let Some((_, held)) = self.sets.list[self.sets.at]
            .targets
            .iter()
            .find(|(target, _)| *target == Target::Canvas)
        else {
            return;
        };
        let Some(encoder) = self.recording.encoder() else {
            return;
        };
        let placed = wgpu::TexelCopyTextureInfo {
            texture: canvas,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: rect[0],
                y: rect[1],
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        };
        let (from, to) = if inward {
            (placed, held.texture.as_image_copy())
        } else {
            (held.texture.as_image_copy(), placed)
        };
        encoder.copy_texture_to_texture(from, to, held.size);
    }

    pub(crate) fn view_canvas(&self) -> Option<wgpu::TextureView> {
        self.set()
            .target(Target::Canvas)
            .cloned()
            .or_else(|| self.canvas.view.clone())
    }
}

#[derive(Default)]
pub(crate) struct Tables {
    pub(crate) textures: Vec<[Option<Allocated>; 2]>,
    pub(crate) sizes: Vec<Option<(wgpu::Extent3d, u32)>>,
    pub(crate) buffers: Vec<Option<(wgpu::Buffer, u64)>>,
}

impl Tables {
    pub(crate) fn of(machine: &Machine) -> Self {
        Self {
            textures: machine
                .declared
                .textures
                .iter()
                .map(|_| [None, None])
                .collect(),
            sizes: machine.declared.textures.iter().map(|_| None).collect(),
            buffers: machine.declared.buffers.iter().map(|_| None).collect(),
        }
    }
}

pub(crate) struct ViewSet {
    pub(crate) view: bevy_ecs::entity::Entity,
    pub(crate) targets: Vec<(Target, Allocated)>,
    pub(crate) tables: Tables,
    pub(crate) post_buffer: wgpu::Buffer,
    pub(crate) groups: std::collections::BTreeMap<Holder, [Option<wgpu::BindGroup>; 2]>,
    pub(crate) stale: std::collections::BTreeSet<Holder>,
    pub(crate) ran: Bits,
}

impl ViewSet {
    pub(crate) fn new(
        device: &wgpu::Device,
        machine: &Machine,
        view: bevy_ecs::entity::Entity,
        holders: &std::collections::BTreeSet<Holder>,
    ) -> Self {
        Self {
            view,
            targets: Vec::new(),
            tables: Tables::of(machine),
            post_buffer: uniform_buffer(
                device,
                "engine/post-uniform",
                size_of::<crate::shared::PostUniform>() as u64,
            ),
            groups: std::collections::BTreeMap::new(),
            stale: holders.clone(),
            ran: Bits::of(machine.graph.passes()),
        }
    }

    pub(crate) fn target(&self, target: Target) -> Option<&wgpu::TextureView> {
        self.targets
            .iter()
            .find(|(known, _)| *known == target)
            .map(|(_, made)| &made.view)
    }

    pub(crate) fn release(&mut self, machine: &Machine, part: Option<u32>) {
        let owned = |owner: u32| part.is_none_or(|part| owner == part);
        for (texture, made) in self.tables.textures.iter_mut().enumerate() {
            if owned(machine.declared.textures[texture].part) {
                for held in made.iter().flatten() {
                    crate::memory::release_any(&held.texture);
                }
                *made = [None, None];
                self.tables.sizes[texture] = None;
            }
        }
        for (buffer, made) in self.tables.buffers.iter_mut().enumerate() {
            if owned(machine.declared.buffers[buffer].part)
                && let Some((held, _)) = made.take()
            {
                crate::memory::release_buffer(&held);
            }
        }
        self.stale.extend(self.groups.keys().copied());
    }
}

pub(crate) struct Sets {
    pub(crate) list: Vec<ViewSet>,
    pub(crate) at: usize,
    pub(crate) holders: std::collections::BTreeSet<Holder>,
    pub(crate) passes: Bits,
    pub(crate) main: Bits,
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

fn placeholder(device: &wgpu::Device, (ty, bytes): (Ty, u64)) -> Placeholder {
    use crate::linker::Sample;
    let buffer = |usage| {
        Placeholder::Buffer(crate::memory::buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("engine/placeholder"),
                size: bytes.max(PLACEHOLDER_BYTES),
                usage,
                mapped_at_creation: false,
            },
        ))
    };
    match ty {
        Ty::Uniform => buffer(wgpu::BufferUsages::UNIFORM),
        Ty::Storage { .. } => buffer(wgpu::BufferUsages::STORAGE),
        Ty::Sampler => Placeholder::Sampler(crate::materials::gpu_sampler(
            device,
            "engine/placeholder",
            engine_core::Sampler::NEAREST,
        )),
        Ty::Comparison => Placeholder::Sampler(crate::materials::gpu_sampler(
            device,
            "engine/placeholder",
            engine_core::Sampler {
                compare: Some(engine_core::Compare::LessEqual),
                ..engine_core::Sampler::NEAREST
            },
        )),
        Ty::Texture { sample, view } => {
            let format = match sample {
                Sample::Float => wgpu::TextureFormat::Rgba8Unorm,
                Sample::Depth => DEPTH_FORMAT,
                Sample::Uint => wgpu::TextureFormat::R32Uint,
                Sample::Sint => wgpu::TextureFormat::R32Sint,
            };
            placeholder_texture(device, format, view, wgpu::TextureUsages::TEXTURE_BINDING)
        }
        Ty::StorageTexture { format, view, .. } => {
            placeholder_texture(device, format, view, wgpu::TextureUsages::STORAGE_BINDING)
        }
    }
}

fn placeholder_texture(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    view: wgpu::TextureViewDimension,
    usage: wgpu::TextureUsages,
) -> Placeholder {
    let layers = match view {
        wgpu::TextureViewDimension::Cube | wgpu::TextureViewDimension::CubeArray => 6,
        wgpu::TextureViewDimension::D1
        | wgpu::TextureViewDimension::D2
        | wgpu::TextureViewDimension::D2Array
        | wgpu::TextureViewDimension::D3 => 1,
    };
    let dimension = match view {
        wgpu::TextureViewDimension::D3 => wgpu::TextureDimension::D3,
        wgpu::TextureViewDimension::D1 => wgpu::TextureDimension::D1,
        wgpu::TextureViewDimension::D2
        | wgpu::TextureViewDimension::D2Array
        | wgpu::TextureViewDimension::Cube
        | wgpu::TextureViewDimension::CubeArray => wgpu::TextureDimension::D2,
    };
    let texture = crate::memory::texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("engine/placeholder"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format,
            usage,
            view_formats: &[],
        },
    );
    Placeholder::Texture(texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(view),
        ..Default::default()
    }))
}
