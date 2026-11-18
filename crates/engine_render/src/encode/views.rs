use super::*;

impl Run {
    pub(crate) fn view_offset(&self) -> u32 {
        self.uniforms.camera_slot * CAMERA_STRIDE as u32
    }

    pub(crate) fn write_cameras(&mut self) {
        let bytes = self.uniforms.cameras.len() as u64 * CAMERA_STRIDE;
        if self.uniforms.camera.size() < bytes {
            self.uniforms.camera = crate::gpu_calls::create_buffer(&self.device, &wgpu::BufferDescriptor {
                label: Some("engine/cameras"),
                size: bytes.next_power_of_two(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            engine_profile::event(
                "rebuilt",
                "engine/cameras",
                &[("why", engine_profile::Fact::Text("views"))],
            );
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
    }

    pub(crate) fn write_post(&mut self) {
        self.queue.write_buffer(
            &self.set().post_buffer,
            0,
            bytemuck::bytes_of(&self.uniforms.post),
        );
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
        let generation = self.assets.textures.generation();
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

    pub(super) fn view_of(&self, target: Target) -> Option<&wgpu::TextureView> {
        match target {
            Target::Canvas => self.canvas.view.as_ref(),
            Target::Shared(_) => self.set().target(target),
        }
    }

    pub(crate) fn texture_view(&mut self, bound: &Bound, ty: Ty) -> Option<wgpu::TextureView> {
        let dimension = match ty {
            Ty::Texture { view, .. } | Ty::StorageTexture { view, .. } => Some(view),
            Ty::Uniform | Ty::Storage { .. } | Ty::Sampler | Ty::Comparison => None,
        };
        match bound {
            Bound::Texture(texture) | Bound::Slice { texture, .. } => {
                let made = self
                    .texture_tables_mut(*texture)
                    .textures
                    .get_mut(*texture as usize)?
                    .as_mut()?;
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
            Bound::Target(name) => self
                .machine
                .declared
                .target_of(*name)
                .and_then(|target| self.view_of(target))
                .cloned(),
            Bound::Buffer(_) | Bound::Window { .. } | Bound::Sampler(_) | Bound::Asset => None,
        }
    }

    pub(super) fn placeholder(&mut self, key: (Ty, u64)) {
        if !self
            .bindings
            .placeholders
            .iter()
            .any(|(known, _)| *known == key)
            && let Some(made) = placeholder(&self.device, key.0)
        {
            self.bindings.placeholders.push((key, made));
        }
    }

    pub(super) fn resource(&mut self, bound: &Bound, ty: Ty) -> Resolved {
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
                match self.texture_view(bound, ty) {
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
        entries: Vec<(u32, Resolved)>,
    ) -> Option<wgpu::BindGroup> {
        let mut keys: Vec<Option<(Ty, u64)>> = Vec::with_capacity(entries.len());
        for (_, resolved) in &entries {
            let key = match resolved {
                Resolved::Placeholder(Ty::Uniform | Ty::Storage { .. }) => return None,
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
        Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine/bound-group"),
            layout,
            entries: &resources,
        }))
    }

    pub(crate) fn refresh_groups(&mut self) {
        if self.bindings.stale.is_empty() && self.set().stale.is_empty() {
            return;
        }
        let saved_pass = self.pass;
        let saved_parity = self.frames.parity;
        let mut stale = std::mem::take(&mut self.bindings.stale);
        stale.append(&mut self.set_mut().stale);
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
                    _ => match self.machine.frame_bound.get(&binding).cloned() {
                        Some(bound) => self.resource(&bound, ty),
                        None => self.core_frame(binding, ty),
                    },
                };
                resolved.push((binding, found));
            }
            let layout = self.machine.variants[variant].layout.clone();
            let group = self.bind_group(&layout, resolved);
            self.keep_group(Holder::Frame(variant as u32), parity, group);
        }
    }

    fn keep_group(&mut self, holder: Holder, parity: usize, group: Option<wgpu::BindGroup>) {
        let made = if self.sets.holders.contains(&holder) {
            let index = self.bindings.holders.index(holder) as usize;
            let groups = &mut self.sets.list[self.sets.at].groups;
            if groups.len() <= index {
                groups.resize_with(index + 1, || None);
            }
            &mut groups[index].get_or_insert_with(|| (holder, [None, None])).1
        } else {
            match holder {
                Holder::Frame(variant) => &mut self.bindings.frame_groups[variant as usize].made,
                Holder::Own(pipeline, at) => {
                    &mut self.bindings.own_groups[pipeline as usize][at as usize].made
                }
            }
        };
        made[parity] = group;
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
                let found = self.resource(&bound, ty);
                resolved.push((binding, found));
            }
            let layout = self.machine.pipelines[pipeline].groups[group]
                .layout
                .clone();
            let made = self.bind_group(&layout, resolved);
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
        let post = self
            .machine
            .frame_slots
            .get(&binding)
            .is_some_and(|slot| slot.post);
        if post {
            Resolved::Buffer(self.set().post_buffer.clone())
        } else {
            Resolved::Placeholder(ty)
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
        let held = (set.canvas.as_ref())
            .is_some_and(|made| made.texture.format() == self.canvas.format);
        if split == held && set.first().is_some_and(|made| made.size == extent) {
            return;
        }
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST;
        let device = self.device.clone();
        let made = |label: &'static str, format, samples| {
            let made = Made {
                label,
                format,
                usage,
                dimension: wgpu::TextureDimension::D2,
                samples,
            };
            Allocated::new(&device, &made, extent, 1)
        };
        let shared: Vec<Allocated> = (self.machine.declared.targets.iter())
            .map(|shared| made(shared.name.name(), shared.format, shared.samples))
            .collect();
        let canvas = split.then(|| made("engine/view-canvas", self.canvas.format, 1));
        let set = self.set_mut();
        set.shared = shared;
        set.canvas = canvas;
        engine_profile::event(
            "rebuilt",
            "engine/view-targets",
            &[("why", engine_profile::Fact::Text("size"))],
        );
        self.touch(Held::Targets);
    }

    pub(crate) fn settle_sets(&mut self, listed: &[engine_core::ViewFacts]) {
        let sets = &self.sets;
        let same = listed.len() == sets.list.len()
            && listed.iter().all(|facts| {
                sets.by_slot
                    .get(facts.slot as usize)
                    .and_then(|&at| sets.list.get(at as usize))
                    .is_some_and(|set| set.view == facts.view)
            });
        if !same {
            self.resettle(listed);
        }
        if let Some(main) = listed.first() {
            self.select(main.slot);
        }
    }

    fn resettle(&mut self, listed: &[engine_core::ViewFacts]) {
        let shown = |view: bevy_ecs::entity::Entity| listed.iter().any(|facts| facts.view == view);
        if let Some(main) = listed.first()
            && !self.sets.list.iter().any(|set| set.view == main.view)
            && let Some(set) = self.sets.list.iter_mut().find(|set| !shown(set.view))
        {
            set.view = main.view;
        }
        let machine = &self.machine;
        self.sets.list.retain_mut(|set| {
            let kept = shown(set.view);
            if !kept {
                set.release(machine, None);
            }
            kept
        });
        for facts in listed {
            if !self.sets.list.iter().any(|set| set.view == facts.view) {
                let set = ViewSet::new(&self.device, &self.machine, facts.view, &self.sets.holders);
                self.sets.list.push(set);
            }
        }
        let slots = listed.iter().map(|facts| facts.slot as usize + 1).max().unwrap_or(0);
        self.sets.by_slot.clear();
        self.sets.by_slot.resize(slots, u32::MAX);
        for (at, set) in self.sets.list.iter_mut().enumerate() {
            if let Some(facts) = listed.iter().find(|facts| facts.view == set.view) {
                set.slot = facts.slot;
                self.sets.by_slot[facts.slot as usize] = at as u32;
            }
        }
        self.sets.at = usize::MAX;
    }

    pub(crate) fn select(&mut self, slot: u32) -> usize {
        let Some(at) = self
            .sets
            .by_slot
            .get(slot as usize)
            .map(|&at| at as usize)
            .filter(|&at| at < self.sets.list.len())
        else {
            return self.sets.at;
        };
        if at != self.sets.at {
            let holding = &self.sets.fixed.holding;
            for filled in &mut self.filled {
                if holding.contains(filled.base as usize) {
                    filled.groups.clear();
                }
            }
            self.sets.at = at;
        }
        at
    }

    pub(crate) fn plan_view(&mut self) -> Result<Arc<Plan>, Error> {
        let Self {
            sets,
            planning,
            machine,
            ..
        } = self;
        let scratch = &sets.scratch;
        let set = &mut sets.list[sets.at];
        let found = (set.plans.iter()).position(|(skips, ..)| *skips == scratch.skips);
        let why = match found.map(|at| &set.plans[at]) {
            Some((_, key, plan))
                if key.runs == planning.runs
                    && key.outputs == scratch.outputs
                    && key.ready == scratch.ready =>
            {
                return Ok(Arc::clone(plan));
            }
            Some((_, key, _)) if key.runs != planning.runs => 1,
            Some((_, key, _)) if key.outputs != scratch.outputs => 2,
            Some(_) => 3,
            None => 0,
        };
        let plan = Arc::new(
            machine
                .graph
                .plan(&planning.runs, &scratch.outputs, &scratch.ready)?,
        );
        match found {
            Some(at) => {
                let (_, key, held) = &mut set.plans[at];
                key.runs.assign(&planning.runs);
                key.outputs.assign(&scratch.outputs);
                key.ready.assign(&scratch.ready);
                *held = Arc::clone(&plan);
            }
            None => {
                let key = PlanKey {
                    runs: planning.runs.clone(),
                    outputs: scratch.outputs.clone(),
                    ready: scratch.ready.clone(),
                };
                set.plans.push((scratch.skips, key, Arc::clone(&plan)));
            }
        }
        sets.counted.plans[why] += 1;
        if engine_profile::recording() {
            engine_profile::event(
                "plan",
                &set.slot.to_string(),
                &[("why", engine_profile::Fact::Text(PLAN_WHY[why]))],
            );
        }
        Ok(plan)
    }

    pub(crate) fn drawing_of(
        &mut self,
        slot: u32,
        handle: TextureHandle,
    ) -> Option<(wgpu::TextureView, wgpu::Texture)> {
        let (texture, _) = self.assets.textures.target(handle)?;
        let set = self
            .sets
            .by_slot
            .get(slot as usize)
            .and_then(|&at| self.sets.list.get_mut(at as usize))?;
        if set.drawing.as_ref().is_none_or(|(held, _)| held != texture) {
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(texture.format().remove_srgb_suffix()),
                ..Default::default()
            });
            set.drawing = Some((texture.clone(), view));
        }
        set.drawing
            .as_ref()
            .map(|(texture, view)| (view.clone(), texture.clone()))
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
        let shared = self.machine.declared.targets.len();
        let pairs: Vec<(wgpu::Texture, wgpu::Texture, wgpu::Extent3d)> = (0..shared)
            .filter_map(|at| {
                let (from, to) = (source.shared.get(at)?, current.shared.get(at)?);
                (from.size == to.size).then(|| (from.texture.clone(), to.texture.clone(), to.size))
            })
            .collect();
        if pairs.len() != shared {
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
        let Some(held) = self.sets.list[self.sets.at].canvas.as_ref() else {
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
    pub(crate) textures: Vec<Option<Allocated>>,
    pub(crate) sizes: Vec<Option<(wgpu::Extent3d, u32)>>,
    pub(crate) buffers: Vec<Option<(wgpu::Buffer, u64)>>,
}

impl Tables {
    pub(crate) fn of(machine: &Machine) -> Self {
        Self {
            textures: machine.declared.textures.iter().map(|_| None).collect(),
            sizes: machine.declared.textures.iter().map(|_| None).collect(),
            buffers: machine.declared.buffers.iter().map(|_| None).collect(),
        }
    }
}

const PLAN_WHY: [&str; 4] = ["first", "runs", "outputs", "ready"];

pub(crate) struct PlanKey {
    runs: Bits,
    outputs: Bits,
    ready: Bits,
}

type HeldGroups = [Option<wgpu::BindGroup>; 2];

pub(crate) struct ViewSet {
    pub(crate) view: bevy_ecs::entity::Entity,
    pub(crate) slot: u32,
    pub(crate) shared: Vec<Allocated>,
    pub(crate) canvas: Option<Allocated>,
    pub(crate) tables: Tables,
    pub(crate) post_buffer: wgpu::Buffer,
    pub(crate) groups: Vec<Option<(Holder, HeldGroups)>>,
    pub(crate) stale: std::collections::BTreeSet<Holder>,
    pub(crate) ran: Bits,
    pub(crate) plans: Vec<(u64, PlanKey, Arc<Plan>)>,
    pub(crate) drawing: Option<(wgpu::Texture, wgpu::TextureView)>,
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
            slot: 0,
            shared: Vec::new(),
            canvas: None,
            tables: Tables::of(machine),
            post_buffer: uniform_buffer(
                device,
                "engine/post-uniform",
                size_of::<crate::shared::PostUniform>() as u64,
            ),
            groups: Vec::new(),
            stale: holders.clone(),
            ran: Bits::of(machine.graph.passes()),
            plans: Vec::new(),
            drawing: None,
        }
    }

    pub(crate) fn group(&self, holder: u32) -> Option<&HeldGroups> {
        self.groups
            .get(holder as usize)?
            .as_ref()
            .map(|(_, made)| made)
    }

    pub(crate) fn allocated(&self, target: Target) -> Option<&Allocated> {
        match target {
            Target::Shared(at) => self.shared.get(at as usize),
            Target::Canvas => self.canvas.as_ref(),
        }
    }

    pub(crate) fn first(&self) -> Option<&Allocated> {
        self.shared.first().or(self.canvas.as_ref())
    }

    pub(crate) fn target(&self, target: Target) -> Option<&wgpu::TextureView> {
        self.allocated(target).map(|made| &made.view)
    }

    pub(crate) fn release(&mut self, machine: &Machine, part: Option<u32>) {
        let owned = |owner: u32| part.is_none_or(|part| owner == part);
        for (texture, made) in self.tables.textures.iter_mut().enumerate() {
            if owned(machine.declared.textures[texture].part) {
                *made = None;
                self.tables.sizes[texture] = None;
            }
        }
        for (buffer, made) in self.tables.buffers.iter_mut().enumerate() {
            if owned(machine.declared.buffers[buffer].part) {
                *made = None;
            }
        }
        self.stale
            .extend(self.groups.iter().flatten().map(|(holder, _)| *holder));
    }
}

pub(crate) struct Fixed {
    pub(crate) holding: Bits,
    pub(crate) part_passes: Vec<Vec<u32>>,
    pub(crate) part_globals: Vec<Vec<u32>>,
    pub(crate) global: Bits,
    pub(crate) main_views: Bits,
    pub(crate) names: Vec<u32>,
}

pub(crate) struct Scratch {
    pub(crate) base: Bits,
    pub(crate) asked: Bits,
    pub(crate) outputs: Bits,
    pub(crate) ready: Bits,
    pub(crate) order: Vec<usize>,
    pub(crate) drawn_on: Vec<(engine_core::Target, usize)>,
    pub(crate) skips: u64,
    #[cfg(any(feature = "dev", feature = "profiling"))]
    pub(crate) seen: Vec<(usize, std::num::NonZeroU64)>,
}

#[derive(Default)]
pub(crate) struct Counted {
    pub(crate) plans: [u64; 4],
    pub(crate) opened: Vec<u64>,
    pub(crate) merged: Vec<u64>,
    pub(crate) why: Vec<Option<&'static str>>,
    pub(crate) shared_open: bool,
}

pub(crate) struct Sets {
    pub(crate) list: Vec<ViewSet>,
    pub(crate) at: usize,
    pub(crate) by_slot: Vec<u32>,
    pub(crate) holders: std::collections::BTreeSet<Holder>,
    pub(crate) passes: Bits,
    pub(crate) main: Bits,
    pub(crate) fixed: Fixed,
    pub(crate) scratch: Scratch,
    pub(crate) counted: Counted,
    pub(crate) settled: Option<Option<u64>>,
}

impl Sets {
    pub(crate) fn of(
        device: &wgpu::Device,
        machine: &Machine,
        holders: std::collections::BTreeSet<Holder>,
    ) -> Self {
        let (passes, main) = view_passes(machine);
        let declared = &machine.declared.passes;
        let count = declared.len();
        let mut holding = Bits::of(machine.pipelines.len());
        for holder in &holders {
            if let Holder::Own(pipeline, _) = holder {
                holding.insert(*pipeline as usize);
            }
        }
        let parts = declared
            .iter()
            .map(|pass| pass.part as usize + 1)
            .max()
            .unwrap_or(0)
            .max(machine.declared.parts.len());
        let mut part_passes = vec![Vec::new(); parts];
        let mut part_globals = vec![Vec::new(); parts];
        let mut global = Bits::of(count);
        let mut main_views = Bits::of(count);
        let mut seen: Vec<&'static str> = Vec::new();
        let mut names = Vec::with_capacity(count);
        for (index, pass) in declared.iter().enumerate() {
            part_passes[pass.part as usize].push(index as u32);
            let viewed = passes.contains(index);
            if !viewed {
                part_globals[pass.part as usize].push(index as u32);
            }
            if !viewed && !main.contains(index) {
                global.insert(index);
            }
            if viewed && pass.views == engine_core::PassViews::Main {
                main_views.insert(index);
            }
            let name = pass.group.unwrap_or(pass.name);
            let at = match seen.iter().position(|known| *known == name) {
                Some(at) => at,
                None => {
                    seen.push(name);
                    seen.len() - 1
                }
            };
            names.push(at as u32);
        }
        let resources = machine.graph.resources();
        let first = ViewSet::new(
            device,
            machine,
            bevy_ecs::entity::Entity::PLACEHOLDER,
            &holders,
        );
        Self {
            list: vec![first],
            at: 0,
            by_slot: vec![0],
            holders,
            passes,
            main,
            fixed: Fixed {
                holding,
                part_passes,
                part_globals,
                global,
                main_views,
                names,
            },
            scratch: Scratch {
                base: Bits::of(count),
                asked: Bits::of(resources),
                outputs: Bits::of(resources),
                ready: Bits::of(resources),
                order: Vec::new(),
                drawn_on: Vec::new(),
                skips: 0,
                #[cfg(any(feature = "dev", feature = "profiling"))]
                seen: Vec::new(),
            },
            counted: Counted {
                opened: vec![0; count],
                merged: vec![0; count],
                why: vec![None; count],
                ..Counted::default()
            },
            settled: None,
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

fn view_passes(machine: &Machine) -> (Bits, Bits) {
    let graph = &machine.graph;
    let declared = &machine.declared.passes;
    let mut passes = Bits::of(graph.passes());
    for (index, pass) in declared.iter().enumerate() {
        if pass.per_view() {
            passes.insert(index);
        }
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
    let spread = graph.spread(passes.clone(), varying);
    let mut main = Bits::of(graph.passes());
    for pass in spread.iter().filter(|&pass| !passes.contains(pass)) {
        main.insert(pass);
    }
    (passes, main)
}

fn placeholder(device: &wgpu::Device, ty: Ty) -> Option<Placeholder> {
    use crate::linker::Sample;
    Some(match ty {
        Ty::Uniform | Ty::Storage { .. } => return None,
        Ty::Sampler => Placeholder::Sampler(crate::textures::gpu_sampler(
            device,
            "engine/placeholder",
            engine_core::Sampler::NEAREST,
        )),
        Ty::Comparison => Placeholder::Sampler(crate::textures::gpu_sampler(
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
                Sample::Depth => wgpu::TextureFormat::Depth32Float,
                Sample::Uint => wgpu::TextureFormat::R32Uint,
                Sample::Sint => wgpu::TextureFormat::R32Sint,
            };
            placeholder_texture(device, format, view, wgpu::TextureUsages::TEXTURE_BINDING)
        }
        Ty::StorageTexture { format, view, .. } => {
            placeholder_texture(device, format, view, wgpu::TextureUsages::STORAGE_BINDING)
        }
    })
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
    let texture = crate::gpu_calls::create_texture(device, &wgpu::TextureDescriptor {
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
    });
    Placeholder::Texture(texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(view),
        ..Default::default()
    }))
}
