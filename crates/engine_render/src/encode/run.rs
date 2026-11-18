use super::*;

pub(crate) struct Run {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: crate::gpu_calls::Queue,
    pub(crate) machine: Machine,
    pub(crate) parts: Vec<Option<Box<dyn Installed>>>,
    pub(crate) assets: Assets,
    pub(crate) part: u32,
    pub(crate) pass: u32,
    pub(crate) point: Option<PassBit>,
    pub(crate) recording: Recording,
    pub(crate) timed: Timed,
    pub(crate) queued: Queued,
    pub(crate) owed_mips: Vec<(u32, Range<u32>)>,
    pub(crate) sets: views::Sets,
    pub(crate) tables: views::Tables,
    pub(crate) spare: Spare,
    pub(crate) reached: Reached,
    pub(crate) filled: Vec<Filled>,
    pub(crate) installs: Installs,
    pub(crate) frames: Frames,
    pub(crate) bindings: Bindings,
    pub(crate) canvas: Canvas,
    pub(crate) uniforms: Uniforms,
    pub(crate) pipelines: Pipelines,
    pub(crate) planning: Planning,
    pub(crate) tallies: Tallies,
    pub(crate) reading: Reading,
    pub(crate) bound: Bound,
    pub(crate) arena: Arena,
}

#[derive(Default)]
pub(crate) struct Bound {
    vertex: Vec<Option<(wgpu::Buffer, u64)>>,
    index: Option<wgpu::Buffer>,
    groups: Vec<Option<(wgpu::BindGroup, usize)>>,
    offsets: Vec<u32>,
    pub(crate) zeros: Vec<u32>,
    pub(crate) given: Vec<u32>,
    render: Option<wgpu::RenderPipeline>,
    compute: Option<wgpu::ComputePipeline>,
    viewport: Option<[u32; 4]>,
    scissor: Option<[u32; 4]>,
    #[cfg(any(feature = "dev", feature = "profiling"))]
    pub(crate) swap: Option<Swap>,
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(crate) struct Swap {
    pub(crate) group: wgpu::BindGroup,
    pub(crate) empty: wgpu::BindGroup,
    pub(crate) render: Vec<(wgpu::RenderPipeline, wgpu::RenderPipeline, u32)>,
    pub(crate) compute: Vec<(wgpu::ComputePipeline, wgpu::ComputePipeline, u32)>,
}

#[cfg(any(feature = "dev", feature = "profiling"))]
impl Swap {
    fn groups(&self, from: u32) -> impl Iterator<Item = (u32, &wgpu::BindGroup)> {
        let group = crate::linker::counting::GROUP;
        (from..group)
            .map(|index| (index, &self.empty))
            .chain(std::iter::once((group, &self.group)))
    }
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(crate) struct Work {
    pub(crate) counter: crate::linker::counting::Counter,
    pub(crate) generation: u64,
    pub(crate) asked: Vec<CountedAsk>,
    pub(crate) live: Option<Live>,
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(crate) struct CountedAsk {
    pub(crate) counted: (u64, u64),
    pub(crate) plain: (u64, u64),
    pub(crate) pipeline: u32,
    pub(crate) generation: u64,
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(crate) struct Live {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) empty: wgpu::BindGroupLayout,
}

impl Bound {
    pub(crate) fn of(limits: crate::machine::Limits) -> Self {
        Self {
            groups: vec![None; limits.groups],
            offsets: vec![0; limits.groups * limits.offsets],
            zeros: vec![0; limits.offsets],
            given: vec![0; limits.offsets],
            ..Self::default()
        }
    }

    pub(crate) fn forget(&mut self) {
        self.vertex.clear();
        self.index = None;
        self.groups.fill(None);
        self.render = None;
        self.compute = None;
        self.viewport = None;
        self.scissor = None;
    }

    fn held_offsets(&self, index: u32, count: usize) -> Option<&[u32]> {
        let start = index as usize * self.zeros.len();
        self.offsets.get(start..start + count)
    }

    fn holds(&self, index: u32, group: &wgpu::BindGroup, offsets: &[u32]) -> bool {
        self.groups
            .get(index as usize)
            .and_then(Option::as_ref)
            .is_some_and(|(held, count)| {
                held == group && self.held_offsets(index, *count) == Some(offsets)
            })
    }

    fn hold(&mut self, index: u32, group: &wgpu::BindGroup, offsets: &[u32]) {
        let start = index as usize * self.zeros.len();
        let kept = offsets.len() <= self.zeros.len()
            && self
                .offsets
                .get_mut(start..start + offsets.len())
                .map(|held| held.copy_from_slice(offsets))
                .is_some();
        if let Some(slot) = self.groups.get_mut(index as usize) {
            *slot = kept.then(|| (group.clone(), offsets.len()));
        }
    }

    pub(crate) fn draw_group(
        &mut self,
        render: &mut wgpu::RenderPass<'static>,
        index: u32,
        group: &wgpu::BindGroup,
        offsets: &[u32],
    ) {
        if self.holds(index, group, offsets) {
            return;
        }
        render.set_bind_group(index, group, offsets);
        self.hold(index, group, offsets);
    }

    pub(crate) fn dispatch_group(
        &mut self,
        pass: &mut wgpu::ComputePass<'static>,
        index: u32,
        group: &wgpu::BindGroup,
        offsets: &[u32],
    ) {
        if self.holds(index, group, offsets) {
            return;
        }
        pass.set_bind_group(index, group, offsets);
        self.hold(index, group, offsets);
    }

    pub(crate) fn draw_pipeline(
        &mut self,
        render: &mut wgpu::RenderPass<'static>,
        pipeline: &wgpu::RenderPipeline,
    ) {
        if self.render.as_ref() == Some(pipeline) {
            return;
        }
        self.render = Some(pipeline.clone());
        #[cfg(any(feature = "dev", feature = "profiling"))]
        if let Some(swap) = self.swap.take() {
            let found = swap.render.iter().find(|(plain, ..)| plain == pipeline);
            if let Some((_, counted, from)) = found {
                render.set_pipeline(counted);
                for (index, group) in swap.groups(*from) {
                    self.draw_group(render, index, group, &[]);
                }
            }
            let counted = found.is_some();
            self.swap = Some(swap);
            if counted {
                return;
            }
        }
        render.set_pipeline(pipeline);
    }

    pub(crate) fn dispatch_pipeline(
        &mut self,
        pass: &mut wgpu::ComputePass<'static>,
        pipeline: &wgpu::ComputePipeline,
    ) {
        if self.compute.as_ref() == Some(pipeline) {
            return;
        }
        self.compute = Some(pipeline.clone());
        #[cfg(any(feature = "dev", feature = "profiling"))]
        if let Some(swap) = self.swap.take() {
            let found = swap.compute.iter().find(|(plain, ..)| plain == pipeline);
            if let Some((_, counted, from)) = found {
                pass.set_pipeline(counted);
                for (index, group) in swap.groups(*from) {
                    self.dispatch_group(pass, index, group, &[]);
                }
            }
            let counted = found.is_some();
            self.swap = Some(swap);
            if counted {
                return;
            }
        }
        pass.set_pipeline(pipeline);
    }

    pub(crate) fn viewport(&mut self, render: &mut wgpu::RenderPass<'static>, rect: [u32; 4]) {
        if self.viewport == Some(rect) {
            return;
        }
        let [x, y, width, height] = rect.map(|value| value as f32);
        render.set_viewport(x, y, width, height, 0.0, 1.0);
        self.viewport = Some(rect);
    }

    pub(crate) fn scissor(&mut self, render: &mut wgpu::RenderPass<'static>, rect: [u32; 4]) {
        if self.scissor == Some(rect) {
            return;
        }
        render.set_scissor_rect(rect[0], rect[1], rect[2], rect[3]);
        self.scissor = Some(rect);
    }

    pub(crate) fn vertex_buffer(
        &mut self,
        render: &mut wgpu::RenderPass<'static>,
        slot: u32,
        buffer: &wgpu::Buffer,
        start: u64,
    ) {
        let at = slot as usize;
        let held = self.vertex.get(at).and_then(Option::as_ref);
        if held.is_some_and(|(held, offset)| held == buffer && *offset == start) {
            return;
        }
        render.set_vertex_buffer(slot, buffer.slice(start..));
        if self.vertex.len() <= at {
            self.vertex.resize(at + 1, None);
        }
        self.vertex[at] = Some((buffer.clone(), start));
    }

    pub(crate) fn within(
        &mut self,
        render: &mut wgpu::RenderPass<'static>,
        (width, height): (u32, u32),
    ) {
        self.viewport(render, [0, 0, width, height]);
        self.scissor(render, [0, 0, width, height]);
    }

    pub(crate) fn index_buffer(
        &mut self,
        render: &mut wgpu::RenderPass<'static>,
        buffer: &wgpu::Buffer,
    ) {
        if self.index.as_ref() == Some(buffer) {
            return;
        }
        render.set_index_buffer(buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.index = Some(buffer.clone());
    }
}

pub(crate) struct Installs {
    pub(crate) turned: Vec<Option<u64>>,
    pub(crate) faults: Vec<Error>,
    pub(crate) reinstalls: Vec<Reinstall>,
    pub(crate) draw_entries: Vec<KindMember>,
    pub(crate) members: Vec<Vec<u32>>,
    pub(crate) installed: Vec<FrameId>,
    pub(crate) gates: Vec<Option<engine_registry::BrickId>>,
    pub(crate) pass_gates: Vec<(u32, engine_registry::BrickId)>,
    pub(crate) features: wgpu::Features,
}

impl Installs {
    fn of(count: usize, features: wgpu::Features) -> Self {
        Self {
            turned: vec![Some(0); count],
            faults: Vec::new(),
            reinstalls: Vec::new(),
            draw_entries: Vec::new(),
            members: Vec::new(),
            installed: vec![FrameId::default(); count],
            gates: Vec::new(),
            pass_gates: Vec::new(),
            features,
        }
    }

    pub(crate) fn settle_members(&mut self, declared: &crate::install::Declared) {
        self.members.clear();
        for pass in &declared.passes {
            let Some((scene, bit)) = pass.scene.and_then(|scene| Some((scene, scene.bit?))) else {
                continue;
            };
            let at = bit.index() as usize;
            if self.members.len() <= at {
                self.members.resize(at + 1, Vec::new());
            }
            let listed = self.draw_entries.iter().enumerate().filter(|(_, member)| {
                pass.members.contains(&member.part) && member.decl.joins(scene.name)
            });
            self.members[at] = listed.map(|(at, _)| at as u32).collect();
        }
    }
}

#[derive(Default)]
pub(crate) struct Frames {
    pub(crate) current: u64,
    pub(crate) asked: u64,
    pub(crate) parity: usize,
}

pub(crate) struct Bindings {
    pub(crate) holders: Holders,
    pub(crate) stale: std::collections::BTreeSet<Holder>,
    pub(crate) frame_groups: Vec<Groups>,
    pub(crate) own_groups: Vec<Vec<Groups>>,
    pub(crate) placeholders: Vec<((Ty, u64), Placeholder)>,
    pub(crate) wanted_textures: Vec<TextureHandle>,
    pub(crate) dimensions: Vec<wgpu::TextureDimension>,
    pub(crate) samplers: Vec<wgpu::Sampler>,
}

pub(crate) struct Canvas {
    pub(crate) view: Option<wgpu::TextureView>,
    pub(crate) texture: Option<wgpu::Texture>,
    pub(crate) surface: wgpu::TextureFormat,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) rect: Option<[u32; 4]>,
    pub(crate) load: bool,
    pub(crate) carried: bool,
    pub(crate) clear: [f64; 4],
    pub(crate) fractions: Vec<(engine_core::TargetName, f32)>,
}

pub(crate) struct Uniforms {
    pub(crate) cameras: Vec<CameraUniform>,
    pub(crate) camera: wgpu::Buffer,
    pub(crate) camera_slot: u32,
    pub(crate) post: crate::shared::PostUniform,
}

#[derive(Default)]
pub(crate) struct Pipelines {
    pub(crate) compiled: Vec<Compiled>,
    pub(crate) listed: Option<Vec<Compiled>>,
    pub(crate) up_front: u32,
    pub(crate) unlisted: u32,
    pub(crate) listing: bool,
    pub(crate) making: super::making::Making,
    pub(crate) refused: Vec<(&'static str, u64, String)>,
    pub(crate) per_pass: Vec<(&'static str, [u32; 2])>,
    pub(crate) cache: Option<wgpu::PipelineCache>,
    pub(crate) warming: super::warming::Warming,
    pub(crate) settled: bool,
    pub(crate) standing: Vec<(u64, u64)>,
    #[cfg(any(feature = "dev", feature = "profiling"))]
    pub(crate) work: Option<Work>,
}

impl Pipelines {
    #[cfg(any(feature = "dev", feature = "profiling"))]
    pub(crate) fn work_landed(
        &mut self,
        written: engine_core::FrameId,
        arrival: crate::readback::Arrival<'_>,
    ) {
        if let (Some(work), crate::readback::Arrival::Read(bytes)) = (self.work.as_mut(), arrival) {
            work.counter.landed(written.0, bytes);
        }
    }

    #[cfg(not(any(feature = "dev", feature = "profiling")))]
    pub(crate) fn work_landed(
        &mut self,
        _written: engine_core::FrameId,
        _arrival: crate::readback::Arrival<'_>,
    ) {
    }

    pub(crate) fn warming_left(&self) -> u32 {
        self.warming.left().saturating_add(self.making.later_left())
    }

    pub(crate) fn made_in(&mut self, pass: &'static str, computed: bool) {
        let at = match self.per_pass.iter().position(|(known, _)| *known == pass) {
            Some(at) => at,
            None => {
                self.per_pass.push((pass, [0, 0]));
                self.per_pass.len() - 1
            }
        };
        self.per_pass[at].1[usize::from(computed)] += 1;
    }

    pub(crate) fn stood_in(&self) -> u32 {
        let left = self
            .standing
            .iter()
            .filter(|&&made| !self.making.waits(made) || self.warming.fed == Some(made));
        u32::try_from(left.count()).unwrap_or(u32::MAX)
    }
}

pub(crate) struct Planning {
    pub(crate) runs: Bits,
    pub(crate) plan: Option<Arc<Plan>>,
    pub(crate) planned_at: usize,
    pub(crate) pass_decls: Vec<Arc<PassDecl>>,
    pub(crate) pass_sets: Vec<Arc<[SetFormat]>>,
    pub(crate) merge: bool,
    pub(crate) labelled: Option<u32>,
    pub(crate) idle: Vec<(u32, engine_core::DrawKind, &'static str)>,
}

#[derive(Default)]
pub(crate) struct Tallies {
    pub(crate) drawn: crate::gpu_calls::Drawn,
    pub(crate) kinds: KindTallies,
    pub(crate) worst: crate::Recorded,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct KindCell {
    pub(crate) joins: bool,
    pub(crate) begun: bool,
    pub(crate) reached: u32,
    pub(crate) skipped: Option<&'static str>,
    pub(crate) reported: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DroppedTally {
    pub(crate) part: u32,
    pub(crate) kind: Option<u32>,
    pub(crate) pass: u32,
    pub(crate) reason: engine_core::DropReason,
    pub(crate) asset: Option<engine_core::Asset>,
    pub(crate) count: u32,
}

#[derive(Default)]
pub(crate) struct KindTallies {
    pub(crate) kinds: Vec<engine_core::DrawKind>,
    pub(crate) scenes: Vec<(&'static str, u32)>,
    pub(crate) cells: Vec<KindCell>,
    pub(crate) ran: Vec<bool>,
    pub(crate) drawing: Option<usize>,
    pub(crate) dropped: Vec<DroppedTally>,
    pub(crate) last_dropped: Vec<DroppedTally>,
    pub(crate) states: Vec<Option<engine_core::KindPassState>>,
    pub(crate) last_states: Vec<Option<engine_core::KindPassState>>,
    pub(crate) totals: Vec<(DroppedTally, u64)>,
    pub(crate) problems: Vec<Error>,
    pub(crate) report: Option<engine_core::DrawReports>,
    pub(crate) passes: Vec<engine_core::KindPass>,
    pub(crate) settled: bool,
}

impl KindTallies {
    pub(crate) fn of(passes: &engine_core::ScenePasses, declared: &[PassDecl]) -> Self {
        let kinds: Vec<engine_core::DrawKind> =
            passes.kinds().iter().map(|decl| decl.kind).collect();
        let scenes: Vec<(&'static str, u32)> = passes
            .listed()
            .iter()
            .filter(|listed| listed.bit.is_some())
            .map(|listed| {
                let pass = declared
                    .iter()
                    .position(|pass| pass.scene.is_some_and(|scene| scene.name == listed.name))
                    .map_or(u32::MAX, |at| at as u32);
                (listed.name, pass)
            })
            .collect();
        let mut cells = Vec::with_capacity(scenes.len() * kinds.len());
        for (name, _) in &scenes {
            for decl in passes.kinds() {
                cells.push(KindCell {
                    joins: decl.joins(name),
                    ..KindCell::default()
                });
            }
        }
        Self {
            ran: vec![false; scenes.len()],
            states: vec![None; cells.len()],
            last_states: vec![None; cells.len()],
            kinds,
            scenes,
            cells,
            ..Self::default()
        }
    }

    pub(crate) fn scened(&mut self, scene: u32) {
        if let Some(ran) = self.ran.get_mut(scene as usize) {
            *ran = true;
        }
    }

    pub(crate) fn begin(&mut self, scene: u32, kind: u32) {
        let at = scene as usize * self.kinds.len() + kind as usize;
        self.drawing = self.cells.get_mut(at).map(|cell| {
            cell.begun = true;
            at
        });
    }

    pub(crate) fn end(&mut self) {
        self.drawing = None;
    }

    pub(crate) fn skip(&mut self, reason: &'static str) {
        if let Some(cell) = self.drawing.and_then(|at| self.cells.get_mut(at)) {
            cell.skipped = Some(reason);
        }
    }

    pub(crate) fn reached(&mut self) {
        if let Some(cell) = self.drawing.and_then(|at| self.cells.get_mut(at)) {
            cell.reached += 1;
        }
    }

    pub(crate) fn kind_at(&self, cell: usize) -> (u32, u32) {
        let kinds = self.kinds.len().max(1);
        ((cell / kinds) as u32, (cell % kinds) as u32)
    }

    pub(crate) fn drop_draws(
        &mut self,
        (part, own_pass): (u32, u32),
        (reason, asset): (engine_core::DropReason, Option<engine_core::Asset>),
        count: u32,
    ) {
        let (kind, pass) = match self.drawing {
            Some(at) => {
                let (scene, kind) = self.kind_at(at);
                let pass = self.scenes.get(scene as usize).map_or(own_pass, |scene| scene.1);
                (Some(kind), pass)
            }
            None => (None, own_pass),
        };
        let same = |known: &DroppedTally| {
            known.part == part
                && known.kind == kind
                && known.pass == pass
                && known.reason == reason
                && known.asset == asset
        };
        let known = match self.dropped.last_mut() {
            Some(last) if same(last) => Some(last),
            Some(_) | None => self.dropped.iter_mut().find(|known| same(known)),
        };
        match known {
            Some(known) => known.count += count,
            None => self.dropped.push(DroppedTally {
                part,
                kind,
                pass,
                reason,
                asset,
                count,
            }),
        }
    }

    pub(crate) fn start_frame(&mut self) {
        self.drawing = None;
        self.ran.fill(false);
        for cell in &mut self.cells {
            cell.begun = false;
            cell.reached = 0;
            cell.skipped = None;
        }
        self.dropped.clear();
    }
}

#[derive(Default)]
pub(crate) struct Reading {
    pub(crate) list: Readbacks,
    pub(crate) arrived: Vec<Arrived>,
    pub(crate) slots: usize,
    pub(crate) slot_bytes: Option<u64>,
}

fn stale_holders(machine: &Machine) -> std::collections::BTreeSet<Holder> {
    (0..machine.variants.len())
        .map(|variant| Holder::Frame(variant as u32))
        .chain(
            machine
                .pipelines
                .iter()
                .enumerate()
                .flat_map(|(pipeline, built)| {
                    (0..built.groups.len())
                        .map(move |group| Holder::Own(pipeline as u32, group as u32))
                }),
        )
        .collect()
}

fn pass_sets_of(machine: &Machine) -> Vec<Arc<[SetFormat]>> {
    machine
        .declared
        .passes
        .iter()
        .map(|pass| Arc::from(machine.declared.pass_set(pass)))
        .collect()
}

fn fresh_groups(machine: &Machine) -> (Vec<Groups>, Vec<Vec<Groups>>) {
    let frame_groups = machine.variants.iter().map(|_| Groups::fresh()).collect();
    let own_groups = machine
        .pipelines
        .iter()
        .map(|pipeline| pipeline.groups.iter().map(|_| Groups::fresh()).collect())
        .collect();
    (frame_groups, own_groups)
}

impl Run {
    pub(crate) fn new(
        (device, queue): (wgpu::Device, crate::gpu_calls::Queue),
        machine: Machine,
        parts: Vec<Option<Box<dyn Installed>>>,
        assets: Assets,
    ) -> Self {
        let camera = uniform_buffer(&device, "engine/cameras", CAMERA_STRIDE);
        let samplers = machine
            .declared
            .samplers
            .iter()
            .map(|desc| crate::textures::gpu_sampler(&device, "engine/part-sampler", *desc))
            .collect();
        let count = parts.len();
        let features = device.features();
        let dimensions = (0..machine.declared.textures.len())
            .map(|texture| texture_dimension(&machine, texture as u32))
            .collect();
        let tables = views::Tables::of(&machine);
        let runs = Bits::of(machine.graph.passes());
        let (frame_groups, own_groups) = fresh_groups(&machine);
        let holders = Holders::of(&machine);
        let view_holders = holders.of_view(&machine);
        let mut stale = stale_holders(&machine);
        stale.retain(|holder| !view_holders.contains(holder));
        let sets = views::Sets::of(&device, &machine, view_holders);
        let pass_decls = machine
            .declared
            .passes
            .iter()
            .map(|pass| Arc::new(pass.clone()))
            .collect();
        let pass_sets = pass_sets_of(&machine);
        let drawn = crate::gpu_calls::Drawn::new(machine.declared.passes.len(), count);
        let bound = Bound::of(machine.limits);
        Self {
            device,
            queue,
            machine,
            parts,
            assets,
            part: 0,
            pass: 0,
            point: None,
            recording: Recording::Idle,
            timed: Timed::default(),
            queued: Queued::default(),
            owed_mips: Vec::new(),
            sets,
            tables,
            spare: Spare::default(),
            reached: Reached::default(),
            filled: Vec::new(),
            installs: Installs::of(count, features),
            frames: Frames::default(),
            bindings: Bindings {
                holders,
                stale,
                frame_groups,
                own_groups,
                placeholders: Vec::new(),
                wanted_textures: Vec::new(),
                dimensions,
                samplers,
            },
            canvas: Canvas::fresh(),
            uniforms: Uniforms {
                cameras: vec![bytemuck::Zeroable::zeroed()],
                camera,
                camera_slot: 0,
                post: bytemuck::Zeroable::zeroed(),
            },
            pipelines: Pipelines::default(),
            planning: Planning {
                runs,
                plan: None,
                planned_at: 0,
                pass_decls,
                pass_sets,
                merge: true,
                labelled: None,
                idle: Vec::new(),
            },
            tallies: Tallies {
                drawn,
                ..Tallies::default()
            },
            reading: Reading {
                slots: 3,
                ..Reading::default()
            },
            bound,
            arena: Arena::default(),
        }
    }
}


impl Canvas {
    fn fresh() -> Self {
        Self {
            view: None,
            texture: None,
            surface: wgpu::TextureFormat::Bgra8Unorm,
            format: wgpu::TextureFormat::Bgra8Unorm,
            rect: None,
            load: false,
            carried: false,
            clear: crate::graph::BLACK,
            fractions: Vec::new(),
        }
    }

    pub(crate) fn fraction(&self, name: engine_core::TargetName) -> f32 {
        (self.fractions.iter())
            .find(|(known, _)| *known == name)
            .map_or(1.0, |&(_, fraction)| fraction)
    }

    pub(crate) fn scale(&mut self, name: engine_core::TargetName, fraction: f32) {
        match self.fractions.iter_mut().find(|(known, _)| *known == name) {
            Some(found) => found.1 = fraction,
            None => self.fractions.push((name, fraction)),
        }
    }
}
