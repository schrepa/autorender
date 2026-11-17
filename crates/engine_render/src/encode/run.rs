use super::*;

pub(crate) struct Run {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: crate::gpu_calls::Queue,
    pub(crate) machine: Machine,
    pub(crate) parts: Vec<Option<Box<dyn Installed>>>,
    pub(crate) assets: Assets,
    pub(crate) part: u32,
    pub(crate) pass: u32,
    pub(crate) point: Option<&'static str>,
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

#[derive(Clone, Copy, PartialEq, Eq)]
struct Offsets {
    held: [u32; MOST_OFFSETS],
    count: usize,
}

impl Offsets {
    fn of(given: &[u32]) -> Option<Self> {
        let mut held = [0; MOST_OFFSETS];
        held.get_mut(..given.len())?.copy_from_slice(given);
        Some(Self {
            held,
            count: given.len(),
        })
    }

    fn listed(&self) -> &[u32] {
        &self.held[..self.count]
    }
}

#[derive(Default)]
pub(crate) struct Bound {
    vertex: Vec<Option<(wgpu::Buffer, u64)>>,
    index: Option<wgpu::Buffer>,
    groups: [Option<(wgpu::BindGroup, Offsets)>; MOST_GROUPS],
    render: Option<wgpu::RenderPipeline>,
    compute: Option<wgpu::ComputePipeline>,
    viewport: Option<[u32; 4]>,
    scissor: Option<[u32; 4]>,
}

impl Bound {
    pub(crate) fn forget(&mut self) {
        self.vertex.clear();
        self.index = None;
        self.groups = Default::default();
        self.render = None;
        self.compute = None;
        self.viewport = None;
        self.scissor = None;
    }

    fn holds(&self, index: u32, group: &wgpu::BindGroup, offsets: &[u32]) -> bool {
        self.groups
            .get(index as usize)
            .and_then(Option::as_ref)
            .is_some_and(|(held, at)| held == group && at.listed() == offsets)
    }

    fn hold(&mut self, index: u32, group: &wgpu::BindGroup, offsets: &[u32]) {
        if let Some(slot) = self.groups.get_mut(index as usize) {
            *slot = Offsets::of(offsets).map(|offsets| (group.clone(), offsets));
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
        render.set_pipeline(pipeline);
        self.render = Some(pipeline.clone());
    }

    pub(crate) fn dispatch_pipeline(
        &mut self,
        pass: &mut wgpu::ComputePass<'static>,
        pipeline: &wgpu::ComputePipeline,
    ) {
        if self.compute.as_ref() == Some(pipeline) {
            return;
        }
        pass.set_pipeline(pipeline);
        self.compute = Some(pipeline.clone());
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
            installed: vec![FrameId::default(); count],
            gates: Vec::new(),
            pass_gates: Vec::new(),
            features,
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
}

impl Pipelines {
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
    pub(crate) drawn: u32,
    pub(crate) part_draws: Vec<u32>,
    pub(crate) rebuilt: Rebuilt,
    pub(crate) uploads: Uploads,
    pub(crate) kinds: KindTallies,
    pub(crate) worst: crate::Recorded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KindAnswer {
    Skipped(&'static str),
    Waiting(&'static str),
    Refused(&'static str),
}

impl KindAnswer {
    const fn rank(self) -> u8 {
        match self {
            Self::Skipped(_) => 0,
            Self::Waiting(_) => 1,
            Self::Refused(_) => 2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct KindTally {
    pub(crate) kind: engine_core::DrawKind,
    pub(crate) pass: &'static str,
    pub(crate) reached: u32,
    pub(crate) answer: Option<KindAnswer>,
}

#[derive(Clone, Debug)]
pub(crate) struct RefusedTally {
    pub(crate) kind: engine_core::DrawKind,
    pub(crate) pass: &'static str,
    pub(crate) error: Error,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DroppedTally {
    pub(crate) part: u32,
    pub(crate) kind: Option<engine_core::DrawKind>,
    pub(crate) pass: &'static str,
    pub(crate) reason: engine_core::DropReason,
    pub(crate) asset: Option<engine_core::Asset>,
    pub(crate) count: u32,
}

#[derive(Default)]
pub(crate) struct KindTallies {
    pub(crate) drawing: Option<usize>,
    pub(crate) ran: Vec<&'static str>,
    pub(crate) listed: Vec<KindTally>,
    pub(crate) dropped: Vec<DroppedTally>,
    pub(crate) refused: Vec<RefusedTally>,
    pub(crate) reported: Vec<(engine_core::DrawKind, &'static str)>,
    pub(crate) problems: Vec<Error>,
    pub(crate) report: Option<engine_core::DrawReports>,
}

impl KindTallies {
    pub(crate) fn scened(&mut self, pass: &'static str) {
        if !self.ran.contains(&pass) {
            self.ran.push(pass);
        }
    }

    pub(crate) fn begin(&mut self, kind: engine_core::DrawKind, pass: &'static str) {
        let at = match self
            .listed
            .iter()
            .position(|known| known.kind == kind && known.pass == pass)
        {
            Some(at) => at,
            None => {
                self.listed.push(KindTally {
                    kind,
                    pass,
                    reached: 0,
                    answer: None,
                });
                self.listed.len() - 1
            }
        };
        self.drawing = Some(at);
    }

    pub(crate) fn end(&mut self) {
        self.drawing = None;
    }

    fn answer(&mut self, answer: KindAnswer) {
        if let Some(tally) = self.drawing.and_then(|at| self.listed.get_mut(at))
            && tally.answer.is_none_or(|held| held.rank() <= answer.rank())
        {
            tally.answer = Some(answer);
        }
    }

    pub(crate) fn skip(&mut self, reason: &'static str) {
        self.answer(KindAnswer::Skipped(reason));
    }

    pub(crate) fn wait(&mut self, on: &'static str) {
        self.answer(KindAnswer::Waiting(on));
    }

    pub(crate) fn refuse(&mut self, on: &'static str, error: &Error) {
        let Some(tally) = self.drawing.and_then(|at| self.listed.get(at)).copied() else {
            return;
        };
        self.answer(KindAnswer::Refused(on));
        let known = self.refused.iter().any(|refused| {
            refused.kind == tally.kind && refused.pass == tally.pass && refused.error == *error
        });
        if !known {
            self.refused.push(RefusedTally {
                kind: tally.kind,
                pass: tally.pass,
                error: error.clone(),
            });
        }
    }

    pub(crate) fn reached(&mut self) {
        if let Some(tally) = self.drawing.and_then(|at| self.listed.get_mut(at)) {
            tally.reached += 1;
        }
    }

    pub(crate) fn drop_draws(
        &mut self,
        (part, own_pass): (u32, &'static str),
        (reason, asset): (engine_core::DropReason, Option<engine_core::Asset>),
        count: u32,
    ) {
        let (kind, pass) = match self.drawing.and_then(|at| self.listed.get(at)) {
            Some(tally) => (Some(tally.kind), tally.pass),
            None => (None, own_pass),
        };
        let known = self.dropped.iter_mut().find(|known| {
            known.part == part
                && known.kind == kind
                && known.pass == pass
                && known.reason == reason
                && known.asset == asset
        });
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
        self.ran.clear();
        self.listed.clear();
        self.dropped.clear();
        self.refused.clear();
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
        .map(|pass| Arc::from(pass_set(pass)))
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
            .map(|desc| crate::materials::gpu_sampler(&device, "engine/part-sampler", *desc))
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
        let sets = first_sets(&device, &machine, view_holders);
        let pass_decls = machine
            .declared
            .passes
            .iter()
            .map(|pass| Arc::new(pass.clone()))
            .collect();
        let pass_sets = pass_sets_of(&machine);
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
                part_draws: vec![0; count],
                ..Tallies::default()
            },
            reading: Reading {
                slots: 3,
                ..Reading::default()
            },
            bound: Bound::default(),
            arena: Arena::default(),
        }
    }
}

fn first_sets(
    device: &wgpu::Device,
    machine: &Machine,
    holders: std::collections::BTreeSet<super::Holder>,
) -> views::Sets {
    let (passes, main) = views::view_passes(machine);
    let first = views::ViewSet::new(
        device,
        machine,
        bevy_ecs::entity::Entity::PLACEHOLDER,
        &holders,
    );
    views::Sets {
        list: vec![first],
        at: 0,
        holders,
        passes,
        main,
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
        }
    }
}
