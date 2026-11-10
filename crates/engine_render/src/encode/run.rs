use super::*;

pub(crate) struct Run {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: crate::gpu_calls::Queue,
    pub(crate) limits: engine_core::DeviceLimits,
    pub(crate) machine: Machine,
    pub(crate) parts: Vec<Option<Box<dyn Installed>>>,
    pub(crate) assets: Assets,
    pub(crate) part: u32,
    pub(crate) pass: u32,
    pub(crate) point: Option<&'static str>,
    pub(crate) recording: Recording,
    pub(crate) timed: Timed,
    pub(crate) queued: Queued,
    pub(crate) region: Option<(u32, u32)>,
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
}

pub(crate) struct Installs {
    pub(crate) turned: Vec<Option<u64>>,
    pub(crate) faults: Vec<(u32, Error)>,
    pub(crate) reinstalls: Vec<Reinstall>,
    pub(crate) draw_entries: Vec<KindMember>,
    pub(crate) installed: Vec<FrameId>,
    pub(crate) gates: Vec<Option<engine_registry::GateId>>,
    pub(crate) pass_gates: Vec<(u32, engine_registry::GateId)>,
    pub(crate) applied: Vec<crate::shared::AppliedModule>,
}

impl Installs {
    fn of(count: usize) -> Self {
        Self {
            turned: vec![Some(0); count],
            faults: Vec::new(),
            reinstalls: Vec::new(),
            draw_entries: Vec::new(),
            installed: vec![FrameId::default(); count],
            gates: Vec::new(),
            pass_gates: Vec::new(),
            applied: Vec::new(),
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
    pub(crate) owned: Vec<OwnedDraw>,
    pub(crate) pieces: wgpu::Buffer,
    pub(crate) piece_layout: Vec<(usize, crate::pieces::PieceAt)>,
    pub(crate) post: crate::shared::PostUniform,
    pub(crate) linear: wgpu::Sampler,
}

#[derive(Default)]
pub(crate) struct Pipelines {
    pub(crate) compiled: Vec<Compiled>,
    pub(crate) listed: Option<Vec<Compiled>>,
    pub(crate) up_front: u32,
    pub(crate) unlisted: u32,
    pub(crate) switches: Vec<(u32, bool)>,
    pub(crate) listing: bool,
    pub(crate) making: super::making::Making,
    pub(crate) refused: Vec<String>,
    pub(crate) cache: Option<wgpu::PipelineCache>,
}

pub(crate) struct Planning {
    pub(crate) runs: Bits,
    pub(crate) plan: Option<Arc<Plan>>,
    pub(crate) planned_at: usize,
    pub(crate) pass_decls: Vec<Arc<PassDecl>>,
    pub(crate) pass_sets: Vec<Arc<[SetFormat]>>,
    pub(crate) live: Vec<bool>,
    pub(crate) merge: bool,
    pub(crate) labelled: Option<u32>,
}

#[derive(Default)]
pub(crate) struct Tallies {
    pub(crate) drawn: u32,
    pub(crate) part_draws: Vec<u32>,
    pub(crate) rebuilt: Rebuilt,
    pub(crate) uploads: Uploads,
    pub(crate) kinds: KindTallies,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct KindTally {
    pub(crate) kind: engine_core::DrawKind,
    pub(crate) pass: &'static str,
    pub(crate) reached: u32,
    pub(crate) skipped: Option<&'static str>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DroppedTally {
    pub(crate) part: u32,
    pub(crate) kind: Option<engine_core::DrawKind>,
    pub(crate) pass: &'static str,
    pub(crate) reason: engine_core::DropReason,
    pub(crate) material: Option<engine_core::MaterialHandle>,
    pub(crate) count: u32,
}

#[derive(Default)]
pub(crate) struct KindTallies {
    pub(crate) drawing: Option<usize>,
    pub(crate) ran: Vec<&'static str>,
    pub(crate) listed: Vec<KindTally>,
    pub(crate) dropped: Vec<DroppedTally>,
    pub(crate) reported: Vec<(engine_core::DrawKind, &'static str)>,
    pub(crate) problems: Vec<Error>,
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
                    skipped: None,
                });
                self.listed.len() - 1
            }
        };
        self.drawing = Some(at);
    }

    pub(crate) fn end(&mut self) {
        self.drawing = None;
    }

    pub(crate) fn skip(&mut self, reason: &'static str) {
        if let Some(tally) = self.drawing.and_then(|at| self.listed.get_mut(at)) {
            tally.skipped = Some(reason);
        }
    }

    pub(crate) fn reached(&mut self) {
        if let Some(tally) = self.drawing.and_then(|at| self.listed.get_mut(at)) {
            tally.reached += 1;
        }
    }

    pub(crate) fn drop_draw(
        &mut self,
        (part, own_pass): (u32, &'static str),
        reason: engine_core::DropReason,
        material: Option<engine_core::MaterialHandle>,
    ) {
        let (kind, pass) = match self.drawing.and_then(|at| self.listed.get(at)) {
            Some(tally) => (Some(tally.kind), tally.pass),
            None => (None, own_pass),
        };
        let known = self.dropped.iter_mut().find(|known| {
            known.part == part && known.kind == kind && known.pass == pass && known.reason == reason
        });
        match known {
            Some(known) => known.count += 1,
            None => self.dropped.push(DroppedTally {
                part,
                kind,
                pass,
                reason,
                material,
                count: 1,
            }),
        }
    }

    pub(crate) fn start_frame(&mut self) {
        self.drawing = None;
        self.ran.clear();
        self.listed.clear();
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
        .map(|pass| Arc::from(pass_set(&machine.outputs, pass)))
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
        (device, queue, limits): (
            wgpu::Device,
            crate::gpu_calls::Queue,
            engine_core::DeviceLimits,
        ),
        machine: Machine,
        parts: Vec<Option<Box<dyn Installed>>>,
        assets: Assets,
    ) -> Self {
        let camera = uniform_buffer(&device, "engine/cameras", CAMERA_STRIDE);
        let linear = crate::materials::gpu_sampler(
            &device,
            "engine/post-sampler",
            engine_core::Sampler::LINEAR_CLAMP,
        );
        let pieces = uniform_buffer(&device, "engine/pieces", crate::pieces::PIECE_ALIGN);
        let samplers = machine
            .declared
            .samplers
            .iter()
            .map(|desc| crate::materials::gpu_sampler(&device, "engine/part-sampler", *desc))
            .collect();
        let count = parts.len();
        let dimensions = (0..machine.declared.textures.len())
            .map(|texture| texture_dimension(&machine, texture as u32))
            .collect();
        let tables = views::Tables::of(&machine);
        let runs = Bits::of(machine.graph.passes());
        let (frame_groups, own_groups) = fresh_groups(&machine);
        let holders = Holders::of(&machine);
        let view_holders = holders.of_view(&machine);
        let (passes, main) = views::view_passes(&machine);
        let mut stale = stale_holders(&machine);
        stale.retain(|holder| !view_holders.contains(holder));
        let first = views::ViewSet::new(
            &device,
            &machine,
            bevy_ecs::entity::Entity::PLACEHOLDER,
            &view_holders,
        );
        let sets = views::Sets {
            list: vec![first],
            at: 0,
            holders: view_holders,
            passes,
            main,
        };
        let pass_decls = machine
            .declared
            .passes
            .iter()
            .map(|pass| Arc::new(pass.clone()))
            .collect();
        let pass_sets = pass_sets_of(&machine);
        let live = machine.outputs.iter().map(crate::machine::layered).collect();
        Self {
            device,
            queue,
            limits,
            machine,
            parts,
            assets,
            part: 0,
            pass: 0,
            point: None,
            recording: Recording::Idle,
            timed: Timed::default(),
            queued: Queued::default(),
            region: None,
            sets,
            tables,
            spare: Spare::default(),
            reached: Reached::default(),
            filled: Vec::new(),
            installs: Installs::of(count),
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
            canvas: Canvas {
                view: None,
                texture: None,
                surface: wgpu::TextureFormat::Bgra8Unorm,
                format: wgpu::TextureFormat::Bgra8Unorm,
                rect: None,
                load: false,
                carried: false,
                clear: crate::graph::BLACK,
            },
            uniforms: Uniforms {
                cameras: vec![bytemuck::Zeroable::zeroed()],
                camera,
                camera_slot: 0,
                owned: Vec::new(),
                pieces,
                piece_layout: Vec::new(),
                post: bytemuck::Zeroable::zeroed(),
                linear,
            },
            pipelines: Pipelines::default(),
            planning: Planning {
                runs,
                plan: None,
                planned_at: 0,
                pass_decls,
                pass_sets,
                live,
                merge: true,
                labelled: None,
            },
            tallies: Tallies {
                part_draws: vec![0; count],
                ..Tallies::default()
            },
            reading: Reading {
                slots: 3,
                ..Reading::default()
            },
        }
    }
}
