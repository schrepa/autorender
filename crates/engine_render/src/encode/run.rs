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
    pub(crate) debug: bool,
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
    pub(crate) applied: Vec<crate::shared::AppliedModule>,
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
}

pub(crate) struct Planning {
    pub(crate) runs: Bits,
    pub(crate) plan: Option<Arc<Plan>>,
    pub(crate) planned_at: usize,
    pub(crate) pass_decls: Vec<Arc<PassDecl>>,
    pub(crate) layer_sets: Vec<(engine_core::AttachPoint, Arc<[SetFormat]>)>,
    pub(crate) merge: bool,
    pub(crate) labelled: Option<u32>,
}

#[derive(Default)]
pub(crate) struct Tallies {
    pub(crate) drawn: u32,
    pub(crate) part_draws: Vec<u32>,
    pub(crate) rebuilt: Rebuilt,
    pub(crate) uploads: Uploads,
}

#[derive(Default)]
pub(crate) struct Reading {
    pub(crate) list: Readbacks,
    pub(crate) arrived: Vec<Arrived>,
    pub(crate) slots: usize,
    pub(crate) slot_bytes: Option<u64>,
}

fn sampler_of(device: &wgpu::Device, desc: &crate::SamplerDesc) -> wgpu::Sampler {
    let [u, v, w] = desc.address;
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("engine/part-sampler"),
        address_mode_u: u,
        address_mode_v: v,
        address_mode_w: w,
        mag_filter: desc.filter,
        min_filter: desc.filter,
        mipmap_filter: desc.mipmap,
        compare: desc.compare,
        anisotropy_clamp: desc.anisotropy.max(1),
        ..Default::default()
    })
}

fn post_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("engine/post-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    })
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

fn layer_sets_of(machine: &Machine) -> Vec<(engine_core::AttachPoint, Arc<[SetFormat]>)> {
    let mut layer_sets: Vec<(engine_core::AttachPoint, Arc<[SetFormat]>)> = Vec::new();
    for layer in machine.declared.passes.iter().filter_map(pass_layer) {
        if !layer_sets.iter().any(|(known, _)| *known == layer) {
            layer_sets.push((layer, Arc::from(set_formats(&machine.outputs, layer))));
        }
    }
    layer_sets
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
        let linear = post_sampler(&device);
        let pieces = uniform_buffer(&device, "engine/pieces", crate::pieces::PIECE_ALIGN);
        let samplers = machine
            .declared
            .samplers
            .iter()
            .map(|desc| sampler_of(&device, desc))
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
        let layer_sets = layer_sets_of(&machine);
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
            debug: false,
            region: None,
            sets,
            tables,
            spare: Spare::default(),
            reached: Reached::default(),
            filled: Vec::new(),
            installs: Installs {
                turned: vec![Some(0); count],
                faults: Vec::new(),
                reinstalls: Vec::new(),
                draw_entries: Vec::new(),
                installed: vec![FrameId::default(); count],
                gates: Vec::new(),
                applied: Vec::new(),
            },
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
                layer_sets,
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
