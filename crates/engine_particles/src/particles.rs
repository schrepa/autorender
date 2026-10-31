#![expect(clippy::expect_used, reason = "drawing reads what this frame prepared")]

use crate::pipeline_layouts::PARTICLE_PIPELINES;
use crate::shaders::entry::particles;
use engine_core::{
    Cut, ParticleBlend, ParticleSettings, ParticleStats, RenderEmitter, RenderStats,
};
use engine_render::bindings::GroupRef;
use engine_render::graph::{GraphTable, ResourceId};
use engine_render::parts::{
    Begin, BeginOut, Create, DrawAt, GpuPart, Hook, PartDecl, PartEntry, Prelude, Step, View,
};
use engine_render::transparent::accumulation_targets;

pub(crate) use crate::gpu_layouts::EmitterData;

impl EmitterData {
    pub(crate) fn new(emitter: &RenderEmitter, first: u32) -> Self {
        let settings = &emitter.emitter;
        Self {
            origin: emitter.origin.to_array(),
            rate: settings.rate,
            velocity: emitter.velocity.to_array(),
            lifetime: settings.lifetime,
            direction: emitter.direction.to_array(),
            speed: settings.speed,
            acceleration: settings.acceleration.to_array(),
            spread: engine_core::math::cos(engine_core::math::clamp(
                settings.spread,
                0.0,
                std::f32::consts::PI,
            )),
            start_color: settings.color[0],
            end_color: settings.color[1],
            size: settings.size,
            emission: settings.emission,
            stretch: settings.stretch,
            radius: settings.radius,
            seed: settings.seed,
            first,
            count: emitter.count,
        }
    }
}

pub(crate) use crate::gpu_layouts::ParticleUniform;

impl ParticleUniform {
    pub(crate) fn new(view_projection: engine_core::glam::Mat4, width: u32, height: u32) -> Self {
        let axis = |row: usize| {
            view_projection
                .row(row)
                .truncate()
                .try_normalize()
                .unwrap_or(engine_core::glam::Vec3::X)
                .extend(0.0)
                .to_array()
        };
        let (width, height) = (width.max(1) as f32, height.max(1) as f32);
        Self {
            right: axis(0),
            up: axis(1),
            size: [width, height, 1.0 / width, 1.0 / height],
        }
    }
}

pub(crate) fn pack_emitters(
    emitters: &[RenderEmitter],
    records: &mut Vec<EmitterData>,
    room: u64,
) -> (u32, u32, u32, u64) {
    records.clear();
    let mut first = 0_u32;
    let (mut alpha, mut mist) = (0, 0);
    let mut cut = 0;
    for blend in [
        ParticleBlend::Alpha,
        ParticleBlend::Mist,
        ParticleBlend::Additive,
    ] {
        for emitter in emitters
            .iter()
            .filter(|emitter| emitter.emitter.blend == blend)
        {
            if u64::from(first) + u64::from(emitter.count) > room {
                cut += u64::from(emitter.count);
                continue;
            }
            records.push(EmitterData::new(emitter, first));
            first += emitter.count;
        }
        match blend {
            ParticleBlend::Alpha => alpha = first,
            ParticleBlend::Mist => mist = first,
            ParticleBlend::Additive => {}
        }
    }
    (alpha, mist, first, cut)
}

fn pack_owners(records: &[EmitterData], owners: &mut Vec<u32>) {
    owners.clear();
    for (index, record) in records.iter().enumerate() {
        owners.extend(std::iter::repeat_n(index as u32, record.count as usize));
    }
}

fn same_instances(a: &[EmitterData], b: &[EmitterData]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| a.first == b.first && a.count == b.count)
}

mod gpu {
    use engine_core::RenderSnapshot;

    use super::*;
    use crate::pipeline_layouts::PARTICLES;
    use engine_render::buffers::Growable;
    use engine_render::cached::{Bound, GroupCache, Versioned};

    pub(crate) struct ParticleFrame {
        pub alpha: u32,
        pub mist: u32,
        pub total: u32,
        pub cut: u64,
    }

    pub(crate) struct ParticlePass {
        layout: engine_render::pipelines::GroupLayout,
        pipeline_layout: wgpu::PipelineLayout,
        alpha: wgpu::RenderPipeline,
        mist: Option<wgpu::RenderPipeline>,
        transparent_targets: Vec<Option<wgpu::ColorTargetState>>,
        additive: Option<wgpu::RenderPipeline>,
        depth_format: wgpu::TextureFormat,
        uniform: wgpu::Buffer,
        written: Option<ParticleUniform>,
        emitters: Growable,
        owners: Growable,
        records: Vec<EmitterData>,
        uploaded: Vec<EmitterData>,
        owner_staging: Vec<u32>,
        packed: Option<(u64, u64)>,
        counts: (u32, u32, u32, u64),
        group: GroupCache<4>,
    }

    impl ParticlePass {
        pub(crate) fn new(
            device: &wgpu::Device,
            depth_format: wgpu::TextureFormat,
            transparent_targets: &[Option<wgpu::ColorTargetState>],
        ) -> Self {
            let layout = engine_render::pipelines::layout(device, &PARTICLES);
            let pipeline_layout = engine_render::pipelines::layout_of(device, &PARTICLE_PIPELINES);
            let alpha = pipeline(
                device,
                &pipeline_layout,
                particles::fs_particle,
                transparent_targets,
                depth_format,
                &[],
            );
            Self {
                layout,
                pipeline_layout,
                alpha,
                mist: None,
                transparent_targets: transparent_targets.to_vec(),
                additive: None,
                depth_format,
                uniform: engine_render::buffers::uniform(
                    device,
                    "engine/particle-uniform",
                    size_of::<ParticleUniform>() as u64,
                ),
                written: None,
                emitters: Growable::new(
                    device,
                    "engine/particle-emitters",
                    128,
                    wgpu::BufferUsages::empty(),
                ),
                owners: Growable::new(
                    device,
                    "engine/particle-owners",
                    4,
                    wgpu::BufferUsages::empty(),
                ),
                records: Vec::new(),
                uploaded: Vec::new(),
                owner_staging: Vec::new(),
                packed: None,
                counts: (0, 0, 0, 0),
                group: GroupCache::default(),
            }
        }

        pub(crate) fn capacity_bytes(&self) -> u64 {
            self.emitters.size() + self.owners.size()
        }

        pub(crate) fn prepare(
            &mut self,
            device: &wgpu::Device,
            queue: &engine_render::gpu_calls::Queue,
            snapshot: &RenderSnapshot,
            emitters: (u64, u64),
            size: engine_render::targets::FrameSize,
            depth: &Versioned<wgpu::TextureView>,
        ) -> Option<ParticleFrame> {
            if snapshot.particles.list.is_empty() {
                return None;
            }
            if self.packed != Some(emitters) {
                self.pack(device, queue, snapshot);
                self.packed = Some(emitters);
            }
            let (alpha, mist, total, cut) = self.counts;
            if total == 0 {
                return None;
            }
            if mist > alpha && self.mist.is_none() {
                self.mist = Some(pipeline(
                    device,
                    &self.pipeline_layout,
                    particles::fs_particle_mist,
                    &self.transparent_targets,
                    self.depth_format,
                    &[],
                ));
            }
            let uniform = ParticleUniform::new(snapshot.view_projection, size.width, size.height);
            if self.written != Some(uniform) {
                queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
                self.written = Some(uniform);
            }
            self.group.bind(
                device,
                "engine/particle-group",
                (&PARTICLES, &self.layout),
                [
                    ("particles", Bound::fixed(self.uniform.as_entire_binding())),
                    ("emitters", self.emitters.bound()),
                    ("owners", self.owners.bound()),
                    ("depth_texture", depth.bound()),
                ],
            );
            Some(ParticleFrame {
                alpha,
                mist,
                total,
                cut,
            })
        }

        fn pack(
            &mut self,
            device: &wgpu::Device,
            queue: &engine_render::gpu_calls::Queue,
            snapshot: &RenderSnapshot,
        ) {
            let room = self.owners.room() / 4;
            self.counts = pack_emitters(&snapshot.particles.list, &mut self.records, room);
            let total = self.counts.2;
            if total == 0 {
                return;
            }
            let emitter_bytes = (self.records.len() * size_of::<EmitterData>()) as u64;
            if self.emitters.grow(device, emitter_bytes) {
                self.uploaded.clear();
            }
            let owner_bytes = u64::from(total) * 4;
            let owners_replaced = self.owners.grow(device, owner_bytes);
            if owners_replaced || !same_instances(&self.records, &self.uploaded) {
                pack_owners(&self.records, &mut self.owner_staging);
                queue.write_buffer(&self.owners, 0, bytemuck::cast_slice(&self.owner_staging));
            }
            if self.records != self.uploaded {
                queue.write_buffer(&self.emitters, 0, bytemuck::cast_slice(&self.records));
                self.uploaded.clone_from(&self.records);
            }
        }

        pub(crate) fn ensure_additive(&mut self, device: &wgpu::Device) {
            if self.additive.is_some() {
                return;
            }
            let add = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            };
            let brightest = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Max,
                },
                alpha: wgpu::BlendComponent::REPLACE,
            };
            let target = |format, blend| {
                Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })
            };
            self.additive = Some(pipeline(
                device,
                &self.pipeline_layout,
                particles::fs_particle_add_post,
                &[
                    target(engine_render::post::SCENE_FORMAT, add),
                    target(engine_render::post::GLOW_FORMAT, brightest),
                ],
                self.depth_format,
                &[],
            ));
        }

        pub(crate) fn draw_alpha(
            &self,
            pass: &mut wgpu::RenderPass<'_>,
            camera: &wgpu::BindGroup,
            frame: &ParticleFrame,
        ) {
            if frame.mist == 0 {
                return;
            }
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(1, self.group.get().expect("prepared_this_frame"), &[]);
            if frame.alpha > 0 {
                pass.set_pipeline(&self.alpha);
                pass.draw(0..4, 0..frame.alpha);
            }
            if let (true, Some(mist)) = (frame.mist > frame.alpha, &self.mist) {
                pass.set_pipeline(mist);
                pass.draw(0..4, frame.alpha..frame.mist);
            }
        }

        pub(crate) fn draw_additive(
            &self,
            pass: &mut wgpu::RenderPass<'_>,
            camera: &wgpu::BindGroup,
            frame: &ParticleFrame,
        ) {
            if frame.total == frame.mist {
                return;
            }
            pass.set_pipeline(self.additive.as_ref().expect("ensured"));
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(1, self.group.get().expect("prepared_this_frame"), &[]);
            pass.draw(0..4, frame.mist..frame.total);
        }
    }

    fn pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        fragment: engine_render::shaders::Entry,
        targets: &[Option<wgpu::ColorTargetState>],
        depth_format: wgpu::TextureFormat,
        constants: &[(&str, f64)],
    ) -> wgpu::RenderPipeline {
        engine_render::pipelines::geometry(
            device,
            &engine_render::pipelines::Geometry {
                label: "engine/particles",
                layout,
                vertex_constants: &[],
                vertex: particles::vs_particle,
                buffers: &[],
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                fragment: Some(engine_render::pipelines::Fragment {
                    entry: fragment,
                    constants,
                    targets,
                }),
            },
        )
    }
}

pub(crate) use gpu::{ParticleFrame, ParticlePass};

pub static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "particles",
        brick: "particles",
        passes: &[],
        resources: &[],
        hooks: &[Hook::Transparent],
        contributes: &[],
        labels: &[],
        lazy: false,
        needs: &[],
        piece: None,
        resolve: None,
    },
    create,
};

fn create(_: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(Particles::default())
}

#[derive(Clone, Copy, Default)]
struct Ids {
    accumulated: Option<ResourceId>,
    revealed: Option<ResourceId>,
}

#[derive(Default)]
struct Particles {
    particles: Option<ParticlePass>,
    frame: Option<ParticleFrame>,
    emits: bool,
    ids: Ids,
}

impl Particles {
    fn prepare_frame(&mut self, cx: &Step<'_>) -> Option<ParticleFrame> {
        let (device, queue) = (&cx.core.device, &cx.core.queue);
        let snapshot = cx.frame.snapshot;
        let emitters = (
            cx.frame.frame.identity(),
            cx.frame.frame.emitters_revision(),
        );
        let (region, depth) = (cx.core.display.region, &cx.core.depth.view);
        let depth_format = cx.core.depth.texture.format();
        let particles = self.particles.get_or_insert_with(|| {
            ParticlePass::new(device, depth_format, &accumulation_targets())
        });
        let prepared = particles.prepare(device, queue, snapshot, emitters, region.render, depth);
        if prepared.is_some() {
            particles.ensure_additive(device);
        }
        prepared
    }
}

impl GpuPart for Particles {
    fn install(&mut self, table: &GraphTable) {
        self.ids = Ids {
            accumulated: table.resource("accumulated"),
            revealed: table.resource("revealed"),
        };
    }

    fn begin(&mut self, cx: &mut Begin<'_>, _out: &mut BeginOut) {
        let snapshot = cx.snapshot;
        self.emits =
            snapshot.section::<ParticleSettings>().enabled && !snapshot.particles.list.is_empty();
        if !self.emits {
            self.particles = None;
        }
    }

    fn wants(&self, resource: ResourceId) -> bool {
        self.emits && self.ids.accumulated == Some(resource)
    }

    fn prepare(&mut self, cx: &mut Step<'_>, _prelude: &mut Prelude) {
        self.frame = None;
        if cx.frame.kind.debug || !self.emits {
            return;
        }
        self.frame = self.prepare_frame(cx);
    }

    fn produces(&self, resource: ResourceId) -> bool {
        self.frame.is_some()
            && (self.ids.accumulated == Some(resource) || self.ids.revealed == Some(resource))
    }

    fn draw(&self, at: DrawAt<'_>, cx: &View<'_>, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        let (DrawAt::Transparent { accumulated, .. }, Some(frame), Some(particles)) =
            (at, &self.frame, &self.particles)
        else {
            return 0;
        };
        let Some(&GroupRef::Shared { group, without }) = PARTICLE_PIPELINES.groups.first() else {
            return 0;
        };
        let camera = cx.core.shared_group(group, without);
        if accumulated {
            particles.draw_alpha(pass, camera, frame);
        } else {
            particles.draw_additive(pass, camera, frame);
        }
        0
    }

    fn capacity_bytes(&self) -> u64 {
        self.particles
            .as_ref()
            .map_or(0, ParticlePass::capacity_bytes)
    }

    fn stats(&self, cx: &View<'_>, stats: &mut RenderStats) {
        let snapshot = cx.frame.snapshot;
        let particles = self.frame.as_ref();
        let drawn = particles.map_or(0, |frame| frame.total);
        let cut = (snapshot.particles.cut as u64 + particles.map_or(0, |frame| frame.cut))
            .min(u64::from(u32::MAX)) as u32;
        stats.particles = ParticleStats {
            drawn,
            invalid: snapshot.particles.invalid.min(u32::MAX as usize) as u32,
            particle_cut: Cut::of(drawn.saturating_add(cut), drawn),
        };
    }
}
