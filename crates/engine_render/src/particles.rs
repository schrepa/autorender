//! Particles: emitter records and the stateless particle
//! pipelines. Alpha particles draw into the transparent accumulation targets,
//! additive ones into the frame's color (and glow) in the transparent
//! composite pass. Everything here is created with the first frame that has
//! emitters.

use engine_core::{ParticleBlend, RenderEmitter};

pub(crate) use crate::gpu_layouts::particles::EmitterData;

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

pub(crate) use crate::gpu_layouts::particles::ParticleUniform;

impl ParticleUniform {
    /// Billboard axes of the unjittered `view_projection`: its first two
    /// rows are the view's right and up axes scaled by the projection.
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

/// Emitter records in draw order: alpha emitters first, then additive ones,
/// with the instances of each, up to `room` particles (what the GPU's
/// bindings hold); returns the alpha and total particle counts and the
/// particles of emitters that did not fit.
pub(crate) fn pack_emitters(
    emitters: &[RenderEmitter],
    records: &mut Vec<EmitterData>,
    room: u64,
) -> (u32, u32, u64) {
    records.clear();
    let mut first = 0_u32;
    let mut alpha = 0;
    let mut dropped = 0;
    for blend in [ParticleBlend::Alpha, ParticleBlend::Additive] {
        for emitter in emitters
            .iter()
            .filter(|emitter| emitter.emitter.blend == blend)
        {
            if u64::from(first) + u64::from(emitter.count) > room {
                dropped += u64::from(emitter.count);
                continue;
            }
            records.push(EmitterData::new(emitter, first));
            first += emitter.count;
        }
        if blend == ParticleBlend::Alpha {
            alpha = first;
        }
    }
    (alpha, first, dropped)
}

/// The emitter of each instance of `records`.
fn pack_owners(records: &[EmitterData], owners: &mut Vec<u32>) {
    owners.clear();
    for (index, record) in records.iter().enumerate() {
        owners.extend(std::iter::repeat_n(index as u32, record.count as usize));
    }
}

/// Whether `a` and `b` give their emitters the same instances.
fn same_instances(a: &[EmitterData], b: &[EmitterData]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| a.first == b.first && a.count == b.count)
}

#[cfg(target_arch = "wasm32")]
mod gpu {
    use engine_core::RenderSnapshot;

    use super::*;
    use crate::bindings::PARTICLES;
    use crate::buffers::{ensure_storage, entry, storage};
    use crate::cached::Cached;
    use crate::gpu_data::PARTICLE_SHADER;

    /// A frame's particles: alpha instances first, then additive ones, and
    /// the particles of emitters beyond the GPU's bindings.
    pub(crate) struct ParticleFrame {
        pub alpha: u32,
        pub total: u32,
        pub dropped: u64,
    }

    pub(crate) struct ParticlePass {
        layout: wgpu::BindGroupLayout,
        pipeline_layout: wgpu::PipelineLayout,
        module: wgpu::ShaderModule,
        alpha: wgpu::RenderPipeline,
        // Additive pipelines into the canvas and into the scene and glow
        // targets, created when first used.
        additive: [Option<wgpu::RenderPipeline>; 2],
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        uniform: wgpu::Buffer,
        written: Option<ParticleUniform>,
        emitters: wgpu::Buffer,
        owners: wgpu::Buffer,
        // Particles a storage binding holds, an owner entry each.
        room: u64,
        records: Vec<EmitterData>,
        uploaded: Vec<EmitterData>,
        owner_staging: Vec<u32>,
        // The render stream and emitters revision the uploads were packed
        // from, and that packing's alpha, total and dropped particles.
        packed: Option<(u64, u64)>,
        counts: (u32, u32, u64),
        group: Cached<(u64, u64), wgpu::BindGroup>,
        buffers_generation: u64,
    }

    impl ParticlePass {
        /// `camera_layout` is group 0 of the mesh pipelines.
        pub(crate) fn new(
            device: &wgpu::Device,
            camera_layout: &wgpu::BindGroupLayout,
            surface_format: wgpu::TextureFormat,
            depth_format: wgpu::TextureFormat,
            transparent_targets: &[Option<wgpu::ColorTargetState>],
        ) -> Self {
            let layout = crate::pipelines::layout(device, &PARTICLES);
            let limits = device.limits();
            let room = limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size)
                / 4;
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("engine/particle-pipeline-layout"),
                bind_group_layouts: &[Some(camera_layout), Some(&layout)],
                immediate_size: 0,
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("engine/particles.wgsl"),
                source: wgpu::ShaderSource::Wgsl(PARTICLE_SHADER.into()),
            });
            let alpha = pipeline(
                device,
                &pipeline_layout,
                &module,
                "fs_particle",
                transparent_targets,
                depth_format,
                &[],
            );
            Self {
                layout,
                pipeline_layout,
                module,
                alpha,
                additive: [None, None],
                surface_format,
                depth_format,
                uniform: crate::buffers::uniform(
                    device,
                    "engine/particle-uniform",
                    size_of::<ParticleUniform>() as u64,
                ),
                written: None,
                emitters: storage(
                    device,
                    "engine/particle-emitters",
                    128,
                    wgpu::BufferUsages::empty(),
                ),
                owners: storage(
                    device,
                    "engine/particle-owners",
                    4,
                    wgpu::BufferUsages::empty(),
                ),
                room,
                records: Vec::new(),
                uploaded: Vec::new(),
                owner_staging: Vec::new(),
                packed: None,
                counts: (0, 0, 0),
                group: Cached::default(),
                buffers_generation: 0,
            }
        }

        pub(crate) fn capacity_bytes(&self) -> u64 {
            self.emitters.size() + self.owners.size()
        }

        /// Uploads `snapshot`'s emitters when `emitters` (the render stream
        /// identity and its emitters revision) changed; `None` when the frame
        /// draws no particles. `depth` is the depth buffer of
        /// `depth_generation`, sampled for soft particles.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn prepare(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            snapshot: &RenderSnapshot,
            emitters: (u64, u64),
            size: (u32, u32),
            depth: &wgpu::TextureView,
            depth_generation: u64,
        ) -> Option<ParticleFrame> {
            if !snapshot.environment.particles.enabled || snapshot.emitters.is_empty() {
                return None;
            }
            if self.packed != Some(emitters) {
                self.pack(device, queue, snapshot);
                self.packed = Some(emitters);
            }
            let (alpha, total, dropped) = self.counts;
            if total == 0 {
                return None;
            }
            let uniform = ParticleUniform::new(snapshot.view_projection, size.0, size.1);
            if self.written != Some(uniform) {
                queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
                self.written = Some(uniform);
            }
            let key = (depth_generation, self.buffers_generation);
            if self.group.stale(&key) {
                let group = crate::pipelines::group(
                    device,
                    "engine/particle-group",
                    &PARTICLES,
                    &self.layout,
                    &[
                        entry("particles", &self.uniform),
                        entry("emitters", &self.emitters),
                        entry("owners", &self.owners),
                        ("depth_texture", crate::pipelines::view(depth)),
                    ],
                );
                self.group.set(key, group);
            }
            Some(ParticleFrame {
                alpha,
                total,
                dropped,
            })
        }

        /// Packs and uploads the snapshot's emitter records, one owner entry
        /// per particle and one record per emitter, within what a storage
        /// binding holds: owners only when emitters' instances changed.
        fn pack(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, snapshot: &RenderSnapshot) {
            self.counts = pack_emitters(&snapshot.emitters, &mut self.records, self.room);
            let total = self.counts.1;
            if total == 0 {
                return;
            }
            let emitter_bytes = (self.records.len() * size_of::<EmitterData>()) as u64;
            if ensure_storage(
                device,
                &mut self.emitters,
                "engine/particle-emitters",
                emitter_bytes,
                wgpu::BufferUsages::empty(),
            ) {
                self.uploaded.clear();
                self.buffers_generation += 1;
            }
            let owner_bytes = u64::from(total) * 4;
            let owners_replaced = ensure_storage(
                device,
                &mut self.owners,
                "engine/particle-owners",
                owner_bytes,
                wgpu::BufferUsages::empty(),
            );
            if owners_replaced {
                self.buffers_generation += 1;
            }
            if owners_replaced || !same_instances(&self.records, &self.uploaded) {
                pack_owners(&self.records, &mut self.owner_staging);
                queue.write_buffer(&self.owners, 0, bytemuck::cast_slice(&self.owner_staging));
            }
            if self.records != self.uploaded {
                queue.write_buffer(&self.emitters, 0, bytemuck::cast_slice(&self.records));
                self.uploaded.clone_from(&self.records);
            }
        }

        /// Creates the additive pipeline into the canvas or, with `post`, the
        /// scene and glow targets.
        pub(crate) fn ensure_additive(&mut self, device: &wgpu::Device, post: bool) {
            let index = usize::from(post);
            if self.additive[index].is_some() {
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
            let pipeline = if post {
                pipeline(
                    device,
                    &self.pipeline_layout,
                    &self.module,
                    "fs_particle_add_post",
                    &[
                        target(crate::post::SCENE_FORMAT, add),
                        target(crate::post::GLOW_FORMAT, brightest),
                    ],
                    self.depth_format,
                    &[],
                )
            } else {
                pipeline(
                    device,
                    &self.pipeline_layout,
                    &self.module,
                    "fs_particle_add",
                    &[target(self.surface_format, add)],
                    self.depth_format,
                    &[],
                )
            };
            self.additive[index] = Some(pipeline);
        }

        /// Draws alpha particles into the transparent accumulation targets.
        pub(crate) fn draw_alpha(
            &self,
            pass: &mut wgpu::RenderPass<'_>,
            camera: &wgpu::BindGroup,
            frame: &ParticleFrame,
        ) {
            if frame.alpha == 0 {
                return;
            }
            pass.set_pipeline(&self.alpha);
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(1, self.group.get().expect("prepared this frame"), &[]);
            pass.draw(0..4, 0..frame.alpha);
        }

        /// Draws additive particles into the frame's color; call
        /// [`Self::ensure_additive`] first.
        pub(crate) fn draw_additive(
            &self,
            pass: &mut wgpu::RenderPass<'_>,
            camera: &wgpu::BindGroup,
            frame: &ParticleFrame,
            post: bool,
        ) {
            if frame.total == frame.alpha {
                return;
            }
            pass.set_pipeline(self.additive[usize::from(post)].as_ref().expect("ensured"));
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(1, self.group.get().expect("prepared this frame"), &[]);
            pass.draw(0..4, frame.alpha..frame.total);
        }
    }

    fn pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        fragment: &str,
        targets: &[Option<wgpu::ColorTargetState>],
        depth_format: wgpu::TextureFormat,
        constants: &[(&str, f64)],
    ) -> wgpu::RenderPipeline {
        crate::pipelines::geometry(
            device,
            &crate::pipelines::Geometry {
                label: "engine/particles",
                layout,
                module,
                vertex_constants: &[],
                vertex: "vs_particle",
                buffers: &[],
                // One four-vertex strip per instance.
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                // Reversed depth.
                depth: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                fragment: Some(crate::pipelines::Fragment {
                    entry: fragment,
                    constants,
                    targets,
                }),
            },
        )
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use gpu::{ParticleFrame, ParticlePass};
