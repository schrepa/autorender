//! Particles (decision 026): emitter records and the stateless particle
//! pipelines. Alpha particles draw into the transparent accumulation targets,
//! additive ones into the frame's color (and glow) in the transparent
//! composite pass. Everything here is created with the first frame that has
//! emitters.

use bytemuck::{Pod, Zeroable};
use engine_core::{ParticleBlend, RenderEmitter};

/// WGSL `Emitter` of `particles.wgsl`: 128 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct EmitterData {
    pub origin: [f32; 3],
    pub rate: f32,
    pub velocity: [f32; 3],
    pub lifetime: f32,
    pub direction: [f32; 3],
    pub speed: f32,
    pub acceleration: [f32; 3],
    /// Cosine of the spread half-angle.
    pub spread: f32,
    pub start_color: [f32; 4],
    pub end_color: [f32; 4],
    pub size: [f32; 2],
    pub emission: f32,
    pub stretch: f32,
    pub radius: f32,
    pub seed: u32,
    pub first: u32,
    pub count: u32,
}

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

/// WGSL `Particles` of `particles.wgsl`: 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct ParticleUniform {
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub size: [f32; 4],
}

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
/// bindings hold); returns the alpha particle count and the particles of
/// emitters that did not fit.
pub(crate) fn pack_emitters(
    emitters: &[RenderEmitter],
    records: &mut Vec<EmitterData>,
    owners: &mut Vec<u32>,
    room: u64,
) -> (u32, u64) {
    records.clear();
    owners.clear();
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
            let index = records.len() as u32;
            records.push(EmitterData::new(emitter, first));
            owners.extend(std::iter::repeat_n(index, emitter.count as usize));
            first += emitter.count;
        }
        if blend == ParticleBlend::Alpha {
            alpha = first;
        }
    }
    (alpha, dropped)
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
        records: Vec<EmitterData>,
        uploaded: Vec<EmitterData>,
        owner_staging: Vec<u32>,
        uploaded_owners: Vec<u32>,
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
                records: Vec::new(),
                uploaded: Vec::new(),
                owner_staging: Vec::new(),
                uploaded_owners: Vec::new(),
                group: Cached::default(),
                buffers_generation: 0,
            }
        }

        pub(crate) fn capacity_bytes(&self) -> u64 {
            self.emitters.size() + self.owners.size()
        }

        /// Uploads `snapshot`'s emitters when they changed; `None` when the
        /// frame draws no particles. `depth` is the depth buffer of
        /// `depth_generation`, sampled for soft particles.
        pub(crate) fn prepare(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            snapshot: &RenderSnapshot,
            size: (u32, u32),
            depth: &wgpu::TextureView,
            depth_generation: u64,
        ) -> Option<ParticleFrame> {
            if !snapshot.environment.particles.enabled || snapshot.emitters.is_empty() {
                return None;
            }
            // One owner entry per particle and one record per emitter, within
            // what a storage binding holds.
            let limits = device.limits();
            let binding = limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size);
            let (alpha, dropped) = pack_emitters(
                &snapshot.emitters,
                &mut self.records,
                &mut self.owner_staging,
                binding / 4,
            );
            let total = self.owner_staging.len() as u32;
            if total == 0 {
                return None;
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
            if ensure_storage(
                device,
                &mut self.owners,
                "engine/particle-owners",
                owner_bytes,
                wgpu::BufferUsages::empty(),
            ) {
                self.uploaded_owners.clear();
                self.buffers_generation += 1;
            }
            if self.records != self.uploaded {
                queue.write_buffer(&self.emitters, 0, bytemuck::cast_slice(&self.records));
                self.uploaded.clone_from(&self.records);
            }
            if self.owner_staging != self.uploaded_owners {
                queue.write_buffer(&self.owners, 0, bytemuck::cast_slice(&self.owner_staging));
                self.uploaded_owners.clone_from(&self.owner_staging);
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
            pass.draw(0..6, 0..frame.alpha);
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
            pass.draw(0..6, frame.alpha..frame.total);
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
                vertex: "vs_particle",
                buffers: &[],
                primitive: wgpu::PrimitiveState::default(),
                // Reversed depth (decision 042).
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
