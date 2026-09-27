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
/// with the instances of each; returns the alpha particle count.
pub(crate) fn pack_emitters(
    emitters: &[RenderEmitter],
    records: &mut Vec<EmitterData>,
    owners: &mut Vec<u32>,
) -> u32 {
    records.clear();
    owners.clear();
    let mut first = 0;
    let mut alpha = 0;
    for blend in [ParticleBlend::Alpha, ParticleBlend::Additive] {
        for emitter in emitters
            .iter()
            .filter(|emitter| emitter.emitter.blend == blend)
        {
            let index = records.len() as u32;
            records.push(EmitterData::new(emitter, first));
            owners.extend(std::iter::repeat_n(index, emitter.count as usize));
            first += emitter.count;
        }
        if blend == ParticleBlend::Alpha {
            alpha = first;
        }
    }
    alpha
}

#[cfg(target_arch = "wasm32")]
mod gpu {
    use engine_core::RenderSnapshot;

    use super::*;
    use crate::gpu_data::PARTICLE_SHADER;

    /// A frame's particles: alpha instances first, then additive ones.
    pub(crate) struct ParticleFrame {
        pub alpha: u32,
        pub total: u32,
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
        group: Option<((u64, u64), wgpu::BindGroup)>,
        buffers_generation: u64,
    }

    fn storage(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(16).next_multiple_of(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
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
            let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
            let read = wgpu::BufferBindingType::Storage { read_only: true };
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("engine/particle-layout"),
                entries: &[
                    buffer(0, wgpu::BufferBindingType::Uniform),
                    buffer(1, read),
                    buffer(2, read),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
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
                uniform: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("engine/particle-uniform"),
                    size: size_of::<ParticleUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                written: None,
                emitters: storage(device, "engine/particle-emitters", 128),
                owners: storage(device, "engine/particle-owners", 4),
                records: Vec::new(),
                uploaded: Vec::new(),
                owner_staging: Vec::new(),
                uploaded_owners: Vec::new(),
                group: None,
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
            if !snapshot.particles.enabled || snapshot.emitters.is_empty() {
                return None;
            }
            let alpha = pack_emitters(
                &snapshot.emitters,
                &mut self.records,
                &mut self.owner_staging,
            );
            let total = self.owner_staging.len() as u32;
            if total == 0 {
                return None;
            }
            let emitter_bytes = (self.records.len() * size_of::<EmitterData>()) as u64;
            if emitter_bytes > self.emitters.size() {
                self.emitters = storage(
                    device,
                    "engine/particle-emitters",
                    emitter_bytes.next_power_of_two(),
                );
                self.uploaded.clear();
                self.buffers_generation += 1;
            }
            let owner_bytes = u64::from(total) * 4;
            if owner_bytes > self.owners.size() {
                self.owners = storage(
                    device,
                    "engine/particle-owners",
                    owner_bytes.next_power_of_two(),
                );
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
            if self.group.as_ref().is_none_or(|(built, _)| *built != key) {
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("engine/particle-group"),
                    layout: &self.layout,
                    entries: &[
                        buffer_entry(0, &self.uniform),
                        buffer_entry(1, &self.emitters),
                        buffer_entry(2, &self.owners),
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(depth),
                        },
                    ],
                });
                self.group = Some((key, group));
            }
            Some(ParticleFrame { alpha, total })
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
                let srgb = if self.surface_format.is_srgb() {
                    1.0
                } else {
                    0.0
                };
                pipeline(
                    device,
                    &self.pipeline_layout,
                    &self.module,
                    "fs_particle_add",
                    &[target(self.surface_format, add)],
                    self.depth_format,
                    &[("SURFACE_IS_SRGB", srgb)],
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
            pass.set_bind_group(1, &self.group.as_ref().expect("prepared this frame").1, &[]);
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
            pass.set_bind_group(1, &self.group.as_ref().expect("prepared this frame").1, &[]);
            pass.draw(0..6, frame.alpha..frame.total);
        }
    }

    fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: buffer.as_entire_binding(),
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
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("engine/particles"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_particle"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some(fragment),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    ..Default::default()
                },
                targets,
            }),
            multiview_mask: None,
            cache: None,
        })
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use gpu::ParticlePass;

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{ParticleEmitter, glam::Vec3};

    #[test]
    fn emitters_pack_alpha_first_with_contiguous_instances() {
        let emitter = |blend, count| RenderEmitter {
            origin: Vec3::ZERO,
            velocity: Vec3::X,
            direction: Vec3::Y,
            emitter: ParticleEmitter {
                blend,
                ..ParticleEmitter::default()
            },
            count,
        };
        let emitters = [
            emitter(ParticleBlend::Additive, 3),
            emitter(ParticleBlend::Alpha, 2),
            emitter(ParticleBlend::Alpha, 4),
        ];
        let (mut records, mut owners) = (Vec::new(), Vec::new());
        let alpha = pack_emitters(&emitters, &mut records, &mut owners);
        assert_eq!(alpha, 6);
        assert_eq!(owners, [0, 0, 1, 1, 1, 1, 2, 2, 2]);
        assert_eq!(
            records
                .iter()
                .map(|record| (record.first, record.count))
                .collect::<Vec<_>>(),
            [(0, 2), (2, 4), (6, 3)]
        );
        assert_eq!(size_of::<EmitterData>(), 128);
        assert_eq!(size_of::<ParticleUniform>(), 48);
        // A straight cone has cosine 1; a full sphere, -1.
        assert_eq!(records[0].spread, engine_core::math::cos(0.3));
    }

    #[test]
    fn billboards_face_the_view_of_the_unjittered_projection() {
        let camera = engine_core::Camera {
            position: Vec3::new(0.0, 5.0, 10.0),
            target: Vec3::ZERO,
            ..engine_core::Camera::default()
        };
        let uniform = ParticleUniform::new(camera.view_projection(2.0), 800, 400);
        let right = Vec3::from_slice(&uniform.right[..3]);
        let up = Vec3::from_slice(&uniform.up[..3]);
        let forward = (camera.target - camera.position).normalize();
        assert!((right - Vec3::X).length() < 1.0e-5);
        assert!(right.dot(forward).abs() < 1.0e-5 && up.dot(forward).abs() < 1.0e-5);
        assert!(up.y > 0.0);
        assert_eq!(uniform.size, [800.0, 400.0, 1.0 / 800.0, 1.0 / 400.0]);
    }
}
