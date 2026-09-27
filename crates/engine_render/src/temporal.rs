//! Temporal antialiasing (decision 017) of GPU-presented display frames.
//!
//! Each frame rasterizes with a sub-pixel jitter into a scene color target
//! and a motion target, then one full-screen pass blends the scene with the
//! reprojected previous output and writes both the canvas and the next
//! history. Culling alternates between two pose buffers, so the previous
//! frame's displayed poses stay readable for motion vectors without copies.

use engine_core::{RenderFrame, glam::Mat4};

use crate::gpu_data::{
    CameraUniform, ResolveUniform, TEMPORAL_SHADER, jitter_clip, jitter_offset, jittered,
};

/// This frame's linear color; the sRGB encoding keeps 8 bits perceptual.
const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Weight of each frame's color against the history: an exponential average
/// over about ten frames, beyond the eight-sample jitter cycle.
const CURRENT_WEIGHT: f32 = 0.1;

struct Targets {
    size: (u32, u32),
    // Scene, motion and both histories, destroyed with the targets.
    textures: [wgpu::Texture; 4],
    scene: wgpu::TextureView,
    motion: wgpu::TextureView,
    history: [wgpu::TextureView; 2],
    // Frames writing history `p` read history `1 - p` through group `p`.
    groups: [wgpu::BindGroup; 2],
}

impl Drop for Targets {
    fn drop(&mut self) {
        // Browsers otherwise keep the memory until garbage collection;
        // submitted work that still reads them completes first.
        for texture in &self.textures {
            texture.destroy();
        }
    }
}

/// What the last submitted temporal frame leaves for the next one.
#[derive(Clone, Copy)]
struct History {
    // Its unjittered view-projection, the origin of camera motion.
    view_projection: Mat4,
    size: (u32, u32),
    // Retained render identity, structure revision and pose storage
    // generation: previous poses are per slot of that structure.
    identity: u64,
    structure: u64,
    pose_generation: u64,
}

/// One temporal frame's settings, from [`TemporalPass::begin`].
pub(crate) struct TemporalFrame {
    /// Pose buffer and history texture this frame writes; it reads the other.
    pub parity: usize,
    pub camera: CameraUniform,
    /// The jittered projection this frame rasterizes and culls with.
    pub rasterized: Mat4,
    pub history_valid: bool,
    /// The other pose buffer holds this structure's previous-frame poses.
    /// Otherwise objects keep this frame's poses: motion is the camera's.
    pub motion_valid: bool,
    history: History,
}

pub(crate) struct TemporalPass {
    mesh_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    pose_layout: wgpu::BindGroupLayout,
    resolve_pipeline: wgpu::RenderPipeline,
    resolve_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    written: Option<ResolveUniform>,
    targets: Option<Targets>,
    // Pose groups for (current, previous) buffer pairs, index 2 * current +
    // previous, of the pose storage generation they were built for.
    pose_groups: Option<(u64, [wgpu::BindGroup; 4])>,
    history: Option<History>,
    frame: u32,
}

impl TemporalPass {
    /// Creates the pipelines and fixed resources; targets follow the surface.
    /// `mesh_shader` is `meshes.wgsl`, drawn with the retained affine vertex
    /// layouts and the renderer's camera and texture layouts.
    pub(crate) fn new(
        device: &wgpu::Device,
        mesh_shader: &wgpu::ShaderModule,
        camera_layout: &wgpu::BindGroupLayout,
        texture_layout: &wgpu::BindGroupLayout,
        vertex_layouts: &[Option<wgpu::VertexBufferLayout<'_>>],
        depth_format: wgpu::TextureFormat,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let storage_entry = |binding, stride: usize| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(stride as u64),
            },
            count: None,
        };
        let pose_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/temporal-pose-layout"),
            entries: &[
                storage_entry(0, size_of::<crate::gpu_data::PoseData>()),
                storage_entry(1, size_of::<crate::gpu_data::ColorData>()),
                storage_entry(2, size_of::<crate::gpu_data::PoseData>()),
            ],
        });
        let mesh_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/temporal-mesh-pipeline-layout"),
            bind_group_layouts: &[
                Some(camera_layout),
                Some(texture_layout),
                Some(&pose_layout),
            ],
            immediate_size: 0,
        });
        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("engine/temporal-meshes"),
            layout: Some(&mesh_layout),
            vertex: wgpu::VertexState {
                module: mesh_shader,
                entry_point: Some("vs_affine_temporal"),
                compilation_options: Default::default(),
                buffers: vertex_layouts,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: mesh_shader,
                entry_point: Some("fs_temporal"),
                compilation_options: Default::default(),
                targets: &[Some(SCENE_FORMAT.into()), Some(MOTION_FORMAT.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sky_pipeline = crate::browser::sky_pipeline(
            device,
            mesh_shader,
            camera_layout,
            "fs_sky_temporal",
            &[Some(SCENE_FORMAT.into()), Some(MOTION_FORMAT.into())],
            None,
        );
        let texture_entry = |binding, filterable| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/taa-resolve-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(size_of::<ResolveUniform>() as u64),
                    },
                    count: None,
                },
                texture_entry(1, false),
                texture_entry(2, false),
                texture_entry(3, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/temporal.wgsl"),
            source: wgpu::ShaderSource::Wgsl(TEMPORAL_SHADER.into()),
        });
        let resolve_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("engine/taa-resolve-pipeline-layout"),
                bind_group_layouts: &[Some(&resolve_layout)],
                immediate_size: 0,
            });
        let resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("engine/taa-resolve"),
            layout: Some(&resolve_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_resolve"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[(
                        "SURFACE_IS_SRGB",
                        if surface_format.is_srgb() { 1.0 } else { 0.0 },
                    )],
                    ..Default::default()
                },
                targets: &[Some(surface_format.into()), Some(HISTORY_FORMAT.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/taa-history-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/taa-resolve-uniform"),
            size: size_of::<ResolveUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            mesh_pipeline,
            sky_pipeline,
            pose_layout,
            resolve_pipeline,
            resolve_layout,
            sampler,
            uniform,
            written: None,
            targets: None,
            pose_groups: None,
            history: None,
            frame: 0,
        }
    }

    /// Target bytes: scene, motion and two history textures of 4 bytes per
    /// pixel each.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.targets.as_ref().map_or(0, |targets| {
            u64::from(targets.size.0) * u64::from(targets.size.1) * 16
        })
    }

    /// Starts a frame of `size` pixels whose culling writes pose buffer
    /// `parity` of storage `pose_generation`: sizes the targets, decides
    /// what history is valid and writes the resolve uniform.
    pub(crate) fn begin(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &RenderFrame<'_>,
        size: (u32, u32),
        pose_generation: u64,
    ) -> TemporalFrame {
        if self
            .targets
            .as_ref()
            .is_none_or(|targets| targets.size != size)
        {
            // Recreated history starts over.
            self.targets = Some(self.create_targets(device, size));
            self.history = None;
        }
        let view_projection = frame.snapshot().view_projection;
        let history = History {
            view_projection,
            size,
            identity: frame.identity(),
            structure: frame.structure_revision(),
            pose_generation,
        };
        let last = self
            .history
            .filter(|last| last.size == size && last.identity == history.identity);
        let jitter = jitter_clip(jitter_offset(self.frame), size.0, size.1);
        let previous = last.map_or(view_projection, |last| last.view_projection);
        let uniform = ResolveUniform {
            texel: [1.0 / size.0 as f32, 1.0 / size.1 as f32],
            current_weight: CURRENT_WEIGHT,
            history_valid: u32::from(last.is_some()),
        };
        if self
            .written
            .is_none_or(|written| bytemuck::bytes_of(&written) != bytemuck::bytes_of(&uniform))
        {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
            self.written = Some(uniform);
        }
        TemporalFrame {
            parity: (self.frame & 1) as usize,
            camera: CameraUniform::temporal(
                view_projection,
                frame.snapshot().camera_position,
                previous,
                jitter,
                size.0,
                size.1,
            ),
            rasterized: jittered(view_projection, jitter),
            history_valid: last.is_some(),
            motion_valid: last.is_some_and(|last| {
                last.structure == history.structure && last.pose_generation == pose_generation
            }),
            history,
        }
    }

    /// Builds the pose bindings over `poses`, the pair of pose storage
    /// `generation`, and `colors`, unless built for that generation.
    pub(crate) fn prepare_pose_groups(
        &mut self,
        device: &wgpu::Device,
        generation: u64,
        poses: [&wgpu::Buffer; 2],
        colors: &wgpu::Buffer,
    ) {
        if self
            .pose_groups
            .as_ref()
            .is_none_or(|(built, _)| *built != generation)
        {
            let group = |current: usize, previous: usize| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("engine/temporal-pose-group"),
                    layout: &self.pose_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: poses[current].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: colors.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: poses[previous].as_entire_binding(),
                        },
                    ],
                })
            };
            self.pose_groups = Some((
                generation,
                [group(0, 0), group(0, 1), group(1, 0), group(1, 1)],
            ));
        }
    }

    /// Pose bindings of `frame`: its pose buffer, the colors, and the
    /// previous frame's poses when valid (otherwise its own again).
    pub(crate) fn pose_group(&self, frame: &TemporalFrame) -> &wgpu::BindGroup {
        let previous = if frame.motion_valid {
            1 - frame.parity
        } else {
            frame.parity
        };
        let (_, groups) = self.pose_groups.as_ref().expect("prepared this frame");
        &groups[2 * frame.parity + previous]
    }

    pub(crate) fn mesh_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.mesh_pipeline
    }

    pub(crate) fn sky_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.sky_pipeline
    }

    /// The scene color and motion attachments of the opaque pass.
    pub(crate) fn attachments(
        &self,
        clear: wgpu::Color,
    ) -> [Option<wgpu::RenderPassColorAttachment<'_>>; 2] {
        let targets = self.targets.as_ref().expect("sized in begin");
        let attachment = |view, clear| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })
        };
        [
            attachment(&targets.scene, clear),
            attachment(&targets.motion, wgpu::Color::TRANSPARENT),
        ]
    }

    /// Records the resolve of `frame` into `canvas` and its history texture.
    pub(crate) fn encode_resolve(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        canvas: &wgpu::TextureView,
        frame: &TemporalFrame,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let targets = self.targets.as_ref().expect("sized in begin");
        // The triangle covers every pixel: clears only spare loading them.
        let attachment = |view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/taa-resolve"),
            color_attachments: &[
                attachment(canvas),
                attachment(&targets.history[frame.parity]),
            ],
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.resolve_pipeline);
        pass.set_bind_group(0, &targets.groups[frame.parity], &[]);
        pass.draw(0..3, 0..1);
    }

    /// Records a submitted frame as the next frame's history.
    pub(crate) fn submitted(&mut self, frame: &TemporalFrame) {
        self.history = Some(frame.history);
        self.frame = self.frame.wrapping_add(1);
    }

    /// A frame drawn without temporal antialiasing ends the history.
    pub(crate) fn reset(&mut self) {
        self.history = None;
    }

    fn create_targets(&self, device: &wgpu::Device, (width, height): (u32, u32)) -> Targets {
        let texture = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let textures = [
            texture("engine/taa-scene", SCENE_FORMAT),
            texture("engine/taa-motion", MOTION_FORMAT),
            texture("engine/taa-history", HISTORY_FORMAT),
            texture("engine/taa-history", HISTORY_FORMAT),
        ];
        let [scene, motion, first, second] = textures
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
        let history = [first, second];
        let group = |previous: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine/taa-resolve-group"),
                layout: &self.resolve_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&scene),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&motion),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(previous),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            })
        };
        let groups = [group(&history[1]), group(&history[0])];
        Targets {
            size: (width, height),
            textures,
            scene,
            motion,
            history,
            groups,
        }
    }
}
