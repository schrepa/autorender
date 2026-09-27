//! Weighted blended order-independent transparency (decision 026).
//!
//! Transparent surfaces and alpha particles add their weighted color and
//! coverage into an accumulation target and multiply what they let through
//! into a revealage target, in any order; a composite then blends their
//! weighted average over the frame's color. Targets exist only while frames
//! draw something transparent.

use crate::gpu_data::TRANSPARENT_SHADER;

/// Weighted premultiplied color and coverage, added.
pub(crate) const ACCUMULATED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The fraction of the opaque color still visible, multiplied.
pub(crate) const REVEALED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// The accumulation pass's color targets and their blending.
pub(crate) fn accumulation_targets() -> [Option<wgpu::ColorTargetState>; 2] {
    let add = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    let multiply = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::OneMinusSrc,
        operation: wgpu::BlendOperation::Add,
    };
    [
        Some(wgpu::ColorTargetState {
            format: ACCUMULATED_FORMAT,
            blend: Some(wgpu::BlendState {
                color: add,
                alpha: add,
            }),
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: REVEALED_FORMAT,
            blend: Some(wgpu::BlendState {
                color: multiply,
                alpha: multiply,
            }),
            write_mask: wgpu::ColorWrites::ALL,
        }),
    ]
}

struct Targets {
    size: (u32, u32),
    textures: [wgpu::Texture; 2],
    accumulated: wgpu::TextureView,
    revealed: wgpu::TextureView,
    group: wgpu::BindGroup,
}

impl Drop for Targets {
    fn drop(&mut self) {
        for texture in &self.textures {
            texture.destroy();
        }
    }
}

pub(crate) struct TransparentPass {
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    module: wgpu::ShaderModule,
    // Composites into the canvas and into the scene target (leaving glow).
    composite: [Option<wgpu::RenderPipeline>; 2],
    surface_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    targets: Option<Targets>,
}

impl TransparentPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/transparent-composite-layout"),
            entries: &[texture(0), texture(1)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/transparent-composite-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/transparent.wgsl"),
            source: wgpu::ShaderSource::Wgsl(TRANSPARENT_SHADER.into()),
        });
        Self {
            layout,
            pipeline_layout,
            module,
            composite: [None, None],
            surface_format,
            depth_format,
            targets: None,
        }
    }

    /// Sizes the accumulation targets to `size`.
    pub(crate) fn begin(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        if self
            .targets
            .as_ref()
            .is_some_and(|targets| targets.size == size)
        {
            return;
        }
        self.targets = None;
        let texture = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
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
            texture("engine/transparent-accumulated", ACCUMULATED_FORMAT),
            texture("engine/transparent-revealed", REVEALED_FORMAT),
        ];
        let [accumulated, revealed] = textures
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine/transparent-composite-group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&accumulated),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&revealed),
                },
            ],
        });
        self.targets = Some(Targets {
            size,
            textures,
            accumulated,
            revealed,
            group,
        });
    }

    /// Releases the targets while frames draw nothing transparent.
    pub(crate) fn release_targets(&mut self) {
        self.targets = None;
    }

    /// Target bytes: 8 accumulated and 1 revealed per pixel.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.targets.as_ref().map_or(0, |targets| {
            u64::from(targets.size.0) * u64::from(targets.size.1) * 9
        })
    }

    /// The accumulation pass's attachments, cleared to nothing added and
    /// everything revealed.
    pub(crate) fn attachments(&self) -> [Option<wgpu::RenderPassColorAttachment<'_>>; 2] {
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
            attachment(&targets.accumulated, wgpu::Color::TRANSPARENT),
            attachment(&targets.revealed, wgpu::Color::WHITE),
        ]
    }

    /// Creates the composite into the canvas or, with `post`, the scene
    /// target beside the glow target.
    pub(crate) fn ensure_composite(&mut self, device: &wgpu::Device, post: bool) {
        let index = usize::from(post);
        if self.composite[index].is_some() {
            return;
        }
        let over = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            // The opaque share of the ambient light alpha stays.
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let format = if post {
            crate::post::SCENE_FORMAT
        } else {
            self.surface_format
        };
        let mut targets = vec![Some(wgpu::ColorTargetState {
            format,
            blend: Some(over),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        if post {
            targets.push(Some(wgpu::ColorTargetState {
                format: crate::post::GLOW_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::empty(),
            }));
        }
        let constants = [("SURFACE_IS_SRGB", if format.is_srgb() { 1.0 } else { 0.0 })];
        self.composite[index] = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("engine/transparent-composite"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &self.module,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: self.depth_format,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.module,
                    entry_point: Some("fs_composite"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            },
        ));
    }

    /// Draws the composite; call [`Self::ensure_composite`] first.
    pub(crate) fn composite(&self, pass: &mut wgpu::RenderPass<'_>, post: bool) {
        let targets = self.targets.as_ref().expect("sized in begin");
        pass.set_pipeline(self.composite[usize::from(post)].as_ref().expect("ensured"));
        pass.set_bind_group(0, &targets.group, &[]);
        pass.draw(0..3, 0..1);
    }
}
