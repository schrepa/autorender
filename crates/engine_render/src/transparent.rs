//! Weighted blended order-independent transparency.
//!
//! Transparent surfaces and alpha particles add their weighted color and
//! coverage into an accumulation target and multiply what they let through
//! into a revealage target, in any order; a composite then blends their
//! weighted average over the frame's color. Targets exist only while frames
//! draw something transparent.

use crate::bindings::TRANSPARENT_COMPOSITE;
use crate::gpu_data::TRANSPARENT_SHADER;
use crate::shaders::entry::transparent;
use crate::targets::{FrameSize, ScreenTarget};

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
    targets: ScreenTarget<Targets>,
}

impl TransparentPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let layout = crate::pipelines::layout(device, &TRANSPARENT_COMPOSITE);
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
            targets: ScreenTarget::default(),
        }
    }

    /// Sizes the accumulation targets to `size`.
    pub(crate) fn begin(&mut self, device: &wgpu::Device, size: FrameSize) {
        if !self.targets.resizes(size) {
            return;
        }
        let texture = |label, format| {
            crate::targets::texture_2d(
                device,
                label,
                size.pixels(),
                1,
                1,
                format,
                crate::targets::RENDER_TARGET,
            )
        };
        let textures = [
            texture("engine/transparent-accumulated", ACCUMULATED_FORMAT),
            texture("engine/transparent-revealed", REVEALED_FORMAT),
        ];
        let [accumulated, revealed] = textures
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
        let group = crate::pipelines::group(
            device,
            "engine/transparent-composite-group",
            &TRANSPARENT_COMPOSITE,
            &self.layout,
            &[
                ("accumulated", crate::pipelines::view(&accumulated)),
                ("revealed", crate::pipelines::view(&revealed)),
            ],
        );
        self.targets.set(
            size,
            Targets {
                textures,
                accumulated,
                revealed,
                group,
            },
        );
    }

    /// Releases the targets while frames draw nothing transparent.
    pub(crate) fn release_targets(&mut self) {
        self.targets.release();
    }

    /// Target bytes: 8 accumulated and 1 revealed per pixel.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.targets
            .size()
            .map_or(0, |size| u64::from(size.width) * u64::from(size.height) * 9)
    }

    /// The accumulation pass's targets, accumulated color and revealage,
    /// which it clears to nothing added and everything revealed.
    pub(crate) fn views(&self) -> [&wgpu::TextureView; 2] {
        let targets = self.targets.get().expect("sized in begin");
        [&targets.accumulated, &targets.revealed]
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
        let constants = [("TARGET_IS_SRGB", if format.is_srgb() { 1.0 } else { 0.0 })];
        self.composite[index] = Some(crate::pipelines::fullscreen(
            device,
            &crate::pipelines::Fullscreen {
                label: "engine/transparent-composite",
                layout: &self.pipeline_layout,
                module: &self.module,
                vertex: transparent::vs_fullscreen,
                fragment: transparent::fs_composite,
                constants: &constants,
                targets: &targets,
                depth: Some(wgpu::DepthStencilState {
                    format: self.depth_format,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
            },
        ));
    }

    /// Draws the composite; call [`Self::ensure_composite`] first.
    pub(crate) fn composite(&self, pass: &mut wgpu::RenderPass<'_>, post: bool) {
        let targets = self.targets.get().expect("sized in begin");
        pass.set_pipeline(self.composite[usize::from(post)].as_ref().expect("ensured"));
        pass.set_bind_group(0, &targets.group, &[]);
        pass.draw(0..3, 0..1);
    }
}
