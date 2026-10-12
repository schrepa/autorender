//! Weighted blended order-independent transparency.
//!
//! Transparent surfaces and alpha particles add their weighted color and
//! coverage into an accumulation target and multiply what they let through
//! into a revealage target, in any order; a composite then blends their
//! weighted average over the frame's color. Targets exist only while frames
//! draw something transparent. With the particles, transparency is a
//! feature of the renderer (`crate::render_features`).

use engine_core::{Drawn, RenderSnapshot};

use crate::bindings::TRANSPARENT_COMPOSITE;
use crate::cached::Versioned;
use crate::gpu_data::TRANSPARENT_SHADER;
use crate::particles::{ParticleFrame, ParticlePass};
use crate::profiling::Span;
use crate::render_features::{Encode, FeaturePass, RenderFeature};
use crate::render_graph::{
    Access, Clear, FrameShape, Node, Pass, RenderGraph, Resource, TRANSPARENT, WHITE,
};
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

struct TransparentPass {
    layout: crate::pipelines::GroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    module: wgpu::ShaderModule,
    // Composites into the canvas and into the scene target (leaving glow).
    composite: [Option<wgpu::RenderPipeline>; 2],
    surface_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    targets: ScreenTarget<Targets>,
}

impl TransparentPass {
    fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let layout = crate::pipelines::layout(device, &TRANSPARENT_COMPOSITE);
        let pipeline_layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::TRANSPARENT_COMPOSITE_PIPELINES,
            &[&layout],
        );
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
    fn begin(&mut self, device: &wgpu::Device, size: FrameSize) {
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
    fn release_targets(&mut self) {
        self.targets.release();
    }

    /// Target bytes: 8 accumulated and 1 revealed per pixel.
    fn capacity_bytes(&self) -> u64 {
        self.targets
            .size()
            .map_or(0, |size| u64::from(size.width) * u64::from(size.height) * 9)
    }

    /// The accumulation pass's targets, accumulated color and revealage,
    /// which it clears to nothing added and everything revealed.
    fn views(&self) -> [&wgpu::TextureView; 2] {
        let targets = self.targets.get().expect("sized in begin");
        [&targets.accumulated, &targets.revealed]
    }

    /// Creates the composite into the canvas or, with `post`, the scene
    /// target beside the glow target.
    fn ensure_composite(&mut self, device: &wgpu::Device, post: bool) {
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
    fn composite(&self, pass: &mut wgpu::RenderPass<'_>, post: bool) {
        let targets = self.targets.get().expect("sized in begin");
        pass.set_pipeline(self.composite[usize::from(post)].as_ref().expect("ensured"));
        pass.set_bind_group(0, &targets.group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Transparent surfaces and particles: the accumulation targets and their
/// composite, created with the first frame drawing something transparent,
/// and the particles' pipelines, with the first frame that has emitters.
#[derive(Default)]
pub(crate) struct Transparency {
    surfaces: Option<TransparentPass>,
    particles: Option<ParticlePass>,
}

/// What a frame draws transparent: whether the renderer's transparent mesh
/// batches draw, and its particles.
pub(crate) struct TransparencyFrame {
    meshes: bool,
    particles: Option<ParticleFrame>,
}

impl Transparency {
    /// Prepares a frame's transparency: its transparent mesh batches when
    /// `meshes`, and `snapshot`'s particles when they change (`emitters`:
    /// the render stream identity and its emitters revision), soft against
    /// the depth buffer `depth`; at the frame's `size`, composited into the
    /// canvas or, with `post`, the effects' targets. `camera` is the main
    /// passes' first group layout, `formats` the canvas's and the depth
    /// buffer's. `None` when nothing transparent draws, whose frames release
    /// the targets.
    #[expect(clippy::too_many_arguments, reason = "the frame's inputs, typed")]
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        snapshot: &RenderSnapshot,
        emitters: (u64, u64),
        (size, depth): (FrameSize, &Versioned<wgpu::TextureView>),
        (camera, (surface_format, depth_format)): (
            &crate::pipelines::GroupLayout,
            (wgpu::TextureFormat, wgpu::TextureFormat),
        ),
        (meshes, post): (bool, bool),
    ) -> Option<TransparencyFrame> {
        // Particles sample the depth buffer for their soft edges.
        let particles =
            if snapshot.environment.particles.enabled && !snapshot.particles.list.is_empty() {
                let particles = self.particles.get_or_insert_with(|| {
                    ParticlePass::new(
                        device,
                        camera,
                        surface_format,
                        depth_format,
                        &accumulation_targets(),
                    )
                });
                let prepared = particles.prepare(device, queue, snapshot, emitters, size, depth);
                if prepared.is_some() {
                    particles.ensure_additive(device, post);
                }
                prepared
            } else {
                None
            };
        if !meshes && particles.is_none() {
            if let Some(surfaces) = self.surfaces.as_mut() {
                surfaces.release_targets();
            }
            return None;
        }
        let surfaces = self
            .surfaces
            .get_or_insert_with(|| TransparentPass::new(device, surface_format, depth_format));
        surfaces.begin(device, size);
        surfaces.ensure_composite(device, post);
        Some(TransparencyFrame { meshes, particles })
    }

    /// The surfaces, which every frame drawing something transparent has.
    fn surfaces(&self) -> &TransparentPass {
        self.surfaces
            .as_ref()
            .expect("frames drawing something transparent create the surfaces")
    }

    /// The accumulation pass's targets, accumulated color and revealage.
    pub(crate) fn views(&self) -> [&wgpu::TextureView; 2] {
        self.surfaces().views()
    }
}

impl RenderFeature for Transparency {
    type Frame = TransparencyFrame;
    type Stats = Drawn;

    /// Transparent surfaces and alpha particles accumulate over read-only
    /// depth, with the frame's shadows, lights and probes, and glass the
    /// opaque color it refracts; their composite, then additive particles,
    /// draw over the main color (and the glow target, which they leave).
    fn declare(_frame: &TransparencyFrame, shape: &FrameShape, graph: &mut RenderGraph) {
        graph
            .add(Pass::Feature(FeaturePass::Transparent))
            .read(Resource::Poses)
            .read_if(shape.gpu, Resource::Visible)
            .read_if(shape.lights, Resource::Lights)
            .read_if(shape.shadows, Resource::ShadowMaps)
            .read_if(shape.refraction, Resource::Scene)
            .read(Resource::Probes)
            .color(
                Resource::Accumulated,
                Access::Clear(Clear::Color(TRANSPARENT)),
            )
            .color(Resource::Revealed, Access::Clear(Clear::Color(WHITE)))
            .depth(Resource::Depth, Access::ReadOnly);
        let scene = shape.scene();
        let composite = graph
            .add(Pass::Feature(FeaturePass::TransparentComposite))
            .read(Resource::Accumulated)
            .read(Resource::Revealed)
            .color(scene, Access::Load);
        let composite = if scene == Resource::Scene {
            composite.color(Resource::Glow, Access::Load)
        } else {
            composite
        };
        composite.depth(Resource::Depth, Access::ReadOnly);
    }

    fn encode(
        &self,
        frame: &TransparencyFrame,
        node: &Node,
        encode: &Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        if node.pass() == Pass::Feature(FeaturePass::TransparentComposite) {
            // The accumulated surfaces over the frame's color, then
            // additive particles.
            let post = encode.targets != crate::render_graph::Targets::Canvas;
            let mut pass = encode.render_pass(
                encoder,
                node,
                "engine/transparent-composite",
                encode
                    .profiler
                    .span("transparent_composite")
                    .map(Span::render),
            );
            self.surfaces().composite(&mut pass, post);
            if let (Some(particles_frame), Some(particles)) = (&frame.particles, &self.particles) {
                particles.draw_additive(&mut pass, encode.camera, particles_frame, post);
            }
            return 0;
        }
        // Read-only depth: surfaces behind the opaque ones stay hidden, and
        // particles sample it.
        let mut pass = encode.render_pass(
            encoder,
            node,
            "engine/transparent-accumulation",
            encode.profiler.span("transparent").map(Span::render),
        );
        let draws = if frame.meshes {
            encode.meshes.transparent(&mut pass)
        } else {
            0
        };
        if let (Some(particles_frame), Some(particles)) = (&frame.particles, &self.particles) {
            particles.draw_alpha(&mut pass, encode.camera, particles_frame);
        }
        draws
    }

    fn capacity_bytes(&self) -> u64 {
        self.surfaces
            .as_ref()
            .map_or(0, TransparentPass::capacity_bytes)
            + self
                .particles
                .as_ref()
                .map_or(0, ParticlePass::capacity_bytes)
    }

    /// Particles drawn, invalid ones, and those left out: the snapshot's and
    /// those beyond the GPU's bindings.
    fn stats(
        _this: Option<&Self>,
        frame: Option<&TransparencyFrame>,
        snapshot: &RenderSnapshot,
    ) -> Drawn {
        let particles = frame.and_then(|frame| frame.particles.as_ref());
        Drawn {
            drawn: particles.map_or(0, |frame| frame.total),
            invalid: snapshot.particles.invalid.min(u32::MAX as usize) as u32,
            dropped: (snapshot.particles.dropped as u64
                + particles.map_or(0, |frame| frame.dropped))
            .min(u64::from(u32::MAX)) as u32,
        }
    }
}
