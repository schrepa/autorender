#![expect(
    clippy::expect_used,
    reason = "the resolve reads the targets and groups its frame prepared"
)]

use crate::shaders::entry::temporal;
use engine_core::{Applied, RenderFrame, RenderSnapshot, glam::Mat4};

use crate::bindings::{TAA_RESOLVE, TEMPORAL_POSES};
use crate::cached::{Bound, GroupCache, Versioned};
use crate::gpu_data::{
    CameraUniform, ResolveUniform, TEMPORAL_SHADER, jitter_clip, jitter_offset, jittered,
};
use crate::post::PostPass;
use crate::profiling::Span;
use crate::render_features::{Encode, FeaturePass, RenderFeature};
use crate::render_graph::{Access, BLACK, Clear, FrameShape, Node, Pass, RenderGraph, Resource};
use crate::targets::{FrameRegion, FrameSize, ScreenTarget};

pub(crate) const MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const CURRENT_WEIGHT: f32 = 0.1;

struct Targets {
    textures: [wgpu::Texture; 3],
    motion: wgpu::TextureView,
    history: [wgpu::TextureView; 2],
}

impl Drop for Targets {
    fn drop(&mut self) {
        for texture in &self.textures {
            texture.destroy();
        }
    }
}

#[derive(Clone, Copy)]
struct History {
    view_projection: Mat4,
    size: FrameSize,
    identity: u64,
    structure: u64,
    pose_generation: u64,
}

pub(crate) struct TemporalFrame {
    pub parity: usize,
    pub camera: CameraUniform,
    pub rasterized: Mat4,
    pub history_valid: bool,
    pub motion_valid: bool,
    history: History,
}

pub(crate) struct TemporalPass {
    mesh_layout: wgpu::PipelineLayout,
    lit_layout: Option<wgpu::PipelineLayout>,
    pose_layout: crate::pipelines::GroupLayout,
    resolve_pipeline: wgpu::RenderPipeline,
    resolve_layout: crate::pipelines::GroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    written: Option<ResolveUniform>,
    targets: ScreenTarget<Targets>,
    no_particles: wgpu::TextureView,
    resolve_groups: [GroupCache<12>; 2],
    pose_groups: [GroupCache<6>; 4],
    history: Option<History>,
    frame: u32,
}

impl TemporalPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_layout: &crate::pipelines::GroupLayout,
        texture_layout: &crate::pipelines::GroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let pose_layout = crate::pipelines::layout(device, &TEMPORAL_POSES);
        let mesh_layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::TEMPORAL_MESH_PIPELINES,
            &[camera_layout, texture_layout, &pose_layout],
        );
        let resolve_layout = crate::pipelines::layout(device, &TAA_RESOLVE);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/temporal.wgsl"),
            source: wgpu::ShaderSource::Wgsl(TEMPORAL_SHADER.into()),
        });
        let resolve_pipeline_layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::TAA_RESOLVE_PIPELINES,
            &[&resolve_layout],
        );
        let resolve_pipeline = crate::pipelines::fullscreen(
            device,
            &crate::pipelines::Fullscreen {
                label: "engine/taa-resolve",
                layout: &resolve_pipeline_layout,
                module: &module,
                vertex: temporal::vs_fullscreen,
                fragment: temporal::fs_resolve,
                constants: &[],
                targets: &[Some(surface_format.into()), Some(HISTORY_FORMAT.into())],
                depth: None,
            },
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/taa-history-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = crate::buffers::uniform(
            device,
            "engine/taa-resolve-uniform",
            size_of::<ResolveUniform>() as u64,
        );
        let no_particles = crate::targets::texture_2d(
            device,
            "engine/taa-no-particles",
            (1, 1),
            1,
            1,
            crate::transparent::REVEALED_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        queue.write_texture(
            no_particles.as_image_copy(),
            &[255; 2],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(2),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        Self {
            mesh_layout,
            lit_layout: None,
            pose_layout,
            resolve_pipeline,
            resolve_layout,
            sampler,
            uniform,
            written: None,
            targets: ScreenTarget::default(),
            no_particles: no_particles.create_view(&Default::default()),
            resolve_groups: Default::default(),
            pose_groups: Default::default(),
            history: None,
            frame: 0,
        }
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.targets.size().map_or(0, |size| {
            u64::from(size.width) * u64::from(size.height) * 12
        })
    }

    pub(crate) fn begin(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &RenderFrame<'_>,
        region: FrameRegion,
        pose_generation: u64,
    ) -> TemporalFrame {
        let (size, render) = (region.display, region.render);
        if self.targets.resizes(size) {
            let targets = self.create_targets(device, size);
            self.targets.set(size, targets);
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
        let offset = jitter_offset(self.frame);
        let jitter = jitter_clip(offset, render.width, render.height);
        let previous = last.map_or(view_projection, |last| last.view_projection);
        let camera = CameraUniform::temporal(
            view_projection,
            frame.snapshot().camera_position,
            frame.snapshot().elapsed_seconds,
            previous,
            jitter,
            render.pixels(),
            u64::from(self.frame),
        );
        let environment = &frame.snapshot().environment;
        let settings = environment.temporal_antialiasing.sanitized();
        let uniform = ResolveUniform {
            texel: [1.0 / size.width as f32, 1.0 / size.height as f32],
            current_weight: CURRENT_WEIGHT,
            history_valid: u32::from(last.is_some()),
            options: [
                if environment.dither.enabled {
                    camera.options[2]
                } else {
                    0.0
                },
                f32::from(u8::from(settings.nearest_motion)),
                f32::from(u8::from(settings.rounded_clamp)),
                settings.particle_response,
            ],
            render: [
                render.width as f32,
                render.height as f32,
                offset[0],
                offset[1],
            ],
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
            camera,
            rasterized: jittered(view_projection, jitter),
            history_valid: last.is_some(),
            motion_valid: last.is_some_and(|last| {
                last.structure == history.structure && last.pose_generation == pose_generation
            }),
            history,
        }
    }

    pub(crate) fn prepare_pose_groups(
        &mut self,
        device: &wgpu::Device,
        poses: [Bound<'_>; 2],
        colors: &Bound<'_>,
        palettes: [Bound<'_>; 2],
        grounds: &Bound<'_>,
    ) {
        for (index, group) in self.pose_groups.iter_mut().enumerate() {
            let (current, previous) = (index / 2, index % 2);
            group.bind(
                device,
                "engine/temporal-pose-group",
                (&TEMPORAL_POSES, &self.pose_layout),
                [
                    ("poses", poses[current].clone()),
                    ("colors", colors.clone()),
                    ("previous_poses", poses[previous].clone()),
                    ("palette", palettes[current].clone()),
                    ("previous_palette", palettes[previous].clone()),
                    ("grounds", grounds.clone()),
                ],
            );
        }
    }

    pub(crate) fn pose_group(&self, frame: &TemporalFrame) -> &wgpu::BindGroup {
        let previous = if frame.motion_valid {
            1 - frame.parity
        } else {
            frame.parity
        };
        self.pose_groups[2 * frame.parity + previous]
            .get()
            .expect("prepared this frame")
    }

    pub(crate) fn ensure_lit_layout(
        &mut self,
        device: &wgpu::Device,
        layouts: [&crate::pipelines::GroupLayout; 3],
    ) {
        if self.lit_layout.is_some() {
            return;
        }
        let [camera, texture, lights] = layouts;
        let layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::LIT_TEMPORAL_MESH_PIPELINES,
            &[camera, texture, &self.pose_layout, lights],
        );
        self.lit_layout = Some(layout);
    }

    pub(crate) fn mesh_layout(&self, lit: bool) -> &wgpu::PipelineLayout {
        if lit {
            self.lit_layout
                .as_ref()
                .expect("created with the first lit frame")
        } else {
            &self.mesh_layout
        }
    }

    pub(crate) fn pose_layout(&self) -> &crate::pipelines::GroupLayout {
        &self.pose_layout
    }

    pub(crate) fn history_target(&self, frame: &ResolveFrame) -> &wgpu::TextureView {
        &self.targets.get().expect("sized in begin").history[frame.parity]
    }

    pub(crate) fn motion(&self) -> &wgpu::TextureView {
        &self.targets.get().expect("sized in begin").motion
    }

    pub(crate) fn prepare_resolve(
        &mut self,
        device: &wgpu::Device,
        post: &PostPass,
        frame: &TemporalFrame,
        depth: &Versioned<wgpu::TextureView>,
        revealed: Option<Bound<'_>>,
    ) -> ResolveFrame {
        let resolve = ResolveFrame {
            parity: frame.parity,
            history: frame.history,
            revealed: revealed.is_some(),
        };
        let Some(targets) = self.targets.get() else {
            return resolve;
        };
        let ([scene, occlusion, bloom, reflections, haze], settings) = post.resolve_inputs();
        let target = |view| Bound::of(targets, crate::pipelines::view(view));
        let revealed =
            revealed.unwrap_or_else(|| Bound::fixed(crate::pipelines::view(&self.no_particles)));
        for (parity, group) in self.resolve_groups.iter_mut().enumerate() {
            group.bind(
                device,
                "engine/taa-resolve-group",
                (&TAA_RESOLVE, &self.resolve_layout),
                [
                    ("resolve", Bound::fixed(self.uniform.as_entire_binding())),
                    ("scene", scene.clone()),
                    ("motion", target(&targets.motion)),
                    ("history", target(&targets.history[1 - parity])),
                    (
                        "history_sampler",
                        Bound::fixed(crate::pipelines::sampler(&self.sampler)),
                    ),
                    ("occlusion", occlusion.clone()),
                    ("bloom", bloom.clone()),
                    ("reflections", reflections.clone()),
                    ("haze", haze.clone()),
                    ("post", settings.clone()),
                    ("depth_texture", depth.bound()),
                    ("revealed", revealed.clone()),
                ],
            );
        }
        resolve
    }

    fn draw_resolve(&self, mut pass: wgpu::RenderPass<'_>, frame: &ResolveFrame) {
        let group = self.resolve_groups[frame.parity]
            .get()
            .expect("bound in prepare_resolve");
        pass.set_pipeline(&self.resolve_pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..3, 0..1);
    }

    pub(crate) fn reset(&mut self) {
        self.history = None;
    }

    fn create_targets(&self, device: &wgpu::Device, size: FrameSize) -> Targets {
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
            texture("engine/taa-motion", MOTION_FORMAT),
            texture("engine/taa-history", HISTORY_FORMAT),
            texture("engine/taa-history", HISTORY_FORMAT),
        ];
        let [motion, first, second] = textures
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
        Targets {
            textures,
            motion,
            history: [first, second],
        }
    }
}

pub(crate) struct ResolveFrame {
    parity: usize,
    history: History,
    revealed: bool,
}

impl RenderFeature for TemporalPass {
    type Frame = ResolveFrame;
    type Stats = Applied;

    const LABELS: &'static [&'static str] = &["engine/taa-", "engine/temporal"];

    fn declare(frame: &ResolveFrame, shape: &FrameShape, graph: &mut RenderGraph) {
        if shape.targets != crate::render_graph::Targets::Temporal {
            return;
        }
        graph
            .add(Pass::Feature(FeaturePass::TemporalResolve))
            .read(Resource::Scene)
            .read(Resource::Motion)
            .read(Resource::Depth)
            .read(Resource::History)
            .read_if(frame.revealed, Resource::Revealed)
            .read_if(shape.effects.any(), Resource::Effects)
            .color(Resource::Canvas, Access::Clear(Clear::Color(BLACK)))
            .color(Resource::History, Access::Clear(Clear::Color(BLACK)));
    }

    fn encode(
        &self,
        frame: &ResolveFrame,
        node: &Node,
        encode: &Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        let pass = encode.render_pass(
            encoder,
            node,
            "engine/taa-resolve",
            encode.profiler.span("taa_resolve").map(Span::render),
        );
        self.draw_resolve(pass, frame);
        0
    }

    fn after_submit(&mut self, frame: Option<&ResolveFrame>) {
        if let Some(frame) = frame {
            self.history = Some(frame.history);
            self.frame = self.frame.wrapping_add(1);
        }
    }

    fn capacity_bytes(&self) -> u64 {
        TemporalPass::capacity_bytes(self)
    }

    fn stats(
        _this: Option<&Self>,
        frame: Option<&ResolveFrame>,
        _snapshot: &RenderSnapshot,
    ) -> Applied {
        Applied {
            active: frame.is_some(),
        }
    }
}
