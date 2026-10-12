//! Temporal antialiasing of GPU-presented display frames.
//!
//! Each frame rasterizes with a sub-pixel jitter into the screen effects'
//! scene and glow targets and a motion target, then one
//! full-screen pass blends the scene with the reprojected previous output,
//! applies ambient occlusion and bloom, and writes both the canvas and the
//! next history. Culling alternates between two pose buffers, so the previous
//! frame's displayed poses stay readable for motion vectors without copies.

use crate::shaders::entry::temporal;
use engine_core::{Applied, RenderFrame, RenderSnapshot, glam::Mat4};

use crate::bindings::{TAA_RESOLVE, TEMPORAL_POSES};
use crate::cached::{Bound, GroupCache};
use crate::gpu_data::{
    CameraUniform, ResolveUniform, TEMPORAL_SHADER, jitter_clip, jitter_offset, jittered,
};
use crate::post::PostPass;
use crate::profiling::Span;
use crate::render_features::{Encode, FeaturePass, RenderFeature, color_operations};
use crate::render_graph::{Access, BLACK, Clear, FrameShape, Node, Pass, RenderGraph, Resource};
use crate::targets::{FrameSize, ScreenTarget};

pub(crate) const MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Weight of each frame's color against the history: an exponential average
/// over about ten frames, beyond the eight-sample jitter cycle.
const CURRENT_WEIGHT: f32 = 0.1;

struct Targets {
    // Motion and both histories, destroyed with the targets.
    textures: [wgpu::Texture; 3],
    motion: wgpu::TextureView,
    history: [wgpu::TextureView; 2],
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
    size: FrameSize,
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
    // Mesh pipelines live with the renderer's other variants;
    // the pass owns their layouts, the lit one created with the first lit
    // temporal frame.
    mesh_layout: wgpu::PipelineLayout,
    lit_layout: Option<wgpu::PipelineLayout>,
    pose_layout: crate::pipelines::GroupLayout,
    resolve_pipeline: wgpu::RenderPipeline,
    resolve_layout: crate::pipelines::GroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    written: Option<ResolveUniform>,
    targets: ScreenTarget<Targets>,
    // Frames writing history `p` read history `1 - p` through group `p`,
    // over the screen effects' targets.
    resolve_groups: [GroupCache<9>; 2],
    // Pose groups for (current, previous) buffer pairs, index 2 * current +
    // previous.
    pose_groups: [GroupCache<6>; 4],
    history: Option<History>,
    frame: u32,
}

impl TemporalPass {
    /// Creates the resolve pipeline, the mesh pipeline layout and fixed
    /// resources; targets follow the surface. Mesh and sky pipelines are the
    /// renderer's.
    pub(crate) fn new(
        device: &wgpu::Device,
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
            resolve_groups: Default::default(),
            pose_groups: Default::default(),
            history: None,
            frame: 0,
        }
    }

    /// Target bytes: motion and two history textures of 4 bytes per pixel
    /// each (the scene target belongs to the screen effects).
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.targets.size().map_or(0, |size| {
            u64::from(size.width) * u64::from(size.height) * 12
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
        size: FrameSize,
        pose_generation: u64,
    ) -> TemporalFrame {
        if self.targets.resizes(size) {
            // Recreated history starts over.
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
        let jitter = jitter_clip(jitter_offset(self.frame), size.width, size.height);
        let previous = last.map_or(view_projection, |last| last.view_projection);
        let uniform = ResolveUniform {
            texel: [1.0 / size.width as f32, 1.0 / size.height as f32],
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
                frame.snapshot().elapsed_seconds,
                previous,
                jitter,
                size.pixels(),
                u64::from(self.frame),
            ),
            rasterized: jittered(view_projection, jitter),
            history_valid: last.is_some(),
            motion_valid: last.is_some_and(|last| {
                last.structure == history.structure && last.pose_generation == pose_generation
            }),
            history,
        }
    }

    /// Builds the pose bindings over the pair of pose buffers `poses`,
    /// `colors`, the joint palettes alternating with the poses and
    /// `grounds`, unless built over them.
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

    /// Pose bindings of `frame`: its pose buffer, the colors, and the
    /// previous frame's poses when valid (otherwise its own again).
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

    /// Creates the lit mesh pipeline layout unless it exists; `layouts` are
    /// the camera, texture and lights layouts.
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

    /// The layout of temporal frames' mesh pipelines, lit or not; lit ones
    /// after [`Self::ensure_lit_layout`].
    pub(crate) fn mesh_layout(&self, lit: bool) -> &wgpu::PipelineLayout {
        if lit {
            self.lit_layout
                .as_ref()
                .expect("created with the first lit frame")
        } else {
            &self.mesh_layout
        }
    }

    /// The pose group layout of temporal frames: poses, colors, previous
    /// poses and joint palettes.
    pub(crate) fn pose_layout(&self) -> &crate::pipelines::GroupLayout {
        &self.pose_layout
    }

    /// The motion target of the opaque pass.
    pub(crate) fn motion(&self) -> &wgpu::TextureView {
        &self.targets.get().expect("sized in begin").motion
    }

    /// Binds the resolve of `frame` to the screen effects' current targets:
    /// what the resolve and the next frame read of it.
    pub(crate) fn prepare_resolve(
        &mut self,
        device: &wgpu::Device,
        post: &PostPass,
        frame: &TemporalFrame,
    ) -> ResolveFrame {
        let resolve = ResolveFrame {
            parity: frame.parity,
            history: frame.history,
        };
        let Some(targets) = self.targets.get() else {
            return resolve;
        };
        let ([scene, occlusion, bloom, reflections], settings) = post.resolve_inputs();
        let target = |view| Bound::of(targets, crate::pipelines::view(view));
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
                    ("post", settings.clone()),
                ],
            );
        }
        resolve
    }

    /// Records the resolve of `frame` into `canvas` and its history texture,
    /// with operations `ops` for each.
    fn encode_resolve(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        canvas: &wgpu::TextureView,
        frame: &ResolveFrame,
        [canvas_ops, history_ops]: [wgpu::Operations<wgpu::Color>; 2],
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let targets = self.targets.get().expect("sized in begin");
        let group = self.resolve_groups[frame.parity]
            .get()
            .expect("bound in prepare_resolve");
        let attachment = |view, ops| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops,
            })
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/taa-resolve"),
            color_attachments: &[
                attachment(canvas, canvas_ops),
                attachment(&targets.history[frame.parity], history_ops),
            ],
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.resolve_pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// A frame drawn without temporal antialiasing ends the history.
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

/// What a frame's resolve writes and leaves: the history texture it writes
/// (its parity), and the history the next frame reads.
pub(crate) struct ResolveFrame {
    parity: usize,
    history: History,
}

impl RenderFeature for TemporalPass {
    type Frame = ResolveFrame;
    type Stats = Applied;

    /// The resolve reads the scene, the motion and the history (and the
    /// effects that run), and writes the canvas and the next history.
    fn declare(_frame: &ResolveFrame, shape: &FrameShape, graph: &mut RenderGraph) {
        if shape.targets != crate::render_graph::Targets::Temporal {
            return;
        }
        graph
            .add(Pass::Feature(FeaturePass::TemporalResolve))
            .read(Resource::Scene)
            .read(Resource::Motion)
            .read(Resource::History)
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
        let mut ops = node.colors().map(|(_, ops)| color_operations(ops));
        let ops = [
            ops.next().expect("the canvas"),
            ops.next().expect("the history"),
        ];
        self.encode_resolve(
            encoder,
            encode.view(Resource::Canvas),
            frame,
            ops,
            encode.profiler.span("taa_resolve").map(Span::render),
        );
        0
    }

    /// A submitted frame's history is the next frame's.
    fn after_submit(&mut self, frame: Option<&ResolveFrame>) {
        if let Some(frame) = frame {
            self.history = Some(frame.history);
            self.frame = self.frame.wrapping_add(1);
        }
    }

    fn capacity_bytes(&self) -> u64 {
        TemporalPass::capacity_bytes(self)
    }

    fn stats(frame: Option<&ResolveFrame>, _snapshot: &RenderSnapshot) -> Applied {
        Applied {
            active: frame.is_some(),
        }
    }
}
