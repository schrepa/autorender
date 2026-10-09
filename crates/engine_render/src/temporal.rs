//! Temporal antialiasing of GPU-presented display frames.
//!
//! Each frame rasterizes with a sub-pixel jitter into the screen effects'
//! scene and glow targets and a motion target, then one
//! full-screen pass blends the scene with the reprojected previous output,
//! applies ambient occlusion and bloom, and writes both the canvas and the
//! next history. Culling alternates between two pose buffers, so the previous
//! frame's displayed poses stay readable for motion vectors without copies.

use engine_core::{RenderFrame, glam::Mat4};

use crate::bindings::{TAA_RESOLVE, TEMPORAL_POSES};
use crate::cached::Cached;
use crate::gpu_data::{
    CameraUniform, ResolveUniform, TEMPORAL_SHADER, jitter_clip, jitter_offset, jittered,
};
use crate::post::PostPass;

pub(crate) const MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Weight of each frame's color against the history: an exponential average
/// over about ten frames, beyond the eight-sample jitter cycle.
const CURRENT_WEIGHT: f32 = 0.1;

struct Targets {
    size: (u32, u32),
    // Motion and both histories, destroyed with the targets.
    textures: [wgpu::Texture; 3],
    motion: wgpu::TextureView,
    history: [wgpu::TextureView; 2],
    // Frames writing history `p` read history `1 - p` through group `p`,
    // over the screen-effect targets of that generation.
    groups: Cached<u64, [wgpu::BindGroup; 2]>,
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
    // Mesh pipelines live with the renderer's other variants;
    // the pass owns their layouts, the lit one created with the first lit
    // temporal frame.
    mesh_layout: wgpu::PipelineLayout,
    lit_layout: Option<wgpu::PipelineLayout>,
    pose_layout: wgpu::BindGroupLayout,
    resolve_pipeline: wgpu::RenderPipeline,
    resolve_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    written: Option<ResolveUniform>,
    targets: Option<Targets>,
    // Pose groups for (current, previous) buffer pairs, index 2 * current +
    // previous, of the pose storage generation they were built for.
    pose_groups: Cached<u64, [wgpu::BindGroup; 4]>,
    history: Option<History>,
    frame: u32,
}

impl TemporalPass {
    /// Creates the resolve pipeline, the mesh pipeline layout and fixed
    /// resources; targets follow the surface. Mesh and sky pipelines are the
    /// renderer's.
    pub(crate) fn new(
        device: &wgpu::Device,
        camera_layout: &wgpu::BindGroupLayout,
        texture_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let pose_layout = crate::pipelines::layout(device, &TEMPORAL_POSES);
        let mesh_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/temporal-mesh-pipeline-layout"),
            bind_group_layouts: &[
                Some(camera_layout),
                Some(texture_layout),
                Some(&pose_layout),
            ],
            immediate_size: 0,
        });
        let resolve_layout = crate::pipelines::layout(device, &TAA_RESOLVE);
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
        let resolve_pipeline = crate::pipelines::fullscreen(
            device,
            &crate::pipelines::Fullscreen {
                label: "engine/taa-resolve",
                layout: &resolve_pipeline_layout,
                module: &module,
                vertex: "vs_fullscreen",
                fragment: "fs_resolve",
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
            targets: None,
            pose_groups: Cached::default(),
            history: None,
            frame: 0,
        }
    }

    /// Target bytes: motion and two history textures of 4 bytes per pixel
    /// each (the scene target belongs to the screen effects).
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.targets.as_ref().map_or(0, |targets| {
            u64::from(targets.size.0) * u64::from(targets.size.1) * 12
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
                frame.snapshot().elapsed_seconds,
                previous,
                jitter,
                size,
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

    /// Builds the pose bindings over `poses`, the pair of pose storage
    /// `generation`, `colors` and the joint palettes alternating with the
    /// poses, unless built for that generation.
    pub(crate) fn prepare_pose_groups(
        &mut self,
        device: &wgpu::Device,
        generation: u64,
        poses: [&wgpu::Buffer; 2],
        colors: &wgpu::Buffer,
        palettes: [&wgpu::Buffer; 2],
        grounds: &wgpu::Buffer,
    ) {
        if self.pose_groups.stale(&generation) {
            let group = |current: usize, previous: usize| {
                crate::pipelines::group(
                    device,
                    "engine/temporal-pose-group",
                    &TEMPORAL_POSES,
                    &self.pose_layout,
                    &[
                        ("poses", poses[current].as_entire_binding()),
                        ("colors", colors.as_entire_binding()),
                        ("previous_poses", poses[previous].as_entire_binding()),
                        ("palette", palettes[current].as_entire_binding()),
                        ("previous_palette", palettes[previous].as_entire_binding()),
                        ("grounds", grounds.as_entire_binding()),
                    ],
                )
            };
            self.pose_groups.set(
                generation,
                [group(0, 0), group(0, 1), group(1, 0), group(1, 1)],
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
        let groups = self.pose_groups.get().expect("prepared this frame");
        &groups[2 * frame.parity + previous]
    }

    /// Creates the lit mesh pipeline layout unless it exists; `layouts` are
    /// the camera, texture and lights layouts.
    pub(crate) fn ensure_lit_layout(
        &mut self,
        device: &wgpu::Device,
        layouts: [&wgpu::BindGroupLayout; 3],
    ) {
        if self.lit_layout.is_some() {
            return;
        }
        let [camera, texture, lights] = layouts;
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/temporal-lit-mesh-pipeline-layout"),
            bind_group_layouts: &[
                Some(camera),
                Some(texture),
                Some(&self.pose_layout),
                Some(lights),
            ],
            immediate_size: 0,
        });
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
    pub(crate) fn pose_layout(&self) -> &wgpu::BindGroupLayout {
        &self.pose_layout
    }

    /// The motion target of the opaque pass.
    pub(crate) fn motion(&self) -> &wgpu::TextureView {
        &self.targets.as_ref().expect("sized in begin").motion
    }

    /// Binds the resolve to the screen effects' current targets.
    pub(crate) fn prepare_resolve(&mut self, device: &wgpu::Device, post: &PostPass) {
        let generation = post.generation();
        let Some(targets) = self.targets.as_mut() else {
            return;
        };
        if !targets.groups.stale(&generation) {
            return;
        }
        let ([scene, occlusion, bloom, reflections], settings) = post.resolve_inputs();
        let group = |previous: &wgpu::TextureView| {
            crate::pipelines::group(
                device,
                "engine/taa-resolve-group",
                &TAA_RESOLVE,
                &self.resolve_layout,
                &[
                    ("resolve", self.uniform.as_entire_binding()),
                    ("scene", crate::pipelines::view(scene)),
                    ("motion", crate::pipelines::view(&targets.motion)),
                    ("history", crate::pipelines::view(previous)),
                    ("history_sampler", crate::pipelines::sampler(&self.sampler)),
                    ("occlusion", crate::pipelines::view(occlusion)),
                    ("bloom", crate::pipelines::view(bloom)),
                    ("reflections", crate::pipelines::view(reflections)),
                    ("post", settings.as_entire_binding()),
                ],
            )
        };
        let groups = [group(&targets.history[1]), group(&targets.history[0])];
        targets.groups.set(generation, groups);
    }

    /// Records the resolve of `frame` into `canvas` and its history texture,
    /// with operations `ops` for each.
    pub(crate) fn encode_resolve(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        canvas: &wgpu::TextureView,
        frame: &TemporalFrame,
        [canvas_ops, history_ops]: [wgpu::Operations<wgpu::Color>; 2],
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let targets = self.targets.as_ref().expect("sized in begin");
        let groups = targets.groups.get().expect("bound in prepare_resolve");
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
        pass.set_bind_group(0, &groups[frame.parity], &[]);
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
            crate::targets::texture_2d(
                device,
                label,
                (width, height),
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
            size: (width, height),
            textures,
            motion,
            history: [first, second],
            groups: Cached::default(),
        }
    }
}
