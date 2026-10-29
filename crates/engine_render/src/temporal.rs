#![expect(
    clippy::expect_used,
    reason = "the resolve reads the targets and groups its frame prepared"
)]

use crate::shaders::entry::temporal;
use engine_core::{Applied, RenderFrame, RenderStats, glam::Mat4};

use crate::bindings::{TAA_RESOLVE, TEMPORAL_POSES};
use crate::cached::{Bound, GroupCache, Versioned};
use crate::composite::{TAA_RESOLVE as TAA_MODULE, composite_entry};
use crate::gpu_data::{CameraUniform, ResolveUniform, jitter_clip, jitter_offset, jittered};
use crate::graph::{
    Access, BLACK, Clear, FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl,
    ResourceId, Stage,
};
use crate::lazy::Lazy;
use crate::parts::{
    Begin, BeginOut, Create, Encode, GpuPart, PartDecl, PartEntry, Prelude, Step, View,
};
use crate::pipeline_layouts::TAA_RESOLVE_PIPELINES;
use crate::post::PostPass;
use crate::profiling::Span;
use crate::targets::{FrameRegion, FrameSize, ScreenTarget};
use crate::transparent::Transparency;

pub(crate) const MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const CURRENT_WEIGHT: f32 = 0.1;

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "temporal",
        brick: "post_effects",
        passes: &[PassDecl {
            name: "temporal_resolve",
            stage: Stage::Composite,
        }],
        resources: &[
            ResourceDecl {
                name: "motion",
                lifetime: Lifetime::Frame,
            },
            ResourceDecl {
                name: "history",
                lifetime: Lifetime::Kept,
            },
        ],
        hooks: &[],
        contributes: &[],
        effects: &[],
        labels: &["engine/taa-", "engine/temporal"],
        lazy: true,
    },
    create,
};

fn create(_: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(TemporalPass::default())
}

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

#[derive(Clone, Copy)]
pub(crate) struct TemporalFrame {
    pub parity: usize,
    pub camera: CameraUniform,
    pub rasterized: Mat4,
    pub history_valid: bool,
    pub motion_valid: bool,
    history: History,
}

#[derive(Clone, Copy, Default)]
struct Ids {
    pass: Option<PassId>,
    canvas: Option<ResourceId>,
    scene: Option<ResourceId>,
    motion: Option<ResourceId>,
    depth: Option<ResourceId>,
    history: Option<ResourceId>,
    revealed: Option<ResourceId>,
    effects: Option<ResourceId>,
}

#[derive(Default)]
pub(crate) struct TemporalPass {
    taa: Lazy<Taa>,
    resolve: Option<ResolveFrame>,
    ids: Ids,
}

impl TemporalPass {
    pub(crate) fn active(&self) -> bool {
        self.taa.get().is_some()
    }

    pub(crate) fn reset(&mut self) {
        if let Some(taa) = self.taa.get_mut() {
            taa.reset();
        }
    }

    pub(crate) fn pose_group(&self, frame: &TemporalFrame) -> Option<&wgpu::BindGroup> {
        self.taa.get()?.pose_group(frame)
    }
}

struct Taa {
    pose_layout: crate::pipelines::GroupLayout,
    resolve_pipeline: wgpu::RenderPipeline,
    resolve_layout: crate::pipelines::GroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    written: Option<ResolveUniform>,
    targets: ScreenTarget<Targets>,
    no_particles: wgpu::TextureView,
    resolve_groups: [GroupCache<13>; 2],
    pose_groups: [GroupCache<4>; 4],
    history: Option<History>,
    frame: u32,
}

impl Taa {
    fn new(
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let pose_layout = crate::pipelines::layout(device, &TEMPORAL_POSES);
        let resolve_layout = crate::pipelines::layout(device, &TAA_RESOLVE);
        let resolve_pipeline_layout = crate::pipelines::layout_of(device, &TAA_RESOLVE_PIPELINES);
        let resolve_pipeline = crate::pipelines::fullscreen(
            device,
            &crate::pipelines::Fullscreen {
                label: "engine/taa-resolve",
                layout: &resolve_pipeline_layout,
                vertex: temporal::vs_fullscreen,
                fragment: composite_entry(TAA_MODULE),
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

    fn capacity_bytes(&self) -> u64 {
        self.targets.size().map_or(0, |size| {
            u64::from(size.width) * u64::from(size.height) * 12
        })
    }

    fn begin(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
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

    fn prepare_pose_groups(
        &mut self,
        device: &wgpu::Device,
        poses: [Bound<'_>; 2],
        colors: &Bound<'_>,
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
                    ("grounds", grounds.clone()),
                ],
            );
        }
    }

    fn pose_group(&self, frame: &TemporalFrame) -> Option<&wgpu::BindGroup> {
        let previous = if frame.motion_valid {
            1 - frame.parity
        } else {
            frame.parity
        };
        self.pose_groups[2 * frame.parity + previous].get()
    }

    fn history_target(&self, frame: &ResolveFrame) -> Option<&wgpu::TextureView> {
        Some(&self.targets.get()?.history[frame.parity])
    }

    fn motion(&self) -> Option<&wgpu::TextureView> {
        Some(&self.targets.get()?.motion)
    }

    fn prepare_resolve(
        &mut self,
        device: &wgpu::Device,
        ([scene, occlusion, bloom, reflections, haze], settings): ([Bound<'_>; 5], Bound<'_>),
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
        let target = |view| Bound::of(targets, crate::pipelines::view(view));
        let revealed =
            revealed.unwrap_or_else(|| Bound::fixed(crate::pipelines::view(&self.no_particles)));
        let sampler = || Bound::fixed(crate::pipelines::sampler(&self.sampler));
        for (parity, group) in self.resolve_groups.iter_mut().enumerate() {
            group.bind(
                device,
                "engine/taa-resolve-group",
                (&TAA_RESOLVE, &self.resolve_layout),
                [
                    ("taa", Bound::fixed(self.uniform.as_entire_binding())),
                    ("scene", scene.clone()),
                    ("motion", target(&targets.motion)),
                    ("history", target(&targets.history[1 - parity])),
                    ("history_sampler", sampler()),
                    ("occlusion", occlusion.clone()),
                    ("bloom", bloom.clone()),
                    ("post", settings.clone()),
                    ("reflections", reflections.clone()),
                    ("depth_texture", depth.bound()),
                    ("revealed", revealed.clone()),
                    ("haze", haze.clone()),
                    ("bloom_sampler", sampler()),
                ],
            );
        }
        resolve
    }

    fn draw_resolve(&self, mut pass: wgpu::RenderPass<'_>, frame: &ResolveFrame) {
        let group = self.resolve_groups[frame.parity]
            .get()
            .expect("bound_in_prepare_resolve");
        pass.set_pipeline(&self.resolve_pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn reset(&mut self) {
        self.history = None;
    }

    fn after_submit(&mut self, frame: &ResolveFrame) {
        self.history = Some(frame.history);
        self.frame = self.frame.wrapping_add(1);
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

#[derive(Clone, Copy)]
struct ResolveFrame {
    parity: usize,
    history: History,
    revealed: bool,
}

impl GpuPart for TemporalPass {
    fn install(&mut self, table: &GraphTable) {
        self.ids = Ids {
            pass: table.pass_named("temporal_resolve"),
            canvas: table.resource("canvas"),
            scene: table.resource("scene"),
            motion: table.resource("motion"),
            depth: table.resource("depth"),
            history: table.resource("history"),
            revealed: table.resource("revealed"),
            effects: table.resource("effects"),
        };
    }

    fn begin(&mut self, cx: &mut Begin<'_>, out: &mut BeginOut) {
        let enabled = cx.snapshot.environment.temporal_antialiasing.enabled;
        if enabled == self.taa.get().is_some() {
            return;
        }
        if enabled {
            let (device, queue) = (&cx.core.device, &cx.core.queue);
            let format = cx.core.canvas_format();
            self.taa
                .create(device, "engine/taa-", || Taa::new(device, queue, format));
        } else {
            self.taa.release();
            out.forget_temporal = true;
        }
    }

    fn temporal_frame(&mut self, cx: &mut Step<'_>) -> Option<TemporalFrame> {
        let taa = self.taa.get_mut()?;
        if !cx.frame.kind.temporal {
            taa.reset();
            return None;
        }
        Some(taa.begin(
            &cx.core.device,
            &cx.core.queue,
            &cx.frame.frame,
            cx.core.display.region,
            cx.core.instances.generation(),
        ))
    }

    fn prepare(&mut self, cx: &mut Step<'_>, _prelude: &mut Prelude) {
        if cx.frame.kind.debug {
            self.resolve = None;
            return;
        }
        let (Some(frame), Some(inputs)) = (
            cx.frame.temporal.as_ref(),
            cx.parts
                .get::<PostPass>()
                .and_then(PostPass::resolve_inputs),
        ) else {
            self.resolve = None;
            return;
        };
        let revealed = cx
            .parts
            .get::<Transparency>()
            .and_then(Transparency::revealed);
        let core = &*cx.core;
        self.resolve = self.taa.get_mut().map(|taa| {
            taa.prepare_resolve(&core.device, inputs, frame, &core.depth.view, revealed)
        });
    }

    fn finish(&mut self, cx: &mut Step<'_>) {
        if cx.frame.temporal.is_none() {
            return;
        }
        let Some(taa) = self.taa.get_mut() else {
            return;
        };
        let core = &*cx.core;
        let poses = core.frame_poses(0);
        let both = [0, 1].map(|parity| poses.bound(parity).expect("temporal_frames_create"));
        taa.prepare_pose_groups(
            &core.device,
            both,
            &poses.colors.bound(),
            &Bound::fixed(core.instances.grounds().as_entire_binding()),
        );
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let (Some(frame), Some(shape)) = (self.resolve, cx.frame.shape) else {
            return;
        };
        if shape.targets != crate::graph::Targets::Temporal {
            return;
        }
        let ids = self.ids;
        let (Some(pass), Some(canvas), Some(history)) = (ids.pass, ids.canvas, ids.history) else {
            return;
        };
        let _ = graph
            .add(cx.table, pass)
            .read(ids.scene)
            .read(ids.motion)
            .read(ids.depth)
            .read(history)
            .read_if(frame.revealed, ids.revealed)
            .read_written(ids.effects)
            .color(canvas, Access::Clear(Clear::Color(BLACK)))
            .color(history, Access::Clear(Clear::Color(BLACK)));
    }

    fn encode(
        &mut self,
        _pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        let (Some(taa), Some(frame)) = (self.taa.get(), self.resolve.as_ref()) else {
            return 0;
        };
        let pass = cx.render_pass(
            encoder,
            "engine/taa-resolve",
            cx.core.profiler.span("taa_resolve").map(Span::render),
        );
        taa.draw_resolve(pass, frame);
        0
    }

    fn attachment(&self, resource: &str) -> Option<wgpu::TextureView> {
        let taa = self.taa.get()?;
        match resource {
            "motion" => taa.motion().cloned(),
            "history" => taa.history_target(self.resolve.as_ref()?).cloned(),
            _ => None,
        }
    }

    fn after_submit(&mut self) {
        if let (Some(taa), Some(frame)) = (self.taa.get_mut(), self.resolve.as_ref()) {
            taa.after_submit(frame);
        }
    }

    fn capacity_bytes(&self) -> u64 {
        self.taa.get().map_or(0, Taa::capacity_bytes)
    }

    fn stats(&self, _cx: &View<'_>, stats: &mut RenderStats) {
        stats.temporal_antialiasing = Applied {
            active: self.resolve.is_some(),
        };
    }

    fn poll(&mut self) -> bool {
        self.taa.poll()
    }

    fn disable(&mut self) -> bool {
        self.taa.disable()
    }

    fn disabled(&self) -> bool {
        self.taa.disabled()
    }

    #[cfg(feature = "dev")]
    fn inject(&mut self) -> bool {
        self.taa.inject()
    }

    #[cfg(feature = "dev")]
    fn encode_fault(&self, encoder: &mut wgpu::CommandEncoder) {
        self.taa.encode_fault(encoder);
    }
}
