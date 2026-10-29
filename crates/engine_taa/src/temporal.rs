#![expect(
    clippy::expect_used,
    reason = "the resolve reads the targets and groups its frame prepared"
)]

use crate::shaders::entry::temporal;
use engine_core::{Applied, Environment, RenderStats};
use engine_render::cached::{Bound, GroupCache, Versioned};
use engine_render::composite::{PixelEffect, PostStage, composite_entry};
use engine_render::graph::{
    Access, BLACK, Clear, FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl,
    ResourceId, Stage,
};
use engine_render::lazy::Lazy;
use engine_render::parts::{
    Begin, BeginOut, Create, Encode, GpuPart, Jitter, PartDecl, PartEntry, Prelude, Step, View,
};
use engine_render::post::PostPass;
use engine_render::profiling::Span;
use engine_render::targets::{FrameRegion, FrameSize, ScreenTarget};
use engine_render::transparent::Transparency;

use crate::bindings::TAA_RESOLVE;
use crate::gpu_layouts::ResolveUniform;
use crate::pipeline_layouts::TAA_RESOLVE_PIPELINES;

pub(crate) const RESOLVE: &str = "taa::resolve";
const COMPOSITE_GROUP: &str = "composite";
const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const CURRENT_WEIGHT: f32 = 0.1;
const JITTER_SAMPLES: u32 = 8;

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "temporal",
        brick: "taa",
        passes: &[PassDecl {
            name: "temporal_resolve",
            stage: Stage::Composite,
        }],
        resources: &[ResourceDecl {
            name: "history",
            lifetime: Lifetime::Kept,
        }],
        hooks: &[],
        contributes: &[],
        effects: &[PixelEffect {
            stage: PostStage::Resolve,
            module: RESOLVE,
        }],
        labels: &["engine/taa-", "engine/temporal"],
        lazy: true,
        faces: &[],
        needs: &[],
        piece: None,
    },
    create,
};

fn create(_: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(TemporalPass::default())
}

fn halton(mut index: u32, base: u32) -> f32 {
    let (mut scale, mut value) = (1.0, 0.0);
    while index > 0 {
        scale /= base as f32;
        value += scale * (index % base) as f32;
        index /= base;
    }
    value
}

fn jitter_offset(frame: u32) -> [f32; 2] {
    let index = frame % JITTER_SAMPLES + 1;
    [halton(index, 2) - 0.5, halton(index, 3) - 0.5]
}

struct Targets {
    textures: [wgpu::Texture; 2],
    history: [wgpu::TextureView; 2],
}

impl Drop for Targets {
    fn drop(&mut self) {
        for texture in &self.textures {
            texture.destroy();
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Ids {
    pass: Option<PassId>,
    canvas: Option<ResourceId>,
    scene: Option<ResourceId>,
    motion: Option<ResourceId>,
    previous_poses: Option<ResourceId>,
    depth: Option<ResourceId>,
    history: Option<ResourceId>,
    revealed: Option<ResourceId>,
    occlusion: Option<ResourceId>,
    reflections: Option<ResourceId>,
    haze: Option<ResourceId>,
    bloom: Option<ResourceId>,
}

#[derive(Default)]
pub(crate) struct TemporalPass {
    taa: Lazy<Taa>,
    resolve: Option<ResolveFrame>,
    ids: Ids,
}

struct Taa {
    resolve_pipeline: wgpu::RenderPipeline,
    resolve_layout: engine_render::pipelines::GroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    written: Option<ResolveUniform>,
    targets: ScreenTarget<Targets>,
    no_particles: wgpu::TextureView,
    resolve_groups: [GroupCache<11>; 2],
    frame: u32,
}

impl Taa {
    fn new(
        device: &wgpu::Device,
        queue: &engine_render::gpu_calls::Queue,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let resolve_layout = engine_render::pipelines::layout(device, &TAA_RESOLVE);
        let resolve_pipeline_layout =
            engine_render::pipelines::layout_of(device, &TAA_RESOLVE_PIPELINES);
        let resolve_pipeline = engine_render::pipelines::fullscreen(
            device,
            &engine_render::pipelines::Fullscreen {
                label: "engine/taa-resolve",
                layout: &resolve_pipeline_layout,
                vertex: temporal::vs_fullscreen,
                fragment: composite_entry(RESOLVE),
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
        let uniform = engine_render::buffers::uniform(
            device,
            "engine/taa-resolve-uniform",
            size_of::<ResolveUniform>() as u64,
        );
        let no_particles = engine_render::targets::texture_2d(
            device,
            "engine/taa-no-particles",
            (1, 1),
            1,
            1,
            engine_render::transparent::REVEALED_FORMAT,
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
            resolve_pipeline,
            resolve_layout,
            sampler,
            uniform,
            written: None,
            targets: ScreenTarget::default(),
            no_particles: no_particles.create_view(&Default::default()),
            resolve_groups: Default::default(),
            frame: 0,
        }
    }

    fn capacity_bytes(&self) -> u64 {
        self.targets.size().map_or(0, |size| {
            u64::from(size.width) * u64::from(size.height) * 8
        })
    }

    fn begin(
        &mut self,
        (device, queue): (&wgpu::Device, &engine_render::gpu_calls::Queue),
        region: FrameRegion,
        history_valid: bool,
        (environment, turn): (&Environment, f32),
    ) {
        let (size, render) = (region.display, region.render);
        if self.targets.resizes(size) {
            let targets = self.create_targets(device, size);
            self.targets.set(size, targets);
        }
        let offset = jitter_offset(self.frame);
        let settings = environment.temporal_antialiasing.sanitized();
        let uniform = ResolveUniform {
            texel: [1.0 / size.width as f32, 1.0 / size.height as f32],
            current_weight: CURRENT_WEIGHT,
            history_valid: u32::from(history_valid),
            options: [
                if environment.dither.enabled {
                    turn
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
    }

    fn history_target(&self, frame: &ResolveFrame) -> Option<&wgpu::TextureView> {
        Some(&self.targets.get()?.history[frame.parity])
    }

    fn prepare_resolve(
        &mut self,
        device: &wgpu::Device,
        ([scene, occlusion, reflections, motion], settings): ([Bound<'_>; 4], Bound<'_>),
        parity: usize,
        (depth, pieces): (&Versioned<wgpu::TextureView>, &wgpu::Buffer),
        revealed: Option<Bound<'_>>,
    ) -> ResolveFrame {
        let resolve = ResolveFrame {
            parity,
            revealed: revealed.is_some(),
        };
        let Some(targets) = self.targets.get() else {
            return resolve;
        };
        let target = |view| Bound::of(targets, engine_render::pipelines::view(view));
        let revealed = revealed
            .unwrap_or_else(|| Bound::fixed(engine_render::pipelines::view(&self.no_particles)));
        let sampler = || Bound::fixed(engine_render::pipelines::sampler(&self.sampler));
        for (parity, group) in self.resolve_groups.iter_mut().enumerate() {
            group.bind(
                device,
                "engine/taa-resolve-group",
                (&TAA_RESOLVE, &self.resolve_layout),
                [
                    ("taa", Bound::fixed(self.uniform.as_entire_binding())),
                    ("scene", scene.clone()),
                    ("motion", motion.clone()),
                    ("history", target(&targets.history[1 - parity])),
                    ("history_sampler", sampler()),
                    ("occlusion", occlusion.clone()),
                    ("post", settings.clone()),
                    ("reflections", reflections.clone()),
                    ("depth_texture", depth.bound()),
                    ("revealed", revealed.clone()),
                    ("pieces", Bound::fixed(pieces.as_entire_binding())),
                ],
            );
        }
        resolve
    }

    fn draw_resolve(
        &self,
        mut pass: wgpu::RenderPass<'_>,
        frame: &ResolveFrame,
        composite: &wgpu::BindGroup,
    ) {
        let group = self.resolve_groups[frame.parity]
            .get()
            .expect("bound_in_prepare_resolve");
        pass.set_pipeline(&self.resolve_pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.set_bind_group(1, composite, &[]);
        pass.draw(0..3, 0..1);
    }

    fn forget_groups(&mut self) {
        for group in &mut self.resolve_groups {
            group.clear();
        }
    }

    fn after_submit(&mut self) {
        self.frame = self.frame.wrapping_add(1);
    }

    fn create_targets(&self, device: &wgpu::Device, size: FrameSize) -> Targets {
        let texture = |label, format| {
            engine_render::targets::texture_2d(
                device,
                label,
                size.pixels(),
                1,
                1,
                format,
                engine_render::targets::RENDER_TARGET,
            )
        };
        let textures = [
            texture("engine/taa-history", HISTORY_FORMAT),
            texture("engine/taa-history", HISTORY_FORMAT),
        ];
        let [first, second] = textures
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
        Targets {
            textures,
            history: [first, second],
        }
    }
}

#[derive(Clone, Copy)]
struct ResolveFrame {
    parity: usize,
    revealed: bool,
}

impl GpuPart for TemporalPass {
    fn install(&mut self, table: &GraphTable) {
        self.ids = Ids {
            pass: table.pass_named("temporal_resolve"),
            canvas: table.resource("canvas"),
            scene: table.resource("scene"),
            motion: table.resource("motion"),
            previous_poses: table.resource("previous_poses"),
            depth: table.resource("depth"),
            history: table.resource("history"),
            revealed: table.resource("revealed"),
            occlusion: table.resource("occlusion"),
            reflections: table.resource("reflections"),
            haze: table.resource("haze"),
            bloom: table.resource("bloom"),
        };
        if let Some(taa) = self.taa.get_mut() {
            taa.forget_groups();
        }
    }

    fn begin(&mut self, cx: &mut Begin<'_>, _out: &mut BeginOut) {
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
        }
    }

    fn wants(&self, resource: ResourceId) -> bool {
        self.taa.get().is_some()
            && (Some(resource) == self.ids.motion || Some(resource) == self.ids.previous_poses)
    }

    fn jitter(&mut self, _cx: &mut Step<'_>) -> Option<Jitter> {
        let taa = self.taa.get()?;
        Some(Jitter {
            offset: jitter_offset(taa.frame),
            sequence: u64::from(taa.frame),
        })
    }

    fn prepare(&mut self, cx: &mut Step<'_>, _prelude: &mut Prelude) {
        if cx.frame.kind.debug {
            self.resolve = None;
            return;
        }
        let (Some(frame), Some(inputs)) = (
            cx.frame.temporal.as_ref(),
            cx.parts.post().and_then(PostPass::resolve_inputs),
        ) else {
            self.resolve = None;
            return;
        };
        let revealed = cx.parts.transparency().and_then(Transparency::revealed);
        let core = &*cx.core;
        let environment = &cx.frame.snapshot.environment;
        self.resolve = self.taa.get_mut().map(|taa| {
            taa.begin(
                (&core.device, &core.queue),
                core.display.region,
                frame.history_valid,
                (environment, frame.camera.options[2]),
            );
            taa.prepare_resolve(
                &core.device,
                inputs,
                frame.parity,
                (&core.depth.view, core.pieces_buffer()),
                revealed,
            )
        });
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let (Some(frame), Some(shape)) = (self.resolve, cx.frame.shape) else {
            return;
        };
        if shape.targets != engine_render::graph::Targets::Temporal {
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
            .read_written(ids.occlusion)
            .read_written(ids.reflections)
            .read_written(ids.haze)
            .read_written(ids.bloom)
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
        taa.draw_resolve(pass, frame, cx.core.shared_group(COMPOSITE_GROUP, &[]));
        0
    }

    fn attachment(&self, resource: &str) -> Option<wgpu::TextureView> {
        let taa = self.taa.get()?;
        match resource {
            "history" => taa.history_target(self.resolve.as_ref()?).cloned(),
            _ => None,
        }
    }

    fn after_submit(&mut self) {
        if let (Some(taa), Some(_)) = (self.taa.get_mut(), self.resolve.as_ref()) {
            taa.after_submit();
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
