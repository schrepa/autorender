#![expect(
    clippy::expect_used,
    reason = "the composite reads the targets and pipelines its frame made"
)]

use engine_core::{Drawn, RenderStats};

use crate::bindings::{GroupRef, TRANSPARENT_COMPOSITE};
use crate::cached::Bound;
use crate::graph::{
    Access, Clear, FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl, ResourceId,
    Stage, TRANSPARENT, WHITE,
};
use crate::parts::{Create, Encode, GpuPart, PartDecl, PartEntry, Prelude, Step, View};
use crate::particles::{ParticleFrame, ParticlePass};
use crate::pipeline_layouts::{PARTICLE_PIPELINES, TRANSPARENT_COMPOSITE_PIPELINES};
use crate::post::PostPass;
use crate::profiling::Span;
use crate::shaders::entry::transparent;
use crate::targets::{FrameSize, ScreenTarget};

pub(crate) const ACCUMULATED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub(crate) const REVEALED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "transparency",
        brick: "core",
        passes: &[
            PassDecl {
                name: "transparent",
                stage: Stage::Transparent,
            },
            PassDecl {
                name: "transparent_composite",
                stage: Stage::Transparent,
            },
        ],
        resources: &[
            ResourceDecl {
                name: "accumulated",
                lifetime: Lifetime::Frame,
            },
            ResourceDecl {
                name: "revealed",
                lifetime: Lifetime::Frame,
            },
        ],
        hooks: &[],
        contributes: &[],
        effects: &[],
        labels: &[],
        lazy: false,
    },
    create,
};

const COMPOSITE_PASS: usize = 1;

fn create(_: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(Transparency::default())
}

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
    composite: [Option<wgpu::RenderPipeline>; 4],
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
        let pipeline_layout = crate::pipelines::layout_of(device, &TRANSPARENT_COMPOSITE_PIPELINES);
        Self {
            layout,
            pipeline_layout,
            composite: Default::default(),
            surface_format,
            depth_format,
            targets: ScreenTarget::default(),
        }
    }

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

    fn release_targets(&mut self) {
        self.targets.release();
    }

    fn capacity_bytes(&self) -> u64 {
        self.targets.size().map_or(0, |size| {
            u64::from(size.width) * u64::from(size.height) * 10
        })
    }

    fn views(&self) -> Option<[&wgpu::TextureView; 2]> {
        let targets = self.targets.get()?;
        Some([&targets.accumulated, &targets.revealed])
    }

    fn ensure_composite(&mut self, device: &wgpu::Device, post: bool, dither: bool) {
        let index = composite_index(post, dither);
        if self.composite[index].is_some() {
            return;
        }
        let over = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
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
        let constants = [
            ("TARGET_IS_SRGB", if format.is_srgb() { 1.0 } else { 0.0 }),
            ("DITHER", if dither { 1.0 } else { 0.0 }),
        ];
        self.composite[index] = Some(crate::pipelines::fullscreen(
            device,
            &crate::pipelines::Fullscreen {
                label: "engine/transparent-composite",
                layout: &self.pipeline_layout,
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

    fn composite(&self, pass: &mut wgpu::RenderPass<'_>, post: bool, dither: bool) {
        let targets = self.targets.get().expect("sized_in_begin");
        pass.set_pipeline(
            self.composite[composite_index(post, dither)]
                .as_ref()
                .expect("ensured"),
        );
        pass.set_bind_group(0, &targets.group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn composite_index(post: bool, dither: bool) -> usize {
    usize::from(post) + 2 * usize::from(dither)
}

#[derive(Clone, Copy, Default)]
struct Ids {
    transparent: Option<PassId>,
    composite: Option<PassId>,
    accumulated: Option<ResourceId>,
    revealed: Option<ResourceId>,
    depth: Option<ResourceId>,
    scene: Option<ResourceId>,
    glow: Option<ResourceId>,
    shadow_maps: Option<ResourceId>,
}

#[derive(Default)]
pub(crate) struct Transparency {
    surfaces: Option<TransparentPass>,
    particles: Option<ParticlePass>,
    frame: Option<TransparencyFrame>,
    ids: Ids,
}

struct TransparencyFrame {
    particles: ParticleFrame,
    dither: bool,
}

impl Transparency {
    fn prepare_frame(&mut self, cx: &Step<'_>, post: bool) -> Option<TransparencyFrame> {
        let (device, queue) = (&cx.core.device, &cx.core.queue);
        let snapshot = cx.frame.snapshot;
        let emitters = (cx.frame.frame.identity(), cx.frame.frame.emitters_revision());
        let (region, depth) = (cx.core.display.region, &cx.core.depth.view);
        let (surface_format, depth_format) =
            (cx.core.canvas_format(), cx.core.depth.texture.format());
        let particles =
            if snapshot.environment.particles.enabled && !snapshot.particles.list.is_empty() {
                let particles = self.particles.get_or_insert_with(|| {
                    ParticlePass::new(
                        device,
                        surface_format,
                        depth_format,
                        &accumulation_targets(),
                    )
                });
                let prepared =
                    particles.prepare(device, queue, snapshot, emitters, region.render, depth);
                if prepared.is_some() {
                    particles.ensure_additive(device, post);
                }
                prepared
            } else {
                None
            };
        let Some(particles) = particles else {
            if let Some(surfaces) = self.surfaces.as_mut() {
                surfaces.release_targets();
            }
            return None;
        };
        let surfaces = self
            .surfaces
            .get_or_insert_with(|| TransparentPass::new(device, surface_format, depth_format));
        surfaces.begin(device, region.display);
        let dither = snapshot.environment.dither.enabled;
        surfaces.ensure_composite(device, post, dither);
        Some(TransparencyFrame { particles, dither })
    }

    fn surfaces(&self) -> &TransparentPass {
        self.surfaces
            .as_ref()
            .expect("transparent_surfaces_created")
    }

    pub(crate) fn revealed(&self) -> Option<Bound<'_>> {
        if self.frame.is_none() {
            return None;
        }
        let targets = self.surfaces.as_ref()?.targets.get()?;
        Some(Bound::of(
            targets,
            crate::pipelines::view(&targets.revealed),
        ))
    }
}

impl GpuPart for Transparency {
    fn install(&mut self, table: &GraphTable) {
        self.ids = Ids {
            transparent: table.pass_named("transparent"),
            composite: table.pass_named("transparent_composite"),
            accumulated: table.resource("accumulated"),
            revealed: table.resource("revealed"),
            depth: table.resource("depth"),
            scene: table.resource("scene"),
            glow: table.resource("glow"),
            shadow_maps: table.resource("shadow_maps"),
        };
    }

    fn prepare(&mut self, cx: &mut Step<'_>, _prelude: &mut Prelude) {
        if cx.frame.kind.debug {
            self.frame = None;
            return;
        }
        let post = cx.parts.get::<PostPass>().is_some_and(PostPass::active);
        self.frame = self.prepare_frame(cx, post);
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let (Some(_), Some(shape)) = (&self.frame, cx.frame.shape) else {
            return;
        };
        let ids = self.ids;
        let (Some(transparent), Some(composite), Some(accumulated), Some(revealed), Some(depth)) = (
            ids.transparent,
            ids.composite,
            ids.accumulated,
            ids.revealed,
            ids.depth,
        ) else {
            return;
        };
        let _ = graph
            .add(cx.table, transparent)
            .read_written(ids.shadow_maps)
            .color(accumulated, Access::Clear(Clear::Color(TRANSPARENT)))
            .color(revealed, Access::Clear(Clear::Color(WHITE)))
            .depth(depth, Access::ReadOnly);
        let Some(scene) = crate::parts::scene_target(cx.table, &shape) else {
            return;
        };
        let mut node = graph
            .add(cx.table, composite)
            .read(accumulated)
            .read(revealed)
            .color(scene, Access::Load);
        if Some(scene) == ids.scene
            && let Some(glow) = ids.glow
        {
            node = node.color(glow, Access::Load);
        }
        let _ = node.depth(depth, Access::ReadOnly);
    }

    fn encode(
        &mut self,
        pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        let Some(frame) = &self.frame else {
            return 0;
        };
        let Some(&GroupRef::Shared { group, without }) = PARTICLE_PIPELINES.groups.first() else {
            return 0;
        };
        let camera = cx.core.shared_group(group, without);
        if pass == COMPOSITE_PASS {
            let post = cx
                .frame
                .shape
                .is_some_and(|shape| shape.targets != crate::graph::Targets::Canvas);
            let mut render = cx.render_pass(
                encoder,
                "engine/transparent-composite",
                cx.core
                    .profiler
                    .span("transparent_composite")
                    .map(Span::render),
            );
            self.surfaces().composite(&mut render, post, frame.dither);
            if let Some(particles) = &self.particles {
                particles.draw_additive(&mut render, camera, &frame.particles, post);
            }
            return 0;
        }
        let mut render = cx.render_pass(
            encoder,
            "engine/transparent-accumulation",
            cx.core.profiler.span("transparent").map(Span::render),
        );
        if let Some(particles) = &self.particles {
            particles.draw_alpha(&mut render, camera, &frame.particles);
        }
        0
    }

    fn attachment(&self, resource: &str) -> Option<wgpu::TextureView> {
        let [accumulated, revealed] = self.surfaces.as_ref()?.views()?;
        match resource {
            "accumulated" => Some(accumulated.clone()),
            "revealed" => Some(revealed.clone()),
            _ => None,
        }
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

    fn stats(&self, cx: &View<'_>, stats: &mut RenderStats) {
        let snapshot = cx.frame.snapshot;
        let particles = self.frame.as_ref().map(|frame| &frame.particles);
        stats.particles = Drawn {
            drawn: particles.map_or(0, |frame| frame.total),
            invalid: snapshot.particles.invalid.min(u32::MAX as usize) as u32,
            dropped: (snapshot.particles.dropped as u64
                + particles.map_or(0, |frame| frame.dropped))
            .min(u64::from(u32::MAX)) as u32,
        };
    }
}
