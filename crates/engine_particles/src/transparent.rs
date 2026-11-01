#![expect(
    clippy::expect_used,
    reason = "the composite reads the targets and pipelines its frame made"
)]

use engine_render::bindings::TRANSPARENT_COMPOSITE;
use engine_render::cached::Bound;
use engine_render::composite::{TRANSPARENT_COMPOSITE as COMPOSITE_MODULE, composite_entry};
use engine_render::graph::{
    Access, Clear, FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl, ResourceId,
    Stage, TRANSPARENT, WHITE,
};
use engine_render::parts::{
    CORE_BRICK, Create, DrawAt, Encode, GpuPart, PartDecl, PartEntry, Prelude, Step, View,
};
use engine_render::pipeline_layouts::TRANSPARENT_COMPOSITE_PIPELINES;
use engine_render::post::shared_groups;
use engine_render::profiling::Span;
use engine_render::shaders::entry::transparent;
use engine_render::targets::{FrameSize, ScreenTarget};

pub const ACCUMULATED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const REVEALED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;

pub static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "transparency",
        brick: CORE_BRICK,
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
        resolve: None,
        labels: &[],
        lazy: false,
        needs: &[],
        piece: None,
    },
    create,
};

const COMPOSITE_PASS: usize = 1;

fn create(_: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(Transparency::default())
}

pub fn accumulation_targets() -> [Option<wgpu::ColorTargetState>; 2] {
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
    layout: engine_render::pipelines::GroupLayout,
    composite: Option<wgpu::RenderPipeline>,
    depth_format: wgpu::TextureFormat,
    targets: ScreenTarget<Targets>,
}

impl TransparentPass {
    fn new(device: &wgpu::Device, depth_format: wgpu::TextureFormat) -> Self {
        let layout = engine_render::pipelines::layout(device, &TRANSPARENT_COMPOSITE);
        Self {
            layout,
            composite: None,
            depth_format,
            targets: ScreenTarget::default(),
        }
    }

    fn begin(&mut self, device: &wgpu::Device, size: FrameSize) {
        if !self.targets.resizes(size) {
            return;
        }
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
            texture("engine/transparent-accumulated", ACCUMULATED_FORMAT),
            texture("engine/transparent-revealed", REVEALED_FORMAT),
        ];
        let [accumulated, revealed] = textures
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
        let group = engine_render::pipelines::group(
            device,
            "engine/transparent-composite-group",
            &TRANSPARENT_COMPOSITE,
            &self.layout,
            &[
                ("accumulated", engine_render::pipelines::view(&accumulated)),
                ("revealed", engine_render::pipelines::view(&revealed)),
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

    fn ensure_composite(&mut self, device: &wgpu::Device) {
        if self.composite.is_some() {
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
        let targets = [
            Some(wgpu::ColorTargetState {
                format: engine_render::post::SCENE_FORMAT,
                blend: Some(over),
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: engine_render::post::GLOW_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::empty(),
            }),
        ];
        let pipeline_layout = engine_render::pipelines::layout_of(device, &TRANSPARENT_COMPOSITE_PIPELINES);
        self.composite = Some(engine_render::pipelines::fullscreen(
            device,
            &engine_render::pipelines::Fullscreen {
                label: "engine/transparent-composite",
                layout: &pipeline_layout,
                vertex: transparent::vs_fullscreen,
                fragment: composite_entry(COMPOSITE_MODULE),
                constants: &[],
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

    fn composite(&self, pass: &mut wgpu::RenderPass<'_>, shared: &[(u32, &wgpu::BindGroup)]) {
        let targets = self.targets.get().expect("sized_in_begin");
        pass.set_pipeline(self.composite.as_ref().expect("ensured"));
        pass.set_bind_group(0, &targets.group, &[]);
        for &(index, group) in shared {
            pass.set_bind_group(index, group, &[]);
        }
        pass.draw(0..3, 0..1);
    }
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
}

#[derive(Default)]
pub struct Transparency {
    surfaces: Option<TransparentPass>,
    drawn: bool,
    ids: Ids,
}

impl Transparency {
    fn surfaces(&self) -> &TransparentPass {
        self.surfaces
            .as_ref()
            .expect("transparent_surfaces_created")
    }

    pub fn revealed(&self) -> Option<Bound<'_>> {
        if !self.drawn {
            return None;
        }
        let targets = self.surfaces.as_ref()?.targets.get()?;
        Some(Bound::of(
            targets,
            engine_render::pipelines::view(&targets.revealed),
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
        };
        if let Some(surfaces) = self.surfaces.as_mut() {
            surfaces.composite = None;
        }
    }

    fn prepare(&mut self, cx: &mut Step<'_>, _prelude: &mut Prelude) {
        if cx.frame.kind.debug {
            self.drawn = false;
            return;
        }
        self.drawn = self
            .ids
            .accumulated
            .is_some_and(|accumulated| cx.parts.iter().any(|(_, part)| part.wants(accumulated)));
        if !self.drawn {
            if let Some(surfaces) = self.surfaces.as_mut() {
                surfaces.release_targets();
            }
            return;
        }
        let core = &*cx.core;
        let device = &core.device;
        let depth_format = core.depth.texture.format();
        let surfaces = self
            .surfaces
            .get_or_insert_with(|| TransparentPass::new(device, depth_format));
        surfaces.begin(device, core.display.region.display);
        surfaces.ensure_composite(device);
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let (true, Some(_)) = (self.drawn, cx.frame.shape) else {
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
            .color(accumulated, Access::Clear(Clear::Color(TRANSPARENT)))
            .color(revealed, Access::Clear(Clear::Color(WHITE)))
            .depth(depth, Access::ReadOnly);
        let (Some(scene), Some(glow)) = (ids.scene, ids.glow) else {
            return;
        };
        let _ = graph
            .add(cx.table, composite)
            .read(accumulated)
            .read(revealed)
            .color(scene, Access::Load)
            .color(glow, Access::Load)
            .depth(depth, Access::ReadOnly);
    }

    fn encode(
        &mut self,
        pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        let (true, Some(shape)) = (self.drawn, cx.frame.shape) else {
            return 0;
        };
        let targets = shape.targets;
        if pass == COMPOSITE_PASS {
            let mut render = cx.render_pass(
                encoder,
                "engine/transparent-composite",
                cx.core
                    .profiler
                    .span("transparent_composite")
                    .map(Span::render),
            );
            self.surfaces().composite(
                &mut render,
                &shared_groups(cx.core, &TRANSPARENT_COMPOSITE_PIPELINES),
            );
            let _ = cx.draw_hook(
                DrawAt::Transparent {
                    targets,
                    accumulated: false,
                },
                &mut render,
            );
            return 0;
        }
        let mut render = cx.render_pass(
            encoder,
            "engine/transparent-accumulation",
            cx.core.profiler.span("transparent").map(Span::render),
        );
        let _ = cx.draw_hook(
            DrawAt::Transparent {
                targets,
                accumulated: true,
            },
            &mut render,
        );
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
    }
}
