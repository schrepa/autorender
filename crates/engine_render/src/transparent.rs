#![expect(
    clippy::expect_used,
    reason = "the composite reads the targets and pipelines its frame made"
)]

use crate::bindings::TRANSPARENT_COMPOSITE;
use crate::cached::Bound;
use crate::composite::{TRANSPARENT_COMPOSITE as COMPOSITE_MODULE, composite_entry};
use crate::graph::{
    Access, Clear, FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl, ResourceId,
    Stage, TRANSPARENT, WHITE,
};
use crate::parts::{
    CORE_BRICK, Create, DrawAt, Encode, GpuPart, PartDecl, PartEntry, Prelude, Step, View,
};
use crate::pipeline_layouts::TRANSPARENT_COMPOSITE_PIPELINES;
use crate::post::{PostPass, shared_groups};
use crate::profiling::Span;
use crate::shaders::entry::transparent;
use crate::targets::{FrameSize, ScreenTarget};

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
        effects: &[],
        labels: &[],
        lazy: false,
        faces: &[],
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
    layout: crate::pipelines::GroupLayout,
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
        Self {
            layout,
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
        let pipeline_layout = crate::pipelines::layout_of(device, &TRANSPARENT_COMPOSITE_PIPELINES);
        self.composite[index] = Some(crate::pipelines::fullscreen(
            device,
            &crate::pipelines::Fullscreen {
                label: "engine/transparent-composite",
                layout: &pipeline_layout,
                vertex: transparent::vs_fullscreen,
                fragment: composite_entry(COMPOSITE_MODULE),
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

    fn composite(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        post: bool,
        shared: &[(u32, &wgpu::BindGroup)],
    ) {
        let targets = self.targets.get().expect("sized_in_begin");
        pass.set_pipeline(self.composite[usize::from(post)].as_ref().expect("ensured"));
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
        };
        if let Some(surfaces) = self.surfaces.as_mut() {
            surfaces.composite = Default::default();
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
        let post = cx.parts.post().is_some_and(PostPass::active);
        let core = &*cx.core;
        let device = &core.device;
        let (surface_format, depth_format) = (core.canvas_format(), core.depth.texture.format());
        let surfaces = self
            .surfaces
            .get_or_insert_with(|| TransparentPass::new(device, surface_format, depth_format));
        surfaces.begin(device, core.display.region.display);
        surfaces.ensure_composite(device, post);
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let (true, Some(shape)) = (self.drawn, cx.frame.shape) else {
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
        let (true, Some(shape)) = (self.drawn, cx.frame.shape) else {
            return 0;
        };
        let targets = shape.targets;
        if pass == COMPOSITE_PASS {
            let post = targets != crate::graph::Targets::Canvas;
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
                post,
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
