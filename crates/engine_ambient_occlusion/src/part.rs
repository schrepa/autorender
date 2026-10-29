use engine_core::{Applied, RenderStats};
use engine_render::cached::{Bound, GroupCache, Versioned};
use engine_render::graph::{
    Access, Clear, FrameGraph, GraphTable, PassDecl, PassId, ResourceId, Stage,
    Targets as FrameTargets, WHITE,
};
use engine_render::lazy::Lazy;
use engine_render::parts::{Create, Encode, GpuPart, PartDecl, PartEntry, Prelude, Step, View};
use engine_render::pipeline_layouts::PipelineLayout;
use engine_render::pipelines::{self, Fullscreen, GroupLayout, fullscreen};
use engine_render::profiling::Span;
use engine_render::targets::{FrameRegion, ScreenTarget, draw_within};

use crate::layouts::{
    AMBIENT, AMBIENT_OCCLUSION, AMBIENT_OCCLUSION_PIPELINES, OCCLUSION_FILTER,
    OCCLUSION_FILTER_PIPELINES,
};
use crate::shaders::entry::ambient_occlusion;

const OCCLUSION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
const RAW_OCCLUSION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

pub static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "ambient_occlusion",
        brick: "ambient_occlusion",
        passes: &[PassDecl {
            name: "ambient_occlusion",
            stage: Stage::PostScene,
        }],
        resources: &[],
        hooks: &[],
        contributes: &[],
        effects: &[],
        labels: &["engine/ambient-occlusion"],
        lazy: true,
        faces: &[],
        needs: &[],
        piece: None,
    },
    create,
};

fn create(_: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(OcclusionPass::default())
}

#[derive(Clone, Copy)]
struct Frame {
    occlusion: bool,
}

#[derive(Clone, Copy, Default)]
struct Ids {
    pass: Option<PassId>,
    depth: Option<ResourceId>,
    occlusion: Option<ResourceId>,
}

#[derive(Default)]
struct OcclusionPass {
    occlusion: Lazy<Occlusion>,
    frame: Option<Frame>,
    ids: Ids,
}

fn ambient_contributed(shared: &engine_render::bindings::SharedGroups) -> bool {
    shared
        .group(AMBIENT)
        .is_some_and(|bindings| !bindings.is_empty())
}

impl GpuPart for OcclusionPass {
    fn install(&mut self, table: &GraphTable) {
        self.ids = Ids {
            pass: table.pass_named("ambient_occlusion"),
            depth: table.resource("depth"),
            occlusion: table.resource("occlusion"),
        };
    }

    fn prepare(&mut self, cx: &mut Step<'_>, _prelude: &mut Prelude) {
        self.frame = None;
        if cx.frame.kind.debug {
            return;
        }
        let core = &*cx.core;
        let device = &core.device;
        let Some(post) = cx.parts.post().filter(|post| post.active()) else {
            if let Some(occlusion) = self.occlusion.get_mut() {
                occlusion.release_targets();
            }
            return;
        };
        if !ambient_contributed(&core.shared) {
            return;
        }
        self.occlusion.create(device, "engine/ambient-occlusion-", || {
            Occlusion::new(device)
        });
        let (Some(occlusion), Some((_, uniform))) =
            (self.occlusion.get_mut(), post.resolve_inputs())
        else {
            return;
        };
        occlusion.size(device, core.display.region);
        let environment = &cx.frame.snapshot.environment;
        let ambient = environment.ambient_occlusion.sanitized().enabled();
        let sky = environment.sky_occlusion.sanitized().darkens() && environment.shadows.enabled;
        if ambient || sky {
            occlusion.bind(device, uniform, &core.depth.view);
            self.frame = Some(Frame { occlusion: ambient });
        }
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let (Some(_), Some(shape), Some(pass), Some(occlusion)) =
            (self.frame, cx.frame.shape, self.ids.pass, self.ids.occlusion)
        else {
            return;
        };
        if shape.targets == FrameTargets::Canvas {
            return;
        }
        let _ = graph
            .add(cx.table, pass)
            .read(self.ids.depth)
            .color(occlusion, Access::Draw(Clear::Color(WHITE)));
    }

    fn encode(
        &mut self,
        _pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        let (Some(_), Some(occlusion)) = (self.frame, self.occlusion.get()) else {
            return 0;
        };
        let (Some(targets), Some(raw), Some(filter)) = (
            occlusion.targets.get(),
            occlusion.occlusion_group.get(),
            occlusion.filter_group.get(),
        ) else {
            return 0;
        };
        let profiler = &cx.core.profiler;
        {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/ambient-occlusion",
                (&targets.raw_occlusion, occlusion.region.render.pixels()),
                profiler.span("ambient_occlusion").map(Span::render),
            );
            pass.set_pipeline(&occlusion.occlusion);
            pass.set_bind_group(0, raw, &[]);
            pass.set_bind_group(1, cx.core.shared_group(AMBIENT, &[]), &[]);
            pass.draw(0..3, 0..1);
        }
        let mut pass = cx.render_pass(
            encoder,
            "engine/ambient-occlusion-filter",
            profiler.span("effects_filter").map(Span::render),
        );
        pass.set_pipeline(&occlusion.filter);
        pass.set_bind_group(0, filter, &[]);
        pass.draw(0..3, 0..1);
        0
    }

    fn capacity_bytes(&self) -> u64 {
        self.occlusion.get().map_or(0, Occlusion::capacity_bytes)
    }

    fn stats(&self, _cx: &View<'_>, stats: &mut RenderStats) {
        stats.ambient_occlusion = Applied {
            active: self.frame.is_some_and(|frame| frame.occlusion),
        };
    }

    fn poll(&mut self) -> bool {
        self.occlusion.poll()
    }

    fn disable(&mut self) -> bool {
        self.occlusion.disable()
    }

    fn disabled(&self) -> bool {
        self.occlusion.disabled()
    }

    #[cfg(feature = "dev")]
    fn inject(&mut self) -> bool {
        self.occlusion.inject()
    }

    #[cfg(feature = "dev")]
    fn encode_fault(&self, encoder: &mut wgpu::CommandEncoder) {
        self.occlusion.encode_fault(encoder);
    }

    fn produces(&self, resource: ResourceId) -> bool {
        self.frame.is_some() && self.ids.occlusion == Some(resource)
    }
}

struct Targets {
    texture: wgpu::Texture,
    raw_occlusion: wgpu::TextureView,
}

impl Drop for Targets {
    fn drop(&mut self) {
        self.texture.destroy();
    }
}

struct Occlusion {
    occlusion_layout: GroupLayout,
    filter_layout: GroupLayout,
    occlusion: wgpu::RenderPipeline,
    filter: wgpu::RenderPipeline,
    targets: ScreenTarget<Targets>,
    region: FrameRegion,
    occlusion_group: GroupCache<2>,
    filter_group: GroupCache<2>,
}

impl Occlusion {
    fn new(device: &wgpu::Device) -> Self {
        let pipeline = |label, declared: &PipelineLayout, entry, format| {
            let layout = pipelines::layout_of(device, declared);
            fullscreen(
                device,
                &Fullscreen {
                    label,
                    layout: &layout,
                    vertex: ambient_occlusion::vs_fullscreen,
                    fragment: entry,
                    constants: &[],
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    depth: None,
                },
            )
        };
        Self {
            occlusion_layout: pipelines::layout(device, &AMBIENT_OCCLUSION),
            filter_layout: pipelines::layout(device, &OCCLUSION_FILTER),
            occlusion: pipeline(
                "engine/ambient-occlusion",
                &AMBIENT_OCCLUSION_PIPELINES,
                ambient_occlusion::fs_occlusion,
                RAW_OCCLUSION_FORMAT,
            ),
            filter: pipeline(
                "engine/ambient-occlusion-filter",
                &OCCLUSION_FILTER_PIPELINES,
                ambient_occlusion::fs_effects_filter,
                OCCLUSION_FORMAT,
            ),
            targets: ScreenTarget::default(),
            region: FrameRegion::default(),
            occlusion_group: GroupCache::default(),
            filter_group: GroupCache::default(),
        }
    }

    fn release_targets(&mut self) {
        self.targets.release();
        self.occlusion_group.clear();
        self.filter_group.clear();
    }

    fn capacity_bytes(&self) -> u64 {
        self.targets
            .size()
            .map_or(0, |size| u64::from(size.width) * u64::from(size.height) * 4)
    }

    fn size(&mut self, device: &wgpu::Device, region: FrameRegion) {
        let size = region.display;
        if self.targets.resizes(size) {
            let texture = engine_render::targets::texture_2d(
                device,
                "engine/ambient-occlusion-raw",
                size.pixels(),
                1,
                1,
                RAW_OCCLUSION_FORMAT,
                engine_render::targets::RENDER_TARGET,
            );
            let raw_occlusion = texture.create_view(&Default::default());
            self.targets.set(
                size,
                Targets {
                    texture,
                    raw_occlusion,
                },
            );
        }
        self.region = region;
    }

    fn bind(
        &mut self,
        device: &wgpu::Device,
        uniform: Bound<'_>,
        depth: &Versioned<wgpu::TextureView>,
    ) {
        let Some(targets) = self.targets.get() else {
            return;
        };
        self.occlusion_group.bind(
            device,
            "engine/ambient-occlusion-group",
            (&AMBIENT_OCCLUSION, &self.occlusion_layout),
            [("post", uniform.clone()), ("depth_texture", depth.bound())],
        );
        self.filter_group.bind(
            device,
            "engine/ambient-occlusion-filter-group",
            (&OCCLUSION_FILTER, &self.filter_layout),
            [
                ("post", uniform),
                (
                    "raw_occlusion",
                    Bound::of(targets, pipelines::view(&targets.raw_occlusion)),
                ),
            ],
        );
    }
}

fn fullscreen_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    label: &str,
    (view, drawn): (&wgpu::TextureView, (u32, u32)),
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) -> wgpu::RenderPass<'e> {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    draw_within(&mut pass, drawn);
    pass
}
