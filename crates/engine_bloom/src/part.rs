use engine_core::{Applied, RenderStats};
use engine_render::cached::{Bound, GroupCache};
use engine_render::composite::{PixelEffect, PostStage};
use engine_render::graph::{
    FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl, ResourceId, Stage,
    Targets as FrameTargets,
};
use engine_render::lazy::Lazy;
use engine_render::parts::{Create, Encode, GpuPart, PartDecl, PartEntry, Prelude, Step, View};
use engine_render::pieces::PieceDecl;
use engine_render::pipelines::{self, Fullscreen, GroupLayout, fullscreen};
use engine_render::profiling::GpuProfiler;
use engine_render::targets::{FrameRegion, FrameSize, ScreenTarget, fullscreen_pass};

use crate::gpu_layouts::BloomPiece;
use crate::layouts::{
    BLOOM_FIRST, BLOOM_FIRST_PIPELINES, BLOOM_LEVEL, BLOOM_LEVEL_PIPELINES, COMPOSITE_BLOOM,
};
use crate::shaders::entry::bloom;

const BLOOM_LEVELS: u32 = 6;

pub static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "bloom",
        brick: "bloom",
        passes: &[PassDecl {
            name: "bloom",
            stage: Stage::PostLight,
        }],
        resources: &[ResourceDecl {
            name: "bloom",
            lifetime: Lifetime::Frame,
        }],
        hooks: &[],
        contributes: &[COMPOSITE_BLOOM],
        effects: &[PixelEffect {
            stage: PostStage::Light,
            module: "bloom::apply",
        }],
        labels: &["engine/bloom"],
        lazy: true,
        faces: &[],
        needs: &[],
        piece: Some(PieceDecl {
            name: "bloom",
            module: "bloom::piece",
            ty: "BloomPiece",
            size: size_of::<BloomPiece>() as u64,
        }),
    },
    create,
};

fn create(_: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(BloomPass::default())
}

fn bloom_format(device: &wgpu::Device) -> wgpu::TextureFormat {
    if device
        .features()
        .contains(wgpu::Features::RG11B10UFLOAT_RENDERABLE)
    {
        wgpu::TextureFormat::Rg11b10Ufloat
    } else {
        wgpu::TextureFormat::Rgba16Float
    }
}

#[derive(Clone, Copy, Default)]
struct Ids {
    pass: Option<PassId>,
    scene: Option<ResourceId>,
    glow: Option<ResourceId>,
    reflections: Option<ResourceId>,
    bloom: Option<ResourceId>,
}

#[derive(Default)]
struct BloomPass {
    bloom: Lazy<Bloom>,
    runs: bool,
    ids: Ids,
    piece: BloomPiece,
}

impl GpuPart for BloomPass {
    fn install(&mut self, table: &GraphTable) {
        self.ids = Ids {
            pass: table.pass_named("bloom"),
            scene: table.resource("scene"),
            glow: table.resource("glow"),
            reflections: table.resource("reflections"),
            bloom: table.resource("bloom"),
        };
    }

    fn prepare(&mut self, cx: &mut Step<'_>, _prelude: &mut Prelude) {
        let settings = cx.frame.snapshot.environment.bloom.sanitized();
        self.piece.threshold = [settings.threshold, 0.0, 0.0, 0.0];
        self.runs = false;
        if cx.frame.kind.debug {
            return;
        }
        let core = &*cx.core;
        let device = &core.device;
        let Some(post) = cx.parts.post().filter(|post| post.active()) else {
            if let Some(bloom) = self.bloom.get_mut() {
                bloom.release_targets();
            }
            return;
        };
        self.bloom
            .create(device, "engine/bloom-", || Bloom::new(device));
        let (Some(bloom), Some((_, uniform)), Some(scene), Some(glow)) = (
            self.bloom.get_mut(),
            post.resolve_inputs(),
            post.scene(),
            post.attachment("glow"),
        ) else {
            return;
        };
        bloom.begin(device, core.display.region, uniform, (scene, &glow));
        self.runs = settings.enabled();
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let (true, Some(shape), Some(pass)) = (self.runs, cx.frame.shape, self.ids.pass) else {
            return;
        };
        if shape.targets == FrameTargets::Canvas {
            return;
        }
        let ids = self.ids;
        let _ = graph
            .add(cx.table, pass)
            .read(ids.scene)
            .read(ids.glow)
            .read_written(ids.reflections)
            .write(ids.bloom);
    }

    fn encode(
        &mut self,
        _pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        if let (true, Some(bloom)) = (self.runs, self.bloom.get()) {
            bloom.encode(encoder, &cx.core.profiler);
        }
        0
    }

    fn contribute<'s>(&'s self, group: &str, out: &mut Vec<(&'static str, Bound<'s>)>) {
        if group != COMPOSITE_BLOOM.group {
            return;
        }
        let Some(bloom) = self.bloom.get() else {
            return;
        };
        let Some(targets) = bloom.targets.get() else {
            return;
        };
        let Some(level) = targets.levels.first() else {
            return;
        };
        out.push(("bloom", Bound::of(targets, pipelines::view(level))));
        out.push((
            "bloom_sampler",
            Bound::fixed(pipelines::sampler(&bloom.sampler)),
        ));
    }

    fn capacity_bytes(&self) -> u64 {
        self.bloom.get().map_or(0, Bloom::capacity_bytes)
    }

    fn stats(&self, _cx: &View<'_>, stats: &mut RenderStats) {
        stats.bloom = Applied { active: self.runs };
    }

    fn poll(&mut self) -> bool {
        self.bloom.poll()
    }

    fn disable(&mut self) -> bool {
        self.bloom.disable()
    }

    fn disabled(&self) -> bool {
        self.bloom.disabled()
    }

    #[cfg(feature = "dev")]
    fn inject(&mut self) -> bool {
        self.bloom.inject()
    }

    #[cfg(feature = "dev")]
    fn encode_fault(&self, encoder: &mut wgpu::CommandEncoder) {
        self.bloom.encode_fault(encoder);
    }

    fn piece(&self) -> Option<&[u8]> {
        Some(bytemuck::bytes_of(&self.piece))
    }

    fn produces(&self, resource: ResourceId) -> bool {
        self.runs && self.ids.bloom == Some(resource)
    }
}

struct Targets {
    texture: wgpu::Texture,
    levels: Vec<wgpu::TextureView>,
}

impl Drop for Targets {
    fn drop(&mut self) {
        self.texture.destroy();
    }
}

struct Bloom {
    first_layout: GroupLayout,
    level_layout: GroupLayout,
    first_pipeline: wgpu::RenderPipeline,
    down_pipeline: wgpu::RenderPipeline,
    up_pipeline: wgpu::RenderPipeline,
    format: wgpu::TextureFormat,
    sampler: wgpu::Sampler,
    targets: ScreenTarget<Targets>,
    region: FrameRegion,
    first_group: GroupCache<4>,
    down_groups: Vec<GroupCache<3>>,
    up_groups: Vec<GroupCache<3>>,
}

impl Bloom {
    fn new(device: &wgpu::Device) -> Self {
        let format = bloom_format(device);
        let first_pipelines = pipelines::layout_of(device, &BLOOM_FIRST_PIPELINES);
        let level_pipelines = pipelines::layout_of(device, &BLOOM_LEVEL_PIPELINES);
        let pipeline = |label, layout: &wgpu::PipelineLayout, entry, blend| {
            fullscreen(
                device,
                &Fullscreen {
                    label,
                    layout,
                    vertex: bloom::vs_fullscreen,
                    fragment: entry,
                    constants: &[],
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    depth: None,
                },
            )
        };
        let add = Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/bloom-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        Self {
            first_layout: pipelines::layout(device, &BLOOM_FIRST),
            level_layout: pipelines::layout(device, &BLOOM_LEVEL),
            first_pipeline: pipeline(
                "engine/bloom-first",
                &first_pipelines,
                bloom::fs_bloom_first,
                None,
            ),
            down_pipeline: pipeline(
                "engine/bloom-down",
                &level_pipelines,
                engine_render::shaders::entry::post::fs_levels_down,
                None,
            ),
            up_pipeline: pipeline("engine/bloom-up", &level_pipelines, bloom::fs_bloom_up, add),
            format,
            sampler,
            targets: ScreenTarget::default(),
            region: FrameRegion::default(),
            first_group: GroupCache::default(),
            down_groups: Vec::new(),
            up_groups: Vec::new(),
        }
    }

    fn release_targets(&mut self) {
        self.targets.release();
        self.first_group.clear();
        self.down_groups.clear();
        self.up_groups.clear();
    }

    fn capacity_bytes(&self) -> u64 {
        self.targets.size().map_or(0, |size| {
            let pixels = u64::from(size.width) * u64::from(size.height);
            pixels / 4 * 8 * 4 / 3
        })
    }

    fn begin(
        &mut self,
        device: &wgpu::Device,
        region: FrameRegion,
        uniform: Bound<'_>,
        (scene, glow): (Bound<'_>, &wgpu::TextureView),
    ) {
        let size = region.display;
        if self.targets.resizes(size) {
            let targets = self.create_targets(device, size);
            self.targets.set(size, targets);
        }
        self.region = region;
        let sampler = || Bound::fixed(pipelines::sampler(&self.sampler));
        self.first_group.bind(
            device,
            "engine/bloom-group",
            (&BLOOM_FIRST, &self.first_layout),
            [
                ("post", uniform.clone()),
                ("scene", scene),
                ("glow", Bound::fixed(pipelines::view(glow))),
                ("linear", sampler()),
            ],
        );
        let Some(targets) = self.targets.get() else {
            return;
        };
        let level = |view| Bound::of(targets, pipelines::view(view));
        let count = targets.levels.len().saturating_sub(1);
        self.down_groups.resize_with(count, GroupCache::default);
        self.up_groups.resize_with(count, GroupCache::default);
        for (index, group) in self.down_groups.iter_mut().enumerate() {
            group.bind(
                device,
                "engine/bloom-group",
                (&BLOOM_LEVEL, &self.level_layout),
                [
                    ("post", uniform.clone()),
                    ("source", level(&targets.levels[index])),
                    ("linear", sampler()),
                ],
            );
        }
        for (index, group) in self.up_groups.iter_mut().enumerate() {
            group.bind(
                device,
                "engine/bloom-group",
                (&BLOOM_LEVEL, &self.level_layout),
                [
                    ("post", uniform.clone()),
                    ("source", level(&targets.levels[index + 1])),
                    ("linear", sampler()),
                ],
            );
        }
    }

    fn create_targets(
        &self,
        device: &wgpu::Device,
        FrameSize { width, height }: FrameSize,
    ) -> Targets {
        let half = ((width / 2).max(1), (height / 2).max(1));
        let levels = BLOOM_LEVELS.min(engine_render::gpu_data::mip_levels(half.0, half.1));
        let texture = engine_render::targets::texture_2d(
            device,
            "engine/bloom",
            half,
            1,
            levels,
            self.format,
            engine_render::targets::RENDER_TARGET,
        );
        let levels = (0..levels)
            .map(|level| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        Targets { texture, levels }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder, profiler: &GpuProfiler) {
        let (Some(targets), Some(first_group)) = (self.targets.get(), self.first_group.get())
        else {
            return;
        };
        let levels = targets.levels.len();
        let span = profiler.span("bloom");
        let timed = |first: bool, last: bool| {
            span.filter(|_| first || last)
                .map(|span| span.render_part(first, last))
        };
        let drawn = |level: usize| self.region.level_region(level as u32);
        {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/bloom",
                (&targets.levels[0], drawn(0)),
                timed(true, levels == 1),
            );
            pass.set_pipeline(&self.first_pipeline);
            pass.set_bind_group(0, first_group, &[]);
            pass.draw(0..3, 0..1);
        }
        for level in 1..levels {
            let Some(group) = self.down_groups[level - 1].get() else {
                continue;
            };
            let mut pass = fullscreen_pass(
                encoder,
                "engine/bloom",
                (&targets.levels[level], drawn(level)),
                None,
            );
            pass.set_pipeline(&self.down_pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
        for level in (0..levels - 1).rev() {
            let Some(group) = self.up_groups[level].get() else {
                continue;
            };
            let mut pass = fullscreen_pass(
                encoder,
                "engine/bloom",
                (&targets.levels[level], drawn(level)),
                timed(false, level == 0),
            );
            pass.set_pipeline(&self.up_pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
