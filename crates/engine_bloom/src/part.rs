use engine_core::{Applied, Member, RenderStats};
use engine_render::cached::{Bound, GroupCache};
use engine_render::graph::{
    FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl, ResourceId, Stage,
};
use engine_render::lazy::Lazy;
use engine_render::parts::{
    Begin, BeginOut, Create, Encode, GpuPart, PartDecl, PartEntry, Prelude, Step, View,
};
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

pub(crate) const MEMBER: Member = Member {
    module: "bloom::apply",
    functions: &["apply"],
    reads: &["bloom"],
};

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
        resolve: None,
        labels: &["engine/bloom"],
        lazy: true,
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

fn create(cx: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(BloomPass {
        bloom: Lazy::default(),
        off: Off::new(&cx.core.device),
        enabled: false,
        runs: false,
        ids: Ids::default(),
        piece: BloomPiece::default(),
    })
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

struct Off {
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

impl Off {
    fn new(device: &wgpu::Device) -> Self {
        let texture = engine_render::targets::texture_2d(
            device,
            "engine/bloom-off",
            (1, 1),
            1,
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        Self {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/bloom-off-sampler"),
                ..Default::default()
            }),
        }
    }
}

struct BloomPass {
    bloom: Lazy<Bloom>,
    off: Off,
    enabled: bool,
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

    fn begin(&mut self, cx: &mut Begin<'_>, _out: &mut BeginOut) {
        let settings = cx.snapshot.section::<crate::Bloom>();
        self.enabled = settings.enabled();
        self.piece.threshold = [settings.threshold, 0.0, 0.0, 0.0];
        self.piece.applied = [settings.intensity, 0.0, 0.0, 0.0];
        if !self.enabled {
            self.bloom.release();
        }
    }

    fn prepare(&mut self, cx: &mut Step<'_>, _prelude: &mut Prelude) {
        self.runs = false;
        if !self.enabled || cx.frame.kind.debug {
            return;
        }
        let core = &*cx.core;
        let device = &core.device;
        let Some(post) = cx.parts.post() else {
            return;
        };
        self.bloom
            .create(device, "engine/bloom-", || Bloom::new(device));
        let (Some(bloom), Some(uniform), Some(scene), Some(glow)) = (
            self.bloom.get_mut(),
            post.uniform(),
            post.scene(),
            post.attachment("glow"),
        ) else {
            return;
        };
        bloom.begin(device, core.display.region, uniform, (scene, &glow));
        self.runs = true;
        self.piece.applied[1] = 1.0;
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let (true, Some(_), Some(pass)) = (self.runs, cx.frame.shape, self.ids.pass) else {
            return;
        };
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
        if let Some(bloom) = self.bloom.get()
            && let Some(targets) = bloom.targets.get()
            && let Some(level) = targets.levels.first()
        {
            out.push(("bloom", Bound::of(targets, pipelines::view(level))));
            out.push((
                "bloom_sampler",
                Bound::fixed(pipelines::sampler(&bloom.sampler)),
            ));
            return;
        }
        out.push(("bloom", Bound::fixed(pipelines::view(&self.off.view))));
        out.push((
            "bloom_sampler",
            Bound::fixed(pipelines::sampler(&self.off.sampler)),
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
