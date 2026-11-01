use engine_core::{Brick, Bricks};
use engine_render::bindings::{self, Sample, Sampling, Stages, Table};
use engine_render::graph::{GraphTable, Targets};
use engine_render::parts::{
    Create, DrawAt, GpuPart, Hook, OpaqueSlot, PartDecl, PartEntry, Step, View,
};
use engine_render::pipelines::{self, Fragment, Geometry};
use engine_render::render_bricks::AddRenderParts;
use engine_render::renderer::{Core, DEPTH_FORMAT};
use engine_render::shaders::{Entry, Stage};
use engine_render::targets::texture_2d;

const LIGHT: [u8; 4] = [224, 64, 32, 255];

const DARK: [u8; 4] = [32, 96, 224, 255];

const TEXELS: [[u8; 4]; 4] = [LIGHT, DARK, DARK, LIGHT];

const SPRITE: Table = Table {
    label: "sprite/sprite-layout",
    bindings: &[
        bindings::texture(
            "checker",
            Sample::Float,
            bindings::View::D2,
            Stages::FRAGMENT,
        ),
        bindings::sampler("checker_sampler", Sampling::Filtering, Stages::FRAGMENT),
    ],
};

engine_render::pipeline_layouts! {
    SPRITE_PIPELINES = [SPRITE];
}

engine_render::shader_modules! {
    "sprite::draw" in "sprite" = "sprite.wgsl";
}

const VERTEX: Entry = Entry {
    module: "sprite::draw",
    name: "vs_sprite",
    stage: Stage::Vertex,
};

const FRAGMENT: Entry = Entry {
    module: "sprite::draw",
    name: "fs_sprite",
    stage: Stage::Fragment,
};

pub(crate) struct SpriteBrick;

impl Brick for SpriteBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .render_part(&PART)
            .shaders(MODULES)
            .pipeline_layouts(PIPELINE_LAYOUTS);
    }
}

static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "sprite",
        brick: "sprite",
        passes: &[],
        resources: &[],
        hooks: &[Hook::Opaque(OpaqueSlot::Objects)],
        contributes: &[],
        resolve: None,
        labels: &[],
        lazy: false,
        needs: &[],
        piece: None,
    },
    create,
};

fn create(cx: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(SpritePart::new(cx.core))
}

struct SpritePart {
    group: wgpu::BindGroup,
    pipeline: Option<wgpu::RenderPipeline>,
    drawn: bool,
}

impl SpritePart {
    fn new(core: &Core) -> Self {
        let device = &core.device;
        let checker = texture_2d(
            device,
            "sprite/checker",
            (2, 2),
            1,
            1,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        core.queue.write_texture(
            checker.as_image_copy(),
            TEXELS.as_flattened(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(2),
            },
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
        );
        let view = checker.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sprite/checker"),
            ..Default::default()
        });
        let layout = pipelines::layout(device, &SPRITE);
        let group = pipelines::group(
            device,
            "sprite/checker",
            &SPRITE,
            &layout,
            &[
                ("checker", pipelines::view(&view)),
                ("checker_sampler", pipelines::sampler(&sampler)),
            ],
        );
        Self {
            group,
            pipeline: None,
            drawn: false,
        }
    }
}

fn pipeline(core: &Core) -> wgpu::RenderPipeline {
    let device = &core.device;
    let layout = pipelines::layout_of(device, &SPRITE_PIPELINES);
    let targets = Targets::Post.color_targets(core.display.format());
    pipelines::geometry(
        device,
        &Geometry {
            label: "sprite/draw",
            layout: &layout,
            vertex: VERTEX,
            vertex_constants: &[],
            buffers: &[],
            primitive: wgpu::PrimitiveState::default(),
            depth: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            fragment: Some(Fragment {
                entry: FRAGMENT,
                constants: &[],
                targets: &targets,
            }),
        },
    )
}

impl GpuPart for SpritePart {
    fn install(&mut self, _table: &GraphTable) {
        self.pipeline = None;
    }

    fn forget_targets(&mut self, _targets: Targets) {
        self.pipeline = None;
    }

    fn finish(&mut self, cx: &mut Step<'_>) {
        self.drawn = cx
            .frame
            .shape
            .is_some_and(|shape| shape.targets == Targets::Post);
        if self.drawn && self.pipeline.is_none() {
            self.pipeline = Some(pipeline(cx.core));
        }
    }

    fn draw(&self, _at: DrawAt<'_>, _cx: &View<'_>, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        let (true, Some(pipeline)) = (self.drawn, &self.pipeline) else {
            return 0;
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.draw(0..6, 0..1);
        1
    }
}
