use std::ops::Range;

use bevy_ecs::prelude::*;
use engine_core::{
    AttachPoint, DeviceLimits, DrawKind, Entry, Error, PassName, PassViews, SlotSet, TargetName,
    Views, Writes, gpu_struct,
};
use engine_render::{
    Access, Attach, Bind, Buffer, DepthUse, Draw, Encode, FULLSCREEN, GpuPart, Install, Lifetime,
    Own, Pass, Pipeline, Prepare, RenderPipeline, Scene, SceneDraw, Shape, Texture, wgpu,
};

use crate::PARTICLES;
use crate::emitters::{Emitter, Packed};
use crate::shaders::particles::{composite, draw, faces::transparency as face};

const ACCUMULATED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const REVEALED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;
const EMITTERS_START: u64 = size_of::<Emitter>() as u64;
const SHOWN_BYTES: u64 = 16;

const ADD: wgpu::BlendComponent = wgpu::BlendComponent {
    src_factor: wgpu::BlendFactor::One,
    dst_factor: wgpu::BlendFactor::One,
    operation: wgpu::BlendOperation::Add,
};

const MULTIPLY: wgpu::BlendComponent = wgpu::BlendComponent {
    src_factor: wgpu::BlendFactor::Zero,
    dst_factor: wgpu::BlendFactor::OneMinusSrc,
    operation: wgpu::BlendOperation::Add,
};

const ACCUMULATION: [(Option<wgpu::BlendState>, wgpu::ColorWrites); 2] = [
    (
        Some(wgpu::BlendState {
            color: ADD,
            alpha: ADD,
        }),
        wgpu::ColorWrites::ALL,
    ),
    (
        Some(wgpu::BlendState {
            color: MULTIPLY,
            alpha: MULTIPLY,
        }),
        wgpu::ColorWrites::ALL,
    ),
];

const ADDITIVE: [(Option<wgpu::BlendState>, wgpu::ColorWrites); 1] = [(
    Some(wgpu::BlendState {
        color: ADD,
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
    }),
    wgpu::ColorWrites::ALL,
)];

const OVER: [(Option<wgpu::BlendState>, wgpu::ColorWrites); 1] = [(
    Some(wgpu::BlendState {
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
    }),
    wgpu::ColorWrites::ALL,
)];

fn weighted(kind: DrawKind) -> bool {
    kind == PARTICLES
}

pub(crate) struct WeightedPass;

impl PassName for WeightedPass {
    const NAME: &'static str = "transparent";
    const PASS: engine_core::Pass = engine_core::Pass {
        writes: Writes(&[TargetName::TRANSPARENCY]),
        reads: SlotSet {
            position: true,
            base_color: true,
            opacity: true,
            emissive: true,
            ..SlotSet::NONE
        },
        views: PassViews::Owned,
        at: AttachPoint::Transparent,
        takes: weighted,
    };
}

pub(crate) struct CompositePass;

impl PassName for CompositePass {
    const NAME: &'static str = "transparent_composite";
    const PASS: engine_core::Pass = engine_core::Pass {
        writes: Writes(&[TargetName::COLOR]),
        reads: SlotSet {
            position: true,
            base_color: true,
            emissive: true,
            ..SlotSet::NONE
        },
        views: PassViews::Owned,
        at: AttachPoint::Transparent,
        takes: weighted,
    };
}

gpu_struct! {
    pub(crate) struct Particles {
        listed: [u32; 4],
    }
}

impl Particles {
    fn new(listed: u32) -> Self {
        Self {
            listed: [listed, 0, 0, 0],
        }
    }
}

fn particle(fragment: Entry) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex: draw::VS_PARTICLE,
        fragment: Some(fragment),
        vertices: &[],
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth: DepthUse::Compare {
            compare: wgpu::CompareFunction::Greater,
            write: false,
            bias: wgpu::DepthBiasState::default(),
        },
        blend: None,
        writes: wgpu::ColorWrites::ALL,
    }
}

fn storage(bytes: u64) -> u64 {
    bytes.max(16).next_multiple_of(16)
}

fn grown(capacity: u64, bytes: u64, room: u64) -> Option<u64> {
    let size = bytes.next_power_of_two().min(room);
    if bytes <= capacity || size <= capacity {
        None
    } else {
        Some(storage(size))
    }
}

#[derive(Clone, Copy)]
struct Counts {
    alpha: u32,
    mist: u32,
    total: u32,
}

pub(crate) struct ParticlesGpu {
    accumulate: Pass<Self>,
    composite: Pass<Self>,
    accumulated: Texture<Self>,
    revealed: Texture<Self>,
    shown: Buffer<Self>,
    uniform: Buffer<Self>,
    emitters: Buffer<Self>,
    compositing: Pipeline<Self, Own>,
    alpha: Pipeline<Self, Scene>,
    mist: Pipeline<Self, Scene>,
    additive: Pipeline<Self, Scene>,
    targets: bool,
    emitter_bytes: Option<u64>,
    drawing: Option<Counts>,
}

impl ParticlesGpu {
    fn show(&self, drawn: bool, cx: &mut Prepare<'_, Self>) {
        cx.allocate_buffer(self.shown, SHOWN_BYTES);
        cx.write(self.shown, 0, &[u32::from(drawn), 0, 0, 0]);
    }

    fn release_held(&mut self, cx: &mut Prepare<'_, Self>) {
        if std::mem::take(&mut self.targets) {
            cx.release(self.accumulated);
            cx.release(self.revealed);
        }
        if self.emitter_bytes.take().is_some() {
            cx.release_buffer(self.uniform);
            cx.release_buffer(self.emitters);
        }
    }

    fn prepare_frame(
        &mut self,
        packed: &Packed,
        room: u64,
        cx: &mut Prepare<'_, Self>,
    ) -> Option<Counts> {
        let held = self
            .emitter_bytes
            .get_or_insert(storage(EMITTERS_START.min(room)));
        if let Some(size) = grown(*held, size_of_val(packed.records.as_slice()) as u64, room) {
            *held = size;
        }
        cx.allocate_buffer(self.uniform, size_of::<Particles>() as u64);
        cx.allocate_buffer(self.emitters, *held);
        if packed.total == 0 {
            return None;
        }
        cx.write(self.emitters, 0, &packed.records);
        let uniform = Particles::new(packed.records.len() as u32);
        cx.write(self.uniform, 0, &[uniform]);
        Some(Counts {
            alpha: packed.alpha,
            mist: packed.mist,
            total: packed.total,
        })
    }

    pub(crate) fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        let Some(counts) = self.drawing else {
            return 0;
        };
        let ranges: [(&Pipeline<Self, Scene>, Range<u32>); 2] =
            if scene.pass == Scene::of::<WeightedPass>() {
                [
                    (&self.alpha, 0..counts.alpha),
                    (&self.mist, counts.alpha..counts.mist),
                ]
            } else if scene.pass == Scene::of::<CompositePass>() {
                [
                    (&self.additive, counts.mist..counts.total),
                    (&self.additive, counts.total..counts.total),
                ]
            } else {
                return 0;
            };
        let mut drawn = 0;
        for (pipeline, instances) in ranges {
            if instances.is_empty() {
                continue;
            }
            draw.draw(
                pipeline,
                Shape::Pulled {
                    vertices: 0..4,
                    instances,
                },
            );
            drawn += 1;
        }
        drawn
    }
}

impl GpuPart for ParticlesGpu {
    type Reads = (
        Res<'static, Packed>,
        Res<'static, Views>,
        Res<'static, DeviceLimits>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let accumulated = cx.texture("accumulated", ACCUMULATED_FORMAT, Lifetime::Frame);
        let revealed = cx.texture("revealed", REVEALED_FORMAT, Lifetime::Kept);
        let shown = cx.buffer("transparent_drawn", Lifetime::Kept);
        let uniform = cx.buffer("particles", Lifetime::Kept);
        let emitters = cx.buffer("emitters", Lifetime::Kept);
        cx.frame(face::TRANSPARENT_REVEALED, Bind::Texture(revealed))?;
        cx.frame(face::TRANSPARENT_DRAWN, Bind::Buffer(shown))?;
        let accumulate = cx
            .scene::<WeightedPass>()
            .at(WeightedPass::PASS.at)
            .span(WeightedPass::NAME)
            .color(accumulated, Access::Clear)
            .color_cleared(revealed, wgpu::Color::WHITE)
            .id();
        let composite_pass = cx
            .scene::<CompositePass>()
            .at(CompositePass::PASS.at)
            .span(CompositePass::NAME)
            .id();
        let compositing = cx.render_pipeline_targets(
            composite_pass,
            &RenderPipeline {
                vertex: FULLSCREEN,
                fragment: Some(composite::FS_COMPOSITE),
                vertices: &[],
                primitive: wgpu::PrimitiveState::default(),
                depth: DepthUse::None,
                blend: None,
                writes: wgpu::ColorWrites::ALL,
            },
            &OVER,
            &[
                (composite::ACCUMULATED, Bind::Texture(accumulated)),
                (composite::REVEALED, Bind::Texture(revealed)),
            ],
        )?;
        let bound = || {
            [
                (draw::PARTICLES, Bind::Buffer(uniform)),
                (draw::EMITTERS, Bind::Buffer(emitters)),
                (draw::DEPTH_TEXTURE, Bind::Target(TargetName::DEPTH)),
            ]
        };
        let (weighted, over) = (Scene::of::<WeightedPass>(), Scene::of::<CompositePass>());
        let alpha = cx.render_pipeline_targets(
            weighted,
            &particle(draw::FS_PARTICLE),
            &ACCUMULATION,
            &bound(),
        )?;
        let mist = cx.render_pipeline_targets(
            weighted,
            &particle(draw::FS_PARTICLE_MIST),
            &ACCUMULATION,
            &bound(),
        )?;
        let additive = cx.render_pipeline_targets(
            over,
            &particle(draw::FS_PARTICLE_ADD_POST),
            &ADDITIVE,
            &bound(),
        )?;
        Ok(Self {
            accumulate,
            composite: composite_pass,
            accumulated,
            revealed,
            shown,
            uniform,
            emitters,
            compositing,
            alpha,
            mist,
            additive,
            targets: false,
            emitter_bytes: None,
            drawing: None,
        })
    }

    fn prepare(
        &mut self,
        (packed, views, limits): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.drawing = None;
        self.show(packed.emits, cx);
        if !packed.emits {
            self.release_held(cx);
            return;
        }
        let room = limits.storage_room / 16 * 16;
        self.drawing = self.prepare_frame(&packed, room, cx);
        let size = views.frame().display_size;
        let extent = wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        };
        cx.allocate(self.accumulated, extent, 1);
        cx.allocate(self.revealed, extent, 1);
        self.targets = true;
        cx.run(self.accumulate);
        cx.run(self.composite);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        let mut draw = cx.render(Attach::Declared);
        if pass == self.composite {
            draw.draw(
                &self.compositing,
                Shape::Pulled {
                    vertices: 0..3,
                    instances: 0..1,
                },
            );
        }
        draw.scene(self)
    }
}
