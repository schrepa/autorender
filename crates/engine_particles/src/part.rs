use bevy_ecs::prelude::*;
use engine_core::{Entry, Error, View, gpu_struct};
use engine_render::{
    Access, Attach, Bind, Buffer, DebugShown, DepthUse, Encode, GpuPart, Install, Layer, Lifetime,
    Own, Pass, Pipeline, Prepare, RenderPipeline, Shape, Texture, ViewTarget, wgpu,
};
use glam::{Mat4, Vec3};

use crate::emitters::{Owners, Packed, STORAGE_ROOM};
use crate::shaders::particles::{composite, draw, faces::transparency as face};

const ACCUMULATED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const REVEALED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;
const EMITTERS_START: u64 = 128;
const OWNERS_START: u64 = 4;

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

const ADDITIVE: [(Option<wgpu::BlendState>, wgpu::ColorWrites); 2] = [
    (
        Some(wgpu::BlendState {
            color: ADD,
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        wgpu::ColorWrites::ALL,
    ),
    (
        Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Max,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        }),
        wgpu::ColorWrites::ALL,
    ),
];

const OVER: [(Option<wgpu::BlendState>, wgpu::ColorWrites); 2] = [
    (
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
    ),
    (None, wgpu::ColorWrites::empty()),
];

gpu_struct! {
    struct Particles {
        right: [f32; 4],
        up: [f32; 4],
        size: [f32; 4],
    }
}

impl Particles {
    fn new(view_projection: Mat4, width: u32, height: u32) -> Self {
        let axis = |row: usize| {
            view_projection
                .row(row)
                .truncate()
                .try_normalize()
                .unwrap_or(Vec3::X)
                .extend(0.0)
                .to_array()
        };
        let (width, height) = (width.max(1) as f32, height.max(1) as f32);
        Self {
            right: axis(0),
            up: axis(1),
            size: [width, height, 1.0 / width, 1.0 / height],
        }
    }
}

fn particle(fragment: Entry) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex: draw::VS_PARTICLE,
        fragment: Some(fragment),
        debug: None,
        vertices: &[],
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth: DepthUse::Test,
        blend: None,
        writes: wgpu::ColorWrites::ALL,
    }
}

fn grown(capacity: u64, bytes: u64) -> u64 {
    let size = bytes.next_power_of_two().min(STORAGE_ROOM);
    if bytes <= capacity || size <= capacity {
        capacity
    } else {
        size
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
    uniform: Buffer<Self>,
    emitters: Buffer<Self>,
    owners: Buffer<Self>,
    alpha: Pipeline<Self, Own>,
    mist: Pipeline<Self, Own>,
    additive: Pipeline<Self, Own>,
    compositing: Pipeline<Self, Own>,
    emitter_bytes: u64,
    owner_bytes: u64,
    drawing: Option<Counts>,
}

impl ParticlesGpu {
    fn upload(
        &mut self,
        packed: &Res<'_, Packed>,
        owners: &Res<'_, Owners>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let emitter_bytes = grown(
            self.emitter_bytes,
            size_of_val(packed.records.as_slice()) as u64,
        );
        let owner_bytes = grown(self.owner_bytes, size_of_val(owners.owners.as_slice()) as u64);
        let owners_grown = owner_bytes != self.owner_bytes;
        (self.emitter_bytes, self.owner_bytes) = (emitter_bytes, owner_bytes);
        cx.allocate_buffer(self.uniform, size_of::<Particles>() as u64);
        cx.allocate_buffer(self.emitters, emitter_bytes);
        cx.allocate_buffer(self.owners, owner_bytes);
        if packed.is_changed() && !packed.records.is_empty() {
            cx.write(self.emitters, 0, &packed.records);
        }
        if (owners.is_changed() || owners_grown) && !owners.owners.is_empty() {
            cx.write(self.owners, 0, &owners.owners);
        }
    }
}

impl GpuPart for ParticlesGpu {
    type Reads = (
        Res<'static, Packed>,
        Res<'static, Owners>,
        Res<'static, View>,
        Res<'static, DebugShown>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let accumulated = cx.texture("accumulated", ACCUMULATED_FORMAT, Lifetime::Frame);
        let revealed = cx.texture("revealed", REVEALED_FORMAT, Lifetime::Frame);
        let uniform = cx.buffer("particles", Lifetime::Kept);
        let emitters = cx.buffer("emitters", Lifetime::Kept);
        let owners = cx.buffer("owners", Lifetime::Kept);
        cx.frame(face::TRANSPARENT_REVEALED, Bind::Texture(revealed))?;
        let accumulate = cx
            .pass("transparent")
            .at(Layer::Blend)
            .color(accumulated, Access::Clear)
            .color_cleared(revealed, wgpu::Color::WHITE)
            .id();
        let composite_pass = cx.pass("transparent_composite").at(Layer::Blend).id();
        let bound = || {
            [
                (draw::PARTICLES, Bind::Buffer(uniform)),
                (draw::EMITTERS, Bind::Buffer(emitters)),
                (draw::OWNERS, Bind::Buffer(owners)),
                (draw::DEPTH_TEXTURE, Bind::Target(ViewTarget::Depth)),
            ]
        };
        let alpha = cx.render_pipeline_targets(
            accumulate,
            &particle(draw::FS_PARTICLE),
            &ACCUMULATION,
            &bound(),
        )?;
        let mist = cx.render_pipeline_targets(
            accumulate,
            &particle(draw::FS_PARTICLE_MIST),
            &ACCUMULATION,
            &bound(),
        )?;
        let additive = cx.render_pipeline_targets(
            composite_pass,
            &particle(draw::FS_PARTICLE_ADD_POST),
            &ADDITIVE,
            &bound(),
        )?;
        let compositing = cx.render_pipeline_targets(
            composite_pass,
            &RenderPipeline {
                vertex: composite::VS_COMPOSITE,
                fragment: Some(composite::FS_COMPOSITE),
                debug: None,
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
        Ok(Self {
            accumulate,
            composite: composite_pass,
            accumulated,
            revealed,
            uniform,
            emitters,
            owners,
            alpha,
            mist,
            additive,
            compositing,
            emitter_bytes: EMITTERS_START,
            owner_bytes: OWNERS_START,
            drawing: None,
        })
    }

    fn prepare(
        &mut self,
        (packed, owners, view, shown): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.drawing = None;
        self.upload(&packed, &owners, cx);
        if shown.owner.is_some() {
            return;
        }
        let size = view.target_size();
        let extent = wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        };
        cx.allocate(self.accumulated, extent, 1);
        cx.allocate(self.revealed, extent, 1);
        cx.run(self.accumulate);
        cx.run(self.composite);
        if packed.total == 0 {
            return;
        }
        let render = view.render_size();
        cx.write(
            self.uniform,
            0,
            &[Particles::new(view.view_projection(), render.x, render.y)],
        );
        self.drawing = Some(Counts {
            alpha: packed.alpha,
            mist: packed.mist,
            total: packed.total,
        });
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        let drawing = self.drawing;
        let mut draw = cx.render(Attach::Declared, None);
        if pass == self.accumulate {
            if let Some(counts) = drawing.filter(|counts| counts.mist > 0) {
                if counts.alpha > 0 {
                    draw.draw(
                        &self.alpha,
                        Shape::Pulled {
                            vertices: 0..4,
                            instances: 0..counts.alpha,
                        },
                    );
                }
                if counts.mist > counts.alpha {
                    draw.draw(
                        &self.mist,
                        Shape::Pulled {
                            vertices: 0..4,
                            instances: counts.alpha..counts.mist,
                        },
                    );
                }
            }
            return 0;
        }
        draw.draw(
            &self.compositing,
            Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
            },
        );
        if let Some(counts) = drawing.filter(|counts| counts.total != counts.mist) {
            draw.draw(
                &self.additive,
                Shape::Pulled {
                    vertices: 0..4,
                    instances: counts.mist..counts.total,
                },
            );
        }
        0
    }
}
