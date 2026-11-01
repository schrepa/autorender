use bevy_ecs::prelude::*;
use engine_core::{DeviceLimits, Entry, Error, View, gpu_struct};
use engine_render::{
    Access, Attach, Bind, Buffer, DepthUse, Encode, FULLSCREEN, GpuPart, Install, Layer,
    Lifetime, Own, Pass, Pipeline, Prepare, RenderPipeline, Shape, Texture, ViewTarget, wgpu,
};
use glam::{Mat4, Vec3};

use crate::emitters::{Emitter, Owners, Packed, same_instances};
use crate::shaders::particles::{composite, draw, faces::transparency as face};

const ACCUMULATED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const REVEALED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;
const EMITTERS_START: u64 = 128;
const OWNERS_START: u64 = 4;
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
    pub(crate) struct Particles {
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

struct Held {
    emitter_bytes: u64,
    owner_bytes: u64,
    uploaded: Vec<Emitter>,
    packed: Option<u64>,
    written: Option<Particles>,
}

impl Held {
    fn new(room: u64) -> Self {
        Self {
            emitter_bytes: storage(EMITTERS_START.min(room)),
            owner_bytes: storage(OWNERS_START.min(room)),
            uploaded: Vec::new(),
            packed: None,
            written: None,
        }
    }
}

pub(crate) struct ParticlesGpu {
    accumulate: Pass<Self>,
    composite: Pass<Self>,
    accumulated: Texture<Self>,
    revealed: Texture<Self>,
    shown: Buffer<Self>,
    uniform: Buffer<Self>,
    emitters: Buffer<Self>,
    owners: Buffer<Self>,
    alpha: Pipeline<Self, Own>,
    mist: Pipeline<Self, Own>,
    additive: Pipeline<Self, Own>,
    compositing: Pipeline<Self, Own>,
    epoch: u64,
    drawn: Option<bool>,
    targets: bool,
    held: Option<Held>,
    drawing: Option<Counts>,
}

impl ParticlesGpu {
    fn show(&mut self, drawn: bool, cx: &mut Prepare<'_, Self>) {
        if self.drawn == Some(drawn) {
            return;
        }
        self.drawn = Some(drawn);
        cx.allocate_buffer(self.shown, SHOWN_BYTES);
        cx.write(self.shown, 0, &[u32::from(drawn), 0, 0, 0]);
    }

    fn drop_held(&mut self, cx: &mut Prepare<'_, Self>) {
        if self.held.take().is_some() {
            cx.release_buffer(self.uniform);
            cx.release_buffer(self.emitters);
            cx.release_buffer(self.owners);
        }
    }

    fn release_targets(&mut self, cx: &mut Prepare<'_, Self>) {
        if std::mem::take(&mut self.targets) {
            cx.release(self.accumulated);
            cx.release(self.revealed);
        }
    }

    fn prepare_frame(
        &mut self,
        (packed, owners, view): (&Packed, &Owners, &View),
        room: u64,
        cx: &mut Prepare<'_, Self>,
    ) -> Option<Counts> {
        let held = self.held.get_or_insert_with(|| Held::new(room));
        cx.allocate_buffer(self.uniform, size_of::<Particles>() as u64);
        cx.allocate_buffer(self.emitters, held.emitter_bytes);
        cx.allocate_buffer(self.owners, held.owner_bytes);
        if held.packed != Some(packed.revision) {
            pack(held, (packed, owners), room, (self.emitters, self.owners), cx);
            held.packed = Some(packed.revision);
        }
        if packed.total == 0 {
            return None;
        }
        let render = view.render_size();
        let uniform = Particles::new(view.view_projection(), render.x, render.y);
        if held.written != Some(uniform) {
            cx.write(self.uniform, 0, &[uniform]);
            held.written = Some(uniform);
        }
        Some(Counts {
            alpha: packed.alpha,
            mist: packed.mist,
            total: packed.total,
        })
    }
}

fn pack(
    held: &mut Held,
    (packed, owners): (&Packed, &Owners),
    room: u64,
    (emitters, owner_buffer): (Buffer<ParticlesGpu>, Buffer<ParticlesGpu>),
    cx: &mut Prepare<'_, ParticlesGpu>,
) {
    if packed.total == 0 {
        return;
    }
    let emitter_bytes = size_of_val(packed.records.as_slice()) as u64;
    if let Some(size) = grown(held.emitter_bytes, emitter_bytes, room) {
        held.emitter_bytes = size;
        cx.allocate_buffer(emitters, size);
        held.uploaded.clear();
    }
    let owner_bytes = u64::from(packed.total) * 4;
    let owners_replaced = match grown(held.owner_bytes, owner_bytes, room) {
        Some(size) => {
            held.owner_bytes = size;
            cx.allocate_buffer(owner_buffer, size);
            true
        }
        None => false,
    };
    if owners_replaced || !same_instances(&packed.records, &held.uploaded) {
        cx.write(owner_buffer, 0, &owners.owners);
    }
    if packed.records != held.uploaded {
        cx.write(emitters, 0, &packed.records);
        held.uploaded.clone_from(&packed.records);
    }
}

impl GpuPart for ParticlesGpu {
    type Reads = (
        Res<'static, Packed>,
        Res<'static, Owners>,
        Res<'static, View>,
        Res<'static, DeviceLimits>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let accumulated = cx.texture("accumulated", ACCUMULATED_FORMAT, Lifetime::Frame);
        let revealed = cx.texture("revealed", REVEALED_FORMAT, Lifetime::Kept);
        let shown = cx.buffer("transparent_drawn", Lifetime::Kept);
        let uniform = cx.buffer("particles", Lifetime::Kept);
        let emitters = cx.buffer("emitters", Lifetime::Kept);
        let owners = cx.buffer("owners", Lifetime::Kept);
        cx.frame(face::TRANSPARENT_REVEALED, Bind::Texture(revealed))?;
        cx.frame(face::TRANSPARENT_DRAWN, Bind::Buffer(shown))?;
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
                vertex: FULLSCREEN,
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
            shown,
            uniform,
            emitters,
            owners,
            alpha,
            mist,
            additive,
            compositing,
            epoch: 0,
            drawn: None,
            targets: false,
            held: None,
            drawing: None,
        })
    }

    fn prepare(
        &mut self,
        (packed, owners, view, limits): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.drawing = None;
        if self.epoch != packed.epoch {
            self.epoch = packed.epoch;
            self.drawn = None;
            self.drop_held(cx);
            self.release_targets(cx);
        }
        if !packed.emits {
            self.drop_held(cx);
        }
        self.show(packed.emits && !view.debug(), cx);
        if view.debug() {
            return;
        }
        if !packed.emits {
            self.release_targets(cx);
            return;
        }
        let size = view.display_size();
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
        let room = limits.storage_room / 16 * 16;
        self.drawing = self.prepare_frame((&*packed, &*owners, &*view), room, cx);
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
