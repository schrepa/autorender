use std::ops::Range;

use bevy_ecs::prelude::*;
use engine_core::{DeviceLimits, Entry, Error, TargetName, gpu_struct};
use engine_render::{
    Allocation, Bind, Buffer, Draw, FaceCull, Form, FormSide, GpuPart, Install, KindForm, Lifetime,
    OwnFragments, Prepare, RoleFragments, Scene, SceneDraw, Shape, Varyings,
};

use crate::PARTICLES;
use crate::emitters::{Emitter, Packed};
use crate::shaders::particles::draw;

const EMITTERS_START: u64 = size_of::<Emitter>() as u64;
const QUAD: Range<u32> = 0..6;

const fn blended(fragment: Entry) -> OwnFragments {
    OwnFragments {
        placed: RoleFragments {
            blended: Some(fragment),
            ..RoleFragments::NONE
        },
        owned: RoleFragments::NONE,
        faces: engine_render::OwnFaces::Culled(FaceCull::None),
    }
}

const ALPHA_FRAGMENTS: OwnFragments = blended(draw::FS_PARTICLE);
const MIST_FRAGMENTS: OwnFragments = blended(draw::FS_PARTICLE_MIST);
const ADDED_FRAGMENTS: OwnFragments = blended(draw::FS_PARTICLE_ADDED);

const fn particle_form(fragments: &'static OwnFragments) -> KindForm {
    KindForm {
        kind: PARTICLES,
        placed: FormSide {
            vertex: draw::VS_PARTICLE,
            streams: &[],
            depth_bias: FormSide::UNBIASED,
        },
        owned: None,
        varyings: Varyings::Own(fragments),
    }
}

const ALPHA_FORM: KindForm = particle_form(&ALPHA_FRAGMENTS);
const MIST_FORM: KindForm = particle_form(&MIST_FRAGMENTS);
const ADDED_FORM: KindForm = particle_form(&ADDED_FRAGMENTS);

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
    uniform: Buffer<Self>,
    emitters: Buffer<Self>,
    alpha: Form<Self>,
    mist: Form<Self>,
    added: Form<Self>,
    emitter_bytes: Option<u64>,
    drawing: Option<Counts>,
}

impl ParticlesGpu {
    fn release_held(&mut self, cx: &mut Prepare<'_, Self>) {
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
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.uniform, size_of::<Particles>() as u64);
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(self.emitters, *held);
        if packed.total == 0 {
            return None;
        }
        cx.write(self.emitters, 0, &packed.records);
        let values = Particles::new(packed.records.len() as u32);
        cx.write(self.uniform, 0, &[values]);
        Some(Counts {
            alpha: packed.alpha,
            mist: packed.mist,
            total: packed.total,
        })
    }

    pub(crate) fn draw(&self, _scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        let Some(counts) = self.drawing else {
            return draw.skip("no particle is alive");
        };
        let ranges: [(&Form<Self>, Range<u32>); 3] = [
            (&self.alpha, 0..counts.alpha),
            (&self.mist, counts.alpha..counts.mist),
            (&self.added, counts.mist..counts.total),
        ];
        let mut drawn = 0;
        for (form, instances) in ranges {
            if instances.is_empty() {
                continue;
            }
            let Some(pipeline) = draw.form(form) else {
                continue;
            };
            draw.draw(
                &pipeline,
                Shape::Pulled {
                    vertices: QUAD,
                    instances,
                    material: None,
                },
            );
            drawn += 1;
        }
        if drawn == 0 {
            return draw.skip("particles only blend");
        }
        drawn
    }
}

impl GpuPart for ParticlesGpu {
    type Reads = (Res<'static, Packed>, Res<'static, DeviceLimits>);

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let uniform = cx.buffer("particles", Lifetime::Kept);
        let emitters = cx.buffer("emitters", Lifetime::Kept);
        let bound = || {
            [
                (draw::PARTICLES, Bind::Buffer(uniform)),
                (draw::EMITTERS, Bind::Buffer(emitters)),
                (draw::DEPTH_TEXTURE, Bind::Target(TargetName::DEPTH)),
            ]
        };
        Ok(Self {
            uniform,
            emitters,
            alpha: cx.form(&ALPHA_FORM, &bound())?,
            mist: cx.form(&MIST_FORM, &bound())?,
            added: cx.form(&ADDED_FORM, &bound())?,
            emitter_bytes: None,
            drawing: None,
        })
    }

    fn prepare(
        &mut self,
        (packed, limits): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.drawing = None;
        if !packed.emits {
            self.release_held(cx);
            return;
        }
        let room = limits.storage_room / 16 * 16;
        self.drawing = self.prepare_frame(&packed, room, cx);
    }
}
