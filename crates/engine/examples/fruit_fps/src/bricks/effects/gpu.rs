use std::ops::Range;

use bevy_ecs::system::SystemParamItem;
use engine::prelude::*;
use engine_core::{Entry, Error, Filter, Sampler, TextureRef, Var, ViewFacts, Wrap};
use engine_drawing::StandIn;
use engine_render::{Access, BatchGroup, Encode, FULLSCREEN, GroupEntry, Own, PerBatch, Scene};

use super::assemble::BOX_VERTICES;
use super::shaders::effects::{barrier, field, glow, items, screen, sheet, slab, solid, volume};
use super::{EFFECTS_PASS, EffectsDrawing, EffectsFrame, Item};
use crate::names::{COLOR, COLOR_PASS, DEPTH};
use crate::play::light::{CLAMP, REPEAT};

const FULLSCREEN_VERTICES: u32 = 3;

const QUAD_VERTICES: u32 = 6;

const SLAB_VERTICES: u32 = 54;

const NOISE_SIDE: u32 = 32;

const NOISE_SAMPLER: Sampler = Sampler {
    wrap_u: Wrap::Repeat,
    wrap_v: Wrap::Repeat,
    wrap_w: Wrap::Repeat,
    mag_filter: Filter::Linear,
    min_filter: Filter::Linear,
    mipmap_filter: None,
    compare: None,
    anisotropy: 1,
};

fn noise_hash(value: u32) -> u32 {
    let mut x = value.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277_803_737);
    (x >> 22) ^ x
}

fn noise_volume() -> Vec<u8> {
    let total = (NOISE_SIDE * NOISE_SIDE * NOISE_SIDE) as usize;
    let mut bytes = Vec::with_capacity(total);
    for z in 0..NOISE_SIDE {
        let kz = (z + 4096).wrapping_mul(83_492_791);
        for y in 0..NOISE_SIDE {
            let kyz = (y + 4096).wrapping_mul(19_349_663) ^ kz;
            for x in 0..NOISE_SIDE {
                let seed = (x + 4096).wrapping_mul(73_856_093) ^ kyz;
                bytes.push((noise_hash(seed) >> 24) as u8);
            }
        }
    }
    bytes
}

pub(super) struct EffectsGpu {
    glow: Pipeline<Self, Scene>,
    solid: Pipeline<Self, Scene>,
    slab: Pipeline<Self, Scene>,
    sheet: Pipeline<Self, Scene>,
    barrier: Pipeline<Self, Scene>,
    smoke: Pipeline<Self, Own>,
    marks: Pipeline<Self, Own>,
    screen: Pipeline<Self, Own>,
    pass: engine_render::Pass<Self>,
    items: Buffer<Self>,
    frame: Buffer<Self>,
    noise: engine_render::Texture<Self>,
    field_group: BatchGroup<Self>,
    unlit: StandIn<Self>,
    field_bound: Option<Option<TextureRef>>,
    sent: Option<EffectsFrame>,
    glows: Vec<(Range<u32>, u32)>,
    solid_range: Range<u32>,
    slab_range: Range<u32>,
    sheet_range: Range<u32>,
    device_range: Range<u32>,
    smoke_range: Range<u32>,
    marks_range: Range<u32>,
    barrier_range: Range<u32>,
    screened: bool,
    own: bool,
}

fn blended() -> Option<wgpu::BlendState> {
    Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING)
}

fn description<'a>(
    vertex: Entry,
    fragment: Entry,
    depth: DepthUse,
    targets: &'a [(Option<wgpu::BlendState>, wgpu::ColorWrites)],
) -> RenderPipeline<'a> {
    RenderPipeline {
        vertex,
        fragment: Some(fragment),
        vertices: &[],
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..wgpu::PrimitiveState::default()
        },
        depth,
        stencil: wgpu::StencilState::default(),
        samples: 1,
        targets,
        constants: &[],
        per_batch: PerBatch::NONE,
    }
}

fn back_faces(mut pipeline: RenderPipeline<'_>) -> RenderPipeline<'_> {
    pipeline.primitive.cull_mode = Some(wgpu::Face::Front);
    pipeline
}

fn tested(write: bool) -> DepthUse {
    DepthUse::Compare {
        compare: wgpu::CompareFunction::GreaterEqual,
        write,
        bias: wgpu::DepthBiasState::default(),
    }
}

fn bound(
    cx: &mut Install<'_, EffectsGpu>,
    entries: [Entry; 2],
    resources: &[(Var, Bind<EffectsGpu>)],
) -> Vec<(Var, Bind<EffectsGpu>)> {
    resources
        .iter()
        .filter(|(var, _)| entries.iter().any(|entry| cx.reaches(*entry, *var)))
        .map(|(var, bind)| (*var, bind.clone()))
        .collect()
}

fn pulled(vertices: u32, instances: &Range<u32>) -> Shape {
    Shape::Pulled {
        vertices: 0..vertices,
        instances: instances.clone(),
        key: None,
    }
}

impl GpuPart for EffectsGpu {
    type Reads = (
        Res<'static, EffectsDrawing>,
        Query<'static, 'static, (), With<ViewSkip>>,
    );
    const NAME: &'static str = "effects";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let items_buffer = cx.buffer("effects_items", Lifetime::Kept);
        let frame_buffer = cx.buffer("effects_frame", Lifetime::Kept);
        let noise = cx.texture_3d("effects_noise", wgpu::TextureFormat::R8Unorm, Lifetime::Kept);
        let noise_sampler = cx.sampler(NOISE_SAMPLER);
        let buffers = [
            (items::EFFECT_ITEMS, Bind::Buffer(items_buffer)),
            (items::EFFECTS_FRAME, Bind::Buffer(frame_buffer)),
            (items::EFFECTS_NOISE_MAP, Bind::Texture(noise)),
            (items::EFFECTS_NOISE_SAMPLER, Bind::Sampler(noise_sampler)),
        ];
        cx.frame(volume::EFFECTS_DEPTH, Bind::Target(DEPTH))?;
        let field_group = cx.batch_group(field::MODULE, 1, &[])?;
        let unlit = StandIn::install(cx, "effects_unlit_field", [0; 4]);
        let scene = Scene::named(COLOR_PASS);
        let over = [(blended(), wgpu::ColorWrites::ALL)];
        let opaque = [(None, wgpu::ColorWrites::ALL)];
        let entries = [glow::VS_GLOW, glow::FS_GLOW];
        let list = bound(cx, entries, &buffers);
        let glow = cx.render_pipeline(
            scene,
            &description(entries[0], entries[1], tested(false), &over),
            &list,
        )?;
        let entries = [solid::VS_SOLID, solid::FS_SOLID];
        let list = bound(cx, entries, &buffers);
        let solid = cx.render_pipeline(
            scene,
            &description(entries[0], entries[1], tested(true), &opaque),
            &list,
        )?;
        let entries = [slab::VS_SLAB, slab::FS_SLAB];
        let list = bound(cx, entries, &buffers);
        let slab = cx.render_pipeline(
            scene,
            &description(entries[0], entries[1], tested(true), &opaque),
            &list,
        )?;
        let entries = [sheet::VS_SHEET, sheet::FS_SHEET];
        let list = bound(cx, entries, &buffers);
        let sheet = cx.render_pipeline(
            scene,
            &description(entries[0], entries[1], tested(true), &over),
            &list,
        )?;
        let entries = [barrier::VS_BARRIER, barrier::FS_BARRIER];
        let list = bound(cx, entries, &buffers);
        let barrier = cx.render_pipeline(
            scene,
            &description(entries[0], entries[1], tested(false), &over),
            &list,
        )?;
        let pass = cx
            .pass(EFFECTS_PASS)
            .label("fruit_fps/effects")
            .span("effects")
            .target(COLOR, Access::Load)
            .views(PassViews::Main)
            .id();
        let entries = [volume::VS_VOLUME, volume::FS_SMOKE];
        let list = bound(cx, entries, &buffers);
        let smoke = cx.render_pipeline(
            pass,
            &back_faces(description(entries[0], entries[1], DepthUse::None, &over)),
            &list,
        )?;
        let entries = [volume::VS_VOLUME, volume::FS_MARKS];
        let list = bound(cx, entries, &buffers);
        let marks = cx.render_pipeline(
            pass,
            &back_faces(description(entries[0], entries[1], DepthUse::None, &over)),
            &list,
        )?;
        let entries = [FULLSCREEN, screen::FS_SCREEN];
        let list = bound(cx, entries, &buffers);
        let screen = cx.render_pipeline(
            pass,
            &description(entries[0], entries[1], DepthUse::None, &over),
            &list,
        )?;
        Ok(Self {
            glow,
            solid,
            slab,
            sheet,
            barrier,
            smoke,
            marks,
            screen,
            pass,
            items: items_buffer,
            frame: frame_buffer,
            noise,
            field_group,
            unlit,
            field_bound: None,
            sent: None,
            glows: Vec::new(),
            solid_range: 0..0,
            slab_range: 0..0,
            sheet_range: 0..0,
            device_range: 0..0,
            smoke_range: 0..0,
            marks_range: 0..0,
            barrier_range: 0..0,
            screened: false,
            own: false,
        })
    }

    fn prepare(
        &mut self,
        (drawing, skipped): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let noise_size = wgpu::Extent3d {
            width: NOISE_SIDE,
            height: NOISE_SIDE,
            depth_or_array_layers: NOISE_SIDE,
        };
        if cx.allocate(self.noise, noise_size) == Allocation::Fresh {
            let volume = noise_volume();
            cx.write_texture(
                self.noise,
                wgpu::Origin3d::ZERO,
                &volume,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(NOISE_SIDE),
                    rows_per_image: Some(NOISE_SIDE),
                },
                noise_size,
            );
        }
        let slots = drawing
            .items
            .len()
            .max(drawing.pooled as usize)
            .max(1)
            .next_power_of_two();
        let _ = cx.allocate_buffer(self.items, (slots * size_of::<Item>()) as u64);
        if !drawing.items.is_empty() {
            cx.write(self.items, 0, &drawing.items);
        }
        let field = self.bind_field(drawing.field, cx);
        let mut frame = drawing.frame;
        if field.is_none() {
            frame.field_shape = [0.0; 4];
        }
        let bytes = size_of::<EffectsFrame>() as u64;
        if cx.allocate_buffer(self.frame, bytes) == Allocation::Fresh || self.sent != Some(frame) {
            cx.write(self.frame, 0, &[frame]);
            self.sent = Some(frame);
        }
        self.glows.clone_from(&drawing.glows);
        self.solid_range = drawing.solid.clone();
        self.slab_range = drawing.slab.clone();
        self.sheet_range = drawing.sheet.clone();
        self.device_range = drawing.device.clone();
        self.smoke_range = drawing.smoke.clone();
        self.marks_range = drawing.marks.clone();
        self.barrier_range = drawing.barrier.clone();
        self.screened = drawing.screen;
        self.own = skipped.is_empty() && drawing.own_pass();
    }

    fn view(&mut self, _view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        if self.own {
            cx.run(self.pass);
        }
    }

    fn encode(&mut self, _pass: engine_render::Pass<Self>, cx: &mut Encode<'_, Self>) {
        let mut draw = cx.render();
        if !self.marks_range.is_empty() {
            draw.draw(&self.marks, pulled(BOX_VERTICES, &self.marks_range));
        }
        if !self.smoke_range.is_empty() {
            draw.draw(&self.smoke, pulled(BOX_VERTICES, &self.smoke_range));
        }
        if self.screened {
            draw.draw(&self.screen, pulled(FULLSCREEN_VERTICES, &(0..1)));
        }
    }
}

impl EffectsGpu {
    fn bind_field(
        &mut self,
        wanted: Option<TextureRef>,
        cx: &mut Prepare<'_, Self>,
    ) -> Option<TextureRef> {
        if self.unlit.written(cx) {
            self.field_bound = None;
        }
        let shown = wanted.filter(|map| cx.texture_resident(map.texture));
        if self.field_bound != Some(shown) {
            let (texture, sampler) = match shown {
                Some(map) => (GroupEntry::Asset(map.texture), map.sampler.unwrap_or(REPEAT)),
                None => (GroupEntry::Texture(self.unlit.texture()), CLAMP),
            };
            let entries = [
                (field::EFFECTS_FIELD_MAP, texture),
                (
                    field::EFFECTS_FIELD_SAMPLER,
                    GroupEntry::AssetSampler(sampler),
                ),
            ];
            match cx.set_batch(self.field_group, None, &entries, None) {
                Ok(()) => self.field_bound = Some(shown),
                Err(error) => cx.report(error),
            }
        }
        self.field_bound.flatten()
    }

    pub(super) fn draw_effects(&self, draw: &mut Draw<'_, Self, Scene>) {
        if !self.solid_range.is_empty() {
            draw.draw(&self.solid, pulled(BOX_VERTICES, &self.solid_range));
        }
        if !self.slab_range.is_empty() {
            draw.draw(&self.slab, pulled(SLAB_VERTICES, &self.slab_range));
        }
        if !self.sheet_range.is_empty() {
            draw.draw(&self.sheet, pulled(BOX_VERTICES, &self.sheet_range));
        }
        for (range, vertices) in &self.glows {
            draw.draw(&self.glow, pulled(*vertices, range));
        }
        if !self.barrier_range.is_empty() {
            draw.draw(&self.barrier, pulled(QUAD_VERTICES, &self.barrier_range));
        }
    }

    pub(super) fn draw_device(&self, draw: &mut Draw<'_, Self, Scene>) {
        if !self.device_range.is_empty() {
            draw.draw(&self.solid, pulled(BOX_VERTICES, &self.device_range));
        }
    }
}
