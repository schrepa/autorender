use std::ops::Range;

use bevy_ecs::system::SystemParamItem;
use engine::prelude::*;
use engine_core::{Entry, Error, Var, ViewFacts};
use engine_render::{Access, Attach, Encode, FULLSCREEN, Own, PerBatch, Scene};

use super::assemble::BOX_VERTICES;
use super::shaders::effects::{glow, items, screen, solid, volume};
use super::{EFFECTS_PASS, EffectsDrawing, EffectsFrame, Item};
use crate::names::{COLOR, COLOR_PASS, DEPTH};

const FULLSCREEN_VERTICES: u32 = 3;

pub(super) struct EffectsGpu {
    glow: Pipeline<Self, Scene>,
    solid: Pipeline<Self, Scene>,
    smoke: Pipeline<Self, Own>,
    marks: Pipeline<Self, Own>,
    screen: Pipeline<Self, Own>,
    pass: engine_render::Pass<Self>,
    items: Buffer<Self>,
    frame: Buffer<Self>,
    sent: Option<EffectsFrame>,
    glow_range: Range<u32>,
    solid_range: Range<u32>,
    device_range: Range<u32>,
    smoke_range: Range<u32>,
    marks_range: Range<u32>,
    glow_vertices: u32,
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
    buffers: [(Var, Buffer<EffectsGpu>); 2],
) -> Vec<(Var, Bind<EffectsGpu>)> {
    buffers
        .into_iter()
        .filter(|(var, _)| entries.iter().any(|entry| cx.reaches(*entry, *var)))
        .map(|(var, buffer)| (var, Bind::Buffer(buffer)))
        .collect()
}

fn pulled(vertices: u32, instances: &Range<u32>) -> Shape<'static, EffectsGpu, Scene> {
    Shape::Pulled {
        vertices: 0..vertices,
        instances: instances.clone(),
        key: None,
    }
}

fn pulled_own(vertices: u32, instances: Range<u32>) -> Shape<'static, EffectsGpu, Own> {
    Shape::Pulled {
        vertices: 0..vertices,
        instances,
        key: None,
    }
}

impl GpuPart for EffectsGpu {
    type Reads = Res<'static, EffectsDrawing>;
    const NAME: &'static str = "effects";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let items_buffer = cx.buffer("effects_items", Lifetime::Kept);
        let frame_buffer = cx.buffer("effects_frame", Lifetime::Kept);
        let buffers = [
            (items::EFFECT_ITEMS, items_buffer),
            (items::EFFECTS_FRAME, frame_buffer),
        ];
        cx.frame(volume::EFFECTS_DEPTH, Bind::Target(DEPTH))?;
        let scene = Scene::named(COLOR_PASS);
        let over = [(blended(), wgpu::ColorWrites::ALL)];
        let opaque = [(None, wgpu::ColorWrites::ALL)];
        let entries = [glow::VS_GLOW, glow::FS_GLOW];
        let list = bound(cx, entries, buffers);
        let glow = cx.render_pipeline(
            scene,
            &description(entries[0], entries[1], tested(false), &over),
            &list,
        )?;
        let entries = [solid::VS_SOLID, solid::FS_SOLID];
        let list = bound(cx, entries, buffers);
        let solid = cx.render_pipeline(
            scene,
            &description(entries[0], entries[1], tested(true), &opaque),
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
        let list = bound(cx, entries, buffers);
        let smoke = cx.render_pipeline(
            pass,
            &description(entries[0], entries[1], DepthUse::None, &over),
            &list,
        )?;
        let entries = [volume::VS_VOLUME, volume::FS_MARKS];
        let list = bound(cx, entries, buffers);
        let marks = cx.render_pipeline(
            pass,
            &description(entries[0], entries[1], DepthUse::None, &over),
            &list,
        )?;
        let entries = [FULLSCREEN, screen::FS_SCREEN];
        let list = bound(cx, entries, buffers);
        let screen = cx.render_pipeline(
            pass,
            &description(entries[0], entries[1], DepthUse::None, &over),
            &list,
        )?;
        Ok(Self {
            glow,
            solid,
            smoke,
            marks,
            screen,
            pass,
            items: items_buffer,
            frame: frame_buffer,
            sent: None,
            glow_range: 0..0,
            solid_range: 0..0,
            device_range: 0..0,
            smoke_range: 0..0,
            marks_range: 0..0,
            glow_vertices: 0,
            screened: false,
            own: false,
        })
    }

    fn prepare(
        &mut self,
        drawing: SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let slots = drawing.items.len().max(1).next_power_of_two();
        let _ = cx.allocate_buffer(self.items, (slots * size_of::<Item>()) as u64);
        if !drawing.items.is_empty() {
            cx.write(self.items, 0, &drawing.items);
        }
        let bytes = size_of::<EffectsFrame>() as u64;
        if cx.allocate_buffer(self.frame, bytes) == Allocation::Fresh
            || self.sent != Some(drawing.frame)
        {
            cx.write(self.frame, 0, &[drawing.frame]);
            self.sent = Some(drawing.frame);
        }
        self.glow_range = drawing.glow.clone();
        self.solid_range = drawing.solid.clone();
        self.device_range = drawing.device.clone();
        self.smoke_range = drawing.smoke.clone();
        self.marks_range = drawing.marks.clone();
        self.glow_vertices = drawing.glow_vertices;
        self.screened = drawing.screen;
        self.own = drawing.own_pass();
    }

    fn view(&mut self, _view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        if self.own {
            cx.run(self.pass);
        }
    }

    fn encode(&mut self, _pass: engine_render::Pass<Self>, cx: &mut Encode<'_, Self>) {
        let mut draw = cx.render(Attach::Declared);
        if !self.marks_range.is_empty() {
            draw.draw(
                &self.marks,
                pulled_own(BOX_VERTICES, self.marks_range.clone()),
            );
        }
        if !self.smoke_range.is_empty() {
            draw.draw(
                &self.smoke,
                pulled_own(BOX_VERTICES, self.smoke_range.clone()),
            );
        }
        if self.screened {
            draw.draw(&self.screen, pulled_own(FULLSCREEN_VERTICES, 0..1));
        }
    }
}

impl EffectsGpu {
    pub(super) fn draw_effects(&self, draw: &mut Draw<'_, Self, Scene>) {
        if !self.solid_range.is_empty() {
            draw.draw(&self.solid, pulled(BOX_VERTICES, &self.solid_range));
        }
        if !self.glow_range.is_empty() {
            draw.draw(&self.glow, pulled(self.glow_vertices, &self.glow_range));
        }
    }

    pub(super) fn draw_device(&self, draw: &mut Draw<'_, Self, Scene>) {
        if !self.device_range.is_empty() {
            draw.draw(&self.solid, pulled(BOX_VERTICES, &self.device_range));
        }
    }
}
