use std::ops::Range;

use bevy_ecs::prelude::*;
use engine_core::math::max;
use engine_core::{Entry, Error, TextureHandle};
use engine_places::ui::UiKind;
use engine_render::{
    Bind, Buffer, DepthUse, Draw, GpuPart, Install, Lifetime, Overlay, Pipeline, Prepare,
    RenderPipeline, SamplerDesc, Shape, wgpu,
};

use crate::UiRect;
use crate::tree::{Flat, UiState};

const MISSING: [f32; 4] = [1.0, 0.0, 1.0, 1.0];
const NO_TINT: [f32; 4] = [1.0; 4];

#[derive(Clone, Copy)]
struct Look {
    texture: Option<TextureHandle>,
    color: [f32; 4],
    border: [f32; 4],
    radius: f32,
    edge: f32,
    uv: [f32; 4],
}

struct Painter {
    rects: Vec<UiRect>,
    runs: Vec<(Option<TextureHandle>, Range<u32>)>,
    cut: u32,
    limit: usize,
    metrics: [f32; 4],
}

impl Painter {
    fn paint(&mut self, flat: &Flat, color: [f32; 4]) {
        let node = &flat.node;
        let frame = Look {
            texture: None,
            color,
            border: node.border_color,
            radius: node.radius,
            edge: node.border_width,
            uv: [0.0; 4],
        };
        let framed = color[3] > 0.0 || (node.border_width > 0.0 && node.border_color[3] > 0.0);
        match node.kind {
            UiKind::Image => {
                let look = if flat.texture.is_some() {
                    Look {
                        texture: flat.texture,
                        color: if color[3] > 0.0 { color } else { NO_TINT },
                        uv: node.image_rect,
                        ..frame
                    }
                } else {
                    Look {
                        color: MISSING,
                        ..frame
                    }
                };
                self.push(flat, flat.at, &look);
            }
            UiKind::Bar => {
                if framed {
                    self.push(flat, flat.at, &frame);
                }
                let [x, y, width, height] = flat.at;
                let filled = Look {
                    color: node.bar_color,
                    edge: 0.0,
                    ..frame
                };
                self.push(flat, [x, y, width * flat.fill, height], &filled);
            }
            UiKind::Panel | UiKind::Label | UiKind::Button | UiKind::List => {
                if framed {
                    self.push(flat, flat.at, &frame);
                }
            }
        }
    }

    fn push(&mut self, flat: &Flat, [x, y, width, height]: [f32; 4], look: &Look) {
        let [display_x, display_y, ratio, scale] = self.metrics;
        let (display_x, display_y) = (max(display_x, 1.0), max(display_y, 1.0));
        let pixel = |value: f32| (value * ratio).round();
        let rect = [pixel(x), pixel(y), pixel(x + width), pixel(y + height)];
        let clip = flat
            .clip
            .map_or([0.0, 0.0, display_x, display_y], |[x, y, width, height]| {
                [pixel(x), pixel(y), pixel(x + width), pixel(y + height)]
            });
        if rect[0] >= rect[2]
            || rect[1] >= rect[3]
            || rect[2] <= clip[0]
            || rect[0] >= clip[2]
            || rect[3] <= clip[1]
            || rect[1] >= clip[3]
        {
            return;
        }
        if self.rects.len() >= self.limit {
            self.cut += 1;
            return;
        }
        let at = self.rects.len() as u32;
        self.rects.push(UiRect {
            place: [
                rect[0] / display_x * 2.0 - 1.0,
                1.0 - rect[1] / display_y * 2.0,
                rect[2] / display_x * 2.0 - 1.0,
                1.0 - rect[3] / display_y * 2.0,
            ],
            rect,
            color: look.color,
            border: look.border,
            shape: [
                look.radius * scale * ratio,
                look.edge * scale * ratio,
                0.0,
                0.0,
            ],
            uv: look.uv,
            clip,
        });
        if let Some((texture, range)) = self.runs.last_mut()
            && *texture == look.texture
        {
            range.end = at + 1;
        } else {
            self.runs.push((look.texture, at..at + 1));
        }
    }
}

fn color_of(state: &UiState, at: u32, flat: &Flat) -> [f32; 4] {
    let node = &flat.node;
    if node.kind != UiKind::Button {
        return node.color;
    }
    let here = Some(at);
    let chosen = if state.pressed == here {
        node.press_color.or(node.hover_color)
    } else if state.hovered == here {
        node.hover_color
    } else if state.focused == here {
        node.focus_color
    } else {
        None
    };
    chosen.unwrap_or(node.color)
}

pub(crate) fn stage(state: &mut UiState, max_rects: usize) {
    let mut painter = Painter {
        rects: std::mem::take(&mut state.rects),
        runs: std::mem::take(&mut state.runs),
        cut: 0,
        limit: max_rects,
        metrics: state.metrics,
    };
    painter.rects.clear();
    painter.runs.clear();
    for (at, flat) in state.nodes.iter().enumerate() {
        painter.paint(flat, color_of(state, at as u32, flat));
    }
    state.rects = painter.rects;
    state.runs = painter.runs;
    state.rects_cut = painter.cut;
}

fn quad(vertex: Entry, fragment: Entry) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex,
        fragment: Some(fragment),
        debug: None,
        vertices: &[],
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::None,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        writes: wgpu::ColorWrites::ALL,
    }
}

pub(crate) struct UiGpu {
    solid: Pipeline<Self, Overlay>,
    image: Pipeline<Self, Overlay>,
    instances: Buffer<Self>,
    drawn: Vec<(Option<Pipeline<Self, Overlay>>, Range<u32>)>,
}

impl GpuPart for UiGpu {
    type Reads = Res<'static, UiState>;

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use crate::shaders::ui::draw;
        let instances = cx.buffer("instances", Lifetime::Kept);
        let sampler = cx.sampler(SamplerDesc::LINEAR_CLAMP);
        let solid = cx.render_pipeline(
            Overlay,
            &quad(draw::VS_RECT, draw::FS_RECT),
            &[(draw::INSTANCES, Bind::Buffer(instances))],
        )?;
        let image = cx.render_pipeline(
            Overlay,
            &quad(draw::VS_RECT, draw::FS_IMAGE),
            &[
                (draw::INSTANCES, Bind::Buffer(instances)),
                (draw::SHEET, Bind::Asset),
                (draw::SHEET_SAMPLER, Bind::Sampler(sampler)),
            ],
        )?;
        Ok(Self {
            solid,
            image,
            instances,
            drawn: Vec::new(),
        })
    }

    fn prepare(&mut self, state: SystemParamItem<'_, '_, Self::Reads>, cx: &mut Prepare<'_, Self>) {
        self.drawn.clear();
        if state.rects.is_empty() {
            return;
        }
        cx.grow_buffer(
            self.instances,
            std::mem::size_of_val(state.rects.as_slice()) as u64,
        );
        cx.write(self.instances, 0, &state.rects);
        for (texture, range) in &state.runs {
            let filled = (*texture).map(|sheet| cx.fill(&self.image, &[sheet]));
            self.drawn.push((filled, range.clone()));
        }
        cx.want_overlay();
    }
}

impl UiGpu {
    pub(crate) fn draw_rects(&self, _: &(), draw: &mut Draw<'_, Self, Overlay>) -> u32 {
        for (filled, range) in &self.drawn {
            draw.draw(
                filled.as_ref().unwrap_or(&self.solid),
                Shape::Pulled {
                    vertices: 0..6,
                    instances: range.clone(),
                },
            );
        }
        self.drawn.len() as u32
    }
}
