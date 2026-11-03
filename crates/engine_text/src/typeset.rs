use std::any::Any;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use bevy_ecs::prelude::ResMut;
use engine_core::{Error, ErrorCode, View};
use engine_places::text::{FontId, Text, TextRun, TextSize, Typeset};
use glam::UVec2;

use crate::atlas::{Atlas, Slot, Upload};
use crate::fonts::{Fonts, Layout, Shapes};
use crate::{GlyphInstance, Label, TextSettings, TextStats};

struct Placing<'a> {
    text: &'a str,
    layout: Layout,
    color: [f32; 4],
    anchor: [f32; 2],
    pivot: [f32; 2],
    offset: [f32; 2],
    clip: Option<[f32; 4]>,
}

pub(crate) struct State {
    fonts: Option<Fonts>,
    scene_fonts: Vec<(Arc<[u8]>, Option<FontId>)>,
    shapes: Shapes,
    atlas: Atlas,
    queue: Vec<TextRun>,
    queued: u32,
    shown: Vec<TextRun>,
    cut: u32,
    ratio: f32,
    limit: u32,
    frame: u64,
    stats: TextStats,
}

impl Default for State {
    fn default() -> Self {
        Self {
            fonts: None,
            scene_fonts: Vec::new(),
            shapes: Shapes::default(),
            atlas: Atlas::default(),
            queue: Vec::new(),
            queued: 0,
            shown: Vec::new(),
            cut: 0,
            ratio: 1.0,
            limit: TextSettings::default().max_glyphs,
            frame: 0,
            stats: TextStats::default(),
        }
    }
}

impl State {
    pub(crate) fn sync(&mut self, settings: &TextSettings, view: &View) {
        self.ratio = view.pixel_ratio();
        self.limit = settings.max_glyphs;
        self.atlas.resize(settings.atlas_size);
    }

    pub(crate) fn stats(&self) -> TextStats {
        self.stats.clone()
    }

    pub(crate) fn runs(&self) -> Vec<TextRun> {
        self.queue.iter().chain(&self.shown).cloned().collect()
    }

    pub(crate) fn forget_atlas(&mut self) {
        self.atlas.forget();
    }

    pub(crate) fn atlas_side(&self) -> u32 {
        self.atlas.side()
    }

    pub(crate) fn take_uploads(&mut self) -> Vec<Upload> {
        self.atlas.take_uploads()
    }

    pub(crate) fn frame<'l>(
        &mut self,
        labels: impl Iterator<Item = &'l Label>,
        settings: &TextSettings,
        view: &View,
        staged: &mut Vec<GlyphInstance>,
    ) {
        self.sync(settings, view);
        self.frame += 1;
        std::mem::swap(&mut self.queue, &mut self.shown);
        self.queue.clear();
        self.queued = 0;
        let mut cut = std::mem::take(&mut self.cut);
        let mut runs = 0;
        staged.clear();
        if settings.enabled {
            let display = view.display_size().max(UVec2::ONE);
            for label in labels.filter(|label| label.visible && !label.text.is_empty()) {
                let placing = self.label_placing(label);
                self.place(&placing, display, staged, &mut cut);
                runs += 1;
            }
            let shown = std::mem::take(&mut self.shown);
            for run in shown.iter().filter(|run| !run.text.is_empty()) {
                let placing = self.run_placing(run);
                self.place(&placing, display, staged, &mut cut);
                runs += 1;
            }
            self.shown = shown;
        }
        let resets = u32::from(self.atlas.settle());
        self.shapes.sweep(self.frame);
        self.stats = TextStats {
            runs,
            glyphs_drawn: staged.len() as u32,
            glyphs_cut: cut,
            atlas_resets: self.stats.atlas_resets + resets,
            fonts: self.fonts.as_ref().map_or(0, Fonts::loaded),
            failed_fonts: self.fonts.as_ref().map_or(0, Fonts::failed),
        };
    }

    fn label_font(&mut self, label: &Label) -> u32 {
        let Some(bytes) = &label.bytes else {
            return 0;
        };
        let known = self
            .scene_fonts
            .iter()
            .find(|(held, _)| Arc::ptr_eq(held, bytes));
        if let Some((_, font)) = known {
            return font.map_or(0, |font| font.0);
        }
        let font = self
            .fonts
            .get_or_insert_with(Fonts::new)
            .load(Arc::clone(bytes));
        self.scene_fonts.push((Arc::clone(bytes), font));
        font.map_or(0, |font| font.0)
    }

    fn label_placing<'l>(&mut self, label: &'l Label) -> Placing<'l> {
        let font = self.label_font(label);
        Placing {
            text: &label.text,
            layout: Layout::of(self.ratio, font, label.size, label.max_width, label.align),
            color: label.color,
            anchor: label.anchor,
            pivot: label.pivot,
            offset: label.offset,
            clip: None,
        }
    }

    fn run_layout(&self, run: &TextRun) -> Layout {
        let font = run.font.map_or(0, |font| font.0);
        Layout::of(self.ratio, font, run.size, run.max_width, run.align)
    }

    fn run_placing<'r>(&self, run: &'r TextRun) -> Placing<'r> {
        Placing {
            text: &run.text,
            layout: self.run_layout(run),
            color: run.color,
            anchor: [0.0; 2],
            pivot: [0.0; 2],
            offset: run.position,
            clip: run.clip,
        }
    }

    fn place(
        &mut self,
        run: &Placing<'_>,
        display: UVec2,
        staged: &mut Vec<GlyphInstance>,
        cut: &mut u32,
    ) {
        let ratio = self.ratio;
        let limit = self.limit as usize;
        let frame = self.frame;
        let Self {
            fonts,
            shapes,
            atlas,
            ..
        } = self;
        let fonts = fonts.get_or_insert_with(Fonts::new);
        let shaped = shapes.get(fonts, run.text, run.layout, frame);
        let screen = display.as_vec2();
        let axis = |at: usize, screen: f32, size: f32| {
            (run.anchor[at] * screen + run.offset[at] * ratio - run.pivot[at] * size).round() as i32
        };
        let origin = [
            axis(0, screen.x, shaped.width),
            axis(1, screen.y, shaped.height),
        ];
        let bounds = bounds_of(run.clip, ratio, display);
        for glyph in &shaped.glyphs {
            if staged.len() >= limit {
                *cut += 1;
                continue;
            }
            let Some(slot) = atlas.slot(glyph.key, fonts) else {
                *cut += 1;
                continue;
            };
            let at = [
                origin[0] + glyph.x + slot.left,
                origin[1] + glyph.y - slot.top,
            ];
            staged.extend(instance(slot, at, bounds, run.color, screen));
        }
    }

    fn enqueue(&mut self, run: TextRun) {
        let count = run.text.chars().count() as u32;
        let queued = self.queued.saturating_add(count);
        if queued > self.limit {
            self.cut = self.cut.saturating_add(count);
            return;
        }
        self.queued = queued;
        self.queue.push(run);
    }
}

fn bounds_of(clip: Option<[f32; 4]>, ratio: f32, display: UVec2) -> [i32; 4] {
    let whole = [0, 0, display.x as i32, display.y as i32];
    let Some([x, y, width, height]) = clip else {
        return whole;
    };
    [
        ((x * ratio).round() as i32).max(whole[0]),
        ((y * ratio).round() as i32).max(whole[1]),
        (((x + width) * ratio).round() as i32).min(whole[2]),
        (((y + height) * ratio).round() as i32).min(whole[3]),
    ]
}

fn instance(
    slot: Slot,
    at: [i32; 2],
    bounds: [i32; 4],
    color: [f32; 4],
    screen: glam::Vec2,
) -> Option<GlyphInstance> {
    let left = at[0].max(bounds[0]);
    let top = at[1].max(bounds[1]);
    let right = (at[0] + slot.width as i32).min(bounds[2]);
    let bottom = (at[1] + slot.height as i32).min(bounds[3]);
    if left >= right || top >= bottom {
        return None;
    }
    let across = |x: i32| x as f32 / screen.x * 2.0 - 1.0;
    let down = |y: i32| 1.0 - y as f32 / screen.y * 2.0;
    Some(GlyphInstance {
        place: [across(left), down(top), across(right), down(bottom)],
        rect: [
            (slot.x as i32 + left - at[0]) as f32,
            (slot.y as i32 + top - at[1]) as f32,
            (right - left) as f32,
            (bottom - top) as f32,
        ],
        color,
    })
}

#[derive(Default)]
pub(crate) struct Typesetter {
    state: Mutex<State>,
}

impl Typesetter {
    pub(crate) fn of(text: &Text) -> Option<&Self> {
        (&*text.0 as &dyn Any).downcast_ref()
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

pub(crate) fn ensure(text: &mut ResMut<'_, Text>) {
    if Typesetter::of(text).is_none() {
        text.0 = Box::new(Typesetter::default());
    }
}

impl Typeset for Typesetter {
    fn register_font(&self, bytes: &[u8]) -> Result<FontId, Error> {
        self.lock()
            .fonts
            .get_or_insert_with(Fonts::new)
            .load(Arc::from(bytes))
            .ok_or_else(|| Error::new(ErrorCode::InvalidFont, "$.font").got(bytes.len()))
    }

    fn measure(&self, run: &TextRun) -> TextSize {
        let mut held = self.lock();
        let state = &mut *held;
        let layout = state.run_layout(run);
        let fonts = state.fonts.get_or_insert_with(Fonts::new);
        let shaped = state.shapes.get(fonts, &run.text, layout, state.frame);
        TextSize {
            width: shaped.width / state.ratio,
            height: shaped.height / state.ratio,
        }
    }

    fn queue(&self, run: TextRun) {
        self.lock().enqueue(run);
    }
}
