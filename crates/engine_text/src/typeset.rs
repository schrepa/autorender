use std::any::Any;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use bevy_ecs::prelude::ResMut;
use engine_core::{Error, ErrorCode, PrimaryView};
use engine_places_rendering::text::{FontId, Glyph, Text, TextRun, TextSize, Typeset};
use glam::UVec2;

use crate::atlas::{Atlas, Slot, Upload};
use crate::fonts::{Fonts, Layout, Shapes};
use crate::{Label, TextSettings, TextStats};

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
    shapes: Shapes,
    atlas: Atlas,
    placed_runs: u32,
    placed_glyphs: u32,
    cut: u32,
    ratio: f32,
    limit: u32,
    enabled: bool,
    frame: u64,
    stats: TextStats,
}

impl Default for State {
    fn default() -> Self {
        let settings = TextSettings::default();
        Self {
            fonts: None,
            shapes: Shapes::default(),
            atlas: Atlas::default(),
            placed_runs: 0,
            placed_glyphs: 0,
            cut: 0,
            ratio: 1.0,
            limit: settings.max_glyphs,
            enabled: settings.enabled,
            frame: 0,
            stats: TextStats::default(),
        }
    }
}

impl State {
    pub(crate) fn sync(&mut self, settings: &TextSettings, view: &PrimaryView) {
        self.ratio = view.pixel_ratio();
        self.limit = settings.max_glyphs;
        self.enabled = settings.enabled;
        self.atlas.resize(settings.atlas_size);
    }

    pub(crate) fn advance(&mut self) {
        self.shapes.sweep(self.frame);
        self.frame += 1;
    }

    pub(crate) fn stats(&self) -> TextStats {
        self.stats.clone()
    }

    pub(crate) fn atlas_side(&self) -> u32 {
        self.atlas.side()
    }

    pub(crate) fn take_uploads(&mut self) -> Vec<Upload> {
        self.atlas.take_uploads()
    }

    pub(crate) fn refill_atlas(&mut self) -> Vec<Upload> {
        if let Some(fonts) = self.fonts.as_mut() {
            self.atlas.refill(fonts);
        }
        self.atlas.take_uploads()
    }

    pub(crate) fn frame<'l>(
        &mut self,
        labels: impl Iterator<Item = &'l Label>,
        settings: &TextSettings,
        view: &PrimaryView,
        staged: &mut Vec<Glyph>,
    ) -> u32 {
        self.sync(settings, view);
        let mut cut = std::mem::take(&mut self.cut);
        let mut runs = std::mem::take(&mut self.placed_runs);
        let placed = std::mem::take(&mut self.placed_glyphs);
        staged.clear();
        if self.enabled {
            let display = view.display_size().max(UVec2::ONE);
            let limit = self.limit as usize;
            for label in labels.filter(|label| label.visible && !label.text.is_empty()) {
                let placing = self.label_placing(label);
                self.place(&placing, display, limit, staged, &mut cut);
                runs += 1;
            }
        }
        let resets = u32::from(self.atlas.settle());
        let shown = staged.len() as u32 + placed;
        self.stats = TextStats {
            runs,
            glyphs_drawn: shown,
            glyphs_cut: cut,
            atlas_resets: self.stats.atlas_resets + resets,
            fonts: self.fonts.as_ref().map_or(0, Fonts::loaded),
            failed_fonts: self.fonts.as_ref().map_or(0, Fonts::failed),
        };
        shown
    }

    fn place_run(&mut self, run: &TextRun, display: UVec2, glyphs: &mut Vec<Glyph>) {
        if !self.enabled || run.text.is_empty() {
            return;
        }
        let placing = self.run_placing(run);
        let before = glyphs.len();
        let mut cut = 0;
        self.place(&placing, display, usize::MAX, glyphs, &mut cut);
        self.cut = self.cut.saturating_add(cut);
        self.placed_runs = self.placed_runs.saturating_add(1);
        self.placed_glyphs = self
            .placed_glyphs
            .saturating_add((glyphs.len() - before) as u32);
    }

    pub(crate) fn load_font(&mut self, bytes: Arc<[u8]>) -> Option<FontId> {
        self.fonts.get_or_insert_with(Fonts::new).load(bytes)
    }

    fn label_placing<'l>(&self, label: &'l Label) -> Placing<'l> {
        let font = label.face.map_or(0, |font| font.0);
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
        limit: usize,
        staged: &mut Vec<Glyph>,
        cut: &mut u32,
    ) {
        let ratio = self.ratio;
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
            staged.extend(glyph(slot, at, bounds, run.color, screen));
        }
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

fn glyph(
    slot: Slot,
    at: [i32; 2],
    bounds: [i32; 4],
    color: [f32; 4],
    screen: glam::Vec2,
) -> Option<Glyph> {
    let left = at[0].max(bounds[0]);
    let top = at[1].max(bounds[1]);
    let right = (at[0] + slot.width as i32).min(bounds[2]);
    let bottom = (at[1] + slot.height as i32).min(bounds[3]);
    if left >= right || top >= bottom {
        return None;
    }
    let across = |x: i32| x as f32 / screen.x * 2.0 - 1.0;
    let down = |y: i32| 1.0 - y as f32 / screen.y * 2.0;
    Some(Glyph {
        place: [across(left), down(top), across(right), down(bottom)],
        texel: [
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

    fn place(&self, run: &TextRun, display: [u32; 2], glyphs: &mut Vec<Glyph>) {
        let display = UVec2::from_array(display).max(UVec2::ONE);
        self.lock().place_run(run, display, glyphs);
    }
}
