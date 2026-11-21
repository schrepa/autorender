engine::brick!(dock);

mod font;

use engine::math;
use engine::prelude::*;
use engine_core::{Error, GpuStruct, PassViews, TargetName, ViewFacts, Views, gpu_struct};
use engine_render::{Access, Encode, Own, PerBatch};
use engine_touch::TouchState;

use crate::state::{CamMode, DensityPreset, TimePreset, VerdantState, WindPreset};

gpu_struct! {
    pub struct UiHeader {
        pub screen: [f32; 4],
    }
}

gpu_struct! {
    pub struct UiQuad {
        pub rect: [f32; 4],
        pub color: [f32; 4],
        pub border: [f32; 4],
        pub params: [f32; 4],
    }
}

const GENERATED: &[&str] = &[<UiHeader as GpuStruct>::WGSL, <UiQuad as GpuStruct>::WGSL];

#[derive(Clone, Copy)]
enum Action {
    Time(TimePreset),
    Wind(WindPreset),
    Density(DensityPreset),
    Cam(CamMode),
}

#[derive(Clone, Copy)]
struct PillItem {
    label: &'static str,
    action: Action,
    active: bool,
    rect: [f32; 4],
    group: u8,
}

#[derive(Clone, Copy)]
struct RowMetrics {
    center_x: f32,
    y: f32,
    h: f32,
    char_w: f32,
    pad_x: f32,
    gap: f32,
    group_gap: f32,
}

#[derive(Clone, Copy)]
struct TextMetrics {
    glyph_w: f32,
    glyph_h: f32,
    step: f32,
}

#[derive(Resource, Default)]
struct DockDraw {
    quads: Vec<UiQuad>,
    screen: Vec2,
    pressed_idx: Option<usize>,
}

pub struct DockBrick;

impl Brick for DockBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .generated(GENERATED)
            .resource::<DockDraw>()
            .part::<DockGpu>()
            .systems(Schedule::BeforeTicks, update_dock);
    }
}

fn build_items(state: &VerdantState) -> Vec<PillItem> {
    let mut items = Vec::with_capacity(12);
    for p in TimePreset::ALL {
        items.push(PillItem { label: p.label(), action: Action::Time(p), active: state.time_preset == p, rect: [0.0; 4], group: 0 });
    }
    for w in WindPreset::ALL {
        items.push(PillItem { label: w.label(), action: Action::Wind(w), active: state.wind_preset == w, rect: [0.0; 4], group: 1 });
    }
    for d in DensityPreset::ALL {
        items.push(PillItem { label: d.label(), action: Action::Density(d), active: state.density == d, rect: [0.0; 4], group: 2 });
    }
    for c in CamMode::ALL {
        items.push(PillItem { label: c.label(), action: Action::Cam(c), active: state.cam_mode == c, rect: [0.0; 4], group: 3 });
    }
    items
}

struct DockLayout {
    items: Vec<PillItem>,
    trays: Vec<[f32; 4]>,
    dividers: Vec<[f32; 4]>,
    text: TextMetrics,
    compact: bool,
}

fn layout_row(
    items: &mut [PillItem],
    range: std::ops::Range<usize>,
    m: RowMetrics,
    max_w: f32,
    dpr: f32,
    dividers: &mut Vec<[f32; 4]>,
) -> [f32; 4] {
    let mut widths = Vec::with_capacity(range.len());
    let mut total = 0.0_f32;
    let mut prev_group = None;
    for item in &items[range.clone()] {
        let w = math::max(item.label.chars().count() as f32 * m.char_w + m.pad_x * 2.0, m.h * 0.78);
        if let Some(pg) = prev_group {
            total += if pg == item.group { m.gap } else { m.group_gap };
        }
        prev_group = Some(item.group);
        widths.push(w);
        total += w;
    }
    let pad_outer = 5.5 * (m.h / 36.0);
    let shrink = if total + pad_outer * 2.0 > max_w && total > 1.0 {
        (max_w - pad_outer * 2.0) / total
    } else {
        1.0
    };
    let eff_total = total * shrink;
    let eff_gap = m.gap * shrink;
    let eff_ggap = m.group_gap * shrink;
    let mut x = m.center_x - eff_total * 0.5;
    let mut prev = None;
    let div_w = math::max(1.2, 1.3 * dpr);
    let div_h = m.h * 0.48;
    for (idx, item) in items[range].iter_mut().enumerate() {
        if let Some(pg) = prev {
            if pg == item.group {
                x += eff_gap;
            } else {
                dividers.push([x + (eff_ggap - div_w) * 0.5, m.y + (m.h - div_h) * 0.5, div_w, div_h]);
                x += eff_ggap;
            }
        }
        prev = Some(item.group);
        let w = widths[idx] * shrink;
        item.rect = [x, m.y, w, m.h];
        x += w;
    }
    [m.center_x - eff_total * 0.5 - pad_outer, m.y - pad_outer, eff_total + pad_outer * 2.0, m.h + pad_outer * 2.0]
}

fn compute_layout(state: &VerdantState, safe: [f32; 4], dpr: f32, logical_w: f32) -> DockLayout {
    let mut items = build_items(state);
    let mut trays = Vec::new();
    let mut dividers = Vec::new();
    let compact = logical_w < 780.0;
    let cx = safe[0] + safe[2] * 0.5;
    let max_row_w = safe[2] - 18.0 * dpr;
    let text = if compact {
        let s = math::clamp(logical_w / 390.0, 0.85, 1.05) * dpr;
        let h = 44.0 * s;
        let step = 6.8 * s;
        let bottom = safe[1] + safe[3] - 14.0 * s;
        let y2 = bottom - h;
        let y1 = y2 - h - 14.0 * s;
        trays.push(layout_row(&mut items, 0..7, RowMetrics { center_x: cx, y: y1, h, char_w: step, pad_x: 6.8 * s, gap: 3.0 * s, group_gap: 10.0 * s }, max_row_w, dpr, &mut dividers));
        trays.push(layout_row(&mut items, 7..12, RowMetrics { center_x: cx, y: y2, h, char_w: step, pad_x: 10.5 * s, gap: 4.5 * s, group_gap: 13.0 * s }, max_row_w, dpr, &mut dividers));
        TextMetrics { glyph_w: 5.8 * s, glyph_h: 9.8 * s, step }
    } else {
        let s = dpr;
        let h = 36.0 * s;
        let step = 9.0 * s;
        let y = safe[1] + safe[3] - 18.0 * s - h;
        trays.push(layout_row(&mut items, 0..12, RowMetrics { center_x: cx, y, h, char_w: step, pad_x: 12.0 * s, gap: 4.0 * s, group_gap: 16.0 * s }, max_row_w, dpr, &mut dividers));
        TextMetrics { glyph_w: 7.5 * s, glyph_h: 11.2 * s, step }
    };
    DockLayout { items, trays, dividers, text, compact }
}

fn push_pill(quads: &mut Vec<UiQuad>, rect: [f32; 4], radius: f32, color: [f32; 4], border: [f32; 4], bw: f32) {
    quads.push(UiQuad { rect, color, border, params: [radius, bw, 0.0, 0.0] });
}

fn push_text(quads: &mut Vec<UiQuad>, text: &str, center: Vec2, metrics: TextMetrics, color: [f32; 4]) {
    let count = text.chars().count();
    if count == 0 {
        return;
    }
    let total_w = (count.saturating_sub(1) as f32) * metrics.step + metrics.glyph_w;
    let start_x = center.x - total_w * 0.5;
    let top_y = center.y - metrics.glyph_h * 0.5;
    for (i, ch) in text.chars().enumerate() {
        if ch == ' ' {
            continue;
        }
        let [lo18, hi17] = font::pack_glyph(ch);
        quads.push(UiQuad {
            rect: [start_x + i as f32 * metrics.step, top_y, metrics.glyph_w, metrics.glyph_h],
            color,
            border: [0.0; 4],
            params: [lo18, hi17, 1.0, 0.0],
        });
    }
}

fn hit_test(rect: [f32; 4], pt: Vec2) -> bool {
    pt.x >= rect[0] && pt.x <= rect[0] + rect[2] && pt.y >= rect[1] && pt.y <= rect[1] + rect[3]
}

fn apply_action(state: &mut VerdantState, action: Action) {
    match action {
        Action::Time(t) => state.time_preset = t,
        Action::Wind(w) => state.wind_preset = w,
        Action::Density(d) => state.density = d,
        Action::Cam(c) => state.cam_mode = c,
    }
}

fn paint_perf_and_stick(
    quads: &mut Vec<UiQuad>,
    state: &VerdantState,
    touch: Option<&TouchState>,
    safe: [f32; 4],
    dpr: f32,
    compact: bool,
) {
    let fps_int = (state.fps_smoothed + 0.5) as u32;
    let perf_text = format!("{} \u{00B7} {} FPS", state.density.short_label(), fps_int);
    let ps = if compact { 0.90 * dpr } else { dpr };
    let metrics = TextMetrics { glyph_w: 7.0 * ps, glyph_h: 10.5 * ps, step: 8.8 * ps };
    let p_h = 28.0 * ps;
    let p_w = perf_text.chars().count() as f32 * metrics.step + 22.0 * ps;
    let p_rect = [safe[0] + safe[2] - 16.0 * ps - p_w, safe[1] + 14.0 * ps, p_w, p_h];
    push_pill(quads, p_rect, p_h * 0.5, [0.045, 0.056, 0.082, 0.64], [1.0, 0.96, 0.88, 0.20], math::max(1.0, dpr));
    push_text(quads, &perf_text, Vec2::new(p_rect[0] + p_rect[2] * 0.5, p_rect[1] + p_rect[3] * 0.5), metrics, [0.96, 0.94, 0.88, 0.94]);

    if let Some(stick) = touch.and_then(TouchState::stick_visual) {
        let r = stick.radius_px;
        let kr = stick.knob_radius_px;
        push_pill(
            quads,
            [stick.center_px.x - r, stick.center_px.y - r, r * 2.0, r * 2.0],
            r,
            [0.06, 0.08, 0.11, 0.28],
            [1.0, 0.94, 0.82, if stick.active { 0.55 } else { 0.28 }],
            math::max(1.5, dpr * 1.5),
        );
        push_pill(
            quads,
            [stick.knob_px.x - kr, stick.knob_px.y - kr, kr * 2.0, kr * 2.0],
            kr,
            [0.96, 0.78, 0.42, if stick.active { 0.82 } else { 0.45 }],
            [1.0, 0.96, 0.88, 0.4],
            math::max(1.0, dpr),
        );
    }
}

fn update_dock(
    views: Res<Views>,
    pointer: Res<Pointer>,
    touch: Option<Res<TouchState>>,
    mut state: ResMut<VerdantState>,
    mut dock: ResMut<DockDraw>,
) {
    let frame = views.frame();
    let sw = frame.display_size.x as f32;
    let sh = frame.display_size.y as f32;
    dock.screen = Vec2::new(sw, sh);
    dock.quads.clear();

    if state.hud_hidden || sw < 10.0 || sh < 10.0 {
        state.ui_hovered = false;
        state.ui_captured_pointer = false;
        return;
    }

    let dpr = frame.effective_pixel_ratio();
    let safe = frame.safe_rect_pixels();
    let logical_w = sw / dpr;
    let mut layout = compute_layout(&state, safe, dpr, logical_w);

    let mut hovered_idx = None;
    let mut over_tray = false;
    if let Some(pt) = pointer.position_pixels {
        for &tray in &layout.trays {
            if hit_test(tray, pt) {
                over_tray = true;
            }
        }
        for (i, item) in layout.items.iter().enumerate() {
            if hit_test(item.rect, pt) {
                hovered_idx = Some(i);
                over_tray = true;
            }
        }
    }

    if pointer.pressed {
        if let Some(idx) = hovered_idx {
            dock.pressed_idx = Some(idx);
            state.ui_captured_pointer = true;
        } else if over_tray {
            state.ui_captured_pointer = true;
        }
    }
    if pointer.released {
        if let Some(idx) = dock.pressed_idx.take()
            && hovered_idx == Some(idx)
        {
            apply_action(&mut state, layout.items[idx].action);
            layout = compute_layout(&state, safe, dpr, logical_w);
        }
        state.ui_captured_pointer = false;
    }
    if !pointer.down && !pointer.released {
        dock.pressed_idx = None;
        state.ui_captured_pointer = false;
    }
    state.ui_hovered = over_tray;

    for &tray in &layout.trays {
        push_pill(&mut dock.quads, tray, tray[3] * 0.48, [0.036, 0.046, 0.070, 0.78], [1.0, 0.96, 0.88, 0.26], math::max(1.0, dpr));
    }
    for &div in &layout.dividers {
        push_pill(&mut dock.quads, div, div[2] * 0.5, [1.0, 0.96, 0.88, 0.24], [0.0; 4], 0.0);
    }
    for (i, item) in layout.items.iter().enumerate() {
        let hovered = hovered_idx == Some(i) && pointer.hovering;
        let fill = if item.active {
            [0.96, 0.75, 0.36, 0.96]
        } else if hovered {
            [1.0, 0.96, 0.90, 0.18]
        } else {
            [1.0, 1.0, 1.0, 0.05]
        };
        let border = if item.active { [1.0, 0.92, 0.74, 0.52] } else { [1.0, 1.0, 1.0, 0.0] };
        let pill_r = if layout.compact { math::min(item.rect[3] * 0.34, item.rect[2] * 0.36) } else { item.rect[3] * 0.44 };
        push_pill(&mut dock.quads, item.rect, pill_r, fill, border, math::max(1.0, dpr));
        let text_col = if item.active {
            [0.07, 0.05, 0.03, 1.0]
        } else if hovered {
            [1.0, 0.98, 0.94, 1.0]
        } else {
            [0.97, 0.97, 0.99, 0.94]
        };
        push_text(&mut dock.quads, item.label, Vec2::new(item.rect[0] + item.rect[2] * 0.5, item.rect[1] + item.rect[3] * 0.5), layout.text, text_col);
    }
    paint_perf_and_stick(&mut dock.quads, &state, touch.as_deref(), safe, dpr, layout.compact);
}

struct DockGpu {
    pass: engine_render::Pass<Self>,
    pipeline: Pipeline<Self, Own>,
    header: Buffer<Self>,
    quads: Buffer<Self>,
    count: u32,
}

impl GpuPart for DockGpu {
    type Reads = Res<'static, DockDraw>;
    const NAME: &'static str = "verdant_dock";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::dock::ui;
        let header = cx.buffer("verdant_ui_header", Lifetime::Kept);
        let quads = cx.buffer("verdant_ui_quads", Lifetime::Kept);
        let pass = cx
            .pass("verdant_dock")
            .label("verdant/dock")
            .span("dock")
            .views(PassViews::Main)
            .target(TargetName::CANVAS, Access::Load)
            .id();
        let pipeline = cx.render_pipeline(
            pass,
            &RenderPipeline {
                vertex: ui::VS_UI,
                fragment: Some(ui::FS_UI),
                vertices: &[],
                primitive: wgpu::PrimitiveState::default(),
                depth: DepthUse::None,
                stencil: wgpu::StencilState::default(),
                samples: 1,
                targets: &[(Some(wgpu::BlendState::ALPHA_BLENDING), wgpu::ColorWrites::ALL)],
                constants: &[],
                per_batch: PerBatch::NONE,
            },
            &[
                (ui::UI_HEADER, Bind::Buffer(header)),
                (ui::UI_QUADS, Bind::Buffer(quads)),
            ],
        )?;
        Ok(Self {
            pass,
            pipeline,
            header,
            quads,
            count: 0,
        })
    }

    fn prepare(&mut self, dock: Res<'_, DockDraw>, cx: &mut Prepare<'_, Self>) {
        self.count = dock.quads.len() as u32;
        let _ = cx.allocate_buffer(self.header, size_of::<UiHeader>() as u64);
        cx.write(self.header, 0, &[UiHeader { screen: [dock.screen.x, dock.screen.y, 0.0, 0.0] }]);
        let wanted = (dock.quads.len().max(1) * size_of::<UiQuad>()) as u64;
        let _ = cx.allocate_buffer(self.quads, wanted.next_power_of_two());
        if !dock.quads.is_empty() {
            cx.write(self.quads, 0, &dock.quads);
        }
    }

    fn view(&mut self, _view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        if self.count > 0 {
            cx.run(self.pass);
        }
    }

    fn encode(&mut self, _pass: engine_render::Pass<Self>, cx: &mut Encode<'_, Self>) {
        cx.render().draw(
            &self.pipeline,
            Shape::Pulled {
                vertices: 0..6,
                instances: 0..self.count,
                key: None,
            },
        );
    }
}
