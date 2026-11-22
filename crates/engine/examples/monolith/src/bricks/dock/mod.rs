engine::brick!(dock);

mod font;

use brick_nanite::{NaniteConfig, NaniteTelemetry, ViewMode};
use engine::math;
use engine::prelude::*;
use engine_core::{Error, GpuStruct, PassViews, TargetName, ViewFacts, Views, gpu_struct};
use engine_render::{Access, Encode, Own, PerBatch};
use engine_touch::TouchState;

use crate::state::{CamBookmark, MonolithState};

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
    View(ViewMode),
    Cam(CamBookmark),
    ToggleFreeze,
    CycleHeap,
    CycleLod,
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

fn build_items(state: &MonolithState, cfg: &NaniteConfig) -> Vec<PillItem> {
    let mut items = Vec::with_capacity(13);
    for v in ViewMode::ALL {
        items.push(PillItem {
            label: v.label(),
            action: Action::View(v),
            active: cfg.view_mode == v,
            rect: [0.0; 4],
            group: 0,
        });
    }
    for c in CamBookmark::ALL {
        items.push(PillItem {
            label: c.label(),
            action: Action::Cam(c),
            active: state.bookmark == c,
            rect: [0.0; 4],
            group: 1,
        });
    }
    items.push(PillItem {
        label: "FREEZE HI-Z",
        action: Action::ToggleFreeze,
        active: cfg.freeze_culling,
        rect: [0.0; 4],
        group: 2,
    });
    items.push(PillItem {
        label: cfg.heap_budget.label(),
        action: Action::CycleHeap,
        active: false,
        rect: [0.0; 4],
        group: 2,
    });
    items.push(PillItem {
        label: cfg.lod_preset.label(),
        action: Action::CycleLod,
        active: false,
        rect: [0.0; 4],
        group: 2,
    });
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
    [
        m.center_x - eff_total * 0.5 - pad_outer,
        m.y - pad_outer,
        eff_total + pad_outer * 2.0,
        m.h + pad_outer * 2.0,
    ]
}

fn compute_layout(
    state: &MonolithState,
    cfg: &NaniteConfig,
    safe: [f32; 4],
    dpr: f32,
    logical_w: f32,
) -> DockLayout {
    let mut items = build_items(state, cfg);
    let mut trays = Vec::new();
    let mut dividers = Vec::new();
    let compact = logical_w < 980.0;
    let cx = safe[0] + safe[2] * 0.5;
    let max_row_w = safe[2] - 18.0 * dpr;
    let text = if compact {
        let s = math::clamp(logical_w / 460.0, 0.78, 1.0) * dpr;
        let h = 38.0 * s;
        let step = 6.6 * s;
        let bottom = safe[1] + safe[3] - 14.0 * s;
        let y2 = bottom - h;
        let y1 = y2 - h - 12.0 * s;
        trays.push(layout_row(
            &mut items,
            0..6,
            RowMetrics { center_x: cx, y: y1, h, char_w: step, pad_x: 7.5 * s, gap: 3.0 * s, group_gap: 10.0 * s },
            max_row_w,
            dpr,
            &mut dividers,
        ));
        trays.push(layout_row(
            &mut items,
            6..13,
            RowMetrics { center_x: cx, y: y2, h, char_w: step, pad_x: 7.5 * s, gap: 3.0 * s, group_gap: 10.0 * s },
            max_row_w,
            dpr,
            &mut dividers,
        ));
        TextMetrics { glyph_w: 5.4 * s, glyph_h: 9.2 * s, step }
    } else {
        let s = dpr;
        let h = 34.0 * s;
        let step = 7.6 * s;
        let y = safe[1] + safe[3] - 16.0 * s - h;
        trays.push(layout_row(
            &mut items,
            0..13,
            RowMetrics { center_x: cx, y, h, char_w: step, pad_x: 9.5 * s, gap: 3.5 * s, group_gap: 13.0 * s },
            max_row_w,
            dpr,
            &mut dividers,
        ));
        TextMetrics { glyph_w: 6.2 * s, glyph_h: 9.8 * s, step }
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

fn apply_action(state: &mut MonolithState, cfg: &mut NaniteConfig, action: Action) {
    match action {
        Action::View(v) => cfg.view_mode = v,
        Action::Cam(c) => {
            state.bookmark = c;
            state.requested_bookmark = Some(c);
        }
        Action::ToggleFreeze => cfg.freeze_culling = !cfg.freeze_culling,
        Action::CycleHeap => cfg.heap_budget = cfg.heap_budget.next(),
        Action::CycleLod => cfg.lod_preset = cfg.lod_preset.next(),
    }
}

#[derive(Clone, Copy)]
struct HudFrame {
    safe: [f32; 4],
    dpr: f32,
    compact: bool,
}

fn paint_telemetry_and_stick(
    quads: &mut Vec<UiQuad>,
    state: &MonolithState,
    cfg: &NaniteConfig,
    tel: &NaniteTelemetry,
    touch: Option<&TouchState>,
    hud: HudFrame,
) {
    let safe = hud.safe;
    let dpr = hud.dpr;
    let compact = hud.compact;
    let fps_int = (state.fps_smoothed + 0.5) as u32;
    let vis_k = (tel.visible_tris() + 500) / 1000;
    let total_m = (tel.total_virtual_tris as f64) / 1_000_000.0;
    let res_mb = (f64::from(tel.resident_pages) * 65_536.0) / (1024.0 * 1024.0);
    let frozen = if cfg.freeze_culling { " \u{00B7} HI-Z FROZEN" } else { "" };
    let line = format!(
        "{total_m:.1}M TRIS \u{00B7} VIS {vis_k}K \u{00B7} HW {}% / SW {}% \u{00B7} PAGES {}/{} ({res_mb:.1}MB) \u{00B7} {fps_int} FPS{frozen}",
        tel.hw_percent(),
        tel.sw_percent(),
        tel.resident_pages,
        tel.total_pages,
    );
    let ps = if compact { 0.78 * dpr } else { 0.94 * dpr };
    let max_w = safe[2] - 24.0 * dpr;
    let mut step = 7.4 * ps;
    let mut gw = 6.0 * ps;
    let chars = line.chars().count().max(1) as f32;
    let raw_w = chars * step + 24.0 * ps;
    if raw_w > max_w {
        let s = (max_w - 18.0 * ps) / (chars * step);
        step *= s;
        gw *= s;
    }
    let metrics = TextMetrics { glyph_w: gw, glyph_h: 9.8 * ps, step };
    let p_h = 28.0 * ps;
    let p_w = math::min(chars * step + 24.0 * ps, max_w);
    let p_rect = [safe[0] + (safe[2] - p_w) * 0.5, safe[1] + 12.0 * dpr, p_w, p_h];
    let border = if cfg.freeze_culling {
        [0.98, 0.72, 0.28, 0.72]
    } else {
        [1.0, 0.94, 0.84, 0.22]
    };
    push_pill(quads, p_rect, p_h * 0.5, [0.055, 0.048, 0.042, 0.72], border, math::max(1.0, dpr));
    push_text(
        quads,
        &line,
        Vec2::new(p_rect[0] + p_rect[2] * 0.5, p_rect[1] + p_rect[3] * 0.5),
        metrics,
        [0.97, 0.94, 0.88, 0.95],
    );

    if let Some(stick) = touch.and_then(TouchState::stick_visual) {
        let r = stick.radius_px;
        let kr = stick.knob_radius_px;
        push_pill(
            quads,
            [stick.center_px.x - r, stick.center_px.y - r, r * 2.0, r * 2.0],
            r,
            [0.06, 0.05, 0.04, 0.28],
            [1.0, 0.92, 0.78, if stick.active { 0.55 } else { 0.28 }],
            math::max(1.5, dpr * 1.5),
        );
        push_pill(
            quads,
            [stick.knob_px.x - kr, stick.knob_px.y - kr, kr * 2.0, kr * 2.0],
            kr,
            [0.95, 0.74, 0.36, if stick.active { 0.82 } else { 0.45 }],
            [1.0, 0.96, 0.88, 0.4],
            math::max(1.0, dpr),
        );
    }
}

#[derive(SystemParam)]
struct DockParams<'w> {
    views: Res<'w, Views>,
    pointer: Res<'w, Pointer>,
    touch: Option<Res<'w, TouchState>>,
    telemetry: Res<'w, NaniteTelemetry>,
    state: ResMut<'w, MonolithState>,
    cfg: ResMut<'w, NaniteConfig>,
    dock: ResMut<'w, DockDraw>,
}

fn update_dock(mut p: DockParams<'_>) {
    let frame = p.views.frame();
    let sw = frame.display_size.x as f32;
    let sh = frame.display_size.y as f32;
    p.dock.screen = Vec2::new(sw, sh);
    p.dock.quads.clear();

    if p.state.hud_hidden || sw < 10.0 || sh < 10.0 {
        p.state.ui_hovered = false;
        p.state.ui_captured_pointer = false;
        return;
    }

    let dpr = frame.effective_pixel_ratio();
    let safe = frame.safe_rect_pixels();
    let logical_w = sw / dpr;
    let mut layout = compute_layout(&p.state, &p.cfg, safe, dpr, logical_w);

    let mut hovered_idx = None;
    let mut over_tray = false;
    if let Some(pt) = p.pointer.position_pixels {
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

    if p.pointer.pressed {
        if let Some(idx) = hovered_idx {
            p.dock.pressed_idx = Some(idx);
            p.state.ui_captured_pointer = true;
        } else if over_tray {
            p.state.ui_captured_pointer = true;
        }
    }
    if p.pointer.released {
        if let Some(idx) = p.dock.pressed_idx.take()
            && hovered_idx == Some(idx)
        {
            apply_action(&mut p.state, &mut p.cfg, layout.items[idx].action);
            layout = compute_layout(&p.state, &p.cfg, safe, dpr, logical_w);
        }
        p.state.ui_captured_pointer = false;
    }
    if !p.pointer.down && !p.pointer.released {
        p.dock.pressed_idx = None;
        p.state.ui_captured_pointer = false;
    }
    p.state.ui_hovered = over_tray;

    for &tray in &layout.trays {
        push_pill(&mut p.dock.quads, tray, tray[3] * 0.48, [0.052, 0.045, 0.040, 0.78], [1.0, 0.94, 0.84, 0.24], math::max(1.0, dpr));
    }
    for &div in &layout.dividers {
        push_pill(&mut p.dock.quads, div, div[2] * 0.5, [1.0, 0.94, 0.84, 0.24], [0.0; 4], 0.0);
    }
    for (i, item) in layout.items.iter().enumerate() {
        let hovered = hovered_idx == Some(i) && p.pointer.hovering;
        let fill = if item.active {
            [0.95, 0.73, 0.34, 0.96]
        } else if hovered {
            [1.0, 0.95, 0.88, 0.18]
        } else {
            [1.0, 1.0, 1.0, 0.05]
        };
        let border = if item.active { [1.0, 0.92, 0.74, 0.52] } else { [1.0, 1.0, 1.0, 0.0] };
        let pill_r = if layout.compact {
            math::min(item.rect[3] * 0.34, item.rect[2] * 0.36)
        } else {
            item.rect[3] * 0.44
        };
        push_pill(&mut p.dock.quads, item.rect, pill_r, fill, border, math::max(1.0, dpr));
        let text_col = if item.active {
            [0.07, 0.05, 0.03, 1.0]
        } else if hovered {
            [1.0, 0.98, 0.94, 1.0]
        } else {
            [0.97, 0.96, 0.94, 0.94]
        };
        push_text(
            &mut p.dock.quads,
            item.label,
            Vec2::new(item.rect[0] + item.rect[2] * 0.5, item.rect[1] + item.rect[3] * 0.5),
            layout.text,
            text_col,
        );
    }
    paint_telemetry_and_stick(
        &mut p.dock.quads,
        &p.state,
        &p.cfg,
        &p.telemetry,
        p.touch.as_deref(),
        HudFrame {
            safe,
            dpr,
            compact: layout.compact,
        },
    );
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
    const NAME: &'static str = "monolith_dock";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::dock::ui;
        let header = cx.buffer("monolith_ui_header", Lifetime::Kept);
        let quads = cx.buffer("monolith_ui_quads", Lifetime::Kept);
        let pass = cx
            .pass("monolith_dock")
            .label("monolith/dock")
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
        cx.write(
            self.header,
            0,
            &[UiHeader {
                screen: [dock.screen.x, dock.screen.y, 0.0, 0.0],
            }],
        );
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
