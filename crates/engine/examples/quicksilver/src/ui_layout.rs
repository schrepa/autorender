use bytemuck::{Pod, Zeroable};
use engine::math;

use crate::font::glyph_uv;
use crate::state::{MaterialKind, QuicksilverState, ScalePreset};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct UiQuad {
    pub rect: [f32; 4],
    pub uv: [f32; 4],
    pub color: [f32; 4],
    pub params: [f32; 4],
}

pub const KIND_GLASS: f32 = 0.0;
pub const KIND_BUTTON: f32 = 1.0;
pub const KIND_SWATCH: f32 = 2.0;
pub const KIND_GLYPH: f32 = 3.0;
pub const KIND_DIVIDER: f32 = 4.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DockAction {
    Material(MaterialKind),
    Drop,
    Vortex,
    ZeroG,
    Scale(ScalePreset),
    OrbitMode,
}

#[derive(Clone, Copy, Debug)]
pub struct DockButton {
    pub rect: [f32; 4],
    pub action: DockAction,
    pub label: &'static str,
    pub active: bool,
    pub swatch: Option<[f32; 4]>,
}

pub struct DockLayout {
    pub panels: Vec<[f32; 5]>,
    pub dividers: Vec<[f32; 4]>,
    pub buttons: Vec<DockButton>,
    pub perf_rect: [f32; 4],
    pub scale: f32,
}

impl DockLayout {
    pub fn hit_test(&self, px: f32, py: f32) -> Option<usize> {
        self.buttons.iter().position(|b| {
            px >= b.rect[0]
                && px <= b.rect[0] + b.rect[2]
                && py >= b.rect[1]
                && py <= b.rect[1] + b.rect[3]
        })
    }

    pub fn covers_point(&self, px: f32, py: f32) -> bool {
        if self.hit_test(px, py).is_some() {
            return true;
        }
        self.panels.iter().any(|p| {
            px >= p[0] && px <= p[0] + p[2] && py >= p[1] && py <= p[1] + p[3]
        })
    }
}

pub fn compute_layout(state: &QuicksilverState) -> DockLayout {
    let w = math::max(state.viewport[0], 1.0);
    let h = math::max(state.viewport[1], 1.0);
    let dpr = math::clamp(state.pixel_ratio, 1.0, 3.0);
    let dp_w = w / dpr;
    let compact = dp_w < 780.0;
    if compact {
        layout_mobile(state, w, h, dpr)
    } else {
        layout_desktop(state, w, h, dpr)
    }
}

fn layout_desktop(state: &QuicksilverState, w: f32, h: f32, dpr: f32) -> DockLayout {
    let s = dpr;
    let pad = 6.0 * s;
    let gap = 4.0 * s;
    let div_gap = 10.0 * s;
    let bh = 38.0 * s;
    let dock_h = bh + pad * 2.0;

    let mat_widths = [86.0, 78.0, 68.0, 82.0];
    let phys_widths = [64.0, 76.0, 78.0];
    let scale_widths = [48.0, 56.0, 56.0];

    let total_mat: f32 = mat_widths.iter().sum::<f32>() * s + gap * 3.0;
    let total_phys: f32 = phys_widths.iter().sum::<f32>() * s + gap * 2.0;
    let total_scale: f32 = scale_widths.iter().sum::<f32>() * s + gap * 2.0;
    let dock_w = pad * 2.0 + total_mat + total_phys + total_scale + div_gap * 2.0;
    let dock_x = (w - dock_w) * 0.5;
    let dock_y = h - dock_h - 22.0 * s;

    let mut buttons = Vec::with_capacity(11);
    let mut dividers = Vec::with_capacity(2);
    let mut x = dock_x + pad;
    let by = dock_y + pad;

    for (mat, bw_dp) in MaterialKind::ALL.iter().zip(mat_widths) {
        let bw = bw_dp * s;
        buttons.push(DockButton {
            rect: [x, by, bw, bh],
            action: DockAction::Material(*mat),
            label: mat.label(),
            active: state.material == *mat,
            swatch: Some(mat.swatch()),
        });
        x += bw + gap;
    }

    x += div_gap - gap;
    dividers.push([x - div_gap * 0.5 - 0.5 * s, by + 7.0 * s, 1.0 * s, bh - 14.0 * s]);

    let phys = [
        (DockAction::Drop, "DROP", false, phys_widths[0]),
        (DockAction::Vortex, "VORTEX", state.vortex, phys_widths[1]),
        (DockAction::ZeroG, "ZERO-G", state.zero_g, phys_widths[2]),
    ];
    for (action, label, active, bw_dp) in phys {
        let bw = bw_dp * s;
        buttons.push(DockButton {
            rect: [x, by, bw, bh],
            action,
            label,
            active,
            swatch: None,
        });
        x += bw + gap;
    }

    x += div_gap - gap;
    dividers.push([x - div_gap * 0.5 - 0.5 * s, by + 7.0 * s, 1.0 * s, bh - 14.0 * s]);

    for (preset, bw_dp) in ScalePreset::ALL.iter().zip(scale_widths) {
        let bw = bw_dp * s;
        buttons.push(DockButton {
            rect: [x, by, bw, bh],
            action: DockAction::Scale(*preset),
            label: preset.label(),
            active: state.preset == *preset,
            swatch: None,
        });
        x += bw + gap;
    }

    let perf_w = 138.0 * s;
    let perf_h = 30.0 * s;
    let perf_rect = [w - perf_w - 18.0 * s, 16.0 * s, perf_w, perf_h];

    DockLayout {
        panels: vec![
            [dock_x, dock_y, dock_w, dock_h, dock_h * 0.5],
            [perf_rect[0], perf_rect[1], perf_rect[2], perf_rect[3], perf_h * 0.5],
        ],
        dividers,
        buttons,
        perf_rect,
        scale: s,
    }
}

fn layout_mobile(state: &QuicksilverState, w: f32, h: f32, dpr: f32) -> DockLayout {
    let s = dpr;
    let margin = 10.0 * s;
    let pad = 6.0 * s;
    let gap = 4.0 * s;
    let dock_w = math::min(w - margin * 2.0, 410.0 * s);
    let inner_w = dock_w - pad * 2.0;
    let bh = 44.0 * s;
    let row_gap = 6.0 * s;
    let dock_h = pad * 2.0 + bh * 2.0 + row_gap;
    let dock_x = (w - dock_w) * 0.5;
    let dock_y = h - dock_h - 18.0 * s;

    let mut buttons = Vec::with_capacity(11);

    let r1_y = dock_y + pad;
    let mat_w = (inner_w - gap * 3.0) * 0.25;
    for (idx, mat) in MaterialKind::ALL.iter().enumerate() {
        let bx = dock_x + pad + (idx as f32) * (mat_w + gap);
        buttons.push(DockButton {
            rect: [bx, r1_y, mat_w, bh],
            action: DockAction::Material(*mat),
            label: mat.label(),
            active: state.material == *mat,
            swatch: Some(mat.swatch()),
        });
    }

    let r2_y = r1_y + bh + row_gap;
    let weights = [1.05_f32, 1.24, 1.24, 0.88, 0.96, 0.96];
    let weight_sum: f32 = weights.iter().sum();
    let avail = inner_w - gap * 5.0;
    let row2 = [
        (DockAction::Drop, "DROP", false),
        (DockAction::Vortex, "VORTEX", state.vortex),
        (DockAction::ZeroG, "ZERO-G", state.zero_g),
        (DockAction::Scale(ScalePreset::Low), "60K", state.preset == ScalePreset::Low),
        (DockAction::Scale(ScalePreset::Mid), "150K", state.preset == ScalePreset::Mid),
        (DockAction::Scale(ScalePreset::High), "250K", state.preset == ScalePreset::High),
    ];
    let mut x = dock_x + pad;
    for ((action, label, active), wt) in row2.into_iter().zip(weights) {
        let bw = avail * (wt / weight_sum);
        buttons.push(DockButton {
            rect: [x, r2_y, bw, bh],
            action,
            label,
            active,
            swatch: None,
        });
        x += bw + gap;
    }

    let top_y = 14.0 * s;
    let orbit_w = 80.0 * s;
    let orbit_h = 44.0 * s;
    let orbit_rect = [margin, top_y, orbit_w, orbit_h];
    buttons.push(DockButton {
        rect: orbit_rect,
        action: DockAction::OrbitMode,
        label: if state.orbit_mode { "ORBIT" } else { "STIR" },
        active: state.orbit_mode,
        swatch: None,
    });

    let perf_w = 128.0 * s;
    let perf_h = 30.0 * s;
    let perf_rect = [w - perf_w - margin, top_y + 7.0 * s, perf_w, perf_h];

    DockLayout {
        panels: vec![
            [dock_x, dock_y, dock_w, dock_h, 22.0 * s],
            [orbit_rect[0], orbit_rect[1], orbit_rect[2], orbit_rect[3], orbit_h * 0.5],
            [perf_rect[0], perf_rect[1], perf_rect[2], perf_rect[3], perf_h * 0.5],
        ],
        dividers: Vec::new(),
        buttons,
        perf_rect,
        scale: s,
    }
}

pub fn build_ui_quads(state: &QuicksilverState, layout: &DockLayout, out: &mut Vec<UiQuad>) {
    out.clear();
    let s = layout.scale;
    let compact = state.viewport[0] / math::max(state.pixel_ratio, 1.0) < 780.0;
    let accent = state.material.swatch();

    for panel in &layout.panels {
        out.push(UiQuad {
            rect: [panel[0], panel[1], panel[2], panel[3]],
            uv: [0.0; 4],
            color: [0.04, 0.048, 0.065, 0.78],
            params: [KIND_GLASS, panel[4], 1.1 * s, 0.0],
        });
    }

    for div in &layout.dividers {
        out.push(UiQuad {
            rect: *div,
            uv: [0.0; 4],
            color: [1.0, 1.0, 1.0, 0.14],
            params: [KIND_DIVIDER, 0.5 * s, 0.0, 0.0],
        });
    }

    for (idx, btn) in layout.buttons.iter().enumerate() {
        let hovered = state.ui.hovered == Some(idx);
        let pressed = state.ui.pressed == Some(idx);
        let radius = btn.rect[3] * 0.5;
        let fill = if btn.active {
            let tint = btn.swatch.unwrap_or(accent);
            [
                0.14 + tint[0] * 0.14,
                0.15 + tint[1] * 0.14,
                0.19 + tint[2] * 0.14,
                0.92,
            ]
        } else if pressed {
            [0.20, 0.23, 0.29, 0.78]
        } else if hovered {
            [0.14, 0.17, 0.22, 0.65]
        } else {
            [0.08, 0.095, 0.125, 0.25]
        };
        let glow = if btn.active {
            1.0
        } else if hovered {
            0.35
        } else {
            0.0
        };
        out.push(UiQuad {
            rect: btn.rect,
            uv: [0.0; 4],
            color: fill,
            params: [KIND_BUTTON, radius, glow, 0.0],
        });

        let font_h = if compact {
            if btn.swatch.is_some() { 10.5 * s } else { 9.5 * s }
        } else {
            11.0 * s
        };
        let tracking = 0.06 * font_h;
        let text_w = measure_text(btn.label, font_h, tracking);

        if let Some(sw) = btn.swatch {
            let dot_d = if compact { 6.5 * s } else { 7.5 * s };
            let dot_gap = 5.0 * s;
            let total = dot_d + dot_gap + text_w;
            let start_x = btn.rect[0] + (btn.rect[2] - total) * 0.5;
            let cy = btn.rect[1] + btn.rect[3] * 0.5;
            out.push(UiQuad {
                rect: [start_x, cy - dot_d * 0.5, dot_d, dot_d],
                uv: [0.0; 4],
                color: sw,
                params: [KIND_SWATCH, dot_d * 0.5, if btn.active { 1.0 } else { 0.45 }, 0.0],
            });
            let text_col = if btn.active {
                [0.98, 0.99, 1.0, 1.0]
            } else if hovered {
                [0.88, 0.91, 0.96, 0.95]
            } else {
                [0.68, 0.73, 0.81, 0.88]
            };
            push_text(out, btn.label, start_x + dot_d + dot_gap, cy, font_h, tracking, text_col);
        } else {
            let start_x = btn.rect[0] + (btn.rect[2] - text_w) * 0.5;
            let cy = btn.rect[1] + btn.rect[3] * 0.5;
            let text_col = if btn.active {
                [0.98, 0.99, 1.0, 1.0]
            } else if hovered {
                [0.90, 0.93, 0.98, 0.95]
            } else {
                [0.70, 0.75, 0.83, 0.88]
            };
            push_text(out, btn.label, start_x, cy, font_h, tracking, text_col);
        }
    }

    let perf_text = format!("{} · {} FPS", state.preset.label(), state.perf.displayed_fps);
    let perf_font = 10.5 * s;
    let perf_track = 0.05 * perf_font;
    let pw = measure_text(&perf_text, perf_font, perf_track);
    let px = layout.perf_rect[0] + (layout.perf_rect[2] - pw) * 0.5;
    let py = layout.perf_rect[1] + layout.perf_rect[3] * 0.5;
    push_text(
        out,
        &perf_text,
        px,
        py,
        perf_font,
        perf_track,
        [0.82, 0.87, 0.95, 0.88],
    );
}

fn measure_text(text: &str, font_h: f32, tracking: f32) -> f32 {
    let mut w = 0.0_f32;
    let mut count = 0u32;
    for ch in text.chars() {
        let (_, adv) = glyph_uv(ch);
        w += adv * font_h;
        count += 1;
    }
    if count > 1 {
        w += (count - 1) as f32 * tracking;
    }
    w
}

fn push_text(
    out: &mut Vec<UiQuad>,
    text: &str,
    mut x: f32,
    center_y: f32,
    font_h: f32,
    tracking: f32,
    color: [f32; 4],
) {
    let cell_h = font_h * 1.48;
    let cell_w = font_h * 1.10;
    for ch in text.chars() {
        let (uv, adv) = glyph_uv(ch);
        let glyph_w = adv * font_h;
        if ch != ' ' {
            let cx = x + glyph_w * 0.5;
            out.push(UiQuad {
                rect: [cx - cell_w * 0.5, center_y - cell_h * 0.5, cell_w, cell_h],
                uv,
                color,
                params: [KIND_GLYPH, 0.0, 0.0, 0.0],
            });
        }
        x += glyph_w + tracking;
    }
}
