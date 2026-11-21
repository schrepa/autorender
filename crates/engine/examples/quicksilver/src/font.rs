use engine::math;

pub const ATLAS_W: u32 = 512;
pub const ATLAS_H: u32 = 128;
pub const COLS: u32 = 16;
pub const ROWS: u32 = 3;
pub const CELL_W: u32 = ATLAS_W / COLS;
pub const CELL_H: u32 = ATLAS_H / ROWS;

const GLYPH_ORDER: &str = "ABCDEFGHIJKLMNOPQRSTUVWX0123456789YZ-·./:+%() ";

#[derive(Clone, Copy)]
enum Stroke {
    Line(f32, f32, f32, f32),
    Arc(f32, f32, f32, f32, f32),
    Dot(f32, f32, f32),
}

pub fn glyph_uv(ch: char) -> ([f32; 4], f32) {
    let idx = GLYPH_ORDER
        .chars()
        .position(|c| c == ch)
        .unwrap_or(GLYPH_ORDER.chars().count() - 1) as u32;
    let col = idx % COLS;
    let row = idx / COLS;
    let u0 = (col * CELL_W) as f32 / ATLAS_W as f32;
    let v0 = (row * CELL_H) as f32 / ATLAS_H as f32;
    let u1 = ((col + 1) * CELL_W) as f32 / ATLAS_W as f32;
    let v1 = ((row + 1) * CELL_H) as f32 / ATLAS_H as f32;
    let adv = match ch {
        'I' | '1' | '.' | ':' | '·' => 0.50,
        ' ' => 0.36,
        '-' => 0.56,
        'M' | 'W' => 0.86,
        _ => 0.72,
    };
    ([u0, v0, u1 - u0, v1 - v0], adv)
}

pub fn build_sdf_atlas() -> Vec<u8> {
    let mut pixels = vec![0u8; (ATLAS_W * ATLAS_H) as usize];
    let stroke_radius = 0.088_f32;
    let spread = 0.24_f32;
    for (idx, ch) in GLYPH_ORDER.chars().enumerate() {
        let strokes = glyph_strokes(ch);
        let col = (idx as u32) % COLS;
        let row = (idx as u32) / COLS;
        let ox = col * CELL_W;
        let oy = row * CELL_H;
        for py in 0..CELL_H {
            let ny = ((py as f32 + 0.5) / CELL_H as f32 - 0.5) * 1.48 + 0.5;
            for px in 0..CELL_W {
                let nx = ((px as f32 + 0.5) / CELL_W as f32 - 0.5) * 1.38 + 0.5;
                let mut dist = 10.0_f32;
                for &s in strokes {
                    dist = math::min(dist, stroke_dist(nx, ny, s));
                }
                let signed = dist - stroke_radius;
                let v = math::clamp(0.5 - signed / (spread * 2.0), 0.0, 1.0);
                let at = ((oy + py) * ATLAS_W + (ox + px)) as usize;
                pixels[at] = (v * 255.0 + 0.5) as u8;
            }
        }
    }
    pixels
}

fn stroke_dist(px: f32, py: f32, stroke: Stroke) -> f32 {
    match stroke {
        Stroke::Line(ax, ay, bx, by) => seg_dist(px, py, ax, ay, bx, by),
        Stroke::Arc(cx, cy, r, a0, a1) => arc_dist(px, py, cx, cy, r, a0, a1),
        Stroke::Dot(cx, cy, extra) => {
            let dx = px - cx;
            let dy = py - cy;
            (dx * dx + dy * dy).sqrt() - extra
        }
    }
}

fn seg_dist(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let abx = bx - ax;
    let aby = by - ay;
    let denom = abx * abx + aby * aby;
    let t = if denom > 1.0e-6 {
        math::clamp(((px - ax) * abx + (py - ay) * aby) / denom, 0.0, 1.0)
    } else {
        0.0
    };
    let dx = px - (ax + abx * t);
    let dy = py - (ay + aby * t);
    (dx * dx + dy * dy).sqrt()
}

fn arc_dist(px: f32, py: f32, cx: f32, cy: f32, r: f32, a0: f32, a1: f32) -> f32 {
    let dx = px - cx;
    let dy = py - cy;
    let mut ang = math::atan2(dy, dx);
    let tau = core::f32::consts::TAU;
    while ang < a0 {
        ang += tau;
    }
    while ang > a0 + tau {
        ang -= tau;
    }
    if ang <= a1 {
        ((dx * dx + dy * dy).sqrt() - r).abs()
    } else {
        let (s0, c0) = math::sin_cos(a0);
        let (s1, c1) = math::sin_cos(a1);
        let d0 = ((px - (cx + c0 * r)).powi(2) + (py - (cy + s0 * r)).powi(2)).sqrt();
        let d1 = ((px - (cx + c1 * r)).powi(2) + (py - (cy + s1 * r)).powi(2)).sqrt();
        math::min(d0, d1)
    }
}

fn glyph_strokes(ch: char) -> &'static [Stroke] {
    match ch {
        'A'..='M' => letter_strokes_a_m(ch),
        'N'..='Z' => letter_strokes_n_z(ch),
        _ => digit_symbol_strokes(ch),
    }
}

fn letter_strokes_a_m(ch: char) -> &'static [Stroke] {
    use Stroke::{Arc, Line};
    const PI: f32 = core::f32::consts::PI;
    const HALF: f32 = core::f32::consts::FRAC_PI_2;
    const TAU: f32 = core::f32::consts::TAU;
    match ch {
        'A' => &[
            Line(0.16, 0.88, 0.50, 0.12),
            Line(0.50, 0.12, 0.84, 0.88),
            Line(0.28, 0.64, 0.72, 0.64),
        ],
        'B' => &[
            Line(0.22, 0.12, 0.22, 0.88),
            Line(0.22, 0.12, 0.56, 0.12),
            Line(0.22, 0.48, 0.56, 0.48),
            Line(0.22, 0.88, 0.58, 0.88),
            Arc(0.56, 0.30, 0.18, -HALF, HALF),
            Arc(0.58, 0.68, 0.20, -HALF, HALF),
        ],
        'C' => &[Arc(0.53, 0.50, 0.36, 0.48, TAU - 0.48)],
        'D' => &[
            Line(0.22, 0.12, 0.22, 0.88),
            Line(0.22, 0.12, 0.46, 0.12),
            Line(0.22, 0.88, 0.46, 0.88),
            Arc(0.46, 0.50, 0.38, -HALF, HALF),
        ],
        'E' => &[
            Line(0.22, 0.12, 0.22, 0.88),
            Line(0.22, 0.12, 0.78, 0.12),
            Line(0.22, 0.50, 0.68, 0.50),
            Line(0.22, 0.88, 0.78, 0.88),
        ],
        'F' => &[
            Line(0.24, 0.12, 0.24, 0.88),
            Line(0.24, 0.12, 0.78, 0.12),
            Line(0.24, 0.50, 0.68, 0.50),
        ],
        'G' => &[
            Arc(0.52, 0.50, 0.36, 0.08, TAU - 0.55),
            Line(0.88, 0.52, 0.54, 0.52),
            Line(0.88, 0.52, 0.88, 0.86),
        ],
        'H' => &[
            Line(0.22, 0.12, 0.22, 0.88),
            Line(0.78, 0.12, 0.78, 0.88),
            Line(0.22, 0.50, 0.78, 0.50),
        ],
        'I' => &[Line(0.50, 0.12, 0.50, 0.88)],
        'J' => &[
            Line(0.68, 0.12, 0.68, 0.66),
            Arc(0.46, 0.66, 0.22, 0.0, PI - 0.2),
        ],
        'K' => &[
            Line(0.22, 0.12, 0.22, 0.88),
            Line(0.78, 0.12, 0.26, 0.52),
            Line(0.42, 0.40, 0.80, 0.88),
        ],
        'L' => &[
            Line(0.24, 0.12, 0.24, 0.88),
            Line(0.24, 0.88, 0.78, 0.88),
        ],
        'M' => &[
            Line(0.14, 0.88, 0.14, 0.12),
            Line(0.14, 0.12, 0.50, 0.60),
            Line(0.50, 0.60, 0.86, 0.12),
            Line(0.86, 0.12, 0.86, 0.88),
        ],
        _ => &[],
    }
}

fn letter_strokes_n_z(ch: char) -> &'static [Stroke] {
    use Stroke::{Arc, Line};
    const PI: f32 = core::f32::consts::PI;
    const HALF: f32 = core::f32::consts::FRAC_PI_2;
    const TAU: f32 = core::f32::consts::TAU;
    match ch {
        'N' => &[
            Line(0.20, 0.88, 0.20, 0.12),
            Line(0.20, 0.12, 0.80, 0.88),
            Line(0.80, 0.88, 0.80, 0.12),
        ],
        'O' => &[Arc(0.50, 0.50, 0.37, 0.0, TAU)],
        'P' => &[
            Line(0.24, 0.12, 0.24, 0.88),
            Line(0.24, 0.12, 0.56, 0.12),
            Line(0.24, 0.52, 0.56, 0.52),
            Arc(0.56, 0.32, 0.20, -HALF, HALF),
        ],
        'Q' => &[
            Arc(0.50, 0.50, 0.36, 0.0, TAU),
            Line(0.62, 0.66, 0.86, 0.90),
        ],
        'R' => &[
            Line(0.22, 0.12, 0.22, 0.88),
            Line(0.22, 0.12, 0.56, 0.12),
            Line(0.22, 0.52, 0.56, 0.52),
            Arc(0.56, 0.32, 0.20, -HALF, HALF),
            Line(0.52, 0.52, 0.78, 0.88),
        ],
        'S' => &[
            Arc(0.50, 0.31, 0.19, HALF, TAU - 0.32),
            Arc(0.50, 0.69, 0.19, -HALF, PI - 0.32),
        ],
        'T' => &[
            Line(0.16, 0.12, 0.84, 0.12),
            Line(0.50, 0.12, 0.50, 0.88),
        ],
        'U' => &[
            Line(0.22, 0.12, 0.22, 0.60),
            Line(0.78, 0.12, 0.78, 0.60),
            Arc(0.50, 0.60, 0.28, 0.0, PI),
        ],
        'V' => &[
            Line(0.18, 0.12, 0.50, 0.88),
            Line(0.50, 0.88, 0.82, 0.12),
        ],
        'W' => &[
            Line(0.12, 0.12, 0.28, 0.88),
            Line(0.28, 0.88, 0.50, 0.36),
            Line(0.50, 0.36, 0.72, 0.88),
            Line(0.72, 0.88, 0.88, 0.12),
        ],
        'X' => &[
            Line(0.20, 0.12, 0.80, 0.88),
            Line(0.80, 0.12, 0.20, 0.88),
        ],
        'Y' => &[
            Line(0.18, 0.12, 0.50, 0.52),
            Line(0.82, 0.12, 0.50, 0.52),
            Line(0.50, 0.52, 0.50, 0.88),
        ],
        'Z' => &[
            Line(0.20, 0.12, 0.80, 0.12),
            Line(0.80, 0.12, 0.20, 0.88),
            Line(0.20, 0.88, 0.80, 0.88),
        ],
        _ => &[],
    }
}

fn digit_symbol_strokes(ch: char) -> &'static [Stroke] {
    use Stroke::{Arc, Dot, Line};
    const PI: f32 = core::f32::consts::PI;
    const HALF: f32 = core::f32::consts::FRAC_PI_2;
    const TAU: f32 = core::f32::consts::TAU;
    match ch {
        '0' => &[
            Line(0.22, 0.34, 0.22, 0.66),
            Line(0.78, 0.34, 0.78, 0.66),
            Arc(0.50, 0.34, 0.28, PI, TAU),
            Arc(0.50, 0.66, 0.28, 0.0, PI),
        ],
        '1' => &[
            Line(0.32, 0.26, 0.52, 0.12),
            Line(0.52, 0.12, 0.52, 0.88),
            Line(0.30, 0.88, 0.74, 0.88),
        ],
        '2' => &[
            Arc(0.50, 0.32, 0.22, PI + 0.2, TAU + 0.55),
            Line(0.68, 0.44, 0.22, 0.88),
            Line(0.22, 0.88, 0.78, 0.88),
        ],
        '3' => &[
            Arc(0.48, 0.30, 0.19, PI + 0.4, TAU + HALF),
            Arc(0.48, 0.69, 0.20, -HALF, PI - 0.35),
        ],
        '4' => &[
            Line(0.66, 0.88, 0.66, 0.12),
            Line(0.66, 0.12, 0.18, 0.64),
            Line(0.18, 0.64, 0.82, 0.64),
        ],
        '5' => &[
            Line(0.74, 0.12, 0.26, 0.12),
            Line(0.26, 0.12, 0.24, 0.46),
            Line(0.24, 0.46, 0.52, 0.46),
            Arc(0.50, 0.67, 0.21, -HALF, PI - 0.28),
        ],
        '6' => &[
            Arc(0.52, 0.42, 0.28, PI, TAU - 0.65),
            Line(0.24, 0.42, 0.24, 0.64),
            Arc(0.50, 0.64, 0.24, 0.0, TAU),
        ],
        '7' => &[
            Line(0.20, 0.12, 0.80, 0.12),
            Line(0.80, 0.12, 0.38, 0.88),
        ],
        '8' => &[
            Arc(0.50, 0.30, 0.18, 0.0, TAU),
            Arc(0.50, 0.68, 0.20, 0.0, TAU),
        ],
        '9' => &[
            Arc(0.50, 0.36, 0.24, 0.0, TAU),
            Line(0.74, 0.36, 0.74, 0.58),
            Arc(0.48, 0.58, 0.26, 0.0, PI - 0.55),
        ],
        '-' => &[Line(0.24, 0.52, 0.76, 0.52)],
        '·' => &[Dot(0.50, 0.50, 0.04)],
        '.' => &[Dot(0.50, 0.82, 0.03)],
        '/' => &[Line(0.72, 0.12, 0.28, 0.88)],
        ':' => &[Dot(0.50, 0.34, 0.03), Dot(0.50, 0.74, 0.03)],
        '+' => &[
            Line(0.50, 0.26, 0.50, 0.74),
            Line(0.26, 0.50, 0.74, 0.50),
        ],
        _ => &[],
    }
}
