use brick_interface::{Area, Case, Pen, Plate, Rgba, Stroke};
use engine::math;
use glam::Vec2;

use super::hud::{HudFacts, Reticle};
use super::menu::ScopeNumbers;
use super::{DamageEdge, Marks, ScreenSource};

const WHITE: Rgba = Rgba([1.0; 4]);
const ARC_PIECES: usize = 5;
const SCOPE_OPEN: f32 = 0.35;
const ARMOR_POP: f32 = 0.3;
const ARMOR_POP_SHARE: f32 = 0.3;

pub(super) const PIECES: [&str; 4] = ["blind", "scope", "hud_marks", "reticle"];

pub(super) fn piece(
    facts: &HudFacts,
    source: &ScreenSource,
    piece: usize,
    area: Area,
    pen: &mut Pen<'_>,
) {
    let marks = &source.marks;
    match piece {
        0 => blind(pen, facts.blind),
        1 => {
            if facts.scope_ease > 0.0 {
                scope(pen, &source.menu.scope, facts.scope_ease);
            }
        }
        2 => hud_marks(pen, facts, marks),
        3 => reticle(
            pen,
            facts,
            &marks.crosshair_outline,
            (area.center().round(), 0.0),
        ),
        _ => {}
    }
}

fn flat(fill: Option<Rgba>, radius: f32, stroke: Option<Stroke>) -> Plate {
    Plate {
        fill,
        radius,
        stroke,
        angle_radians: 0.0,
        soft: 1.0,
        sweep: 1.0,
    }
}

fn blind(pen: &mut Pen<'_>, amount: f32) {
    if amount <= 0.0 {
        return;
    }
    let size = pen.size();
    let whole = Area {
        x: 0.0,
        y: 0.0,
        w: size.x,
        h: size.y,
    };
    let fill = Some(WHITE.faded(math::min(amount, 1.0)));
    pen.plate(whole, flat(fill, 0.0, None), None);
}

fn scope(pen: &mut Pen<'_>, scope: &ScopeNumbers, ease: f32) {
    let (Some(color), Some(shade)) = (pen.palette(&scope.color), pen.palette(&scope.shade))
    else {
        return;
    };
    let ease = math::clamp(ease, 0.0, 1.0);
    let size = pen.size();
    let center = (size * 0.5).round();
    let clear = size.y * scope.ring_share * 0.5 * (1.0 + SCOPE_OPEN * (1.0 - ease));
    let outer = size.length() * 0.5 + 2.0;
    let whole = Area {
        x: center.x - outer,
        y: center.y - outer,
        w: 2.0 * outer,
        h: 2.0 * outer,
    };
    let surround = Stroke {
        width: outer - clear,
        inset: 0.0,
        color: shade.faded(ease),
    };
    pen.plate(whole, flat(None, outer, Some(surround)), None);
    let ring = Area {
        x: center.x - clear,
        y: center.y - clear,
        w: 2.0 * clear,
        h: 2.0 * clear,
    };
    let stroke = Stroke {
        width: scope.width_px,
        inset: 0.0,
        color: color.faded(ease),
    };
    pen.plate(ring, flat(None, clear, Some(stroke)), None);
    let line = scope.line_px;
    let across = Area {
        x: ring.x,
        y: center.y - line * 0.5,
        w: ring.w,
        h: line,
    };
    let down = Area {
        x: center.x - line * 0.5,
        y: ring.y,
        w: line,
        h: ring.h,
    };
    pen.plate(across, flat(Some(color.faded(ease)), 0.0, None), None);
    pen.plate(down, flat(Some(color.faded(ease)), 0.0, None), None);
}

fn hud_marks(pen: &mut Pen<'_>, facts: &HudFacts, marks: &Marks) {
    let cross = facts.menu.settings.reticle;
    let reach = cross.gap + cross.length + cross.outline + facts.spread_px;
    if pen.covered(reach) {
        return;
    }
    let center = (pen.size() * 0.5).round();
    let sniping = facts.scope_ease > 0.0;
    if !sniping {
        reticle(
            pen,
            facts,
            &marks.crosshair_outline,
            (center, facts.spread_px),
        );
    }
    armor_and_damage(pen, facts, marks);
}

fn armor_and_damage(pen: &mut Pen<'_>, facts: &HudFacts, marks: &Marks) {
    let center = (pen.size() * 0.5).round();
    let ink = pen.palette(&marks.outline_color);
    let atlas = pen.atlas(&marks.atlas);
    if let (Some(age), Some(atlas)) = (facts.armor_break, atlas) {
        let numbers = &marks.armor;
        let s = pen.scale();
        let left = 1.0 - age / numbers.seconds;
        let pop =
            1.0 + ARMOR_POP * math::max(0.0, 1.0 - age / (numbers.seconds * ARMOR_POP_SHARE));
        let at = center + Vec2::new(numbers.offset_px[0], numbers.offset_px[1]) * s;
        let size = numbers.size_px * s * pop;
        if let Some([w, h]) = pen.cell_size(atlas, &numbers.icon) {
            pen.cell_turned(
                (atlas, &numbers.icon),
                at,
                (size / math::max(w, h), 0.0),
                WHITE.faded(left),
            );
        }
    }
    damage_marks(pen, facts, (&marks.damage, atlas), ink);
}

fn damage_marks(
    pen: &mut Pen<'_>,
    facts: &HudFacts,
    (edge, atlas): (&DamageEdge, Option<&str>),
    ink: Option<Rgba>,
) {
    let s = pen.scale();
    let size = pen.size();
    let center = size * 0.5;
    let radius = edge.radius_share * size.y;
    if radius <= 0.0 || edge.fade_seconds <= 0.0 {
        return;
    }
    let span = edge.length_deg.to_radians();
    let (band, rim) = (edge.band_px * s, edge.rim_px * s);
    let color = pen.palette(&edge.color);
    for mark in &facts.damage {
        let t = math::clamp(mark.age / edge.fade_seconds, 0.0, 1.0);
        let left = (1.0 - t) * (1.0 - t);
        let toward = -mark.angle_radians;
        let ends: [Vec2; ARC_PIECES + 1] = std::array::from_fn(|at| {
            let along = at as f32 / ARC_PIECES as f32 - 0.5;
            let (sin, cos) = math::sin_cos(toward + along * span);
            center + Vec2::new(sin, -cos) * radius
        });
        for (paint, thick) in [(ink, band + 2.0 * rim), (color, band)] {
            let Some(paint) = paint else {
                continue;
            };
            for pair in ends.windows(2) {
                let step = pair[1] - pair[0];
                let angle = math::atan2(step.y, step.x);
                round_bar(
                    pen,
                    (pair[0] + pair[1]) * 0.5,
                    (step.length() + thick, thick),
                    (angle, paint.faded(left)),
                );
            }
        }
        let (sin, cos) = math::sin_cos(toward);
        if let Some(atlas) = atlas {
            pen.cell_turned(
                (atlas, &edge.tip),
                center + Vec2::new(sin, -cos) * radius,
                (edge.tip_scale * s, toward),
                WHITE.faded(left),
            );
        }
    }
}

fn round_bar(pen: &mut Pen<'_>, at: Vec2, (w, h): (f32, f32), (angle, color): (f32, Rgba)) {
    let area = Area {
        x: at.x - w * 0.5,
        y: at.y - h * 0.5,
        w,
        h,
    };
    let plate = Plate {
        angle_radians: angle,
        ..flat(Some(color), h * 0.5, None)
    };
    pen.plate(area, plate, None);
}

fn reticle(
    pen: &mut Pen<'_>,
    facts: &HudFacts,
    outline_color: &str,
    (center, spread_px): (Vec2, f32),
) {
    let settings = &facts.menu.settings;
    let reticle: Reticle = settings.reticle;
    let Some(color) = pen.palette(&settings.crosshair_color) else {
        return;
    };
    let ink = pen.palette(outline_color);
    let thick = math::max(reticle.thickness.round(), 1.0);
    let gap = reticle.gap.round() + spread_px.round();
    let length = reticle.length.round();
    let outline = reticle.outline.round();
    let near = (thick * 0.5).floor();
    let (cx, cy) = (center.x - near, center.y - near);
    let mut pieces = vec![
        [cx - gap - length, cy, length, thick],
        [cx + thick + gap, cy, length, thick],
        [cx, cy - gap - length, thick, length],
        [cx, cy + thick + gap, thick, length],
    ];
    if reticle.dot {
        pieces.push([cx, cy, thick, thick]);
    }
    let case = ink.filter(|_| outline > 0.0).map(|ink| Case {
        color: ink,
        grow: outline,
        offset: [0.0, 0.0],
        doubled: false,
        rim: None,
    });
    for [x, y, w, h] in pieces {
        if w <= 0.0 || h <= 0.0 {
            continue;
        }
        pen.plate(Area { x, y, w, h }, flat(Some(color), 0.0, None), case);
    }
}
