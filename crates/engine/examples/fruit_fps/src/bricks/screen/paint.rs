use std::fmt::Write;

use engine::math;
use glam::Vec2;

use crate::play::screen::{
    Align, Area, Cover, Fit, HudFacts, Plate, Rgba, ScreenDraw, Shown, Stroke, TextStyle, Tone,
};

const KNOB_RING_PX: f32 = 6.0;

use super::layout::{
    Cond, Dir, Element, FlowKind, Format, Kind, Knob, Layout, Markers, Number, Paint, Piece, Set,
    StrokeRef, Template,
};
use super::menu::ScopeNumbers;
use super::paths::Val;
use super::{MapFrame, MarkNumbers, Marks};

pub(super) struct Painter<'a> {
    pub draw: &'a mut ScreenDraw,
    pub facts: &'a HudFacts,
    pub layout: &'a Layout,
    pub map: &'a MapFrame,
    pub scratch: &'a mut String,
    pub spare: &'a mut String,
}

struct Look<'a> {
    fill: Option<&'a Paint>,
    color: Option<&'a Paint>,
    stroke: Option<&'a StrokeRef>,
    icon: Option<&'a Template>,
}

fn whole_number(value: f32, out: &mut String) {
    let _ = write!(out, "{}", value as i64);
}

fn grouped(value: i64, out: &mut String) {
    if value < 0 {
        out.push('-');
    }
    let digits = value.unsigned_abs();
    let mut place = 1_u64;
    while place * 1000 <= digits {
        place *= 1000;
    }
    let mut first = true;
    while place > 0 {
        let group = (digits / place) % 1000;
        if first {
            let _ = write!(out, "{group}");
            first = false;
        } else {
            let _ = write!(out, ",{group:03}");
        }
        place /= 1000;
    }
}

fn formatted(value: Val<'_>, format: Format, out: &mut String) {
    match (value, format) {
        (Val::Unknown | Val::Missing | Val::Bool(_), _) => {}
        (Val::Text(text), Format::Upper) => {
            for character in text.chars() {
                out.extend(character.to_uppercase());
            }
        }
        (
            Val::Text(text),
            Format::Plain | Format::Int | Format::Clock | Format::Credits | Format::Sec1,
        ) => {
            out.push_str(text);
        }
        (Val::Num(number), Format::Int) => whole_number(number.ceil(), out),
        (Val::Num(number), Format::Clock) => {
            let seconds = math::max(number.ceil(), 0.0) as i64;
            let _ = write!(out, "{}:{:02}", seconds / 60, seconds % 60);
        }
        (Val::Num(number), Format::Credits) => grouped(number as i64, out),
        (Val::Num(number), Format::Sec1) => {
            let _ = write!(out, "{number:.1}s");
        }
        (Val::Num(number), Format::Plain | Format::Upper) => {
            if number.fract() == 0.0 {
                whole_number(number, out);
            } else {
                let _ = write!(out, "{number}");
            }
        }
    }
}

fn number_of(facts: &HudFacts, number: &Number) -> Option<f32> {
    match number {
        Number::Fixed(value) => Some(*value),
        Number::Path(path) => facts.value(path).number(),
    }
}

impl Painter<'_> {
    pub(super) fn holds(&self, cond: Option<&Cond>) -> bool {
        cond.is_none_or(|cond| self.facts.value(&cond.path).truthy() != cond.negated)
    }

    fn color(&self, paint: Option<&Paint>) -> Option<Rgba> {
        match paint? {
            Paint::None => None,
            Paint::Fixed(color) => Some(*color),
            Paint::Path(path) => self
                .facts
                .value(path)
                .text()
                .and_then(|key| self.layout.palette.get(key).copied()),
        }
    }

    fn look<'e>(&self, element: &'e Element) -> Look<'e> {
        let mut look = Look {
            fill: element.look.fill.as_ref(),
            color: element.look.color.as_ref(),
            stroke: element.look.stroke.as_ref(),
            icon: element.look.icon.as_ref(),
        };
        for (cond, set) in &element.styles {
            if self.holds(Some(cond)) {
                apply(&mut look, set);
            }
        }
        look
    }

    fn render(&self, template: &Template, out: &mut String) {
        out.clear();
        for piece in &template.pieces {
            match piece {
                Piece::Text(text) => out.push_str(text),
                Piece::Value(path, format) => formatted(self.facts.value(path), *format, out),
            }
        }
    }

    fn stroke(&self, stroke: Option<&StrokeRef>, s: f32) -> Option<Stroke> {
        let stroke = stroke?;
        Some(Stroke {
            width: stroke.width * s,
            inset: self.layout.stroke_rim * s,
            color: self.color(Some(&stroke.color))?,
        })
    }

    fn plate(&mut self, area: Area, fill: Option<Rgba>, radius: f32, stroke: Option<Stroke>) {
        self.draw.plate(
            area,
            Plate {
                fill,
                radius,
                stroke,
                angle_radians: 0.0,
            },
        );
    }

    pub(super) fn screen(&mut self, shown: Shown, spread_px: f32) {
        let Some(screen) = self.layout.screens.get(&shown) else {
            return;
        };
        if let Some(over) = screen.over {
            self.screen(over, spread_px);
        }
        for element in &screen.elements {
            self.element(element, spread_px);
        }
    }

    fn crosshair_shift(&self, area: Area) -> Vec2 {
        let center = (self.draw.size * 0.5).round();
        let delta = area.center() - center;
        let step = |value: f32| {
            if value >= 1.0 {
                1.0
            } else if value <= -1.0 {
                -1.0
            } else {
                0.0
            }
        };
        Vec2::new(step(delta.x), step(delta.y))
    }

    fn element(&mut self, element: &Element, spread_px: f32) {
        if !self.holds(element.when.as_ref()) {
            return;
        }
        let look = self.look(element);
        let mut area = self
            .draw
            .place(element.rect, element.anchor, element.unscaled);
        if element.unscaled {
            let shift = self.crosshair_shift(area) * spread_px;
            area.x += shift.x;
            area.y += shift.y;
        }
        let s = if element.unscaled {
            1.0
        } else {
            self.draw.scale()
        };
        let radius = element.radius * s;
        match &element.kind {
            Kind::Plate => {
                let fill = self.color(look.fill);
                let stroke = self.stroke(look.stroke, s);
                self.plate(area, fill, radius, stroke);
            }
            Kind::Text(text) => {
                let Some(color) = self.color(look.color) else {
                    return;
                };
                let px = self.draw.text_px(text.size, element.unscaled);
                let mut out = std::mem::take(self.scratch);
                self.render(&text.text, &mut out);
                self.draw.text(
                    &out,
                    area,
                    TextStyle {
                        px,
                        weight: text.weight,
                        align: text.align,
                        color,
                    },
                );
                *self.scratch = out;
            }
            Kind::Icon(fit) => {
                let (Some(color), Some(icon)) = (self.color(look.color), look.icon) else {
                    return;
                };
                let mut out = std::mem::take(self.scratch);
                self.render(icon, &mut out);
                if !out.is_empty() {
                    self.draw.icon(&out, area, (color, *fit));
                }
                *self.scratch = out;
            }
            Kind::Image(picture) => {
                let Some(color) = self.color(look.color) else {
                    return;
                };
                let mut out = std::mem::take(self.scratch);
                let mut name = std::mem::take(self.spare);
                self.render(&picture.image, &mut name);
                if let Some(small) = &picture.small {
                    self.render(small, &mut out);
                    if self
                        .draw
                        .picture_size(&out)
                        .is_some_and(|size| area.h <= size[1])
                    {
                        std::mem::swap(&mut name, &mut out);
                    }
                }
                let fit = if picture.cover {
                    Cover::Fill
                } else {
                    Cover::Fit
                };
                match &picture.cell {
                    Some((_, key)) => {
                        self.render(key, &mut out);
                        if !out.is_empty() {
                            self.draw.image((&name, Some(&out), fit), area, color);
                        }
                    }
                    None if !name.is_empty() => self.draw.image((&name, None, fit), area, color),
                    None => {}
                }
                *self.scratch = out;
                *self.spare = name;
            }
            Kind::Markers(markers) => self.markers(markers, area, s, self.color(look.color)),
            Kind::Bar {
                value,
                max,
                track,
                dir,
                knob,
            } => {
                let end = self.bar(
                    area,
                    radius,
                    (value, max),
                    (self.color(Some(track)), self.color(look.fill)),
                    *dir,
                );
                if let Some(knob) = knob {
                    self.knob(knob, end, s);
                }
            }
            Kind::Pips { .. } => self.pips(element, area, s),
            Kind::Flow { .. } => self.flow(element, &look, area, s),
        }
    }

    fn bar(
        &mut self,
        area: Area,
        radius: f32,
        (value, max): (&Number, &Number),
        (track, fill): (Option<Rgba>, Option<Rgba>),
        dir: Dir,
    ) -> Vec2 {
        if track.is_some() {
            self.plate(area, track, radius, None);
        }
        let value = number_of(self.facts, value).unwrap_or(0.0);
        let max = number_of(self.facts, max)
            .filter(|max| *max != 0.0)
            .unwrap_or(1.0);
        let share = math::clamp(value / max, 0.0, 1.0);
        let filled = match dir {
            Dir::LeftToRight => Area {
                w: area.w * share,
                ..area
            },
            Dir::RightToLeft => Area {
                x: area.x + area.w * (1.0 - share),
                w: area.w * share,
                ..area
            },
            Dir::BottomToTop => Area {
                y: area.y + area.h * (1.0 - share),
                h: area.h * share,
                ..area
            },
        };
        let rounded = if dir == Dir::BottomToTop && share < 1.0 {
            radius
        } else {
            math::min(radius, math::min(filled.w, filled.h) * 0.5)
        };
        self.plate(filled, fill, rounded, None);
        match dir {
            Dir::LeftToRight => Vec2::new(filled.x + filled.w, area.y + area.h * 0.5),
            Dir::RightToLeft => Vec2::new(filled.x, area.y + area.h * 0.5),
            Dir::BottomToTop => Vec2::new(area.x + area.w * 0.5, filled.y),
        }
    }

    fn knob(&mut self, knob: &Knob, at: Vec2, s: f32) {
        let disc = |wide: f32| Area {
            x: at.x - wide * 0.5,
            y: at.y - wide * 0.5,
            w: wide,
            h: wide,
        };
        let size = knob.size * s;
        if let Some(ring) = knob.ring.as_ref().and_then(|ring| self.color(Some(ring))) {
            let wide = size + KNOB_RING_PX * s;
            self.plate(disc(wide), Some(ring), wide * 0.5, None);
        }
        let fill = self.color(Some(&knob.fill));
        self.plate(disc(size), fill, size * 0.5, None);
    }

    fn pips(&mut self, element: &Element, area: Area, s: f32) {
        let Kind::Pips {
            count,
            value,
            size,
            gap,
            on,
            off,
            full,
            align,
        } = &element.kind
        else {
            return;
        };
        let count = number_of(self.facts, count).unwrap_or(0.0).max_whole();
        let value = number_of(self.facts, value).unwrap_or(0.0).max_whole();
        let (pw, ph, g) = (size[0] * s, size[1] * s, gap * s);
        let total = count as f32 * pw + count.saturating_sub(1) as f32 * g;
        let x = match align {
            Align::Left => area.x,
            Align::Center | Align::Right => area.x + (area.w - total) * 0.5,
        };
        let y = area.y + (area.h - ph) * 0.5;
        let full = full.as_ref().filter(|_| value >= count);
        let (on, off, full) = (
            self.color(Some(on)),
            self.color(Some(off)),
            full.and_then(|full| self.color(Some(full))),
        );
        for index in 0..count {
            let color = match full {
                Some(full) => Some(full),
                None if index < value => on,
                None => off,
            };
            let pip = Area {
                x: x + index as f32 * (pw + g),
                y,
                w: pw,
                h: ph,
            };
            self.plate(pip, color, math::min(pw, ph) * 0.5, None);
        }
    }

    fn flow(&mut self, element: &Element, look: &Look<'_>, area: Area, s: f32) {
        let Kind::Flow {
            align,
            padding,
            gap,
            items,
        } = &element.kind
        else {
            return;
        };
        let mut out = std::mem::take(self.scratch);
        let (mut widths, mut shown) = (0.0, 0_u32);
        for item in items {
            if !self.holds(item.when.as_ref()) {
                continue;
            }
            widths += match &item.kind {
                FlowKind::Text { text, .. } => {
                    self.render(&text.text, &mut out);
                    self.draw
                        .measure(&out, self.draw.text_px(text.size, false), text.weight)
                }
                FlowKind::Icon { w, .. } => w * s,
            };
            shown += 1;
        }
        let (pad, gap) = (padding * s, gap * s);
        let width = 2.0 * pad + widths + gap * shown.saturating_sub(1) as f32;
        let x = match align {
            Align::Left => area.x,
            Align::Center => area.x + (area.w - width) * 0.5,
            Align::Right => area.x + area.w - width,
        };
        let fill = self.color(look.fill);
        let stroke = self.stroke(look.stroke, s);
        self.plate(
            Area {
                x,
                w: width,
                ..area
            },
            fill,
            element.radius * s,
            stroke,
        );
        let mut at = x + pad;
        for item in items {
            if !self.holds(item.when.as_ref()) {
                continue;
            }
            match &item.kind {
                FlowKind::Text { text, color } => {
                    self.render(&text.text, &mut out);
                    let px = self.draw.text_px(text.size, false);
                    let wide = self.draw.measure(&out, px, text.weight);
                    if let Some(color) = self.color(Some(color)) {
                        let style = TextStyle {
                            px,
                            weight: text.weight,
                            align: Align::Left,
                            color,
                        };
                        self.draw.text(
                            &out,
                            Area {
                                x: at,
                                w: wide,
                                ..area
                            },
                            style,
                        );
                    }
                    at += wide + gap;
                }
                FlowKind::Icon { icon, w, h, color } => {
                    let (iw, ih) = (w * s, h * s);
                    self.render(icon, &mut out);
                    if let Some(color) = self.color(Some(color)) {
                        let place = Area {
                            x: at,
                            y: area.y + (area.h - ih) * 0.5,
                            w: iw,
                            h: ih,
                        };
                        self.draw.icon(&out, place, (color, Fit::Cell));
                    }
                    at += iw + gap;
                }
            }
        }
        *self.scratch = out;
    }

    fn markers(&mut self, markers: &Markers, area: Area, s: f32, color: Option<Rgba>) {
        let facts = self.facts;
        let Some(items) = facts.map_items(&markers.items) else {
            return;
        };
        let map = self.map;
        if map.size[0] <= 0.0 || map.size[1] <= 0.0 {
            return;
        }
        let size = markers.size * s;
        let backing = markers
            .backing
            .as_ref()
            .and_then(|(fill, wide)| Some((self.color(Some(fill))?, wide * s)));
        let outline = markers
            .outline
            .as_ref()
            .and_then(|(paint, icon)| Some((self.color(Some(paint))?, icon.as_str())));
        for item in items {
            let px = (item.pos[0] - map.origin_m[0]) * map.px_per_m;
            let py = (map.origin_m[1] - item.pos[1]) * map.px_per_m;
            let center = Vec2::new(
                area.x + px * area.w / map.size[0],
                area.y + py * area.h / map.size[1],
            );
            let angle = if markers.rotate {
                item.yaw_degrees.to_radians()
            } else {
                0.0
            };
            if let Some((fill, wide)) = backing {
                let disc = Area {
                    x: center.x - wide * 0.5,
                    y: center.y - wide * 0.5,
                    w: wide,
                    h: wide,
                };
                self.plate(disc, Some(fill), wide * 0.5, None);
            }
            if let Some((ink, icon)) = outline {
                self.draw.marker(icon, center, size, (ink, angle));
            }
            let tint = item.color.and_then(|tone| self.layout.tone(tone)).or(color);
            if let Some(tint) = tint {
                self.draw.marker(&markers.icon, center, size, (tint, angle));
            }
        }
    }

    pub(super) fn scope(&mut self, scope: &ScopeNumbers) {
        let Some(color) = self.layout.palette.get(&scope.color).copied() else {
            return;
        };
        let center = (self.draw.size * 0.5).round();
        let wide = self.draw.size.y * scope.ring_share;
        let ring = Area {
            x: center.x - wide * 0.5,
            y: center.y - wide * 0.5,
            w: wide,
            h: wide,
        };
        let stroke = Stroke {
            width: scope.width_px,
            inset: 0.0,
            color,
        };
        self.plate(ring, None, wide * 0.5, Some(stroke));
        let line = scope.line_px;
        let across = Area {
            x: ring.x,
            y: center.y - line * 0.5,
            w: wide,
            h: line,
        };
        let down = Area {
            x: center.x - line * 0.5,
            y: ring.y,
            w: line,
            h: wide,
        };
        self.plate(across, Some(color), 0.0, None);
        self.plate(down, Some(color), 0.0, None);
    }

    pub(super) fn backdrop(&mut self, color: &str) {
        let Some(color) = self.layout.palette.get(color).copied() else {
            return;
        };
        let size = self.draw.size;
        let whole = Area {
            x: 0.0,
            y: 0.0,
            w: size.x,
            h: size.y,
        };
        self.plate(whole, Some(color), 0.0, None);
    }

    pub(super) fn marks(&mut self, marks: &Marks) {
        let center = (self.draw.size * 0.5).round();
        if let Some(hit) = self.facts.hit {
            let tone = if hit.killed {
                Tone::Danger
            } else if hit.headshot {
                Tone::Ready
            } else {
                Tone::Text
            };
            let numbers = &marks.hit;
            if let Some(color) = self.layout.tone(tone) {
                let color = color.faded(1.0 - hit.age / numbers.seconds);
                let reach = numbers.gap_px + numbers.length_px * 0.5;
                for quarter in [1.0_f32, 3.0, 5.0, 7.0] {
                    let angle = quarter * std::f32::consts::FRAC_PI_4;
                    let (sin, cos) = math::sin_cos(angle);
                    self.bar_mark(
                        center + Vec2::new(cos, sin) * reach,
                        numbers,
                        1.0,
                        (angle, color),
                    );
                }
            }
        }
        let s = self.draw.scale();
        let numbers = &marks.damage;
        let Some(color) = self.layout.tone(Tone::Danger) else {
            return;
        };
        let facts = self.facts;
        for mark in &facts.damage {
            let (sin, cos) = math::sin_cos(mark.angle_radians);
            let at = center + Vec2::new(-sin, -cos) * numbers.gap_px * s;
            let faded = color.faded(1.0 - mark.age / numbers.seconds);
            self.bar_mark(at, numbers, s, (-mark.angle_radians, faded));
        }
    }

    fn bar_mark(&mut self, at: Vec2, numbers: &MarkNumbers, s: f32, (angle, color): (f32, Rgba)) {
        let (w, h) = (numbers.length_px * s, numbers.width_px * s);
        self.draw.plate(
            Area {
                x: at.x - w * 0.5,
                y: at.y - h * 0.5,
                w,
                h,
            },
            Plate {
                fill: Some(color),
                radius: h * 0.5,
                stroke: None,
                angle_radians: angle,
            },
        );
    }
}

fn apply<'e>(look: &mut Look<'e>, set: &'e Set) {
    if let Some(fill) = &set.fill {
        look.fill = Some(fill);
    }
    if let Some(color) = &set.color {
        look.color = Some(color);
    }
    if let Some(stroke) = &set.stroke {
        look.stroke = Some(stroke);
    }
    if let Some(icon) = &set.icon {
        look.icon = Some(icon);
    }
}

trait Whole {
    fn max_whole(self) -> u32;
}

impl Whole for f32 {
    fn max_whole(self) -> u32 {
        if self.is_finite() && self > 0.0 {
            self as u32
        } else {
            0
        }
    }
}
