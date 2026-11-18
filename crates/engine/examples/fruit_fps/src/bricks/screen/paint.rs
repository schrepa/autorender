use std::fmt::Write;

use engine::math;
use glam::Vec2;

use crate::play::screen::{
    Align, Anchor, Area, Case, Cover, Fit, HudFacts, Plate, Reticle, Rgba, ScreenDraw, Shown,
    Stroke, TextStyle, Weight,
};

const KNOB_RING_PX: f32 = 6.0;
const KNOB_INSET_PX: f32 = 2.0;
const WHITE: Rgba = Rgba([1.0; 4]);
const ARC_PIECES: usize = 5;
const SCOPE_OPEN: f32 = 0.35;
const ARMOR_POP: f32 = 0.3;
const ARMOR_POP_SHARE: f32 = 0.3;

use super::layout::{
    Arc, Cond, Dir, Element, FlowItem, FlowKind, Format, Kind, Knob, Layout, Markers, Number,
    Paint, Picture, Piece, Screen, Set, Shadow, Stack, StrokeRef, Template, TextKind,
};
use super::menu::ScopeNumbers;
use super::paths::Val;
use super::{MapFrame, Marks};

pub(super) struct Painter<'a> {
    pub draw: &'a mut ScreenDraw,
    pub facts: &'a HudFacts,
    pub layout: &'a Layout,
    pub map: &'a MapFrame,
    pub scratch: &'a mut String,
    pub spare: &'a mut String,
    pub pulses: &'a mut [PulseState],
    pub rows: &'a mut [RowState],
    pub now: f64,
    pub zoom: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RowState {
    span: Option<(f32, f32)>,
    lift: Option<f32>,
}

fn place_rows(rows: &mut [RowState], stack: &Stack, (gap, ceiling, floor): (f32, f32, f32)) {
    for row in stack.rows.clone() {
        let Some((top, bottom)) = rows[row].span else {
            continue;
        };
        let tall = bottom - top;
        let taken = |rows: &[RowState], y: f32| {
            rows[stack.rows.start..row].iter().find_map(|placed| {
                let (Some((above, below)), Some(lift)) = (placed.span, placed.lift) else {
                    return None;
                };
                let (above, below) = (above + lift, below + lift);
                (y < below + gap && y + tall + gap > above).then_some((above, below))
            })
        };
        let mut y = top;
        while let Some((_, below)) = taken(rows, y) {
            y = below + gap;
        }
        if y + tall <= floor || y == top {
            rows[row].lift = Some(y - top);
            continue;
        }
        y = top;
        while let Some((above, _)) = taken(rows, y) {
            y = above - gap - tall;
        }
        rows[row].lift = (y >= ceiling).then_some(y - top);
    }
}

struct Measured {
    lead: f32,
    share: f32,
    over: f32,
    content: f32,
    longest: Option<(usize, f32)>,
    tallest: f32,
}

#[derive(Clone, Debug, Default)]
pub(super) struct PulseState {
    last: String,
    primed: bool,
    since: f64,
}

struct Look<'a> {
    fill: Option<&'a Paint>,
    color: Option<&'a Paint>,
    stroke: Option<&'a StrokeRef>,
    icon: Option<&'a Template>,
    grey: f32,
    shadows: &'a [Shadow],
}

fn equal(value: Val<'_>, wanted: &str) -> bool {
    match value {
        Val::Text(text) => text == wanted,
        Val::Num(number) => wanted.parse::<f32>().is_ok_and(|parsed| parsed == number),
        Val::Bool(flag) => wanted == if flag { "true" } else { "false" },
        Val::Unknown | Val::Missing => wanted.is_empty(),
    }
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
    fn px(&self, size: f32, unscaled: bool) -> f32 {
        self.draw.text_px(size, unscaled) * self.zoom
    }

    fn pulse<'e>(&mut self, element: &'e Element) -> (f32, Option<&'e Paint>) {
        let Some(pulse) = &element.pulse else {
            return (1.0, None);
        };
        let Some(state) = self.pulses.get_mut(element.serial) else {
            return (1.0, None);
        };
        let mut now = std::mem::take(self.spare);
        now.clear();
        formatted(self.facts.value(&pulse.watch), Format::Plain, &mut now);
        if state.last != now {
            if state.primed {
                state.since = self.now;
            }
            state.primed = true;
            std::mem::swap(&mut state.last, &mut now);
        }
        *self.spare = now;
        let t = ((self.now - state.since) / f64::from(pulse.seconds)) as f32;
        if !(0.0..1.0).contains(&t) {
            return (1.0, None);
        }
        let left = (1.0 - t) * (1.0 - t);
        (1.0 + (pulse.scale - 1.0) * left, pulse.color.as_ref())
    }

    pub(super) fn holds(&self, conds: &[Cond]) -> bool {
        conds.iter().all(|cond| {
            let value = self.facts.value(&cond.path);
            let met = match &cond.equals {
                None => value.truthy(),
                Some(wanted) => equal(value, wanted),
            };
            met != cond.negated
        })
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
            grey: element.look.grey.unwrap_or(0.0),
            shadows: element.look.shadows.as_deref().unwrap_or(&[]),
        };
        for (conds, set) in &element.styles {
            if self.holds(conds) {
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
                soft: 1.0,
                sweep: 1.0,
            },
        );
    }

    fn shadow_plates(&mut self, shadows: &[Shadow], area: Area, (radius, s): (f32, f32)) {
        for shadow in shadows {
            let Some(color) = self.color(Some(&shadow.color)) else {
                continue;
            };
            let grow = shadow.grow * s;
            let moved = Area {
                x: area.x + shadow.offset[0] * s - grow,
                y: area.y + shadow.offset[1] * s - grow,
                w: area.w + 2.0 * grow,
                h: area.h + 2.0 * grow,
            };
            self.draw.plate(
                moved,
                Plate {
                    fill: Some(color),
                    radius: radius + grow,
                    stroke: None,
                    angle_radians: 0.0,
                    soft: 1.0 + shadow.soft * s,
                    sweep: 1.0,
                },
            );
        }
    }

    fn case_of(&self, shadows: &[Shadow], s: f32) -> Option<Case> {
        if let [pair @ .., last] = shadows
            && pair.len() == 2
        {
            let case = self.case_of(pair, s)?;
            let color = self.color(Some(&last.color))?;
            let inside =
                last.offset == [0.0, 0.0] && last.soft == 0.0 && last.grow * s <= case.grow;
            return inside.then_some(Case {
                rim: Some((color, last.grow * s)),
                ..case
            });
        }
        let (first, rest) = shadows.split_first()?;
        let color = self.color(Some(&first.color))?;
        let alike = |other: &Shadow| {
            other.grow == first.grow
                && other.soft == 0.0
                && self.color(Some(&other.color)) == Some(color)
        };
        let (offset, doubled) = match rest {
            [] => (first.offset, false),
            [second] if alike(second) && second.offset == [0.0, 0.0] => (first.offset, true),
            [second] if alike(second) && first.offset == [0.0, 0.0] => (second.offset, true),
            _ => return None,
        };
        (first.soft == 0.0).then(|| Case {
            color,
            grow: first.grow * s,
            offset: offset.map(|along| along * s),
            doubled,
            rim: None,
        })
    }

    fn cased(
        &mut self,
        shadows: &[Shadow],
        area: Area,
        (fill, radius, stroke): (Option<Rgba>, f32, Option<Stroke>),
        s: f32,
    ) {
        let case = self.case_of(shadows, s);
        if case.is_none() {
            self.shadow_plates(shadows, area, (radius, s));
        }
        self.draw.cased_plate(
            area,
            Plate {
                fill,
                radius,
                stroke,
                angle_radians: 0.0,
                soft: 1.0,
                sweep: 1.0,
            },
            case,
        );
    }

    fn shadows(&mut self, element: &Element, look: &Look<'_>, area: Area, (radius, s): (f32, f32)) {
        if matches!(element.kind, Kind::Plate) {
            self.shadow_plates(look.shadows, area, (radius, s));
            return;
        }
        for shadow in look.shadows {
            let Some(color) = self.color(Some(&shadow.color)) else {
                continue;
            };
            let grow = shadow.grow * s;
            match &element.kind {
                Kind::Text(text) => {
                    let mut out = std::mem::take(self.scratch);
                    let px = self.fitted(text, area.w, element.unscaled, &mut out);
                    let placed = Area {
                        x: area.x + shadow.offset[0] * s,
                        y: area.y + shadow.offset[1] * s,
                        ..area
                    };
                    let style = TextStyle {
                        px,
                        weight: text.weight,
                        align: text.align,
                        color,
                        grow,
                        soft: shadow.soft * s,
                    };
                    self.draw.text(&out, placed, style);
                    *self.scratch = out;
                }
                Kind::Plate
                | Kind::Icon(..)
                | Kind::Arc(_)
                | Kind::Image(_)
                | Kind::Markers(_)
                | Kind::Bar { .. }
                | Kind::Pips { .. }
                | Kind::Flow { .. }
                | Kind::Reticle => {}
            }
        }
    }

    fn icon(&mut self, look: &Look<'_>, area: Area, (fit, atlas): (Fit, Option<&str>)) {
        let Some(icon) = look.icon else {
            return;
        };
        let color = self.color(look.color);
        let mut out = std::mem::take(self.scratch);
        self.render(icon, &mut out);
        match (atlas, color) {
            _ if out.is_empty() => {}
            (Some(atlas), color) => {
                self.draw
                    .sticker((atlas, &out), area, (color.unwrap_or(WHITE), look.grey));
            }
            (None, Some(color)) => {
                self.draw.icon(&out, area, (color, fit));
            }
            (None, None) => {}
        }
        *self.scratch = out;
    }

    fn arc(&mut self, arc: &Arc, area: Area, s: f32, color: Option<Rgba>) {
        let side = math::min(area.w, area.h);
        let ring = Area {
            x: area.x + (area.w - side) * 0.5,
            y: area.y + (area.h - side) * 0.5,
            w: side,
            h: side,
        };
        let width = arc.width * s;
        let band = |color: Rgba, angle_radians: f32, sweep: f32| Plate {
            fill: None,
            radius: side * 0.5,
            stroke: Some(Stroke {
                width,
                inset: 0.0,
                color,
            }),
            angle_radians,
            soft: 1.0,
            sweep,
        };
        if let Some(track) = arc.track.as_ref().and_then(|track| self.color(Some(track))) {
            self.draw.plate(ring, band(track, 0.0, 1.0));
        }
        let value = number_of(self.facts, &arc.value).unwrap_or(0.0);
        let max = number_of(self.facts, &arc.max)
            .filter(|max| *max != 0.0)
            .unwrap_or(1.0);
        let share = math::clamp(value / max, 0.0, 1.0);
        let Some(color) = color.filter(|_| share > 0.0) else {
            return;
        };
        let turn = std::f32::consts::TAU;
        let start = arc.start_deg.to_radians() - if arc.clockwise { 0.0 } else { share * turn };
        self.draw.plate(ring, band(color, start, share));
        if share >= 1.0 {
            return;
        }
        let center = ring.center();
        let reach = (side - width) * 0.5;
        for angle in [start, start + share * turn] {
            let (sin, cos) = math::sin_cos(angle);
            let at = center + Vec2::new(sin, -cos) * reach;
            let cap = Area {
                x: at.x - width * 0.5,
                y: at.y - width * 0.5,
                w: width,
                h: width,
            };
            self.plate(cap, Some(color), width * 0.5, None);
        }
    }

    pub(super) fn screen(&mut self, shown: Shown, spread_px: f32) {
        let Some(screen) = self.layout.screens.get(&shown) else {
            return;
        };
        if let Some(over) = screen.over {
            self.screen(over, spread_px);
        }
        self.stack(screen);
        for element in &screen.elements {
            self.element(element, spread_px);
        }
    }

    fn stack(&mut self, screen: &Screen) {
        self.rows.fill(RowState::default());
        for element in &screen.elements {
            let Some(row) = element.row else {
                continue;
            };
            if !self.holds(&element.when) {
                continue;
            }
            let area = self
                .draw
                .place(element.rect, element.anchor, element.unscaled);
            let (top, bottom) = (area.y, area.y + area.h);
            let Some(state) = self.rows.get_mut(row) else {
                continue;
            };
            state.span = Some(state.span.map_or((top, bottom), |(above, below)| {
                (math::min(above, top), math::max(below, bottom))
            }));
        }
        let layout = self.layout;
        let s = self.draw.scale();
        for stack in &layout.stacks {
            let edge = |(rect, anchor): ([f32; 4], Anchor)| self.draw.place(rect, anchor, false).y;
            let bounds = (stack.gap * s, edge(stack.ceiling), edge(stack.floor));
            place_rows(self.rows, stack, bounds);
        }
    }

    fn placed(&self, element: &Element) -> Option<Area> {
        let mut area = self
            .draw
            .place(element.rect, element.anchor, element.unscaled);
        if let Some(row) = element.row {
            area.y += self.rows.get(row).and_then(|state| state.lift)?;
        }
        Some(area)
    }

    pub(super) fn crossed(&self, shown: Shown, reach: f32) -> bool {
        let Some(screen) = self.layout.screens.get(&shown) else {
            return false;
        };
        let center = (self.draw.size * 0.5).round();
        screen.elements.iter().any(|element| {
            matches!(
                element.kind,
                Kind::Plate | Kind::Flow { .. } | Kind::Image(_)
            ) && self.holds(&element.when)
                && self.placed(element).is_some_and(|area| {
                    area.x < center.x + reach
                        && area.x + area.w > center.x - reach
                        && area.y < center.y + reach
                        && area.y + area.h > center.y - reach
                })
        })
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

    fn framed<'e>(
        &mut self,
        element: &'e Element,
        spread_px: f32,
    ) -> Option<(Look<'e>, Area, f32, f32)> {
        let mut area = self.placed(element)?;
        let mut look = self.look(element);
        let (zoom, flash) = self.pulse(element);
        if let Some(flash) = flash {
            look.fill = look.fill.map(|_| flash);
            look.color = look.color.map(|_| flash);
        }
        self.zoom = zoom;
        if zoom != 1.0 {
            let center = area.center();
            area.w *= zoom;
            area.h *= zoom;
            area.x = center.x - area.w * 0.5;
            area.y = center.y - area.h * 0.5;
        }
        if element.unscaled {
            let shift = self.crosshair_shift(area) * spread_px;
            area.x += shift.x.round();
            area.y += shift.y.round();
        }
        let s = zoom
            * if element.unscaled {
                1.0
            } else {
                self.draw.scale()
            };
        let sweep = element
            .sweep
            .as_ref()
            .and_then(|sweep| number_of(self.facts, sweep))
            .unwrap_or(1.0);
        Some((look, area, s, sweep))
    }

    fn element(&mut self, element: &Element, spread_px: f32) {
        if !self.holds(&element.when) {
            return;
        }
        let Some((look, area, s, sweep)) = self.framed(element, spread_px) else {
            return;
        };
        let radius = element.radius * s;
        let whole = matches!(element.kind, Kind::Plate) && sweep >= 1.0;
        if !whole {
            self.shadows(element, &look, area, (radius, s));
        }
        match &element.kind {
            Kind::Plate => {
                let fill = self.color(look.fill);
                let stroke = self.stroke(look.stroke, s);
                if whole {
                    self.cased(look.shadows, area, (fill, radius, stroke), s);
                } else {
                    self.draw.plate(
                        area,
                        Plate {
                            fill,
                            radius,
                            stroke,
                            angle_radians: 0.0,
                            soft: 1.0,
                            sweep,
                        },
                    );
                }
            }
            Kind::Text(text) => {
                let Some(color) = self.color(look.color) else {
                    return;
                };
                let mut out = std::mem::take(self.scratch);
                let px = self.fitted(text, area.w, element.unscaled, &mut out);
                self.draw.text(
                    &out,
                    area,
                    TextStyle {
                        px,
                        weight: text.weight,
                        align: text.align,
                        color,
                        grow: 0.0,
                        soft: 0.0,
                    },
                );
                *self.scratch = out;
            }
            Kind::Icon(fit, atlas) => self.icon(&look, area, (*fit, atlas.as_deref())),
            Kind::Arc(arc) => self.arc(arc, area, s, self.color(look.color)),
            Kind::Image(picture) => self.image(picture, &look, area, (radius, s)),
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
                    (radius, look.shadows, s),
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
            Kind::Reticle => self.reticle(area.center().round(), 0.0),
        }
    }

    fn image(&mut self, picture: &Picture, look: &Look<'_>, area: Area, (radius, s): (f32, f32)) {
        let Some(color) = self.color(look.color) else {
            return;
        };
        let case = self.case_of(look.shadows, s);
        if case.is_none() {
            self.shadow_plates(look.shadows, area, (radius, s));
        }
        let frame = (radius, case);
        let mut out = std::mem::take(self.scratch);
        let mut name = std::mem::take(self.spare);
        self.render(&picture.image, &mut name);
        let fit = Cover::Fit;
        match &picture.cell {
            Some((_, key)) => {
                self.render(key, &mut out);
                if !out.is_empty() {
                    self.draw
                        .image((&name, Some(&out), fit, look.grey), area, (color, frame));
                }
            }
            None if !name.is_empty() => {
                self.draw
                    .image((&name, None, fit, look.grey), area, (color, frame));
            }
            None => {}
        }
        *self.scratch = out;
        *self.spare = name;
    }

    fn bar(
        &mut self,
        area: Area,
        (radius, shadows, s): (f32, &[Shadow], f32),
        (value, max): (&Number, &Number),
        (track, fill): (Option<Rgba>, Option<Rgba>),
        dir: Dir,
    ) -> Vec2 {
        if track.is_some() {
            self.cased(shadows, area, (track, radius, None), s);
        } else {
            self.shadow_plates(shadows, area, (radius, s));
        }
        let value = number_of(self.facts, value).unwrap_or(0.0);
        let max = number_of(self.facts, max)
            .filter(|max| *max != 0.0)
            .unwrap_or(1.0);
        let share = math::clamp(value / max, 0.0, 1.0);
        let across = if share > 0.0 {
            math::min(area.w, math::max(area.w * share, area.h))
        } else {
            0.0
        };
        let filled = match dir {
            Dir::LeftToRight => Area { w: across, ..area },
            Dir::RightToLeft => Area {
                x: area.x + area.w - across,
                w: across,
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
            Dir::LeftToRight => Vec2::new(area.x + area.w * share, area.y + area.h * 0.5),
            Dir::RightToLeft => Vec2::new(area.x + area.w * (1.0 - share), area.y + area.h * 0.5),
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
        let inner = size - KNOB_INSET_PX * s;
        self.plate(disc(inner), fill, inner * 0.5, None);
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
            track,
        } = &element.kind
        else {
            return;
        };
        let count = number_of(self.facts, count).unwrap_or(0.0).max_whole();
        let value = number_of(self.facts, value).unwrap_or(0.0).max_whole();
        if count == 0 {
            return;
        }
        let pad = track.as_ref().map_or(0.0, |(_, pad)| *pad);
        let span = |k: f32| {
            (count as f32 * size[0] + count.saturating_sub(1) as f32 * gap + 2.0 * pad) * k
        };
        let k = if span(s) > area.w && area.w > 0.0 {
            s * area.w / span(s)
        } else {
            s
        };
        let (pw, ph, g, pad) = (size[0] * k, size[1] * k, gap * k, pad * k);
        let total = span(k);
        let x = match align {
            Align::Left => area.x,
            Align::Center | Align::Right => area.x + (area.w - total) * 0.5,
        };
        let tall = ph + 2.0 * pad;
        let y = area.y + (area.h - tall) * 0.5;
        if let Some(fill) = track.as_ref().and_then(|(fill, _)| self.color(Some(fill))) {
            let strip = Area {
                x,
                y,
                w: total,
                h: tall,
            };
            self.plate(strip, Some(fill), tall * 0.5, None);
        }
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
                x: x + pad + index as f32 * (pw + g),
                y: y + pad,
                w: pw,
                h: ph,
            };
            self.plate(pip, color, math::min(pw, ph) * 0.5, None);
        }
    }

    fn cap_px(&self, px: f32, weight: Weight) -> f32 {
        self.draw
            .font
            .as_ref()
            .and_then(|font| font.face(weight))
            .map_or(0.0, |face| face.cap * px / face.em_px)
    }

    fn shrunk(&self, size: f32, share: f32) -> f32 {
        let px = self.px(size, false);
        math::max(px * share, math::min(px, self.draw.min_text_px))
    }

    fn cut(&self, out: &mut String, (px, weight): (f32, Weight), room: f32) {
        let mark = self.layout.fit.mark.as_str();
        let keep = room - self.draw.measure(mark, px, weight);
        while !out.is_empty() && self.draw.measure(out, px, weight) > keep {
            out.pop();
        }
        out.truncate(out.trim_end().len());
        out.push_str(mark);
    }

    fn fitted(&mut self, text: &TextKind, room: f32, unscaled: bool, out: &mut String) -> f32 {
        self.render(&text.text, out);
        let px = self.px(text.size, unscaled);
        let wide = self.draw.measure(out, px, text.weight);
        if room <= 0.0 || wide <= room {
            return px;
        }
        let least = math::max(
            px * self.layout.fit.floor,
            math::min(px, self.draw.min_text_px),
        );
        let fitted = math::max(least, px * room / wide);
        self.draw.fitted += 1;
        if self.draw.measure(out, fitted, text.weight) > room {
            self.cut(out, (fitted, text.weight), room);
        }
        fitted
    }

    fn measured(
        &self,
        items: &[FlowItem],
        (padding, inset, gap): (f32, f32, f32),
        (room, s): (f32, f32),
        out: &mut String,
    ) -> Measured {
        let (mut fixed, mut texts, mut count) = (0.0, 0.0, 0_u32);
        let (mut first, mut last) = (None, false);
        for item in items {
            if !self.holds(&item.when) {
                continue;
            }
            let round = !matches!(item.kind, FlowKind::Text { cap: false, .. });
            first.get_or_insert(round);
            last = round;
            count += 1;
            match &item.kind {
                FlowKind::Text { text, cap, .. } => {
                    self.render(&text.text, out);
                    let wide = self
                        .draw
                        .measure(out, self.px(text.size, false), text.weight);
                    if *cap {
                        fixed += self.capped(wide, true, s);
                    } else {
                        texts += wide;
                    }
                }
                FlowKind::Icon { w, .. } => fixed += w * s,
            }
        }
        let end = |round: bool| if round { inset * s } else { padding * s };
        let lead = end(first.unwrap_or(false));
        let frame = lead + end(last) + gap * s * count.saturating_sub(1) as f32 + fixed;
        let share = if texts > 0.0 && frame + texts > room {
            math::clamp((room - frame) / texts, self.layout.fit.floor, 1.0)
        } else {
            1.0
        };
        let mut measured = Measured {
            lead,
            share,
            over: frame - room,
            content: frame,
            longest: None,
            tallest: 0.0,
        };
        for (index, item) in items.iter().enumerate() {
            if !self.holds(&item.when) {
                continue;
            }
            if let FlowKind::Text { text, cap, .. } = &item.kind {
                let px = self.shrunk(text.size, if *cap { 1.0 } else { share });
                measured.tallest = math::max(measured.tallest, self.cap_px(px, text.weight));
                if !*cap {
                    self.render(&text.text, out);
                    let wide = self.draw.measure(out, px, text.weight);
                    measured.over += wide;
                    measured.content += wide;
                    if measured.longest.is_none_or(|(_, most)| wide > most) {
                        measured.longest = Some((index, wide));
                    }
                }
            }
        }
        measured.content = math::min(measured.content, room);
        measured
    }

    fn flow(&mut self, element: &Element, look: &Look<'_>, area: Area, s: f32) {
        let Kind::Flow {
            align,
            padding,
            inset,
            gap,
            min_w,
            items,
        } = &element.kind
        else {
            return;
        };
        let mut out = std::mem::take(self.scratch);
        let fit = self.measured(items, (*padding, *inset, *gap), (area.w, s), &mut out);
        let width = math::min(area.w, math::max(fit.content, min_w * s));
        let x = match align {
            Align::Left => area.x,
            Align::Center => area.x + (area.w - width) * 0.5,
            Align::Right => area.x + area.w - width,
        };
        let fill = self.color(look.fill);
        let stroke = self.stroke(look.stroke, s);
        let pill = Area {
            x,
            w: width,
            ..area
        };
        self.cased(look.shadows, pill, (fill, element.radius * s, stroke), s);
        let mut at = x + fit.lead + (width - fit.content) * 0.5;
        for (index, item) in items.iter().enumerate() {
            if !self.holds(&item.when) {
                continue;
            }
            match &item.kind {
                FlowKind::Text { text, color, cap } => {
                    self.render(&text.text, &mut out);
                    let px = self.shrunk(text.size, if *cap { 1.0 } else { fit.share });
                    let mut wide = self.draw.measure(&out, px, text.weight);
                    if fit.over > 0.0 && fit.longest.is_some_and(|(most, _)| most == index) {
                        self.cut(&mut out, (px, text.weight), wide - fit.over);
                        wide = self.draw.measure(&out, px, text.weight);
                    }
                    let room = self.capped(wide, *cap, s);
                    let place = Area {
                        x: at,
                        w: room,
                        ..area
                    };
                    if *cap {
                        self.key_cap(place, s);
                    }
                    if let Some(color) = self.color(Some(color)) {
                        let style = TextStyle {
                            px,
                            weight: text.weight,
                            align: Align::Center,
                            color,
                            grow: 0.0,
                            soft: 0.0,
                        };
                        let line = Area {
                            y: area.y + (fit.tallest - self.cap_px(px, text.weight)) * 0.5,
                            ..place
                        };
                        self.draw.text(&out, line, style);
                    }
                    at += room + gap * s;
                }
                FlowKind::Icon {
                    icon,
                    w,
                    h,
                    color,
                    atlas,
                } => {
                    let (iw, ih) = (w * s, h * s);
                    self.render(icon, &mut out);
                    let place = Area {
                        x: at,
                        y: area.y + (area.h - ih) * 0.5,
                        w: iw,
                        h: ih,
                    };
                    self.flow_icon((&out, atlas.as_deref()), place, color.as_ref());
                    at += iw + gap * s;
                }
            }
        }
        *self.scratch = out;
    }

    fn flow_icon(
        &mut self,
        (name, atlas): (&str, Option<&str>),
        place: Area,
        color: Option<&Paint>,
    ) {
        let tint = self.color(color);
        match (atlas, tint) {
            _ if name.is_empty() => {}
            (Some(atlas), tint) => {
                self.draw
                    .sticker((atlas, name), place, (tint.unwrap_or(WHITE), 0.0));
            }
            (None, Some(tint)) => {
                self.draw.icon(name, place, (tint, Fit::Cell));
            }
            (None, None) => {}
        }
    }

    fn capped(&self, wide: f32, cap: bool, s: f32) -> f32 {
        match self.layout.key_cap.as_ref().filter(|_| cap) {
            Some(piece) => math::max(piece.height * s, wide + 2.0 * piece.padding_x * s),
            None => wide,
        }
    }

    fn key_cap(&mut self, place: Area, s: f32) {
        let layout = self.layout;
        let Some(piece) = &layout.key_cap else {
            return;
        };
        let h = piece.height * s;
        let cap = Area {
            y: place.y + (place.h - h) * 0.5,
            h,
            ..place
        };
        let fill = self.color(Some(&piece.fill));
        self.cased(&piece.shadows, cap, (fill, piece.radius * s, None), s);
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
            let kept = 1.0 - math::clamp(item.fade, 0.0, 1.0);
            let tint = item
                .color
                .and_then(|tone| self.layout.tone(tone))
                .or(color)
                .map(|tint| tint.faded(kept));
            if markers.disc {
                let wide = 2.0 * item.radius_m * map.px_per_m * area.w / map.size[0];
                let disc = Area {
                    x: center.x - wide * 0.5,
                    y: center.y - wide * 0.5,
                    w: wide,
                    h: wide,
                };
                self.plate(disc, tint, wide * 0.5, None);
                continue;
            }
            let icon = if item.icon.is_empty() {
                markers.icon.as_str()
            } else {
                item.icon.as_str()
            };
            if let Some(atlas) = &markers.atlas {
                if let Some([w, h]) = self.draw.sticker_size(atlas, icon) {
                    let scale = size / math::max(math::max(w, h), 1.0);
                    self.draw.sticker_turned(
                        (atlas, icon),
                        center,
                        (scale, angle),
                        WHITE.faded(kept),
                    );
                }
                continue;
            }
            if let Some((fill, wide)) = backing {
                let disc = Area {
                    x: center.x - wide * 0.5,
                    y: center.y - wide * 0.5,
                    w: wide,
                    h: wide,
                };
                self.plate(disc, Some(fill), wide * 0.5, None);
            }
            if let Some((ink, rim)) = outline {
                self.draw
                    .marker(rim, center, size, (ink.faded(kept), angle));
            }
            if let Some(tint) = tint {
                self.draw.marker(icon, center, size, (tint, angle));
            }
        }
    }

    pub(super) fn blind(&mut self, amount: f32) {
        if amount <= 0.0 {
            return;
        }
        let size = self.draw.size;
        let whole = Area {
            x: 0.0,
            y: 0.0,
            w: size.x,
            h: size.y,
        };
        self.plate(whole, Some(WHITE.faded(math::min(amount, 1.0))), 0.0, None);
    }

    pub(super) fn scope(&mut self, scope: &ScopeNumbers, ease: f32) {
        let (Some(color), Some(shade)) = (self.palette(&scope.color), self.palette(&scope.shade))
        else {
            return;
        };
        let ease = math::clamp(ease, 0.0, 1.0);
        let center = (self.draw.size * 0.5).round();
        let clear = self.draw.size.y * scope.ring_share * 0.5 * (1.0 + SCOPE_OPEN * (1.0 - ease));
        let outer = self.draw.size.length() * 0.5 + 2.0;
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
        self.plate(whole, None, outer, Some(surround));
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
        self.plate(ring, None, clear, Some(stroke));
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
        self.plate(across, Some(color.faded(ease)), 0.0, None);
        self.plate(down, Some(color.faded(ease)), 0.0, None);
    }

    fn palette(&self, key: &str) -> Option<Rgba> {
        self.layout.palette.get(key).copied()
    }

    pub(super) fn marks(&mut self, marks: &Marks) {
        let center = (self.draw.size * 0.5).round();
        let ink = self.palette(&marks.outline_color);
        if let (Some(age), Some(atlas)) = (self.facts.armor_break, self.layout.stickers.clone()) {
            let numbers = &marks.armor;
            let s = self.draw.scale();
            let left = 1.0 - age / numbers.seconds;
            let pop =
                1.0 + ARMOR_POP * math::max(0.0, 1.0 - age / (numbers.seconds * ARMOR_POP_SHARE));
            let at = center + Vec2::new(numbers.offset_px[0], numbers.offset_px[1]) * s;
            let size = numbers.size_px * s * pop;
            if let Some([w, h]) = self.draw.sticker_size(&atlas, &numbers.icon) {
                self.draw.sticker_turned(
                    (&atlas, &numbers.icon),
                    at,
                    (size / math::max(w, h), 0.0),
                    WHITE.faded(left),
                );
            }
        }
        self.damage_marks(ink);
    }

    fn damage_marks(&mut self, ink: Option<Rgba>) {
        let Some(edge) = self.layout.damage_edge.as_ref() else {
            return;
        };
        let s = self.draw.scale();
        let center = self.draw.size * 0.5;
        let radius = edge.radius_share * self.draw.size.y;
        if radius <= 0.0 || edge.fade_seconds <= 0.0 {
            return;
        }
        let span = edge.length_deg.to_radians();
        let (band, rim) = (edge.band_px * s, edge.rim_px * s);
        let facts = self.facts;
        for mark in &facts.damage {
            let t = math::clamp(mark.age / edge.fade_seconds, 0.0, 1.0);
            let left = (1.0 - t) * (1.0 - t);
            let toward = -mark.angle_radians;
            let ends: [Vec2; ARC_PIECES + 1] = std::array::from_fn(|at| {
                let along = at as f32 / ARC_PIECES as f32 - 0.5;
                let (sin, cos) = math::sin_cos(toward + along * span);
                center + Vec2::new(sin, -cos) * radius
            });
            for (paint, thick) in [(ink, band + 2.0 * rim), (Some(edge.color), band)] {
                let Some(paint) = paint else {
                    continue;
                };
                for pair in ends.windows(2) {
                    let step = pair[1] - pair[0];
                    let angle = math::atan2(step.y, step.x);
                    self.round_bar(
                        (pair[0] + pair[1]) * 0.5,
                        (step.length() + thick, thick),
                        (angle, paint.faded(left)),
                    );
                }
            }
            let (sin, cos) = math::sin_cos(toward);
            self.draw.sticker_turned(
                (&edge.atlas, &edge.tip),
                center + Vec2::new(sin, -cos) * radius,
                (edge.tip_scale * s, toward),
                WHITE.faded(left),
            );
        }
    }

    fn round_bar(&mut self, at: Vec2, (w, h): (f32, f32), (angle, color): (f32, Rgba)) {
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
                soft: 1.0,
                sweep: 1.0,
            },
        );
    }

    pub(super) fn reticle(&mut self, center: Vec2, spread_px: f32) {
        let settings = &self.facts.menu.settings;
        let reticle: Reticle = settings.reticle;
        let Some(color) = self.palette(&settings.crosshair_color) else {
            return;
        };
        let ink = self.layout.crosshair.map(|piece| piece.outline);
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
            self.draw.cased_plate(
                Area { x, y, w, h },
                Plate {
                    fill: Some(color),
                    radius: 0.0,
                    stroke: None,
                    angle_radians: 0.0,
                    soft: 1.0,
                    sweep: 1.0,
                },
                case,
            );
        }
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
    if let Some(grey) = set.grey {
        look.grey = grey;
    }
    if let Some(shadows) = &set.shadows {
        look.shadows = shadows;
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
