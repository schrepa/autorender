use std::collections::BTreeMap;
use std::fmt::Write;

use engine::math;
use glam::Vec2;

use super::draw::{Align, Anchor, Area, Case, InterfaceDraw, Plate, Rgba, Stroke, TextStyle, Weight};
use super::layout::{
    Arc, Cond, Element, FlowItem, FlowKind, Format, Kind, Knob, Layout, Markers, Number, Paint,
    Picture, Piece, Screen, Set, Shadow, Stack, Template, TextKind, Want,
};
use super::value::Val;
use super::{Facts, MapFrame, Point};

const WHITE: Rgba = Rgba([1.0; 4]);

pub(super) struct Painter<'a, F: Facts> {
    pub draw: &'a mut InterfaceDraw,
    pub facts: &'a F,
    pub look: &'a F::Look,
    pub layout: &'a Layout,
    pub maps: &'a BTreeMap<String, MapFrame>,
    pub elements: &'a [Element],
    pub scratch: &'a mut String,
    pub spare: &'a mut String,
    pub pulses: &'a mut [PulseState],
    pub rows: &'a mut [RowState],
    pub said: &'a mut Vec<Said>,
    pub pulsing: &'a mut bool,
    pub now: f64,
    pub zoom: f32,
}

#[derive(Clone, Debug, Default)]
pub(super) struct Said {
    held: bool,
    text: String,
    advance: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RowState {
    span: Option<(f32, f32)>,
    lift: Option<f32>,
}

pub struct Pen<'a> {
    draw: &'a mut InterfaceDraw,
    layout: &'a Layout,
    rows: &'a [RowState],
    elements: &'a [Element],
    holds: &'a dyn Fn(&[Cond]) -> bool,
}

impl<'a> Pen<'a> {
    pub fn size(&self) -> Vec2 {
        self.draw.size
    }

    pub fn scale(&self) -> f32 {
        self.draw.scale()
    }

    pub fn palette(&self, name: &str) -> Option<Rgba> {
        self.layout.palette.get(name).copied()
    }

    pub fn atlas(&self, name: &str) -> Option<&'a str> {
        self.layout.atlases.get(name).map(String::as_str)
    }

    pub fn plate(&mut self, area: Area, plate: Plate, case: Option<Case>) {
        self.draw.cased_plate(area, plate, case);
    }

    pub fn cell_size(&self, atlas: &str, cell: &str) -> Option<[f32; 2]> {
        self.draw.cell_size(atlas, cell)
    }

    pub fn cell_turned(&mut self, cell: (&str, &str), center: Vec2, turn: (f32, f32), color: Rgba) {
        self.draw.cell_turned(cell, center, turn, color);
    }

    pub fn covered(&self, reach: f32) -> bool {
        let center = (self.draw.size * 0.5).round();
        self.elements.iter().any(|element| {
            matches!(
                element.kind,
                Kind::Plate | Kind::Flow { .. } | Kind::Image(_)
            ) && (self.holds)(&element.when)
                && placed(self.draw, self.rows, element).is_some_and(|area| {
                    area.x < center.x + reach
                        && area.x + area.w > center.x - reach
                        && area.y < center.y + reach
                        && area.y + area.h > center.y - reach
                })
        })
    }
}

fn placed(draw: &InterfaceDraw, rows: &[RowState], element: &Element) -> Option<Area> {
    let mut area = draw.place(element.rect, element.anchor);
    if let Some(row) = element.row {
        area.y += rows.get(row).and_then(|state| state.lift)?;
    }
    Some(area)
}

fn met<F: Facts>(facts: &F, conds: &[Cond]) -> bool {
    conds.iter().all(|cond| {
        let value = facts.read(cond.slot);
        let met = match &cond.equals {
            None => value.truthy(),
            Some(wanted) => equal(value, wanted),
        };
        met != cond.negated
    })
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

#[derive(Clone, Debug, PartialEq)]
enum Seen {
    Num(f32),
    Text(String),
}

impl Default for Seen {
    fn default() -> Self {
        Seen::Text(String::new())
    }
}

impl Seen {
    fn moved(&mut self, value: Val<'_>) -> bool {
        let same = match (&*self, value) {
            (Seen::Num(held), Val::Num(number)) => *held == number,
            (Seen::Text(held), Val::Text(text)) => held.as_str() == text,
            (Seen::Text(held), Val::Unknown | Val::Missing | Val::Bool(_)) => held.is_empty(),
            (Seen::Num(_), Val::Text(_) | Val::Unknown | Val::Missing | Val::Bool(_))
            | (Seen::Text(_), Val::Num(_)) => false,
        };
        if !same {
            *self = match value {
                Val::Num(number) => Seen::Num(number),
                Val::Text(text) => Seen::Text(text.to_owned()),
                Val::Unknown | Val::Missing | Val::Bool(_) => Seen::default(),
            };
        }
        !same
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct PulseState {
    last: Seen,
    primed: bool,
    since: f64,
}

struct Look<'a> {
    fill: Option<&'a Paint>,
    color: Option<&'a Paint>,
    icon: Option<&'a Template>,
    grey: f32,
    shadows: &'a [Shadow],
}

fn equal(value: Val<'_>, wanted: &Want) -> bool {
    match value {
        Val::Text(text) => text == wanted.text,
        Val::Num(number) => wanted.number == Some(number),
        Val::Bool(flag) => wanted.text == if flag { "true" } else { "false" },
        Val::Unknown | Val::Missing => wanted.text.is_empty(),
    }
}

fn whole_number(value: f32, out: &mut String) {
    let _ = write!(out, "{}", value as i64);
}

pub fn thousands(value: i64, out: &mut String) {
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
            Format::Plain | Format::Int | Format::Clock | Format::Thousands | Format::Tenths,
        ) => {
            out.push_str(text);
        }
        (Val::Num(number), Format::Int) => whole_number(number.ceil(), out),
        (Val::Num(number), Format::Clock) => {
            let seconds = math::max(number.ceil(), 0.0) as i64;
            let _ = write!(out, "{}:{:02}", seconds / 60, seconds % 60);
        }
        (Val::Num(number), Format::Thousands) => thousands(number as i64, out),
        (Val::Num(number), Format::Tenths) => {
            let _ = write!(out, "{number:.1}");
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

fn number_of<F: Facts>(facts: &F, number: &Number) -> Option<f32> {
    match number {
        Number::Fixed(value) => Some(*value),
        Number::Read(slot) => facts.read(*slot).number(),
    }
}

impl<F: Facts> Painter<'_, F> {
    fn px(&self, size: f32) -> f32 {
        self.draw.text_px(size) * self.zoom
    }

    fn pulse<'e>(&mut self, element: &'e Element) -> (f32, Option<&'e Paint>) {
        let Some(pulse) = &element.pulse else {
            return (1.0, None);
        };
        let Some(state) = self.pulses.get_mut(element.serial) else {
            return (1.0, None);
        };
        if state.last.moved(self.facts.read(pulse.watch)) {
            if state.primed {
                state.since = self.now;
            }
            state.primed = true;
        }
        let t = ((self.now - state.since) / f64::from(pulse.seconds)) as f32;
        if !(0.0..1.0).contains(&t) {
            return (1.0, None);
        }
        *self.pulsing = true;
        let left = (1.0 - t) * (1.0 - t);
        (1.0 + (pulse.scale - 1.0) * left, pulse.color.as_ref())
    }

    fn holds(&self, conds: &[Cond]) -> bool {
        met(self.facts, conds)
    }

    fn color(&self, paint: Option<&Paint>) -> Option<Rgba> {
        match paint? {
            Paint::None => None,
            Paint::Fixed(color) => Some(*color),
            Paint::Read(slot) => self
                .facts
                .read(*slot)
                .text()
                .and_then(|key| self.layout.palette.get(key).copied()),
        }
    }

    fn look<'e>(&self, element: &'e Element) -> Look<'e> {
        let mut look = Look {
            fill: element.look.fill.as_ref(),
            color: element.look.color.as_ref(),
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
                Piece::Value(slot, format) => formatted(self.facts.read(*slot), *format, out),
            }
        }
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
        (fill, radius): (Option<Rgba>, f32),
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
                stroke: None,
                angle_radians: 0.0,
                soft: 1.0,
                sweep: 1.0,
            },
            case,
        );
    }

    fn icon(&mut self, look: &Look<'_>, area: Area, atlas: &str) {
        let Some(icon) = look.icon else {
            return;
        };
        let color = self.color(look.color);
        let mut out = std::mem::take(self.scratch);
        self.render(icon, &mut out);
        if !out.is_empty() {
            self.draw
                .cell((atlas, &out), area, (color.unwrap_or(WHITE), look.grey));
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
        let start = arc.start_deg.to_radians();
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

    pub(super) fn screen(&mut self, shown: usize) {
        let Some(screen) = self.layout.screens.get(shown) else {
            return;
        };
        self.elements = &screen.elements;
        self.stack(screen);
        for element in &screen.elements {
            self.element(element);
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
            let area = self.draw.place(element.rect, element.anchor);
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
            let edge = |(rect, anchor): ([f32; 4], Anchor)| self.draw.place(rect, anchor).y;
            let bounds = (stack.gap * s, edge(stack.ceiling), edge(stack.floor));
            place_rows(self.rows, stack, bounds);
        }
    }

    fn framed<'e>(&mut self, element: &'e Element) -> Option<(Look<'e>, Area, f32)> {
        let mut area = placed(self.draw, self.rows, element)?;
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
        let s = zoom * self.draw.scale();
        Some((look, area, s))
    }

    fn element(&mut self, element: &Element) {
        if !self.holds(&element.when) {
            return;
        }
        let Some((look, area, s)) = self.framed(element) else {
            return;
        };
        let radius = element.radius * s;
        match &element.kind {
            Kind::Plate => {
                let fill = self.color(look.fill);
                self.cased(look.shadows, area, (fill, radius), s);
            }
            Kind::Text(text) => {
                let mut out = std::mem::take(self.scratch);
                let (px, wide) = self.fitted(text, area.w, &mut out);
                for shadow in look.shadows {
                    let Some(color) = self.color(Some(&shadow.color)) else {
                        continue;
                    };
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
                        grow: shadow.grow * s,
                        soft: shadow.soft * s,
                    };
                    self.draw.text((&out, wide), placed, style);
                }
                if let Some(color) = self.color(look.color) {
                    let style = TextStyle {
                        px,
                        weight: text.weight,
                        align: text.align,
                        color,
                        grow: 0.0,
                        soft: 0.0,
                    };
                    self.draw.text((&out, wide), area, style);
                }
                *self.scratch = out;
            }
            Kind::Icon(atlas) => self.icon(&look, area, atlas),
            Kind::Arc(arc) => self.arc(arc, area, s, self.color(look.color)),
            Kind::Image(picture) => self.image(picture, &look, area, (radius, s)),
            Kind::Markers(markers) => self.markers(markers, area, s, self.color(look.color)),
            Kind::Bar {
                value,
                max,
                track,
                knob,
            } => {
                let end = self.bar(
                    area,
                    (radius, look.shadows, s),
                    (value, max),
                    (self.color(Some(track)), self.color(look.fill)),
                );
                if let Some(knob) = knob {
                    self.knob(knob, end, s);
                }
            }
            Kind::Pips { .. } => self.pips(element, area, s),
            Kind::Flow { .. } => self.flow(element, &look, area, s),
            Kind::Piece(name) => {
                let facts = self.facts;
                let holds = |conds: &[Cond]| met(facts, conds);
                let mut pen = Pen {
                    draw: &mut *self.draw,
                    layout: self.layout,
                    rows: &*self.rows,
                    elements: self.elements,
                    holds: &holds,
                };
                facts.piece(self.look, *name, area, &mut pen);
            }
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
        match &picture.cell {
            Some((_, key)) => {
                self.render(key, &mut out);
                if !out.is_empty() {
                    self.draw
                        .image((&name, Some(&out), look.grey), area, (color, frame));
                }
            }
            None if !name.is_empty() => {
                self.draw
                    .image((&name, None, look.grey), area, (color, frame));
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
    ) -> Vec2 {
        if track.is_some() {
            self.cased(shadows, area, (track, radius), s);
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
        let filled = Area { w: across, ..area };
        let rounded = math::min(radius, math::min(filled.w, filled.h) * 0.5);
        self.plate(filled, fill, rounded, None);
        Vec2::new(area.x + area.w * share, area.y + area.h * 0.5)
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
            let wide = size + knob.ring_px * s;
            self.plate(disc(wide), Some(ring), wide * 0.5, None);
        }
        let fill = self.color(Some(&knob.fill));
        let inner = size - knob.inset_px * s;
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
        let px = self.px(size);
        math::max(px * share, math::min(px, self.draw.min_text_px))
    }

    fn cut(&self, out: &mut String, (px, weight): (f32, Weight), room: f32) -> f32 {
        let mark = self.layout.fit.mark.as_str();
        let keep = room - self.draw.measure(mark, px, weight);
        out.truncate(self.draw.kept(out, (px, weight), keep));
        out.truncate(out.trim_end().len());
        out.push_str(mark);
        self.draw.measure(out, px, weight)
    }

    fn fitted(&mut self, text: &TextKind, room: f32, out: &mut String) -> (f32, f32) {
        self.render(&text.text, out);
        let px = self.px(text.size);
        let advance = self.draw.advance(out, text.weight);
        let wide = self.draw.width(advance, px, text.weight);
        if room <= 0.0 || wide <= room {
            return (px, wide);
        }
        let least = math::max(
            px * self.layout.fit.floor,
            math::min(px, self.draw.min_text_px),
        );
        let fitted = math::max(least, px * room / wide);
        self.draw.fitted += 1;
        let narrow = self.draw.width(advance, fitted, text.weight);
        if narrow > room {
            return (fitted, self.cut(out, (fitted, text.weight), room));
        }
        (fitted, narrow)
    }

    fn say(&self, items: &[FlowItem], said: &mut Vec<Said>) {
        said.resize_with(items.len(), Said::default);
        for (item, one) in items.iter().zip(said.iter_mut()) {
            one.held = self.holds(&item.when);
            if let (true, FlowKind::Text { text, .. }) = (one.held, &item.kind) {
                self.render(&text.text, &mut one.text);
                one.advance = self.draw.advance(&one.text, text.weight);
            }
        }
    }

    fn measured(
        &self,
        (items, said): (&[FlowItem], &[Said]),
        (padding, inset, gap): (f32, f32, f32),
        (room, s): (f32, f32),
    ) -> Measured {
        let (mut fixed, mut texts, mut count) = (0.0, 0.0, 0_u32);
        let (mut first, mut last) = (None, false);
        for (item, one) in items.iter().zip(said).filter(|(_, one)| one.held) {
            let round = !matches!(item.kind, FlowKind::Text { cap: false, .. });
            first.get_or_insert(round);
            last = round;
            count += 1;
            match &item.kind {
                FlowKind::Text { text, cap, .. } => {
                    let wide = self.draw.width(one.advance, self.px(text.size), text.weight);
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
        for (index, (item, one)) in items.iter().zip(said).enumerate() {
            if !one.held {
                continue;
            }
            if let FlowKind::Text { text, cap, .. } = &item.kind {
                let px = self.shrunk(text.size, if *cap { 1.0 } else { share });
                measured.tallest = math::max(measured.tallest, self.cap_px(px, text.weight));
                if !*cap {
                    let wide = self.draw.width(one.advance, px, text.weight);
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
        let mut said = std::mem::take(self.said);
        self.say(items, &mut said);
        let fit = self.measured((items, &said), (*padding, *inset, *gap), (area.w, s));
        let width = math::min(area.w, math::max(fit.content, min_w * s));
        let x = match align {
            Align::Left => area.x,
            Align::Center => area.x + (area.w - width) * 0.5,
            Align::Right => area.x + area.w - width,
        };
        let fill = self.color(look.fill);
        let pill = Area {
            x,
            w: width,
            ..area
        };
        self.cased(look.shadows, pill, (fill, element.radius * s), s);
        let mut out = std::mem::take(self.scratch);
        let mut at = x + fit.lead + (width - fit.content) * 0.5;
        for (index, (item, one)) in items.iter().zip(said.iter_mut()).enumerate() {
            if !one.held {
                continue;
            }
            match &item.kind {
                FlowKind::Text { text, color, cap } => {
                    let px = self.shrunk(text.size, if *cap { 1.0 } else { fit.share });
                    let mut wide = self.draw.width(one.advance, px, text.weight);
                    if fit.over > 0.0 && fit.longest.is_some_and(|(most, _)| most == index) {
                        wide = self.cut(&mut one.text, (px, text.weight), wide - fit.over);
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
                        self.draw.text((&one.text, wide), line, style);
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
                    self.flow_icon((&out, atlas), place, color.as_ref());
                    at += iw + gap * s;
                }
            }
        }
        *self.scratch = out;
        *self.said = said;
    }

    fn flow_icon(&mut self, (name, atlas): (&str, &str), place: Area, color: Option<&Paint>) {
        if name.is_empty() {
            return;
        }
        let tint = self.color(color).unwrap_or(WHITE);
        self.draw.cell((atlas, name), place, (tint, 0.0));
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
        self.cased(&piece.shadows, cap, (fill, piece.radius * s), s);
    }

    fn markers(&mut self, markers: &Markers, area: Area, s: f32, color: Option<Rgba>) {
        let (facts, maps) = (self.facts, self.maps);
        let Some(map) = maps.get(&markers.map) else {
            return;
        };
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
        facts.points(markers.items, &mut |item: Point<'_>| {
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
                .and_then(|key| self.layout.palette.get(key).copied())
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
                return;
            }
            let icon = if item.icon.is_empty() {
                markers.icon.as_str()
            } else {
                item.icon
            };
            if let Some(atlas) = &markers.atlas {
                if let Some([w, h]) = self.draw.cell_size(atlas, icon) {
                    let scale = size / math::max(math::max(w, h), 1.0);
                    self.draw.cell_turned(
                        (atlas, icon),
                        center,
                        (scale, angle),
                        WHITE.faded(kept),
                    );
                }
                return;
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
        });
    }
}

fn apply<'e>(look: &mut Look<'e>, set: &'e Set) {
    if let Some(fill) = &set.fill {
        look.fill = Some(fill);
    }
    if let Some(color) = &set.color {
        look.color = Some(color);
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
