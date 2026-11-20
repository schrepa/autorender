use std::collections::BTreeMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use engine::math;
use engine::prelude::Resource;
use engine_core::{Asset, Texture};
use glam::Vec2;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::Slot;
use super::layout::{Element, Layout};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Area {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Area {
    pub fn contains(&self, point: Vec2) -> bool {
        point.x >= self.x
            && point.y >= self.y
            && point.x < self.x + self.w
            && point.y < self.y + self.h
    }

    pub fn center(&self) -> Vec2 {
        Vec2::new(self.x + self.w * 0.5, self.y + self.h * 0.5)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgba(pub [f32; 4]);

impl Rgba {
    #[must_use]
    pub fn faded(self, by: f32) -> Rgba {
        let [r, g, b, a] = self.0;
        Rgba([r, g, b, a * math::clamp(by, 0.0, 1.0)])
    }

    pub fn packed(self) -> u32 {
        let byte = |value: f32| (math::clamp(value, 0.0, 1.0) * 255.0 + 0.5) as u32;
        let [r, g, b, a] = self.0;
        byte(r) | (byte(g) << 8) | (byte(b) << 16) | (byte(a) << 24)
    }

    pub fn visible(self) -> bool {
        self.0[3] > 0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stroke {
    pub width: f32,
    pub inset: f32,
    pub color: Rgba,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Case {
    pub color: Rgba,
    pub grow: f32,
    pub offset: [f32; 2],
    pub doubled: bool,
    pub rim: Option<(Rgba, f32)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plate {
    pub fill: Option<Rgba>,
    pub radius: f32,
    pub stroke: Option<Stroke>,
    pub angle_radians: f32,
    pub soft: f32,
    pub sweep: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Weight(pub usize);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Align {
    #[default]
    Left,
    Center,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextStyle {
    pub px: f32,
    pub weight: Weight,
    pub align: Align,
    pub color: Rgba,
    pub grow: f32,
    pub soft: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Anchor {
    At([f32; 2]),
    Stretch,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Glyph {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub offset_x: f32,
    pub offset_y: f32,
    pub advance: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Thin {
    pub below_px: f32,
    pub span_px: f32,
    pub boost: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Face {
    pub line_height: f32,
    pub ascent: f32,
    pub descent: f32,
    pub cap: f32,
    pub em_px: f32,
    pub spread_px: f32,
    pub slot: u32,
    pub thin: Option<Thin>,
    pub glyphs: Vec<Option<Glyph>>,
}

impl Face {
    pub fn glyph(&self, character: char) -> Option<&Glyph> {
        self.glyphs.get(character as usize)?.as_ref()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Font {
    pub faces: Vec<Option<Face>>,
}

impl Font {
    pub fn face(&self, weight: Weight) -> Option<&Face> {
        self.faces.get(weight.0)?.as_ref()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Icons {
    pub cells: BTreeMap<String, [f32; 4]>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Picture {
    pub name: String,
    pub atlas: String,
    pub size: [f32; 2],
    pub cells: BTreeMap<String, [f32; 4]>,
}

pub const GLYPH_SLOTS: usize = 2;
pub const PICTURE_SLOTS: usize = 12;
pub const BOUND_SLOTS: usize = 1 + GLYPH_SLOTS + PICTURE_SLOTS;

pub const PLATE: u32 = 0;
pub const ICON: u32 = 1;
pub const GLYPH: u32 = 2;
pub const IMAGE: u32 = 3;

pub const GLYPH_MARGIN_PX: f32 = 1.5;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Quad {
    pub rect: [f32; 4],
    pub uv: [f32; 4],
    pub fill: u32,
    pub edge: u32,
    pub mode: u32,
    pub radius: f32,
    pub stroke_width: f32,
    pub stroke_inset: f32,
    pub scale: f32,
    pub angle: f32,
    pub case: u32,
    pub case_grow: f32,
    pub case_offset: [f32; 2],
    pub rim: u32,
    pub rim_grow: f32,
    pub align: [u32; 2],
}

impl Quad {
    pub fn covered(&self) -> f32 {
        let [_, _, w, h] = self.rect;
        let framed = self.mode == IMAGE && (self.case != 0 || self.radius > 0.0);
        if self.mode != PLATE && !framed {
            return w * h;
        }
        let pad = 2.0 * PLATE_PAD * if framed { 1.0 } else { self.scale };
        let (mut wide, mut tall) = (w, h);
        if self.case != 0 {
            let doubled = if framed {
                self.stroke_inset > 0.5
            } else {
                self.uv[1] > 0.5
            };
            let [x, y] = self.case_offset;
            wide = span(w, self.case_grow, (x, doubled));
            tall = span(h, self.case_grow, (y, doubled));
        }
        (wide + pad) * (tall + pad)
    }
}

pub const PLATE_PAD: f32 = 1.0;

fn plain(quad: &Quad) -> bool {
    quad.mode == PLATE
        && quad.case == 0
        && quad.stroke_width == 0.0
        && quad.radius == 0.0
        && quad.angle == 0.0
        && quad.uv[0] == 0.0
}

fn covers([x, y, w, h]: [f32; 4], [cx, cy, cw, ch]: [f32; 4]) -> bool {
    x <= cx && y <= cy && x + w >= cx + cw && y + h >= cy + ch
}

fn opaque(color: u32) -> bool {
    color >> 24 == 0xFF
}

fn area_of([left, top, right, low]: [f32; 4]) -> f32 {
    (right - left) * (low - top)
}

fn inside(quad: &Quad) -> Option<[f32; 4]> {
    if quad.mode != PLATE || !opaque(quad.fill) || quad.angle != 0.0 || quad.uv[0] != 0.0 {
        return None;
    }
    let [x, y, w, h] = quad.rect;
    let r = math::clamp(quad.radius, 0.0, math::min(w, h) * 0.5);
    let margin = math::max(quad.scale, 1.0);
    let (across, down) = if w - 2.0 * r >= h - 2.0 * r {
        (r + margin, margin)
    } else {
        (margin, r + margin)
    };
    let box_ = [
        (x + across).ceil(),
        (y + down).ceil(),
        (x + w - across).floor(),
        (y + h - down).floor(),
    ];
    (box_[2] > box_[0] && box_[3] > box_[1]).then_some(box_)
}

fn over(top: u32, below: u32) -> u32 {
    let unpack = |color: u32| [0, 8, 16, 24].map(|shift| ((color >> shift) & 0xFF) as f32 / 255.0);
    let [tr, tg, tb, ta] = unpack(top);
    let [br, bg, bb, ba] = unpack(below);
    let a = ta + ba * (1.0 - ta);
    if a <= 0.0 {
        return 0;
    }
    let mix = |t: f32, b: f32| (t * ta + b * ba * (1.0 - ta)) / a;
    Rgba([mix(tr, br), mix(tg, bg), mix(tb, bb), a]).packed()
}

fn span(size: f32, grow: f32, (offset, doubled): (f32, bool)) -> f32 {
    let half = size * 0.5;
    let (far, near) = if doubled {
        (math::max(offset, 0.0), math::min(offset, 0.0))
    } else {
        (offset, offset)
    };
    math::max(half, half + grow + far) + math::max(half, half + grow - near)
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct InterfaceDraw {
    pub(super) size: Vec2,
    pub(super) reference: Vec2,
    pub(super) min_text_px: f32,
    pub(super) sharpen_levels: f32,
    pub(super) font: Option<Font>,
    pub(super) icons: Icons,
    pub(super) pictures: Vec<Picture>,
    pub(super) atlases: Option<[Asset<Texture>; BOUND_SLOTS]>,
    pub(super) quads: Vec<Quad>,
    pub(super) first: usize,
    pub(super) floor: Vec<Quad>,
    pub(super) held: Option<Quad>,
    pub(super) revision: u64,
    pub(super) texts: u32,
    pub(super) glyphs: u32,
    pub(super) glyphs_missing: u32,
    pub(super) icons_missing: u32,
    pub(super) fitted: u32,
    pub(super) opaque: bool,
    pub(super) floor_quad: Option<Quad>,
    pub(super) layout: Option<Arc<Layout>>,
}

impl InterfaceDraw {
    pub fn element(&self, screen: usize, spot: Slot) -> Option<Area> {
        let screen = self.layout.as_deref()?.screens.get(screen)?;
        let element = screen
            .elements
            .iter()
            .find(|element| element.spot == Some(spot))?;
        Some(self.place(element.rect, element.anchor))
    }

    pub fn under(&self, screen: usize, point: Vec2) -> impl Iterator<Item = &Element> {
        let screen = self.layout.as_deref().and_then(|layout| layout.screens.get(screen));
        screen
            .into_iter()
            .flat_map(|screen| screen.elements.iter().rev())
            .filter(move |element| self.place(element.rect, element.anchor).contains(point))
    }

    pub(super) fn begin(&mut self, size: Vec2) {
        self.size = size;
        self.quads.clear();
        self.first = 0;
        self.floor.clear();
        self.texts = 0;
        self.glyphs = 0;
        self.glyphs_missing = 0;
        self.icons_missing = 0;
        self.fitted = 0;
        self.opaque = false;
    }

    pub(super) fn settle(&mut self) {
        let canvas = [0.0, 0.0, self.size.x, self.size.y];
        let whole = |quad: &Quad| plain(quad) && covers(quad.rect, canvas);
        let found = self
            .quads
            .iter()
            .rposition(|quad| whole(quad) && opaque(quad.fill));
        self.opaque = found.is_some();
        self.floor_quad = found.map(|at| self.quads[at]);
        let mut first = found.unwrap_or(0);
        let Some(mut bottom) = self.quads.get(first).copied() else {
            self.first = first;
            return;
        };
        first += 1;
        while whole(&bottom)
            && let Some(above) = self.quads.get(first).filter(|quad| whole(quad))
        {
            bottom = Quad {
                fill: over(above.fill, bottom.fill),
                ..*above
            };
            first += 1;
        }
        self.first = first;
        self.punch(bottom);
    }

    fn punch(&mut self, bottom: Quad) {
        let hole = self.quads[self.first..]
            .iter()
            .filter_map(inside)
            .max_by(|a, b| area_of(*a).total_cmp(&area_of(*b)))
            .filter(|_| plain(&bottom));
        let [x, y, w, h] = bottom.rect;
        let cut = hole
            .map(|[hx, hy, hr, hb]| {
                [
                    math::max(hx, x),
                    math::max(hy, y),
                    math::min(hr, x + w),
                    math::min(hb, y + h),
                ]
            })
            .filter(|[left, top, right, low]| right > left && low > top);
        let Some([left, top, right, low]) = cut else {
            self.floor.push(bottom);
            return;
        };
        let band = |rect: [f32; 4]| Quad {
            rect,
            scale: math::min(bottom.scale, 1.0),
            ..bottom
        };
        let bands = [
            [x, y, w, top - y],
            [x, low, w, y + h - low],
            [x, top, left - x, low - top],
            [right, top, x + w - right, low - top],
        ];
        self.floor.extend(
            bands
                .into_iter()
                .filter(|rect| rect[2] > 0.0 && rect[3] > 0.0)
                .map(band),
        );
    }

    pub(super) fn parts(&self) -> [&[Quad]; 3] {
        [
            self.held.as_slice(),
            &self.floor,
            &self.quads[self.first..],
        ]
    }

    pub(super) fn scale(&self) -> f32 {
        if self.reference.x <= 0.0 || self.reference.y <= 0.0 {
            return 0.0;
        }
        math::min(
            self.size.x / self.reference.x,
            self.size.y / self.reference.y,
        )
    }

    pub(super) fn place(&self, rect: [f32; 4], anchor: Anchor) -> Area {
        let [x, y, w, h] = rect;
        let [ax, ay] = match anchor {
            Anchor::Stretch => {
                let kx = self.size.x / self.reference.x;
                let ky = self.size.y / self.reference.y;
                return Area {
                    x: x * kx,
                    y: y * ky,
                    w: w * kx,
                    h: h * ky,
                };
            }
            Anchor::At(at) => at,
        };
        let s = self.scale();
        let origin_x = (ax * self.size.x).round();
        let origin_y = (ay * self.size.y).round();
        Area {
            x: origin_x + (x - ax * self.reference.x) * s,
            y: origin_y + (y - ay * self.reference.y) * s,
            w: w * s,
            h: h * s,
        }
    }

    pub(super) fn text_px(&self, size: f32) -> f32 {
        math::max(self.min_text_px, size * self.scale())
    }

    pub(super) fn plate(&mut self, rect: Area, plate: Plate) {
        self.cased_plate(rect, plate, None);
    }

    pub(super) fn cased_plate(&mut self, rect: Area, plate: Plate, case: Option<Case>) {
        let fill = plate.fill.filter(|fill| fill.visible());
        let stroke = plate
            .stroke
            .filter(|stroke| stroke.width > 0.0 && stroke.color.visible());
        let case = case.filter(|case| case.color.visible());
        if rect.w <= 0.01
            || rect.h <= 0.01
            || (fill.is_none() && stroke.is_none() && case.is_none())
        {
            return;
        }
        let half = math::min(rect.w, rect.h) * 0.5;
        self.quads.push(Quad {
            rect: [rect.x, rect.y, rect.w, rect.h],
            uv: [
                1.0 - math::clamp(plate.sweep, 0.0, 1.0),
                if case.is_some_and(|case| case.doubled) {
                    1.0
                } else {
                    0.0
                },
                0.0,
                0.0,
            ],
            fill: fill.map_or(0, Rgba::packed),
            edge: stroke.map_or(0, |stroke| stroke.color.packed()),
            mode: PLATE,
            radius: math::clamp(plate.radius, 0.0, half),
            stroke_width: stroke.map_or(0.0, |stroke| stroke.width),
            stroke_inset: stroke.map_or(0.0, |stroke| stroke.inset),
            scale: plate.soft,
            angle: plate.angle_radians,
            case: case.map_or(0, |case| case.color.packed()),
            case_grow: case.map_or(0.0, |case| case.grow),
            case_offset: case.map_or([0.0; 2], |case| case.offset),
            rim: case
                .and_then(|case| case.rim)
                .map_or(0, |(color, _)| color.packed()),
            rim_grow: case.and_then(|case| case.rim).map_or(0.0, |(_, grow)| grow),
            align: [0; 2],
        });
    }

    pub(super) fn image(
        &mut self,
        (name, cell, grey): (&str, Option<&str>, f32),
        rect: Area,
        (color, (radius, case)): (Rgba, (f32, Option<Case>)),
    ) {
        let Some(index) = self
            .pictures
            .iter()
            .position(|picture| picture.name == name)
        else {
            self.icons_missing += 1;
            return;
        };
        let before = self.quads.len();
        self.picture_quad(index, (cell, grey), rect, color);
        let case = case.filter(|case| case.color.visible());
        if let Some(quad) = self.quads.get_mut(before) {
            let [_, _, w, h] = quad.rect;
            quad.radius = math::clamp(radius, 0.0, math::min(w, h) * 0.5);
            if let Some(case) = case {
                quad.case = case.color.packed();
                quad.case_grow = case.grow;
                quad.case_offset = case.offset;
                if let Some((color, grow)) = case.rim {
                    quad.rim = color.packed();
                    quad.rim_grow = grow;
                }
                quad.stroke_inset = if case.doubled { 1.0 } else { 0.0 };
            }
        }
    }

    pub(super) fn cell(
        &mut self,
        (atlas, cell): (&str, &str),
        rect: Area,
        (color, grey): (Rgba, f32),
    ) {
        let Some(index) = self
            .pictures
            .iter()
            .position(|picture| picture.atlas == atlas)
        else {
            self.icons_missing += 1;
            return;
        };
        self.picture_quad(index, (Some(cell), grey), rect, color);
    }

    pub(super) fn cell_size(&self, atlas: &str, cell: &str) -> Option<[f32; 2]> {
        let picture = self
            .pictures
            .iter()
            .find(|picture| picture.atlas == atlas)?;
        let [_, _, w, h] = *picture.cells.get(cell)?;
        Some([w, h])
    }

    pub(super) fn cell_turned(
        &mut self,
        (atlas, cell): (&str, &str),
        center: Vec2,
        (scale, angle): (f32, f32),
        color: Rgba,
    ) {
        let found = self
            .pictures
            .iter()
            .enumerate()
            .find(|(_, picture)| picture.atlas == atlas)
            .and_then(|(index, picture)| Some((index, *picture.cells.get(cell)?)));
        let Some((index, [ix, iy, iw, ih])) = found else {
            self.icons_missing += 1;
            return;
        };
        if !color.visible() || iw <= 0.0 || ih <= 0.0 || scale <= 0.0 {
            return;
        }
        let (w, h) = (iw * scale, ih * scale);
        self.quads.push(Quad {
            rect: [center.x - w * 0.5, center.y - h * 0.5, w, h],
            uv: [ix, iy, iw, ih],
            fill: color.packed(),
            edge: index as u32,
            mode: IMAGE,
            scale,
            angle,
            ..Quad::default()
        });
    }

    fn picture_quad(
        &mut self,
        index: usize,
        (cell, grey): (Option<&str>, f32),
        rect: Area,
        color: Rgba,
    ) {
        let Some(picture) = self.pictures.get(index) else {
            return;
        };
        let [ix, iy, iw, ih] = match cell {
            None => [0.0, 0.0, picture.size[0], picture.size[1]],
            Some(key) => match picture.cells.get(key) {
                Some(found) => *found,
                None => {
                    self.icons_missing += 1;
                    return;
                }
            },
        };
        if !color.visible() || iw <= 0.0 || ih <= 0.0 {
            return;
        }
        let k = math::min(rect.w / iw, rect.h / ih);
        let (w, h) = (iw * k, ih * k);
        self.quads.push(Quad {
            rect: [
                (rect.x + (rect.w - w) * 0.5).round(),
                (rect.y + (rect.h - h) * 0.5).round(),
                w,
                h,
            ],
            uv: [ix, iy, iw, ih],
            fill: color.packed(),
            edge: index as u32,
            mode: IMAGE,
            stroke_width: grey,
            scale: k,
            ..Quad::default()
        });
    }

    pub(super) fn marker(
        &mut self,
        name: &str,
        center: Vec2,
        size: f32,
        (color, angle_radians): (Rgba, f32),
    ) {
        let Some(&[ix, iy, iw, ih]) = self.icons.cells.get(name) else {
            self.icons_missing += 1;
            return;
        };
        if !color.visible() || iw <= 0.0 || ih <= 0.0 || size <= 0.0 {
            return;
        }
        self.quads.push(Quad {
            rect: [center.x - size * 0.5, center.y - size * 0.5, size, size],
            uv: [ix, iy, iw, ih],
            fill: color.packed(),
            mode: ICON,
            scale: size / math::max(iw, ih),
            angle: angle_radians,
            ..Quad::default()
        });
    }

    pub(super) fn advance(&self, text: &str, weight: Weight) -> f32 {
        let Some(face) = self.font.as_ref().and_then(|font| font.face(weight)) else {
            return 0.0;
        };
        text.chars()
            .filter_map(|character| face.glyph(character))
            .map(|glyph| glyph.advance)
            .sum()
    }

    pub(super) fn width(&self, advance: f32, px: f32, weight: Weight) -> f32 {
        self.font
            .as_ref()
            .and_then(|font| font.face(weight))
            .map_or(0.0, |face| advance * px / face.em_px)
    }

    pub(super) fn measure(&self, text: &str, px: f32, weight: Weight) -> f32 {
        self.width(self.advance(text, weight), px, weight)
    }

    pub(super) fn kept(&self, text: &str, (px, weight): (f32, Weight), room: f32) -> usize {
        let face = self.font.as_ref().and_then(|font| font.face(weight));
        let mut advance = 0.0;
        let mut end = 0;
        for (at, character) in text.char_indices() {
            if let Some(glyph) = face.and_then(|face| face.glyph(character)) {
                advance += glyph.advance;
            }
            if self.width(advance, px, weight) > room {
                return end;
            }
            end = at + character.len_utf8();
        }
        end
    }

    pub(super) fn text(&mut self, (text, mut width): (&str, f32), rect: Area, style: TextStyle) {
        if text.is_empty() || !style.color.visible() {
            return;
        }
        let mut style = style;
        if rect.w > 0.0 && width > rect.w && style.px > self.min_text_px {
            let px = math::max(self.min_text_px, style.px * rect.w / width);
            width *= px / style.px;
            style.px = px;
            self.fitted += 1;
        }
        let Some(face) = self.font.as_ref().and_then(|font| font.face(style.weight)) else {
            return;
        };
        self.texts += 1;
        let k = style.px / face.em_px;
        let baseline = (rect.y + rect.h * 0.5 + face.cap * k * 0.5).round();
        let mut pen = match style.align {
            Align::Left => rect.x,
            Align::Center => rect.x + (rect.w - width) * 0.5,
            Align::Right => rect.x + rect.w - width,
        }
        .round();
        let fill = style.color.packed();
        let boost = match face.thin {
            Some(thin) => {
                thin.boost * math::clamp((thin.below_px - style.px) / thin.span_px, 0.0, 1.0)
            }
            None => 0.0,
        };
        let reach = (boost + style.grow + style.soft + GLYPH_MARGIN_PX) / k;
        let trim = math::max(0.0, face.spread_px - reach);
        for character in text.chars() {
            let Some(glyph) = face.glyph(character) else {
                self.glyphs_missing += 1;
                continue;
            };
            if glyph.w > 0.0 && glyph.h > 0.0 {
                self.glyphs += 1;
                let across = math::min(trim, glyph.w * 0.5);
                let down = math::min(trim, glyph.h * 0.5);
                let (w, h) = (glyph.w - 2.0 * across, glyph.h - 2.0 * down);
                self.quads.push(Quad {
                    rect: [
                        pen + (glyph.offset_x + across) * k,
                        baseline + (glyph.offset_y + down) * k,
                        w * k,
                        h * k,
                    ],
                    uv: [glyph.x + across, glyph.y + down, w, h],
                    fill,
                    edge: face.slot,
                    mode: GLYPH,
                    radius: face.spread_px,
                    stroke_width: boost + style.grow,
                    stroke_inset: style.soft,
                    scale: k,
                    ..Quad::default()
                });
            }
            pen += glyph.advance * k;
        }
    }

    pub(super) fn covered_pixels(&self) -> u64 {
        self.parts()
            .into_iter()
            .flatten()
            .map(|quad| quad.covered() as u64)
            .sum()
    }
}
