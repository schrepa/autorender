use std::collections::BTreeMap;

use bytemuck::{Pod, Zeroable};
use engine::TextureHandle;
use engine::math;
use engine::prelude::Resource;
use glam::Vec2;

use crate::rules::{BuyRequest, Phase};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Shown {
    #[default]
    Hud,
    Buy,
    Scoreboard,
    RoundEnd,
    MatchEnd,
    Title,
    Select,
    Loading,
    Pause,
    Settings,
}

impl Shown {
    pub const ALL: [Shown; 10] = [
        Shown::Hud,
        Shown::Buy,
        Shown::Scoreboard,
        Shown::RoundEnd,
        Shown::MatchEnd,
        Shown::Title,
        Shown::Select,
        Shown::Loading,
        Shown::Pause,
        Shown::Settings,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Shown::Hud => "hud",
            Shown::Buy => "buy",
            Shown::Scoreboard => "scoreboard",
            Shown::RoundEnd => "round_end",
            Shown::MatchEnd => "match_end",
            Shown::Title => "title",
            Shown::Select => "character_select",
            Shown::Loading => "loading",
            Shown::Pause => "pause",
            Shown::Settings => "settings",
        }
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScreenState {
    pub shown: Shown,
    pub buy_open: bool,
    pub board_held: bool,
    pub holds_pointer: bool,
}

#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct ScreenBuys {
    pub requests: Vec<BuyRequest>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tone {
    #[default]
    Text,
    Attack,
    Defend,
    Ready,
    Danger,
}

impl Tone {
    pub const ALL: [Tone; 5] = [
        Tone::Text,
        Tone::Attack,
        Tone::Defend,
        Tone::Ready,
        Tone::Danger,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Tone::Text => "text",
            Tone::Attack => "attack",
            Tone::Defend => "defend",
            Tone::Ready => "ready",
            Tone::Danger => "danger",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoundFacts {
    pub time: f32,
    pub number: u32,
    pub planted: bool,
    pub fuse: f32,
    pub site: String,
    pub time_low: bool,
    pub fuse_low: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TeamFacts {
    pub color: Tone,
    pub name: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlateFacts {
    pub icon: String,
    pub alive: bool,
    pub health: f32,
    pub ult_ready: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FeedLine {
    pub killer: String,
    pub killer_color: Tone,
    pub weapon: String,
    pub headshot: bool,
    pub victim: String,
    pub victim_color: Tone,
    pub mine: bool,
    pub ability: String,
    pub world: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlayerFacts {
    pub health: f32,
    pub health_low: bool,
    pub armor: f32,
    pub armor_icon: String,
    pub credits: u32,
    pub char_name: String,
    pub role: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AbilityFacts {
    pub icon: String,
    pub max: u32,
    pub charges: u32,
    pub empty: bool,
    pub cooling: bool,
    pub cooldown: f32,
    pub points: u32,
    pub points_needed: u32,
    pub ready: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct WeaponFacts {
    pub name: String,
    pub mag: Option<u32>,
    pub mag_size: Option<u32>,
    pub reserve: Option<u32>,
    pub low: bool,
    pub reload: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ActionFacts {
    pub active: bool,
    pub label: String,
    pub left: f32,
    pub progress: f32,
    pub color: Tone,
    pub checkpoint: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Card {
    pub kind: String,
    pub name: String,
    pub icon: String,
    pub price: String,
    pub owned: bool,
    pub afford: bool,
    pub max: u32,
    pub charges: u32,
    pub points: u32,
    pub stats: WeaponStats,
    pub buy: Option<BuyRequest>,
    pub sell: Option<BuyRequest>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WeaponStats {
    pub damage: f32,
    pub rate: f32,
    pub range: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mate {
    pub name: String,
    pub icon: String,
    pub credits: u32,
    pub can_rifle: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BuyFacts {
    pub time: f32,
    pub time_low: bool,
    pub weapons: Vec<Card>,
    pub armor: Vec<Card>,
    pub abilities: Vec<Card>,
    pub reward_win: u32,
    pub reward_loss: u32,
    pub team: Vec<Mate>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MapItem {
    pub pos: [f32; 2],
    pub yaw_degrees: f32,
    pub color: Option<Tone>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MinimapFacts {
    pub me: Vec<MapItem>,
    pub players: Vec<MapItem>,
    pub device: Vec<MapItem>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BoardRow {
    pub name: String,
    pub icon: String,
    pub k: u32,
    pub d: u32,
    pub a: u32,
    pub adr: u32,
    pub loadout: String,
    pub credits: String,
    pub ult: u32,
    pub ult_need: u32,
    pub alive: bool,
    pub me: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResultFacts {
    pub title: String,
    pub reason: String,
    pub color: Tone,
    pub mvp: String,
    pub mvp_icon: String,
    pub mvp_color: Tone,
    pub mvp_portrait: String,
    pub mvp_name: String,
    pub mvp_stats: String,
    pub mvp_char: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AbilityCard {
    pub icon: String,
    pub key: String,
    pub name: String,
    pub desc: String,
    pub ultimate: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SelectCard {
    pub key: String,
    pub image: String,
    pub name: String,
    pub role: String,
    pub selected: bool,
    pub abilities: [AbilityCard; 4],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SelectFacts {
    pub cards: Vec<SelectCard>,
    pub level: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoadingFacts {
    pub map: String,
    pub mode: String,
    pub percent: u32,
    pub progress: f32,
    pub site_a_note: String,
    pub site_b_note: String,
    pub tips: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Slider {
    pub frac: f32,
    pub text: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeyRow {
    pub action: String,
    pub binding: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SettingsFacts {
    pub sliders: [Slider; 5],
    pub crosshair: usize,
    pub crosshair_color: String,
    pub invert: bool,
    pub keys: Vec<KeyRow>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MenuFacts {
    pub screen: Option<Shown>,
    pub focus: String,
    pub select: SelectFacts,
    pub loading: LoadingFacts,
    pub settings: SettingsFacts,
    pub can_quit: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpectateFacts {
    pub active: bool,
    pub name: String,
    pub icon: String,
    pub color: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cover {
    Fit,
    Fill,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PickupFacts {
    pub active: bool,
    pub key: String,
    pub weapon: String,
    pub icon: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CastFacts {
    pub active: bool,
    pub key: String,
    pub icon: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlaceFacts {
    pub name: String,
    pub callout: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HitMark {
    pub age: f32,
    pub headshot: bool,
    pub killed: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DamageMark {
    pub angle_radians: f32,
    pub age: f32,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct HudFacts {
    pub round: RoundFacts,
    pub score: [u32; 2],
    pub teams: [TeamFacts; 2],
    pub players: [Vec<PlateFacts>; 2],
    pub feed: Vec<FeedLine>,
    pub player: PlayerFacts,
    pub abilities: [AbilityFacts; 4],
    pub weapon: WeaponFacts,
    pub spread_px: f32,
    pub action: ActionFacts,
    pub buy: BuyFacts,
    pub board: [Vec<BoardRow>; 2],
    pub result: ResultFacts,
    pub minimap: MinimapFacts,
    pub menu: MenuFacts,
    pub spectate: SpectateFacts,
    pub place: PlaceFacts,
    pub scope: u32,
    pub pickup: PickupFacts,
    pub cast: CastFacts,
    pub hit: Option<HitMark>,
    pub damage: Vec<DamageMark>,
    pub phase: Option<Phase>,
    pub live: bool,
}

impl HudFacts {
    pub fn cards(&self) -> impl Iterator<Item = &Card> + '_ {
        self.buy
            .weapons
            .iter()
            .chain(&self.buy.armor)
            .chain(&self.buy.abilities)
    }
}

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
pub struct Plate {
    pub fill: Option<Rgba>,
    pub radius: f32,
    pub stroke: Option<Stroke>,
    pub angle_radians: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Weight {
    #[default]
    Regular,
    Bold,
}

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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Face {
    pub line_height: f32,
    pub ascent: f32,
    pub descent: f32,
    pub cap: f32,
    pub glyphs: Vec<Option<Glyph>>,
}

impl Face {
    pub fn glyph(&self, character: char) -> Option<&Glyph> {
        self.glyphs.get(character as usize)?.as_ref()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Font {
    pub em_px: f32,
    pub spread_px: f32,
    pub regular: Face,
    pub bold: Face,
}

impl Font {
    pub fn face(&self, weight: Weight) -> &Face {
        match weight {
            Weight::Regular => &self.regular,
            Weight::Bold => &self.bold,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Icons {
    pub cells: BTreeMap<String, [f32; 4]>,
    pub trims: BTreeMap<String, [f32; 4]>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Fit {
    #[default]
    Cell,
    Trim,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Picture {
    pub name: String,
    pub size: [f32; 2],
    pub cells: BTreeMap<String, [f32; 4]>,
}

pub const PLATE: u32 = 0;
pub const ICON: u32 = 1;
pub const GLYPH: u32 = 2;
pub const IMAGE: u32 = 3;

pub const THIN_BELOW_PX: f32 = 28.0;
pub const THIN_SPAN_PX: f32 = 12.0;
pub const THIN_BOOST: f32 = 0.2;

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
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct ScreenDraw {
    pub size: Vec2,
    pub reference: Vec2,
    pub min_text_px: f32,
    pub font: Option<Font>,
    pub icons: Icons,
    pub pictures: Vec<Picture>,
    pub atlases: Option<[TextureHandle; 11]>,
    pub quads: Vec<Quad>,
    pub revision: u64,
    pub texts: u32,
    pub glyphs: u32,
    pub glyphs_missing: u32,
    pub icons_missing: u32,
    pub fitted: u32,
}

impl ScreenDraw {
    pub fn begin(&mut self, size: Vec2) {
        self.size = size;
        self.quads.clear();
        self.texts = 0;
        self.glyphs = 0;
        self.glyphs_missing = 0;
        self.icons_missing = 0;
        self.fitted = 0;
    }

    pub fn scale(&self) -> f32 {
        if self.reference.x <= 0.0 || self.reference.y <= 0.0 {
            return 0.0;
        }
        math::min(
            self.size.x / self.reference.x,
            self.size.y / self.reference.y,
        )
    }

    pub fn place(&self, rect: [f32; 4], anchor: Anchor, unscaled: bool) -> Area {
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
        let s = if unscaled { 1.0 } else { self.scale() };
        let origin_x = (ax * self.size.x).round();
        let origin_y = (ay * self.size.y).round();
        Area {
            x: origin_x + (x - ax * self.reference.x) * s,
            y: origin_y + (y - ay * self.reference.y) * s,
            w: w * s,
            h: h * s,
        }
    }

    pub fn text_px(&self, size: f32, unscaled: bool) -> f32 {
        let s = if unscaled { 1.0 } else { self.scale() };
        math::max(self.min_text_px, size * s)
    }

    pub fn plate(&mut self, rect: Area, plate: Plate) {
        let fill = plate.fill.filter(|fill| fill.visible());
        let stroke = plate
            .stroke
            .filter(|stroke| stroke.width > 0.0 && stroke.color.visible());
        if rect.w <= 0.01 || rect.h <= 0.01 || (fill.is_none() && stroke.is_none()) {
            return;
        }
        let half = math::min(rect.w, rect.h) * 0.5;
        self.quads.push(Quad {
            rect: [rect.x, rect.y, rect.w, rect.h],
            uv: [0.0; 4],
            fill: fill.map_or(0, Rgba::packed),
            edge: stroke.map_or(0, |stroke| stroke.color.packed()),
            mode: PLATE,
            radius: math::clamp(plate.radius, 0.0, half),
            stroke_width: stroke.map_or(0.0, |stroke| stroke.width),
            stroke_inset: stroke.map_or(0.0, |stroke| stroke.inset),
            scale: 1.0,
            angle: plate.angle_radians,
        });
    }

    pub fn picture_size(&self, name: &str) -> Option<[f32; 2]> {
        self.pictures
            .iter()
            .find(|picture| picture.name == name)
            .map(|picture| picture.size)
    }

    pub fn image(
        &mut self,
        (name, cell, fit): (&str, Option<&str>, Cover),
        rect: Area,
        color: Rgba,
    ) {
        let Some((index, picture)) = self
            .pictures
            .iter()
            .enumerate()
            .find(|(_, picture)| picture.name == name)
        else {
            self.icons_missing += 1;
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
        let (quad, uv) = match fit {
            Cover::Fit => {
                let k = math::min(rect.w / iw, rect.h / ih);
                let (w, h) = (iw * k, ih * k);
                let at = [
                    (rect.x + (rect.w - w) * 0.5).round(),
                    (rect.y + (rect.h - h) * 0.5).round(),
                ];
                ([at[0], at[1], w, h], [ix, iy, iw, ih])
            }
            Cover::Fill => {
                let k = math::max(rect.w / iw, rect.h / ih);
                let (sw, sh) = (rect.w / k, rect.h / k);
                (
                    [rect.x, rect.y, rect.w, rect.h],
                    [ix + (iw - sw) * 0.5, iy + (ih - sh) * 0.5, sw, sh],
                )
            }
        };
        let k = quad[2] / uv[2];
        self.quads.push(Quad {
            rect: quad,
            uv,
            fill: color.packed(),
            edge: index as u32,
            mode: IMAGE,
            scale: k,
            ..Quad::default()
        });
    }

    pub fn marker(
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

    pub fn icon(&mut self, name: &str, rect: Area, (color, fit): (Rgba, Fit)) -> bool {
        let cells = match fit {
            Fit::Cell => &self.icons.cells,
            Fit::Trim => &self.icons.trims,
        };
        let Some(&[ix, iy, iw, ih]) = cells.get(name) else {
            self.icons_missing += 1;
            return false;
        };
        if !color.visible() || iw <= 0.0 || ih <= 0.0 {
            return true;
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
            edge: 0,
            mode: ICON,
            radius: 0.0,
            stroke_width: 0.0,
            stroke_inset: 0.0,
            scale: k,
            angle: 0.0,
        });
        true
    }

    pub fn measure(&self, text: &str, px: f32, weight: Weight) -> f32 {
        let Some(font) = &self.font else {
            return 0.0;
        };
        let face = font.face(weight);
        let advance: f32 = text
            .chars()
            .filter_map(|character| face.glyph(character))
            .map(|glyph| glyph.advance)
            .sum();
        advance * px / font.em_px
    }

    pub fn text(&mut self, text: &str, rect: Area, style: TextStyle) {
        if text.is_empty() || !style.color.visible() {
            return;
        }
        let mut style = style;
        let mut width = self.measure(text, style.px, style.weight);
        if rect.w > 0.0 && width > rect.w {
            style.px *= rect.w / width;
            width = rect.w;
            self.fitted += 1;
        }
        let Some(font) = &self.font else {
            return;
        };
        self.texts += 1;
        let face = font.face(style.weight);
        let k = style.px / font.em_px;
        let baseline = (rect.y + rect.h * 0.5 + face.cap * k * 0.5).round();
        let mut pen = match style.align {
            Align::Left => rect.x,
            Align::Center => rect.x + (rect.w - width) * 0.5,
            Align::Right => rect.x + rect.w - width,
        }
        .round();
        let fill = style.color.packed();
        let boost = match style.weight {
            Weight::Regular => {
                THIN_BOOST * math::clamp((THIN_BELOW_PX - style.px) / THIN_SPAN_PX, 0.0, 1.0)
            }
            Weight::Bold => 0.0,
        };
        for character in text.chars() {
            let Some(glyph) = face.glyph(character) else {
                self.glyphs_missing += 1;
                continue;
            };
            if glyph.w > 0.0 && glyph.h > 0.0 {
                self.glyphs += 1;
                self.quads.push(Quad {
                    rect: [
                        pen + glyph.offset_x * k,
                        baseline + glyph.offset_y * k,
                        glyph.w * k,
                        glyph.h * k,
                    ],
                    uv: [glyph.x, glyph.y, glyph.w, glyph.h],
                    fill,
                    edge: 0,
                    mode: GLYPH,
                    radius: font.spread_px,
                    stroke_width: boost,
                    stroke_inset: 0.0,
                    scale: k,
                    angle: 0.0,
                });
            }
            pen += glyph.advance * k;
        }
    }

    pub fn covered_pixels(&self) -> u64 {
        self.quads
            .iter()
            .map(|quad| {
                let pad = if quad.mode == PLATE { 2.0 } else { 0.0 };
                ((quad.rect[2] + pad) * (quad.rect[3] + pad)) as u64
            })
            .sum()
    }
}
