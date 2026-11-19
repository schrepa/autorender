use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::play::screen::{Align, Anchor, Rgba, Shown, Tone, Weight};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Seg {
    Key(String),
    Index(usize),
}

pub(super) type Path = Vec<Seg>;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Cond {
    pub path: Path,
    pub negated: bool,
    pub equals: Option<String>,
}

pub(super) type Conds = Vec<Cond>;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Paint {
    None,
    Fixed(Rgba),
    Path(Path),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Number {
    Fixed(f32),
    Path(Path),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Format {
    Plain,
    Int,
    Clock,
    Credits,
    Sec1,
    Upper,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Piece {
    Text(String),
    Value(Path, Format),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Template {
    pub pieces: Vec<Piece>,
}

impl Template {
    pub(super) fn fixed(&self) -> Option<&str> {
        match self.pieces.as_slice() {
            [Piece::Text(text)] => Some(text),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Set {
    pub fill: Option<Paint>,
    pub color: Option<Paint>,
    pub icon: Option<Template>,
    pub grey: Option<f32>,
    pub shadows: Option<Vec<Shadow>>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Shadow {
    pub offset: [f32; 2],
    pub grow: f32,
    pub soft: f32,
    pub color: Paint,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct TextKind {
    pub text: Template,
    pub size: f32,
    pub face: (String, String),
    pub weight: Weight,
    pub align: Align,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum FlowKind {
    Text {
        text: TextKind,
        color: Paint,
        cap: bool,
    },
    Icon {
        icon: Template,
        w: f32,
        h: f32,
        color: Option<Paint>,
        atlas: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct FlowItem {
    pub when: Conds,
    pub kind: FlowKind,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Arc {
    pub value: Number,
    pub max: Number,
    pub width: f32,
    pub track: Option<Paint>,
    pub start_deg: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Kind {
    Plate,
    Text(TextKind),
    Icon(String),
    Arc(Arc),
    Image(Picture),
    Markers(Markers),
    Bar {
        value: Number,
        max: Number,
        track: Paint,
        knob: Option<Knob>,
    },
    Pips {
        count: Number,
        value: Number,
        size: [f32; 2],
        gap: f32,
        on: Paint,
        off: Paint,
        full: Option<Paint>,
        align: Align,
        track: Option<(Paint, f32)>,
    },
    Flow {
        align: Align,
        padding: f32,
        inset: f32,
        gap: f32,
        min_w: f32,
        items: Vec<FlowItem>,
    },
    Reticle,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Knob {
    pub size: f32,
    pub fill: Paint,
    pub ring: Option<Paint>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Picture {
    pub image: Template,
    pub cell: Option<(String, Template)>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Markers {
    pub map: String,
    pub items: Path,
    pub icon: String,
    pub size: f32,
    pub rotate: bool,
    pub backing: Option<(Paint, f32)>,
    pub outline: Option<(Paint, String)>,
    pub atlas: Option<String>,
    pub disc: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CardList {
    Weapons,
    Armor,
    Abilities,
    Team,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Element {
    pub id: String,
    pub rect: [f32; 4],
    pub anchor: Anchor,
    pub when: Conds,
    pub styles: Vec<(Conds, Set)>,
    pub look: Set,
    pub radius: f32,
    pub pulse: Option<Pulse>,
    pub serial: usize,
    pub kind: Kind,
    pub card: Option<(CardList, usize)>,
    pub row: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Stack {
    pub gap: f32,
    pub ceiling: ([f32; 4], Anchor),
    pub floor: ([f32; 4], Anchor),
    pub rows: std::ops::Range<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Pulse {
    pub watch: Path,
    pub seconds: f32,
    pub scale: f32,
    pub color: Option<Paint>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Screen {
    pub elements: Vec<Element>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Layout {
    pub reference: [f32; 2],
    pub min_text_px: f32,
    pub palette: BTreeMap<String, Rgba>,
    pub screens: BTreeMap<Shown, Screen>,
    pub unread: Vec<String>,
    pub skipped: Vec<String>,
    pub ability_texts: BTreeMap<String, String>,
    pub tips: Vec<String>,
    pub key_cap: Option<KeyCap>,
    pub damage_edge: Option<DamageEdge>,
    pub crosshair: Option<CrosshairPiece>,
    pub stickers: Option<String>,
    pub stacks: Vec<Stack>,
    pub rows: usize,
    pub fit: Fitting,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Fitting {
    pub floor: f32,
    pub mark: String,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct KeyCap {
    pub fill: Paint,
    pub shadows: Vec<Shadow>,
    pub radius: f32,
    pub padding_x: f32,
    pub height: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct DamageEdge {
    pub atlas: String,
    pub tip: String,
    pub tip_scale: f32,
    pub radius_share: f32,
    pub length_deg: f32,
    pub band_px: f32,
    pub rim_px: f32,
    pub color: Rgba,
    pub fade_seconds: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct CrosshairPiece {
    pub outline: Rgba,
}

impl Layout {
    pub(super) fn tone(&self, tone: Tone) -> Option<Rgba> {
        self.palette.get(tone.key()).copied()
    }

    pub(super) fn elements(&self) -> usize {
        self.screens
            .values()
            .map(|screen| screen.elements.len())
            .sum()
    }

    pub(super) fn faces(&mut self, names: &[(String, String)]) {
        let skipped = &mut self.skipped;
        let settle = |text: &mut TextKind| match names.iter().position(|name| *name == text.face) {
            Some(index) => {
                text.weight = Weight(index);
                Ok(())
            }
            None => Err(format!(
                "face `{}` is not in the bound font `{}`",
                text.face.1, text.face.0
            )),
        };
        for screen in self.screens.values_mut() {
            screen.elements.retain_mut(|element| {
                let settled = match &mut element.kind {
                    Kind::Text(text) => settle(text),
                    Kind::Flow { items, .. } => {
                        items.iter_mut().try_for_each(|item| match &mut item.kind {
                            FlowKind::Text { text, .. } => settle(text),
                            FlowKind::Icon { .. } => Ok(()),
                        })
                    }
                    Kind::Plate
                    | Kind::Icon(..)
                    | Kind::Arc(_)
                    | Kind::Image(_)
                    | Kind::Markers(_)
                    | Kind::Bar { .. }
                    | Kind::Pips { .. }
                    | Kind::Reticle => Ok(()),
                };
                match settled {
                    Ok(()) => true,
                    Err(problem) => {
                        skipped.push(format!("element `{}`: {problem}; skipped", element.id));
                        false
                    }
                }
            });
        }
    }

    pub(super) fn number(&mut self) {
        let mut serial = 0;
        for screen in self.screens.values_mut() {
            for element in &mut screen.elements {
                element.serial = serial;
                serial += 1;
            }
        }
    }

    pub(super) fn retain(&mut self, mut kept: impl FnMut(&Element) -> Result<(), String>) {
        let skipped = &mut self.skipped;
        for screen in self.screens.values_mut() {
            screen.elements.retain(|element| match kept(element) {
                Ok(()) => true,
                Err(problem) => {
                    skipped.push(problem);
                    false
                }
            });
        }
    }
}

type Fields = Map<String, Value>;

struct Reader<'a> {
    fields: &'a Fields,
    at: String,
}

impl<'a> Reader<'a> {
    fn of(value: &'a Value, at: String) -> Result<Self, String> {
        let fields = value
            .as_object()
            .ok_or_else(|| format!("{at}: expected an object"))?;
        Ok(Self { fields, at })
    }

    fn get(&self, key: &str) -> Option<&'a Value> {
        self.fields.get(key)
    }

    fn text(&self, key: &str) -> Result<&'a str, String> {
        self.get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{}.{key}: expected a string", self.at))
    }

    fn number(&self, key: &str) -> Result<f32, String> {
        self.get(key)
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .ok_or_else(|| format!("{}.{key}: expected a number", self.at))
    }

    fn number_or(&self, key: &str, absent: f32) -> Result<f32, String> {
        match self.get(key) {
            None => Ok(absent),
            Some(_) => self.number(key),
        }
    }

    fn flag(&self, key: &str) -> Result<bool, String> {
        match self.get(key) {
            None => Ok(false),
            Some(value) => value
                .as_bool()
                .ok_or_else(|| format!("{}.{key}: expected true or false", self.at)),
        }
    }

    fn pair(&self, key: &str) -> Result<[f32; 2], String> {
        let list = self.list(key)?;
        match list {
            [a, b] => match (a.as_f64(), b.as_f64()) {
                (Some(a), Some(b)) => Ok([a as f32, b as f32]),
                (Some(_) | None, Some(_) | None) => {
                    Err(format!("{}.{key}: expected two numbers", self.at))
                }
            },
            _ => Err(format!("{}.{key}: expected two numbers", self.at)),
        }
    }

    fn list(&self, key: &str) -> Result<&'a [Value], String> {
        self.get(key)
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .ok_or_else(|| format!("{}.{key}: expected a list", self.at))
    }

    fn within(&self, key: &str) -> String {
        format!("{}.{key}", self.at)
    }
}

fn path(text: &str) -> Path {
    text.split('.')
        .map(|part| match part.parse::<usize>() {
            Ok(index) => Seg::Index(index),
            Err(_) => Seg::Key(part.to_owned()),
        })
        .collect()
}

fn cond(text: &str) -> Cond {
    let (negated, rest) = match text.strip_prefix('!') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let (name, equals) = match rest.split_once('=') {
        Some((name, value)) => (name, Some(value.to_owned())),
        None => (rest, None),
    };
    Cond {
        path: path(name),
        negated,
        equals,
    }
}

fn conds_of(value: Option<&Value>, at: &str) -> Result<Conds, String> {
    match value {
        None => Ok(Vec::new()),
        Some(Value::String(text)) => Ok(vec![cond(text)]),
        Some(Value::Array(listed)) => listed
            .iter()
            .map(|one| {
                one.as_str()
                    .map(cond)
                    .ok_or_else(|| format!("{at}: expected a list of conditions"))
            })
            .collect(),
        Some(Value::Null | Value::Bool(_) | Value::Number(_) | Value::Object(_)) => {
            Err(format!("{at}: expected a condition or a list of them"))
        }
    }
}

pub(super) fn hex(text: &str) -> Option<Rgba> {
    let digits = text.strip_prefix('#')?;
    let byte = |at: usize| {
        digits
            .get(at..at + 2)
            .and_then(|pair| u8::from_str_radix(pair, 16).ok())
            .map(|value| f32::from(value) / 255.0)
    };
    let alpha = match digits.len() {
        6 => Some(1.0),
        8 => byte(6),
        _ => None,
    }?;
    Some(Rgba([byte(0)?, byte(2)?, byte(4)?, alpha]))
}

fn paint(value: &Value, palette: &BTreeMap<String, Rgba>, at: &str) -> Result<Paint, String> {
    let text = value
        .as_str()
        .ok_or_else(|| format!("{at}: expected a colour name"))?;
    if text == "none" {
        return Ok(Paint::None);
    }
    if let Some(rest) = text.strip_prefix('@') {
        return Ok(Paint::Path(path(rest)));
    }
    palette
        .get(text)
        .copied()
        .or_else(|| hex(text))
        .map(Paint::Fixed)
        .ok_or_else(|| format!("{at}: `{text}` is not in the palette"))
}

fn number_of(value: &Value, at: &str) -> Result<Number, String> {
    match value {
        Value::String(text) => Ok(Number::Path(path(text))),
        Value::Number(number) => number
            .as_f64()
            .map(|value| Number::Fixed(value as f32))
            .ok_or_else(|| format!("{at}: expected a number")),
        Value::Null | Value::Bool(_) | Value::Array(_) | Value::Object(_) => {
            Err(format!("{at}: expected a number or a data path"))
        }
    }
}

fn format_of(name: &str, at: &str) -> Result<Format, String> {
    match name {
        "int" => Ok(Format::Int),
        "clock" => Ok(Format::Clock),
        "credits" => Ok(Format::Credits),
        "sec1" => Ok(Format::Sec1),
        "upper" => Ok(Format::Upper),
        _ => Err(format!("{at}: unknown format `{name}`")),
    }
}

pub(super) fn template(text: &str, at: &str) -> Result<Template, String> {
    let mut pieces = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find('{') {
        if open > 0 {
            pieces.push(Piece::Text(rest[..open].to_owned()));
        }
        let after = &rest[open + 1..];
        let close = after
            .find('}')
            .ok_or_else(|| format!("{at}: `{{` without `}}`"))?;
        let inner = &after[..close];
        let (name, format) = match inner.split_once('|') {
            Some((name, format)) => (name, format_of(format, at)?),
            None => (inner, Format::Plain),
        };
        pieces.push(Piece::Value(path(name), format));
        rest = &after[close + 1..];
    }
    if !rest.is_empty() {
        pieces.push(Piece::Text(rest.to_owned()));
    }
    Ok(Template { pieces })
}

fn align(text: &str, at: &str) -> Result<Align, String> {
    match text {
        "left" => Ok(Align::Left),
        "center" => Ok(Align::Center),
        "right" => Ok(Align::Right),
        _ => Err(format!("{at}: unknown alignment `{text}`")),
    }
}

fn set_of(value: &Value, palette: &BTreeMap<String, Rgba>, at: String) -> Result<Set, String> {
    let read = Reader::of(value, at)?;
    let mut set = Set::default();
    for (key, given) in read.fields {
        let at = read.within(key);
        match key.as_str() {
            "fill" => set.fill = Some(paint(given, palette, &at)?),
            "color" => set.color = Some(paint(given, palette, &at)?),
            "icon" => {
                let text = given
                    .as_str()
                    .ok_or_else(|| format!("{at}: expected a string"))?;
                set.icon = Some(template(text, &at)?);
            }
            "grey" => set.grey = Some(unit(given, &at)?),
            "shadows" => set.shadows = Some(shadow_list(given, palette, &at)?),
            _ => return Err(format!("{at}: style_if cannot set `{key}`")),
        }
    }
    Ok(set)
}

fn text_kind(read: &Reader<'_>, cx: &Context<'_>) -> Result<TextKind, String> {
    let name = read.text("face")?;
    let face = cx
        .fonts
        .get(name)
        .cloned()
        .ok_or_else(|| format!("{}: `{name}` is not in `fonts`", read.within("face")))?;
    Ok(TextKind {
        text: template(read.text("text")?, &read.within("text"))?,
        size: read.number("size")?,
        face,
        weight: Weight::default(),
        align: match read.get("align") {
            None => Align::Left,
            Some(_) => align(read.text("align")?, &read.within("align"))?,
        },
    })
}

fn flow_items(read: &Reader<'_>, cx: &Context<'_>) -> Result<Vec<FlowItem>, String> {
    let palette = cx.palette;
    let mut items = Vec::new();
    for (index, value) in read.list("items")?.iter().enumerate() {
        let item = Reader::of(value, format!("{}.items[{index}]", read.at))?;
        let color = match item.get("color") {
            None => None,
            Some(given) => Some(paint(given, palette, &item.within("color"))?),
        };
        let kind = match item.text("type")? {
            kind @ ("text" | "key") => {
                let cap = kind == "key";
                if cap && !cx.key_cap {
                    return Err(format!("{}: key items need `$.pieces.key_cap`", item.at));
                }
                FlowKind::Text {
                    text: text_kind(&item, cx)?,
                    color: color.ok_or_else(|| {
                        format!("{}: expected a colour name", item.within("color"))
                    })?,
                    cap,
                }
            }
            "icon" => FlowKind::Icon {
                icon: template(item.text("icon")?, &item.within("icon"))?,
                w: item.number("w")?,
                h: item.number("h")?,
                color,
                atlas: sticker_of(&item, cx)?,
            },
            other => return Err(format!("{}.type: unknown item `{other}`", item.at)),
        };
        items.push(FlowItem {
            when: conds_of(item.get("when"), &item.within("when"))?,
            kind,
        });
    }
    Ok(items)
}

fn sticker_of(read: &Reader<'_>, cx: &Context<'_>) -> Result<String, String> {
    let named = read.text("atlas")?;
    Ok(cx
        .atlases
        .get(named)
        .map_or(named, String::as_str)
        .to_owned())
}

fn arc_of(read: &Reader<'_>, palette: &BTreeMap<String, Rgba>) -> Result<Arc, String> {
    let number_at = |key: &str| number_of(read.get(key).unwrap_or(&Value::Null), &read.within(key));
    Ok(Arc {
        value: number_at("value")?,
        max: number_at("max")?,
        width: read.number("width")?,
        track: match read.get("track") {
            None => None,
            Some(given) => Some(paint(given, palette, &read.within("track"))?),
        },
        start_deg: read.number_or("start_deg", 0.0)?,
    })
}

fn kind_of(read: &Reader<'_>, cx: &Context<'_>) -> Result<Kind, String> {
    let palette = cx.palette;
    let paint_at = |key: &str| {
        paint(
            read.get(key).unwrap_or(&Value::Null),
            palette,
            &read.within(key),
        )
    };
    let number_at = |key: &str| number_of(read.get(key).unwrap_or(&Value::Null), &read.within(key));
    let aligned = |absent: Align| match read.get("align") {
        None => Ok(absent),
        Some(_) => align(read.text("align")?, &read.within("align")),
    };
    Ok(match read.text("kind")? {
        "plate" => Kind::Plate,
        "reticle" => Kind::Reticle,
        "text" => Kind::Text(text_kind(read, cx)?),
        "icon" => Kind::Icon(sticker_of(read, cx)?),
        "arc" => Kind::Arc(arc_of(read, palette)?),
        "image" => Kind::Image(Picture {
            image: template(read.text("image")?, &read.within("image"))?,
            cell: match read.get("atlas") {
                None => None,
                Some(_) => Some((
                    read.text("atlas")?.to_owned(),
                    template(read.text("key")?, &read.within("key"))?,
                )),
            },
        }),
        "markers" => Kind::Markers(markers_of(read, cx)?),
        "bar" => Kind::Bar {
            value: number_at("value")?,
            max: number_at("max")?,
            track: match read.get("track") {
                None => Paint::None,
                Some(_) => paint_at("track")?,
            },
            knob: match read.get("knob") {
                None => None,
                Some(value) => Some(knob_of(value, palette, read.within("knob"))?),
            },
        },
        "pips" => Kind::Pips {
            count: number_at("count")?,
            value: number_at("value")?,
            size: read.pair("size")?,
            gap: read.number("gap")?,
            on: paint_at("on")?,
            off: paint_at("off")?,
            full: match read.get("full") {
                None => None,
                Some(_) => Some(paint_at("full")?),
            },
            align: aligned(Align::Center)?,
            track: match read.get("track") {
                None => None,
                Some(given) => {
                    let track = Reader::of(given, read.within("track"))?;
                    let fill = paint(
                        track.get("fill").unwrap_or(&Value::Null),
                        palette,
                        &track.within("fill"),
                    )?;
                    Some((fill, track.number("pad")?))
                }
            },
        },
        "flow" => Kind::Flow {
            align: aligned(Align::Left)?,
            padding: read.number("padding")?,
            inset: read.number_or("inset", read.number("padding")?)?,
            gap: read.number("gap")?,
            min_w: read.number_or("min_w", 0.0)?,
            items: flow_items(read, cx)?,
        },
        other => return Err(format!("{}: unknown kind `{other}`", read.within("kind"))),
    })
}

fn knob_of(value: &Value, palette: &BTreeMap<String, Rgba>, at: String) -> Result<Knob, String> {
    let read = Reader::of(value, at)?;
    Ok(Knob {
        size: read.number("size")?,
        fill: paint(
            read.get("fill").unwrap_or(&Value::Null),
            palette,
            &read.within("fill"),
        )?,
        ring: match read.get("ring") {
            None => None,
            Some(ring) => Some(paint(ring, palette, &read.within("ring"))?),
        },
    })
}

fn texts_of(read: &Reader<'_>) -> Result<(BTreeMap<String, String>, Vec<String>), String> {
    let texts = Reader::of(
        read.get("texts").unwrap_or(&Value::Null),
        read.within("texts"),
    )?;
    let abilities = Reader::of(
        texts.get("abilities").unwrap_or(&Value::Null),
        texts.within("abilities"),
    )?;
    let mut named = BTreeMap::new();
    for (name, value) in abilities.fields {
        let line = value
            .as_str()
            .ok_or_else(|| format!("{}: expected a string", abilities.within(name)))?;
        named.insert(name.clone(), line.to_owned());
    }
    let tips = texts
        .list("tips")?
        .iter()
        .map(|tip| {
            tip.as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{}: expected strings", texts.within("tips")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((named, tips))
}

fn markers_of(read: &Reader<'_>, cx: &Context<'_>) -> Result<Markers, String> {
    let palette = cx.palette;
    let disc = read.flag("disc")?;
    let backing = match read.get("backing") {
        None => None,
        Some(value) => {
            let backing = Reader::of(value, read.within("backing"))?;
            let fill = paint(
                backing.get("fill").unwrap_or(&Value::Null),
                palette,
                &backing.within("fill"),
            )?;
            Some((fill, backing.number("size")?))
        }
    };
    let outline = match read.get("outline") {
        None => None,
        Some(value) => Some((
            paint(value, palette, &read.within("outline"))?,
            read.text("outline_icon")?.to_owned(),
        )),
    };
    Ok(Markers {
        map: read.text("map")?.to_owned(),
        items: path(read.text("items")?),
        icon: if disc {
            String::new()
        } else {
            read.text("icon")?.to_owned()
        },
        size: if disc { 0.0 } else { read.number("size")? },
        rotate: read.flag("rotate")?,
        backing,
        outline,
        atlas: match read.get("atlas") {
            None => None,
            Some(_) => Some(sticker_of(read, cx)?),
        },
        disc,
    })
}

fn card_of(styles: &[(Conds, Set)]) -> Option<(CardList, usize)> {
    styles
        .iter()
        .flat_map(|(conds, _)| conds)
        .find_map(|cond| match cond.path.as_slice() {
            [Seg::Key(buy), Seg::Key(list), Seg::Index(index), ..] if buy == "buy" => {
                let list = match list.as_str() {
                    "weapons" => CardList::Weapons,
                    "armor" => CardList::Armor,
                    "abilities" => CardList::Abilities,
                    "team" => CardList::Team,
                    _ => return None,
                };
                Some((list, *index))
            }
            _ => None,
        })
}

struct Context<'a> {
    palette: &'a BTreeMap<String, Rgba>,
    anchors: &'a BTreeMap<String, Anchor>,
    fonts: &'a BTreeMap<String, (String, String)>,
    atlases: &'a BTreeMap<String, String>,
    key_cap: bool,
    rows: &'a [String],
}

const SHADOWED: &str = "shadows draw under plate, text, flow, bar and image elements only";

fn shadowed(kind: &Kind) -> bool {
    match kind {
        Kind::Plate | Kind::Text(_) | Kind::Flow { .. } | Kind::Bar { .. } | Kind::Image(_) => true,
        Kind::Icon(..) | Kind::Arc(_) | Kind::Markers(_) | Kind::Pips { .. } | Kind::Reticle => {
            false
        }
    }
}

fn element(value: &Value, cx: &Context<'_>, at: String) -> Result<Element, String> {
    let read = Reader::of(value, at)?;
    let rect = match read.list("rect")? {
        [x, y, w, h] => [x, y, w, h].map(|value| value.as_f64().unwrap_or(f64::NAN) as f32),
        _ => return Err(format!("{}: expected [x, y, w, h]", read.within("rect"))),
    };
    if rect.iter().any(|value| !value.is_finite()) {
        return Err(format!("{}: expected four numbers", read.within("rect")));
    }
    let anchor_name = read.text("anchor")?;
    let anchor = *cx
        .anchors
        .get(anchor_name)
        .ok_or_else(|| format!("{}: unknown anchor `{anchor_name}`", read.within("anchor")))?;
    let mut styles = Vec::new();
    if let Some(rules) = read.get("style_if") {
        let rules = rules
            .as_array()
            .ok_or_else(|| format!("{}: expected a list", read.within("style_if")))?;
        for (index, rule) in rules.iter().enumerate() {
            let rule_read = Reader::of(rule, format!("{}.style_if[{index}]", read.at))?;
            let when = conds_of(
                Some(rule_read.get("when").unwrap_or(&Value::Null)),
                &rule_read.within("when"),
            )?;
            let set = set_of(
                rule_read.get("set").unwrap_or(&Value::Null),
                cx.palette,
                rule_read.within("set"),
            )?;
            styles.push((when, set));
        }
    }
    let look = Set {
        fill: match read.get("fill") {
            None => None,
            Some(given) => Some(paint(given, cx.palette, &read.within("fill"))?),
        },
        color: match read.get("color") {
            None => None,
            Some(given) => Some(paint(given, cx.palette, &read.within("color"))?),
        },
        icon: match read.get("icon") {
            None => None,
            Some(_) => Some(template(read.text("icon")?, &read.within("icon"))?),
        },
        grey: match read.get("grey") {
            None => None,
            Some(given) => Some(unit(given, &read.within("grey"))?),
        },
        shadows: match read.get("shadows") {
            None => None,
            Some(given) => Some(shadow_list(given, cx.palette, &read.within("shadows"))?),
        },
    };
    let card = card_of(&styles);
    let kind = kind_of(&read, cx)?;
    let pulse = match read.get("pulse") {
        None => None,
        Some(given) => Some(pulse_of(given, cx.palette, read.within("pulse"))?),
    };
    let any_shadows = look.shadows.is_some() || styles.iter().any(|(_, set)| set.shadows.is_some());
    if any_shadows && !shadowed(&kind) {
        return Err(format!("{}: {SHADOWED}", read.within("shadows")));
    }
    Ok(Element {
        id: read.text("id")?.to_owned(),
        rect,
        anchor,
        when: conds_of(read.get("when"), &read.within("when"))?,
        styles,
        look,
        radius: read.number_or("radius", 0.0)?,
        pulse,
        serial: 0,
        kind,
        card,
        row: match read.get("stack") {
            None => None,
            Some(_) => {
                let name = read.text("stack")?;
                Some(cx.rows.iter().position(|row| row == name).ok_or_else(|| {
                    format!("{}: no stack has the row `{name}`", read.within("stack"))
                })?)
            }
        },
    })
}

type Named = (BTreeMap<String, String>, BTreeMap<String, (String, String)>);

fn named_files(read: &Reader<'_>) -> Result<Named, String> {
    let atlases: BTreeMap<String, String> = read
        .fields
        .iter()
        .filter_map(|(name, value)| {
            let file = value.as_object()?.get("atlas")?.as_str()?;
            Some((name.clone(), file.to_owned()))
        })
        .collect();
    let mut fonts = BTreeMap::new();
    if let Some(listed) = read.get("fonts").and_then(Value::as_object) {
        for (name, value) in listed {
            if !value.is_object() {
                continue;
            }
            let one = Reader::of(value, format!("$.fonts.{name}"))?;
            fonts.insert(
                name.clone(),
                (one.text("atlas")?.to_owned(), one.text("face")?.to_owned()),
            );
        }
    }
    Ok((atlases, fonts))
}

type Pieces = (Option<KeyCap>, Option<DamageEdge>, Option<CrosshairPiece>);

fn swatch(read: &Reader<'_>, key: &str, palette: &BTreeMap<String, Rgba>) -> Result<Rgba, String> {
    let name = read.text(key)?;
    palette
        .get(name)
        .copied()
        .ok_or_else(|| format!("{}: `{name}` is not in the palette", read.within(key)))
}

fn edge_of(
    pieces: &Reader<'_>,
    palette: &BTreeMap<String, Rgba>,
    atlases: &BTreeMap<String, String>,
) -> Result<Option<DamageEdge>, String> {
    let Some(given) = pieces.get("damage_edge") else {
        return Ok(None);
    };
    let edge = Reader::of(given, pieces.within("damage_edge"))?;
    let tip = Reader::of(
        pieces.get("damage_tip").unwrap_or(&Value::Null),
        pieces.within("damage_tip"),
    )?;
    let atlas = atlases.get(STICKERS).ok_or_else(|| {
        format!(
            "{}: the tip is a sticker and `$.{STICKERS}` is missing",
            edge.at
        )
    })?;
    Ok(Some(DamageEdge {
        atlas: atlas.clone(),
        tip: tip.text("icon")?.to_owned(),
        tip_scale: edge.number("tip_scale")?,
        radius_share: edge.number("radius_share")?,
        length_deg: edge.number("length_deg")?,
        band_px: edge.number("band_px")?,
        rim_px: edge.number("rim_px")?,
        color: swatch(&edge, "color", palette)?,
        fade_seconds: edge.number("fade_seconds")?,
    }))
}

fn key_cap_of(cap: &Reader<'_>, palette: &BTreeMap<String, Rgba>) -> Result<KeyCap, String> {
    let border = Reader::of(
        cap.get("border").unwrap_or(&Value::Null),
        cap.within("border"),
    )?;
    Ok(KeyCap {
        fill: paint(
            cap.get("fill").unwrap_or(&Value::Null),
            palette,
            &cap.within("fill"),
        )?,
        shadows: vec![Shadow {
            offset: [0.0, 0.0],
            grow: border.number("width")?,
            soft: 0.0,
            color: paint(
                border.get("color").unwrap_or(&Value::Null),
                palette,
                &border.within("color"),
            )?,
        }],
        radius: cap.number("radius")?,
        padding_x: cap.number("padding_x")?,
        height: cap.number("height")?,
    })
}

fn pieces_of(
    read: &Reader<'_>,
    palette: &BTreeMap<String, Rgba>,
    atlases: &BTreeMap<String, String>,
) -> Result<Pieces, String> {
    let Some(given) = read.get("pieces") else {
        return Ok((None, None, None));
    };
    let pieces = Reader::of(given, read.within("pieces"))?;
    let key_cap = match pieces.get("key_cap") {
        None => None,
        Some(given) => Some(key_cap_of(
            &Reader::of(given, pieces.within("key_cap"))?,
            palette,
        )?),
    };
    let crosshair = match pieces.get("crosshair") {
        None => None,
        Some(given) => {
            let cross = Reader::of(given, pieces.within("crosshair"))?;
            Some(CrosshairPiece {
                outline: swatch(&cross, "outline", palette)?,
            })
        }
    };
    Ok((key_cap, edge_of(&pieces, palette, atlases)?, crosshair))
}

const STICKERS: &str = "stickers";

fn edge_at(
    given: &Value,
    at: String,
    anchors: &BTreeMap<String, Anchor>,
) -> Result<([f32; 4], Anchor), String> {
    let read = Reader::of(given, at)?;
    let named = read.text("anchor")?;
    let anchor = *anchors
        .get(named)
        .ok_or_else(|| format!("{}: unknown anchor `{named}`", read.within("anchor")))?;
    Ok(([0.0, read.number("y")?, 0.0, 0.0], anchor))
}

fn stacks_of(
    read: &Reader<'_>,
    anchors: &BTreeMap<String, Anchor>,
) -> Result<(Vec<Stack>, Vec<String>), String> {
    let mut stacks = Vec::new();
    let mut rows: Vec<String> = Vec::new();
    let Some(_) = read.get("stacks") else {
        return Ok((stacks, rows));
    };
    for (index, given) in read.list("stacks")?.iter().enumerate() {
        let stack = Reader::of(given, format!("{}[{index}]", read.within("stacks")))?;
        let first = rows.len();
        for (place, row) in stack.list("rows")?.iter().enumerate() {
            let at = format!("{}[{place}]", stack.within("rows"));
            let name = row
                .as_str()
                .ok_or_else(|| format!("{at}: expected a string"))?;
            if rows.iter().any(|known| known == name) {
                return Err(format!("{at}: the row `{name}` is in two places"));
            }
            rows.push(name.to_owned());
        }
        let ceiling = edge_at(
            stack.get("ceiling").unwrap_or(&Value::Null),
            stack.within("ceiling"),
            anchors,
        )?;
        let floor = edge_at(
            stack.get("floor").unwrap_or(&Value::Null),
            stack.within("floor"),
            anchors,
        )?;
        stacks.push(Stack {
            gap: stack.number("gap")?,
            ceiling,
            floor,
            rows: first..rows.len(),
        });
    }
    Ok((stacks, rows))
}

fn shown_of(name: &str) -> Option<Shown> {
    Shown::ALL.into_iter().find(|shown| shown.key() == name)
}

pub(super) fn parse(text: &str) -> Result<Layout, String> {
    let whole: Value =
        serde_json::from_str(text).map_err(|error| format!("layout.json: {error}"))?;
    let read = Reader::of(&whole, "$".to_owned())?;
    let mut palette = BTreeMap::new();
    let colours = Reader::of(
        read.get("palette").unwrap_or(&Value::Null),
        read.within("palette"),
    )?;
    for (name, value) in colours.fields {
        let colour = value
            .as_str()
            .and_then(hex)
            .ok_or_else(|| format!("{}: expected #RRGGBB or #RRGGBBAA", colours.within(name)))?;
        palette.insert(name.clone(), colour);
    }
    let mut anchors = BTreeMap::new();
    let listed = Reader::of(
        read.get("anchors").unwrap_or(&Value::Null),
        read.within("anchors"),
    )?;
    for (name, value) in listed.fields {
        let anchor = match value.as_array().map(Vec::as_slice) {
            Some([x, y]) => match (x.as_f64(), y.as_f64()) {
                (Some(x), Some(y)) => Anchor::At([x as f32, y as f32]),
                (Some(_) | None, Some(_) | None) => {
                    return Err(format!("{}: expected two numbers", listed.within(name)));
                }
            },
            Some(_) => return Err(format!("{}: expected two numbers", listed.within(name))),
            None => Anchor::Stretch,
        };
        anchors.insert(name.clone(), anchor);
    }
    let font = Reader::of(
        read.get("font").unwrap_or(&Value::Null),
        read.within("font"),
    )?;
    let (atlases, fonts) = named_files(&read)?;
    let (key_cap, damage_edge, crosshair) = pieces_of(&read, &palette, &atlases)?;
    let (stacks, rows) = stacks_of(&read, &anchors)?;
    let cx = Context {
        palette: &palette,
        anchors: &anchors,
        fonts: &fonts,
        atlases: &atlases,
        key_cap: key_cap.is_some(),
        rows: &rows,
    };
    let listed = Reader::of(
        read.get("screens").unwrap_or(&Value::Null),
        read.within("screens"),
    )?;
    let mut screens = BTreeMap::new();
    let mut unread = Vec::new();
    let mut skipped = Vec::new();
    for (name, value) in listed.fields {
        let at = listed.within(name);
        let Some(shown) = shown_of(name) else {
            unread.push(name.clone());
            continue;
        };
        let screen = Reader::of(value, at.clone())?;
        let mut elements = Vec::new();
        for (index, value) in screen.list("elements")?.iter().enumerate() {
            match element(value, &cx, format!("{at}.elements[{index}]")) {
                Ok(found) => elements.push(found),
                Err(problem) => skipped.push(problem),
            }
        }
        screens.insert(shown, Screen { elements });
    }
    for missing in Shown::ALL {
        if let std::collections::btree_map::Entry::Vacant(empty) = screens.entry(missing) {
            skipped.push(format!(
                "$.screens: `{}` is missing; it draws nothing",
                missing.key()
            ));
            empty.insert(Screen {
                elements: Vec::new(),
            });
        }
    }
    let (ability_texts, tips) = texts_of(&read)?;
    let reference = read.pair("reference")?;
    if reference[0] <= 0.0 || reference[1] <= 0.0 {
        return Err("$.reference: expected two sizes above 0".to_owned());
    }
    Ok(Layout {
        reference,
        min_text_px: font.number("min_px")?,
        palette,
        screens,
        unread,
        skipped,
        ability_texts,
        tips,
        key_cap,
        damage_edge,
        crosshair,
        stickers: atlases.get(STICKERS).cloned(),
        stacks,
        rows: rows.len(),
        fit: fit_of(&read)?,
    })
}

fn fit_of(read: &Reader<'_>) -> Result<Fitting, String> {
    let rules = Reader::of(
        read.get("rules").unwrap_or(&Value::Null),
        read.within("rules"),
    )?;
    let fitting = Reader::of(
        rules.get("fit").unwrap_or(&Value::Null),
        rules.within("fit"),
    )?;
    let fit = Fitting {
        floor: fitting.number("floor")?,
        mark: fitting.text("mark")?.to_owned(),
    };
    if !(fit.floor > 0.0 && fit.floor <= 1.0) {
        return Err("$.rules.fit.floor: expected a share above 0, at most 1".to_owned());
    }
    Ok(fit)
}

fn paint_paths(paint: &Paint, found: &mut Vec<Path>) {
    if let Paint::Path(path) = paint {
        found.push(path.clone());
    }
}

fn template_paths(template: &Template, found: &mut Vec<Path>) {
    for piece in &template.pieces {
        if let Piece::Value(path, _) = piece {
            found.push(path.clone());
        }
    }
}

fn number_paths(number: &Number, found: &mut Vec<Path>) {
    if let Number::Path(path) = number {
        found.push(path.clone());
    }
}

fn set_paths(set: &Set, found: &mut Vec<Path>) {
    for given in [&set.fill, &set.color].into_iter().flatten() {
        paint_paths(given, found);
    }
    if let Some(icon) = &set.icon {
        template_paths(icon, found);
    }
    for shadow in set.shadows.iter().flatten() {
        paint_paths(&shadow.color, found);
    }
}

fn cond_paths(conds: &[Cond], found: &mut Vec<Path>) {
    found.extend(conds.iter().map(|cond| cond.path.clone()));
}

fn unit(value: &Value, at: &str) -> Result<f32, String> {
    value
        .as_f64()
        .map(|value| value as f32)
        .filter(|value| (0.0..=1.0).contains(value))
        .ok_or_else(|| format!("{at}: expected a number from 0 to 1"))
}

fn pulse_of(value: &Value, palette: &BTreeMap<String, Rgba>, at: String) -> Result<Pulse, String> {
    let read = Reader::of(value, at)?;
    let seconds = read.number("seconds")?;
    if seconds <= 0.0 {
        return Err(format!(
            "{}: expected seconds above 0",
            read.within("seconds")
        ));
    }
    Ok(Pulse {
        watch: path(read.text("watch")?),
        seconds,
        scale: read.number_or("scale", 1.0)?,
        color: match read.get("color") {
            None => None,
            Some(given) => Some(paint(given, palette, &read.within("color"))?),
        },
    })
}

fn shadow_list(
    listed: &Value,
    palette: &BTreeMap<String, Rgba>,
    at: &str,
) -> Result<Vec<Shadow>, String> {
    let listed = listed
        .as_array()
        .ok_or_else(|| format!("{at}: expected a list"))?;
    let mut shadows = Vec::new();
    for (index, value) in listed.iter().enumerate() {
        let one = Reader::of(value, format!("{at}[{index}]"))?;
        shadows.push(Shadow {
            offset: match one.get("offset") {
                None => [0.0, 0.0],
                Some(_) => one.pair("offset")?,
            },
            grow: one.number_or("grow", 0.0)?,
            soft: one.number_or("soft", 0.0)?,
            color: paint(
                one.get("color").unwrap_or(&Value::Null),
                palette,
                &one.within("color"),
            )?,
        });
    }
    Ok(shadows)
}

fn kind_paths(kind: &Kind, found: &mut Vec<Path>) {
    match kind {
        Kind::Plate | Kind::Icon(..) | Kind::Reticle => {}
        Kind::Arc(arc) => {
            number_paths(&arc.value, found);
            number_paths(&arc.max, found);
            if let Some(track) = &arc.track {
                paint_paths(track, found);
            }
        }
        Kind::Image(picture) => {
            template_paths(&picture.image, found);
            if let Some((_, key)) = &picture.cell {
                template_paths(key, found);
            }
        }
        Kind::Markers(markers) => {
            found.push(markers.items.clone());
            if let Some((fill, _)) = &markers.backing {
                paint_paths(fill, found);
            }
            if let Some((color, _)) = &markers.outline {
                paint_paths(color, found);
            }
        }
        Kind::Text(text) => template_paths(&text.text, found),
        Kind::Bar {
            value,
            max,
            track,
            knob,
            ..
        } => {
            if let Some(knob) = knob {
                paint_paths(&knob.fill, found);
                if let Some(ring) = &knob.ring {
                    paint_paths(ring, found);
                }
            }
            number_paths(value, found);
            number_paths(max, found);
            paint_paths(track, found);
        }
        Kind::Pips {
            count,
            value,
            on,
            off,
            full,
            track,
            ..
        } => {
            number_paths(count, found);
            number_paths(value, found);
            let track = track.as_ref().map(|(fill, _)| fill);
            for given in [Some(on), Some(off), full.as_ref(), track]
                .into_iter()
                .flatten()
            {
                paint_paths(given, found);
            }
        }
        Kind::Flow { items, .. } => {
            for item in items {
                cond_paths(&item.when, found);
                match &item.kind {
                    FlowKind::Text { text, color, .. } => {
                        template_paths(&text.text, found);
                        paint_paths(color, found);
                    }
                    FlowKind::Icon { icon, color, .. } => {
                        template_paths(icon, found);
                        if let Some(color) = color {
                            paint_paths(color, found);
                        }
                    }
                }
            }
        }
    }
}

impl Element {
    pub(super) fn listed(&self) -> Vec<(String, Path)> {
        let mut found = Vec::new();
        self.paths(&mut found);
        found
    }

    fn paths(&self, found: &mut Vec<(String, Path)>) {
        let mut listed = Vec::new();
        cond_paths(&self.when, &mut listed);
        set_paths(&self.look, &mut listed);
        if let Some(pulse) = &self.pulse {
            listed.push(pulse.watch.clone());
            if let Some(color) = &pulse.color {
                paint_paths(color, &mut listed);
            }
        }
        for (conds, given) in &self.styles {
            cond_paths(conds, &mut listed);
            set_paths(given, &mut listed);
        }
        kind_paths(&self.kind, &mut listed);
        found.extend(listed.into_iter().map(|path| (self.id.clone(), path)));
    }
}
