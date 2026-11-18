use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::play::screen::{Align, Anchor, Fit, Rgba, Shown, Tone, Weight};

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
}

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

#[derive(Clone, Debug, PartialEq)]
pub(super) struct StrokeRef {
    pub width: f32,
    pub color: Paint,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Set {
    pub fill: Option<Paint>,
    pub color: Option<Paint>,
    pub stroke: Option<StrokeRef>,
    pub icon: Option<Template>,
    pub grey: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Shadow {
    pub offset: [f32; 2],
    pub grow: f32,
    pub soft: f32,
    pub color: Paint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Dir {
    LeftToRight,
    RightToLeft,
    BottomToTop,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct TextKind {
    pub text: Template,
    pub size: f32,
    pub face: String,
    pub weight: Weight,
    pub align: Align,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum FlowKind {
    Text {
        text: TextKind,
        color: Paint,
    },
    Icon {
        icon: Template,
        w: f32,
        h: f32,
        color: Paint,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct FlowItem {
    pub when: Option<Cond>,
    pub kind: FlowKind,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Kind {
    Plate,
    Text(TextKind),
    Icon(Fit),
    Image(Picture),
    Markers(Markers),
    Bar {
        value: Number,
        max: Number,
        track: Paint,
        dir: Dir,
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
    },
    Flow {
        align: Align,
        padding: f32,
        gap: f32,
        items: Vec<FlowItem>,
    },
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
    pub small: Option<Template>,
    pub cover: bool,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CardList {
    Weapons,
    Armor,
    Abilities,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Element {
    pub id: String,
    pub rect: [f32; 4],
    pub anchor: Anchor,
    pub unscaled: bool,
    pub when: Option<Cond>,
    pub styles: Vec<(Cond, Set)>,
    pub look: Set,
    pub radius: f32,
    pub shadows: Vec<Shadow>,
    pub sweep: Option<Number>,
    pub pulse: Option<Pulse>,
    pub serial: usize,
    pub kind: Kind,
    pub card: Option<(CardList, usize)>,
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
    pub over: Option<Shown>,
    pub elements: Vec<Element>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Layout {
    pub reference: [f32; 2],
    pub stroke_rim: f32,
    pub min_text_px: f32,
    pub palette: BTreeMap<String, Rgba>,
    pub screens: BTreeMap<Shown, Screen>,
    pub unread: Vec<String>,
    pub skipped: Vec<String>,
    pub ability_texts: BTreeMap<String, String>,
    pub tips: Vec<String>,
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

    pub(super) fn faces(&mut self, names: &[String]) {
        let skipped = &mut self.skipped;
        let settle = |text: &mut TextKind| match names.iter().position(|name| *name == text.face) {
            Some(index) => {
                text.weight = Weight(index);
                Ok(())
            }
            None => Err(format!(
                "weight `{}` is no face of the font atlas",
                text.face
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
                    | Kind::Icon(_)
                    | Kind::Image(_)
                    | Kind::Markers(_)
                    | Kind::Bar { .. }
                    | Kind::Pips { .. } => Ok(()),
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
    match text.strip_prefix('!') {
        Some(rest) => Cond {
            path: path(rest),
            negated: true,
        },
        None => Cond {
            path: path(text),
            negated: false,
        },
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

fn stroke(
    value: &Value,
    palette: &BTreeMap<String, Rgba>,
    at: String,
) -> Result<StrokeRef, String> {
    let read = Reader::of(value, at)?;
    Ok(StrokeRef {
        width: read.number("width")?,
        color: paint(
            read.get("color").unwrap_or(&Value::Null),
            palette,
            &read.within("color"),
        )?,
    })
}

fn set_of(value: &Value, palette: &BTreeMap<String, Rgba>, at: String) -> Result<Set, String> {
    let read = Reader::of(value, at)?;
    let mut set = Set::default();
    for (key, given) in read.fields {
        let at = read.within(key);
        match key.as_str() {
            "fill" => set.fill = Some(paint(given, palette, &at)?),
            "color" => set.color = Some(paint(given, palette, &at)?),
            "stroke" => set.stroke = Some(stroke(given, palette, at)?),
            "icon" => {
                let text = given
                    .as_str()
                    .ok_or_else(|| format!("{at}: expected a string"))?;
                set.icon = Some(template(text, &at)?);
            }
            "grey" => set.grey = Some(unit(given, &at)?),
            _ => return Err(format!("{at}: style_if cannot set `{key}`")),
        }
    }
    Ok(set)
}

fn text_kind(read: &Reader<'_>) -> Result<TextKind, String> {
    Ok(TextKind {
        text: template(read.text("text")?, &read.within("text"))?,
        size: read.number("size")?,
        face: read.text("weight")?.to_owned(),
        weight: Weight::default(),
        align: match read.get("align") {
            None => Align::Left,
            Some(_) => align(read.text("align")?, &read.within("align"))?,
        },
    })
}

fn flow_items(
    read: &Reader<'_>,
    palette: &BTreeMap<String, Rgba>,
) -> Result<Vec<FlowItem>, String> {
    let mut items = Vec::new();
    for (index, value) in read.list("items")?.iter().enumerate() {
        let item = Reader::of(value, format!("{}.items[{index}]", read.at))?;
        let color = paint(
            item.get("color").unwrap_or(&Value::Null),
            palette,
            &item.within("color"),
        )?;
        let kind = match item.text("type")? {
            "text" => FlowKind::Text {
                text: text_kind(&item)?,
                color,
            },
            "icon" => FlowKind::Icon {
                icon: template(item.text("icon")?, &item.within("icon"))?,
                w: item.number("w")?,
                h: item.number("h")?,
                color,
            },
            other => return Err(format!("{}.type: unknown item `{other}`", item.at)),
        };
        items.push(FlowItem {
            when: item.get("when").and_then(Value::as_str).map(cond),
            kind,
        });
    }
    Ok(items)
}

fn kind_of(read: &Reader<'_>, palette: &BTreeMap<String, Rgba>) -> Result<Kind, String> {
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
        "text" => Kind::Text(text_kind(read)?),
        "icon" => Kind::Icon(match read.get("fit") {
            None => Fit::Cell,
            Some(_) => match read.text("fit")? {
                "cell" => Fit::Cell,
                "trim" => Fit::Trim,
                other => return Err(format!("{}: unknown fit `{other}`", read.within("fit"))),
            },
        }),
        "image" => Kind::Image(Picture {
            image: template(read.text("image")?, &read.within("image"))?,
            small: match read.get("image_small") {
                None => None,
                Some(_) => Some(template(
                    read.text("image_small")?,
                    &read.within("image_small"),
                )?),
            },
            cover: match read.get("fit") {
                None => false,
                Some(_) => match read.text("fit")? {
                    "cover" => true,
                    "contain" => false,
                    other => return Err(format!("{}: unknown fit `{other}`", read.within("fit"))),
                },
            },
            cell: match read.get("atlas") {
                None => None,
                Some(_) => Some((
                    read.text("atlas")?.to_owned(),
                    template(read.text("key")?, &read.within("key"))?,
                )),
            },
        }),
        "markers" => Kind::Markers(markers_of(read, palette)?),
        "bar" => Kind::Bar {
            value: number_at("value")?,
            max: number_at("max")?,
            track: match read.get("track") {
                None => Paint::None,
                Some(_) => paint_at("track")?,
            },
            dir: match read.text("dir")? {
                "ltr" => Dir::LeftToRight,
                "rtl" => Dir::RightToLeft,
                "btt" => Dir::BottomToTop,
                other => {
                    return Err(format!(
                        "{}: unknown direction `{other}`",
                        read.within("dir")
                    ));
                }
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
        },
        "flow" => Kind::Flow {
            align: aligned(Align::Left)?,
            padding: read.number("padding")?,
            gap: read.number("gap")?,
            items: flow_items(read, palette)?,
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

fn markers_of(read: &Reader<'_>, palette: &BTreeMap<String, Rgba>) -> Result<Markers, String> {
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
        icon: read.text("icon")?.to_owned(),
        size: read.number("size")?,
        rotate: read.flag("rotate")?,
        backing,
        outline,
    })
}

fn card_of(styles: &[(Cond, Set)]) -> Option<(CardList, usize)> {
    styles
        .iter()
        .find_map(|(cond, _)| match cond.path.as_slice() {
            [Seg::Key(buy), Seg::Key(list), Seg::Index(index), ..] if buy == "buy" => {
                let list = match list.as_str() {
                    "weapons" => CardList::Weapons,
                    "armor" => CardList::Armor,
                    "abilities" => CardList::Abilities,
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
            let when = cond(rule_read.text("when")?);
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
        stroke: match read.get("stroke") {
            None => None,
            Some(given) => Some(stroke(given, cx.palette, read.within("stroke"))?),
        },
        icon: match read.get("icon") {
            None => None,
            Some(_) => Some(template(read.text("icon")?, &read.within("icon"))?),
        },
        grey: match read.get("grey") {
            None => None,
            Some(given) => Some(unit(given, &read.within("grey"))?),
        },
    };
    let card = card_of(&styles);
    let kind = kind_of(&read, cx.palette)?;
    let shadows = shadows_of(&read, cx.palette)?;
    let sweep = match read.get("sweep") {
        None => None,
        Some(given) => Some(number_of(given, &read.within("sweep"))?),
    };
    if sweep.is_some() && !matches!(kind, Kind::Plate) {
        return Err(format!(
            "{}: sweep draws on plate elements only",
            read.within("sweep")
        ));
    }
    let pulse = match read.get("pulse") {
        None => None,
        Some(given) => Some(pulse_of(given, cx.palette, read.within("pulse"))?),
    };
    if !shadows.is_empty() && !matches!(kind, Kind::Plate | Kind::Text(_)) {
        return Err(format!(
            "{}: shadows draw under plate and text elements only",
            read.within("shadows")
        ));
    }
    Ok(Element {
        id: read.text("id")?.to_owned(),
        rect,
        anchor,
        unscaled: read.flag("unscaled")?,
        when: read.get("when").and_then(Value::as_str).map(cond),
        styles,
        look,
        radius: read.number_or("radius", 0.0)?,
        shadows,
        sweep,
        pulse,
        serial: 0,
        kind,
        card,
    })
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
    if let Some(tone) = Tone::ALL
        .into_iter()
        .find(|tone| !palette.contains_key(tone.key()))
    {
        return Err(format!("$.palette: `{}` is missing", tone.key()));
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
    let cx = Context {
        palette: &palette,
        anchors: &anchors,
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
        let over = match screen.get("over") {
            None => None,
            Some(_) => {
                let over = screen.text("over")?;
                Some(shown_of(over).ok_or_else(|| format!("{at}.over: unknown screen `{over}`"))?)
            }
        };
        let mut elements = Vec::new();
        for (index, value) in screen.list("elements")?.iter().enumerate() {
            match element(value, &cx, format!("{at}.elements[{index}]")) {
                Ok(found) => elements.push(found),
                Err(problem) => skipped.push(problem),
            }
        }
        screens.insert(shown, Screen { over, elements });
    }
    for missing in Shown::ALL {
        if let std::collections::btree_map::Entry::Vacant(empty) = screens.entry(missing) {
            skipped.push(format!(
                "$.screens: `{}` is missing; it draws nothing",
                missing.key()
            ));
            empty.insert(Screen {
                over: None,
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
        stroke_rim: read.number("stroke_rim")?,
        min_text_px: font.number("min_px")?,
        palette,
        screens,
        unread,
        skipped,
        ability_texts,
        tips,
    })
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
    if let Some(stroke) = &set.stroke {
        paint_paths(&stroke.color, found);
    }
    if let Some(icon) = &set.icon {
        template_paths(icon, found);
    }
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

fn shadows_of(read: &Reader<'_>, palette: &BTreeMap<String, Rgba>) -> Result<Vec<Shadow>, String> {
    let Some(listed) = read.get("shadows") else {
        return Ok(Vec::new());
    };
    let listed = listed
        .as_array()
        .ok_or_else(|| format!("{}: expected a list", read.within("shadows")))?;
    let mut shadows = Vec::new();
    for (index, value) in listed.iter().enumerate() {
        let one = Reader::of(value, format!("{}.shadows[{index}]", read.at))?;
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
        Kind::Plate | Kind::Icon(_) => {}
        Kind::Image(picture) => {
            template_paths(&picture.image, found);
            if let Some(small) = &picture.small {
                template_paths(small, found);
            }
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
            ..
        } => {
            number_paths(count, found);
            number_paths(value, found);
            for given in [Some(on), Some(off), full.as_ref()].into_iter().flatten() {
                paint_paths(given, found);
            }
        }
        Kind::Flow { items, .. } => {
            for item in items {
                if let Some(when) = &item.when {
                    found.push(when.path.clone());
                }
                match &item.kind {
                    FlowKind::Text { text, color } => {
                        template_paths(&text.text, found);
                        paint_paths(color, found);
                    }
                    FlowKind::Icon { icon, color, .. } => {
                        template_paths(icon, found);
                        paint_paths(color, found);
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
        if let Some(when) = &self.when {
            listed.push(when.path.clone());
        }
        set_paths(&self.look, &mut listed);
        for shadow in &self.shadows {
            paint_paths(&shadow.color, &mut listed);
        }
        if let Some(sweep) = &self.sweep {
            number_paths(sweep, &mut listed);
        }
        if let Some(pulse) = &self.pulse {
            listed.push(pulse.watch.clone());
            if let Some(color) = &pulse.color {
                paint_paths(color, &mut listed);
            }
        }
        for (cond, given) in &self.styles {
            listed.push(cond.path.clone());
            set_paths(given, &mut listed);
        }
        kind_paths(&self.kind, &mut listed);
        found.extend(listed.into_iter().map(|path| (self.id.clone(), path)));
    }
}
