engine::brick!(screen);

mod control;
mod demo;
mod facts;
mod gpu;
mod layout;
mod live;
mod menu;
mod more;

pub use menu::Options;
mod paint;
mod paths;

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::prelude::*;
use engine::{Error, ErrorCode};
use engine_core::{SceneContext, SceneCx};
use engine_render::RenderBricks;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::play::flow::Flow;
use crate::play::numbers::Input;
use crate::play::screen::{
    Face, Font, Glyph, HudFacts, Icons, Picture, Quad, ScreenBuys, ScreenDraw, ScreenState, Shown,
};

use layout::{Layout, Seg};
use paths::Val;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScreenKeys {
    pub take: Vec<String>,
    pub give_back: Vec<String>,
    pub pick: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Lows {
    pub time_seconds: f32,
    pub fuse_seconds: f32,
    pub health: f32,
    pub magazine_share: f32,
    pub buy_seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FeedNumbers {
    pub lines: usize,
    pub seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MarkNumbers {
    pub gap_px: f32,
    pub length_px: f32,
    pub width_px: f32,
    pub seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Marks {
    pub hit: MarkNumbers,
    pub damage: MarkNumbers,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Words {
    pub attackers: String,
    pub defenders: String,
    pub round_won: String,
    pub round_lost: String,
    pub victory: String,
    pub defeat: String,
    pub elimination: String,
    pub exploded: String,
    pub defused: String,
    pub time_out: String,
    pub planting: String,
    pub defusing: String,
    pub free: String,
    pub armor_kind: String,
    pub light: String,
    pub heavy: String,
    pub per_charge: String,
    pub hidden: String,
    pub mvp_round: String,
    pub mvp_match: String,
    pub mvp_stats: String,
    pub mvp_char: String,
}

pub(super) struct Keys {
    take: Vec<Input>,
    give_back: Vec<Input>,
    pick: Vec<Input>,
}

pub(super) struct MapFrame {
    pub px_per_m: f32,
    pub origin_m: [f32; 2],
    pub size: [f32; 2],
}

pub(super) struct Loaded {
    layout: Layout,
    font: Font,
    icons: Icons,
    map: MapFrame,
    pictures: Vec<Picture>,
    picture_textures: Vec<String>,
    select_images: BTreeMap<String, String>,
    keys: Keys,
    menu_keys: menu::MenuInputs,
    demo: HudFacts,
}

#[derive(Component, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScreenSource {
    pub layout: String,
    pub icons: String,
    pub font: String,
    pub minimap: String,
    pub portraits: String,
    pub icons_texture: String,
    pub font_texture: String,
    pub pictures: BTreeMap<String, String>,
    pub sized: BTreeMap<String, [f32; 2]>,
    pub thin_faces: Vec<String>,
    pub atlases: Vec<String>,
    pub menu: menu::MenuSource,
    pub keys: ScreenKeys,
    pub low: Lows,
    pub feed: FeedNumbers,
    pub marks: Marks,
    pub words: Words,
    #[serde(skip)]
    loaded: Option<Arc<Loaded>>,
}

#[derive(Deserialize)]
struct GlyphFile {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    offset_x: f32,
    offset_y: f32,
    advance: f32,
}

#[derive(Deserialize)]
struct FaceFile {
    line_height: f32,
    ascent: f32,
    descent: f32,
    cap_height: f32,
    glyphs: BTreeMap<String, GlyphFile>,
}

#[derive(Deserialize)]
struct FontFile {
    em_px: f32,
    spread_px: f32,
    faces: BTreeMap<String, FaceFile>,
}

#[derive(Deserialize)]
struct IconsFile {
    icons: BTreeMap<String, [f32; 4]>,
    trim: BTreeMap<String, [f32; 4]>,
}

#[derive(Deserialize)]
struct PortraitsFile {
    image: String,
    size: [f32; 2],
    portraits: BTreeMap<String, [f32; 4]>,
    select: BTreeMap<String, SelectFile>,
}

#[derive(Deserialize)]
struct AtlasFile {
    image: String,
    size: [f32; 2],
    cells: BTreeMap<String, [f32; 4]>,
}

#[derive(Deserialize)]
struct SelectFile {
    image: String,
    size: [f32; 2],
}

pub(super) const PICTURE_SLOTS: usize = 9;

type Pictures = (Vec<Picture>, Vec<String>, BTreeMap<String, String>);

#[derive(Deserialize)]
struct MapFile {
    image: String,
    size: [f32; 2],
    px_per_m: f32,
    origin_m: [f32; 2],
}

fn face_of(file: FaceFile, name: &str) -> Result<Face, String> {
    let mut glyphs: Vec<Option<Glyph>> = Vec::new();
    for (key, glyph) in file.glyphs {
        let code: usize = key
            .parse()
            .map_err(|_| format!("faces.{name}.glyphs: `{key}` is not a decimal code point"))?;
        if glyphs.len() <= code {
            glyphs.resize(code + 1, None);
        }
        glyphs[code] = Some(Glyph {
            x: glyph.x,
            y: glyph.y,
            w: glyph.w,
            h: glyph.h,
            offset_x: glyph.offset_x,
            offset_y: glyph.offset_y,
            advance: glyph.advance,
        });
    }
    Ok(Face {
        line_height: file.line_height,
        ascent: file.ascent,
        descent: file.descent,
        cap: file.cap_height,
        glyphs,
    })
}

fn font_of(text: &str, thin_faces: &[String]) -> Result<(Font, Vec<String>), String> {
    let file: FontFile = serde_json::from_str(text).map_err(|error| error.to_string())?;
    if !(file.em_px > 0.0 && file.spread_px > 0.0) {
        return Err("em_px and spread_px must be above 0".to_owned());
    }
    let mut names = Vec::new();
    let mut faces = Vec::new();
    let mut thin = Vec::new();
    for (name, found) in file.faces {
        faces.push(Some(face_of(found, &name)?));
        thin.push(thin_faces.contains(&name));
        names.push(name);
    }
    Ok((
        Font {
            em_px: file.em_px,
            spread_px: file.spread_px,
            faces,
            thin,
        },
        names,
    ))
}
fn inputs(names: &[String]) -> Result<Vec<Input>, String> {
    names
        .iter()
        .map(|name| {
            serde_json::from_value::<Input>(Value::String(name.clone()))
                .map_err(|error| error.to_string())
        })
        .collect()
}

fn spoken(path: &[Seg]) -> String {
    let parts: Vec<String> = path
        .iter()
        .map(|seg| match seg {
            Seg::Key(name) => name.clone(),
            Seg::Index(index) => index.to_string(),
        })
        .collect();
    parts.join(".")
}

fn text_of<'a>(bytes: &'a [u8], uri: &str) -> Result<&'a str, String> {
    std::str::from_utf8(bytes).map_err(|_| format!("{uri} is not UTF-8"))
}

fn above_zero(value: f32) -> bool {
    value > 0.0
}

fn file_name(uri: &str) -> &str {
    uri.rsplit(['/', '\\']).next().unwrap_or(uri)
}

fn json_of<T: serde::de::DeserializeOwned>(bytes: &[u8], uri: &str) -> Result<T, String> {
    serde_json::from_str(text_of(bytes, uri)?).map_err(|error| format!("{uri}: {error}"))
}

impl ScreenSource {
    fn files(&self) -> Vec<String> {
        vec![
            self.layout.clone(),
            self.icons.clone(),
            self.font.clone(),
            self.minimap.clone(),
            self.portraits.clone(),
        ]
        .into_iter()
        .chain(self.atlases.iter().cloned())
        .collect()
    }

    pub(super) fn loaded(&self) -> Option<&Loaded> {
        self.loaded.as_deref()
    }

    fn checked(&self) -> Result<(), String> {
        let marks = [
            ("marks.hit", &self.marks.hit),
            ("marks.damage", &self.marks.damage),
        ];
        for (name, numbers) in marks {
            if !above_zero(numbers.seconds) {
                return Err(format!("{name}.seconds must be above 0"));
            }
        }
        if !above_zero(self.feed.seconds) {
            return Err("feed.seconds must be above 0".to_owned());
        }
        Ok(())
    }

    fn element_problem(
        &self,
        element: &layout::Element,
        pictures: &[Picture],
        demo: &HudFacts,
    ) -> Result<(), String> {
        let at = |problem: String| {
            format!(
                "{}: element `{}`: {problem}; skipped",
                self.layout, element.id
            )
        };
        match &element.kind {
            layout::Kind::Image(image) => {
                if let Some(name) = image.image.fixed() {
                    let Some(picture) = pictures.iter().find(|picture| picture.name == name) else {
                        return Err(at(format!("image `{name}` is none of the bound pictures")));
                    };
                    let known = |named: &str| {
                        named == file_name(&self.portraits)
                            || self.atlases.iter().any(|uri| file_name(uri) == named)
                    };
                    if let Some((named, _)) = &image.cell
                        && (!known(named) || picture.cells.is_empty())
                    {
                        return Err(at(format!(
                            "image atlas `{named}` is neither portraits.json nor one of `atlases`"
                        )));
                    }
                }
            }
            layout::Kind::Markers(markers) => {
                let wanted = file_name(&self.minimap);
                if markers.map != wanted {
                    return Err(at(format!(
                        "markers map `{}` is not `{wanted}`",
                        markers.map
                    )));
                }
                if demo.map_items(&markers.items).is_none() {
                    return Err(at(format!(
                        "`{}` is not a marker list",
                        spoken(&markers.items)
                    )));
                }
            }
            layout::Kind::Plate
            | layout::Kind::Text(_)
            | layout::Kind::Icon(_)
            | layout::Kind::Bar { .. }
            | layout::Kind::Pips { .. }
            | layout::Kind::Flow { .. } => {}
        }
        for (_, path) in element.listed() {
            if demo.value(&path) == Val::Unknown {
                return Err(at(format!("reads unknown data `{}`", spoken(&path))));
            }
        }
        Ok(())
    }
    fn pictures(
        &self,
        (map, portraits, atlases): (&MapFile, PortraitsFile, Vec<AtlasFile>),
        skipped: &mut Vec<String>,
    ) -> Pictures {
        let mut found = vec![
            Picture {
                name: map.image.clone(),
                size: map.size,
                cells: BTreeMap::new(),
            },
            Picture {
                name: portraits.image,
                size: portraits.size,
                cells: portraits.portraits,
            },
        ];
        let mut select_images = BTreeMap::new();
        for (key, select) in portraits.select {
            select_images.insert(key, select.image.clone());
            found.push(Picture {
                name: select.image,
                size: select.size,
                cells: BTreeMap::new(),
            });
        }
        for atlas in atlases {
            found.push(Picture {
                name: atlas.image,
                size: atlas.size,
                cells: atlas.cells,
            });
        }
        for (name, size) in &self.sized {
            found.push(Picture {
                name: name.clone(),
                size: *size,
                cells: BTreeMap::new(),
            });
        }
        let mut pictures = Vec::new();
        let mut textures = Vec::new();
        for picture in found {
            let Some(texture) = self.pictures.get(&picture.name) else {
                skipped.push(format!(
                    "pictures: `{}` names no scene texture; not bound",
                    picture.name
                ));
                continue;
            };
            if pictures.len() == PICTURE_SLOTS {
                skipped.push(format!("pictures: `{}` is past the {PICTURE_SLOTS} slots the screen pass binds; not bound", picture.name));
                continue;
            }
            textures.push(texture.clone());
            pictures.push(picture);
        }
        (pictures, textures, select_images)
    }
    fn read(&self, files: [&[u8]; 5], atlases: &[Arc<[u8]>]) -> Result<Loaded, String> {
        self.checked()?;
        let [layout, icons, font, map, portraits] = files;
        let mut layout = layout::parse(text_of(layout, &self.layout)?)?;
        let icons: IconsFile = json_of(icons, &self.icons)?;
        let map: MapFile = json_of(map, &self.minimap)?;
        let portraits: PortraitsFile = json_of(portraits, &self.portraits)?;
        let (font, faces) = font_of(text_of(font, &self.font)?, &self.thin_faces)
            .map_err(|error| format!("{}: {error}", self.font))?;
        layout.faces(&faces);
        let demo = demo::demo(&self.words);
        let atlases = self
            .atlases
            .iter()
            .zip(atlases)
            .map(|(uri, bytes)| json_of::<AtlasFile>(bytes, uri))
            .collect::<Result<Vec<_>, _>>()?;
        let (pictures, picture_textures, select_images) =
            self.pictures((&map, portraits, atlases), &mut layout.skipped);
        layout.retain(|element| self.element_problem(element, &pictures, &demo));
        layout.number();
        let keys = &self.keys;
        let menu = &self.menu.keys;
        Ok(Loaded {
            layout,
            font,
            icons: Icons {
                cells: icons.icons,
                trims: icons.trim,
            },
            map: MapFrame {
                px_per_m: map.px_per_m,
                origin_m: map.origin_m,
                size: map.size,
            },
            pictures,
            picture_textures,
            select_images,
            keys: Keys {
                take: inputs(&keys.take)?,
                give_back: inputs(&keys.give_back)?,
                pick: inputs(&keys.pick)?,
            },
            menu_keys: menu::MenuInputs {
                confirm: inputs(&menu.confirm)?,
                back: inputs(&menu.back)?,
                up: inputs(&menu.up)?,
                down: inputs(&menu.down)?,
                left: inputs(&menu.left)?,
                right: inputs(&menu.right)?,
                levels: inputs(&menu.levels)?,
            },
            demo,
        })
    }
    fn prepared(mut self, cx: &SceneCx<'_>, path: &str) -> Result<Self, Error> {
        let mut files: Vec<Arc<[u8]>> = Vec::with_capacity(5 + self.atlases.len());
        let named = [
            ("layout", &self.layout),
            ("icons", &self.icons),
            ("font", &self.font),
            ("minimap", &self.minimap),
            ("portraits", &self.portraits),
        ];
        for (field, uri) in named {
            if let Some(bytes) = cx.buffer(uri, &format!("{path}.{field}"))? {
                files.push(bytes);
            }
        }
        for (index, uri) in self.atlases.iter().enumerate() {
            if let Some(bytes) = cx.buffer(uri, &format!("{path}.atlases[{index}]"))? {
                files.push(bytes);
            }
        }
        let [layout, icons, font, map, portraits, atlases @ ..] = files.as_slice() else {
            return Ok(self);
        };
        if atlases.len() != self.atlases.len() {
            return Ok(self);
        }
        let loaded = self
            .read([layout, icons, font, map, portraits], atlases)
            .map_err(|problem| Error::new(ErrorCode::InvalidRecord, path).got(problem))?;
        self.loaded = Some(Arc::new(loaded));
        Ok(self)
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Showing {
    #[default]
    Auto,
    Hud,
    Buy,
    Scoreboard,
    RoundEnd,
    MatchEnd,
    Title,
    CharacterSelect,
    Loading,
    Pause,
    Settings,
}

settings! {
    #[section("screen_show")]
    #[serde(deny_unknown_fields)]
    pub struct ScreenSetup {
        pub show: Showing = Showing::Auto => any,
        pub demo: bool = false => any,
    }
}

stats! {
    #[section("screen")]
    pub struct ScreenStats {
        pub shown: String,
        pub live: bool,
        pub quads: u32,
        pub bytes_written: u64,
        pub covered_pixels: u64,
        pub texts: u32,
        pub glyphs: u32,
        pub glyphs_missing: u32,
        pub icons_missing: u32,
        pub texts_fitted: u32,
        pub pointer: Option<[f32; 2]>,
        pub hovered: Option<String>,
        pub buys: u32,
        pub elements: u32,
        pub screens: u32,
        pub screens_unread: u32,
        pub elements_skipped: u32,
        pub skipped: Vec<String>,
        pub flow: String,
        pub flow_settings_open: bool,
        pub character: String,
        pub level: String,
        pub quit_asked: u32,
        pub options_problem: Option<String>,
        pub problem: Option<String>,
    }
}

#[derive(Resource, Default)]
pub(super) struct ScreenBook {
    applied: Option<Arc<Loaded>>,
    previous: Vec<Quad>,
    previous_size: Vec2,
    buys: u32,
    hovered: Option<(layout::CardList, usize)>,
    pointer: Option<Vec2>,
    pulses: Vec<paint::PulseState>,
}

pub struct ScreenBrick;

impl Brick for ScreenBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<ScreenSource>("screen")
                    .schema(engine_core::schema_of::<ScreenSource>)
                    .files(ScreenSource::files)
                    .prepare(ScreenSource::prepared),
            )
            .settings::<ScreenSetup>()
            .stats::<ScreenStats>()
            .resource::<HudFacts>()
            .resource::<ScreenDraw>()
            .resource::<ScreenState>()
            .resource::<ScreenBuys>()
            .resource::<ScreenBook>()
            .resource::<facts::Recent>()
            .settings::<menu::Options>()
            .resource::<menu::OptionsWrite>()
            .resource::<menu::MenuState>()
            .part::<gpu::ScreenGpu>()
            .systems(Schedule::FixedSimulation, facts::collect)
            .systems(Schedule::Presentation, facts::fill)
            .systems(Schedule::Presentation, more::fill_more)
            .systems(Schedule::Presentation, more::fill_menu)
            .systems(Schedule::Presentation, control::control)
            .systems(Schedule::Presentation, draw_screen)
            .systems(Schedule::Presentation, report_flow)
            .systems(Schedule::Presentation, menu::write_options);
    }
}

fn adopt(draw: &mut ScreenDraw, book: &mut ScreenBook, loaded: Option<&Arc<Loaded>>) {
    let same = match (&book.applied, loaded) {
        (Some(held), Some(given)) => Arc::ptr_eq(held, given),
        (None, None) => true,
        (Some(_), None) | (None, Some(_)) => false,
    };
    if same {
        return;
    }
    book.applied = loaded.cloned();
    book.pulses.clear();
    book.pulses.resize_with(
        loaded.map_or(0, |loaded| loaded.layout.elements()),
        paint::PulseState::default,
    );
    draw.atlases = None;
    match loaded {
        Some(loaded) => {
            draw.reference = Vec2::from(loaded.layout.reference);
            draw.min_text_px = loaded.layout.min_text_px;
            draw.font = Some(loaded.font.clone());
            draw.icons = loaded.icons.clone();
            draw.pictures.clone_from(&loaded.pictures);
        }
        None => {
            draw.font = None;
            draw.icons = Icons::default();
        }
    }
}

fn atlases(
    draw: &mut ScreenDraw,
    source: &ScreenSource,
    (loaded, scene): (&Loaded, Option<&SceneContext>),
) -> Option<String> {
    if draw.atlases.is_some() {
        return None;
    }
    let textures = &scene?.assets.textures;
    let found = |id: &str| textures.get(id).copied();
    let (Some(icons), Some(font)) = (found(&source.icons_texture), found(&source.font_texture))
    else {
        return Some(format!(
            "scene textures `{}` and `{}` are not both in the scene",
            source.icons_texture, source.font_texture
        ));
    };
    let mut bound = [icons; 2 + PICTURE_SLOTS];
    bound[1] = font;
    for (slot, id) in loaded.picture_textures.iter().enumerate() {
        let Some(handle) = found(id) else {
            return Some(format!("scene texture `{id}` is not in the scene"));
        };
        bound[2 + slot] = handle;
    }
    let first = bound[2];
    for unused in &mut bound[2 + loaded.picture_textures.len()..] {
        *unused = first;
    }
    draw.atlases = Some(bound);
    None
}
type Shared<'w, 's> = (
    Query<'w, 's, &'static ScreenSource>,
    Option<Res<'w, SceneContext>>,
    Res<'w, Views>,
    Res<'w, HudFacts>,
    Res<'w, ScreenState>,
    Option<Res<'w, Flow>>,
    Res<'w, FrameTime>,
);

fn draw_screen(
    (sources, scene, views, facts, state, flow, time): Shared,
    mut draw: ResMut<ScreenDraw>,
    mut book: ResMut<ScreenBook>,
    mut stats: ResMut<ScreenStats>,
    (mut scratch, mut spare): (Local<String>, Local<String>),
) {
    let shown = views.frame().display_size;
    let size = Vec2::new(shown.x as f32, shown.y as f32);
    let source = sources.iter().next();
    let draw = &mut *draw;
    let book = &mut *book;
    adopt(draw, book, source.and_then(|source| source.loaded.as_ref()));
    let mut problem = match source {
        None => Some("no entity carries the `screen` component".to_owned()),
        Some(source) if source.loaded.is_none() => {
            Some("the screen files are not read yet".to_owned())
        }
        Some(source) => match source.loaded.as_deref() {
            Some(loaded) => atlases(draw, source, (loaded, scene.as_deref())),
            None => None,
        },
    };
    draw.begin(size);
    if let (Some(source), Some(applied)) = (source, book.applied.clone()) {
        let mut painter = paint::Painter {
            draw: &mut *draw,
            facts: &facts,
            layout: &applied.layout,
            map: &applied.map,
            scratch: &mut scratch,
            spare: &mut spare,
            pulses: &mut book.pulses,
            now: time.real_elapsed,
            zoom: 1.0,
        };
        let behind_menu = matches!(state.shown, Shown::Title | Shown::Select)
            || (state.shown == Shown::Settings
                && flow
                    .as_deref()
                    .is_some_and(|flow| !flow.stage.holds_match()));
        if behind_menu {
            painter.backdrop(&source.menu.backdrop);
        }
        if facts.scope > 0 && state.shown == Shown::Hud {
            painter.scope(&source.menu.scope);
        }
        painter.screen(state.shown, facts.spread_px);
        if matches!(state.shown, Shown::Hud | Shown::RoundEnd) {
            painter.marks(&source.marks);
        }
    }
    let changed = draw.quads != book.previous || size != book.previous_size;
    if changed {
        book.previous.clone_from(&draw.quads);
        book.previous_size = size;
        draw.revision += 1;
    }
    let stats = &mut *stats;
    live::set(&mut stats.shown, state.shown.key());
    stats.live = facts.live;
    stats.quads = draw.quads.len() as u32;
    stats.bytes_written = if changed {
        16 + (draw.quads.len() * size_of::<Quad>()) as u64
    } else {
        0
    };
    stats.covered_pixels = draw.covered_pixels();
    stats.texts = draw.texts;
    stats.glyphs = draw.glyphs;
    stats.glyphs_missing = draw.glyphs_missing;
    stats.icons_missing = draw.icons_missing;
    stats.texts_fitted = draw.fitted;
    stats.pointer = book.pointer.map(|pointer| [pointer.x, pointer.y]);
    match book
        .hovered
        .and_then(|hovered| control::card(&facts, hovered))
    {
        Some(card) => live::set(stats.hovered.get_or_insert_with(String::new), &card.name),
        None => stats.hovered = None,
    }
    stats.buys = book.buys;
    let applied = book.applied.as_deref();
    stats.elements = applied.map_or(0, |loaded| loaded.layout.elements() as u32);
    stats.screens = applied.map_or(0, |loaded| loaded.layout.screens.len() as u32);
    stats.screens_unread = applied.map_or(0, |loaded| loaded.layout.unread.len() as u32);
    let skipped = applied.map_or(&[][..], |loaded| loaded.layout.skipped.as_slice());
    stats.elements_skipped = skipped.len() as u32;
    if stats.skipped.as_slice() != skipped {
        stats.skipped = skipped.to_vec();
    }
    if stats.problem != problem {
        stats.problem = problem.take();
    }
}

type FlowReads<'w> = (
    Option<Res<'w, Flow>>,
    Res<'w, menu::MenuState>,
    Res<'w, menu::OptionsWrite>,
);

fn report_flow((flow, state, writes): FlowReads, mut stats: ResMut<ScreenStats>) {
    let stats = &mut *stats;
    if let Some(flow) = flow.as_deref() {
        live::set(&mut stats.flow, flow.stage.key());
        stats.flow_settings_open = flow.settings_open;
        live::set(&mut stats.character, &flow.character);
        live::set(&mut stats.level, &flow.level);
    }
    stats.quit_asked = state.quit_asked;
    if stats.options_problem != writes.problem {
        stats.options_problem.clone_from(&writes.problem);
    }
}
