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
mod tactics;

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::prelude::*;
use engine::{Error, ErrorCode};
use engine_core::{SceneContext, SceneCx};
use engine_render::RenderBricks;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::play::camera::TestCamera;
use crate::play::flow::Flow;
use crate::play::numbers::Input;
use crate::play::screen::{
    BOUND_SLOTS, Covered, Face, Font, GLYPH_SLOTS, Glyph, HudFacts, Icons, PICTURE_SLOTS, Picture,
    Quad, ScreenBuys, ScreenDraw, ScreenState, Shown,
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
    pub kill_seconds: f32,
    pub time_seconds: f32,
    pub fuse_seconds: f32,
    pub health: f32,
    pub magazine_share: f32,
    pub buy_seconds: f32,
    pub go_seconds: f32,
    pub point_seconds: f32,
    pub lost_seconds: f32,
    pub ult_seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FeedNumbers {
    pub lines: usize,
    pub seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArmorMark {
    pub icon: String,
    pub size_px: f32,
    pub offset_px: [f32; 2],
    pub seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Marks {
    pub armor: ArmorMark,
    pub outline_color: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FontSource {
    pub file: String,
    pub texture: String,
    pub thin_faces: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Words {
    pub kills: Vec<String>,
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
    pub signature_free: String,
    pub shop_zone: String,
    pub shop_dead: String,
    pub shop_closed: String,
    pub point_kill: String,
    pub point_death: String,
    pub point_plant: String,
    pub point_defuse: String,
    pub point_orb: String,
    pub status_slowed: String,
    pub status_sight: String,
    pub status_vulnerable: String,
    pub status_revealed: String,
    pub status_boosted: String,
    pub status_chained: String,
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
    font_textures: Vec<String>,
    icons: Icons,
    map: MapFrame,
    pictures: Vec<Picture>,
    picture_textures: Vec<String>,
    keys: Keys,
    menu_keys: menu::MenuInputs,
    demo: HudFacts,
}

#[derive(Component, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScreenSource {
    pub layout: String,
    pub icons: String,
    pub minimap: String,
    pub portraits: String,
    pub icons_texture: String,
    pub fonts: Vec<FontSource>,
    pub pictures: BTreeMap<String, String>,
    pub sized: BTreeMap<String, [f32; 2]>,
    pub atlases: Vec<String>,
    pub menu: menu::MenuSource,
    pub keys: ScreenKeys,
    pub key_names: menu::KeyNames,
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
}

#[derive(Deserialize)]
struct PortraitsFile {
    image: String,
    size: [f32; 2],
    portraits: BTreeMap<String, [f32; 4]>,
}

#[derive(Deserialize)]
struct AtlasFile {
    image: String,
    size: [f32; 2],
    #[serde(alias = "icons")]
    cells: BTreeMap<String, [f32; 4]>,
}

type Pictures = (Vec<Picture>, Vec<String>);

#[derive(Deserialize)]
struct MapFile {
    image: String,
    size: [f32; 2],
    px_per_m: f32,
    origin_m: [f32; 2],
}

fn face_of(file: FaceFile, name: &str, (font, slot): (&FontFile, u32)) -> Result<Face, String> {
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
        em_px: font.em_px,
        spread_px: font.spread_px,
        slot,
        thin: false,
        glyphs,
    })
}

type Blobs<'a> = &'a [Arc<[u8]>];

type Faces = (Font, Vec<(String, String)>, Vec<String>);

fn fonts_of(
    sources: &[FontSource],
    files: &[Arc<[u8]>],
    skipped: &mut Vec<String>,
) -> Result<Faces, String> {
    let mut font = Font::default();
    let mut names = Vec::new();
    let mut textures = Vec::new();
    for (source, bytes) in sources.iter().zip(files) {
        if textures.len() == GLYPH_SLOTS {
            skipped.push(format!(
                "fonts: `{}` is past the {GLYPH_SLOTS} font slots the screen pass binds; not bound",
                source.file
            ));
            continue;
        }
        let mut file: FontFile = json_of(bytes, &source.file)?;
        if !(file.em_px > 0.0 && file.spread_px > 0.0) {
            return Err(format!(
                "{}: em_px and spread_px must be above 0",
                source.file
            ));
        }
        let slot = textures.len() as u32;
        for (name, found) in std::mem::take(&mut file.faces) {
            let mut face = face_of(found, &name, (&file, slot))
                .map_err(|error| format!("{}: {error}", source.file))?;
            face.thin = source.thin_faces.contains(&name);
            font.faces.push(Some(face));
            names.push((file_name(&source.file).to_owned(), name));
        }
        textures.push(source.texture.clone());
    }
    Ok((font, names, textures))
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

fn bound_atlas(pictures: &[Picture], named: &str) -> bool {
    pictures
        .iter()
        .any(|picture| picture.atlas == named && !picture.cells.is_empty())
}

fn json_of<T: serde::de::DeserializeOwned>(bytes: &[u8], uri: &str) -> Result<T, String> {
    serde_json::from_str(text_of(bytes, uri)?).map_err(|error| format!("{uri}: {error}"))
}

impl ScreenSource {
    fn files(&self) -> Vec<String> {
        vec![
            self.layout.clone(),
            self.icons.clone(),
            self.minimap.clone(),
            self.portraits.clone(),
        ]
        .into_iter()
        .chain(self.fonts.iter().map(|font| font.file.clone()))
        .chain(self.atlases.iter().cloned())
        .collect()
    }

    pub(super) fn loaded(&self) -> Option<&Loaded> {
        self.loaded.as_deref()
    }

    fn checked(&self) -> Result<(), String> {
        if !above_zero(self.marks.armor.seconds) {
            return Err("marks.armor.seconds must be above 0".to_owned());
        }
        if self.words.kills.is_empty() {
            return Err("words.kills must name at least one kill".to_owned());
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
                if let Some(name) = image.image.fixed()
                    && !pictures.iter().any(|picture| picture.name == name)
                {
                    return Err(at(format!("image `{name}` is none of the bound pictures")));
                }
                if let Some((named, _)) = &image.cell
                    && !bound_atlas(pictures, named)
                {
                    return Err(at(format!("atlas `{named}` is no bound picture atlas")));
                }
            }
            layout::Kind::Icon(named) if !bound_atlas(pictures, named) => {
                return Err(at(format!("atlas `{named}` is no bound picture atlas")));
            }
            layout::Kind::Flow { items, .. } => {
                for item in items {
                    if let layout::FlowKind::Icon { atlas: named, .. } = &item.kind
                        && !bound_atlas(pictures, named)
                    {
                        return Err(at(format!("atlas `{named}` is no bound picture atlas")));
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
                if let Some(named) = &markers.atlas
                    && !bound_atlas(pictures, named)
                {
                    return Err(at(format!("atlas `{named}` is no bound picture atlas")));
                }
            }
            layout::Kind::Plate
            | layout::Kind::Text(_)
            | layout::Kind::Icon(..)
            | layout::Kind::Arc(_)
            | layout::Kind::Bar { .. }
            | layout::Kind::Pips { .. }
            | layout::Kind::Reticle => {}
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
        (map, portraits, atlases): (&MapFile, PortraitsFile, Vec<(String, AtlasFile)>),
        skipped: &mut Vec<String>,
    ) -> Pictures {
        let plain = |name: String, size: [f32; 2]| Picture {
            name,
            atlas: String::new(),
            size,
            cells: BTreeMap::new(),
        };
        let mut found = vec![
            plain(map.image.clone(), map.size),
            Picture {
                name: portraits.image,
                atlas: file_name(&self.portraits).to_owned(),
                size: portraits.size,
                cells: portraits.portraits,
            },
        ];
        for (uri, atlas) in atlases {
            found.push(Picture {
                name: atlas.image,
                atlas: file_name(&uri).to_owned(),
                size: atlas.size,
                cells: atlas.cells,
            });
        }
        for (name, size) in &self.sized {
            found.push(plain(name.clone(), *size));
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
                skipped.push(format!(
                    "pictures: `{}` is past the {PICTURE_SLOTS} slots the screen pass binds; not bound",
                    picture.name
                ));
                continue;
            }
            textures.push(texture.clone());
            pictures.push(picture);
        }
        (pictures, textures)
    }

    fn read(
        &self,
        files: [&[u8]; 4],
        (fonts, atlases): (Blobs<'_>, Blobs<'_>),
    ) -> Result<Loaded, String> {
        self.checked()?;
        let [layout, icons, map, portraits] = files;
        let mut layout = layout::parse(text_of(layout, &self.layout)?)?;
        let icons: IconsFile = json_of(icons, &self.icons)?;
        let map: MapFile = json_of(map, &self.minimap)?;
        let portraits: PortraitsFile = json_of(portraits, &self.portraits)?;
        let (font, faces, font_textures) = fonts_of(&self.fonts, fonts, &mut layout.skipped)?;
        layout.faces(&faces);
        let demo = demo::demo(&self.words);
        let atlases = self
            .atlases
            .iter()
            .zip(atlases)
            .map(|(uri, bytes)| Ok((uri.clone(), json_of::<AtlasFile>(bytes, uri)?)))
            .collect::<Result<Vec<_>, String>>()?;
        let (pictures, picture_textures) =
            self.pictures((&map, portraits, atlases), &mut layout.skipped);
        layout.retain(|element| self.element_problem(element, &pictures, &demo));
        layout.number();
        let keys = &self.keys;
        let menu = &self.menu.keys;
        Ok(Loaded {
            layout,
            font,
            font_textures,
            icons: Icons { cells: icons.icons },
            map: MapFrame {
                px_per_m: map.px_per_m,
                origin_m: map.origin_m,
                size: map.size,
            },
            pictures,
            picture_textures,
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
        let mut files: Vec<Arc<[u8]>> =
            Vec::with_capacity(4 + self.fonts.len() + self.atlases.len());
        let named = [
            ("layout", &self.layout),
            ("icons", &self.icons),
            ("minimap", &self.minimap),
            ("portraits", &self.portraits),
        ];
        for (field, uri) in named {
            if let Some(bytes) = cx.buffer(uri, &format!("{path}.{field}"))? {
                files.push(bytes);
            }
        }
        for (index, font) in self.fonts.iter().enumerate() {
            if let Some(bytes) = cx.buffer(&font.file, &format!("{path}.fonts[{index}].file"))? {
                files.push(bytes);
            }
        }
        for (index, uri) in self.atlases.iter().enumerate() {
            if let Some(bytes) = cx.buffer(uri, &format!("{path}.atlases[{index}]"))? {
                files.push(bytes);
            }
        }
        let [layout, icons, map, portraits, rest @ ..] = files.as_slice() else {
            return Ok(self);
        };
        if rest.len() != self.fonts.len() + self.atlases.len() {
            return Ok(self);
        }
        let (fonts, atlases) = rest.split_at(self.fonts.len());
        let loaded = self
            .read([layout, icons, map, portraits], (fonts, atlases))
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
        pub demo_blind: f32 = 0.0 => range(0.0, 1.0),
        pub demo_channel: bool = false => any,
        pub demo_spectate: bool = false => any,
        pub demo_scope: f32 = 0.0 => range(0.0, 1.0),
        pub demo_kills: u32 = 0 => range(0, 5),
        pub demo_armor: bool = false => any,
        pub demo_poor: bool = false => any,
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
        pub covered_share: f32,
        pub opaque: bool,
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
        pub hud_hidden: bool,
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
    rows: Vec<paint::RowState>,
    held: Option<Quad>,
    opaque: bool,
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
            .resource::<Covered>()
            .resource::<facts::Recent>()
            .settings::<menu::Options>()
            .resource::<menu::OptionsWrite>()
            .resource::<menu::MenuState>()
            .part::<gpu::ScreenGpu>()
            .systems(Schedule::FixedSimulation, facts::collect)
            .systems(Schedule::Presentation, facts::fill)
            .systems(Schedule::Presentation, more::fill_more)
            .systems(Schedule::Presentation, tactics::fill_tactics)
            .systems(Schedule::Presentation, more::fill_menu)
            .systems(Schedule::Presentation, control::control)
            .systems(Schedule::FrameStart, cover)
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
    book.rows.clear();
    book.rows.resize(
        loaded.map_or(0, |loaded| loaded.layout.rows),
        paint::RowState::default(),
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
    let Some(icons) = found(&source.icons_texture) else {
        return Some(format!(
            "scene texture `{}` is not in the scene",
            source.icons_texture
        ));
    };
    let mut bound = [icons; BOUND_SLOTS];
    let groups = [
        (1, GLYPH_SLOTS, &loaded.font_textures),
        (1 + GLYPH_SLOTS, PICTURE_SLOTS, &loaded.picture_textures),
    ];
    for (start, slots, ids) in groups {
        for (slot, id) in ids.iter().enumerate() {
            let Some(handle) = found(id) else {
                return Some(format!("scene texture `{id}` is not in the scene"));
            };
            bound[start + slot] = handle;
        }
        let first = bound[start];
        for unused in &mut bound[start + ids.len()..start + slots] {
            *unused = first;
        }
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
    Res<'w, FrameTime>,
    Option<Res<'w, TestCamera>>,
);

const IN_ROUND: [Shown; 4] = [Shown::Hud, Shown::Buy, Shown::Scoreboard, Shown::RoundEnd];

fn cover(book: Res<ScreenBook>, mut covered: ResMut<Covered>) {
    covered.set_if_neq(Covered(book.opaque));
}

fn draw_screen(
    (sources, scene, views, facts, state, time, camera): Shared,
    mut draw: ResMut<ScreenDraw>,
    mut book: ResMut<ScreenBook>,
    mut stats: ResMut<ScreenStats>,
    (mut scratch, mut spare): (Local<String>, Local<String>),
    covered: Res<Covered>,
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
    let hud_hidden = camera.as_deref().is_some_and(|camera| camera.hud_hidden)
        && IN_ROUND.contains(&state.shown);
    if let (Some(source), Some(applied), false) = (source, book.applied.clone(), hud_hidden) {
        let mut painter = paint::Painter {
            draw: &mut *draw,
            facts: &facts,
            layout: &applied.layout,
            map: &applied.map,
            scratch: &mut scratch,
            spare: &mut spare,
            pulses: &mut book.pulses,
            rows: &mut book.rows,
            now: time.real_elapsed,
            zoom: 1.0,
        };
        if matches!(
            state.shown,
            Shown::Hud | Shown::RoundEnd | Shown::Scoreboard
        ) {
            painter.blind(facts.blind);
        }
        let sniping = facts.scope_ease > 0.0 && state.shown == Shown::Hud;
        if sniping {
            painter.scope(&source.menu.scope, facts.scope_ease);
        }
        painter.screen(state.shown);
        let cross = facts.menu.settings.reticle;
        let reach = cross.gap + cross.length + cross.outline + facts.spread_px;
        if state.shown == Shown::Hud && !painter.crossed(Shown::Hud, reach) {
            let center = (painter.draw.size * 0.5).round();
            if !sniping {
                painter.reticle(center, facts.spread_px);
            }
            painter.marks(&source.marks);
        }
    }
    draw.settle();
    if covered.0
        && !draw.opaque
        && let Some(held) = book.held
    {
        draw.quads.insert(0, held);
    }
    book.held = draw.floor_quad;
    book.opaque = draw.opaque;
    let changed = draw.quads != book.previous || size != book.previous_size;
    if changed {
        book.previous.clone_from(&draw.quads);
        book.previous_size = size;
        draw.revision += 1;
    }
    let stats = &mut *stats;
    live::set(&mut stats.shown, state.shown.key());
    stats.live = facts.live;
    stats.hud_hidden = hud_hidden;
    stats.quads = draw.quads.len() as u32;
    stats.bytes_written = if changed {
        16 + (draw.quads.len() * size_of::<Quad>()) as u64
    } else {
        0
    };
    stats.covered_pixels = draw.covered_pixels();
    stats.opaque = draw.opaque;
    let screen = draw.size.x * draw.size.y;
    stats.covered_share = if screen > 0.0 {
        stats.covered_pixels as f32 / screen
    } else {
        0.0
    };
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
