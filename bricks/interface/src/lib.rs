#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

engine::brick!();

mod draw;
mod env;
mod gpu;
mod layout;
mod paint;
mod touch_paint;
mod value;

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use engine::prelude::*;
use engine::{Error, ErrorCode, Expected};
use engine_core::{Asset, AssetId, AssetKind, AssetKinds, Assets, Stats, Texture};
use engine_render::RenderBricks;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub use draw::{Area, Case, InterfaceDraw, Plate, Rgba, Stroke};
pub use layout::{Element, Seg};
pub use paint::{Pen, thousands};
pub use value::Val;

use draw::{BOUND_SLOTS, Face, Font, GLYPH_SLOTS, Glyph, Icons, PICTURE_SLOTS, Picture, Quad, Thin};

pub type Slot = u32;

pub trait Facts: Resource + Sized {
    type Look: Component;
    const SCREENS: &'static [&'static str];
    const PIECES: &'static [&'static str];
    fn slot(look: &Self::Look, path: &[Seg]) -> Option<Slot>;
    fn spot(look: &Self::Look, id: &str) -> Option<Slot>;
    fn read(&self, slot: Slot) -> Val<'_>;
    fn version(&self) -> u64;
    fn screen(&self) -> Option<usize>;
    fn points(&self, slot: Slot, each: &mut dyn FnMut(Point<'_>));
    fn piece(&self, look: &Self::Look, piece: usize, area: Area, pen: &mut Pen<'_>);
}

pub struct Point<'a> {
    pub pos: [f32; 2],
    pub yaw_degrees: f32,
    pub color: Option<&'a str>,
    pub fade: f32,
    pub radius_m: f32,
    pub icon: &'a str,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FontSource {
    file: Asset<FontFile>,
    texture: Asset<Texture>,
    thin_faces: Vec<String>,
    thin: Thin,
}

#[derive(Component, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Interface {
    layout: Asset<LayoutFile>,
    icons: Asset<AtlasFile>,
    maps: Vec<Asset<MapFile>>,
    icons_texture: Asset<Texture>,
    fonts: Vec<FontSource>,
    pictures: BTreeMap<String, Asset<Texture>>,
    sized: BTreeMap<String, [f32; 2]>,
    atlases: Vec<Asset<AtlasFile>>,
    sharpen_levels: f32,
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
struct AtlasFile {
    image: String,
    size: [f32; 2],
    cells: BTreeMap<String, [f32; 4]>,
}

#[derive(Deserialize)]
struct MapFile {
    image: String,
    size: [f32; 2],
    px_per_m: f32,
    origin_m: [f32; 2],
}

struct LayoutFile(serde_json::Value);

struct MapFrame {
    px_per_m: f32,
    origin_m: [f32; 2],
    size: [f32; 2],
}

struct Loaded {
    layout: Arc<layout::Layout>,
    font: Font,
    icons: Icons,
    maps: BTreeMap<String, MapFrame>,
    pictures: Vec<Picture>,
    atlases: [Asset<Texture>; BOUND_SLOTS],
    sharpen_levels: f32,
}

fn text<'a>(bytes: &'a [u8], at: &str) -> Result<&'a str, Error> {
    std::str::from_utf8(bytes)
        .map_err(|_| Error::new(ErrorCode::InvalidRecord, at.to_owned()).expected(Expected::Utf8))
}

fn json<T: DeserializeOwned>(bytes: Arc<[u8]>, at: &str) -> Result<T, Error> {
    engine::bricks::parse_json(text(&bytes, at)?).map_err(|error| error.within(at))
}

fn layout_file(bytes: Arc<[u8]>, uri: &str) -> Result<LayoutFile, Error> {
    serde_json::from_str(text(&bytes, uri)?)
        .map(LayoutFile)
        .map_err(|error| Error::new(ErrorCode::InvalidRecord, uri).got(error.to_string()))
}

fn face_of(file: &FaceFile, name: &str, (font, slot): (&FontFile, u32)) -> Result<Face, String> {
    let mut glyphs: Vec<Option<Glyph>> = Vec::new();
    for (key, glyph) in &file.glyphs {
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
        thin: None,
        glyphs,
    })
}

type Faces = (Font, Vec<(String, String)>, Vec<Asset<Texture>>);

fn fonts_of(
    fonts: &[(&FontSource, &str, &FontFile)],
    skipped: &mut Vec<String>,
) -> Result<Faces, String> {
    let mut font = Font::default();
    let mut names = Vec::new();
    let mut textures = Vec::new();
    for &(source, uri, file) in fonts {
        if textures.len() == GLYPH_SLOTS {
            skipped.push(format!(
                "fonts: `{uri}` is past the {GLYPH_SLOTS} font slots the interface pass binds; not bound"
            ));
            continue;
        }
        if !(file.em_px > 0.0 && file.spread_px > 0.0) {
            return Err(format!("{uri}: em_px and spread_px must be above 0"));
        }
        let slot = textures.len() as u32;
        for (name, found) in &file.faces {
            let mut face =
                face_of(found, name, (file, slot)).map_err(|error| format!("{uri}: {error}"))?;
            face.thin = source.thin_faces.contains(name).then_some(source.thin);
            font.faces.push(Some(face));
            names.push((file_name(uri).to_owned(), name.clone()));
        }
        textures.push(source.texture);
    }
    Ok((font, names, textures))
}

fn file_name(uri: &str) -> &str {
    uri.rsplit(['/', '\\']).next().unwrap_or(uri)
}

fn bound_atlas(pictures: &[Picture], named: &str) -> bool {
    pictures
        .iter()
        .any(|picture| picture.atlas == named && !picture.cells.is_empty())
}

fn bound(
    icons: Asset<Texture>,
    fonts: &[Asset<Texture>],
    pictures: &[Asset<Texture>],
) -> [Asset<Texture>; BOUND_SLOTS] {
    let mut bound = [icons; BOUND_SLOTS];
    let groups = [
        (1, GLYPH_SLOTS, fonts),
        (1 + GLYPH_SLOTS, PICTURE_SLOTS, pictures),
    ];
    for (start, slots, ids) in groups {
        bound[start..start + ids.len()].copy_from_slice(ids);
        let first = bound[start];
        bound[start + ids.len()..start + slots].fill(first);
    }
    bound
}

type Maps<'a> = [(&'a str, &'a MapFile)];

fn element_problem(
    element: &Element,
    (layout_uri, maps): (&str, &Maps<'_>),
    pictures: &[Picture],
) -> Result<(), String> {
    let at =
        |problem: String| format!("{layout_uri}: element `{}`: {problem}; skipped", element.id);
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
            if !maps.iter().any(|(uri, _)| file_name(uri) == markers.map) {
                return Err(at(format!(
                    "markers map `{}` is none of the bound maps",
                    markers.map
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
        | layout::Kind::Piece(_) => {}
    }
    Ok(())
}

impl Interface {
    fn pictures(
        &self,
        (maps, atlases): (&Maps<'_>, Vec<(&str, &AtlasFile)>),
        skipped: &mut Vec<String>,
    ) -> (Vec<Picture>, Vec<Asset<Texture>>) {
        let plain = |name: String, size: [f32; 2]| Picture {
            name,
            atlas: String::new(),
            size,
            cells: BTreeMap::new(),
        };
        let mut found: Vec<Picture> = maps
            .iter()
            .map(|(_, map)| plain(map.image.clone(), map.size))
            .collect();
        for (uri, atlas) in atlases {
            found.push(Picture {
                name: atlas.image.clone(),
                atlas: file_name(uri).to_owned(),
                size: atlas.size,
                cells: atlas.cells.clone(),
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
                    "pictures: `{}` is past the {PICTURE_SLOTS} slots the interface pass binds; not bound",
                    picture.name
                ));
                continue;
            }
            textures.push(*texture);
            pictures.push(picture);
        }
        (pictures, textures)
    }

    fn read<F: Facts>(&self, look: &F::Look, assets: &Assets) -> Result<Loaded, Error> {
        let uri = |id: AssetId| assets.uri(id).unwrap_or_default();
        let refused =
            |problem: String| Error::new(ErrorCode::InvalidRecord, "$.interface").got(problem);
        let file = assets.get(self.layout)?;
        let slot = |path: &[Seg]| F::slot(look, path);
        let spot = |id: &str| F::spot(look, id);
        let mut layout = layout::parse(&file.0, (F::SCREENS, F::PIECES), (&slot, &spot))
            .map_err(|problem| {
                Error::new(ErrorCode::InvalidRecord, uri(self.layout.id())).got(problem)
            })?;
        let icons = assets.get(self.icons)?;
        let maps = self
            .maps
            .iter()
            .map(|map| Ok::<_, Error>((uri(map.id()), assets.get(*map)?)))
            .collect::<Result<Vec<_>, Error>>()?;
        let fonts = self
            .fonts
            .iter()
            .map(|font| Ok::<_, Error>((font, uri(font.file.id()), assets.get(font.file)?)))
            .collect::<Result<Vec<_>, Error>>()?;
        let atlases = self
            .atlases
            .iter()
            .map(|atlas| Ok::<_, Error>((uri(atlas.id()), assets.get(*atlas)?)))
            .collect::<Result<Vec<_>, Error>>()?;
        let named = (uri(self.layout.id()), maps.as_slice());
        let (font, faces, font_textures) = fonts_of(&fonts, &mut layout.skipped).map_err(refused)?;
        layout.faces(&faces);
        let (pictures, picture_textures) = self.pictures((&maps, atlases), &mut layout.skipped);
        layout.retain(|element| element_problem(element, named, &pictures));
        layout.number();
        let frames = maps.iter().map(|(uri, map)| {
            let frame = MapFrame {
                px_per_m: map.px_per_m,
                origin_m: map.origin_m,
                size: map.size,
            };
            (file_name(uri).to_owned(), frame)
        });
        Ok(Loaded {
            layout: Arc::new(layout),
            font,
            icons: Icons {
                cells: icons.cells.clone(),
            },
            maps: frames.collect(),
            pictures,
            atlases: bound(self.icons_texture, &font_textures, &picture_textures),
            sharpen_levels: self.sharpen_levels,
        })
    }
}

#[derive(Resource, Default)]
struct InterfaceLoad {
    loaded: Option<Arc<Loaded>>,
    problem: Option<String>,
}

fn load<F: Facts>(
    sources: Query<(Ref<Interface>, &F::Look)>,
    assets: Res<Assets>,
    mut held: ResMut<InterfaceLoad>,
    mut draw: ResMut<InterfaceDraw>,
) {
    let Some((source, look)) = sources.iter().next() else {
        if held.loaded.is_some() {
            *held = InterfaceLoad::default();
            draw.layout = None;
        }
        return;
    };
    if !source.is_added() {
        return;
    }
    *held = match source.read::<F>(look, &assets) {
        Ok(loaded) => InterfaceLoad {
            loaded: Some(Arc::new(loaded)),
            problem: None,
        },
        Err(error) => InterfaceLoad {
            loaded: None,
            problem: Some(error.to_string()),
        },
    };
    draw.layout = held.loaded.as_ref().map(|loaded| Arc::clone(&loaded.layout));
}

stats! {
    #[section("interface")]
    pub struct InterfaceStats {
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
        pub elements: u32,
        pub screens: u32,
        pub screens_unread: u32,
        pub elements_skipped: u32,
        pub skipped: Vec<String>,
        pub problem: Option<String>,
    }
}

#[derive(Resource, Default)]
struct InterfaceBook {
    applied: Option<Arc<Loaded>>,
    painted: Option<(Option<u64>, FrameFacts, env::EnvFacts, Option<u64>)>,
    pulsing: bool,
    pulses: Vec<paint::PulseState>,
    rows: Vec<paint::RowState>,
    said: Vec<paint::Said>,
    held: Option<Quad>,
    opaque: bool,
}

pub struct InterfaceBrick<F>(PhantomData<fn() -> F>);

impl<F: Facts> Brick for InterfaceBrick<F> {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<Interface>("interface")
                    .schema(engine_core::schema_of::<Interface>),
            )
            .join::<AssetKinds>(AssetKind::of("interface_layout", layout_file))
            .join::<AssetKinds>(AssetKind::of("interface_map", json::<MapFile>))
            .join::<AssetKinds>(AssetKind::of("interface_font", json::<FontFile>))
            .join::<AssetKinds>(AssetKind::of("interface_atlas", json::<AtlasFile>))
            .resource::<InterfaceLoad>()
            .resource::<InterfaceDraw>()
            .resource::<InterfaceBook>()
            .part::<gpu::InterfaceGpu>()
            .systems(Schedule::BeforeTicks, load::<F>)
            .systems(Schedule::BeforeDraw, cover::<F>)
            .systems(Schedule::Presentation, draw_interface::<F>)
            .dev(|dev| {
                dev.stats::<InterfaceStats>();
            });
    }
}

fn adopt(draw: &mut InterfaceDraw, book: &mut InterfaceBook, loaded: Option<&Arc<Loaded>>) {
    let same = match (&book.applied, loaded) {
        (Some(held), Some(given)) => Arc::ptr_eq(held, given),
        (None, None) => true,
        (Some(_), None) | (None, Some(_)) => false,
    };
    if same {
        return;
    }
    book.applied = loaded.cloned();
    book.painted = None;
    draw.revision += 1;
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
    draw.atlases = loaded.map(|loaded| loaded.atlases);
    match loaded {
        Some(loaded) => {
            draw.reference = Vec2::from(loaded.layout.reference);
            draw.min_text_px = loaded.layout.min_text_px;
            draw.sharpen_levels = loaded.sharpen_levels;
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

fn cover<F: Facts>(
    book: Res<InterfaceBook>,
    views: Res<Views>,
    listed: Query<(Entity, Has<ViewSkip>), With<View>>,
    mut spawner: Spawner<InterfaceBrick<F>>,
) {
    let main = views.main().map(|facts| facts.view);
    for (view, skipped) in &listed {
        let wanted = book.opaque && Some(view) == main;
        if wanted && !skipped {
            spawner.insert(view, ViewSkip);
        } else if skipped && !wanted {
            spawner.remove::<ViewSkip>(view);
        }
    }
}

type Shared<'w, 's, F> = (
    Query<'w, 's, &'static Interface>,
    Query<'w, 's, Ref<'static, <F as Facts>::Look>>,
    Res<'w, InterfaceLoad>,
    Res<'w, Views>,
    Option<Res<'w, F>>,
    Res<'w, FrameTime>,
    Option<Res<'w, InputProfile>>,
    Option<Res<'w, engine_touch::TouchState>>,
);

fn draw_interface<F: Facts>(
    (sources, looks, load, views, facts, time, profile, touch): Shared<F>,
    mut draw: ResMut<InterfaceDraw>,
    mut book: ResMut<InterfaceBook>,
    mut stats: Stats<InterfaceStats>,
    (mut scratch, mut spare): (Local<String>, Local<String>),
    skipped: Query<(), With<ViewSkip>>,
) {
    let frame_facts = views.frame();
    let env = env::EnvFacts::of(frame_facts, profile.as_deref().copied());
    let touch_rev = touch.as_deref().map(engine_touch::TouchState::revision);
    let source = sources.iter().next();
    let look = looks.iter().next();
    let draw = &mut *draw;
    let book = &mut *book;
    adopt(draw, book, load.loaded.as_ref());
    let facts = facts.as_deref();
    let problem = match (source, &load.loaded, facts) {
        (None, _, _) => Some("no entity carries the `interface` component".to_owned()),
        (Some(_), None, _) => Some(
            load.problem
                .clone()
                .unwrap_or_else(|| "the interface files are not read yet".to_owned()),
        ),
        (Some(_), Some(_), None) => Some("no resource holds the interface facts".to_owned()),
        (Some(_), Some(_), Some(_)) => None,
    };
    let ready = source
        .and(look.as_deref())
        .zip(book.applied.clone())
        .zip(facts);
    let mark = ready.as_ref().map(|(_, facts)| facts.version());
    let key = (mark, frame_facts, env, touch_rev);
    let fresh = book.pulsing
        || book.painted != Some(key)
        || look.as_ref().is_some_and(|look| look.is_changed());
    if fresh {
        draw.begin(frame_facts);
        book.pulsing = false;
        if let Some(((look, applied), facts)) = ready
            && let Some(screen) = facts.screen()
        {
            let mut painter = paint::Painter {
                draw: &mut *draw,
                facts,
                env,
                look,
                layout: &applied.layout,
                maps: &applied.maps,
                elements: &[],
                scratch: &mut scratch,
                spare: &mut spare,
                pulses: &mut book.pulses,
                rows: &mut book.rows,
                said: &mut book.said,
                pulsing: &mut book.pulsing,
                now: time.real_elapsed,
                zoom: 1.0,
            };
            painter.screen(screen);
        }
        if let Some(touch) = touch.as_deref() {
            touch_paint::paint_touch(draw, touch);
        }
        draw.settle();
        book.painted = Some(key);
    }
    let held = book.held.filter(|_| !skipped.is_empty() && !draw.opaque);
    let changed = fresh || draw.held != held;
    draw.held = held;
    book.held = draw.floor_quad;
    book.opaque = draw.opaque;
    if changed {
        draw.revision += 1;
    }
    let count: usize = draw.parts().iter().map(|part| part.len()).sum();
    stats.edit(|stats| {
        stats.quads = count as u32;
        stats.bytes_written = if changed {
            16 + (count * size_of::<Quad>()) as u64
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
            stats.problem = problem;
        }
    });
}
