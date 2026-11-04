use std::hash::BuildHasher;
use std::sync::Arc;

use cosmic_text::{
    Align, Attrs, Buffer, CacheKey, Family, FontSystem, LayoutGlyph, Metrics, Shaping, Stretch,
    Style, SwashCache, SwashContent, Weight, fontdb,
};
use engine_core::collections::{FixedState, HashMap};
use engine_core::math::{max, min};
use engine_places::text::{FontId, TextAlign};

const DEFAULT_FONT: &[u8] = include_bytes!("fonts/FiraMono-subset.ttf");
const LOCALE: &str = "en-US";
const LINE_SPACING: f32 = 1.2;
const LARGEST_PIXELS: f32 = 2048.0;
const KEPT_FRAMES: u64 = 240;

struct Face {
    family: String,
    weight: Weight,
    style: Style,
    stretch: Stretch,
}

impl Face {
    fn attrs(&self) -> Attrs<'_> {
        Attrs::new()
            .family(Family::Name(&self.family))
            .weight(self.weight)
            .style(self.style)
            .stretch(self.stretch)
    }
}

pub(crate) struct Raster {
    pub(crate) left: i32,
    pub(crate) top: i32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) coverage: Vec<u8>,
}

pub(crate) struct Fonts {
    system: FontSystem,
    cache: SwashCache,
    faces: Vec<Face>,
    known: HashMap<u64, Option<FontId>>,
    failed: u32,
}

impl Fonts {
    pub(crate) fn new() -> Self {
        let mut fonts = Self {
            system: FontSystem::new_with_locale_and_db(LOCALE.to_owned(), fontdb::Database::new()),
            cache: SwashCache::new(),
            faces: Vec::new(),
            known: HashMap::default(),
            failed: 0,
        };
        if fonts.add(Arc::new(DEFAULT_FONT)).is_none() {
            fonts.failed += 1;
        }
        fonts
    }

    fn add(&mut self, bytes: Arc<dyn AsRef<[u8]> + Send + Sync>) -> Option<FontId> {
        let loaded = self
            .system
            .db_mut()
            .load_font_source(fontdb::Source::Binary(bytes));
        let face = self.system.db().face(*loaded.first()?)?;
        let family = face.families.first()?.0.clone();
        let added = Face {
            family,
            weight: face.weight,
            style: face.style,
            stretch: face.stretch,
        };
        self.faces.push(added);
        Some(FontId(self.faces.len() as u32 - 1))
    }

    pub(crate) fn load(&mut self, bytes: Arc<[u8]>) -> Option<FontId> {
        let hash = FixedState::default().hash_one(&*bytes);
        if let Some(known) = self.known.get(&hash) {
            return *known;
        }
        let added = self.add(Arc::new(bytes));
        if added.is_none() {
            self.failed += 1;
        }
        self.known.insert(hash, added);
        added
    }

    pub(crate) fn loaded(&self) -> u32 {
        self.faces.len() as u32
    }

    pub(crate) fn failed(&self) -> u32 {
        self.failed
    }

    pub(crate) fn raster(&mut self, key: CacheKey) -> Option<Raster> {
        let image = self.cache.get_image_uncached(&mut self.system, key)?;
        let placement = image.placement;
        let coverage = match image.content {
            SwashContent::Mask => image.data,
            SwashContent::SubpixelMask | SwashContent::Color => image
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|texel| texel[3])
                .collect(),
        };
        Some(Raster {
            left: placement.left,
            top: placement.top,
            width: placement.width,
            height: placement.height,
            coverage,
        })
    }

    fn shape(&mut self, text: &str, layout: Layout) -> Shaped {
        let size = f32::from_bits(layout.size);
        let mut buffer = Buffer::new_empty(Metrics::new(size, (size * LINE_SPACING).ceil()));
        buffer.set_size(layout.width.map(f32::from_bits), None);
        let face = self
            .faces
            .get(layout.font as usize)
            .or_else(|| self.faces.first());
        let attrs = face.map_or_else(Attrs::new, Face::attrs);
        buffer.set_text(text, &attrs, Shaping::Advanced, Some(aligned(layout.align)));
        buffer.shape_until_scroll(&mut self.system, false);
        let mut shaped = Shaped {
            text: text.to_owned(),
            layout,
            glyphs: Vec::new(),
            width: 0.0,
            height: 0.0,
            used: 0,
        };
        for line in buffer.layout_runs() {
            shaped.width = max(shaped.width, line.line_w);
            shaped.height = max(shaped.height, line.line_top + line.line_height);
            shaped
                .glyphs
                .extend(line.glyphs.iter().map(|glyph| placed(glyph, line.line_y)));
        }
        shaped
    }
}

fn aligned(align: TextAlign) -> Align {
    match align {
        TextAlign::Left => Align::Left,
        TextAlign::Center => Align::Center,
        TextAlign::Right => Align::Right,
    }
}

fn placed(glyph: &LayoutGlyph, baseline: f32) -> ShapedGlyph {
    let (key, x, y) = CacheKey::new(
        glyph.font_id,
        glyph.glyph_id,
        glyph.font_size,
        (
            (glyph.x + glyph.font_size * glyph.x_offset).round(),
            (glyph.y - glyph.font_size * glyph.y_offset + baseline).round(),
        ),
        glyph.font_weight,
        glyph.cache_key_flags,
    );
    ShapedGlyph { key, x, y }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Layout {
    font: u32,
    size: u32,
    width: Option<u32>,
    align: TextAlign,
}

impl Layout {
    pub(crate) fn of(
        ratio: f32,
        font: u32,
        size: f32,
        max_width: Option<f32>,
        align: TextAlign,
    ) -> Self {
        let pixels = size * ratio;
        let pixels = if pixels >= 1.0 {
            min(pixels, LARGEST_PIXELS)
        } else {
            1.0
        };
        Self {
            font,
            size: pixels.to_bits(),
            width: max_width
                .filter(|width| width.is_finite() && *width > 0.0)
                .map(|width| (width * ratio).to_bits()),
            align,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ShapedGlyph {
    pub(crate) key: CacheKey,
    pub(crate) x: i32,
    pub(crate) y: i32,
}

pub(crate) struct Shaped {
    text: String,
    layout: Layout,
    pub(crate) glyphs: Vec<ShapedGlyph>,
    pub(crate) width: f32,
    pub(crate) height: f32,
    used: u64,
}

#[derive(Default)]
pub(crate) struct Shapes {
    held: HashMap<u64, Shaped>,
}

impl Shapes {
    pub(crate) fn get(
        &mut self,
        fonts: &mut Fonts,
        text: &str,
        layout: Layout,
        frame: u64,
    ) -> &Shaped {
        let hash = FixedState::default().hash_one((text, layout));
        let shaped = self
            .held
            .entry(hash)
            .and_modify(|held| {
                if held.layout != layout || held.text != text {
                    *held = fonts.shape(text, layout);
                }
            })
            .or_insert_with(|| fonts.shape(text, layout));
        shaped.used = frame;
        shaped
    }

    pub(crate) fn sweep(&mut self, frame: u64) {
        self.held
            .retain(|_, held| frame.saturating_sub(held.used) < KEPT_FRAMES);
    }
}
