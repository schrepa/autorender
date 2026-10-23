//! Tiles' shading: the control values and smooth surface's gradients a
//! terrain's tiles of level 1 and coarser shade with, level 0's averaged
//! ([`TileShading`]), so the ground seen far away is the ground seen near,
//! averaged: its materials, how wet it is and how its slopes turn to the
//! light. Each sample of a level-1 tile takes the tent average of the
//! level-0 samples within one of its spacings, weights 1, 2 and 1 along each
//! axis, over those within the terrain: their control values as the
//! surface's rules give the level-0 tiles' samples, from the heights and
//! rivers those tiles' records hold, and the gradients those heights give.
//! Each coarser level's samples take the same average of the level finer's
//! shading. Every sum runs in a fixed order, materials' weights and wetness
//! in whole numbers, so every host makes the same bits.

use std::ops::Range;
use std::sync::Arc;

use crate::codec::{self, CodecError, Reader, Writer};
use crate::surface::smooth_gradient;
use crate::tile::{SurfaceInputs, control_row};
use crate::{
    BiomeRules, Control, Generator, Layers, Pyramid, Stamp, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE,
    Tile, TileKey,
};

/// Samples along each side of a tile's shading, its edges included, and
/// its samples.
const SIDE: usize = TILE_SAMPLES as usize;
const AREA: usize = SIDE * SIDE;

/// The level-0 samples along each side a level-1 tile's shading reads: its
/// own at twice its spacing, and one past them on each side.
const WINDOW: usize = 2 * TILE_QUADS as usize + 3;

/// A tile's shading: its key, and at each of its samples (0 to 256 along x
/// and z, rows along z) its packed control value ([`Control::packed`]) and
/// the gradient of its smooth surface, its rise per meter along x and z as
/// the two halves of a word, the low one along x: as the GPU packs a normal
/// map's texel.
#[derive(Clone, Debug, PartialEq)]
pub struct TileShading {
    key: TileKey,
    control: Box<[u32]>,
    gradients: Box<[u32]>,
}

impl TileShading {
    /// Whether a terrain's tiles of `level` shade with their shading, level
    /// 0's averaged: those of level 1 and coarser, but for a terrain's
    /// surroundings (`around`), whose ground lies mostly where no level-0
    /// tile stands and whose tiles shade with their own samples, as level-0
    /// tiles do.
    pub fn shades(level: u8, around: bool) -> bool {
        level > 0 && !around
    }

    /// The tiles of the level finer than tile `key` (of level 1 or coarser)
    /// of `pyramid` whose samples its shading averages: the 4 × 4 around its
    /// 2 × 2 children, as its edge samples read one sample into the tiles
    /// around them, those within the pyramid, row by row along z. None for
    /// a level-0 tile.
    pub fn reads(pyramid: &Pyramid, key: TileKey) -> Vec<TileKey> {
        let Some(finer) = key.level.checked_sub(1) else {
            return Vec::new();
        };
        let [columns, rows] = pyramid.tiles(u32::from(finer));
        let span = |at: u16, count: u32| {
            let first = (2 * i64::from(at) - 1).max(0);
            let last = (2 * i64::from(at) + 2).min(i64::from(count) - 1);
            first..=last
        };
        let mut tiles = Vec::with_capacity(16);
        for z in span(key.z, rows) {
            for x in span(key.x, columns) {
                tiles.push(TileKey::new(finer, x as u16, z as u16));
            }
        }
        tiles
    }

    pub fn key(&self) -> TileKey {
        self.key
    }

    /// Each sample's packed control value, rows along z.
    pub fn control(&self) -> &[u32] {
        &self.control
    }

    /// Each sample's gradient as the two halves of a word, rows along z.
    pub fn gradients(&self) -> &[u32] {
        &self.gradients
    }

    /// The control value of sample `(x, z)`, the nearest sample's past the
    /// tile's.
    pub fn control_at(&self, x: i32, z: i32) -> Control {
        let last = SIDE as i32 - 1;
        let (x, z) = (x.clamp(0, last) as usize, z.clamp(0, last) as usize);
        Control::unpack(self.control[z * SIDE + x])
    }

    /// Its control values as a tile's layers hold them: its samples with a
    /// border of one, each border texel its nearest sample's, row by row.
    pub fn control_texels(&self) -> Box<[u32]> {
        stored(&self.control)
    }

    /// Its gradients as a tile's normal maps hold them: its samples with a
    /// border of one, each border texel its nearest sample's, row by row.
    pub fn gradient_texels(&self) -> Box<[u32]> {
        stored(&self.gradients)
    }

    /// Bytes it holds in memory.
    pub fn bytes(&self) -> u64 {
        ((self.control.len() + self.gradients.len()) * 4) as u64
    }
}

/// `plane`, a value at each of a tile's samples, laid out as a tile stores
/// its samples ([`TILE_STRIDE`] a row): each border texel its nearest
/// sample's.
fn stored(plane: &[u32]) -> Box<[u32]> {
    let stride = TILE_STRIDE as usize;
    (0..stride * stride)
        .map(|at| {
            let x = (at % stride).clamp(1, SIDE) - 1;
            let z = (at / stride).clamp(1, SIDE) - 1;
            plane[z * SIDE + x]
        })
        .collect()
}

/// A tile's shading travels as its key, then its control values' four
/// bytes as planes and its gradients' two halves as planes of 16-bit
/// values, each written as its differences from what its neighbours
/// predict ([`codec::put_planes`]), which smooth ground and the long runs
/// of one material leave mostly zero for the cook's compression.
impl codec::Codec for TileShading {
    fn write(&self, out: &mut Writer) {
        out.put(&self.key);
        let mut bytes = vec![0_u8; 4 * AREA];
        for (at, value) in self.control.iter().enumerate() {
            for (plane, byte) in value.to_le_bytes().into_iter().enumerate() {
                bytes[plane * AREA + at] = byte;
            }
        }
        codec::put_planes(out, &bytes, SIDE);
        let mut halves = vec![0_u16; 2 * AREA];
        for (at, &word) in self.gradients.iter().enumerate() {
            halves[at] = word as u16;
            halves[AREA + at] = (word >> 16) as u16;
        }
        codec::put_wide_planes(out, &halves, SIDE);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        let key = input.get()?;
        let bytes = codec::get_planes(input, SIDE)?;
        let halves = codec::get_wide_planes(input, SIDE)?;
        if bytes.len() != 4 * AREA || halves.len() != 2 * AREA {
            return Err(CodecError("invalid tile shading"));
        }
        let control = (0..AREA)
            .map(|at| {
                u32::from_le_bytes([
                    bytes[at],
                    bytes[AREA + at],
                    bytes[2 * AREA + at],
                    bytes[3 * AREA + at],
                ])
            })
            .collect();
        let gradients = (0..AREA)
            .map(|at| u32::from(halves[at]) | u32::from(halves[AREA + at]) << 16)
            .collect();
        Ok(Self {
            key,
            control,
            gradients,
        })
    }
}

/// The half-precision bits nearest `value`, ties to even (beyond the
/// largest half, infinity): each half of a normal map's texel, as the GPU's
/// `pack2x16float` writes it.
fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        // Infinity, or a quiet NaN.
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x200 };
    }
    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased < -25 {
        return sign;
    }
    // Normal halves keep the mantissa's top ten bits under their exponent;
    // subnormal ones (below 2^-14) shift the whole mantissa, its implicit
    // bit with it, into units of 2^-24.
    let (full, shift, base) = if unbiased >= -14 {
        (mantissa, 13, ((unbiased + 15) as u32) << 10)
    } else {
        (mantissa | 0x80_0000, (-1 - unbiased) as u32, 0)
    };
    let kept = base | (full >> shift);
    let rest = full & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    // A carry out of the mantissa raises the exponent, as it should.
    let rounded = kept + u32::from(rest > half || (rest == half && (kept & 1) == 1));
    sign | rounded as u16
}

/// The value of half-precision `bits`, as the GPU's `unpack2x16float`
/// reads each half of a normal map's texel.
fn half_value(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from(bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x3ff);
    match exponent {
        // Subnormal: the mantissa's units are 2^-24, exactly.
        0 => f32::from_bits(sign | (mantissa as f32 * (1.0 / 16_777_216.0)).to_bits()),
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)),
        _ => f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13)),
    }
}

/// The gradient `[x, z]` as a normal map's texel: its two halves.
fn gradient_word([x, z]: [f32; 2]) -> u32 {
    u32::from(half_bits(x)) | u32::from(half_bits(z)) << 16
}

/// The gradient a normal map's texel `word` holds.
fn word_gradient(word: u32) -> [f32; 2] {
    [half_value(word as u16), half_value((word >> 16) as u16)]
}

/// The samples of the level finer whose tent average gives a sample at
/// `sample` (along an axis, counted from the terrain's first corner at its
/// level), `last` the last sample of the level finer within the terrain,
/// and their weights: 1, 2 and 1 over those within the terrain, or the
/// nearest within it alone where none is; and how many there are.
fn taps(sample: i64, last: i64) -> ([(i64, u32); 3], usize) {
    let mut taps = [(0, 0); 3];
    let mut count = 0;
    for (at, weight) in [(2 * sample - 1, 1), (2 * sample, 2), (2 * sample + 1, 1)] {
        if (0..=last).contains(&at) {
            taps[count] = (at, weight);
            count += 1;
        }
    }
    if count == 0 {
        taps[0] = ((2 * sample).clamp(0, last), 1);
        count = 1;
    }
    (taps, count)
}

/// A sample's taps summed: each material's weight (its taps' weights times
/// its share of each, in 255ths), the taps' weights together, their
/// wetness times their weights, and their gradients times their weights.
#[derive(Default)]
struct Sums {
    materials: Vec<(u8, u32)>,
    total: u32,
    wet: u32,
    gradient: [f32; 2],
}

impl Sums {
    fn clear(&mut self) {
        self.materials.clear();
        (self.total, self.wet, self.gradient) = (0, 0, [0.0; 2]);
    }

    /// Adds a tap of `weight` holding `control` and `gradient`: its first
    /// material at its weight times one less its blend, its second at its
    /// weight times its blend.
    fn add(&mut self, control: Control, gradient: [f32; 2], weight: u32) {
        let blend = u32::from(control.blend);
        for (material, share) in [(control.a, 255 - blend), (control.b, blend)] {
            let part = weight * share;
            if part == 0 {
                continue;
            }
            match self.materials.iter_mut().find(|(held, _)| *held == material) {
                Some((_, sum)) => *sum += part,
                None => self.materials.push((material, part)),
            }
        }
        self.total += weight;
        self.wet += weight * u32::from(control.wet);
        let scale = weight as f32;
        self.gradient = [
            self.gradient[0] + scale * gradient[0],
            self.gradient[1] + scale * gradient[1],
        ];
    }

    /// What the taps average to: the two heaviest materials (ties to the
    /// lower index, so every host picks the same), the second's share of
    /// the two and the mean wetness, packed, each rounded to the nearest
    /// 255th; and the mean gradient as a normal map's texel.
    fn mean(&self) -> (u32, u32) {
        let heavier = |a: (u8, u32), b: (u8, u32)| a.1 > b.1 || (a.1 == b.1 && a.0 < b.0);
        let (mut first, mut second): (Option<(u8, u32)>, Option<(u8, u32)>) = (None, None);
        for &material in &self.materials {
            if first.is_none_or(|held| heavier(material, held)) {
                second = first;
                first = Some(material);
            } else if second.is_none_or(|held| heavier(material, held)) {
                second = Some(material);
            }
        }
        let (a, heaviest) = first.unwrap_or((0, 0));
        let (b, next) = second.unwrap_or((a, 0));
        let pair = heaviest + next;
        let blend = if pair > 0 {
            (255 * next + pair / 2) / pair
        } else {
            0
        };
        let total = self.total.max(1);
        let control = Control {
            a,
            b,
            blend: blend as u8,
            wet: ((self.wet + total / 2) / total) as u8,
        };
        let weight = total as f32;
        let gradient = [self.gradient[0] / weight, self.gradient[1] / weight];
        (control.packed(), gradient_word(gradient))
    }
}

/// Tile `key`'s shading over `pyramid` from the samples of the level finer:
/// `sample` gives each one's packed control value and gradient by its place
/// on that level (x and z, counted from the terrain's first corner). Each
/// sample of the tile is the tent average of those within one of its
/// spacings ([`taps`]), summed in their order, rows along z first.
fn averaged(
    pyramid: &Pyramid,
    key: TileKey,
    sample: &dyn Fn(i64, i64) -> (u32, [f32; 2]),
) -> TileShading {
    let finer = u32::from(key.level).saturating_sub(1);
    let quads = i64::from(TILE_QUADS);
    let base = pyramid.base_tiles();
    let first = [i64::from(key.x) * quads, i64::from(key.z) * quads];
    let along = |axis: usize| -> Vec<([(i64, u32); 3], usize)> {
        let last = (quads * i64::from(base[axis])) >> finer;
        (0..SIDE as i64)
            .map(|index| taps(first[axis] + index, last))
            .collect()
    };
    let (columns, rows) = (along(0), along(1));
    let mut control = Vec::with_capacity(AREA);
    let mut gradients = Vec::with_capacity(AREA);
    let mut sums = Sums::default();
    for (row, row_count) in &rows {
        for (column, column_count) in &columns {
            sums.clear();
            for &(z, weight_z) in &row[..*row_count] {
                for &(x, weight_x) in &column[..*column_count] {
                    let (value, gradient) = sample(x, z);
                    sums.add(Control::unpack(value), gradient, weight_x * weight_z);
                }
            }
            let (value, gradient) = sums.mean();
            control.push(value);
            gradients.push(gradient);
        }
    }
    TileShading {
        key,
        control: control.into_boxed_slice(),
        gradients: gradients.into_boxed_slice(),
    }
}

/// Making a tile's shading (the job [`crate::jobs::ShadingJob`]) over a
/// pyramid: a level-1 tile's from the level-0 tiles it reads, a coarser
/// tile's from the shading of those of the level finer it reads.
pub(crate) struct ShadingMaking {
    pyramid: Pyramid,
    key: TileKey,
    reads: Reads,
}

/// What a tile's shading is made from: a level-1 tile's ground, read tile by
/// tile; or the shading of the level finer, by its place among the 4 × 4
/// tiles around the tile's children ([`TileShading::reads`]), none where a
/// tile lies outside the pyramid.
enum Reads {
    Ground(Box<Ground>),
    Finer(Box<[Option<TileShading>; 16]>),
}

/// A level-1 tile's ground as its shading reads it: the stack its tiles
/// generate from (with its bakes) and the surface's rules; the level-0
/// tiles it reads, each with the edits reaching it, and the next to
/// generate; the one generating, over its layers under those edits; the one
/// whose samples it reads, a row at a time; the samples read so far; and a
/// row's inputs to the rules, reused.
struct Ground {
    layers: Arc<Layers>,
    rules: Arc<BiomeRules>,
    tiles: Vec<(TileKey, Vec<Stamp>)>,
    next: usize,
    generating: Option<(Arc<Layers>, Generator)>,
    reading: Option<Reading>,
    window: Window,
    inputs: SurfaceInputs,
}

/// A level-0 tile whose samples a level-1 tile reads: the tile as its record
/// holds it, the layers it generated from, each stored column's x in their
/// frame, the samples it gives along x and z (counted from the terrain's
/// first corner) and the next row of them to read.
struct Reading {
    tile: Tile,
    layers: Arc<Layers>,
    xs: Box<[f32]>,
    columns: Range<i64>,
    rows: Range<i64>,
    next: i64,
}

/// The level-0 samples a level-1 tile's shading reads, [`WINDOW`] along x
/// and z from its first sample's less one (counted from the terrain's first
/// corner), row by row along z: each one's packed control value and
/// gradient once read.
struct Window {
    first: [i64; 2],
    control: Vec<u32>,
    gradients: Vec<[f32; 2]>,
}

impl Window {
    /// Where sample `(x, z)` lies in the window.
    fn at(&self, x: i64, z: i64) -> usize {
        (z - self.first[1]) as usize * WINDOW + (x - self.first[0]) as usize
    }
}

impl ShadingMaking {
    /// Level-1 tile `key`'s shading over `pyramid` from the level-0 tiles
    /// `tiles` (those it reads, [`TileShading::reads`], each with the edits
    /// reaching it), generated from `layers` and given control values by
    /// `rules`.
    pub(crate) fn from_ground(
        (layers, rules): (Arc<Layers>, Arc<BiomeRules>),
        pyramid: Pyramid,
        key: TileKey,
        tiles: Vec<(TileKey, Vec<Stamp>)>,
    ) -> Self {
        let quads = i64::from(TILE_QUADS);
        let first = [
            2 * i64::from(key.x) * quads - 1,
            2 * i64::from(key.z) * quads - 1,
        ];
        let ground = Ground {
            layers,
            rules,
            tiles,
            next: 0,
            generating: None,
            reading: None,
            window: Window {
                first,
                control: vec![0; WINDOW * WINDOW],
                gradients: vec![[0.0; 2]; WINDOW * WINDOW],
            },
            inputs: SurfaceInputs::default(),
        };
        Self {
            pyramid,
            key,
            reads: Reads::Ground(Box::new(ground)),
        }
    }

    /// Tile `key`'s shading over `pyramid`, of level 2 or coarser, from the
    /// shading of the tiles of the level finer it reads, `finer` (those of
    /// [`TileShading::reads`], in its order).
    pub(crate) fn from_finer(pyramid: Pyramid, key: TileKey, finer: Vec<TileShading>) -> Self {
        let mut places: Box<[Option<TileShading>; 16]> = Box::default();
        let (x0, z0) = (2 * i64::from(key.x) - 1, 2 * i64::from(key.z) - 1);
        for shading in finer {
            let at = shading.key();
            let (dx, dz) = (i64::from(at.x) - x0, i64::from(at.z) - z0);
            if (0..4).contains(&dx) && (0..4).contains(&dz) {
                places[(dz * 4 + dx) as usize] = Some(shading);
            }
        }
        Self {
            pyramid,
            key,
            reads: Reads::Finer(places),
        }
    }

    /// Works for about `budget` units (a sample generated or given its
    /// control value each): the tile's shading once done; an error where a
    /// tile it reads does not generate or read back.
    pub(crate) fn advance(&mut self, budget: &mut u64) -> Result<Option<TileShading>, String> {
        let (pyramid, key) = (self.pyramid, self.key);
        let shading = match &mut self.reads {
            Reads::Ground(ground) => {
                if !ground.advance(&pyramid, budget)? {
                    return Ok(None);
                }
                let window = &ground.window;
                averaged(&pyramid, key, &|x, z| {
                    let at = window.at(x, z);
                    (window.control[at], window.gradients[at])
                })
            }
            Reads::Finer(places) => {
                let level = u32::from(key.level) - 1;
                let [columns, rows] = pyramid.tiles(level).map(i64::from);
                let (x0, z0) = (2 * i64::from(key.x) - 1, 2 * i64::from(key.z) - 1);
                let quads = i64::from(TILE_QUADS);
                averaged(&pyramid, key, &|x, z| {
                    // The tile holding the sample, the far edge's its last.
                    let (tx, tz) = ((x / quads).min(columns - 1), (z / quads).min(rows - 1));
                    let place = ((tz - z0) * 4 + (tx - x0)) as usize;
                    let Some(shading) = places.get(place).and_then(Option::as_ref) else {
                        return (0, [0.0; 2]);
                    };
                    let at = (z - tz * quads) as usize * SIDE + (x - tx * quads) as usize;
                    (shading.control[at], word_gradient(shading.gradients[at]))
                })
            }
        };
        Ok(Some(shading))
    }
}

impl Ground {
    /// Generates and reads the level-0 tiles for about `budget` units:
    /// whether every one was read.
    fn advance(&mut self, pyramid: &Pyramid, budget: &mut u64) -> Result<bool, String> {
        loop {
            if *budget == 0 {
                return Ok(false);
            }
            if self.reading.is_some() {
                self.read_row(budget);
                continue;
            }
            let Some((key, stamps)) = self.tiles.get(self.next) else {
                return Ok(true);
            };
            let key = *key;
            if self.generating.is_none() {
                let layers = crate::jobs::stamped(&self.layers, stamps)?;
                let generator = Generator::new(layers.clone(), pyramid, key);
                self.generating = Some((layers, generator));
            }
            let Some((layers, generator)) = self.generating.as_mut() else {
                continue;
            };
            let Some(made) = generator.advance(budget) else {
                if generator.failed() {
                    return Err("the ground's heights are invalid".into());
                }
                return Ok(false);
            };
            let layers = layers.clone();
            self.generating = None;
            self.next += 1;
            // As its record holds it: heights at their steps of the grid.
            let tile: Tile =
                codec::decode(&codec::encode(&made)).map_err(|error| error.to_string())?;
            self.reading = Reading::of(tile, layers, pyramid, &self.window);
        }
    }

    /// Reads the next row of the samples of the tile being read: each one's
    /// control value as the surface's rules give it and its smooth surface's
    /// gradient, into the window; the tile done once its last row is.
    fn read_row(&mut self, budget: &mut u64) {
        let Self {
            rules,
            reading,
            window,
            inputs,
            ..
        } = self;
        let Some(read) = reading.as_mut() else {
            return;
        };
        let stride = TILE_STRIDE as usize;
        let key = read.tile.key();
        let spacing = read.tile.spacing();
        let quads = i64::from(TILE_QUADS);
        let (first_x, first_z) = (i64::from(key.x) * quads, i64::from(key.z) * quads);
        // Stored rows and columns hold samples -1 to 257.
        let row = (read.next - first_z + 1) as usize;
        let columns =
            (read.columns.start - first_x + 1) as usize..(read.columns.end - first_x + 1) as usize;
        let z = read.layers.offset_samples(spacing)[1] + first_z + row as i64 - 1;
        let start = window.at(read.columns.start, read.next);
        let heights = read.tile.heights();
        control_row(
            (&**rules, &*read.layers),
            (heights, (read.tile.river_edges(), read.tile.river_beds())),
            (&read.xs[..], z as f32 * spacing, spacing),
            (row, columns.clone()),
            (&mut window.control[start..start + columns.len()], inputs),
        );
        let height = |r: usize, c: usize| heights[r * stride + c];
        for (offset, column) in columns.clone().enumerate() {
            let around = [row - 1, row, row + 1]
                .map(|r| [column - 1, column, column + 1].map(|c| height(r, c)));
            window.gradients[start + offset] = smooth_gradient(around, spacing);
        }
        *budget = budget.saturating_sub(columns.len() as u64 * rules.cost());
        read.next += 1;
        if read.next >= read.rows.end {
            *reading = None;
        }
    }
}

impl Reading {
    /// Level-0 tile `tile` of `pyramid`, generated from `layers`, as it
    /// gives `window` its samples: those it holds that the window takes,
    /// each sample of the terrain held by the tile it lies in, the far
    /// edge's by the last; `None` where it gives none.
    fn of(tile: Tile, layers: Arc<Layers>, pyramid: &Pyramid, window: &Window) -> Option<Self> {
        let key = tile.key();
        let quads = i64::from(TILE_QUADS);
        let tiles = pyramid.tiles(0);
        let span = |axis: usize, at: u16| {
            let first = i64::from(at) * quads;
            let last_tile = i64::from(at) + 1 == i64::from(tiles[axis]);
            let end = first + quads + i64::from(last_tile);
            first.max(window.first[axis])..end.min(window.first[axis] + WINDOW as i64)
        };
        let (columns, rows) = (span(0, key.x), span(1, key.z));
        if columns.is_empty() || rows.is_empty() {
            return None;
        }
        let spacing = tile.spacing();
        let first = i64::from(key.x) * quads + layers.offset_samples(spacing)[0];
        let xs = (-1..=i64::from(TILE_SAMPLES))
            .map(|index| (first + index) as f32 * spacing)
            .collect();
        Some(Self {
            tile,
            layers,
            xs,
            next: rows.start,
            columns,
            rows,
        })
    }
}
