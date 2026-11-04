use std::ops::Range;
use std::sync::Arc;

use engine_codec::{self as codec, Error, Reader, Writer};
use engine_heightfield::layer::LayerKind;
use engine_heightfield::tile::{
    BiomeRules, Control, Generator, SurfaceInputs, Tile, control_row, gradient_word,
    smooth_gradient, stamped, word_gradient,
};
use engine_heightfield::{Layers, Pyramid, Stamp, TILE_QUADS, TILE_SAMPLES, TileKey};

use crate::gpu::TILE_STRIDE;

const SIDE: usize = TILE_SAMPLES as usize;
const AREA: usize = SIDE * SIDE;

const WINDOW: usize = 2 * TILE_QUADS as usize + 3;

#[derive(Clone, Debug, PartialEq)]
pub struct TileShading {
    key: TileKey,
    control: Box<[u32]>,
    gradients: Box<[u32]>,
}

impl TileShading {
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

    pub(crate) fn ground_span(pyramid: &Pyramid, key: TileKey) -> (TileKey, TileKey) {
        let scale = 1_i64 << key.level;
        let [columns, rows] = pyramid.tiles(0);
        let span = |at: u16, count: u32| {
            let at = i64::from(at);
            let first = (scale * at - (scale - 1)).max(0);
            let last = (scale * at + 2 * (scale - 1)).min(i64::from(count) - 1);
            (first as u16, last as u16)
        };
        let (first_x, last_x) = span(key.x, columns);
        let (first_z, last_z) = span(key.z, rows);
        (
            TileKey::new(0, first_x, first_z),
            TileKey::new(0, last_x, last_z),
        )
    }

    pub fn key(&self) -> TileKey {
        self.key
    }

    pub fn control(&self) -> &[u32] {
        &self.control
    }

    pub fn gradients(&self) -> &[u32] {
        &self.gradients
    }

    pub fn control_at(&self, x: i32, z: i32) -> Control {
        let last = SIDE as i32 - 1;
        let (x, z) = (x.clamp(0, last) as usize, z.clamp(0, last) as usize);
        Control::unpack(self.control[z * SIDE + x])
    }

    pub fn control_texels(&self) -> Box<[u32]> {
        stored(&self.control)
    }

    pub fn gradient_texels(&self) -> Box<[u32]> {
        stored(&self.gradients)
    }

    pub fn bytes(&self) -> u64 {
        ((self.control.len() + self.gradients.len()) * 4) as u64
    }
}

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

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
        let key = input.get()?;
        let bytes = codec::get_planes(input, SIDE)?;
        let halves = codec::get_wide_planes(input, SIDE)?;
        if bytes.len() != 4 * AREA || halves.len() != 2 * AREA {
            return Err(codec::invalid("$.shading"));
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

    fn add(&mut self, control: Control, gradient: [f32; 2], weight: u32) {
        let blend = u32::from(control.blend);
        for (material, share) in [(control.a, 255 - blend), (control.b, blend)] {
            let part = weight * share;
            if part == 0 {
                continue;
            }
            match self
                .materials
                .iter_mut()
                .find(|(held, _)| *held == material)
            {
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

    fn mean(&self) -> (u32, u32) {
        let heavier = |a: (u8, u32), b: (u8, u32)| a.1 > b.1 || (a.1 == b.1 && a.0 < b.0);
        let mut first: Option<(u8, u32)> = None;
        let mut second: Option<(u8, u32)> = None;
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
        let blend = (255 * next + pair / 2).checked_div(pair).unwrap_or(0);
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

pub(crate) struct ShadingMaking {
    pyramid: Pyramid,
    key: TileKey,
    reads: Reads,
}

enum Reads {
    Ground(Box<Ground>),
    Finer(Box<[Option<TileShading>; 16]>),
}

struct Ground {
    layers: Arc<Layers>,
    rules: Arc<BiomeRules>,
    kinds: Vec<&'static LayerKind>,
    tiles: Vec<(TileKey, Vec<Stamp>)>,
    next: usize,
    generating: Option<(Arc<Layers>, Generator)>,
    reading: Option<Reading>,
    window: Window,
    inputs: SurfaceInputs,
}

struct Reading {
    tile: Tile,
    layers: Arc<Layers>,
    xs: Box<[f32]>,
    columns: Range<i64>,
    rows: Range<i64>,
    next: i64,
}

struct Window {
    first: [i64; 2],
    control: Vec<u32>,
    gradients: Vec<[f32; 2]>,
}

impl Window {
    fn at(&self, x: i64, z: i64) -> usize {
        (z - self.first[1]) as usize * WINDOW + (x - self.first[0]) as usize
    }
}

impl ShadingMaking {
    pub(crate) fn from_ground(
        (layers, rules, kinds): (Arc<Layers>, Arc<BiomeRules>, Vec<&'static LayerKind>),
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
            kinds,
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

    pub(crate) fn advance(&mut self, budget: &mut u64) -> Result<Option<TileShading>, Error> {
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
    fn advance(&mut self, pyramid: &Pyramid, budget: &mut u64) -> Result<bool, Error> {
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
                let layers = stamped(&self.layers, stamps, &self.kinds)?;
                let generator = Generator::new(layers.clone(), pyramid, key);
                self.generating = Some((layers, generator));
            }
            let Some((layers, generator)) = self.generating.as_mut() else {
                continue;
            };
            let Some(made) = generator.advance(budget) else {
                if generator.failed() {
                    return Err(crate::jobs::invalid_heights());
                }
                return Ok(false);
            };
            let layers = layers.clone();
            self.generating = None;
            self.next += 1;
            let tile: Tile = codec::decode(&codec::encode(&made))?;
            self.reading = Reading::of(tile, layers, pyramid, &self.window);
        }
    }

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
