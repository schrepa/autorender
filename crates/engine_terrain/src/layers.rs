use std::sync::Arc;

use engine_core::{Error, ErrorCode};
use engine_math::{max, min};
use engine_settings::{InvalidSetting, Rule, Setting, settings};
use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::given::GivenRiver;
use crate::hydrology::Hydrology;
use crate::layer::{ChangeMap, NoFiles, Place, TerrainLayer, layer_kind, read_entry, write_entry};
use crate::noise::smooth;
use crate::path::{Bridge, PathSample};
use crate::rivers::Rivers;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum Blend {
    #[default]
    Add,
    Max,
    Min,
    Replace,
}

impl Setting for Blend {}

impl Blend {
    #[inline(always)]
    pub fn apply(self, below: f32, value: f32, weight: f32) -> f32 {
        let target = match self {
            Self::Add => below + value,
            Self::Max => max(below, value),
            Self::Min => min(below, value),
            Self::Replace => value,
        };
        below + (target - below) * weight
    }
}

pub const MAX_HEIGHT: f32 = 20_000.0;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum StampOp {
    Raise(#[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = MAX_HEIGHT)))] f32),
    Lower(#[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = MAX_HEIGHT)))] f32),
    Flatten(
        #[cfg_attr(feature = "schema", schemars(range(min = -MAX_HEIGHT, max = MAX_HEIGHT)))] f32,
    ),
    Hole,
}

impl Setting for StampOp {
    fn check(&self, field: &'static str, _: Rule) -> Result<(), InvalidSetting> {
        let (variant, height, rule) = match *self {
            Self::Raise(height) => ("raise", height, Rule::Range(0.0, MAX_HEIGHT as f64)),
            Self::Lower(height) => ("lower", height, Rule::Range(0.0, MAX_HEIGHT as f64)),
            Self::Flatten(height) => (
                "flatten",
                height,
                Rule::Range(-MAX_HEIGHT as f64, MAX_HEIGHT as f64),
            ),
            Self::Hole => return Ok(()),
        };
        height
            .check(variant, rule)
            .map_err(|invalid| invalid.within(field))
    }
}

settings! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct Stamp {
        pub center: [f32; 2] = [0.0; 2] => range(-Layers::MAX_COORDINATE, Layers::MAX_COORDINATE),
        pub radius: f32 = 1.0 => range(0.0, Layers::MAX_COORDINATE),
        #[serde(default)]
        pub falloff: f32 = 0.0 => range(0.0, Layers::MAX_COORDINATE),
        pub op: StampOp = StampOp::Hole => any,
    }
}

impl Stamp {
    #[inline]
    pub fn weight(&self, x: f32, z: f32) -> f32 {
        let dx = x - self.center[0];
        let dz = z - self.center[1];
        let distance = (dx * dx + dz * dz).sqrt();
        1.0 - smooth((distance - self.radius) / max(self.falloff, 1.0e-6))
    }

    pub fn reach(&self) -> f32 {
        self.radius + self.falloff
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HeightImage {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) samples: Box<[u16]>,
    pub(crate) levels: Vec<ImageLevel>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ImageLevel {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) samples: Box<[f32]>,
}

impl ImageLevel {
    fn below(width: u32, height: u32, samples: &[f32]) -> Self {
        let half = |size: u32| (size - 1).div_ceil(2) + 1;
        let (next_width, next_height) = (half(width), half(height));
        let at = |x: i64, z: i64| {
            let x = x.clamp(0, i64::from(width) - 1) as usize;
            let z = z.clamp(0, i64::from(height) - 1) as usize;
            samples[z * width as usize + x]
        };
        let mut rows = vec![0.0_f32; next_width as usize * height as usize];
        for z in 0..i64::from(height) {
            for x in 0..i64::from(next_width) {
                let center = 2 * x;
                rows[z as usize * next_width as usize + x as usize] =
                    0.25 * at(center - 1, z) + 0.5 * at(center, z) + 0.25 * at(center + 1, z);
            }
        }
        let row = |x: i64, z: i64| {
            let z = z.clamp(0, i64::from(height) - 1) as usize;
            rows[z * next_width as usize + x as usize]
        };
        let mut next = vec![0.0_f32; next_width as usize * next_height as usize];
        for z in 0..i64::from(next_height) {
            for x in 0..i64::from(next_width) {
                let center = 2 * z;
                next[z as usize * next_width as usize + x as usize] =
                    0.25 * row(x, center - 1) + 0.5 * row(x, center) + 0.25 * row(x, center + 1);
            }
        }
        Self {
            width: next_width,
            height: next_height,
            samples: next.into_boxed_slice(),
        }
    }
}

impl HeightImage {
    pub fn new(width: u32, height: u32, samples: Vec<u16>) -> Option<Self> {
        if !(width >= 2 && height >= 2 && samples.len() == width as usize * height as usize) {
            return None;
        }
        let mut levels: Vec<ImageLevel> = Vec::new();
        let mut current: (u32, u32, Vec<f32>) = (
            width,
            height,
            samples.iter().map(|&sample| f32::from(sample)).collect(),
        );
        while current.0 > 2 || current.1 > 2 {
            let level = ImageLevel::below(current.0, current.1, &current.2);
            current = (level.width, level.height, level.samples.to_vec());
            levels.push(level);
        }
        Some(Self {
            width,
            height,
            samples: samples.into_boxed_slice(),
            levels,
        })
    }

    pub fn from_le_bytes(width: u32, height: u32, bytes: &[u8]) -> Option<Self> {
        if bytes.len() != width as usize * height as usize * 2 {
            return None;
        }
        let samples = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect();
        Self::new(width, height, samples)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn levels(&self) -> usize {
        self.levels.len()
    }

    #[inline]
    pub fn sample_smooth(&self, u: f32, v: f32, level: usize) -> f32 {
        let mut sample = 0.0;
        self.sample_smooth_row([u], v, level, |_, value| sample = value);
        sample
    }

    #[inline]
    pub fn sample_smooth_row(
        &self,
        us: impl IntoIterator<Item = f32>,
        v: f32,
        level: usize,
        each: impl FnMut(usize, f32),
    ) {
        match level
            .checked_sub(1)
            .and_then(|index| self.levels.get(index))
        {
            None => smooth_row(us, v, 1.0, each, |x, z| {
                let x = x.clamp(0, self.width as i32 - 1) as usize;
                let z = z.clamp(0, self.height as i32 - 1) as usize;
                f32::from(self.samples[z * self.width as usize + x])
            }),
            Some(coarse) => {
                let scale = 1.0 / (1_u32 << level) as f32;
                smooth_row(us, v, scale, each, |x, z| {
                    let x = x.clamp(0, coarse.width as i32 - 1) as usize;
                    let z = z.clamp(0, coarse.height as i32 - 1) as usize;
                    coarse.samples[z * coarse.width as usize + x]
                });
            }
        }
    }

    #[inline]
    pub fn sample(&self, u: f32, v: f32, level: usize) -> f32 {
        match level
            .checked_sub(1)
            .and_then(|index| self.levels.get(index))
        {
            None => catmull_rom_2d(u, v, |x, z| {
                let x = x.clamp(0, self.width as i32 - 1) as usize;
                let z = z.clamp(0, self.height as i32 - 1) as usize;
                f32::from(self.samples[z * self.width as usize + x])
            }),
            Some(coarse) => {
                let scale = 1.0 / (1_u32 << level) as f32;
                catmull_rom_2d(u * scale, v * scale, |x, z| {
                    let x = x.clamp(0, coarse.width as i32 - 1) as usize;
                    let z = z.clamp(0, coarse.height as i32 - 1) as usize;
                    coarse.samples[z * coarse.width as usize + x]
                })
            }
        }
    }
}

#[inline(always)]
fn catmull_rom_2d(u: f32, v: f32, texel: impl Fn(i32, i32) -> f32) -> f32 {
    let u0 = u.floor();
    let v0 = v.floor();
    let (x, z) = (u0 as i32, v0 as i32);
    let wx = catmull_rom(u - u0);
    let wz = catmull_rom(v - v0);
    let mut sum = 0.0;
    for (row, weight_z) in wz.iter().enumerate() {
        let zi = z - 1 + row as i32;
        let line = wx[0] * texel(x - 1, zi)
            + wx[1] * texel(x, zi)
            + wx[2] * texel(x + 1, zi)
            + wx[3] * texel(x + 2, zi);
        sum += weight_z * line;
    }
    sum
}

#[inline(always)]
fn smooth_row(
    us: impl IntoIterator<Item = f32>,
    v: f32,
    scale: f32,
    mut each: impl FnMut(usize, f32),
    texel: impl Fn(i32, i32) -> f32,
) {
    let v = v * scale;
    let v0 = v.floor();
    let z = v0 as i32;
    let wz = b_spline(v - v0);
    let column = |x: i32| {
        wz[0] * texel(x, z - 1)
            + wz[1] * texel(x, z)
            + wz[2] * texel(x, z + 1)
            + wz[3] * texel(x, z + 2)
    };
    let mut cell: Option<(i32, [f32; 4])> = None;
    for (index, u) in us.into_iter().enumerate() {
        let u = u * scale;
        let u0 = u.floor();
        let x = u0 as i32;
        let columns = match cell {
            Some((at, columns)) if at == x => columns,
            Some((at, [_, b, c, d])) if x.checked_sub(at) == Some(1) => [b, c, d, column(x + 2)],
            _ => [column(x - 1), column(x), column(x + 1), column(x + 2)],
        };
        cell = Some((x, columns));
        let wx = b_spline(u - u0);
        each(
            index,
            wx[0] * columns[0] + wx[1] * columns[1] + wx[2] * columns[2] + wx[3] * columns[3],
        );
    }
}

#[inline(always)]
fn b_spline(t: f32) -> [f32; 4] {
    let s = 1.0 - t;
    let t2 = t * t;
    let t3 = t2 * t;
    [
        s * s * s * (1.0 / 6.0),
        (3.0 * t3 - 6.0 * t2 + 4.0) * (1.0 / 6.0),
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) * (1.0 / 6.0),
        t3 * (1.0 / 6.0),
    ]
}

#[inline(always)]
pub fn catmull_rom(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        0.5 * (-t3 + 2.0 * t2 - t),
        0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
        0.5 * (-3.0 * t3 + 4.0 * t2 + t),
        0.5 * (t3 - t2),
    ]
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layers {
    layers: Vec<Arc<dyn TerrainLayer>>,
    pub(crate) offset: Vec2,
    pub(crate) fade: Option<Fade>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Fade {
    pub(crate) low: Vec2,
    pub(crate) high: Vec2,
    pub(crate) blend: f32,
}

impl Layers {
    pub const MAX_COORDINATE: f32 = 1.0e7;

    pub fn new(layers: Vec<Arc<dyn TerrainLayer>>) -> Result<Self, Error> {
        for (index, layer) in layers.iter().enumerate() {
            layer
                .check(Place {
                    layers: &layers,
                    index,
                })
                .map_err(|error| error.within(&format!("$.layers[{index}]")))?;
        }
        Ok(Self {
            layers,
            offset: Vec2::ZERO,
            fade: None,
        })
    }

    pub fn layers(&self) -> &[Arc<dyn TerrainLayer>] {
        &self.layers
    }

    pub fn may_hole(&self) -> bool {
        self.layers.iter().any(|layer| layer.may_hole())
    }

    pub fn has_hydrology(&self) -> bool {
        self.layers.iter().any(|layer| layer.makes_water())
    }

    #[must_use]
    pub fn surroundings(&self, offset: Vec2, (low, high): (Vec2, Vec2), blend: f32) -> Self {
        Self {
            offset,
            fade: Some(Fade {
                low,
                high,
                blend: max(blend, 0.0),
            }),
            ..self.clone()
        }
    }

    pub fn shown(&self, x: f32, z: f32) -> f32 {
        let Some(fade) = self.fade else {
            return 1.0;
        };
        let point = Vec2::new(x, z);
        let outside = (fade.low - point)
            .max(point - fade.high)
            .max(Vec2::ZERO)
            .length();
        if outside <= 0.0 {
            return 1.0;
        }
        if fade.blend <= 0.0 {
            return 0.0;
        }
        let t = min(outside / fade.blend, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    }

    pub fn offset_samples(&self, spacing: f32) -> [i64; 2] {
        [
            (self.offset.x / spacing).round() as i64,
            (self.offset.y / spacing).round() as i64,
        ]
    }

    pub fn with_stamps(&self, stamps: &[Stamp]) -> Result<Self, Error> {
        let mut layers = self.layers.clone();
        if !stamps.is_empty() {
            let kind = layer_kind(STAMP).ok_or_else(|| {
                Error::new(ErrorCode::MissingLayer, "$.stamps").variants([STAMP])
            })?;
            for (offset, stamp) in stamps.iter().enumerate() {
                let at = format!("$.layers[{}]", self.layers.len() + offset);
                stamp.validate().map_err(|invalid| {
                    Error::from(invalid)
                        .coded(ErrorCode::InvalidTerrain)
                        .within(&at)
                })?;
                let bytes = crate::codec::encode(stamp);
                let made =
                    (kind.read)(&mut Reader::new(&bytes)).map_err(|error| error.within(&at))?;
                layers.push(made);
            }
        }
        Ok(Self {
            layers,
            offset: self.offset,
            fade: self.fade,
        })
    }

    #[must_use]
    pub fn with_drawn(&self, drawn: &[GivenRiver], baked: Option<&Self>) -> Self {
        let Some(first) = self.layers.iter().position(|layer| layer.makes_water()) else {
            return self.clone();
        };
        let mut layers: Vec<Arc<dyn TerrainLayer>> = self.layers.iter().map(unbaked).collect();
        if let Some(baked) = baked {
            for (layer, done) in layers[..first].iter_mut().zip(&baked.layers) {
                if unbaked(done).same(layer.as_ref()) {
                    *layer = done.clone();
                }
            }
        }
        if !drawn.is_empty()
            && let Some(rivers) = drawn_rivers(drawn)
        {
            layers.insert(first, rivers);
        }
        Self {
            layers,
            offset: self.offset,
            fade: self.fade,
        }
    }

    pub fn changes_from(&self, before: &Self) -> Option<Vec<(Vec2, Vec2)>> {
        let shown = |stack: &Self| -> Vec<Arc<dyn TerrainLayer>> {
            stack
                .layers
                .iter()
                .filter(|layer| layer.given_rivers().is_none())
                .cloned()
                .collect()
        };
        let (now, then) = (shown(self), shown(before));
        if now.len() != then.len() || self.offset != before.offset || self.fade != before.fade {
            return None;
        }
        let mut regions = Vec::new();
        for (now, then) in now.iter().zip(&then) {
            if now.same(then.as_ref()) {
                continue;
            }
            regions.extend(now.changed_regions(then.as_ref())?);
        }
        Some(regions)
    }

    #[must_use]
    pub fn below(&self, index: usize) -> Self {
        Self {
            layers: self.layers[..index].to_vec(),
            offset: self.offset,
            fade: self.fade,
        }
    }

    pub fn unbaked(&self) -> Option<usize> {
        self.layers.iter().position(|layer| layer.needs_bake())
    }

    #[must_use]
    pub fn with_baked(&self, index: usize, baked: Arc<dyn TerrainLayer>) -> Self {
        let mut layers = self.clone();
        layers.layers[index] = baked;
        layers
    }

    pub fn hydrology(&self) -> Option<&Hydrology> {
        self.layers.iter().find_map(|layer| layer.water())
    }

    pub(crate) fn hydrology_layer(&self) -> Option<(usize, &Hydrology)> {
        self.layers
            .iter()
            .enumerate()
            .find_map(|(index, layer)| layer.water().map(|water| (index, water)))
    }

    pub(crate) fn rivers_of(&self, index: usize) -> Option<&Rivers> {
        self.layers
            .get(index)?
            .water()
            .map(|water| &water.rivers)
            .filter(|rivers| !rivers.is_empty())
    }

    pub fn bridges(&self) -> Vec<Bridge> {
        self.layers.iter().flat_map(|layer| layer.bridges()).collect()
    }

    pub fn path_at(&self, x: f32, z: f32) -> PathSample {
        nearest_path(&self.layers, Vec2::new(x, z))
    }

    pub fn changes(&self) -> impl Iterator<Item = ChangeMap<'_>> + '_ {
        self.layers.iter().filter_map(|layer| layer.change())
    }

    pub fn holes_reach(&self, min: Vec2, max: Vec2) -> bool {
        self.layers.iter().any(|layer| layer.holes_reach(min, max))
    }

    pub fn hole_at(&self, x: f32, z: f32) -> bool {
        self.layers.iter().any(|layer| layer.hole(x, z))
    }

    pub(crate) fn cut_rivers(
        &self,
        raster: &crate::rivers::RiverRaster,
        at: usize,
        row: &mut [f32],
        xs: &[f32],
        z: f32,
    ) {
        for (column, (height, &x)) in row.iter_mut().zip(xs).enumerate() {
            let cut = raster.cut(at, column, *height);
            *height = self.faded(*height, cut, x, z);
        }
    }

    pub(crate) fn faded(&self, height: f32, changed: f32, x: f32, z: f32) -> f32 {
        let shown = self.shown(x, z);
        if shown >= 1.0 {
            changed
        } else {
            height + (changed - height) * shown
        }
    }
}

const STAMP: &str = "stamp";

const RIVERS: &str = "rivers";

fn unbaked(layer: &Arc<dyn TerrainLayer>) -> Arc<dyn TerrainLayer> {
    layer.unbaked().unwrap_or_else(|| layer.clone())
}

fn drawn_rivers(drawn: &[GivenRiver]) -> Option<Arc<dyn TerrainLayer>> {
    let kind = layer_kind(RIVERS)?;
    let lines = serde_json::to_value(drawn).ok()?;
    let mut settings = serde_json::Map::new();
    settings.insert("lines".to_owned(), lines);
    (kind.from_json)(&serde_json::Value::Object(settings), &NoFiles).ok()
}

pub(crate) fn nearest_path(layers: &[Arc<dyn TerrainLayer>], point: Vec2) -> PathSample {
    layers
        .iter()
        .fold(PathSample::FAR, |nearest, layer| match layer.path(point) {
            Some(sample) if sample.edge() < nearest.edge() => sample,
            Some(_) | None => nearest,
        })
}

pub fn reaches(center: [f32; 2], radius: f32, min: Vec2, max: Vec2) -> bool {
    let nearest = Vec2::from(center).clamp(min, max);
    nearest.distance_squared(Vec2::from(center)) <= radius * radius
}

use crate::codec::{Codec, Reader, Writer, record, tagged};

tagged!(Blend {
    0 => Add,
    1 => Max,
    2 => Min,
    3 => Replace,
});

tagged!(StampOp {
    0 => Raise(meters),
    1 => Lower(meters),
    2 => Flatten(height),
    3 => Hole,
});

record!(Stamp {
    center,
    radius,
    falloff,
    op,
});
record!(HeightImage {
    width,
    height,
    samples,
    levels,
});
record!(ImageLevel {
    width,
    height,
    samples,
});
record!(Fade { low, high, blend });

impl Codec for Layers {
    fn write(&self, out: &mut Writer) {
        out.put(&(self.layers.len() as u64));
        for layer in &self.layers {
            write_entry(out, layer.as_ref());
        }
        out.put(&self.offset).put(&self.fade);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
        let count: u64 = input.get()?;
        let mut layers = Vec::new();
        for _ in 0..count {
            layers.push(read_entry(input)?);
        }
        let offset = input.get()?;
        let fade = input.get()?;
        let mut stack = Self::new(layers)?;
        stack.offset = offset;
        stack.fade = fade;
        Ok(stack)
    }
}
