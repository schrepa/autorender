use engine_math::{clamp, min};

use crate::bytes::{Reader, Writer};

pub const SHAPE_SIZE: u32 = 128;
pub const DETAIL_SIZE: u32 = 256;
const SHAPE_LAYERS: u32 = 32;
const DETAIL_LAYERS: u32 = 16;

const TAG: &[u8; 4] = b"CLDN";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Volume {
    Shape,
    Detail,
}

impl Volume {
    pub const fn size(self) -> u32 {
        match self {
            Self::Shape => SHAPE_SIZE,
            Self::Detail => DETAIL_SIZE,
        }
    }

    pub const fn layers(self) -> u32 {
        match self {
            Self::Shape => SHAPE_LAYERS,
            Self::Detail => DETAIL_LAYERS,
        }
    }

    const fn code(self) -> u32 {
        match self {
            Self::Shape => 0,
            Self::Detail => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Part {
    pub volume: Volume,
    pub first: u32,
}

impl Part {
    pub fn identity(&self) -> Vec<u8> {
        let mut writer = Writer::new(TAG, 8);
        writer.u32(self.volume.code()).u32(self.first);
        writer.finish()
    }

    fn from_identity(bytes: &[u8]) -> Result<Self, String> {
        let mut reader = Reader::new(bytes, TAG)?;
        let volume = match reader.u32()? {
            0 => Volume::Shape,
            1 => Volume::Detail,
            _ => return Err("no such cloud noise volume".into()),
        };
        let first = reader.u32()?;
        reader.finish()?;
        if first % volume.layers() != 0 || first >= volume.size() {
            return Err("no such slab of the cloud noise".into());
        }
        Ok(Self { volume, first })
    }

    fn texels(&self) -> usize {
        let size = self.volume.size() as usize;
        size * size * self.volume.layers() as usize
    }
}

pub fn parts() -> Vec<Part> {
    [Volume::Shape, Volume::Detail]
        .into_iter()
        .flat_map(|volume| {
            (0..volume.size())
                .step_by(volume.layers() as usize)
                .map(move |first| Part { volume, first })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct CloudNoise {
    pub shape: Vec<u8>,
    pub detail: Vec<u8>,
}

impl CloudNoise {
    pub fn empty() -> Self {
        let side = |volume: Volume| (volume.size() as usize).pow(3);
        Self {
            shape: vec![0; side(Volume::Shape)],
            detail: vec![0; side(Volume::Detail)],
        }
    }

    pub fn insert(&mut self, record: &[u8]) -> Result<(), String> {
        let mut reader = Reader::new(record, TAG)?;
        let identity = {
            let mut writer = Writer::new(TAG, 8);
            writer.u32(reader.u32()?).u32(reader.u32()?);
            writer.finish()
        };
        let part = Part::from_identity(&identity)?;
        let residuals = reader.bytes(part.texels())?;
        reader.finish()?;
        let size = part.volume.size() as usize;
        let start = part.first as usize * size * size;
        let volume = match part.volume {
            Volume::Shape => &mut self.shape,
            Volume::Detail => &mut self.detail,
        };
        let texels = &mut volume[start..start + residuals.len()];
        for (index, residual) in residuals.iter().enumerate() {
            texels[index] = residual.wrapping_add(predicted(texels, size, index));
        }
        Ok(())
    }
}

fn predicted(texels: &[u8], size: usize, index: usize) -> u8 {
    let (x, y, z) = (index % size, index / size % size, index / (size * size));
    let at = |dx: usize, dy: usize, dz: usize| {
        if x < dx || y < dy || z < dz {
            0
        } else {
            texels[index - dx - dy * size - dz * size * size]
        }
    };
    at(1, 0, 0)
        .wrapping_add(at(0, 1, 0))
        .wrapping_add(at(0, 0, 1))
        .wrapping_sub(at(1, 1, 0))
        .wrapping_sub(at(1, 0, 1))
        .wrapping_sub(at(0, 1, 1))
        .wrapping_add(at(1, 1, 1))
}

pub(crate) struct NoisePart {
    part: Part,
    octaves: Octaves,
    texels: Vec<u8>,
}

impl NoisePart {
    pub(crate) fn new(identity: &[u8]) -> Result<Self, String> {
        let part = Part::from_identity(identity)?;
        Ok(Self {
            part,
            octaves: Octaves::new(part.volume),
            texels: Vec::with_capacity(part.texels()),
        })
    }

    pub(crate) fn advance(&mut self, budget: &mut u64) -> Option<Vec<u8>> {
        let size = self.part.volume.size() as usize;
        let total = self.part.texels();
        let end = total.min(
            self.texels
                .len()
                .saturating_add(usize::try_from(*budget).unwrap_or(usize::MAX)),
        );
        *budget -= (end - self.texels.len()) as u64;
        for index in self.texels.len()..end {
            let (x, y, z) = (
                index % size,
                index / size % size,
                index / (size * size) + self.part.first as usize,
            );
            let p = [x, y, z].map(|axis| (axis as f32 + 0.5) / size as f32);
            let value = match self.part.volume {
                Volume::Shape => self.octaves.shape(p),
                Volume::Detail => self.octaves.detail(p),
            };
            self.texels
                .push((clamp(value, 0.0, 1.0) * 255.0 + 0.5) as u8);
        }
        (self.texels.len() == total).then(|| self.record())
    }

    fn record(&self) -> Vec<u8> {
        let size = self.part.volume.size() as usize;
        let mut writer = Writer::new(TAG, 8 + self.texels.len());
        writer.u32(self.part.volume.code()).u32(self.part.first);
        let residuals: Vec<u8> = (0..self.texels.len())
            .map(|index| self.texels[index].wrapping_sub(predicted(&self.texels, size, index)))
            .collect();
        writer.bytes(&residuals);
        writer.finish()
    }
}

pub(crate) fn record(identity: &[u8]) -> Result<Vec<u8>, String> {
    let mut part = NoisePart::new(identity)?;
    let mut budget = u64::MAX;
    part.advance(&mut budget)
        .ok_or_else(|| "a cloud noise slab stopped short".into())
}

struct Octaves {
    worley: Vec<Lattice>,
    perlin: Vec<Lattice>,
}

struct Lattice {
    cells: i32,
    vectors: Vec<[f32; 3]>,
}

impl Lattice {
    fn worley(cells: i32) -> Self {
        Self::of(cells, |cell| hash3(cell.map(|axis| axis + cells as u32)))
    }

    fn perlin(period: i32) -> Self {
        Self::of(period, |cell| {
            let vector =
                hash3([cell[0] + 7, cell[1] + 11, cell[2] + 13]).map(|axis| axis * 2.0 - 1.0);
            let length = dot(vector, vector).sqrt();
            if length > 0.0 {
                vector.map(|axis| axis / length)
            } else {
                [0.0; 3]
            }
        })
    }

    fn of(cells: i32, vector: impl Fn([u32; 3]) -> [f32; 3]) -> Self {
        let count = cells as u32;
        let vectors = (0..count * count * count)
            .map(|index| {
                vector([
                    index % count,
                    index / count % count,
                    index / (count * count),
                ])
            })
            .collect();
        Self { cells, vectors }
    }

    fn around(&self, cell: [i32; 3]) -> [[usize; 3]; 3] {
        cell.map(|axis| [-1, 0, 1].map(|step| (axis + step).rem_euclid(self.cells) as usize))
    }

    fn at(&self, [x, y, z]: [usize; 3]) -> [f32; 3] {
        let cells = self.cells as usize;
        self.vectors[(z * cells + y) * cells + x]
    }

    fn worley_at(&self, p: [f32; 3]) -> f32 {
        let scaled = p.map(|axis| axis * self.cells as f32);
        let cell = scaled.map(|axis| axis.floor() as i32);
        let f: [f32; 3] = std::array::from_fn(|axis| scaled[axis] - cell[axis] as f32);
        let [xs, ys, zs] = self.around(cell);
        let mut nearest = 1.0e9_f32;
        for (dz, &z) in zs.iter().enumerate() {
            for (dy, &y) in ys.iter().enumerate() {
                for (dx, &x) in xs.iter().enumerate() {
                    let point = self.at([x, y, z]);
                    let offset = [dx, dy, dz].map(|step| step as f32 - 1.0);
                    let away: [f32; 3] =
                        std::array::from_fn(|axis| (offset[axis] + point[axis]) - f[axis]);
                    nearest = min(nearest, dot(away, away));
                }
            }
        }
        1.0 - clamp(nearest.sqrt(), 0.0, 1.0)
    }

    fn perlin_at(&self, p: [f32; 3]) -> f32 {
        let scaled = p.map(|axis| axis * self.cells as f32);
        let cell = scaled.map(|axis| axis.floor() as i32);
        let f: [f32; 3] = std::array::from_fn(|axis| scaled[axis] - cell[axis] as f32);
        let u = f.map(|t| t * t * t * (t * (t * 6.0 - 15.0) + 10.0));
        let [xs, ys, zs] = self.around(cell);
        let corner = |index: usize| {
            let offset = [index & 1, (index >> 1) & 1, index >> 2];
            let gradient = self.at([xs[offset[0] + 1], ys[offset[1] + 1], zs[offset[2] + 1]]);
            let toward: [f32; 3] = std::array::from_fn(|axis| f[axis] - offset[axis] as f32);
            dot(gradient, toward)
        };
        let corners: [f32; 8] = std::array::from_fn(corner);
        let x0 = mix(corners[0], corners[1], u[0]);
        let x1 = mix(corners[2], corners[3], u[0]);
        let x2 = mix(corners[4], corners[5], u[0]);
        let x3 = mix(corners[6], corners[7], u[0]);
        mix(mix(x0, x1, u[1]), mix(x2, x3, u[1]), u[2])
    }
}

impl Octaves {
    fn new(volume: Volume) -> Self {
        Self {
            worley: [4, 8, 16, 32, 64]
                .into_iter()
                .take(match volume {
                    Volume::Shape => 3,
                    Volume::Detail => 5,
                })
                .map(Lattice::worley)
                .collect(),
            perlin: match volume {
                Volume::Shape => [4, 8, 16, 32].into_iter().map(Lattice::perlin).collect(),
                Volume::Detail => Vec::new(),
            },
        }
    }

    fn shape(&self, p: [f32; 3]) -> f32 {
        let [w4, w8, w16] = [0, 1, 2].map(|octave| self.worley[octave].worley_at(p));
        let low = w4 * 0.625 + w8 * 0.25 + w16 * 0.125;
        let mut sum = 0.0;
        let mut amplitude = 0.5;
        for octave in &self.perlin {
            sum += octave.perlin_at(p) * amplitude;
            amplitude *= 0.5;
        }
        let billows = clamp(sum * 0.9 + 0.5, 0.0, 1.0);
        clamp(remap(billows, low - 1.0, 1.0, 0.0, 1.0), 0.0, 1.0)
    }

    fn detail(&self, p: [f32; 3]) -> f32 {
        let [w4, w8, w16, w32, w64] =
            [0, 1, 2, 3, 4].map(|octave| self.worley[octave].worley_at(p));
        let coarse = w4 * 0.625 + w8 * 0.25 + w16 * 0.125;
        let middle = w8 * 0.625 + w16 * 0.25 + w32 * 0.125;
        let fine = w16 * 0.625 + w32 * 0.25 + w64 * 0.125;
        coarse * 0.625 + middle * 0.25 + fine * 0.125
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

fn remap(value: f32, low: f32, high: f32, to_low: f32, to_high: f32) -> f32 {
    to_low + (value - low) * (to_high - to_low) / engine_math::max(high - low, 1.0e-5)
}

fn hash3(p: [u32; 3]) -> [f32; 3] {
    let mut v = p.map(|axis| axis.wrapping_mul(1_664_525).wrapping_add(1_013_904_223));
    let mix = |v: &mut [u32; 3]| {
        v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[2]));
        v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
        v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    };
    mix(&mut v);
    v = v.map(|axis| axis ^ (axis >> 16));
    mix(&mut v);
    v.map(|axis| axis as f32 * (1.0 / 4_294_967_296.0))
}
