//! The spectra of water's waves: [`TILES`] tiles of wind-driven waves,
//! each repeating at its own length, whose spectra together are one
//! spectrum the wind shapes (Phillips's, after Tessendorf), split into
//! bands of wave numbers, each tile taking those its grid holds densely
//! enough. They are drawn from fixed noise for a [`WaveShape`]: the wind
//! and the ripples' size. The renderer evolves them each frame into the
//! surface's slopes.

use std::f64::consts::TAU;

use crate::bytes::{Reader, Writer};

/// Texels along a side of a tile: wave numbers of its spectrum, and
/// points of its slopes.
pub const TEXELS: u32 = 256;
/// Tiles of waves, each repeating at its own length.
pub const TILES: usize = 3;

/// Each tile's meters, as shares of the ripples' size: the first holds the
/// swell longer than the largest ripples, the second ripples down to an
/// eighth of them, the third the finer ones. Each band spans an eighth of
/// its tile's wave numbers, so its shortest waves span eight texels and
/// filter smoothly.
const TILE_SHARES: [f64; TILES] = [32.0, 4.0, 0.5];
/// Where each band ends, in wave numbers of the next tile's grid: a tile
/// takes the waves from its first wave number to this many of the next
/// tile's, so every tile holds its band with enough wave numbers.
const BAND_END: f64 = 4.0;
/// The surface's root mean square slope at full ripples.
const FULL_SLOPE: f64 = 0.5;
/// Meters under which waves fade out: the shortest ripples tilt the
/// surface too little to see.
const SHORTEST: f64 = 0.01;
/// The calmest wind the spectrum takes (meters a second), and the share
/// of waves running against it.
const CALMEST: f64 = 1.0;
const AGAINST: f64 = 0.2;
/// Meters a second squared of gravity.
const GRAVITY: f64 = 9.80665;
/// The smallest ripples' size, meters.
const SMALLEST_SIZE: f32 = 0.1;

const TAG: &[u8; 4] = b"WAVE";

/// What the spectra are drawn for: the wind's velocity (meters a second
/// along x and z) and the meters across the largest ripples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveShape {
    pub wind: [f32; 2],
    pub size: f32,
}

impl WaveShape {
    /// The spectra's job: its identity and input.
    pub fn identity(&self) -> Vec<u8> {
        let mut writer = Writer::new(TAG, 12);
        writer.f32(self.wind[0]).f32(self.wind[1]).f32(self.size);
        writer.finish()
    }

    fn from_identity(bytes: &[u8]) -> Result<Self, String> {
        let mut reader = Reader::new(bytes, TAG)?;
        let shape = Self {
            wind: [reader.f32()?, reader.f32()?],
            size: reader.f32()?,
        };
        reader.finish()?;
        Ok(shape)
    }

    /// Each tile's meters, longest first.
    pub fn sizes(&self) -> [f32; TILES] {
        self.meters().map(|meters| meters as f32)
    }

    fn meters(&self) -> [f64; TILES] {
        let size = f64::from(engine_math::max(self.size, SMALLEST_SIZE));
        TILE_SHARES.map(|share| size * share)
    }
}

/// The tiles' spectra for a shape, as water's storage buffer holds them:
/// at each texel of each tile, row by row, its wave number's amplitude and
/// its opposite's (complex).
#[derive(Clone, Debug, PartialEq)]
pub struct WaveSpectra {
    pub shape: WaveShape,
    pub texels: Vec<[f32; 4]>,
}

/// The spectra's record for the shape `identity` names.
pub(crate) fn record(identity: &[u8]) -> Result<Vec<u8>, String> {
    let shape = WaveShape::from_identity(identity)?;
    let texels = spectra(shape.meters(), shape.wind.map(f64::from));
    let mut writer = Writer::new(TAG, 12 + texels.len() * 16);
    writer.f32(shape.wind[0]).f32(shape.wind[1]).f32(shape.size);
    for texel in &texels {
        for value in texel {
            writer.f32(*value);
        }
    }
    Ok(writer.finish())
}

/// The spectra `bytes` hold.
pub fn decode(bytes: &[u8]) -> Result<WaveSpectra, String> {
    let mut reader = Reader::new(bytes, TAG)?;
    let shape = WaveShape {
        wind: [reader.f32()?, reader.f32()?],
        size: reader.f32()?,
    };
    let count = (TEXELS * TEXELS) as usize * TILES;
    let mut texels = Vec::with_capacity(count);
    for _ in 0..count {
        texels.push([reader.f32()?, reader.f32()?, reader.f32()?, reader.f32()?]);
    }
    reader.finish()?;
    Ok(WaveSpectra { shape, texels })
}

/// The tiles' spectra, tiles of `sizes` meters (longest first) under wind
/// `wind` (meters a second along x and z): at each texel of each tile, row
/// by row, its wave number's amplitude and its opposite's (complex). Each
/// amplitude is fixed noise (two normal deviates) times the square root of
/// half Phillips's spectrum there, where the wave number lies in the tile's
/// band, scaled so the slopes' variance over all tiles is [`FULL_SLOPE`]
/// squared. Wave numbers at a grid's edge, which have no opposite there,
/// hold none.
fn spectra(sizes: [f64; TILES], wind: [f64; 2]) -> Vec<[f32; 4]> {
    let n = TEXELS as usize;
    let blown = libm::hypot(wind[0], wind[1]);
    let speed = blown.max(CALMEST);
    let heading = if blown > 1.0e-3 {
        [wind[0] / blown, wind[1] / blown]
    } else {
        [1.0, 0.0]
    };
    // The longest waves the wind raises.
    let longest = speed * speed / GRAVITY;
    let mut amplitudes = vec![[0.0_f64; 2]; n * n * TILES];
    let mut slopes = 0.0;
    for (tile, &size) in sizes.iter().enumerate() {
        let step = TAU / size;
        let band = sizes
            .get(tile + 1)
            .map_or(f64::INFINITY, |next| BAND_END * TAU / next);
        let first = if tile == 0 { 0.0 } else { BAND_END * step };
        for z in 1..n {
            for x in 1..n {
                let kx = (x as f64 - (n / 2) as f64) * step;
                let kz = (z as f64 - (n / 2) as f64) * step;
                let squared = kx * kx + kz * kz;
                let k = squared.sqrt();
                if k <= 0.0 || k < first || k >= band {
                    continue;
                }
                let along = (kx * heading[0] + kz * heading[1]) / k;
                let directional = along * along * if along < 0.0 { AGAINST } else { 1.0 };
                let power = libm::exp(-1.0 / (squared * longest * longest)) / (squared * squared)
                    * directional
                    * libm::exp(-squared * SHORTEST * SHORTEST);
                let [a, b] = normal_pair(tile as u32, x as u32, z as u32);
                let scale = (power * 0.5).sqrt();
                amplitudes[(tile * n + z) * n + x] = [a * scale, b * scale];
                slopes += squared * (a * a + b * b) * power * 0.5;
            }
        }
    }
    // Each wave number and its opposite both carry slopes.
    let gain = if slopes > 0.0 {
        FULL_SLOPE / (2.0 * slopes).sqrt()
    } else {
        0.0
    };
    (0..n * n * TILES)
        .map(|index| {
            let (tile, x, z) = (index / (n * n), index % n, index / n % n);
            let opposite = if x == 0 || z == 0 {
                [0.0; 2]
            } else {
                amplitudes[(tile * n + n - z) * n + n - x]
            };
            let own = amplitudes[index];
            [own[0], own[1], opposite[0], opposite[1]].map(|value| (value * gain) as f32)
        })
        .collect()
}

/// Two independent standard normal deviates for texel `(x, z)` of tile
/// `tile`: hashed uniform numbers through Box and Muller's transform.
fn normal_pair(tile: u32, x: u32, z: u32) -> [f64; 2] {
    let first = hash(tile.wrapping_mul(0x9e37_79b9) ^ hash(x ^ hash(z)));
    let second = hash(first ^ 0x85eb_ca6b);
    let unit = |bits: u32| (f64::from(bits >> 8) + 0.5) / f64::from(1_u32 << 24);
    let radius = (-2.0 * libm::log(unit(first))).sqrt();
    let angle = TAU * unit(second);
    [radius * libm::cos(angle), radius * libm::sin(angle)]
}

/// A 32-bit integer hash (Wellons's lowbias32).
pub(crate) fn hash(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    value
}
