//! The tiles of water's waves: [`WIND_TILES`] tiles of the wind's waves,
//! each repeating at its own period and holding its own band of wave
//! lengths, then the tile of the current's turbulence ([`TURBULENCE_TILE`]).
//! Each tile holds unit slope variance spread evenly over the octaves of its
//! band (the saturation range, its heights' spectrum falling as k⁻³), the
//! wind's spread about its heading and the turbulence's alike every way.
//! Water's shaders scale each wind tile by the share of the sea the wind
//! raises where they read it (`meshes/water_spectrum.wgsl`), and read the
//! turbulence's at the lengths the current sets. They are drawn from fixed
//! noise for a [`WaveShape`]: the wind's heading. The renderer evolves the
//! wind's tiles each frame into the surface's slopes and heights, and the
//! turbulence's once, as it stands.

use std::f64::consts::TAU;

use crate::bytes::{Reader, Writer};

/// Texels along a side of a tile: wave numbers of its spectrum, and
/// points of its slopes.
pub const TEXELS: u32 = 256;
/// Tiles of the wind's waves; the tile of the current's turbulence, after
/// them; and every tile.
pub const WIND_TILES: usize = 4;
pub const TURBULENCE_TILE: usize = WIND_TILES;
pub const TILES: usize = WIND_TILES + 1;

/// The wind's first tile's period (meters), and how many times shorter each
/// next tile's is: 40, 10, 2.5 and 0.625 m.
const LONGEST_PERIOD: f64 = 40.0;
const TILE_RATIO: f64 = 4.0;
/// The waves a period from which each wind tile's band runs: a tile holds
/// the waves from a quarter of its period (the first all its longer ones
/// too) down to a quarter of the next tile's, so the shortest span 16
/// texels and filter smoothly.
const BAND_START: f64 = 4.0;
/// The shortest of the wind's waves the last tile holds, meters: about
/// where surface tension takes over from gravity (1.7 cm), past which the
/// saturation range of gravity waves the tiles follow ends.
const SHORTEST: f64 = 0.01;
/// The turbulence tile's nominal period (meters) and the band it holds, in
/// waves a period: one octave, its longest waves a meter long and its
/// shortest spanning 8 texels. Water reads it at the lengths the current
/// sets, its band scaled alike.
const TURBULENCE_PERIOD: f64 = 16.0;
const TURBULENCE_BAND: [f64; 2] = [16.0, 32.0];
/// The share of the wind's waves running against it.
const AGAINST: f64 = 0.2;

const TAG: &[u8; 4] = b"WAVE";

/// What the tiles are drawn for: the wind's heading, a unit vector along x
/// and z (along x in still air).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveShape {
    pub heading: [f32; 2],
}

impl WaveShape {
    /// The shape for the wind blowing at `wind` (meters a second along x and
    /// z).
    pub fn blown(wind: [f32; 2]) -> Self {
        let [x, z] = wind.map(f64::from);
        let speed = libm::hypot(x, z);
        let heading = if speed > 1.0e-3 {
            [(x / speed) as f32, (z / speed) as f32]
        } else {
            [1.0, 0.0]
        };
        Self { heading }
    }

    /// The tiles' job: its identity and input.
    pub fn identity(&self) -> Vec<u8> {
        let mut writer = Writer::new(TAG, 8);
        writer.f32(self.heading[0]).f32(self.heading[1]);
        writer.finish()
    }

    fn from_identity(bytes: &[u8]) -> Result<Self, String> {
        let mut reader = Reader::new(bytes, TAG)?;
        let shape = Self {
            heading: [reader.f32()?, reader.f32()?],
        };
        reader.finish()?;
        Ok(shape)
    }
}

/// A tile as water reads it: its period (meters), the longest and the
/// shortest waves it holds (meters), and the root mean square height of its
/// waves at unit slope variance (meters).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WaveTile {
    pub period: f32,
    pub longest: f32,
    pub shortest: f32,
    pub height: f32,
}

/// The tiles for a shape, as water's storage buffer holds them: each tile,
/// and at each texel of each tile, row by row, its wave number's amplitude
/// and its opposite's (complex).
#[derive(Clone, Debug, PartialEq)]
pub struct WaveSpectra {
    pub shape: WaveShape,
    pub tiles: [WaveTile; TILES],
    pub texels: Vec<[f32; 4]>,
}

/// The tiles' record for the shape `identity` names.
pub(crate) fn record(identity: &[u8]) -> Result<Vec<u8>, String> {
    let shape = WaveShape::from_identity(identity)?;
    let (tiles, texels) = spectra(shape.heading.map(f64::from));
    let mut writer = Writer::new(TAG, 8 + TILES * 16 + texels.len() * 16);
    writer.f32(shape.heading[0]).f32(shape.heading[1]);
    for tile in &tiles {
        writer
            .f32(tile.period)
            .f32(tile.longest)
            .f32(tile.shortest)
            .f32(tile.height);
    }
    for texel in &texels {
        for value in texel {
            writer.f32(*value);
        }
    }
    Ok(writer.finish())
}

/// The tiles `bytes` hold.
pub fn decode(bytes: &[u8]) -> Result<WaveSpectra, String> {
    let mut reader = Reader::new(bytes, TAG)?;
    let shape = WaveShape {
        heading: [reader.f32()?, reader.f32()?],
    };
    let mut tiles = [WaveTile::default(); TILES];
    for tile in &mut tiles {
        *tile = WaveTile {
            period: reader.f32()?,
            longest: reader.f32()?,
            shortest: reader.f32()?,
            height: reader.f32()?,
        };
    }
    let count = (TEXELS * TEXELS) as usize * TILES;
    let mut texels = Vec::with_capacity(count);
    for _ in 0..count {
        texels.push([reader.f32()?, reader.f32()?, reader.f32()?, reader.f32()?]);
    }
    reader.finish()?;
    Ok(WaveSpectra {
        shape,
        tiles,
        texels,
    })
}

/// Tile `tile`'s period (meters) and the band of waves it holds, in waves a
/// period: from its longest up to its shortest.
fn layout(tile: usize) -> (f64, [f64; 2]) {
    if tile == TURBULENCE_TILE {
        return (TURBULENCE_PERIOD, TURBULENCE_BAND);
    }
    let period = (0..tile).fold(LONGEST_PERIOD, |period, _| period / TILE_RATIO);
    let first = if tile == 0 { 1.0 } else { BAND_START };
    let end = if tile + 1 == WIND_TILES {
        period / SHORTEST
    } else {
        BAND_START * TILE_RATIO
    };
    (period, [first, end])
}

/// The tiles, and their spectra for the wind's heading `heading` (a unit
/// vector along x and z): at each texel of each tile, row by row, its wave
/// number's amplitude and its opposite's (complex). Each amplitude is fixed
/// noise (two normal deviates) times the square root of half the tile's
/// spectrum there, where the wave number lies in the tile's band: k⁻⁴, so
/// each octave holds as much slope variance, times the spread about the
/// wind's heading, cos² of the angle from it and AGAINST of that against
/// it, and even every way for the turbulence. Each tile scales to unit
/// slope variance: the wind's as their waves run, each wave number and its
/// opposite carrying their own (the cross terms of their sum come and go as
/// the two run apart); the turbulence's as it stands, the two summed. Wave
/// numbers at a grid's edge, which have no opposite there, hold none.
fn spectra(heading: [f64; 2]) -> ([WaveTile; TILES], Vec<[f32; 4]>) {
    let n = TEXELS as usize;
    let middle = (n / 2) as f64;
    let mut amplitudes = vec![[0.0_f64; 2]; n * n * TILES];
    let mut tiles = [WaveTile::default(); TILES];
    let mut gains = [0.0_f64; TILES];
    for (tile, described) in tiles.iter_mut().enumerate() {
        let (period, band) = layout(tile);
        let step = TAU / period;
        for z in 1..n {
            for x in 1..n {
                // Waves a period along x and z.
                let (along_x, along_z) = (x as f64 - middle, z as f64 - middle);
                let waves = (along_x * along_x + along_z * along_z).sqrt();
                if !(band[0]..band[1]).contains(&waves) {
                    continue;
                }
                let spread = if tile == TURBULENCE_TILE {
                    1.0
                } else {
                    let facing = (along_x * heading[0] + along_z * heading[1]) / waves;
                    facing * facing * if facing < 0.0 { AGAINST } else { 1.0 }
                };
                let k = waves * step;
                let power = spread / (k * k * k * k);
                let [a, b] = normal_pair(tile as u32, x as u32, z as u32);
                let scale = (power * 0.5).sqrt();
                amplitudes[(tile * n + z) * n + x] = [a * scale, b * scale];
            }
        }
        // The slopes' and the heights' variance of the tile as drawn.
        let (mut slopes, mut heights) = (0.0, 0.0);
        for z in 1..n {
            for x in 1..n {
                let own = amplitudes[(tile * n + z) * n + x];
                let opposite = amplitudes[(tile * n + n - z) * n + n - x];
                let power = if tile == TURBULENCE_TILE {
                    let (real, imaginary) = (own[0] + opposite[0], own[1] - opposite[1]);
                    real * real + imaginary * imaginary
                } else {
                    own[0] * own[0]
                        + own[1] * own[1]
                        + opposite[0] * opposite[0]
                        + opposite[1] * opposite[1]
                };
                let (kx, kz) = ((x as f64 - middle) * step, (z as f64 - middle) * step);
                slopes += (kx * kx + kz * kz) * power;
                heights += power;
            }
        }
        let gain = if slopes > 0.0 {
            1.0 / slopes.sqrt()
        } else {
            0.0
        };
        gains[tile] = gain;
        *described = WaveTile {
            period: period as f32,
            longest: (period / band[0]) as f32,
            shortest: (period / band[1]) as f32,
            height: (heights.sqrt() * gain) as f32,
        };
    }
    let texels = (0..n * n * TILES)
        .map(|index| {
            let (tile, x, z) = (index / (n * n), index % n, index / n % n);
            let opposite = if x == 0 || z == 0 {
                [0.0; 2]
            } else {
                amplitudes[(tile * n + n - z) * n + n - x]
            };
            let own = amplitudes[index];
            [own[0], own[1], opposite[0], opposite[1]].map(|value| (value * gains[tile]) as f32)
        })
        .collect();
    (tiles, texels)
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
