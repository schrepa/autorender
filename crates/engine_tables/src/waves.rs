//! The tiles of water's waves: [`WIND_TILES`] tiles of the wind's waves,
//! each repeating at its own period and holding its own band of wave
//! lengths, then the tile of the current's turbulence ([`TURBULENCE_TILE`]).
//! The wind's tiles' periods never divide one another and each one's lattice
//! turns by its own angle from the world's axes, so no tile's repeat lines
//! up with another's or with the grids of the ground and the water: water
//! reads each once, straight where it stands, with no patches. Each tile
//! holds unit slope variance spread evenly over the octaves of its band (the
//! saturation range, its heights' spectrum falling as k⁻³), the wind's
//! spread about its heading as measured seas spread (Donelan, Hamilton and
//! Hui 1985; Banner 1990: [`spreading`]), narrow at the sea's peak and wider
//! above it, and the turbulence's alike every way.
//! Water's shaders scale each wind tile by the share of the sea the wind
//! raises where they read it (`meshes/water_spectrum.wgsl`), and read the
//! turbulence's at the lengths the current sets. They are drawn from fixed
//! noise for a [`WaveShape`]: the wind's heading. The renderer evolves the
//! wind's tiles each frame into the surface's slopes and heights, and the
//! turbulence's once, as it stands. The sea's own waves head by the same
//! spread ([`spread_turn`]).

use std::f64::consts::{FRAC_PI_2, PI, TAU};

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
/// next tile's is: φ³ = 2 + √5, the cube of the golden ratio, irrational,
/// so no tile's period divides another's, and with a continued fraction of
/// fours alone, so neighboring tiles' periods lie far from any ratio of
/// small whole numbers: 40, 9.44, 2.23 and 0.526 m.
const LONGEST_PERIOD: f64 = 40.0;
const TILE_RATIO: f64 = 4.236_067_977_499_79;
/// The waves a period from which each wind tile's band runs: a tile holds
/// the waves from a quarter of its period (the first all its longer ones
/// too) down to a quarter of the next tile's, so the shortest span about 15
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
/// shortest waves it holds (meters), the root mean square height of its
/// waves at unit slope variance (meters), and the angle its lattice turns by
/// from the world's axes (radians, from +x toward +z): water reads it at
/// each world point turned by that angle, and turns its slopes back.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WaveTile {
    pub period: f32,
    pub longest: f32,
    pub shortest: f32,
    pub height: f32,
    pub turn: f32,
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
    let mut writer = Writer::new(TAG, 8 + TILES * 20 + texels.len() * 16);
    writer.f32(shape.heading[0]).f32(shape.heading[1]);
    for tile in &tiles {
        writer
            .f32(tile.period)
            .f32(tile.longest)
            .f32(tile.shortest)
            .f32(tile.height)
            .f32(tile.turn);
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
            turn: reader.f32()?,
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

/// The spread of the wind's waves about its heading as measured seas spread
/// (Donelan, Hamilton and Hui 1985; Banner 1990, as Horvath 2015 gives
/// them), D(θ) = β / (2 tanh(βπ)) sech²(βθ) of the angle θ from the
/// heading, from −π to π: its β for waves whose angular frequency is
/// `ratio` times the peak's, r, a ratio under 0.56 spreading as 0.56 does.
/// β is 2.61 r^1.3 below 0.95, 2.28 r^−1.3 below 1.6, and
/// 10^(−0.4 + 0.8393 e^(−0.567 ln r²)) beyond, so the spread is narrowest at
/// the peak (β about 2.3), where almost none of the waves run against the
/// wind, and wider above it.
pub fn spreading(ratio: f64) -> f64 {
    let ratio = ratio.max(0.56);
    if ratio < 0.95 {
        2.61 * libm::pow(ratio, 1.3)
    } else if ratio < 1.6 {
        2.28 * libm::pow(ratio, -1.3)
    } else {
        libm::pow(
            10.0,
            -0.4 + 0.8393 * libm::exp(-0.567 * libm::log(ratio * ratio)),
        )
    }
}

/// The angle (radians) from the wind's heading below which the spread of
/// [`spreading`] puts the share (1 + `draw`) / 2 of the waves `ratio` times
/// the peak's angular frequency, `draw` running from −1 to 1: D's inverse
/// distribution, atanh(u tanh(βπ)) / β for u = `draw`, so draws spread
/// evenly over that range head waves as the sea spreads them, either way
/// from the heading as `draw`'s sign is.
pub fn spread_turn(ratio: f64, draw: f64) -> f64 {
    let beta = spreading(ratio);
    libm::atanh(draw * libm::tanh(beta * PI)) / beta
}

/// D(θ) of [`spreading`] for its β `beta` and the angle θ `angle` (radians)
/// from the wind's heading: the share of waves of one wave number heading
/// there, per radian, one over the circle for every wave number.
fn spread_at(beta: f64, angle: f64) -> f64 {
    let cosh = libm::cosh(beta * angle);
    beta / (2.0 * libm::tanh(beta * PI) * cosh * cosh)
}

/// Tile `tile`'s period (meters), the band of waves it holds, in waves a
/// period: from its longest up to its shortest, and the angle its lattice
/// turns by from the world's axes (radians, from +x toward +z). The wind's
/// tiles turn by (i + ½) (π/2) / [`WIND_TILES`], spread evenly over a
/// quarter turn, a square lattice's own symmetry, so none lies along the
/// world's axes, where the ground's and the water's grids run; the
/// turbulence's, drawn evenly every way, turns by none, its reads turning
/// to the flow.
fn layout(tile: usize) -> (f64, [f64; 2], f64) {
    if tile == TURBULENCE_TILE {
        return (TURBULENCE_PERIOD, TURBULENCE_BAND, 0.0);
    }
    let period = (0..tile).fold(LONGEST_PERIOD, |period, _| period / TILE_RATIO);
    let first = if tile == 0 { 1.0 } else { BAND_START };
    let end = if tile + 1 == WIND_TILES {
        period / SHORTEST
    } else {
        BAND_START * TILE_RATIO
    };
    let turn = (tile as f64 + 0.5) * FRAC_PI_2 / WIND_TILES as f64;
    (period, [first, end], turn)
}

/// The tiles, and their spectra for the wind's heading `heading` (a unit
/// vector along x and z): at each texel of each tile, row by row, its wave
/// number's amplitude and its opposite's (complex). Each amplitude is fixed
/// noise (two normal deviates) times the square root of half the tile's
/// spectrum there, where the wave number lies in the tile's band: k⁻⁴, so
/// each octave holds as much slope variance, times the spread about the
/// wind's heading turned as the tile's lattice is (`spread_at`), one over
/// the circle for each wave number, and even every way for the turbulence.
/// A tile is drawn once while the sea's peak changes from place to place,
/// so each wind tile spreads its waves as the sea does where its band
/// carries the peak, where it shows most: the peak at its longest waves,
/// each wave's angular frequency over the peak's the root of its wave
/// number over theirs (deep water's ω² = g k), so its longest waves spread
/// narrowest and its shortest widest. Each tile scales to unit slope
/// variance: the wind's as their waves run, each wave number and its
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
        let (period, band, turn) = layout(tile);
        let step = TAU / period;
        // The wind's heading in the tile's lattice: turned by its angle, so
        // that its waves run along the wind where water reads it turned.
        let (sin, cos) = (libm::sin(turn), libm::cos(turn));
        let facing = [
            heading[0] * cos - heading[1] * sin,
            heading[0] * sin + heading[1] * cos,
        ];
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
                    // Its angle from the heading, and its angular frequency
                    // over that of the tile's longest waves.
                    let angle = libm::atan2(
                        facing[0] * along_z - facing[1] * along_x,
                        facing[0] * along_x + facing[1] * along_z,
                    );
                    spread_at(spreading((waves / band[0]).sqrt()), angle)
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
            turn: turn as f32,
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
