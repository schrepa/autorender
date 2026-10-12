//! `ETR1` terrain files: a tile pyramid's tiles, each fetched on its own
//! by byte range.
//!
//! Little-endian. A 64-byte header (magic, version, level-0 tiles along x
//! and z, spacing, levels, the height step, tile count, where the directory
//! starts and the prefix's length), then a directory of 24-byte entries,
//! one per tile, levels coarsest first and rows along z of columns along x
//! (the tile's byte offset as `u64`, its length, flags, and its lowest and
//! highest height), then the tiles. The prefix (header and directory) is
//! all a host loads up front.
//!
//! A tile holds heights quantized to multiples of the step, absolute, so
//! neighbors and levels decode shared samples to the same bits: its
//! [`TILE_STRIDE`]-squared samples row by row, each predicted from its left,
//! upper and upper-left neighbors (LOCO-I's median predictor) and the
//! residual Rice-coded with a parameter per row; then, with the holes flag,
//! its hole bits as raw 64-bit words.
// A file comes from wherever a scene points: every narrowing cast is
// checked.
#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use crate::{Pyramid, TILE_SAMPLES, TILE_STRIDE, Tile, TileKey};

pub const TERRAIN_FILE_MAGIC: [u8; 4] = *b"ETR1";
pub const TERRAIN_FILE_VERSION: u32 = 1;
const HEADER_LENGTH: u32 = 64;
const HEADER_BYTES: usize = HEADER_LENGTH as usize;
const ENTRY_BYTES: usize = 24;
const FLAG_HOLES: u32 = 1;
const HOLE_WORDS: usize = ((TILE_SAMPLES * TILE_SAMPLES) as usize).div_ceil(64);
/// Unary quotients this long escape to a raw 32-bit value.
const ESCAPE: u32 = 24;

/// Why a terrain file or tile is refused.
#[derive(Clone, Debug, PartialEq)]
pub enum FormatError {
    Magic,
    Version(u32),
    /// The header's dimensions make no pyramid.
    Pyramid(String),
    /// The step must be finite and positive.
    Step,
    /// Fewer bytes than the header and directory, or a tile's payload,
    /// need.
    Truncated,
    /// The directory or a payload contradicts the header.
    Corrupt(&'static str),
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Magic => write!(formatter, "not an ETR1 terrain file"),
            Self::Version(version) => write!(formatter, "unsupported ETR1 version {version}"),
            Self::Pyramid(error) => write!(formatter, "invalid terrain dimensions: {error}"),
            Self::Step => write!(formatter, "the height step must be finite and positive"),
            Self::Truncated => write!(formatter, "truncated terrain data"),
            Self::Corrupt(what) => write!(formatter, "corrupt terrain data: {what}"),
        }
    }
}

impl std::error::Error for FormatError {}

/// A tile's directory entry: where its payload lies, whether it has
/// holes, and its height bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FileTile {
    pub offset: u64,
    pub length: u32,
    pub holes: bool,
    pub bounds: [f32; 2],
}

/// A terrain file's header and directory.
#[derive(Clone, Debug, PartialEq)]
pub struct FileHeader {
    pyramid: Pyramid,
    step: f32,
    tiles: Vec<FileTile>,
    /// Directory index of each level's first tile.
    level_starts: Vec<usize>,
}

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

impl FileHeader {
    /// Bytes of the header and directory, from the file's first 64 bytes.
    pub fn prefix_bytes(first: &[u8]) -> Result<usize, FormatError> {
        if first.len() < HEADER_BYTES {
            return Err(FormatError::Truncated);
        }
        if first[..4] != TERRAIN_FILE_MAGIC {
            return Err(FormatError::Magic);
        }
        Ok(word(first, 36) as usize)
    }

    /// Parses the header and directory from the file's prefix (or the whole
    /// file).
    pub fn parse(bytes: &[u8]) -> Result<Self, FormatError> {
        let prefix = Self::prefix_bytes(bytes)?;
        let version = word(bytes, 4);
        if version != TERRAIN_FILE_VERSION {
            return Err(FormatError::Version(version));
        }
        let pyramid = Pyramid::new(
            [word(bytes, 8), word(bytes, 12)],
            f32::from_bits(word(bytes, 16)),
        )
        .map_err(|error| FormatError::Pyramid(error.to_string()))?;
        if word(bytes, 20) != pyramid.levels() {
            return Err(FormatError::Corrupt("level count"));
        }
        let step = f32::from_bits(word(bytes, 24));
        if !(step.is_finite() && step > 0.0) {
            return Err(FormatError::Step);
        }
        let count = word(bytes, 28) as usize;
        if count != pyramid.tile_count() as usize {
            return Err(FormatError::Corrupt("tile count"));
        }
        let directory = word(bytes, 32) as usize;
        if directory != HEADER_BYTES || prefix != directory + count * ENTRY_BYTES {
            return Err(FormatError::Corrupt("directory"));
        }
        if bytes.len() < prefix {
            return Err(FormatError::Truncated);
        }
        let tiles = (0..count)
            .map(|index| {
                let at = directory + index * ENTRY_BYTES;
                let entry = &bytes[at..at + ENTRY_BYTES];
                FileTile {
                    offset: u64::from_le_bytes(entry[..8].try_into().expect("eight bytes")),
                    length: word(entry, 8),
                    holes: word(entry, 12) & FLAG_HOLES != 0,
                    bounds: [
                        f32::from_bits(word(entry, 16)),
                        f32::from_bits(word(entry, 20)),
                    ],
                }
            })
            .collect::<Vec<_>>();
        if tiles
            .iter()
            .any(|tile| tile.offset < prefix as u64 || tile.length < 4)
        {
            return Err(FormatError::Corrupt("tile range"));
        }
        Ok(Self {
            level_starts: level_starts(&pyramid),
            pyramid,
            step,
            tiles,
        })
    }

    pub fn pyramid(&self) -> &Pyramid {
        &self.pyramid
    }

    /// Meters between quantized heights.
    pub fn step(&self) -> f32 {
        self.step
    }

    /// `key`'s directory entry.
    pub fn tile(&self, key: TileKey) -> Option<&FileTile> {
        self.pyramid
            .contains(key)
            .then(|| &self.tiles[directory_index(&self.pyramid, &self.level_starts, key)])
    }
}

/// Directory index of each level's first tile, coarsest level first.
fn level_starts(pyramid: &Pyramid) -> Vec<usize> {
    let mut starts = vec![0; pyramid.levels() as usize];
    let mut next = 0;
    for level in (0..pyramid.levels()).rev() {
        starts[level as usize] = next;
        let [x, z] = pyramid.tiles(level);
        next += (x * z) as usize;
    }
    starts
}

fn directory_index(pyramid: &Pyramid, starts: &[usize], key: TileKey) -> usize {
    let [columns, _] = pyramid.tiles(u32::from(key.level));
    starts[key.level as usize] + usize::from(key.z) * columns as usize + usize::from(key.x)
}

/// LOCO-I's median predictor of a sample from its left, upper and
/// upper-left neighbors.
#[inline]
fn predict(left: i32, up: i32, corner: i32) -> i32 {
    if corner >= left.max(up) {
        left.min(up)
    } else if corner <= left.min(up) {
        left.max(up)
    } else {
        left + up - corner
    }
}

struct BitWriter {
    bytes: Vec<u8>,
    word: u64,
    bits: u32,
}

impl BitWriter {
    fn push(&mut self, value: u32, count: u32) {
        debug_assert!(count <= 32, "a push writes at most 32 bits");
        if count == 0 {
            return;
        }
        self.word |= u64::from(value & (u32::MAX >> (32 - count))) << self.bits;
        self.bits += count;
        while self.bits >= 8 {
            self.bytes.push(self.word.to_le_bytes()[0]);
            self.word >>= 8;
            self.bits -= 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            self.bytes.push(self.word.to_le_bytes()[0]);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    at: usize,
    word: u64,
    bits: u32,
}

impl BitReader<'_> {
    fn refill(&mut self) {
        while self.bits <= 56 && self.at < self.bytes.len() {
            self.word |= u64::from(self.bytes[self.at]) << self.bits;
            self.at += 1;
            self.bits += 8;
        }
    }

    fn take(&mut self, count: u32) -> Result<u32, FormatError> {
        debug_assert!(count <= 32, "a take reads at most 32 bits");
        if count == 0 {
            return Ok(0);
        }
        self.refill();
        if self.bits < count {
            return Err(FormatError::Truncated);
        }
        let [a, b, c, d, ..] = self.word.to_le_bytes();
        let value = u32::from_le_bytes([a, b, c, d]) & (u32::MAX >> (32 - count));
        self.word >>= count;
        self.bits -= count;
        Ok(value)
    }

    /// Ones before the next zero, up to `limit`.
    fn ones(&mut self, limit: u32) -> Result<u32, FormatError> {
        let mut count = 0;
        while count < limit {
            if self.take(1)? == 0 {
                return Ok(count);
            }
            count += 1;
        }
        Ok(limit)
    }
}

fn zigzag(value: i32) -> u32 {
    ((value << 1) ^ (value >> 31)).cast_unsigned()
}

fn unzigzag(value: u32) -> i32 {
    (value >> 1).cast_signed() ^ -(value & 1).cast_signed()
}

/// Encodes `tile`'s samples quantized to `step` meters, and its holes.
fn encode_tile(tile: &Tile, step: f32) -> Vec<u8> {
    let stride = TILE_STRIDE as usize;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "heights lie far inside i32"
    )]
    let quantized: Vec<i32> = tile
        .heights()
        .iter()
        .map(|&height| (f64::from(height) / f64::from(step)).round() as i32)
        .collect();
    let mut writer = BitWriter {
        bytes: Vec::with_capacity(stride * stride),
        word: 0,
        bits: 0,
    };
    let mut residuals = vec![0u32; stride];
    for z in 0..stride {
        for (x, residual) in residuals.iter_mut().enumerate() {
            let at = z * stride + x;
            let value = quantized[at];
            let predicted = match (x, z) {
                (0, 0) => 0,
                (_, 0) => quantized[at - 1],
                (0, _) => quantized[at - stride],
                _ => predict(
                    quantized[at - 1],
                    quantized[at - stride],
                    quantized[at - stride - 1],
                ),
            };
            *residual = zigzag(value.wrapping_sub(predicted));
        }
        let mean = residuals.iter().map(|&r| u64::from(r)).sum::<u64>() / stride as u64;
        let k = (u64::BITS - mean.leading_zeros()).min(31);
        writer.push(k, 5);
        for &residual in &residuals {
            let quotient = residual >> k;
            if quotient < ESCAPE {
                writer.push((1u32 << quotient) - 1, quotient);
                writer.push(0, 1);
                writer.push(residual, k);
            } else {
                writer.push((1u32 << ESCAPE) - 1, ESCAPE);
                writer.push(residual, 32);
            }
        }
    }
    let mut bytes = writer.finish();
    if let Some(holes) = tile.hole_bits() {
        for word in holes {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }
    bytes
}

/// Decodes `key`'s tile from its payload `bytes`.
pub fn decode_tile(header: &FileHeader, key: TileKey, bytes: &[u8]) -> Result<Tile, FormatError> {
    let entry = header.tile(key).ok_or(FormatError::Corrupt("tile key"))?;
    if bytes.len() != entry.length as usize {
        return Err(FormatError::Truncated);
    }
    let stride = TILE_STRIDE as usize;
    let hole_bytes = if entry.holes { HOLE_WORDS * 8 } else { 0 };
    let samples_end = bytes
        .len()
        .checked_sub(hole_bytes)
        .ok_or(FormatError::Truncated)?;
    let mut reader = BitReader {
        bytes: &bytes[..samples_end],
        at: 0,
        word: 0,
        bits: 0,
    };
    let mut quantized = vec![0i32; stride * stride];
    for z in 0..stride {
        let k = reader.take(5)?;
        if k > 31 {
            return Err(FormatError::Corrupt("rice parameter"));
        }
        for x in 0..stride {
            let at = z * stride + x;
            let quotient = reader.ones(ESCAPE)?;
            let residual = if quotient < ESCAPE {
                (quotient << k) | reader.take(k)?
            } else {
                reader.take(32)?
            };
            let predicted = match (x, z) {
                (0, 0) => 0,
                (_, 0) => quantized[at - 1],
                (0, _) => quantized[at - stride],
                _ => predict(
                    quantized[at - 1],
                    quantized[at - stride],
                    quantized[at - stride - 1],
                ),
            };
            quantized[at] = predicted.wrapping_add(unzigzag(residual));
        }
    }
    let step = header.step();
    #[expect(clippy::cast_possible_truncation, reason = "heights are f32 values")]
    let heights = quantized
        .iter()
        .map(|&value| (f64::from(value) * f64::from(step)) as f32)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let holes = entry.holes.then(|| {
        bytes[samples_end..]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| u64::from_le_bytes(*chunk))
            .collect::<Vec<_>>()
            .into_boxed_slice()
    });
    let spacing = header.pyramid().spacing(u32::from(key.level));
    Tile::new(key, spacing, heights, holes).ok_or(FormatError::Corrupt("tile samples"))
}

/// Writes a terrain file of `pyramid`'s tiles, which `tile` provides, with
/// heights quantized to `step` meters.
pub fn encode_file(
    pyramid: &Pyramid,
    step: f32,
    mut tile: impl FnMut(TileKey) -> Tile,
) -> Result<Vec<u8>, FormatError> {
    if !(step.is_finite() && step > 0.0) {
        return Err(FormatError::Step);
    }
    let count = pyramid.tile_count() as usize;
    let prefix = HEADER_BYTES + count * ENTRY_BYTES;
    let mut bytes = vec![0u8; prefix];
    bytes[..4].copy_from_slice(&TERRAIN_FILE_MAGIC);
    let put = |bytes: &mut Vec<u8>, at: usize, value: u32| {
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    };
    put(&mut bytes, 4, TERRAIN_FILE_VERSION);
    put(&mut bytes, 8, pyramid.base_tiles()[0]);
    put(&mut bytes, 12, pyramid.base_tiles()[1]);
    put(&mut bytes, 16, pyramid.spacing(0).to_bits());
    put(&mut bytes, 20, pyramid.levels());
    put(&mut bytes, 24, step.to_bits());
    put(&mut bytes, 28, pyramid.tile_count());
    put(&mut bytes, 32, HEADER_LENGTH);
    let prefix_length = u32::try_from(prefix).map_err(|_| FormatError::Corrupt("directory"))?;
    put(&mut bytes, 36, prefix_length);
    let mut index = 0;
    for level in (0..pyramid.levels()).rev() {
        for key in pyramid.level_tiles(level) {
            let tile = tile(key);
            let payload = encode_tile(&tile, step);
            let at = HEADER_BYTES + index * ENTRY_BYTES;
            let offset = bytes.len() as u64;
            bytes[at..at + 8].copy_from_slice(&offset.to_le_bytes());
            let length =
                u32::try_from(payload.len()).map_err(|_| FormatError::Corrupt("tile length"))?;
            put(&mut bytes, at + 8, length);
            put(
                &mut bytes,
                at + 12,
                if tile.hole_bits().is_some() {
                    FLAG_HOLES
                } else {
                    0
                },
            );
            // Bounds of the quantized heights decoders see.
            let [low, high] = tile.bounds();
            #[expect(clippy::cast_possible_truncation, reason = "bounds are f32 values")]
            let quantize = |height: f32| {
                ((f64::from(height) / f64::from(step)).round() * f64::from(step)) as f32
            };
            put(&mut bytes, at + 16, quantize(low).to_bits());
            put(&mut bytes, at + 20, quantize(high).to_bits());
            bytes.extend_from_slice(&payload);
            index += 1;
        }
    }
    Ok(bytes)
}
