#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod astc;
mod bc7;
mod mips;

use engine_assets::{Texture, TextureFormat, TextureSummary};
use engine_codec::{Error, ErrorCode, Expected};

const COOK_HEADER: usize = 10;

pub fn cook_input(source: &Source, family: Family) -> Vec<u8> {
    let mut input = Vec::with_capacity(COOK_HEADER + source.pixels.len());
    input.extend_from_slice(&source.width.to_le_bytes());
    input.extend_from_slice(&source.height.to_le_bytes());
    input.push(match source.usage {
        Usage::Color => 0,
        Usage::Data => 1,
        Usage::Normal => 2,
    });
    input.push(match family {
        Family::Bc7 => 0,
        Family::Astc => 1,
    });
    input.extend_from_slice(&source.pixels);
    input
}

pub fn cook_job(input: &[u8]) -> Result<Vec<u8>, Error> {
    let header = input.get(..COOK_HEADER).ok_or_else(|| {
        Error::new(ErrorCode::InvalidTexture, "$.input")
            .expected(Expected::Present)
            .at_least(COOK_HEADER as u64)
            .got(input.len())
    })?;
    let word = |at: usize| {
        u32::from_le_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]])
    };
    let usage = match header[8] {
        0 => Usage::Color,
        1 => Usage::Data,
        2 => Usage::Normal,
        other => {
            return Err(Error::new(ErrorCode::InvalidTexture, "$.input.usage")
                .range(0.0, 2.0)
                .got(other));
        }
    };
    let family = match header[9] {
        0 => Family::Bc7,
        1 => Family::Astc,
        other => {
            return Err(Error::new(ErrorCode::InvalidTexture, "$.input.family")
                .range(0.0, 1.0)
                .got(other));
        }
    };
    let source = Source {
        width: word(0),
        height: word(4),
        pixels: input[COOK_HEADER..].to_vec(),
        usage,
    };
    cook(&source, family).map(|texture| record(&texture))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Usage {
    Color,
    Data,
    Normal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Family {
    Bc7,
    Astc,
}

impl Family {
    pub const ALL: [Family; 2] = [Family::Bc7, Family::Astc];

    pub fn format(self) -> TextureFormat {
        match self {
            Self::Bc7 => TextureFormat::Bc7,
            Self::Astc => TextureFormat::Astc4x4,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bc7 => "bc7",
            Self::Astc => "astc",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Source {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    pub usage: Usage,
}

pub fn levels(source: &Source) -> Vec<Vec<u8>> {
    mips::chain(source.width, source.height, &source.pixels, source.usage)
}

pub fn cook(source: &Source, family: Family) -> Result<Texture, Error> {
    let expected = TextureFormat::Rgba8.level_bytes(source.width, source.height);
    if source.pixels.len() != expected {
        return Err(Error::new(ErrorCode::InvalidTexture, "$.pixels")
            .expected(Expected::Matching)
            .range(expected as f64, expected as f64)
            .got(source.pixels.len()));
    }
    let levels = levels(source);
    let compressed = levels
        .iter()
        .enumerate()
        .map(|(level, pixels)| {
            let width = (source.width >> level).max(1);
            let height = (source.height >> level).max(1);
            compress(width, height, pixels, family)
        })
        .collect();
    let masked = source
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .any(|texel| texel[3] != 255);
    Ok(
        Texture::from_levels(family.format(), source.width, source.height, compressed)?
            .with_summary(TextureSummary::of_rgba8(&source.pixels))
            .with_emission_mask(masked),
    )
}

pub fn compress(width: u32, height: u32, pixels: &[u8], family: Family) -> Vec<u8> {
    let (columns, rows) = (width.div_ceil(4) as usize, height.div_ceil(4) as usize);
    let block_row = |row: usize| -> Vec<u8> {
        let mut out = Vec::with_capacity(columns * 16);
        for column in 0..columns {
            let texels: [[u8; 4]; 16] = std::array::from_fn(|index| {
                let x = (column * 4 + index % 4).min(width as usize - 1);
                let y = (row * 4 + index / 4).min(height as usize - 1);
                let at = (y * width as usize + x) * 4;
                [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]
            });
            out.extend_from_slice(&match family {
                Family::Bc7 => bc7::encode_block(&texels),
                Family::Astc => astc::encode_block(&texels),
            });
        }
        out
    };
    rows_in_parallel(rows, block_row).concat()
}

fn rows_in_parallel<T: Send>(rows: usize, make: impl Fn(usize) -> T + Sync) -> Vec<T> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        if threads > 1 && rows > 1 {
            let per = rows.div_ceil(threads);
            let make = &make;
            return std::thread::scope(|scope| {
                let parts: Vec<_> = (0..rows)
                    .step_by(per)
                    .map(|first| {
                        scope.spawn(move || {
                            (first..(first + per).min(rows))
                                .map(make)
                                .collect::<Vec<T>>()
                        })
                    })
                    .collect();
                parts
                    .into_iter()
                    .flat_map(|part| {
                        part.join()
                            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                    })
                    .collect()
            });
        }
    }
    (0..rows).map(make).collect()
}

pub fn solid_block(format: TextureFormat, color: [u8; 4]) -> Option<[u8; 16]> {
    match format {
        TextureFormat::Bc7 => Some(bc7::encode_block(&[color; 16])),
        TextureFormat::Astc4x4 => Some(astc::encode_block(&[color; 16])),
        TextureFormat::Rgba8 | TextureFormat::Etc2Rgba8 => None,
    }
}

pub fn decompress(texture: &Texture) -> Result<Texture, Error> {
    if texture.format() != TextureFormat::Bc7 {
        return Err(Error::new(ErrorCode::InvalidTexture, "$.format").expected(Expected::Supported));
    }
    let levels = texture
        .levels()
        .iter()
        .enumerate()
        .map(|(level, blocks)| {
            let width = (texture.width() >> level).max(1) as usize;
            let height = (texture.height() >> level).max(1) as usize;
            let columns = width.div_ceil(4);
            let mut pixels = vec![0_u8; width * height * 4];
            for (index, block) in blocks.as_chunks::<16>().0.iter().enumerate() {
                let (column, row) = (index % columns, index / columns);
                for (texel, color) in bc7::decode_block(block).iter().enumerate() {
                    let (x, y) = (column * 4 + texel % 4, row * 4 + texel / 4);
                    if x < width && y < height {
                        let at = (y * width + x) * 4;
                        pixels[at..at + 4].copy_from_slice(color);
                    }
                }
            }
            pixels
        })
        .collect();
    let decoded = Texture::from_levels(
        TextureFormat::Rgba8,
        texture.width(),
        texture.height(),
        levels,
    )?
    .with_emission_mask(texture.has_emission_mask());
    Ok(match texture.summary() {
        Some(summary) => decoded.with_summary(summary),
        None => decoded,
    })
}

const MAGIC: &[u8; 4] = b"ETEX";
const HEADER: usize = 4 + 4 + 8 + 48;

pub fn record(texture: &Texture) -> Vec<u8> {
    let summary = texture.summary().unwrap_or(TextureSummary {
        color: [1.0; 3],
        mean: [1.0; 4],
        slopes: 0.0,
        deviation: [0.0; 4],
    });
    let mut out = Vec::with_capacity(HEADER + texture.byte_len());
    out.extend_from_slice(MAGIC);
    out.push(match texture.format() {
        TextureFormat::Rgba8 => 0,
        TextureFormat::Bc7 => 1,
        TextureFormat::Etc2Rgba8 => 2,
        TextureFormat::Astc4x4 => 3,
    });
    out.push(u8::from(texture.has_emission_mask()));
    out.push(texture.levels().len() as u8);
    out.push(0);
    out.extend_from_slice(&texture.width().to_le_bytes());
    out.extend_from_slice(&texture.height().to_le_bytes());
    for value in summary
        .color
        .iter()
        .chain(&summary.mean)
        .chain([&summary.slopes])
        .chain(&summary.deviation)
    {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for level in texture.levels() {
        out.extend_from_slice(level);
    }
    out
}

pub fn decode(bytes: &[u8]) -> Result<Texture, String> {
    let header = bytes
        .get(..HEADER)
        .ok_or("a texture record is shorter than its header")?;
    if &header[..4] != MAGIC {
        return Err("not a texture record".into());
    }
    let format = match header[4] {
        0 => TextureFormat::Rgba8,
        1 => TextureFormat::Bc7,
        2 => TextureFormat::Etc2Rgba8,
        3 => TextureFormat::Astc4x4,
        other => return Err(format!("a texture record names format {other}")),
    };
    let masked = header[5] != 0;
    let count = usize::from(header[6]);
    let word = |at: usize| {
        u32::from_le_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]])
    };
    let (width, height) = (word(8), word(12));
    let value = |index: usize| f32::from_bits(word(16 + 4 * index));
    let summary = TextureSummary {
        color: [value(0), value(1), value(2)],
        mean: [value(3), value(4), value(5), value(6)],
        slopes: value(7),
        deviation: [value(8), value(9), value(10), value(11)],
    };
    let mut levels = Vec::with_capacity(count);
    let mut at = HEADER;
    for level in 0..count {
        let length = format.level_bytes((width >> level).max(1), (height >> level).max(1));
        let level_bytes = bytes
            .get(at..at + length)
            .ok_or("a texture record ends within its levels")?;
        levels.push(level_bytes.to_vec());
        at += length;
    }
    if at != bytes.len() {
        return Err("a texture record holds bytes past its levels".into());
    }
    Texture::from_levels(format, width, height, levels)
        .map(|texture| texture.with_summary(summary).with_emission_mask(masked))
        .map_err(|error| error.to_string())
}
