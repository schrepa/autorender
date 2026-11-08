#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

#[doc(hidden)]
pub mod bc7;
mod image;
mod mips;

pub use image::{IMAGE_FORMATS, Image, ImageFormat, decode_image, image_format};

use engine_assets::{ColorSpace, Texture, TextureFormat, TextureSummary};
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

pub fn cook_source(input: &[u8]) -> Result<(Source, Family), Error> {
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
    Ok((source, family))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Usage {
    Color,
    Data,
    Normal,
}

impl Usage {
    pub fn space(self) -> ColorSpace {
        match self {
            Self::Color => ColorSpace::Srgb,
            Self::Data | Self::Normal => ColorSpace::Linear,
        }
    }
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

pub fn plain(source: &Source) -> Result<Texture, Error> {
    let expected = TextureFormat::Rgba8.level_bytes(source.width, source.height);
    if source.pixels.len() != expected {
        return Err(Error::new(ErrorCode::InvalidTexture, "$.pixels")
            .expected(Expected::Matching)
            .range(expected as f64, expected as f64)
            .got(source.pixels.len()));
    }
    Ok(
        Texture::from_levels(TextureFormat::Rgba8, source.width, source.height, levels(source))?
            .with_summary(TextureSummary::of_rgba8(&source.pixels))
            .with_color_space(source.usage.space()),
    )
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
    .with_color_space(texture.color_space());
    Ok(match texture.summary() {
        Some(summary) => decoded.with_summary(summary),
        None => decoded,
    })
}

const MAGIC: &[u8; 4] = b"ETEX";
const HEADER: usize = 4 + 3 + 8 + 28;

pub fn record(texture: &Texture) -> Vec<u8> {
    let summary = texture.summary().unwrap_or(TextureSummary {
        color: [1.0; 3],
        mean: [1.0; 4],
    });
    let mut out = Vec::with_capacity(HEADER + texture.byte_len());
    out.extend_from_slice(MAGIC);
    out.push(match texture.format() {
        TextureFormat::Rgba8 => 0,
        TextureFormat::Bc7 => 1,
        TextureFormat::Etc2Rgba8 => 2,
        TextureFormat::Astc4x4 => 3,
    });
    out.push(texture.levels().len() as u8);
    out.push(texture.color_space().tag());
    out.extend_from_slice(&texture.width().to_le_bytes());
    out.extend_from_slice(&texture.height().to_le_bytes());
    for value in summary.color.iter().chain(&summary.mean) {
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
    let count = usize::from(header[5]);
    let space = ColorSpace::of_tag(header[6])
        .ok_or_else(|| format!("a texture record names colour space {}", header[6]))?;
    let word = |at: usize| {
        u32::from_le_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]])
    };
    let (width, height) = (word(7), word(11));
    let value = |index: usize| f32::from_bits(word(15 + 4 * index));
    let summary = TextureSummary {
        color: [value(0), value(1), value(2)],
        mean: [value(3), value(4), value(5), value(6)],
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
        .map(|texture| {
            texture
                .with_summary(summary)
                .with_color_space(space)
        })
        .map_err(|error| error.to_string())
}
