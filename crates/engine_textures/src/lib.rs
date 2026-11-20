#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use std::ops::Range;
use std::sync::Arc;

use engine_assets::{ColorSpace, Texture, TextureFormat};
use engine_codec::{Error, ErrorCode, Expected};

const IDENTIFIER: [u8; 12] = [
    0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
];

const HEADER: usize = 80;

const FORMATS: [(u32, TextureFormat, ColorSpace); 5] = [
    (37, TextureFormat::Rgba8, ColorSpace::Linear),
    (43, TextureFormat::Rgba8, ColorSpace::Srgb),
    (141, TextureFormat::Bc5, ColorSpace::Linear),
    (145, TextureFormat::Bc7, ColorSpace::Linear),
    (146, TextureFormat::Bc7, ColorSpace::Srgb),
];

struct Ktx2 {
    format: TextureFormat,
    space: ColorSpace,
    width: u32,
    height: u32,
    levels: Vec<Range<usize>>,
}

fn invalid(field: &str) -> Error {
    Error::new(ErrorCode::InvalidTexture, format!("$.{field}"))
}

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four_bytes"))
}

fn read_fields(bytes: &[u8]) -> Result<Ktx2, Error> {
    if !bytes.starts_with(&IDENTIFIER) {
        return Err(Error::new(ErrorCode::InvalidTexture, "$").expected(Expected::Image));
    }
    if bytes.len() < HEADER {
        return Err(invalid("header").at_least(HEADER as u64).got(bytes.len()));
    }
    let vk_format = word(bytes, 12);
    let (format, space) = FORMATS
        .iter()
        .find(|(known, _, _)| *known == vk_format)
        .map(|(_, format, space)| (*format, *space))
        .ok_or_else(|| {
            invalid("vkFormat")
                .expected(Expected::Supported)
                .got(vk_format)
        })?;
    let (width, height) = (word(bytes, 20), word(bytes, 24));
    for (field, at, held) in [
        ("pixelDepth", 28, 0),
        ("layerCount", 32, 0),
        ("faceCount", 36, 1),
        ("supercompressionScheme", 44, 0),
    ] {
        let value = word(bytes, at);
        if value != held {
            return Err(invalid(field).expected(Expected::Supported).got(value));
        }
    }
    let count = word(bytes, 40) as usize;
    let table = count
        .checked_mul(24)
        .and_then(|length| length.checked_add(HEADER));
    if table.is_none_or(|end| end > bytes.len()) {
        return Err(invalid("levels")
            .at_least(table.unwrap_or(usize::MAX) as u64)
            .got(bytes.len()));
    }
    let mut levels = Vec::with_capacity(count);
    for level in 0..count {
        let at = HEADER + 24 * level;
        let (offset, length) = (word(bytes, at) as usize, word(bytes, at + 8) as usize);
        if word(bytes, at + 4) != 0 || word(bytes, at + 12) != 0 {
            return Err(invalid(&format!("levels[{level}]"))
                .expected(Expected::Supported)
                .got(length));
        }
        levels.push(offset..offset.saturating_add(length));
    }
    Ok(Ktx2 {
        format,
        space,
        width,
        height,
        levels,
    })
}

pub fn read_texture(bytes: Arc<[u8]>, path: &str) -> Result<Texture, Error> {
    let Ktx2 {
        format,
        space,
        width,
        height,
        levels,
    } = read_fields(&bytes).map_err(|error| error.within(path))?;
    Texture::from_file(format, space, width, height, bytes, levels)
        .map_err(|error| error.within(path))
}

fn sample(offset: u32, length: u32, channel: u32, lower: u32, upper: u32) -> [u32; 4] {
    [
        offset | ((length - 1) << 16) | (channel << 24),
        0,
        lower,
        upper,
    ]
}

fn descriptor(format: TextureFormat, space: ColorSpace) -> Vec<u8> {
    let srgb = space == ColorSpace::Srgb;
    let transfer = if srgb { 2 } else { 1 };
    let (model, dimensions, plane, samples): (u32, u32, u32, Vec<[u32; 4]>) = match format {
        TextureFormat::Rgba8 => {
            let alpha = if srgb { 15 | 0x10 } else { 15 };
            (
                1,
                0,
                4,
                vec![
                    sample(0, 8, 0, 0, 255),
                    sample(8, 8, 1, 0, 255),
                    sample(16, 8, 2, 0, 255),
                    sample(24, 8, alpha, 0, 255),
                ],
            )
        }
        TextureFormat::Bc5 => (
            132,
            3 | (3 << 8),
            16,
            vec![
                sample(0, 64, 0, 0, u32::MAX),
                sample(64, 64, 1, 0, u32::MAX),
            ],
        ),
        TextureFormat::Bc7 => (134, 3 | (3 << 8), 16, vec![sample(0, 128, 0, 0, u32::MAX)]),
    };
    let block = 24 + 16 * samples.len() as u32;
    let mut words = vec![
        4 + block,
        0,
        2 | (block << 16),
        model | (1 << 8) | (transfer << 16),
        dimensions,
        plane,
        0,
    ];
    for held in samples {
        words.extend_from_slice(&held);
    }
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

pub fn write(
    format: TextureFormat,
    space: ColorSpace,
    width: u32,
    height: u32,
    levels: &[&[u8]],
) -> Result<Vec<u8>, Error> {
    let vk_format = FORMATS
        .iter()
        .find(|(_, known, held)| (*known, *held) == (format, space))
        .map(|(vk_format, _, _)| *vk_format)
        .ok_or_else(|| {
            invalid("vkFormat")
                .expected(Expected::Supported)
                .got(format!("{format:?} {space:?}"))
        })?;
    format.check_levels(width, height, levels)?;
    let align = format.block().1.max(4) as usize;
    let count = levels.len();
    let index_end = HEADER + 24 * count;
    let dfd = descriptor(format, space);
    let mut at = index_end + dfd.len();
    let mut placed = vec![0_usize; count];
    for level in (0..count).rev() {
        at = at.next_multiple_of(align);
        placed[level] = at;
        at += levels[level].len();
    }
    let mut out = vec![0_u8; at];
    out[..12].copy_from_slice(&IDENTIFIER);
    let words = [
        vk_format,
        1,
        width,
        height,
        0,
        0,
        1,
        count as u32,
        0,
        index_end as u32,
        dfd.len() as u32,
        0,
        0,
    ];
    for (index, word) in words.iter().enumerate() {
        out[12 + 4 * index..16 + 4 * index].copy_from_slice(&word.to_le_bytes());
    }
    for (level, (offset, bytes)) in placed.iter().zip(levels).enumerate() {
        let base = HEADER + 24 * level;
        let length = bytes.len() as u64;
        out[base..base + 8].copy_from_slice(&(*offset as u64).to_le_bytes());
        out[base + 8..base + 16].copy_from_slice(&length.to_le_bytes());
        out[base + 16..base + 24].copy_from_slice(&length.to_le_bytes());
        out[*offset..*offset + bytes.len()].copy_from_slice(bytes);
    }
    out[index_end..index_end + dfd.len()].copy_from_slice(&dfd);
    Ok(out)
}
