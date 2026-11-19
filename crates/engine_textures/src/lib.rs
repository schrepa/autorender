#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine_assets::{ColorSpace, Texture, TextureFormat};
use engine_codec::{Error, ErrorCode, Expected};

const KTX2_IDENTIFIER: [u8; 12] = [
    0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
];

const KTX2_FORMATS: [(u32, TextureFormat, ColorSpace); 5] = [
    (37, TextureFormat::Rgba8, ColorSpace::Linear),
    (43, TextureFormat::Rgba8, ColorSpace::Srgb),
    (141, TextureFormat::Bc5, ColorSpace::Linear),
    (145, TextureFormat::Bc7, ColorSpace::Linear),
    (146, TextureFormat::Bc7, ColorSpace::Srgb),
];

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four_bytes"))
}

pub fn decode_texture(uri: &str, bytes: &[u8]) -> Result<Texture, Error> {
    let invalid = |expected: Expected, field: &str, got: serde_json::Value| {
        Error::new(ErrorCode::InvalidTexture, format!("$.{field}"))
            .expected(expected)
            .got(serde_json::json!({ "file": uri, "value": got }))
    };
    if !bytes.starts_with(&KTX2_IDENTIFIER) {
        return Err(Error::new(ErrorCode::InvalidTexture, "$")
            .expected(Expected::Image)
            .got(uri));
    }
    if bytes.len() < 80 {
        return Err(invalid(Expected::Image, "header", bytes.len().into()));
    }
    let vk_format = word(bytes, 12);
    let (format, space) = KTX2_FORMATS
        .iter()
        .find(|(known, _, _)| *known == vk_format)
        .map(|(_, format, space)| (*format, *space))
        .ok_or_else(|| invalid(Expected::Supported, "vkFormat", vk_format.into()))?;
    let (width, height) = (word(bytes, 20), word(bytes, 24));
    for (field, at, most) in [
        ("pixelDepth", 28, 1),
        ("layerCount", 32, 1),
        ("faceCount", 36, 1),
        ("supercompressionScheme", 44, 0),
    ] {
        let value = word(bytes, at);
        if value > most || (field == "faceCount" && value != 1) {
            return Err(invalid(Expected::Supported, field, value.into()));
        }
    }
    let count = word(bytes, 40) as usize;
    let chain = (u32::BITS - width.max(height).max(1).leading_zeros()) as usize;
    if count == 0 || count > chain || bytes.len() < 80 + 24 * count {
        return Err(invalid(Expected::Supported, "levelCount", count.into()));
    }
    let mut levels = Vec::with_capacity(count);
    for level in 0..count {
        let at = 80 + 24 * level;
        let (offset, length) = (word(bytes, at) as usize, word(bytes, at + 8) as usize);
        let expected = format.level_bytes((width >> level).max(1), (height >> level).max(1));
        let field = format!("levels[{level}]");
        if word(bytes, at + 4) != 0 || word(bytes, at + 12) != 0 || length != expected {
            return Err(invalid(Expected::Matching, &field, length.into()));
        }
        let held = bytes
            .get(offset..offset.saturating_add(length))
            .ok_or_else(|| invalid(Expected::Image, &field, offset.into()))?;
        levels.push(held.to_vec());
    }
    Texture::from_levels(format, width, height, levels).map(|texture| texture.with_color_space(space))
}
