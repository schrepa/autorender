use engine_assets::{ColorSpace, Texture, TextureFormat};
use engine_codec::{Error, ErrorCode, Expected};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageFormat {
    pub name: &'static str,
}

pub const IMAGE_FORMATS: &[ImageFormat] = &[
    ImageFormat { name: "png" },
    ImageFormat { name: "jpeg" },
    ImageFormat { name: "ktx2" },
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

pub fn image_format(bytes: &[u8]) -> Option<&'static ImageFormat> {
    let name = if bytes.starts_with(&KTX2_IDENTIFIER) {
        "ktx2"
    } else if bytes.starts_with(b"\x89PNG") {
        "png"
    } else if bytes.starts_with(&[0xff, 0xd8]) {
        "jpeg"
    } else {
        return None;
    };
    IMAGE_FORMATS.iter().find(|format| format.name == name)
}

fn failed(uri: &str) -> impl Fn(String) -> Error + '_ {
    move |error: String| {
        Error::new(ErrorCode::InvalidTexture, "$")
            .expected(Expected::Image)
            .got(serde_json::json!({ "file": uri, "failed": error }))
    }
}

pub fn decode_image(uri: &str, bytes: &[u8]) -> Result<Image, Error> {
    match image_format(bytes).map(|format| format.name) {
        Some("png") => decode_png(bytes).map_err(failed(uri)),
        Some("jpeg") => decode_jpeg(bytes).map_err(failed(uri)),
        Some(_) | None => Err(Error::new(ErrorCode::InvalidTexture, "$")
            .expected(Expected::Image)
            .got(uri)),
    }
}

pub fn decode_texture(uri: &str, bytes: &[u8]) -> Result<Texture, Error> {
    match image_format(bytes).map(|format| format.name) {
        Some("ktx2") => read_ktx2(uri, bytes),
        Some(_) | None => {
            let image = decode_image(uri, bytes)?;
            Texture::from_rgba(image.width, image.height, image.pixels)
        }
    }
}

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

fn read_ktx2(uri: &str, bytes: &[u8]) -> Result<Texture, Error> {
    let invalid = |expected: Expected, field: &str, got: serde_json::Value| {
        Error::new(ErrorCode::InvalidTexture, format!("$.{field}"))
            .expected(expected)
            .got(serde_json::json!({ "file": uri, "value": got }))
    };
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
    Texture::from_levels(format, width, height, levels)
        .map(|texture| texture.with_color_space(space))
}

fn decode_png(bytes: &[u8]) -> Result<Image, String> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|error| error.to_string())?;
    let mut buffer = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buffer)
        .map_err(|error| error.to_string())?;
    buffer.truncate(info.buffer_size());
    let pixels = match info.color_type {
        png::ColorType::Rgba => buffer,
        png::ColorType::Rgb => buffer
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|&[red, green, blue]| [red, green, blue, 255])
            .collect(),
        png::ColorType::GrayscaleAlpha => buffer
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|&[gray, alpha]| [gray, gray, gray, alpha])
            .collect(),
        png::ColorType::Grayscale => buffer
            .iter()
            .flat_map(|&gray| [gray, gray, gray, 255])
            .collect(),
        png::ColorType::Indexed => return Err("indexed".into()),
    };
    Ok(Image {
        width: info.width,
        height: info.height,
        pixels,
    })
}

fn decode_jpeg(bytes: &[u8]) -> Result<Image, String> {
    let options = zune_jpeg::zune_core::options::DecoderOptions::default()
        .jpeg_set_out_colorspace(zune_jpeg::zune_core::colorspace::ColorSpace::RGBA);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(
        zune_jpeg::zune_core::bytestream::ZCursor::new(bytes),
        options,
    );
    let pixels = decoder.decode().map_err(|error| error.to_string())?;
    let (width, height) = decoder.dimensions().ok_or("dimensions")?;
    Ok(Image {
        width: width as u32,
        height: height as u32,
        pixels,
    })
}
