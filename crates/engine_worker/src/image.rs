#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageFormat {
    pub name: &'static str,
    pub what: &'static str,
}

pub const IMAGE_FORMATS: &[ImageFormat] = &[
    ImageFormat {
        name: "png",
        what: "PNG: palettes, low bit depths and transparency expand to 8-bit RGBA, 16-bit channels drop to 8",
    },
    ImageFormat {
        name: "jpeg",
        what: "JPEG, to RGBA with opaque alpha",
    },
    ImageFormat {
        name: "ktx2-rgba8",
        what: "uncompressed RGBA8 KTX2 (VK_FORMAT_R8G8B8A8_UNORM or _SRGB), its base level as stored",
    },
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

pub fn image_format(bytes: &[u8]) -> Option<&'static ImageFormat> {
    let name = if bytes.starts_with(&KTX2_IDENTIFIER) {
        "ktx2-rgba8"
    } else if bytes.starts_with(b"\x89PNG") {
        "png"
    } else if bytes.starts_with(&[0xff, 0xd8]) {
        "jpeg"
    } else {
        return None;
    };
    IMAGE_FORMATS.iter().find(|format| format.name == name)
}

pub fn decode_image(uri: &str, bytes: &[u8]) -> Result<Image, String> {
    match image_format(bytes).map(|format| format.name) {
        Some("png") => decode_png(bytes).map_err(|error| format!("{uri}: {error}")),
        Some("jpeg") => decode_jpeg(bytes).map_err(|error| format!("{uri}: {error}")),
        Some("ktx2-rgba8") => read_rgba8_ktx2(uri, bytes),
        Some(_) | None => Err(format!("{uri} is not a PNG, JPEG or KTX2 image")),
    }
}

const KTX2_IDENTIFIER: [u8; 12] = [
    0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
];

const RGBA8_FORMATS: [u32; 2] = [37, 43];

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

fn read_rgba8_ktx2(uri: &str, bytes: &[u8]) -> Result<Image, String> {
    let invalid = |what: &str| format!("{uri} {what}");
    if bytes.len() < 104 {
        return Err(invalid("is not a valid KTX2 image"));
    }
    if !RGBA8_FORMATS.contains(&word(bytes, 12)) || word(bytes, 44) != 0 {
        return Err(invalid("must be uncompressed RGBA8"));
    }
    let (width, height) = (word(bytes, 20), word(bytes, 24));
    if word(bytes, 28) > 1
        || word(bytes, 32) > 1
        || word(bytes, 36) != 1
        || width == 0
        || height == 0
    {
        return Err(invalid("must be a single 2D image"));
    }
    let (offset, length) = (word(bytes, 80) as usize, word(bytes, 88) as usize);
    if word(bytes, 84) != 0
        || word(bytes, 92) != 0
        || length != width as usize * height as usize * 4
        || offset + length > bytes.len()
    {
        return Err(invalid("has an invalid base level"));
    }
    Ok(Image {
        width,
        height,
        pixels: bytes[offset..offset + length].to_vec(),
    })
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
        png::ColorType::Indexed => return Err("indexed colors did not expand".into()),
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
    let (width, height) = decoder
        .dimensions()
        .ok_or("the decoder found no dimensions")?;
    Ok(Image {
        width: width as u32,
        height: height as u32,
        pixels,
    })
}
