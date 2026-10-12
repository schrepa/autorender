//! Images as hosts decode them: PNG, JPEG and uncompressed RGBA8 KTX2 to
//! RGBA8 pixels with color and alpha exactly as stored (no
//! premultiplication, no color conversion), with the engine's decoders;
//! which host decodes which format how is [`IMAGE_FORMATS`]'.

/// Who decodes an image format on a host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageDecoder {
    /// The engine's decoder ([`decode_image`]): on the page's job workers,
    /// in place without them, and natively in place.
    Engine,
    /// The page's vendored Basis Universal transcoder, to the block-
    /// compressed format the GPU samples.
    Transcoder,
}

impl ImageDecoder {
    /// Its name, as the page's `IMAGE_FORMATS` names it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Engine => "engine",
            Self::Transcoder => "transcoder",
        }
    }
}

/// An image format a scene's files may hold, and who decodes it on each
/// host; none where the host cannot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageFormat {
    /// Its name, as messages and the page's `IMAGE_FORMATS` name it.
    pub name: &'static str,
    /// What it is.
    pub what: &'static str,
    pub page: Option<ImageDecoder>,
    pub native: Option<ImageDecoder>,
}

/// Every image format a scene's files may hold. Basis KTX2 stays the
/// page's: native hosts have no transcoder.
pub const IMAGE_FORMATS: &[ImageFormat] = &[
    ImageFormat {
        name: "png",
        what: "PNG: palettes, low bit depths and transparency expand to 8-bit RGBA, 16-bit channels drop to 8",
        page: Some(ImageDecoder::Engine),
        native: Some(ImageDecoder::Engine),
    },
    ImageFormat {
        name: "jpeg",
        what: "JPEG, to RGBA with opaque alpha",
        page: Some(ImageDecoder::Engine),
        native: Some(ImageDecoder::Engine),
    },
    ImageFormat {
        name: "ktx2-rgba8",
        what: "uncompressed RGBA8 KTX2 (VK_FORMAT_R8G8B8A8_UNORM or _SRGB), its base level as stored",
        page: Some(ImageDecoder::Engine),
        native: Some(ImageDecoder::Engine),
    },
    ImageFormat {
        name: "ktx2-basis",
        what: "Basis Universal KTX2",
        page: Some(ImageDecoder::Transcoder),
        native: None,
    },
];

/// An image's RGBA8 pixels, rows from the top.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// The format of an image's `bytes`, by their first bytes: one of
/// [`IMAGE_FORMATS`], or none.
pub fn image_format(bytes: &[u8]) -> Option<&'static ImageFormat> {
    let name = if bytes.starts_with(&KTX2_IDENTIFIER) {
        let rgba8 =
            bytes.len() >= 48 && RGBA8_FORMATS.contains(&word(bytes, 12)) && word(bytes, 44) == 0;
        if rgba8 { "ktx2-rgba8" } else { "ktx2-basis" }
    } else if bytes.starts_with(b"\x89PNG") {
        "png"
    } else if bytes.starts_with(&[0xff, 0xd8]) {
        "jpeg"
    } else {
        return None;
    };
    IMAGE_FORMATS.iter().find(|format| format.name == name)
}

/// Image `uri`'s `bytes` decoded to RGBA8 with color and alpha as stored:
/// PNG, JPEG or uncompressed RGBA8 KTX2, the formats the engine's decoder
/// reads.
pub fn decode_image(uri: &str, bytes: &[u8]) -> Result<Image, String> {
    match image_format(bytes).map(|format| format.name) {
        Some("png") => decode_png(bytes).map_err(|error| format!("{uri}: {error}")),
        Some("jpeg") => decode_jpeg(bytes).map_err(|error| format!("{uri}: {error}")),
        Some("ktx2-rgba8") => read_rgba8_ktx2(uri, bytes),
        Some(name) => Err(format!(
            "{uri} is a {name} image, which the engine's decoder does not read (IMAGE_FORMATS)"
        )),
        None => Err(format!("{uri} is not a PNG, JPEG or KTX2 image")),
    }
}

const KTX2_IDENTIFIER: [u8; 12] = [
    0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
];

// Uncompressed RGBA8 formats: VK_FORMAT_R8G8B8A8_UNORM and _SRGB. Whether
// a map decodes sRGB is the material's choice, not the file's.
const RGBA8_FORMATS: [u32; 2] = [37, 43];

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

/// The base level of an uncompressed RGBA8 KTX2 image.
fn read_rgba8_ktx2(uri: &str, bytes: &[u8]) -> Result<Image, String> {
    let invalid = |what: &str| format!("{uri} {what}");
    if bytes.len() < 104 {
        return Err(invalid("is not a valid KTX2 image"));
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
    // The base level's entry leads the level index.
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
    // Palettes, low bit depths and transparency chunks expand to 8-bit
    // channels; 16-bit channels drop to 8.
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
