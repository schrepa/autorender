//! Files from disk as the page fetches them: a scene's images, buffers,
//! virtual meshes and terrain files, and the prefixes of streamed files,
//! whose other bytes renderers read by range from the same path.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use engine::scene::{SceneDocument, SceneFileKind, SceneFiles};
use engine::{Error, ErrorCode};

/// The failure of reading `path` for `what`.
fn unreadable(path: &Path, error: impl std::fmt::Display) -> Error {
    Error::new(
        ErrorCode::MissingFile,
        "$",
        format!("cannot read {}: {error}", path.display()),
    )
}

/// The files scene `source`, read from `path`, references, relative to its
/// folder: images decoded to RGBA8 (PNG, JPEG or uncompressed KTX2),
/// buffers whole, and the prefixes of virtual meshes and terrain files.
pub fn scene_files(path: &Path, source: &str) -> engine::Result<SceneFiles> {
    let document = SceneDocument::from_json(source)?;
    let folder = path.parent().unwrap_or(Path::new("."));
    let mut files = SceneFiles::default();
    for file in document.files() {
        let path = folder.join(&file.uri);
        let url = path.to_string_lossy().into_owned();
        match file.kind {
            SceneFileKind::VirtualMesh => {
                let prefix = read_prefix(&path, 128, |header| {
                    engine::VirtualMesh::prefix_bytes(header)
                })?;
                files.add_virtual_mesh(&file.uri, prefix, Some(&url));
            }
            SceneFileKind::Terrain => {
                let prefix = read_prefix(&path, 64, |header| {
                    engine::host::FileHeader::prefix_bytes(header).ok()
                })?;
                files.add_terrain(&file.uri, prefix, Some(&url));
            }
            SceneFileKind::Buffer => {
                let bytes = std::fs::read(&path).map_err(|error| unreadable(&path, error))?;
                files.add_buffer(&file.uri, bytes);
            }
            SceneFileKind::Image => {
                let bytes = std::fs::read(&path).map_err(|error| unreadable(&path, error))?;
                let image = decode_image(&file.uri, &bytes)?;
                files.add_image(&file.uri, image.width, image.height, image.pixels);
            }
        }
    }
    Ok(files)
}

/// The prefix of the streamed file at `path` whose first `header` bytes say
/// how long it is (`prefix_bytes`).
pub fn read_prefix(
    path: &Path,
    header: usize,
    prefix_bytes: impl FnOnce(&[u8]) -> Option<usize>,
) -> engine::Result<Vec<u8>> {
    let mut file = File::open(path).map_err(|error| unreadable(path, error))?;
    let mut head = vec![0; header];
    file.read_exact(&mut head)
        .map_err(|error| unreadable(path, error))?;
    let length = prefix_bytes(&head).ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidMesh,
            "$",
            format!(
                "{} is not a streamed file of this engine version",
                path.display()
            ),
        )
    })?;
    if length <= head.len() {
        head.truncate(length);
        return Ok(head);
    }
    let mut rest = vec![0; length - head.len()];
    file.read_exact(&mut rest)
        .map_err(|error| unreadable(path, error))?;
    head.extend_from_slice(&rest);
    Ok(head)
}

/// An image's RGBA8 pixels, rows from the top.
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Image `uri`'s `bytes` decoded to RGBA8 with color and alpha as stored:
/// PNG, JPEG or uncompressed RGBA8 KTX2. Basis-compressed KTX2 needs the
/// page's transcoder, which native hosts lack.
pub fn decode_image(uri: &str, bytes: &[u8]) -> engine::Result<Image> {
    let invalid = |message: String| Error::new(ErrorCode::InvalidTexture, "$", message);
    if bytes.starts_with(&KTX2_IDENTIFIER) {
        return read_rgba8_ktx2(uri, bytes)?.ok_or_else(|| {
            invalid(format!(
                "{uri} is a Basis-compressed KTX2 image, which only the page transcodes; \
                 native hosts read uncompressed RGBA8 KTX2, PNG and JPEG"
            ))
        });
    }
    if bytes.starts_with(b"\x89PNG") {
        return decode_png(bytes).map_err(|error| invalid(format!("{uri}: {error}")));
    }
    if bytes.starts_with(&[0xff, 0xd8]) {
        return decode_jpeg(bytes).map_err(|error| invalid(format!("{uri}: {error}")));
    }
    Err(invalid(format!("{uri} is not a PNG, JPEG or KTX2 image")))
}

const KTX2_IDENTIFIER: [u8; 12] = [
    0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
];

// Uncompressed RGBA8 formats: VK_FORMAT_R8G8B8A8_UNORM and _SRGB. Whether
// a map decodes sRGB is the material's choice, not the file's.
const RGBA8_FORMATS: [u32; 2] = [37, 43];

/// The base level of an uncompressed RGBA8 KTX2 image, `None` for any
/// other KTX2 image.
pub fn read_rgba8_ktx2(uri: &str, bytes: &[u8]) -> engine::Result<Option<Image>> {
    let invalid = |what: &str| Error::new(ErrorCode::InvalidTexture, "$", format!("{uri} {what}"));
    if bytes.len() < 104 {
        return Err(invalid("is not a valid KTX2 image"));
    }
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"));
    if !RGBA8_FORMATS.contains(&word(12)) || word(44) != 0 {
        return Ok(None);
    }
    let (width, height) = (word(20), word(24));
    if word(28) > 1 || word(32) > 1 || word(36) != 1 || width == 0 || height == 0 {
        return Err(invalid("must be a single 2D image"));
    }
    // The base level's entry leads the level index.
    let (offset, length) = (word(80) as usize, word(88) as usize);
    if word(84) != 0
        || word(92) != 0
        || length != width as usize * height as usize * 4
        || offset + length > bytes.len()
    {
        return Err(invalid("has an invalid base level"));
    }
    Ok(Some(Image {
        width,
        height,
        pixels: bytes[offset..offset + length].to_vec(),
    }))
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
