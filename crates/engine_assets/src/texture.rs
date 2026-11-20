use std::ops::Range;
use std::sync::Arc;

use engine_codec::{Error, ErrorCode, Expected};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    Rgba8,
    Bc7,
    Bc5,
}

impl TextureFormat {
    pub const fn block(self) -> (u32, u32) {
        match self {
            Self::Rgba8 => (1, 4),
            Self::Bc7 | Self::Bc5 => (4, 16),
        }
    }

    pub fn level_bytes(self, width: u32, height: u32) -> usize {
        let (side, bytes) = self.block();
        (width.div_ceil(side) as usize)
            .saturating_mul(height.div_ceil(side) as usize)
            .saturating_mul(bytes as usize)
    }

    pub fn check_levels(self, width: u32, height: u32, levels: &[&[u8]]) -> Result<(), Error> {
        check_size(self, width, height)?;
        let chain = 32 - width.max(height).leading_zeros();
        if levels.is_empty() || levels.len() > chain as usize {
            return Err(invalid("$.levels")
                .expected(Expected::Array)
                .range(1.0, f64::from(chain))
                .got(levels.len()));
        }
        for (level, bytes) in levels.iter().enumerate() {
            let expected = self.level_bytes((width >> level).max(1), (height >> level).max(1));
            if bytes.len() != expected {
                return Err(invalid(format!("$.levels[{level}]"))
                    .expected(Expected::Matching)
                    .range(expected as f64, expected as f64)
                    .got(bytes.len()));
            }
        }
        Ok(())
    }

    pub fn is_compressed(self) -> bool {
        self != Self::Rgba8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ColorSpace {
    Srgb,
    Linear,
}

#[derive(Debug)]
enum Contents {
    File,
    Base,
    Drawn,
}

#[derive(Debug)]
pub struct Texture {
    width: u32,
    height: u32,
    format: TextureFormat,
    space: ColorSpace,
    bytes: Arc<[u8]>,
    levels: Box<[Range<usize>]>,
    contents: Contents,
}

impl Texture {
    pub fn new(width: u32, height: u32, rgba8: Vec<u8>) -> Result<Self, Error> {
        let texture = Self::from_rgba(width, height, rgba8)?;
        let pixels = texture.pixels().as_chunks::<4>().0;
        if let Some(pixel) = pixels.iter().position(|pixel| pixel[3] != 255) {
            return Err(invalid(format!("$.rgba8[{}]", pixel * 4 + 3))
                .expected(Expected::Opaque)
                .got(pixels[pixel][3])
                .nearest("Texture::from_rgba"));
        }
        Ok(texture)
    }

    pub fn from_rgba(width: u32, height: u32, rgba8: Vec<u8>) -> Result<Self, Error> {
        TextureFormat::Rgba8.check_levels(width, height, &[&rgba8])?;
        let levels = std::iter::once(0..rgba8.len()).collect();
        Ok(Self {
            width,
            height,
            format: TextureFormat::Rgba8,
            space: ColorSpace::Srgb,
            bytes: rgba8.into(),
            levels,
            contents: Contents::Base,
        })
    }

    pub fn target(width: u32, height: u32, space: ColorSpace) -> Result<Self, Error> {
        check_size(TextureFormat::Rgba8, width, height)?;
        Ok(Self {
            width,
            height,
            format: TextureFormat::Rgba8,
            space,
            bytes: Arc::new([]),
            levels: Box::new([]),
            contents: Contents::Drawn,
        })
    }

    pub fn from_file(
        format: TextureFormat,
        space: ColorSpace,
        width: u32,
        height: u32,
        bytes: Arc<[u8]>,
        levels: Vec<Range<usize>>,
    ) -> Result<Self, Error> {
        let held = levels
            .iter()
            .enumerate()
            .map(|(level, range)| {
                bytes.get(range.clone()).ok_or_else(|| {
                    invalid(format!("$.levels[{level}]"))
                        .at_most(bytes.len() as u64)
                        .got(range.end)
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        format.check_levels(width, height, &held)?;
        Ok(Self {
            width,
            height,
            format,
            space,
            bytes,
            levels: levels.into_boxed_slice(),
            contents: Contents::File,
        })
    }

    pub fn color_space(&self) -> ColorSpace {
        self.space
    }

    pub fn is_target(&self) -> bool {
        matches!(self.contents, Contents::Drawn)
    }

    pub fn makes_levels(&self) -> bool {
        matches!(self.contents, Contents::Base)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn pixels(&self) -> &[u8] {
        self.levels().next().unwrap_or_default()
    }

    pub fn format(&self) -> TextureFormat {
        self.format
    }

    pub fn levels(&self) -> impl ExactSizeIterator<Item = &[u8]> + '_ {
        self.levels.iter().map(|range| &self.bytes[range.clone()])
    }

    pub fn byte_len(&self) -> usize {
        self.levels.iter().map(|range| range.len()).sum()
    }
}

fn check_size(format: TextureFormat, width: u32, height: u32) -> Result<(), Error> {
    for (path, size) in [("$.width", width), ("$.height", height)] {
        if size == 0 {
            return Err(invalid(path).at_least(1).got(size));
        }
        if format.is_compressed() && !size.is_multiple_of(4) {
            return Err(invalid(path)
                .expected(Expected::Multiple)
                .got(size)
                .nearest(size.next_multiple_of(4)));
        }
    }
    Ok(())
}

fn invalid(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidTexture, path)
}
