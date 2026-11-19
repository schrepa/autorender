use std::fmt;
use std::ops::Deref;

use engine_codec::{Error, ErrorCode, Expected};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    Rgba8,
    Bc7,
    Bc5,
}

impl TextureFormat {
    pub fn level_bytes(self, width: u32, height: u32) -> usize {
        let (columns, rows, block) = match self {
            Self::Rgba8 => (width, height, 4),
            Self::Bc7 | Self::Bc5 => (width.div_ceil(4), height.div_ceil(4), 16),
        };
        (columns as usize)
            .saturating_mul(rows as usize)
            .saturating_mul(block)
    }

    pub fn is_compressed(self) -> bool {
        self != Self::Rgba8
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ColorSpace {
    #[default]
    Srgb,
    Linear,
}

pub struct Level(Box<[u8]>);

impl Deref for Level {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Debug for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl PartialEq for Level {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

#[derive(Debug)]
enum Contents {
    Levels(Box<[Level]>),
    Base(Box<[Level]>),
    Drawn,
}

#[derive(Debug)]
pub struct Texture {
    width: u32,
    height: u32,
    format: TextureFormat,
    space: ColorSpace,
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
        let levels = [Level(rgba8.into_boxed_slice())];
        check_levels(TextureFormat::Rgba8, width, height, &levels)?;
        Ok(Self::of_contents(
            TextureFormat::Rgba8,
            (width, height),
            Contents::Base(Box::new(levels)),
        ))
    }

    pub fn target(width: u32, height: u32, space: ColorSpace) -> Result<Self, Error> {
        check_size(TextureFormat::Rgba8, width, height)?;
        Ok(Self {
            width,
            height,
            format: TextureFormat::Rgba8,
            space,
            contents: Contents::Drawn,
        })
    }

    pub fn from_levels(
        format: TextureFormat,
        width: u32,
        height: u32,
        levels: Vec<Vec<u8>>,
    ) -> Result<Self, Error> {
        check_levels(format, width, height, &levels)?;
        let levels = levels
            .into_iter()
            .map(|bytes| Level(bytes.into_boxed_slice()))
            .collect();
        Ok(Self::of_contents(
            format,
            (width, height),
            Contents::Levels(levels),
        ))
    }

    fn of_contents(format: TextureFormat, (width, height): (u32, u32), contents: Contents) -> Self {
        Self {
            width,
            height,
            format,
            space: ColorSpace::Srgb,
            contents,
        }
    }

    #[must_use]
    pub fn with_color_space(self, space: ColorSpace) -> Self {
        Self { space, ..self }
    }

    pub fn color_space(&self) -> ColorSpace {
        self.space
    }

    pub fn is_target(&self) -> bool {
        matches!(self.contents, Contents::Drawn)
    }

    pub fn makes_levels(&self) -> bool {
        matches!(self.contents, Contents::Base(_))
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn pixels(&self) -> &[u8] {
        self.levels().first().map_or(&[][..], |level| &level[..])
    }

    pub fn format(&self) -> TextureFormat {
        self.format
    }

    pub fn levels(&self) -> &[Level] {
        match &self.contents {
            Contents::Levels(levels) | Contents::Base(levels) => levels,
            Contents::Drawn => &[],
        }
    }

    pub fn byte_len(&self) -> usize {
        self.levels().iter().map(|level| level.len()).sum()
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

fn check_levels(
    format: TextureFormat,
    width: u32,
    height: u32,
    levels: &[impl Deref<Target = [u8]>],
) -> Result<(), Error> {
    check_size(format, width, height)?;
    if levels.is_empty() || levels.len() > full_chain(width, height) as usize {
        return Err(invalid("$.levels")
            .expected(Expected::Array)
            .range(1.0, f64::from(full_chain(width, height)))
            .got(levels.len()));
    }
    for (level, bytes) in levels.iter().enumerate() {
        let expected = format.level_bytes((width >> level).max(1), (height >> level).max(1));
        if bytes.len() != expected {
            return Err(invalid(format!("$.levels[{level}]"))
                .expected(Expected::Matching)
                .range(expected as f64, expected as f64)
                .got(bytes.len()));
        }
    }
    Ok(())
}

fn invalid(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidTexture, path)
}

fn full_chain(width: u32, height: u32) -> u32 {
    32 - width.max(height).leading_zeros()
}
