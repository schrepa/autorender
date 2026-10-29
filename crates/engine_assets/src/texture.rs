use std::sync::OnceLock;

use engine_settings::{Error, ErrorCode, Expected};

pub const MAX_TEXTURE_DIMENSION: u32 = 8192;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    Rgba8,
    Bc7,
    Etc2Rgba8,
    Astc4x4,
}

impl TextureFormat {
    pub fn level_bytes(self, width: u32, height: u32) -> usize {
        match self {
            Self::Rgba8 => width as usize * height as usize * 4,
            Self::Bc7 | Self::Etc2Rgba8 | Self::Astc4x4 => {
                width.div_ceil(4) as usize * height.div_ceil(4) as usize * 16
            }
        }
    }

    pub fn is_compressed(self) -> bool {
        self != Self::Rgba8
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureSummary {
    pub color: [f32; 3],
    pub mean: [f32; 4],
    pub slopes: f32,
    pub deviation: [f32; 4],
}

impl TextureSummary {
    pub fn of_rgba8(pixels: &[u8]) -> Self {
        let texels = pixels.as_chunks::<4>().0;
        let count = texels.len().max(1) as f64;
        let decode = srgb_decoded();
        let mut color = [0.0_f64; 3];
        let mut mean = [0_u64; 4];
        let mut squares = [0_u64; 4];
        let mut slopes = 0.0_f64;
        let unit = |value: u8| f64::from(value) / 127.5 - 1.0;
        for texel in texels {
            for channel in 0..3 {
                color[channel] += f64::from(decode[usize::from(texel[channel])]);
            }
            for ((sum, square), &value) in mean.iter_mut().zip(&mut squares).zip(texel) {
                *sum += u64::from(value);
                *square += u64::from(value) * u64::from(value);
            }
            let (x, y, z) = (unit(texel[0]), unit(texel[1]), unit(texel[2]).max(0.25));
            slopes += (x * x + y * y) / (z * z);
        }
        let mut deviation = [0.0_f32; 4];
        for ((spread, &sum), &square) in deviation.iter_mut().zip(&mean).zip(&squares) {
            let average = sum as f64 / count;
            let variance = (square as f64 / count - average * average).max(0.0);
            *spread = (variance.sqrt() / 255.0) as f32;
        }
        Self {
            color: color.map(|sum| (sum / count) as f32),
            mean: mean.map(|sum| (sum as f64 / (255.0 * count)) as f32),
            slopes: (slopes / count) as f32,
            deviation,
        }
    }
}

pub fn srgb_decoded() -> &'static [f32; 256] {
    static TABLE: OnceLock<[f32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|value| {
            let value = value as f32 / 255.0;
            if value <= 0.040_45 {
                value / 12.92
            } else {
                libm::powf((value + 0.055) / 1.055, 2.4)
            }
        })
    })
}

#[derive(Debug)]
pub struct Texture {
    width: u32,
    height: u32,
    format: TextureFormat,
    levels: Box<[Box<[u8]>]>,
    masked: bool,
    summary: OnceLock<TextureSummary>,
}

impl Texture {
    pub fn new(width: u32, height: u32, rgba8: Vec<u8>) -> Result<Self, Error> {
        let texture = Self::from_rgba(width, height, rgba8)?;
        let pixels = texture.pixels().as_chunks::<4>().0;
        if let Some(pixel) = pixels.iter().position(|pixel| pixel[3] != 255) {
            return Err(invalid(format!("$.rgba8[{}]", pixel * 4 + 3))
                .expected(Expected::Opaque)
                .got(pixels[pixel][3]));
        }
        Ok(texture)
    }

    pub fn from_rgba(width: u32, height: u32, rgba8: Vec<u8>) -> Result<Self, Error> {
        Self::from_levels(TextureFormat::Rgba8, width, height, vec![rgba8])
    }

    pub fn from_levels(
        format: TextureFormat,
        width: u32,
        height: u32,
        levels: Vec<Vec<u8>>,
    ) -> Result<Self, Error> {
        for (path, size) in [("$.width", width), ("$.height", height)] {
            if !(1..=MAX_TEXTURE_DIMENSION).contains(&size) {
                return Err(invalid(path)
                    .range(1.0, f64::from(MAX_TEXTURE_DIMENSION))
                    .got(size));
            }
            if format.is_compressed() && !size.is_multiple_of(4) {
                return Err(invalid(path)
                    .expected(Expected::Multiple)
                    .got(size)
                    .nearest(size.next_multiple_of(4)));
            }
        }
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
        let masked = !format.is_compressed()
            && levels[0]
                .as_chunks::<4>()
                .0
                .iter()
                .any(|pixel| pixel[3] != 255);
        Ok(Self {
            width,
            height,
            format,
            levels: levels.into_iter().map(Vec::into_boxed_slice).collect(),
            masked: masked || format.is_compressed(),
            summary: OnceLock::new(),
        })
    }

    #[must_use]
    pub fn with_summary(self, summary: TextureSummary) -> Self {
        let summary = OnceLock::from(summary);
        Self { summary, ..self }
    }

    #[must_use]
    pub fn with_emission_mask(self, masked: bool) -> Self {
        Self { masked, ..self }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn pixels(&self) -> &[u8] {
        &self.levels[0]
    }

    pub fn format(&self) -> TextureFormat {
        self.format
    }

    pub fn levels(&self) -> &[Box<[u8]>] {
        &self.levels
    }

    pub fn has_mips(&self) -> bool {
        self.levels.len() == full_chain(self.width, self.height) as usize
    }

    pub fn summary(&self) -> Option<TextureSummary> {
        if let Some(summary) = self.summary.get() {
            return Some(*summary);
        }
        (self.format == TextureFormat::Rgba8).then(|| {
            *self
                .summary
                .get_or_init(|| TextureSummary::of_rgba8(self.pixels()))
        })
    }

    pub fn has_emission_mask(&self) -> bool {
        self.masked
    }

    pub fn byte_len(&self) -> usize {
        self.levels.iter().map(|level| level.len()).sum()
    }
}

fn invalid(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidTexture, path)
}

pub fn full_chain(width: u32, height: u32) -> u32 {
    32 - width.max(height).leading_zeros()
}
