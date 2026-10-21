use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::AssetError;

/// Maximum admitted width or height: the largest 2D texture every WebGPU
/// device holds.
pub const MAX_TEXTURE_DIMENSION: u32 = 8192;

/// Ids handed to textures so far, across libraries.
static NEXT_TEXTURE: AtomicU64 = AtomicU64::new(1);

/// How a texture's texels are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    /// Four bytes per texel, red, green, blue and alpha.
    Rgba8,
    /// BC7 blocks of 4 × 4 texels in 16 bytes, for devices
    /// with WebGPU's `texture-compression-bc`.
    Bc7,
    /// ETC2 RGBA8 (EAC alpha) blocks of 4 × 4 texels in 16 bytes, for
    /// devices with `texture-compression-etc2`.
    Etc2Rgba8,
    /// ASTC blocks of 4 × 4 texels in 16 bytes, for devices with
    /// `texture-compression-astc`.
    Astc4x4,
}

impl TextureFormat {
    /// Bytes of one level `width` × `height`: whole blocks for compressed
    /// formats.
    pub fn level_bytes(self, width: u32, height: u32) -> usize {
        match self {
            Self::Rgba8 => width as usize * height as usize * 4,
            Self::Bc7 | Self::Etc2Rgba8 | Self::Astc4x4 => {
                width.div_ceil(4) as usize * height.div_ceil(4) as usize * 16
            }
        }
    }

    /// Whether texels are stored in blocks of 4 × 4.
    pub fn is_compressed(self) -> bool {
        self != Self::Rgba8
    }
}

/// What a texture's texels average to, which renderers stand in for its
/// texels where they repeat too small on screen to show: the mean of its
/// red, green and blue decoded from sRGB, the mean of each channel as
/// stored (0 to 1), and the mean square of the slopes its texels make read
/// as a tangent-space normal map, each normal at most about 76 degrees
/// from flat.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureSummary {
    pub color: [f32; 3],
    pub mean: [f32; 4],
    pub slopes: f32,
}

impl TextureSummary {
    /// The summary of `pixels`, RGBA8 texels: what every texture made from
    /// those texels summarizes to, whatever its format.
    pub fn of_rgba8(pixels: &[u8]) -> Self {
        let texels = pixels.as_chunks::<4>().0;
        let count = texels.len().max(1) as f64;
        let decode = srgb_decoded();
        let mut color = [0.0_f64; 3];
        let mut mean = [0_u64; 4];
        let mut slopes = 0.0_f64;
        let unit = |value: u8| f64::from(value) / 127.5 - 1.0;
        for texel in texels {
            for channel in 0..3 {
                color[channel] += f64::from(decode[usize::from(texel[channel])]);
            }
            for (sum, &value) in mean.iter_mut().zip(texel) {
                *sum += u64::from(value);
            }
            // At most about 76 degrees from flat.
            let (x, y, z) = (unit(texel[0]), unit(texel[1]), unit(texel[2]).max(0.25));
            slopes += (x * x + y * y) / (z * z);
        }
        Self {
            color: color.map(|sum| (sum / count) as f32),
            mean: mean.map(|sum| (sum as f64 / (255.0 * count)) as f32),
            slopes: (slopes / count) as f32,
        }
    }
}

/// Each 8-bit sRGB value decoded to linear light.
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

/// Immutable texels in tightly packed row-major RGBA8 order, or
/// block-compressed, with their mip levels where the texture carries them.
///
/// Rows run from top to bottom, matching UV origin (0, 0) at the upper left.
/// A texture holds its base level and, where it was cooked, every level
/// below it; renderers generate the levels of RGBA8 textures holding their
/// base level alone. Decoding and transcoding happen before admission;
/// there are no asynchronous loading states in this type. Surfaces are
/// opaque: alpha is the linear mask of a material's emission, 255 where it
/// emits fully.
#[derive(Debug)]
pub struct Texture {
    id: u64,
    width: u32,
    height: u32,
    format: TextureFormat,
    levels: Box<[Box<[u8]>]>,
    masked: bool,
    summary: OnceLock<TextureSummary>,
}

impl Texture {
    /// Takes ownership of exactly `width * height * 4` bytes. Dimensions must be
    /// nonzero and at most [`MAX_TEXTURE_DIMENSION`]; every alpha byte must be 255.
    pub fn new(width: u32, height: u32, rgba8: Vec<u8>) -> Result<Self, AssetError> {
        let texture = Self::from_rgba(width, height, rgba8)?;
        if let Some(pixel) = texture
            .pixels()
            .as_chunks::<4>()
            .0
            .iter()
            .position(|pixel| pixel[3] != 255)
        {
            return Err(AssetError::UnsupportedTextureAlpha { pixel });
        }
        Ok(texture)
    }

    /// [`Self::new`] with any alpha: coverage for masked and blended
    /// materials; for opaque ones it masks the emission, from
    /// 0 (none) to 255 (the material's full emission).
    pub fn from_rgba(width: u32, height: u32, rgba8: Vec<u8>) -> Result<Self, AssetError> {
        Self::from_levels(TextureFormat::Rgba8, width, height, vec![rgba8])
    }

    /// The levels of a `width` × `height` texture in `format`: the base
    /// level first, each next level half as large (rounding down, at least
    /// 1), at most down to 1 × 1, each exactly its texels or, compressed,
    /// its whole blocks. Compressed dimensions are multiples of 4, as WebGPU
    /// requires. Alpha is kept, as [`Self::from_rgba`] keeps it; compressed
    /// textures may mask emission anywhere.
    pub fn from_levels(
        format: TextureFormat,
        width: u32,
        height: u32,
        levels: Vec<Vec<u8>>,
    ) -> Result<Self, AssetError> {
        let invalid = |reason| Err(AssetError::InvalidCompressedTexture { reason });
        if width == 0
            || height == 0
            || width > MAX_TEXTURE_DIMENSION
            || height > MAX_TEXTURE_DIMENSION
        {
            return Err(AssetError::InvalidTextureDimensions {
                width,
                height,
                limit: MAX_TEXTURE_DIMENSION,
            });
        }
        if format.is_compressed() && (!width.is_multiple_of(4) || !height.is_multiple_of(4)) {
            return invalid("dimensions must be multiples of the 4 × 4 block");
        }
        if levels.is_empty() || levels.len() > full_chain(width, height) as usize {
            return invalid("a texture has from one level to a full mip chain");
        }
        // The dimension limit bounds every product below usize::MAX on Wasm32.
        for (level, bytes) in levels.iter().enumerate() {
            let expected = format.level_bytes((width >> level).max(1), (height >> level).max(1));
            if bytes.len() != expected {
                return Err(AssetError::InvalidTextureLength {
                    expected,
                    actual: bytes.len(),
                });
            }
        }
        // One pass records whether any texel limits emission.
        let masked = !format.is_compressed()
            && levels[0]
                .as_chunks::<4>()
                .0
                .iter()
                .any(|pixel| pixel[3] != 255);
        Ok(Self {
            id: NEXT_TEXTURE.fetch_add(1, Ordering::Relaxed),
            width,
            height,
            format,
            levels: levels.into_iter().map(Vec::into_boxed_slice).collect(),
            masked: masked || format.is_compressed(),
            summary: OnceLock::new(),
        })
    }

    /// This texture whose texels summarize to `summary`, as the cook
    /// measured them on its source.
    #[must_use]
    pub fn with_summary(self, summary: TextureSummary) -> Self {
        let summary = OnceLock::from(summary);
        Self { summary, ..self }
    }

    /// This texture with `masked` saying whether any texel's alpha limits
    /// emission, as the cook found on its source.
    #[must_use]
    pub fn with_emission_mask(self, masked: bool) -> Self {
        Self { masked, ..self }
    }

    /// Unique among the textures of a process: what renderers tell
    /// textures apart by, a texture replaced in its library included.
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// The base level: RGBA8 pixels, or blocks of compressed formats.
    pub fn pixels(&self) -> &[u8] {
        &self.levels[0]
    }

    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// Every stored level from the base down.
    pub fn levels(&self) -> &[Box<[u8]>] {
        &self.levels
    }

    /// Whether it holds every level down to 1 × 1.
    pub fn has_mips(&self) -> bool {
        self.levels.len() == full_chain(self.width, self.height) as usize
    }

    /// What its texels average to: as cooked, or measured on its RGBA8
    /// base level once asked; `None` for compressed textures not cooked
    /// with one.
    pub fn summary(&self) -> Option<TextureSummary> {
        if let Some(summary) = self.summary.get() {
            return Some(*summary);
        }
        (self.format == TextureFormat::Rgba8)
            .then(|| *self.summary.get_or_init(|| TextureSummary::of_rgba8(self.pixels())))
    }

    /// Whether any texel's alpha may be below 255, limiting emission there;
    /// compressed textures always may, unless cooked from texels that do
    /// not.
    pub fn has_emission_mask(&self) -> bool {
        self.masked
    }

    /// Owned pixel payload bytes; excludes allocation and `Arc` metadata.
    pub fn byte_len(&self) -> usize {
        self.levels.iter().map(|level| level.len()).sum()
    }
}

/// Levels of a full mip chain of a `width` × `height` texture.
pub fn full_chain(width: u32, height: u32) -> u32 {
    32 - width.max(height).leading_zeros()
}
