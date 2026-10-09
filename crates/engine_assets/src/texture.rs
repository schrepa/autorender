use crate::AssetError;

/// Maximum admitted width or height: the largest 2D texture every WebGPU
/// device holds.
pub const MAX_TEXTURE_DIMENSION: u32 = 8192;

/// How a texture's texels are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    /// Four bytes per texel, red, green, blue and alpha; one level, whose
    /// mip chain renderers generate.
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
}

/// Immutable sRGB color pixels in tightly packed row-major RGBA8 order, or
/// block-compressed levels.
///
/// Rows run from top to bottom, matching UV origin (0, 0) at the upper left.
/// RGBA8 textures store one mip level; compressed ones store theirs, from the
/// base level down, since renderers cannot generate them. Decoding and
/// transcoding happen before admission; there are no asynchronous loading
/// states in this type. Surfaces are opaque: alpha is the linear mask of a
/// material's emission, 255 where it emits fully.
#[derive(Debug)]
pub struct Texture {
    width: u32,
    height: u32,
    format: TextureFormat,
    levels: Box<[Box<[u8]>]>,
    masked: bool,
}

impl Texture {
    /// Takes ownership of exactly `width * height * 4` bytes. Dimensions must be
    /// nonzero and at most [`MAX_TEXTURE_DIMENSION`]; every alpha byte must be 255.
    pub fn new(width: u32, height: u32, rgba8: Vec<u8>) -> Result<Self, AssetError> {
        let texture = Self::with_emission_mask(width, height, rgba8)?;
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
    pub fn with_alpha(width: u32, height: u32, rgba8: Vec<u8>) -> Result<Self, AssetError> {
        Self::with_emission_mask(width, height, rgba8)
    }

    /// [`Self::with_alpha`], named for its use by opaque materials: alpha
    /// masks the emission of materials using this texture, from 0 (none) to
    /// 255 (the material's full emission).
    pub fn with_emission_mask(width: u32, height: u32, rgba8: Vec<u8>) -> Result<Self, AssetError> {
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
        // The dimension limit bounds the product below usize::MAX on Wasm32.
        let expected = width as usize * height as usize * 4;
        if rgba8.len() != expected {
            return Err(AssetError::InvalidTextureLength {
                expected,
                actual: rgba8.len(),
            });
        }
        // One pass records whether any texel limits emission.
        let masked = rgba8.as_chunks::<4>().0.iter().any(|pixel| pixel[3] != 255);
        Ok(Self {
            width,
            height,
            format: TextureFormat::Rgba8,
            levels: vec![rgba8.into_boxed_slice()].into_boxed_slice(),
            masked,
        })
    }

    /// Block-compressed levels of a `width` × `height`
    /// texture in a compressed `format`: the base level first, each next
    /// level half as large (rounding down, at least 1), at most down to
    /// 1 × 1, each exactly its whole blocks. Dimensions are multiples of 4,
    /// as WebGPU requires. Alpha is kept, as [`Self::with_alpha`] keeps it.
    pub fn compressed(
        format: TextureFormat,
        width: u32,
        height: u32,
        levels: Vec<Vec<u8>>,
    ) -> Result<Self, AssetError> {
        let invalid = |reason| Err(AssetError::InvalidCompressedTexture { reason });
        if format == TextureFormat::Rgba8 {
            return invalid("RGBA8 textures take one level through Texture::new");
        }
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
        if !width.is_multiple_of(4) || !height.is_multiple_of(4) {
            return invalid("dimensions must be multiples of the 4 × 4 block");
        }
        let chain = 32 - (width.max(height)).leading_zeros();
        if levels.is_empty() || levels.len() > chain as usize {
            return invalid("a texture has from one level to a full mip chain");
        }
        for (level, bytes) in levels.iter().enumerate() {
            let expected = format.level_bytes((width >> level).max(1), (height >> level).max(1));
            if bytes.len() != expected {
                return Err(AssetError::InvalidTextureLength {
                    expected,
                    actual: bytes.len(),
                });
            }
        }
        Ok(Self {
            width,
            height,
            format,
            levels: levels.into_iter().map(Vec::into_boxed_slice).collect(),
            masked: true,
        })
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

    /// Every stored level from the base down: one for RGBA8 textures.
    pub fn levels(&self) -> &[Box<[u8]>] {
        &self.levels
    }

    /// Whether any texel's alpha may be below 255, limiting emission there;
    /// compressed textures always may.
    pub fn has_emission_mask(&self) -> bool {
        self.masked
    }

    /// Owned pixel payload bytes; excludes allocation and `Arc` metadata.
    pub fn byte_len(&self) -> usize {
        self.levels.iter().map(|level| level.len()).sum()
    }
}
