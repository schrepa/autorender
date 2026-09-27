use crate::AssetError;

/// Maximum admitted width or height: the largest 2D texture every WebGPU
/// device holds.
pub const MAX_TEXTURE_DIMENSION: u32 = 8192;

/// Immutable sRGB color pixels in tightly packed row-major RGBA8 order.
///
/// Rows run from top to bottom, matching UV origin (0, 0) at the upper left.
/// Only one mip level is stored. Decoding and validation happen before admission;
/// there are no encoded formats or asynchronous loading states in this type.
/// Surfaces are opaque: alpha is the linear mask of a material's emission
/// (decision 019), 255 where it emits fully.
#[derive(Debug)]
pub struct Texture {
    width: u32,
    height: u32,
    pixels: Box<[u8]>,
    masked: bool,
}

impl Texture {
    /// Takes ownership of exactly `width * height * 4` bytes. Dimensions must be
    /// nonzero and at most [`MAX_TEXTURE_DIMENSION`]; every alpha byte must be 255.
    pub fn new(width: u32, height: u32, rgba8: Vec<u8>) -> Result<Self, AssetError> {
        let texture = Self::with_emission_mask(width, height, rgba8)?;
        if let Some(pixel) = texture
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .position(|pixel| pixel[3] != 255)
        {
            return Err(AssetError::UnsupportedTextureAlpha { pixel });
        }
        Ok(texture)
    }

    /// [`Self::new`] with any alpha: it masks the emission of materials using
    /// this texture, from 0 (none) to 255 (the material's full emission).
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
            pixels: rgba8.into_boxed_slice(),
            masked,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Whether any texel's alpha is below 255, limiting emission there.
    pub fn has_emission_mask(&self) -> bool {
        self.masked
    }

    /// Owned pixel payload bytes; excludes allocation and `Arc` metadata.
    pub fn byte_len(&self) -> usize {
        self.pixels.len()
    }
}
