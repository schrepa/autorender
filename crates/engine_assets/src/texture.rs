use crate::AssetError;

/// Maximum admitted width or height for the portable texture profile.
pub const MAX_TEXTURE_DIMENSION: u32 = 2048;

/// Immutable opaque sRGB color pixels in tightly packed row-major RGBA8 order.
///
/// Rows run from top to bottom, matching UV origin (0, 0) at the upper left.
/// Only one mip level is stored. Decoding and validation happen before admission;
/// there are no encoded formats or asynchronous loading states in this type.
#[derive(Debug)]
pub struct Texture {
    width: u32,
    height: u32,
    pixels: Box<[u8]>,
}

impl Texture {
    /// Takes ownership of exactly `width * height * 4` bytes. Dimensions must be
    /// nonzero and at most [`MAX_TEXTURE_DIMENSION`]; every alpha byte must be 255.
    pub fn new(width: u32, height: u32, rgba8: Vec<u8>) -> Result<Self, AssetError> {
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
        if let Some(pixel) = rgba8
            .as_chunks::<4>()
            .0
            .iter()
            .position(|pixel| pixel[3] != 255)
        {
            return Err(AssetError::UnsupportedTextureAlpha { pixel });
        }
        Ok(Self {
            width,
            height,
            pixels: rgba8.into_boxed_slice(),
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

    /// Owned pixel payload bytes; excludes allocation and `Arc` metadata.
    pub fn byte_len(&self) -> usize {
        self.pixels.len()
    }
}
