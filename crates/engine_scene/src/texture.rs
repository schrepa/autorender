use base64::{Engine as _, engine::general_purpose::STANDARD};
use engine_core::Texture;
use serde::{Deserialize, Serialize};

use crate::SceneError;

/// Immutable opaque RGBA8 sRGB pixels in row-major, top-to-bottom order.
///
/// PNG/GLB decoding happens offline. Base64 is decoded only during validation
/// and loading, never in simulation or rendering. Scenes store one level; the
/// renderer generates mip levels and samples them trilinearly with
/// clamp-to-edge. The complete scene still must fit 1 MiB.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTexture {
    #[cfg_attr(
        feature = "schema",
        schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
    )]
    pub id: String,
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 2048)))]
    pub width: u32,
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 2048)))]
    pub height: u32,
    /// Canonical padded RFC 4648 base64; each alpha byte must equal 255.
    #[cfg_attr(feature = "schema", schemars(length(max = 1_048_576)))]
    pub rgba8: String,
}

impl SceneTexture {
    pub(crate) fn texture(&self, path: &str) -> Result<Texture, SceneError> {
        if !(1..=2048).contains(&self.width) || !(1..=2048).contains(&self.height) {
            return Err(SceneError::new(
                "invalid_texture",
                path,
                "texture dimensions must be 1..2048",
            ));
        }
        let bytes = self.width as usize * self.height as usize * 4;
        let encoded = bytes.div_ceil(3) * 4;
        if self.rgba8.len() > super::document::MAX_SOURCE_BYTES || self.rgba8.len() != encoded {
            return Err(SceneError::new(
                "invalid_texture",
                format!("{path}.rgba8"),
                "base64 length must match width × height × 4 bytes and fit the scene source limit",
            ));
        }
        let pixels = STANDARD.decode(&self.rgba8).map_err(|error| {
            SceneError::new(
                "invalid_texture",
                format!("{path}.rgba8"),
                error.to_string(),
            )
        })?;
        Texture::new(self.width, self.height, pixels)
            .map_err(|error| SceneError::new("invalid_texture", path, error.to_string()))
    }
}
