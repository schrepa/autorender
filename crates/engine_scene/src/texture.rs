use base64::{Engine as _, engine::general_purpose::STANDARD};
use engine_core::{MAX_TEXTURE_DIMENSION, Texture};
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};

use crate::{SceneError, SceneFiles, files::valid_uri};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTexture {
    #[cfg_attr(
        feature = "schema",
        schemars(transform = engine_settings::stated(engine_settings::rule::id()))
    )]
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schema",
        schemars(range(min = 1, max = engine_core::MAX_TEXTURE_DIMENSION))
    )]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schema",
        schemars(range(min = 1, max = engine_core::MAX_TEXTURE_DIMENSION))
    )]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgba8: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schema",
        schemars(
            length(min = 1, max = 256),
            regex(pattern = r"^[A-Za-z0-9_-][A-Za-z0-9_./-]*$")
        )
    )]
    pub uri: Option<String>,
}

impl SceneTexture {
    pub fn inline(id: impl Into<String>, width: u32, height: u32, pixels: &[u8]) -> Self {
        Self {
            id: id.into(),
            width: Some(width),
            height: Some(height),
            rgba8: Some(STANDARD.encode(pixels)),
            uri: None,
        }
    }

    pub fn file(id: impl Into<String>, uri: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            width: None,
            height: None,
            rgba8: None,
            uri: Some(uri.into()),
        }
    }

    pub(crate) fn texture(
        &self,
        path: &str,
        files: Option<&SceneFiles>,
    ) -> Result<Texture, SceneError> {
        match (&self.uri, self.width, self.height, &self.rgba8) {
            (Some(uri), None, None, None) => {
                let uri_path = format!("{path}.uri");
                valid_uri(uri, &uri_path)?;
                match files {
                    None => Ok(Texture::new(1, 1, vec![255; 4]).expect("opaque_texel_valid")),
                    Some(files) => files.texture(uri, &uri_path),
                }
            }
            (None, Some(width), Some(height), Some(rgba8)) => {
                if !(1..=MAX_TEXTURE_DIMENSION).contains(&width)
                    || !(1..=MAX_TEXTURE_DIMENSION).contains(&height)
                {
                    return Err(SceneError::new(
                        ErrorCode::InvalidTexture,
                        path,
                        format!("texture dimensions must be 1..{MAX_TEXTURE_DIMENSION}"),
                    ));
                }
                let bytes = width as usize * height as usize * 4;
                let encoded = bytes.div_ceil(3) * 4;
                if rgba8.len() != encoded {
                    return Err(SceneError::new(
                        ErrorCode::InvalidTexture,
                        format!("{path}.rgba8"),
                        "base64 length must match width × height × 4 bytes",
                    ));
                }
                let pixels = STANDARD.decode(rgba8).map_err(|error| {
                    SceneError::new(
                        ErrorCode::InvalidTexture,
                        format!("{path}.rgba8"),
                        error.to_string(),
                    )
                })?;
                Texture::from_rgba(width, height, pixels).map_err(|error| {
                    SceneError::new(ErrorCode::InvalidTexture, path, error.to_string())
                })
            }
            _ => Err(SceneError::new(
                ErrorCode::InvalidTexture,
                path,
                "a texture has either width, height and rgba8, or a uri",
            )),
        }
    }
}
