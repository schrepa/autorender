use base64::{Engine as _, engine::general_purpose::STANDARD};
use engine_core::{Error, Expected, MAX_TEXTURE_DIMENSION, Texture};
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};

use crate::{SceneFiles, files::valid_uri, validation::identifier};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTexture {
    #[cfg_attr(
        feature = "schema",
        schemars(transform = engine_settings::stated(engine_settings::fields::id()))
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schema",
        schemars(transform = engine_settings::stated(engine_settings::fields::id()))
    )]
    pub provided: Option<String>,
}

impl SceneTexture {
    pub fn inline(id: impl Into<String>, width: u32, height: u32, pixels: &[u8]) -> Self {
        Self {
            id: id.into(),
            width: Some(width),
            height: Some(height),
            rgba8: Some(STANDARD.encode(pixels)),
            uri: None,
            provided: None,
        }
    }

    pub fn file(id: impl Into<String>, uri: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            width: None,
            height: None,
            rgba8: None,
            uri: Some(uri.into()),
            provided: None,
        }
    }

    pub(crate) fn texture(&self, path: &str, files: Option<&SceneFiles>) -> Result<Texture, Error> {
        match (
            &self.provided,
            &self.uri,
            self.width,
            self.height,
            &self.rgba8,
        ) {
            (Some(name), None, None, None, None) => {
                identifier(name, &format!("{path}.provided"))?;
                Ok(Texture::new(1, 1, vec![255; 4]).expect("opaque_texel_valid"))
            }
            (None, Some(uri), None, None, None) => {
                let uri_path = format!("{path}.uri");
                valid_uri(uri, &uri_path)?;
                match files {
                    None => Ok(Texture::new(1, 1, vec![255; 4]).expect("opaque_texel_valid")),
                    Some(files) => files.texture(uri, &uri_path),
                }
            }
            (None, None, Some(width), Some(height), Some(rgba8)) => {
                for (field, size) in [("width", width), ("height", height)] {
                    if !(1..=MAX_TEXTURE_DIMENSION).contains(&size) {
                        return Err(Error::new(
                            ErrorCode::InvalidTexture,
                            format!("{path}.{field}"),
                        )
                        .range(1.0, f64::from(MAX_TEXTURE_DIMENSION))
                        .got(size));
                    }
                }
                let bytes = width as usize * height as usize * 4;
                let encoded = bytes.div_ceil(3) * 4;
                if rgba8.len() != encoded {
                    return Err(
                        Error::new(ErrorCode::InvalidTexture, format!("{path}.rgba8"))
                            .expected(Expected::Base64)
                            .range(encoded as f64, encoded as f64)
                            .got(rgba8.len()),
                    );
                }
                let pixels = STANDARD.decode(rgba8).map_err(|_| {
                    Error::new(ErrorCode::InvalidTexture, format!("{path}.rgba8"))
                        .expected(Expected::Base64)
                })?;
                Texture::from_rgba(width, height, pixels)
                    .map_err(|error| error.coded(ErrorCode::InvalidTexture).within(path))
            }
            (provided, uri, width, height, rgba8) => {
                Err(Error::new(ErrorCode::InvalidTexture, path)
                    .got(
                        [
                            ("provided", provided.is_some()),
                            ("uri", uri.is_some()),
                            ("width", width.is_some()),
                            ("height", height.is_some()),
                            ("rgba8", rgba8.is_some()),
                        ]
                        .into_iter()
                        .filter_map(|(field, given)| given.then_some(field))
                        .collect::<Vec<_>>(),
                    )
                    .variants(["uri", "width,height,rgba8", "provided"]))
            }
        }
    }
}
