use base64::{Engine as _, engine::general_purpose::STANDARD};
use engine_core::{Error, Texture};
use engine_settings::{ErrorCode, Expected, valid_uri};
use serde::{Deserialize, Serialize};

use crate::SceneFiles;

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
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
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
    pub(crate) fn texture(
        &self,
        path: &str,
        files: Option<&SceneFiles>,
    ) -> Result<(Texture, bool), Error> {
        match (&self.uri, self.width, self.height, &self.rgba8) {
            (Some(uri), None, None, None) => {
                let uri_path = format!("{path}.uri");
                valid_uri(uri, &uri_path)?;
                match files {
                    None => Ok((
                        Texture::new(1, 1, vec![255; 4]).expect("opaque_texel_valid"),
                        false,
                    )),
                    Some(files) => files.texture(uri, &uri_path).map(|texture| (texture, true)),
                }
            }
            (None, Some(width), Some(height), Some(rgba8)) => {
                for (field, size) in [("width", width), ("height", height)] {
                    if size == 0 {
                        return Err(Error::new(
                            ErrorCode::InvalidTexture,
                            format!("{path}.{field}"),
                        )
                        .at_least(1)
                        .got(size));
                    }
                }
                let bytes = engine_core::TextureFormat::Rgba8.level_bytes(width, height);
                let encoded = bytes.div_ceil(3).saturating_mul(4);
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
                    .map(|texture| (texture, false))
                    .map_err(|error| error.coded(ErrorCode::InvalidTexture).within(path))
            }
            (uri, width, height, rgba8) => Err(Error::new(ErrorCode::InvalidTexture, path)
                .got(
                    [
                        ("uri", uri.is_some()),
                        ("width", width.is_some()),
                        ("height", height.is_some()),
                        ("rgba8", rgba8.is_some()),
                    ]
                    .into_iter()
                    .filter_map(|(field, given)| given.then_some(field))
                    .collect::<Vec<_>>(),
                )
                .variants(["uri", "width,height,rgba8"])),
        }
    }
}
