use std::collections::BTreeMap;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use engine_core::{Error, Texture};
use engine_settings::{ErrorCode, Expected, valid_uri};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "dev", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTexture {
    #[cfg_attr(
        feature = "dev",
        schemars(transform = engine_settings::stated(engine_settings::fields::id()))
    )]
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "dev", schemars(range(min = 1)))]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "dev", schemars(range(min = 1)))]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgba8: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "dev",
        schemars(
            length(min = 1, max = 256),
            regex(pattern = r"^[A-Za-z0-9_-][A-Za-z0-9_./-]*$")
        )
    )]
    pub uri: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub variants: BTreeMap<String, String>,
}

pub(crate) enum TextureSource {
    Made(Texture),
    File(String),
    Unpicked,
}

impl SceneTexture {
    pub(crate) fn within(&mut self, folder: &str) {
        for uri in self.uri.iter_mut().chain(self.variants.values_mut()) {
            *uri = format!("{folder}{uri}");
        }
    }

    pub(crate) fn unpicked(&self, path: &str) -> Error {
        Error::new(ErrorCode::UnknownVariant, format!("{path}.variants"))
            .expected(Expected::Present)
            .variants(self.variants.keys().cloned())
    }

    pub(crate) fn source(&self, path: &str, variant: Option<&str>) -> Result<TextureSource, Error> {
        let fields = (self.width, self.height, &self.rgba8);
        match (&self.uri, self.variants.is_empty(), fields) {
            (Some(uri), true, (None, None, None)) => {
                valid_uri(uri, &format!("{path}.uri"))?;
                Ok(TextureSource::File(uri.clone()))
            }
            (None, false, (None, None, None)) => {
                for (name, uri) in &self.variants {
                    valid_uri(uri, &format!("{path}.variants.{name}"))?;
                }
                let Some(name) = variant else {
                    return Ok(TextureSource::Unpicked);
                };
                match self.variants.get(name) {
                    Some(uri) => Ok(TextureSource::File(uri.clone())),
                    None => Err(self.unpicked(path).got(name)),
                }
            }
            (None, true, (Some(width), Some(height), Some(rgba8))) => {
                inline(path, width, height, rgba8).map(TextureSource::Made)
            }
            (uri, single, (width, height, rgba8)) => {
                Err(Error::new(ErrorCode::InvalidTexture, path)
                    .got(
                        [
                            ("uri", uri.is_some()),
                            ("variants", !single),
                            ("width", width.is_some()),
                            ("height", height.is_some()),
                            ("rgba8", rgba8.is_some()),
                        ]
                        .into_iter()
                        .filter_map(|(field, given)| given.then_some(field))
                        .collect::<Vec<_>>(),
                    )
                    .variants(["uri", "variants", "width,height,rgba8"]))
            }
        }
    }
}

fn inline(path: &str, width: u32, height: u32, rgba8: &str) -> Result<Texture, Error> {
    for (field, size) in [("width", width), ("height", height)] {
        if size == 0 {
            return Err(
                Error::new(ErrorCode::InvalidTexture, format!("{path}.{field}"))
                    .at_least(1)
                    .got(size),
            );
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
        Error::new(ErrorCode::InvalidTexture, format!("{path}.rgba8")).expected(Expected::Base64)
    })?;
    Texture::from_rgba(width, height, pixels)
        .map_err(|error| error.coded(ErrorCode::InvalidTexture).within(path))
}
