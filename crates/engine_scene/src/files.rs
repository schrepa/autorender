#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use engine_settings::{ErrorCode, Expected};
use std::collections::BTreeMap;
use std::sync::Arc;

use engine_assets::MeshBuffer;
use engine_core::{Error, SceneBuffers, Texture};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneFileKind {
    Image,
    Buffer,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SceneFile {
    pub uri: String,
    pub kind: SceneFileKind,
}

#[derive(Clone, Debug, Default)]
pub struct SceneFiles {
    images: BTreeMap<String, (u32, u32, Vec<u8>)>,
    buffers: Arc<SceneBuffers>,
}

impl SceneFiles {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_image(&mut self, uri: &str, width: u32, height: u32, pixels: Vec<u8>) {
        self.images.insert(uri.to_owned(), (width, height, pixels));
    }

    pub fn add_buffer(&mut self, uri: &str, bytes: Vec<u8>) {
        Arc::make_mut(&mut self.buffers).insert(uri.to_owned(), bytes.into());
    }

    pub fn buffer(&self, uri: &str, path: &str) -> Result<&[u8], Error> {
        self.buffers
            .get(uri)
            .map(Arc::as_ref)
            .ok_or_else(|| missing(uri, path))
    }

    pub(crate) fn buffers(&self) -> Arc<SceneBuffers> {
        Arc::clone(&self.buffers)
    }

    pub(crate) fn texture(&self, uri: &str, path: &str) -> Result<Texture, Error> {
        let (width, height, pixels) = self.images.get(uri).ok_or_else(|| missing(uri, path))?;
        Texture::from_rgba(*width, *height, pixels.clone())
            .map_err(|error| error.coded(ErrorCode::InvalidTexture).within(path))
    }

    pub(crate) fn mesh(&self, uri: &str, path: &str) -> Result<MeshBuffer, Error> {
        let bytes = self.buffers.get(uri).ok_or_else(|| missing(uri, path))?;
        MeshBuffer::parse(bytes, path)
    }
}

fn missing(uri: &str, path: &str) -> Error {
    Error::new(ErrorCode::MissingFile, path)
        .expected(Expected::File)
        .got(uri)
}

