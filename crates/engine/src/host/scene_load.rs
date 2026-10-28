use engine_core::{ErrorCode, TextureFormat, VirtualMesh};
use engine_scene::{SceneDocument, SceneFileKind, SceneFiles};
pub use engine_worker::{
    IMAGE_FORMATS, Image, ImageDecoder, ImageFormat, decode_image, image_format,
};

use crate::{Error, Result};

const VIRTUAL_HEADER: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadingHost {
    Page,
    Native,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileRead {
    pub uri: String,
    pub prefix: Option<usize>,
}

#[derive(Debug)]
pub struct ImageRead {
    pub decoder: ImageDecoder,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Asked,
    Prefix,
    Decoding,
    Held,
}

#[derive(Debug)]
pub struct SceneLoad {
    host: LoadingHost,
    files: SceneFiles,
    entries: Vec<(String, SceneFileKind, Stage)>,
    reads: Vec<FileRead>,
}

impl SceneLoad {
    pub fn new(source: &str, host: LoadingHost) -> Result<Self> {
        let document = SceneDocument::from_json(source)?;
        let entries: Vec<(String, SceneFileKind, Stage)> = document
            .files()
            .into_iter()
            .map(|file| (file.uri, file.kind, Stage::Asked))
            .collect();
        let reads = entries
            .iter()
            .map(|(uri, kind, _)| FileRead {
                uri: uri.clone(),
                prefix: match kind {
                    SceneFileKind::VirtualMesh => Some(VIRTUAL_HEADER),
                    SceneFileKind::Image | SceneFileKind::Buffer => None,
                },
            })
            .collect();
        Ok(Self {
            host,
            files: SceneFiles::default(),
            entries,
            reads,
        })
    }

    pub fn take_reads(&mut self) -> Vec<FileRead> {
        std::mem::take(&mut self.reads)
    }

    pub fn read(&mut self, uri: &str, bytes: Vec<u8>, url: &str) -> Result<Option<ImageRead>> {
        let at = self
            .entries
            .iter()
            .position(|(file, _, _)| file == uri)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::MissingFile,
                    "$",
                    format!("the scene names no file {uri}"),
                )
            })?;
        let (kind, stage) = (self.entries[at].1, self.entries[at].2);
        match kind {
            SceneFileKind::Buffer => {
                self.files.add_buffer(uri, bytes);
                self.entries[at].2 = Stage::Held;
                Ok(None)
            }
            SceneFileKind::Image => {
                let format = engine_worker::image_format(&bytes).ok_or_else(|| {
                    invalid_texture(format!("{uri} is not a PNG, JPEG or KTX2 image"))
                })?;
                let decoder = match self.host {
                    LoadingHost::Page => format.page,
                    LoadingHost::Native => format.native,
                }
                .ok_or_else(|| {
                    invalid_texture(format!(
                        "{uri} is a {} image, which only the page reads (IMAGE_FORMATS)",
                        format.what
                    ))
                })?;
                self.entries[at].2 = Stage::Decoding;
                Ok(Some(ImageRead { decoder, bytes }))
            }
            SceneFileKind::VirtualMesh => {
                let length = bytes
                    .get(..VIRTUAL_HEADER)
                    .and_then(VirtualMesh::prefix_bytes);
                let Some(length) = length else {
                    return Err(Error::new(
                        ErrorCode::InvalidMesh,
                        "$",
                        format!("{url} is not a virtual mesh file of this engine version"),
                    ));
                };
                if bytes.len() < length {
                    if stage == Stage::Prefix {
                        return Err(Error::new(
                            ErrorCode::MissingFile,
                            "$",
                            format!("{url} ends within its prefix"),
                        ));
                    }
                    self.entries[at].2 = Stage::Prefix;
                    self.reads.push(FileRead {
                        uri: uri.to_owned(),
                        prefix: Some(length),
                    });
                    return Ok(None);
                }
                let mut prefix = bytes;
                prefix.truncate(length);
                self.files.add_virtual_mesh(uri, prefix, Some(url));
                self.entries[at].2 = Stage::Held;
                Ok(None)
            }
        }
    }

    pub fn image(&mut self, uri: &str, image: Image) {
        self.files
            .add_image(uri, image.width, image.height, image.pixels);
        self.held(uri);
    }

    pub fn decode(&mut self, uri: &str, bytes: &[u8]) -> Result<()> {
        let image = decode_image(uri, bytes).map_err(invalid_texture)?;
        self.image(uri, image);
        Ok(())
    }

    pub fn compressed_image(
        &mut self,
        uri: &str,
        format: TextureFormat,
        width: u32,
        height: u32,
        levels: Vec<Vec<u8>>,
    ) {
        self.files
            .add_compressed_image(uri, format, width, height, levels);
        self.held(uri);
    }

    pub fn finish(self) -> Result<SceneFiles> {
        if let Some((uri, _, _)) = self
            .entries
            .iter()
            .find(|(_, _, stage)| *stage != Stage::Held)
        {
            return Err(Error::new(
                ErrorCode::MissingFile,
                "$",
                format!("the scene's file {uri} did not load"),
            ));
        }
        Ok(self.files)
    }

    fn held(&mut self, uri: &str) {
        if let Some(entry) = self.entries.iter_mut().find(|(file, _, _)| file == uri) {
            entry.2 = Stage::Held;
        }
    }
}

fn invalid_texture(message: String) -> Error {
    Error::new(ErrorCode::InvalidTexture, "$", message)
}
