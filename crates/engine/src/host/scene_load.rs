use engine_core::{ErrorCode, VirtualMesh};
use engine_scene::{SceneDocument, SceneFileKind, SceneFiles};
pub use engine_worker::{IMAGE_FORMATS, Image, ImageFormat, decode_image, image_format};

use crate::{Error, Result};

const VIRTUAL_HEADER: usize = 128;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileRead {
    pub uri: String,
    pub prefix: Option<usize>,
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
    files: SceneFiles,
    entries: Vec<(String, SceneFileKind, Stage)>,
    reads: Vec<FileRead>,
}

impl SceneLoad {
    pub fn new(source: &str) -> Result<Self> {
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
            files: SceneFiles::default(),
            entries,
            reads,
        })
    }

    pub fn take_reads(&mut self) -> Vec<FileRead> {
        std::mem::take(&mut self.reads)
    }

    pub fn read(&mut self, uri: &str, bytes: Vec<u8>, url: &str) -> Result<Option<Vec<u8>>> {
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
                engine_worker::image_format(&bytes).ok_or_else(|| {
                    invalid_texture(format!("{uri} is not a PNG, JPEG or KTX2 image"))
                })?;
                self.entries[at].2 = Stage::Decoding;
                Ok(Some(bytes))
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
