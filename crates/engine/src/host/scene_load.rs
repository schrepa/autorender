use engine_core::ErrorCode;
use engine_scene::{SceneDocument, SceneFileKind, SceneFiles, SceneKinds};
use engine_settings::Expected;
pub use engine_worker::{IMAGE_FORMATS, Image, ImageFormat, decode_image, image_format};

use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileRead {
    pub uri: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Asked,
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
    pub fn new(source: &str, kinds: &SceneKinds) -> Result<Self> {
        let document = SceneDocument::from_json(source)?;
        document.validate(kinds)?;
        let entries: Vec<(String, SceneFileKind, Stage)> = document
            .files(kinds)
            .into_iter()
            .map(|file| (file.uri, file.kind, Stage::Asked))
            .collect();
        let reads = entries
            .iter()
            .map(|(uri, _, _)| FileRead { uri: uri.clone() })
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

    pub fn read(&mut self, uri: &str, bytes: Vec<u8>) -> Result<Option<Vec<u8>>> {
        let at = self
            .entries
            .iter()
            .position(|(file, _, _)| file == uri)
            .ok_or_else(|| {
                Error::new(ErrorCode::MissingFile, "$")
                    .got(uri)
                    .variants(self.entries.iter().map(|(file, _, _)| file.clone()))
            })?;
        match self.entries[at].1 {
            SceneFileKind::Buffer => {
                self.files.add_buffer(uri, bytes);
                self.entries[at].2 = Stage::Held;
                Ok(None)
            }
            SceneFileKind::Image => {
                engine_worker::image_format(&bytes).ok_or_else(|| {
                    Error::new(ErrorCode::InvalidTexture, "$")
                        .expected(Expected::Image)
                        .got(uri)
                })?;
                self.entries[at].2 = Stage::Decoding;
                Ok(Some(bytes))
            }
        }
    }

    pub fn image(&mut self, uri: &str, image: Image) {
        self.files
            .add_image(uri, image.width, image.height, image.pixels);
        self.held(uri);
    }

    pub fn decode(&mut self, uri: &str, bytes: &[u8]) -> Result<()> {
        let image = decode_image(uri, bytes)?;
        self.image(uri, image);
        Ok(())
    }

    pub fn finish(self) -> Result<SceneFiles> {
        if let Some((uri, _, _)) = self
            .entries
            .iter()
            .find(|(_, _, stage)| *stage != Stage::Held)
        {
            return Err(Error::new(ErrorCode::MissingFile, "$")
                .expected(Expected::Present)
                .got(uri));
        }
        Ok(self.files)
    }

    fn held(&mut self, uri: &str) {
        if let Some(entry) = self.entries.iter_mut().find(|(file, _, _)| file == uri) {
            entry.2 = Stage::Held;
        }
    }
}
