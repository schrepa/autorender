use engine_core::ErrorCode;
use engine_scene::{SceneDocument, SceneFileKind, SceneFiles, SceneKinds};
use engine_settings::Expected;

use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileRead {
    pub uri: String,
}

#[derive(Debug)]
pub struct SceneLoad {
    document: SceneDocument,
    files: SceneFiles,
    pending: Vec<(String, SceneFileKind)>,
    reads: Vec<FileRead>,
}

impl SceneLoad {
    pub fn new(source: &str, kinds: &SceneKinds) -> Result<Self> {
        let document = SceneDocument::from_json(source)?;
        document.validate(kinds)?;
        let pending: Vec<(String, SceneFileKind)> = document
            .files(kinds)
            .into_iter()
            .map(|file| (file.uri, file.kind))
            .collect();
        let reads = pending
            .iter()
            .map(|(uri, _)| FileRead { uri: uri.clone() })
            .collect();
        Ok(Self {
            document,
            files: SceneFiles::default(),
            pending,
            reads,
        })
    }

    pub fn take_reads(&mut self) -> Vec<FileRead> {
        std::mem::take(&mut self.reads)
    }

    pub fn read(&mut self, uri: &str, bytes: Vec<u8>) -> Result<()> {
        let at = self
            .pending
            .iter()
            .position(|(file, _)| file == uri)
            .ok_or_else(|| {
                Error::new(ErrorCode::MissingFile, "$")
                    .got(uri)
                    .variants(self.pending.iter().map(|(file, _)| file.clone()))
            })?;
        match self.pending[at].1 {
            SceneFileKind::Buffer => self.files.add_buffer(uri, bytes),
            SceneFileKind::Image => self.files.add_texture_file(uri, bytes)?,
        }
        self.pending.swap_remove(at);
        Ok(())
    }

    pub fn opened(self) -> Result<(SceneDocument, SceneFiles)> {
        if let Some((uri, _)) = self.pending.first() {
            return Err(Error::new(ErrorCode::MissingFile, "$")
                .expected(Expected::Present)
                .got(uri));
        }
        Ok((self.document, self.files))
    }
}
