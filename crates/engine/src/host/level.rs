use engine_core::LevelProgress;
#[cfg(not(target_arch = "wasm32"))]
use engine_core::{Error, ErrorCode};
#[cfg(not(target_arch = "wasm32"))]
use engine_scene::{SceneFiles, SceneKinds};

#[cfg(not(target_arch = "wasm32"))]
use super::{FileRead, SceneLoad};
use crate::Result;

const QUIET_FRAMES: u32 = 8;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub(crate) struct LevelRead {
    folder: std::path::PathBuf,
    source: String,
    load: SceneLoad,
    reads: Vec<FileRead>,
    total: u32,
}

#[cfg(not(target_arch = "wasm32"))]
impl LevelRead {
    fn open(path: &str, kinds: &SceneKinds) -> Result<Self> {
        let source = std::fs::read_to_string(path).map_err(|error| {
            Error::new(ErrorCode::SourceIo, "$.path").got(format!("{path}: {error}"))
        })?;
        let mut load = SceneLoad::new(&source, kinds)?;
        let reads = load.take_reads();
        let folder = std::path::Path::new(path)
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_owned();
        Ok(Self {
            folder,
            source,
            load,
            total: reads.len() as u32,
            reads,
        })
    }

    fn read_one(&mut self, read: &FileRead) -> Result<()> {
        let path = self.folder.join(&read.uri);
        let bytes = std::fs::read(&path).map_err(|error| {
            Error::new(ErrorCode::SourceIo, "$.path").got(format!("{}: {error}", path.display()))
        })?;
        if let Some(image) = self.load.read(&read.uri, bytes)? {
            self.load.decode(&read.uri, &image)?;
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub(crate) enum LevelLoad {
    #[default]
    Idle,
    #[cfg(not(target_arch = "wasm32"))]
    Reading(Box<LevelRead>),
    Building {
        quiet: u32,
    },
}

pub(crate) enum LevelStep {
    Progress(LevelProgress),
    #[cfg(not(target_arch = "wasm32"))]
    Ready {
        source: String,
        files: SceneFiles,
    },
}

impl LevelLoad {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn open(path: &str, kinds: &SceneKinds) -> Result<Self> {
        Ok(Self::Reading(Box::new(LevelRead::open(path, kinds)?)))
    }

    pub(crate) fn step(&mut self, busy: bool) -> Result<LevelStep> {
        match self {
            Self::Idle => Ok(LevelStep::Progress(LevelProgress::Idle)),
            #[cfg(not(target_arch = "wasm32"))]
            Self::Reading(reading) => {
                if let Some(read) = reading.reads.pop() {
                    reading.read_one(&read)?;
                    return Ok(LevelStep::Progress(LevelProgress::Reading {
                        done: reading.total - reading.reads.len() as u32,
                        total: reading.total,
                    }));
                }
                let Self::Reading(finished) = std::mem::take(self) else {
                    return Ok(LevelStep::Progress(LevelProgress::Idle));
                };
                let LevelRead { source, load, .. } = *finished;
                Ok(LevelStep::Ready {
                    source,
                    files: load.finish()?,
                })
            }
            Self::Building { quiet } => {
                *quiet = if busy { 0 } else { *quiet + 1 };
                if *quiet < QUIET_FRAMES {
                    return Ok(LevelStep::Progress(LevelProgress::Building));
                }
                *self = Self::Idle;
                Ok(LevelStep::Progress(LevelProgress::Idle))
            }
        }
    }
}
