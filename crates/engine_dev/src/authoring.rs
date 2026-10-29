use engine_settings::{Error, ErrorCode, Expected};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use engine_scene::{MAX_SOURCE_BYTES, SceneDocument};
use serde::Serialize;

static NEXT_SAVE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Serialize)]
pub struct ValidationReport {
    pub id: String,
    pub revision: String,
    pub entities: usize,
    pub prefabs: usize,
    pub expanded_entities: usize,
    pub source_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct SaveReport {
    pub id: String,
    pub revision: String,
    pub source_bytes: usize,
    pub path: PathBuf,
    pub persisted: bool,
}

pub fn validate_scene(source: &str) -> Result<ValidationReport, Error> {
    let scene = SceneDocument::from_json(source)?;
    let expanded_entities = scene.expanded_entities()?.len();
    Ok(ValidationReport {
        id: scene.id,
        revision: scene.revision.to_string(),
        entities: scene.entities.len(),
        prefabs: scene.prefabs.len(),
        expanded_entities,
        source_bytes: source.len(),
    })
}

pub fn read_scene_source(path: &Path) -> Result<String, Error> {
    let file = File::open(path).map_err(|error| io_error(path, &error))?;
    let length = file
        .metadata()
        .map_err(|error| io_error(path, &error))?
        .len();
    if length > MAX_SOURCE_BYTES as u64 {
        return Err(
            Error::new(ErrorCode::SourceTooLarge, path.display().to_string())
                .at_most(MAX_SOURCE_BYTES as u64)
                .got(length),
        );
    }
    let mut bytes = Vec::new();
    file.take((MAX_SOURCE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| io_error(path, &error))?;
    if bytes.len() > MAX_SOURCE_BYTES {
        return Err(
            Error::new(ErrorCode::SourceTooLarge, path.display().to_string())
                .at_most(MAX_SOURCE_BYTES as u64),
        );
    }
    String::from_utf8(bytes).map_err(|_| {
        Error::new(ErrorCode::InvalidUtf8, path.display().to_string()).expected(Expected::Utf8)
    })
}

pub fn save_scene(
    path: &Path,
    candidate: &str,
    expected_revision: u64,
) -> Result<SaveReport, Error> {
    let candidate_scene = SceneDocument::from_json(candidate)?;
    let next = expected_revision.checked_add(1).ok_or_else(|| {
        Error::new(ErrorCode::RevisionExhausted, "$.revision").got(expected_revision.to_string())
    })?;
    if candidate_scene.revision != next {
        return Err(Error::new(ErrorCode::InvalidRevision, "$.revision")
            .got(candidate_scene.revision.to_string())
            .nearest(next.to_string()));
    }
    let path = fs::canonicalize(path).map_err(|error| io_error(path, &error))?;
    let name = path
        .file_name()
        .ok_or_else(|| {
            Error::new(ErrorCode::InvalidPath, path.display().to_string()).expected(Expected::File)
        })?
        .to_string_lossy();
    let lock_path = path.with_file_name(format!(".{name}.engine-dev.lock"));
    let lock_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                Error::new(ErrorCode::SourceLocked, lock_path.display().to_string())
                    .expected(Expected::Absent)
            } else {
                io_error(&lock_path, &error)
            }
        })?;
    let _lock = TemporaryFile {
        path: lock_path,
        file: Some(lock_file),
    };
    let original = read_scene_source(&path)?;
    let current = SceneDocument::from_json(&original)?;
    if current.revision != expected_revision {
        return Err(Error::new(ErrorCode::RevisionConflict, "$.revision")
            .got(expected_revision.to_string())
            .nearest(current.revision.to_string()));
    }
    if candidate_scene.id != current.id {
        return Err(Error::new(ErrorCode::SceneIdMismatch, "$.id")
            .got(&candidate_scene.id)
            .nearest(&current.id));
    }
    let sequence = NEXT_SAVE.fetch_add(1, Ordering::Relaxed);
    let temp_path = path.with_file_name(format!(
        ".{name}.engine-dev.{}.{sequence}.tmp",
        std::process::id()
    ));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)
        .map_err(|error| io_error(&temp_path, &error))?;
    let mut temporary = TemporaryFile {
        path: temp_path,
        file: Some(file),
    };
    let output = temporary.file.as_mut().expect("temporary_file_open");
    output
        .write_all(candidate.as_bytes())
        .map_err(|error| io_error(&temporary.path, &error))?;
    output
        .sync_all()
        .map_err(|error| io_error(&temporary.path, &error))?;
    temporary.file.take();
    if read_scene_source(&path)? != original {
        return Err(Error::new(
            ErrorCode::SourceChanged,
            path.display().to_string(),
        ));
    }
    fs::rename(&temporary.path, &path).map_err(|error| io_error(&path, &error))?;
    Ok(SaveReport {
        id: candidate_scene.id,
        revision: next.to_string(),
        source_bytes: candidate.len(),
        path,
        persisted: true,
    })
}

struct TemporaryFile {
    path: PathBuf,
    file: Option<File>,
}

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        self.file.take();
        let _ = fs::remove_file(&self.path);
    }
}

fn io_error(path: &Path, error: &std::io::Error) -> Error {
    Error::new(ErrorCode::SourceIo, path.display().to_string()).got(format!("{:?}", error.kind()))
}
