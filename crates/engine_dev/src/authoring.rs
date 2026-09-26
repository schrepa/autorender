//! Native source authoring shared by command-line clients. Saves preserve the
//! validated candidate bytes and replace the source only after revision checks.

use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use engine_scene::SceneDocument;
use serde::Serialize;

use crate::DevError;

pub const MAX_SOURCE_BYTES: usize = 1024 * 1024;
static NEXT_SAVE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Serialize)]
pub struct ValidationReport {
    pub id: String,
    /// Decimal u64, preserving precision when tools parse the result in JS.
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

/// Parses and validates using the same functions as runtime scene loading.
pub fn validate_scene(source: &str) -> Result<ValidationReport, DevError> {
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

/// Reads at most 1 MiB plus one overflow byte. This guards allocation even if a
/// source changes size between its metadata check and the actual read.
pub fn read_scene_source(path: &Path) -> Result<String, DevError> {
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    if file
        .metadata()
        .map_err(|error| io_error(path, error))?
        .len()
        > MAX_SOURCE_BYTES as u64
    {
        return Err(DevError::new(
            "source_too_large",
            path.display().to_string(),
            "scene source exceeds 1048576 bytes",
        ));
    }
    let mut bytes = Vec::new();
    file.take((MAX_SOURCE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| io_error(path, error))?;
    if bytes.len() > MAX_SOURCE_BYTES {
        return Err(DevError::new(
            "source_too_large",
            path.display().to_string(),
            "scene source exceeds 1048576 bytes",
        ));
    }
    String::from_utf8(bytes).map_err(|error| {
        DevError::new(
            "invalid_utf8",
            path.display().to_string(),
            error.to_string(),
        )
    })
}

/// Saves one existing scene with optimistic revision protection.
///
/// The source ID must match and candidate revision must be expected + 1. A
/// same-directory exclusive sidecar lock serializes cooperating writers through
/// read, validation, and rename. A synced temporary file replaces the target
/// atomically; source bytes stay unchanged on validation or conflict errors.
/// Direct editors do not participate in the lock, so stop such writers while
/// saving. A process crash can leave `.NAME.engine-dev.lock`; remove it only
/// after confirming no save is running. This function never removes stale locks.
/// Existing sources are required; project creation is deliberately separate.
pub fn save_scene(
    path: &Path,
    candidate: &str,
    expected_revision: u64,
) -> Result<SaveReport, DevError> {
    let candidate_scene = SceneDocument::from_json(candidate)?;
    let next = expected_revision.checked_add(1).ok_or_else(|| {
        DevError::new(
            "revision_exhausted",
            "$.revision",
            "scene revision cannot advance past u64::MAX",
        )
    })?;
    if candidate_scene.revision != next {
        return Err(DevError::new(
            "invalid_revision",
            "$.revision",
            format!("candidate revision must be {next}"),
        ));
    }
    let path = fs::canonicalize(path).map_err(|error| io_error(path, error))?;
    let name = path
        .file_name()
        .ok_or_else(|| {
            DevError::new(
                "invalid_path",
                path.display().to_string(),
                "source must name a file",
            )
        })?
        .to_string_lossy();
    let lock_path = path.with_file_name(format!(".{name}.engine-dev.lock"));
    let lock_file = OpenOptions::new().write(true).create_new(true).open(&lock_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::AlreadyExists {
            DevError::new("source_locked", lock_path.display().to_string(), "another save owns this source lock; after a crashed save, confirm no writer is running before removing the lock")
        } else { io_error(&lock_path, error) }
    })?;
    let _lock = TemporaryFile {
        path: lock_path,
        file: Some(lock_file),
    };
    let original = read_scene_source(&path)?;
    let current = SceneDocument::from_json(&original)?;
    if current.revision != expected_revision {
        return Err(DevError::new(
            "revision_conflict",
            "$.revision",
            format!(
                "expected revision {expected_revision}, saved source is {}",
                current.revision
            ),
        ));
    }
    if candidate_scene.id != current.id {
        return Err(DevError::new(
            "scene_id_mismatch",
            "$.id",
            "candidate must retain the saved scene ID",
        ));
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
        .map_err(|error| io_error(&temp_path, error))?;
    let mut temporary = TemporaryFile {
        path: temp_path,
        file: Some(file),
    };
    let output = temporary.file.as_mut().expect("temporary file is open");
    output
        .write_all(candidate.as_bytes())
        .map_err(|error| io_error(&temporary.path, error))?;
    output
        .sync_all()
        .map_err(|error| io_error(&temporary.path, error))?;
    // Close before rename so Windows cannot retain an incompatible file handle.
    temporary.file.take();
    // Catch noncooperating changes made during validation without presenting
    // this best-effort check as a filesystem compare-and-swap guarantee.
    if read_scene_source(&path)? != original {
        return Err(DevError::new(
            "source_changed",
            path.display().to_string(),
            "source changed during save; re-read it before retrying",
        ));
    }
    fs::rename(&temporary.path, &path).map_err(|error| io_error(&path, error))?;
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

fn io_error(path: &Path, error: std::io::Error) -> DevError {
    DevError::new("source_io", path.display().to_string(), error.to_string())
}
