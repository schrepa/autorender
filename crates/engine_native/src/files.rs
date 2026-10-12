//! Files from disk as the page fetches them: a scene's images, buffers,
//! virtual meshes and terrain files as every host loads them, and the
//! prefixes of streamed files, whose other bytes renderers read by range
//! from the same path.

use std::fs::File;
use std::io::Read;
use std::path::Path;

pub use engine::host::{Image, decode_image, image_format};
use engine::host::{LoadingHost, SceneLoad};
use engine::scene::SceneFiles;
use engine::{Error, ErrorCode};

/// The failure of reading `path` for `what`.
fn unreadable(path: &Path, error: impl std::fmt::Display) -> Error {
    Error::new(
        ErrorCode::MissingFile,
        "$",
        format!("cannot read {}: {error}", path.display()),
    )
}

/// The files scene `source`, read from `path`, references, relative to its
/// folder, as every host loads them (`engine::host::SceneLoad`): images
/// decoded with the engine's decoder (PNG, JPEG or uncompressed KTX2),
/// buffers whole, and the prefixes of virtual meshes and terrain files.
pub fn scene_files(path: &Path, source: &str) -> engine::Result<SceneFiles> {
    let mut load = SceneLoad::new(source, LoadingHost::Native)?;
    let folder = path.parent().unwrap_or(Path::new("."));
    loop {
        let reads = load.take_reads();
        if reads.is_empty() {
            break;
        }
        for read in reads {
            let path = folder.join(&read.uri);
            let bytes = read_start(&path, read.prefix)?;
            let url = path.to_string_lossy().into_owned();
            // Natively the engine's decoder decodes every image it reads.
            if let Some(image) = load.read(&read.uri, bytes, &url)? {
                load.decode(&read.uri, &image.bytes)?;
            }
        }
    }
    load.finish()
}

/// File `path`'s first `prefix` bytes (fewer where it is shorter), or the
/// whole file.
fn read_start(path: &Path, prefix: Option<usize>) -> engine::Result<Vec<u8>> {
    let Some(prefix) = prefix else {
        return std::fs::read(path).map_err(|error| unreadable(path, error));
    };
    let mut bytes = Vec::with_capacity(prefix);
    File::open(path)
        .and_then(|file| file.take(prefix as u64).read_to_end(&mut bytes))
        .map_err(|error| unreadable(path, error))?;
    Ok(bytes)
}

/// The prefix of the streamed file at `path` whose first `header` bytes say
/// how long it is (`prefix_bytes`).
pub fn read_prefix(
    path: &Path,
    header: usize,
    prefix_bytes: impl FnOnce(&[u8]) -> Option<usize>,
) -> engine::Result<Vec<u8>> {
    let mut file = File::open(path).map_err(|error| unreadable(path, error))?;
    let mut head = vec![0; header];
    file.read_exact(&mut head)
        .map_err(|error| unreadable(path, error))?;
    let length = prefix_bytes(&head).ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidMesh,
            "$",
            format!(
                "{} is not a streamed file of this engine version",
                path.display()
            ),
        )
    })?;
    if length <= head.len() {
        head.truncate(length);
        return Ok(head);
    }
    let mut rest = vec![0; length - head.len()];
    file.read_exact(&mut rest)
        .map_err(|error| unreadable(path, error))?;
    head.extend_from_slice(&rest);
    Ok(head)
}
