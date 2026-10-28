use std::fs::File;
use std::io::Read;
use std::path::Path;

pub use engine::host::{Image, decode_image, image_format};
use engine::host::SceneLoad;
use engine::scene::SceneFiles;
use engine::{Error, ErrorCode};

fn unreadable(path: &Path, error: impl std::fmt::Display) -> Error {
    Error::new(
        ErrorCode::MissingFile,
        "$",
        format!("cannot read {}: {error}", path.display()),
    )
}

pub fn scene_files(path: &Path, source: &str) -> engine::Result<SceneFiles> {
    let mut load = SceneLoad::new(source)?;
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
            if let Some(image) = load.read(&read.uri, bytes, &url)? {
                load.decode(&read.uri, &image)?;
            }
        }
    }
    load.finish()
}

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
