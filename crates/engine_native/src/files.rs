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
            let bytes = std::fs::read(&path).map_err(|error| unreadable(&path, error))?;
            if let Some(image) = load.read(&read.uri, bytes)? {
                load.decode(&read.uri, &image)?;
            }
        }
    }
    load.finish()
}
