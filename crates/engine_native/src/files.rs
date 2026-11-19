use std::path::Path;

use engine::host::SceneLoad;
use engine::scene::{SceneDocument, SceneFiles, SceneKinds};

pub fn scene_files(path: &Path, source: &str, kinds: &SceneKinds) -> engine::Result<SceneFiles> {
    scene_opened(path, source, kinds).map(|(_, files)| files)
}

pub fn scene_opened(
    path: &Path,
    source: &str,
    kinds: &SceneKinds,
) -> engine::Result<(SceneDocument, SceneFiles)> {
    let mut load = SceneLoad::new(source, kinds)?;
    let folder = path.parent().unwrap_or(Path::new("."));
    for read in load.take_reads() {
        let path = folder.join(&read.uri);
        let bytes = std::fs::read(&path).map_err(|error| crate::file_error("$", &path, &error))?;
        load.read(&read.uri, bytes)?;
    }
    load.opened()
}
