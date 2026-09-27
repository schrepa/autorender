use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use engine_scene::SceneDocument;

const MAX_SOURCE_BYTES: u64 = 1024 * 1024;
const SCHEMA_PATH: &str = "docs/schema/scene-v7.schema.json";

pub fn validate_sources(root: &Path) -> Result<(), String> {
    let paths = scene_paths(root)?;
    if paths.is_empty() {
        return Err("no scene sources found in web/public/scenes/*.scene.json".to_owned());
    }
    for path in &paths {
        let display = path.strip_prefix(root).unwrap_or(path).display();
        let source = read_source(path).map_err(|error| format!("{display}: {error}"))?;
        let document =
            SceneDocument::from_json(&source).map_err(|error| format!("{display}: {error}"))?;
        println!(
            "Validated {display}: scene={} revision={} authored_entities={} prefabs={} expanded_entities={}",
            document.id,
            document.revision,
            document.entities.len(),
            document.prefabs.len(),
            document
                .expanded_entities()
                .map_err(|error| format!("{display}: {error}"))?
                .len()
        );
    }
    println!("Validated {} scene source(s).", paths.len());
    Ok(())
}

fn scene_paths(root: &Path) -> Result<Vec<PathBuf>, String> {
    let directory = root.join("web/public/scenes");
    let mut paths = Vec::new();
    let entries = fs::read_dir(&directory)
        .map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("cannot read scene directory entry: {error}"))?;
        let path = entry.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".scene.json"))
            && path.is_file()
        {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn read_source(path: &Path) -> Result<String, String> {
    let file =
        File::open(path).map_err(|error| format!("source_io at $: cannot open scene: {error}"))?;
    let length = file
        .metadata()
        .map_err(|error| format!("source_io at $: cannot read scene metadata: {error}"))?
        .len();
    if length > MAX_SOURCE_BYTES {
        return Err(format!(
            "source_too_large at $: scene source must be at most {MAX_SOURCE_BYTES} bytes"
        ));
    }
    // A bounded reader also protects against a file growing after metadata.
    let mut bytes = Vec::with_capacity(length as usize);
    file.take(MAX_SOURCE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("source_io at $: cannot read scene: {error}"))?;
    if bytes.len() as u64 > MAX_SOURCE_BYTES {
        return Err(format!(
            "source_too_large at $: scene source must be at most {MAX_SOURCE_BYTES} bytes"
        ));
    }
    String::from_utf8(bytes)
        .map_err(|error| format!("invalid_utf8 at $: scene source must be UTF-8: {error}"))
}

pub fn write_schema(root: &Path) -> Result<(), String> {
    let path = root.join(SCHEMA_PATH);
    fs::create_dir_all(root.join("docs/schema"))
        .map_err(|error| format!("cannot create schema directory: {error}"))?;
    fs::write(&path, SceneDocument::schema_json())
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    println!("Generated {SCHEMA_PATH} from engine_scene authoring types.");
    Ok(())
}

pub fn check_schema(root: &Path) -> Result<(), String> {
    let saved = fs::read_to_string(root.join(SCHEMA_PATH))
        .map_err(|error| format!("cannot read {SCHEMA_PATH}: {error}; run `cargo xtask schema`"))?;
    if saved.replace("\r\n", "\n") != SceneDocument::schema_json() {
        return Err(format!(
            "{SCHEMA_PATH} differs from engine_scene; run `cargo xtask schema` and review the change"
        ));
    }
    println!("Generated scene schema matches {SCHEMA_PATH}.");
    Ok(())
}
