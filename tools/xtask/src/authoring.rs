use std::{
    fs,
    path::{Path, PathBuf},
};

use engine_dev::authoring::{read_scene_source, validate_scene};
use engine_scene::{SceneDocument, schema_path};

pub fn validate_sources(root: &Path) -> Result<(), String> {
    let paths = scene_paths(root)?;
    if paths.is_empty() {
        return Err("no scene sources found in web/public/scenes/*.scene.json".to_owned());
    }
    for path in &paths {
        let display = path.strip_prefix(root).unwrap_or(path).display();
        // The same bounded reader and validation as `cargo xtask scene validate`.
        let report = read_scene_source(path)
            .and_then(|source| validate_scene(&source))
            .map_err(|error| format!("{display}: {error}"))?;
        println!(
            "Validated {display}: scene={} revision={} authored_entities={} prefabs={} expanded_entities={}",
            report.id, report.revision, report.entities, report.prefabs, report.expanded_entities
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

pub fn write_schema(root: &Path) -> Result<(), String> {
    let schema = schema_path();
    let path = root.join(&schema);
    fs::create_dir_all(root.join("docs/schema"))
        .map_err(|error| format!("cannot create schema directory: {error}"))?;
    fs::write(&path, SceneDocument::schema_json())
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    println!("Generated {schema} from engine_scene authoring types.");
    Ok(())
}

/// Where the protocol summary the web host and scripts read lives.
const PROTOCOL_PATH: &str = "web/src/generated/protocol.json";

/// Writes the development protocol's summary for the web host and scripts.
pub fn write_protocol(root: &Path) -> Result<(), String> {
    fs::create_dir_all(root.join("web/src/generated"))
        .map_err(|error| format!("cannot create the generated directory: {error}"))?;
    fs::write(root.join(PROTOCOL_PATH), engine_dev::PROTOCOL_SUMMARY)
        .map_err(|error| format!("cannot write {PROTOCOL_PATH}: {error}"))?;
    println!("Generated {PROTOCOL_PATH} from engine_dev's protocol.");
    Ok(())
}

pub fn check_protocol(root: &Path) -> Result<(), String> {
    let saved = fs::read_to_string(root.join(PROTOCOL_PATH)).map_err(|error| {
        format!("cannot read {PROTOCOL_PATH}: {error}; run `cargo xtask schema`")
    })?;
    if saved.replace("\r\n", "\n") != engine_dev::PROTOCOL_SUMMARY {
        return Err(format!(
            "{PROTOCOL_PATH} differs from engine_dev's protocol; run `cargo xtask schema`"
        ));
    }
    println!("Generated protocol summary matches {PROTOCOL_PATH}.");
    Ok(())
}

pub fn check_schema(root: &Path) -> Result<(), String> {
    let schema = schema_path();
    let saved = fs::read_to_string(root.join(&schema))
        .map_err(|error| format!("cannot read {schema}: {error}; run `cargo xtask schema`"))?;
    if saved.replace("\r\n", "\n") != SceneDocument::schema_json() {
        return Err(format!(
            "{schema} differs from engine_scene; run `cargo xtask schema` and review the change"
        ));
    }
    println!("Generated scene schema matches {schema}.");
    Ok(())
}
