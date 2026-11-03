use std::fs;
use std::path::{Component, Path, PathBuf};

use serde_json::json;
use zip::CompressionMethod;
use zip::write::SimpleFileOptions;

use crate::Failure;
use crate::checkouts::copy_folder;
use crate::cook;
use crate::process::{Project, project_option, run, slashed};

const USAGE: &str = "usage: cargo xtask ship --project DIR";
const SHIPPED: &str = "target/ship";
const ASSETS: &str = "assets";
const COOKED: &str = "cooked";
const LOCAL_ONLY: &str = "web/public";

pub fn ship(root: &Path, options: &[String]) -> Result<(), Failure> {
    let (folder, rest) = project_option(options)?;
    let (Some(folder), []) = (folder, rest.as_slice()) else {
        return Err(USAGE.into());
    };
    let project = Project::read(root, &folder)?;
    let check = ["--project".to_owned(), folder.clone(), "--check".to_owned()];
    cook::cook(root, &check)?;
    let packs = cook::project_packs(root, &folder, &project)?;
    run(
        root,
        "cargo",
        &[
            "build",
            "-p",
            &project.package,
            "--bin",
            &project.native,
            "--profile",
            "release",
            "--locked",
        ],
    )?;
    let shipped = root.join(SHIPPED).join(&project.package);
    if shipped.exists() {
        fs::remove_dir_all(&shipped)
            .map_err(|error| format!("cannot clear {}: {error}", shipped.display()))?;
    }
    let assets = shipped.join(ASSETS);
    fs::create_dir_all(assets.join(COOKED))
        .map_err(|error| format!("cannot create {}: {error}", assets.display()))?;
    let program = project.program();
    copied(
        &root.join("target/release").join(&program),
        &shipped.join(&program),
    )?;
    for pack in &packs {
        let name = pack.file_name().unwrap_or_default();
        copied(pack, &assets.join(COOKED).join(name))?;
    }
    let own = root.join(&folder).join(ASSETS);
    if own.is_dir() {
        copy_folder(&own, &assets)?;
    }
    scenes(root, &folder, &project, &assets)?;
    let local = format!("{LOCAL_ONLY}/{}", project.package);
    if root.join(&local).is_dir() {
        println!("{}", json!({"event": "ship_left_out", "folder": local}));
    }
    let archive = root.join(SHIPPED).join(format!("{}.zip", project.package));
    zipped(&shipped, &archive, &program)?;
    let bytes = fs::metadata(&archive).map_or(0, |metadata| metadata.len());
    let record = json!({
        "event": "shipped",
        "package": project.package,
        "folder": slashed(&shipped),
        "archive": slashed(&archive),
        "packs": packs.len(),
        "bytes": bytes,
    });
    println!("{record}");
    Ok(())
}

fn copied(from: &Path, to: &Path) -> Result<(), String> {
    if let Some(folder) = to.parent() {
        fs::create_dir_all(folder)
            .map_err(|error| format!("cannot create {}: {error}", folder.display()))?;
    }
    fs::copy(from, to).map_err(|error| format!("cannot copy {}: {error}", from.display()))?;
    Ok(())
}

fn inside(file: &str) -> Result<(), String> {
    let leaves = Path::new(file)
        .components()
        .any(|part| part == Component::ParentDir);
    if leaves {
        return Err(json!({"code": "ship_outside_project", "file": file}).to_string());
    }
    Ok(())
}

fn scenes(root: &Path, folder: &str, project: &Project, assets: &Path) -> Result<(), String> {
    if project.scenes.is_empty() {
        return Ok(());
    }
    let kinds = crate::bricks::standard_kinds().map_err(|error| error.to_string())?;
    for scene in &project.scenes {
        inside(scene)?;
        let from = root.join(folder).join(scene);
        let text = fs::read_to_string(&from)
            .map_err(|error| format!("cannot read {}: {error}", from.display()))?;
        let document = engine_scene::SceneDocument::from_json(&text)
            .map_err(|error| json!({ "file": scene, "error": error }).to_string())?;
        let to = assets.join(scene);
        copied(&from, &to)?;
        let (Some(near), Some(beside)) = (from.parent(), to.parent()) else {
            continue;
        };
        for file in document.files(&kinds) {
            inside(&file.uri)?;
            copied(&near.join(&file.uri), &beside.join(&file.uri))?;
        }
    }
    Ok(())
}

fn files_under(folder: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(folder)
        .map_err(|error| format!("cannot read {}: {error}", folder.display()))?;
    for entry in entries {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_dir() {
            files_under(&path, files)?;
        } else {
            files.push(path);
        }
    }
    Ok(())
}

fn zipped(folder: &Path, archive: &Path, program: &str) -> Result<(), String> {
    let mut files = Vec::new();
    files_under(folder, &mut files)?;
    files.sort();
    let out = fs::File::create(archive)
        .map_err(|error| format!("cannot create {}: {error}", archive.display()))?;
    let mut writer = zip::ZipWriter::new(out);
    let options = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .large_file(true);
    for path in files {
        let relative = path
            .strip_prefix(folder)
            .map_err(|error| error.to_string())?;
        let name = slashed(relative);
        let mode = if name == program { 0o755 } else { 0o644 };
        writer
            .start_file(name, options.unix_permissions(mode))
            .map_err(|error| error.to_string())?;
        let mut file = fs::File::open(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        std::io::copy(&mut file, &mut writer)
            .map_err(|error| format!("cannot pack {}: {error}", path.display()))?;
    }
    writer.finish().map_err(|error| error.to_string())?;
    Ok(())
}
