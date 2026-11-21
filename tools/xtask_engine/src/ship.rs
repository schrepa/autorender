use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};

use engine_checks::Package;
use flate2::Compression;
use flate2::write::GzEncoder;
use serde_json::{Value, json};
use xtask::pipeline;
use xtask::process::{DEV_FEATURES, clients, run, slashed};
use zip::CompressionMethod;
use zip::write::SimpleFileOptions;

const SHIPPED: &str = "target/ship";
const PAGES: &str = "web/dist";
const SCENE_SUFFIX: &str = ".scene.json";
const MIN_PRECOMPRESS_BYTES: u64 = 512;
const PRECOMPRESS_EXTENSIONS: &[&str] = &[
    "ktx2", "emb", "clip", "skel", "json", "bin", "svg", "html", "css", "js", "wasm",
];

pub fn ship(root: &Path, options: &[String]) -> Result<(), String> {
    if !options.is_empty() {
        return Err("usage: ship".to_owned());
    }
    crate::authoring::generate(root, &["--check".to_owned()])?;
    for package in crate::rules::repository(root)?.projects() {
        game(root, package)?;
    }
    Ok(())
}

fn declared<'a>(package: &'a Package, key: &str) -> Result<&'a str, String> {
    package.engine[key].as_str().ok_or_else(|| {
        format!("{}: [package.metadata.engine] declares no {key}", package.name)
    })
}

fn game(root: &Path, package: &Package) -> Result<(), String> {
    let name = package.name.as_str();
    let (game, native) = (declared(package, "game")?, declared(package, "native")?);
    let cook = declared(package, "cook")?;
    let cooking = ["run", "-p", name, "--bin", cook, "--features", DEV_FEATURES, "--locked", "--"];
    run(root, "cargo", &cooking)?;
    run(root, "cargo", &[&cooking[..], &["--verify"][..]].concat())?;
    run(root, "cargo", &["build", "-p", name, "--bin", native, "--release", "--locked"])?;
    let client = clients()
        .iter()
        .find(|client| client.package == name)
        .ok_or_else(|| format!("{name} builds no page"))?;
    pipeline::build_page(root, client)?;
    let shipped = root.join(SHIPPED).join(game);
    if shipped.exists() {
        fs::remove_dir_all(&shipped)
            .map_err(|error| format!("cannot clear {}: {error}", shipped.display()))?;
    }
    let (hand, mut content) = (hand(package)?, content(&package.folder)?);
    let hands = hand.len();
    content.extend(hand);
    let program = format!("{native}{}", std::env::consts::EXE_SUFFIX);
    let native = shipped.join("native");
    copied(&root.join("target/release").join(&program), &native.join(&program))?;
    let bytes = placed(&package.folder, &content, &native)?;
    archived(game, &native, (content.len(), hands), bytes, (0, 0), Some(program.as_str()))?;
    let web = shipped.join("web");
    let pages = root.join(PAGES);
    let mut built = Vec::new();
    files_under(&pages, &mut built)?;
    for page in built {
        let relative = page.strip_prefix(&pages).map_err(|error| error.to_string())?;
        copied(&page, &web.join(relative))?;
    }
    let bytes = placed(&package.folder, &content, &web.join(game))?;
    let precompressed = precompress(&web)?;
    archived(game, &web, (content.len(), hands), bytes, precompressed, None)
}

fn hand(package: &Package) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for listed in package.engine["ship"].as_array().into_iter().flatten() {
        let path = listed.as_str().map(PathBuf::from).ok_or_else(|| {
            format!("{}: [package.metadata.engine] ship lists paths", package.name)
        })?;
        inside(&path)?;
        let at = package.folder.join(&path);
        if at.is_dir() {
            let mut found = Vec::new();
            files_under(&at, &mut found)?;
            for file in found {
                let relative = file.strip_prefix(&package.folder).map_err(|e| e.to_string())?;
                files.push(relative.to_path_buf());
            }
        } else if at.is_file() {
            files.push(path);
        } else {
            return Err(json!({"code": "ship_missing", "file": slashed(&path)}).to_string());
        }
    }
    Ok(files)
}

fn content(folder: &Path) -> Result<Vec<PathBuf>, String> {
    let mut listed = Vec::new();
    for scene in scenes(folder)? {
        let cooked = read(&folder.join(&scene))?["cooked"].as_str().map(PathBuf::from);
        listed.push(scene);
        let Some(cooked) = cooked else {
            continue;
        };
        inside(&cooked)?;
        let near = cooked.parent().map(Path::to_path_buf).unwrap_or_default();
        for file in read(&folder.join(&cooked))?["files"].as_array().into_iter().flatten() {
            let uri = file["uri"]
                .as_str()
                .ok_or_else(|| format!("{}: a file without a uri", slashed(&cooked)))?;
            let at = near.join(uri);
            inside(&at)?;
            listed.push(at);
        }
        listed.push(cooked);
    }
    listed.sort();
    listed.dedup();
    Ok(listed)
}

fn scenes(folder: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = fs::read_dir(folder)
        .map_err(|error| format!("cannot read {}: {error}", folder.display()))?;
    let mut scenes = Vec::new();
    for entry in entries {
        let name = entry.map_err(|error| error.to_string())?.file_name();
        if name.to_string_lossy().ends_with(SCENE_SUFFIX) {
            scenes.push(PathBuf::from(name));
        }
    }
    scenes.sort();
    Ok(scenes)
}

fn read(path: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
}

fn inside(file: &Path) -> Result<(), String> {
    if file.components().all(|part| matches!(part, Component::Normal(_))) {
        return Ok(());
    }
    Err(json!({"code": "ship_outside_project", "file": slashed(file)}).to_string())
}

fn placed(folder: &Path, content: &[PathBuf], to: &Path) -> Result<u64, String> {
    let mut bytes = 0;
    for file in content {
        bytes += copied(&folder.join(file), &to.join(file))?;
    }
    Ok(bytes)
}

fn copied(from: &Path, to: &Path) -> Result<u64, String> {
    if let Some(folder) = to.parent() {
        fs::create_dir_all(folder)
            .map_err(|error| format!("cannot create {}: {error}", folder.display()))?;
    }
    fs::copy(from, to).map_err(|error| format!("cannot copy {}: {error}", from.display()))
}

fn archived(
    game: &str,
    folder: &Path,
    (files, hand_files): (usize, usize),
    bytes: u64,
    (precompressed_files, precompressed_bytes): (usize, u64),
    program: Option<&str>,
) -> Result<(), String> {
    let archive = folder.with_extension("zip");
    zipped(folder, &archive, program)?;
    let record = json!({
        "event": "shipped",
        "game": game,
        "folder": slashed(folder),
        "files": files,
        "hand_files": hand_files,
        "bytes": bytes,
        "precompressed_files": precompressed_files,
        "precompressed_bytes": precompressed_bytes,
        "archive": slashed(&archive),
        "archive_bytes": fs::metadata(&archive).map_or(0, |metadata| metadata.len()),
    });
    println!("{record}");
    Ok(())
}

fn should_precompress(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };
    if !PRECOMPRESS_EXTENSIONS
        .iter()
        .any(|wanted| extension.eq_ignore_ascii_case(wanted))
    {
        return false;
    }
    let sidecar = sidecar_gz(path);
    if sidecar.exists() {
        return false;
    }
    fs::metadata(path).is_ok_and(|metadata| metadata.len() >= MIN_PRECOMPRESS_BYTES)
}

fn sidecar_gz(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".gz");
    PathBuf::from(name)
}

fn compress_file(path: &Path) -> Result<Option<u64>, String> {
    let raw = fs::read(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
    encoder
        .write_all(&raw)
        .map_err(|error| format!("cannot compress {}: {error}", path.display()))?;
    let compressed = encoder
        .finish()
        .map_err(|error| format!("cannot finish {}: {error}", path.display()))?;
    if compressed.len() >= raw.len() {
        return Ok(None);
    }
    let out = sidecar_gz(path);
    fs::write(&out, &compressed)
        .map_err(|error| format!("cannot write {}: {error}", out.display()))?;
    Ok(Some(compressed.len() as u64))
}

fn precompress(folder: &Path) -> Result<(usize, u64), String> {
    let mut files = Vec::new();
    files_under(folder, &mut files)?;
    let targets: Vec<PathBuf> = files
        .into_iter()
        .filter(|path| should_precompress(path))
        .collect();
    if targets.is_empty() {
        return Ok((0, 0));
    }
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(targets.len());
    let chunk = targets.len().div_ceil(threads);
    let results: Vec<Result<(usize, u64), String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = targets
            .chunks(chunk)
            .map(|slice| {
                scope.spawn(move || -> Result<(usize, u64), String> {
                    let mut count = 0;
                    let mut bytes = 0;
                    for path in slice {
                        if let Some(written) = compress_file(path)? {
                            count += 1;
                            bytes += written;
                        }
                    }
                    Ok((count, bytes))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    });
    let mut total_count = 0;
    let mut total_bytes = 0;
    for result in results {
        let (count, bytes) = result?;
        total_count += count;
        total_bytes += bytes;
    }
    Ok((total_count, total_bytes))
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

fn zipped(folder: &Path, archive: &Path, program: Option<&str>) -> Result<(), String> {
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
        let mode = if Some(name.as_str()) == program { 0o755 } else { 0o644 };
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
