//! Offline shader analysis: exact-input reuse, verified compiler bundles,
//! success-only publication and complete materialization in caller outputs.

use crate::{
    cache::{Entry, Store},
    files, jobs,
};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

pub const TARGETS: [&str; 3] = ["gfx1100", "gfx1150", "gfx1201"];
const VERSION: &str = "2.14.2";
#[cfg(windows)]
const ARCHIVE: Option<(&str, &str)> = Some((
    "https://github.com/GPUOpen-Tools/radeon_gpu_analyzer/releases/download/2.14.2/rga-windows-x64-2.14.2.zip",
    "5b46941a72722ddb27c9427ca4413bccc6892b4185ec1d5b9120740687fa166d",
));
#[cfg(target_os = "linux")]
const ARCHIVE: Option<(&str, &str)> = Some((
    "https://github.com/GPUOpen-Tools/radeon_gpu_analyzer/releases/download/2.14.2/rga-2.14.2.tgz",
    "341e7ca4f531e467be6de80a28342582230c54a42667a4eddda1b78489c35d62",
));
#[cfg(not(any(windows, target_os = "linux")))]
const ARCHIVE: Option<(&str, &str)> = None;

/// A verified tool lease outlives every analyzer subprocess in this report.
pub struct Analyzer {
    program: PathBuf,
    identity: Value,
    store: Store,
    reuse: bool,
    _tool: Option<Entry>,
    hits: AtomicU64,
    misses: AtomicU64,
    failed: AtomicU64,
}

impl Analyzer {
    /// A PATH override is usable but not cached without an explicit bundle
    /// root. Hashing one executable cannot identify its companion compiler.
    pub fn open(root: &Path, reuse: bool) -> Result<Option<Self>, String> {
        let store = Store::open(root)?;
        let name = format!("rga{}", std::env::consts::EXE_SUFFIX);
        let explicit = std::env::var_os("ENGINE_RGA_ROOT").map(PathBuf::from);
        let path_program = std::env::var_os("PATH").and_then(|path| {
            std::env::split_paths(&path)
                .map(|p| p.join(&name))
                .find(|p| p.is_file())
        });
        let (program, identity, tool, verified) = if let Some(bundle) = explicit {
            let bundle = bundle.canonicalize().map_err(|e| e.to_string())?;
            let manifest = files::manifest(&bundle)?;
            let program = find(&bundle, &name).ok_or("ENGINE_RGA_ROOT contains no analyzer")?;
            (
                program,
                json!({"kind": "explicit-bundle", "sha256": files::digest(manifest.to_string().as_bytes())}),
                None,
                true,
            )
        } else if let Some(program) = path_program {
            let identity = json!({"kind": "path-override", "executable_sha256": files::hash(&program)?, "cache_disabled_reason": "compiler bundle unknown; set ENGINE_RGA_ROOT to the complete installation"});
            (program, identity, None, false)
        } else {
            let Some((url, digest)) = ARCHIVE else {
                return Ok(None);
            };
            let entry = match store.read("analyzer-tool", digest)? {
                Some(entry) => entry,
                None => {
                    let entry = store.entry("analyzer-tool", digest)?;
                    if !entry.valid()? {
                        install(&entry, url, digest)?;
                    }
                    entry.shared()?
                }
            };
            let program =
                find(&entry.data(), &name).ok_or("analyzer archive contains no executable")?;
            (
                program,
                json!({"kind": "pinned-bundle", "release": VERSION, "archive_sha256": digest, "manifest_sha256": files::hash(&entry.path().join("manifest.json"))?}),
                Some(entry),
                true,
            )
        };
        Ok(Some(Self {
            program,
            identity,
            store,
            reuse: reuse && verified,
            _tool: tool,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            failed: AtomicU64::new(0),
        }))
    }

    /// Actual tool provenance; never label a PATH override as a pinned release.
    pub fn description(&self) -> Value {
        json!({"tool": "Radeon GPU Analyzer, offline SPIR-V mode (no driver)", "program": self.program, "identity": self.identity, "targets": TARGETS, "cache_enabled": self.reuse, "caution": "the offline compiler approximates the driver's; registers and instruction counts compare shaders and changes, not exact live costs"})
    }

    /// Cache telemetry is separate from observations and original tool output.
    pub fn telemetry(&self) -> Value {
        json!({"hits": self.hits.load(Ordering::Relaxed), "compiled": self.misses.load(Ordering::Relaxed), "failed": self.failed.load(Ordering::Relaxed)})
    }

    /// Preserve full entry/target coverage on cache hits and explicit errors.
    pub fn report(
        &self,
        spirv: &Path,
        entry: &str,
        stage: &str,
        folder: &Path,
        stem: &str,
    ) -> Value {
        let mut result = serde_json::Map::new();
        for target in TARGETS {
            let value = match self.target(spirv, entry, stage, target, folder, stem) {
                Ok(value) => value,
                Err(error) => {
                    self.failed.fetch_add(1, Ordering::Relaxed);
                    json!({"error": error})
                }
            };
            result.insert(target.into(), value);
        }
        Value::Object(result)
    }

    fn target(
        &self,
        spirv: &Path,
        entry: &str,
        stage: &str,
        target: &str,
        folder: &Path,
        stem: &str,
    ) -> Result<Value, String> {
        let start = Instant::now();
        let inputs = json!({"version": 4, "execution_environment": "bundle-system-c-locale-v1", "spirv": files::hash(spirv)?, "entry": entry, "stage": stage, "target": target, "mode": "vk-spv-offline", "tool": self.identity, "os": std::env::consts::OS, "arch": std::env::consts::ARCH});
        let key = files::digest(inputs.to_string().as_bytes());
        let cache = self.store.entry("shader-analysis", &key)?;
        let hit = self.reuse && cache.valid()?;
        if hit {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            let stage_dir = cache.staging()?;
            let attempt = compile(&self.program, spirv, stage, target, &stage_dir);
            if let Err(error) = attempt {
                let failed = folder
                    .join("amd/failures")
                    .join(format!("{stem}.{target}.{}", files::unique()));
                files::copy_tree(&stage_dir, &failed)?;
                return Err(format!("{error}; attempt: {}", failed.display()));
            }
            files::atomic_json(&stage_dir.join("inputs.json"), &inputs)?;
            // Disabled caching also validates and materializes fresh bytes, but
            // never replaces a reusable result with a different experiment.
            if !self.reuse {
                return materialize(&stage_dir, folder, stem, target, false, &key, start);
            }
            cache.publish()?;
        }
        materialize(&cache.data(), folder, stem, target, hit, &key, start)
    }
}

fn materialize(
    from: &Path,
    folder: &Path,
    stem: &str,
    target: &str,
    hit: bool,
    key: &str,
    start: Instant,
) -> Result<Value, String> {
    let out = folder.join("amd");
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let isa = out.join(format!("{stem}.{target}.isa"));
    let csv = out.join(format!("{stem}.{target}.csv"));
    fs::copy(from.join("output.isa"), &isa).map_err(|e| e.to_string())?;
    fs::copy(from.join("output.csv"), &csv).map_err(|e| e.to_string())?;
    let mut result = files::json_file(&from.join("summary.json"))?;
    result["isa"] = json!(isa);
    result["analysis"] = json!(csv);
    result["cache"] = json!({"hit": hit, "key": key, "elapsed_ms": start.elapsed().as_millis()});
    Ok(result)
}

fn compile(
    program: &Path,
    spirv: &Path,
    stage: &str,
    target: &str,
    output: &Path,
) -> Result<(), String> {
    let flag = match stage {
        "vertex" => "--vert",
        "fragment" => "--frag",
        "compute" => "--comp",
        _ => return Err(format!("unsupported shader stage: {stage}")),
    };
    // RGA's Windows compiler still has MAX_PATH restrictions. Keep both its
    // input and output short, then export every raw file before cleanup.
    let temp = output.join("scratch");
    if cfg!(windows) && temp.as_os_str().len() > 180 {
        return Err("analyzer cache path is too long; select a short ENGINE_CACHE_DIR".into());
    }
    fs::create_dir(&temp).map_err(|e| e.to_string())?;
    let result = (|| {
        fs::copy(spirv, temp.join("input.spv")).map_err(|e| e.to_string())?;
        let mut command = Command::new(program);
        command
            .current_dir(&temp)
            .args([
                "-s",
                "vk-spv-offline",
                "-c",
                target,
                "--isa",
                "output.isa",
                "-a",
                "output.csv",
                flag,
                "input.spv",
            ])
            .env("TEMP", &temp)
            .env("TMP", &temp)
            .env("TMPDIR", &temp);
        let mut search = vec![
            program
                .parent()
                .ok_or("analyzer has no parent directory")?
                .to_path_buf(),
        ];
        if cfg!(windows) {
            let system = std::env::var_os("SystemRoot").ok_or("SystemRoot is unavailable")?;
            search.push(PathBuf::from(system).join("System32"));
        } else {
            search.extend([PathBuf::from("/usr/bin"), PathBuf::from("/bin")]);
        }
        command.env(
            "PATH",
            std::env::join_paths(search).map_err(|e| e.to_string())?,
        );
        // Cargo injects dynamic-library paths even on Windows. Only this
        // verified installation and standard system libraries are tool inputs.
        let parent = program.parent().ok_or("analyzer has no parent")?;
        let libraries = std::env::join_paths([parent.to_path_buf(), parent.join("lib")])
            .map_err(|e| e.to_string())?;
        command
            .env("LD_LIBRARY_PATH", &libraries)
            .env("DYLD_LIBRARY_PATH", &libraries)
            .env("LANG", "C")
            .env("LC_ALL", "C");
        let run = jobs::run(&mut command, &temp.join("logs"), &jobs::Limits::default())?;
        if !run.status.success() {
            return Err(format!("analyzer exited {}", run.status));
        }
        let isa = produced(&temp, "isa", target)?;
        let csv = produced(&temp, "csv", target)?;
        let summary = resources(&isa, &csv)?;
        fs::copy(isa, output.join("output.isa")).map_err(|e| e.to_string())?;
        fs::copy(csv, output.join("output.csv")).map_err(|e| e.to_string())?;
        files::atomic_json(&output.join("summary.json"), &summary)
    })();
    let logs = files::copy_tree(&temp.join("logs"), &output.join("logs"));
    if result.is_err() {
        files::copy_tree(&temp, &output.join("failed-tool-output"))?;
    }
    let removed = fs::remove_dir_all(&temp).map_err(|e| format!("{}: {e}", temp.display()));
    result.and(logs).and(removed)
}

fn produced(folder: &Path, extension: &str, target: &str) -> Result<PathBuf, String> {
    let paths: Vec<_> = files::list(folder)?
        .into_iter()
        .filter(|p| {
            p.extension().is_some_and(|e| e == extension)
                && p.file_stem().is_some_and(|s| {
                    let name = s.to_string_lossy();
                    name == "output" || (name.starts_with(target) && name.contains("output"))
                })
        })
        .collect();
    match paths.as_slice() {
        [path] => Ok(folder.join(path)),
        _ => Err(format!(
            "analyzer produced {} matching {extension} files, expected one",
            paths.len()
        )),
    }
}

/// Missing/malformed data is an error, never an invented zero resource count.
pub fn resources(isa: &Path, analysis: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(isa).map_err(|e| e.to_string())?;
    let instructions = text
        .lines()
        .filter(|line| {
            let line = line.trim_start();
            line.starts_with(|c: char| c.is_ascii_lowercase()) && !line.ends_with(':')
        })
        .count();
    if instructions == 0 {
        return Err("empty or unrecognized ISA output".into());
    }
    let table = fs::read_to_string(analysis).map_err(|e| e.to_string())?;
    let mut lines = table.lines();
    let header: Vec<_> = lines
        .next()
        .ok_or("missing CSV header")?
        .split(',')
        .map(str::trim)
        .collect();
    let values: Vec<_> = lines
        .next()
        .ok_or("missing CSV resources")?
        .split(',')
        .map(str::trim)
        .collect();
    if header.len() != values.len() || header.is_empty() {
        return Err("incomplete analyzer CSV row".into());
    }
    let mut fields = serde_json::Map::new();
    for (key, value) in header.iter().zip(values) {
        if key.is_empty() || value.is_empty() {
            return Err("empty analyzer CSV cell".into());
        }
        fields.insert(
            key.to_ascii_lowercase(),
            value
                .parse::<f64>()
                .map_or_else(|_| json!(value), |v| json!(v)),
        );
    }
    for name in ["used_vgprs", "used_sgprs"] {
        if !fields.get(name).is_some_and(Value::is_number) {
            return Err(format!("missing numeric analyzer resource {name}"));
        }
    }
    Ok(json!({"instructions": instructions, "resources": fields}))
}

fn install(entry: &Entry, url: &str, expected: &str) -> Result<(), String> {
    entry.require_capacity(1024 * 1024 * 1024)?;
    let folder = entry.staging()?;
    let archive = entry.path().join("download");
    eprintln!("Downloading Radeon GPU Analyzer {VERSION}");
    let status = Command::new("curl")
        .args(["-sSL", "--fail", "--max-time", "600", "-o"])
        .arg(&archive)
        .arg(url)
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("analyzer download failed".into());
    }
    if files::hash(&archive)? != expected {
        fs::remove_file(archive).map_err(|e| e.to_string())?;
        return Err("analyzer archive checksum differs".into());
    }
    let tar = if cfg!(windows) {
        std::env::var_os("SystemRoot")
            .map(|p| PathBuf::from(p).join("System32/tar.exe"))
            .unwrap_or_else(|| PathBuf::from("tar"))
    } else {
        PathBuf::from("tar")
    };
    let status = Command::new(tar)
        .arg("-xf")
        .arg(&archive)
        .arg("-C")
        .arg(&folder)
        .status()
        .map_err(|e| e.to_string())?;
    fs::remove_file(archive).map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("analyzer extraction failed".into());
    }
    entry.publish()
}

fn find(folder: &Path, name: &str) -> Option<PathBuf> {
    files::list(folder)
        .ok()?
        .into_iter()
        .find(|p| p.file_name().is_some_and(|s| s == name))
        .map(|p| folder.join(p))
}
