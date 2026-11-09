use engine_core::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use naga::valid::{Capabilities, ModuleInfo, ValidationFlags, Validator};
use serde_json::{Value, json};

const RGA_VERSION: &str = "2.14.2";
#[cfg(windows)]
const RGA_ARCHIVE: Option<(&str, &str)> = Some((
    "https://github.com/GPUOpen-Tools/radeon_gpu_analyzer/releases/download/2.14.2/rga-windows-x64-2.14.2.zip",
    "5b46941a72722ddb27c9427ca4413bccc6892b4185ec1d5b9120740687fa166d",
));
#[cfg(target_os = "linux")]
const RGA_ARCHIVE: Option<(&str, &str)> = Some((
    "https://github.com/GPUOpen-Tools/radeon_gpu_analyzer/releases/download/2.14.2/rga-2.14.2.tgz",
    "341e7ca4f531e467be6de80a28342582230c54a42667a4eddda1b78489c35d62",
));
#[cfg(not(any(windows, target_os = "linux")))]
const RGA_ARCHIVE: Option<(&str, &str)> = None;

const TARGETS: [&str; 3] = ["gfx1100", "gfx1150", "gfx1201"];

static RGA_RUNS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct Checked {
    name: String,
    module: naga::Module,
    info: ModuleInfo,
}

fn validated(text: &str) -> Result<(naga::Module, ModuleInfo), String> {
    let module = naga::front::wgsl::parse_str(text).map_err(|error| error.emit_to_string(text))?;
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator
        .validate(&module)
        .map_err(|error| error.emit_to_string(text))?;
    Ok((module, info))
}

fn modules() -> Result<Vec<Checked>, String> {
    let mut engine = crate::bricks::standard_engine().map_err(|error| error.to_string())?;
    let installed = engine_checks::install::noop(&mut engine)?;
    let mut checked = Vec::new();
    let mut failures = Vec::new();
    for (path, text) in installed.modules.into_iter().zip(installed.texts) {
        match validated(&text) {
            Ok((module, info)) => checked.push(Checked {
                name: path,
                module,
                info,
            }),
            Err(error) => failures.push(format!("{path}: {error}")),
        }
    }
    if failures.is_empty() {
        Ok(checked)
    } else {
        Err(failures.join("\n"))
    }
}

pub fn run(root: &Path, options: &[String]) -> Result<(), String> {
    let (folder, rga) = match options {
        [flag, folder] if flag == "--report" => (PathBuf::from(folder), true),
        [flag, folder, no] if flag == "--report" && no == "--no-rga" => {
            (PathBuf::from(folder), false)
        }
        _ => return Err("usage: cargo xtask gpu --report DIR [--no-rga]".into()),
    };
    let modules = modules()?;
    let rga = if rga { rga_program(root)? } else { None };
    std::fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
    let work: Vec<(&Checked, &naga::EntryPoint)> = modules
        .iter()
        .flat_map(|checked| {
            checked
                .module
                .entry_points
                .iter()
                .map(move |entry| (checked, entry))
        })
        .collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done: Vec<std::sync::Mutex<Option<Result<Value, String>>>> =
        work.iter().map(|_| std::sync::Mutex::new(None)).collect();
    let workers = std::thread::available_parallelism().map_or(4, usize::from);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(checked, entry)) = work.get(index) else {
                        break;
                    };
                    let report = entry_report(checked, entry, &folder, rga.as_deref());
                    *done[index].lock().expect("worker_not_panicked") = Some(report);
                }
            });
        }
    });
    let entries = done
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .expect("worker_not_panicked")
                .expect("every_entry_ran")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let report = json!({
        "version": 1,
        "naga": "counts over each entry point's function and every function it calls",
        "amd": rga.as_ref().map(|program| json!({
            "tool": format!("Radeon GPU Analyzer {RGA_VERSION}, offline SPIR-V mode (no driver)"),
            "program": program.display().to_string(),
            "targets": TARGETS,
            "caution": "the offline compiler approximates the driver's; registers and instruction counts compare shaders and changes, not exact live costs",
        })),
        "amd_unavailable": rga.is_none().then_some(
            if RGA_ARCHIVE.is_none() { "Radeon GPU Analyzer runs on Windows and Linux" } else { "skipped with --no-rga" }
        ),
        "entries": entries,
    });
    let path = folder.join("report.json");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&report).expect("report_serializes"),
    )
    .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    print_summary(&report);
    println!(
        "{}",
        json!({"event": "written", "file": path.display().to_string()})
    );
    Ok(())
}

fn entry_report(
    checked: &Checked,
    entry: &naga::EntryPoint,
    folder: &Path,
    rga: Option<&Path>,
) -> Result<Value, String> {
    let module = &checked.module;
    let mut counts = Counts::default();
    let mut seen = HashSet::default();
    counts.function(module, &entry.function, &mut seen);
    let options = naga::back::spv::Options {
        lang_version: (1, 3),
        ..naga::back::spv::Options::default()
    };
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: entry.stage,
        entry_point: entry.name.clone(),
    };
    let words = naga::back::pipeline_constants::process_overrides(
        module,
        &checked.info,
        Some((entry.stage, &entry.name)),
        &naga::back::PipelineConstants::default(),
    )
    .map_err(|error| error.to_string())
    .and_then(|(module, info)| {
        naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline))
            .map_err(|error| error.to_string())
    });
    let words = match words {
        Ok(words) => words,
        Err(error) => {
            return Ok(json!({
                "module": checked.name,
                "entry_point": entry.name,
                "naga": counts.to_json(),
                "spirv_error": error,
            }));
        }
    };
    let stem = format!("{}.{}", checked.name.replace("::", "."), entry.name);
    let spirv = folder.join("spirv").join(format!("{stem}.spv"));
    std::fs::create_dir_all(spirv.parent().expect("file_has_folder"))
        .map_err(|error| error.to_string())?;
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    std::fs::write(&spirv, &bytes).map_err(|error| error.to_string())?;
    let stage = match entry.stage {
        naga::ShaderStage::Vertex => "vertex",
        naga::ShaderStage::Fragment => "fragment",
        naga::ShaderStage::Compute => "compute",
        _ => "other",
    };
    let amd = rga.map(|program| amd_report(program, &spirv, stage, folder, &stem));
    Ok(json!({
        "module": checked.name,
        "entry_point": entry.name,
        "stage": stage,
        "workgroup_size": (entry.stage == naga::ShaderStage::Compute).then_some(entry.workgroup_size),
        "naga": counts.to_json(),
        "spirv_words": words.len(),
        "spirv": spirv.display().to_string(),
        "amd": amd,
    }))
}

#[derive(Default)]
struct Counts {
    functions: u32,
    expressions: u64,
    samples: u32,
    loads: u32,
    derivatives: u32,
    loops: u32,
    calls: u32,
    barriers: u32,
    atomics: u32,
    branches: u32,
}

impl Counts {
    fn function(
        &mut self,
        module: &naga::Module,
        function: &naga::Function,
        seen: &mut HashSet<naga::Handle<naga::Function>>,
    ) {
        self.functions += 1;
        self.expressions += function.expressions.len() as u64;
        for (_, expression) in function.expressions.iter() {
            match expression {
                naga::Expression::ImageSample { .. } => self.samples += 1,
                naga::Expression::ImageLoad { .. } => self.loads += 1,
                naga::Expression::Derivative { .. } => self.derivatives += 1,
                _ => {}
            }
        }
        self.block(module, &function.body, seen);
    }

    fn block(
        &mut self,
        module: &naga::Module,
        block: &naga::Block,
        seen: &mut HashSet<naga::Handle<naga::Function>>,
    ) {
        for statement in block.iter() {
            match statement {
                naga::Statement::Block(inner) => self.block(module, inner, seen),
                naga::Statement::If { accept, reject, .. } => {
                    self.branches += 1;
                    self.block(module, accept, seen);
                    self.block(module, reject, seen);
                }
                naga::Statement::Switch { cases, .. } => {
                    self.branches += 1;
                    for case in cases {
                        self.block(module, &case.body, seen);
                    }
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    self.loops += 1;
                    self.block(module, body, seen);
                    self.block(module, continuing, seen);
                }
                naga::Statement::Call { function, .. } => {
                    self.calls += 1;
                    if seen.insert(*function) {
                        self.function(module, &module.functions[*function], seen);
                    }
                }
                naga::Statement::ControlBarrier(_) | naga::Statement::MemoryBarrier(_) => {
                    self.barriers += 1;
                }
                naga::Statement::Atomic { .. } | naga::Statement::ImageAtomic { .. } => {
                    self.atomics += 1;
                }
                _ => {}
            }
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "functions": self.functions,
            "expressions": self.expressions,
            "texture_samples": self.samples,
            "texture_loads": self.loads,
            "derivatives": self.derivatives,
            "loops": self.loops,
            "calls": self.calls,
            "branches": self.branches,
            "barriers": self.barriers,
            "atomics": self.atomics,
        })
    }
}

fn amd_report(program: &Path, spirv: &Path, stage: &str, folder: &Path, stem: &str) -> Value {
    let flag = match stage {
        "vertex" => "--vert",
        "fragment" => "--frag",
        _ => "--comp",
    };
    let out = folder.join("amd");
    let _ = std::fs::create_dir_all(&out);
    let mut targets = serde_json::Map::new();
    for target in TARGETS {
        let isa = out.join(format!("{stem}.{target}.isa"));
        let analysis = out.join(format!("{stem}.{target}.csv"));
        let temp = std::env::temp_dir().join(format!(
            "rga-{}-{}",
            std::process::id(),
            RGA_RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = std::fs::create_dir_all(&temp);
        let output = Command::new(program)
            .args(["-s", "vk-spv-offline", "-c", target, "--isa"])
            .arg(&isa)
            .arg("-a")
            .arg(&analysis)
            .arg(flag)
            .arg(spirv)
            .env("TEMP", &temp)
            .env("TMP", &temp)
            .env("TMPDIR", &temp)
            .output();
        let _ = std::fs::remove_dir_all(&temp);
        let result = match output {
            Ok(output) => {
                let isa_file = produced(&out, &isa, target);
                let analysis_file = produced(&out, &analysis, target);
                match (isa_file, analysis_file) {
                    (Some(isa), Some(analysis)) => resources(&isa, &analysis),
                    _ => {
                        let said = format!(
                            "{}{}",
                            String::from_utf8_lossy(&output.stdout),
                            String::from_utf8_lossy(&output.stderr)
                        );
                        let lines: Vec<&str> = said
                            .lines()
                            .filter(|line| !line.trim().is_empty())
                            .collect();
                        json!({"error": lines[lines.len().saturating_sub(4)..].join(" ")})
                    }
                }
            }
            Err(error) => json!({"error": error.to_string()}),
        };
        targets.insert(target.to_owned(), result);
    }
    Value::Object(targets)
}

fn produced(folder: &Path, requested: &Path, target: &str) -> Option<PathBuf> {
    if requested.is_file() {
        return Some(requested.to_owned());
    }
    let stem = requested.file_stem()?.to_string_lossy().into_owned();
    let extension = requested.extension()?.to_string_lossy().into_owned();
    std::fs::read_dir(folder)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| {
            let name = path
                .file_name()
                .map_or(String::new(), |name| name.to_string_lossy().into_owned());
            name.starts_with(target)
                && name.contains(&stem)
                && path
                    .extension()
                    .is_some_and(|found| found.to_string_lossy() == extension)
        })
}

fn resources(isa: &Path, analysis: &Path) -> Value {
    let instructions = std::fs::read_to_string(isa).map_or(0, |text| {
        text.lines()
            .filter(|line| {
                let line = line.trim_start();
                line.starts_with(|first: char| first.is_ascii_lowercase()) && !line.ends_with(':')
            })
            .count()
    });
    let table = std::fs::read_to_string(analysis).unwrap_or_default();
    let mut lines = table.lines();
    let header: Vec<&str> = lines
        .next()
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .collect();
    let values: Vec<&str> = lines
        .next()
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .collect();
    let mut fields = serde_json::Map::new();
    for (key, value) in header.iter().zip(&values) {
        let key = key.to_ascii_lowercase();
        let value = value
            .parse::<f64>()
            .map_or_else(|_| json!(value), |number| json!(number));
        fields.insert(key, value);
    }
    json!({
        "isa": isa.display().to_string(),
        "instructions": instructions,
        "resources": fields,
    })
}

fn rga_program(root: &Path) -> Result<Option<PathBuf>, String> {
    let executable = if cfg!(windows) { "rga.exe" } else { "rga" };
    if let Some(found) = std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|folder| folder.join(executable))
            .find(|file| file.is_file())
    }) {
        return Ok(Some(found));
    }
    let Some((url, sha256)) = RGA_ARCHIVE else {
        return Ok(None);
    };
    let folder = root.join("target/tools").join(format!("rga-{RGA_VERSION}"));
    if let Some(found) = find(&folder, executable) {
        return Ok(Some(found));
    }
    std::fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
    let archive = folder.join(url.rsplit('/').next().expect("url_has_name"));
    let record = json!({
        "event": "downloading",
        "tool": "radeon_gpu_analyzer",
        "version": RGA_VERSION,
        "url": url,
    });
    println!("{record}");
    let status = Command::new("curl")
        .args(["-sSL", "--fail", "-o"])
        .arg(&archive)
        .arg(url)
        .status()
        .map_err(|error| format!("cannot run curl: {error}"))?;
    if !status.success() {
        return Err(format!("downloading {url} failed"));
    }
    let bytes = std::fs::read(&archive).map_err(|error| error.to_string())?;
    let digest: String = {
        use sha2::Digest;
        sha2::Sha256::digest(&bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    };
    if digest != sha256 {
        let _ = std::fs::remove_file(&archive);
        return Err(format!(
            "{url} has SHA-256 {digest}, not the pinned {sha256}"
        ));
    }
    let tar = if cfg!(windows) {
        std::env::var_os("SystemRoot").map_or_else(
            || PathBuf::from("tar"),
            |system| PathBuf::from(system).join("System32/tar.exe"),
        )
    } else {
        PathBuf::from("tar")
    };
    let status = Command::new(tar)
        .arg("-xf")
        .arg(&archive)
        .arg("-C")
        .arg(&folder)
        .status()
        .map_err(|error| format!("cannot run tar: {error}"))?;
    if !status.success() {
        return Err(format!("unpacking {} failed", archive.display()));
    }
    let _ = std::fs::remove_file(&archive);
    find(&folder, executable)
        .map(Some)
        .ok_or_else(|| format!("{} holds no {executable}", folder.display()))
}

fn find(folder: &Path, name: &str) -> Option<PathBuf> {
    let direct = folder.join(name);
    if direct.is_file() {
        return Some(direct);
    }
    for entry in std::fs::read_dir(folder).ok()?.flatten() {
        let path = entry.path();
        if path.is_dir() {
            for candidate in [path.join(name), path.join("bin").join(name)] {
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

fn print_summary(report: &Value) {
    let Some(entries) = report["entries"].as_array() else {
        return;
    };
    let vgprs = |entry: &Value| {
        entry["amd"]["gfx1100"]["resources"]["used_vgprs"]
            .as_f64()
            .unwrap_or(-1.0)
    };
    let mut ranked: Vec<&Value> = entries.iter().collect();
    ranked.sort_by(|a, b| {
        vgprs(b).total_cmp(&vgprs(a)).then(
            b["naga"]["expressions"]
                .as_u64()
                .cmp(&a["naga"]["expressions"].as_u64()),
        )
    });
    println!(
        "{}",
        json!({"event": "entry_points", "entries": entries.len()})
    );
    for entry in ranked.iter().take(15) {
        let amd = &entry["amd"]["gfx1100"];
        let record = json!({
            "event": "costly_entry_point",
            "module": entry["module"],
            "entry_point": entry["entry_point"],
            "rdna3_vgprs": amd["resources"]["used_vgprs"],
            "rdna3_instructions": amd["instructions"],
            "naga_expressions": entry["naga"]["expressions"],
            "texture_samples": entry["naga"]["texture_samples"],
        });
        println!("{record}");
    }
}
