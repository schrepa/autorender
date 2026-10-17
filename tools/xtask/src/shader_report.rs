//! `cargo xtask gpu --report DIR`: what every shader entry point costs,
//! offline, on any machine, no GPU needed. Everywhere: naga's counts of the
//! work each entry point does (expressions, texture samples and loads,
//! derivatives, loops, calls, barriers, atomics) and its SPIR-V. Where Radeon
//! GPU Analyzer runs (Windows and Linux; downloaded once into
//! the shared byte-bounded developer cache, with a verified bundle): the AMD machine
//! code each entry point compiles to on RDNA 3, 3.5 (integrated) and 4,
//! the architectures its offline compiler knows, with its vector and scalar
//! registers, local memory and scratch. The report is
//! `DIR/report.json`; SPIR-V and machine code sit beside it.

use engine_core::collections::HashSet;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::gpu::{Checked, modules};

/// Runs `gpu --report DIR [--no-rga]`.
pub fn run(root: &Path, options: &[String]) -> Result<(), String> {
    let use_cache = !options.iter().any(|option| option == "--no-cache");
    let options: Vec<_> = options
        .iter()
        .filter(|option| option.as_str() != "--no-cache")
        .cloned()
        .collect();
    let (folder, rga) = match options.as_slice() {
        [flag, folder] if flag == "--report" => (PathBuf::from(folder), true),
        [flag, folder, no] if flag == "--report" && no == "--no-rga" => {
            (PathBuf::from(folder), false)
        }
        _ => return Err("usage: cargo xtask gpu [--report DIR [--no-rga]]".into()),
    };
    let modules = modules()?;
    let rga = if rga {
        xtask::analyzer::Analyzer::open(root, use_cache)?
    } else {
        None
    };
    std::fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
    // Entry points compile independently: across the processors.
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
                    let report = entry_report(checked, entry, &folder, rga.as_ref());
                    *done[index].lock().expect("no worker panics") = Some(report);
                }
            });
        }
    });
    let entries = done
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .expect("no worker panics")
                .expect("every entry ran")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let report = json!({
        "version": 1,
        "naga": "counts over each entry point's function and every function it calls",
        "amd": rga.as_ref().map(xtask::analyzer::Analyzer::description),
        "amd_unavailable": rga.is_none().then_some("disabled or unsupported platform"),
        "cache": rga.as_ref().map(xtask::analyzer::Analyzer::telemetry),
        "entries": entries,
    });
    let path = folder.join("report.json");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&report).expect("reports serialize"),
    )
    .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    print_summary(&report);
    println!("Wrote {}.", path.display());
    Ok(())
}

/// The report of `entry` in `checked`: naga's counts, its SPIR-V, and with
/// `rga` its AMD machine code per target.
fn entry_report(
    checked: &Checked,
    entry: &naga::EntryPoint,
    folder: &Path,
    rga: Option<&xtask::analyzer::Analyzer>,
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
    // Pipeline-overridable constants at their defaults, as pipelines that
    // set none compile them.
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
    let stem = format!("{}.{}", checked.name, entry.name);
    let spirv = folder.join("spirv").join(format!("{stem}.spv"));
    std::fs::create_dir_all(spirv.parent().expect("files have folders"))
        .map_err(|error| error.to_string())?;
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    std::fs::write(&spirv, &bytes).map_err(|error| error.to_string())?;
    let stage = match entry.stage {
        naga::ShaderStage::Vertex => "vertex",
        naga::ShaderStage::Fragment => "fragment",
        naga::ShaderStage::Compute => "compute",
        _ => "other",
    };
    let amd = rga.map(|program| program.report(&spirv, &entry.name, stage, folder, &stem));
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

/// What an entry point's functions do, as naga's IR says.
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
    /// Counts `function` and, once each, the functions it calls.
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

/// The entry points costing most: by AMD vector registers where measured,
/// else by naga's expressions.
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
        "{} entry points. Most costly (RDNA 3 vector registers and instructions, naga expressions and samples):",
        entries.len()
    );
    for entry in ranked.iter().take(15) {
        let amd = &entry["amd"]["gfx1100"];
        println!(
            "  {:<44} {:>8} vgprs {:>5} instructions {:>5} expressions {:>3} samples",
            format!(
                "{}::{}",
                entry["module"].as_str().unwrap_or(""),
                entry["entry_point"].as_str().unwrap_or("")
            ),
            amd["resources"]["used_vgprs"]
                .as_f64()
                .map_or("-".into(), |value| value.to_string()),
            amd["instructions"]
                .as_u64()
                .map_or("-".into(), |value| value.to_string()),
            entry["naga"]["expressions"],
            entry["naga"]["texture_samples"],
        );
    }
}
