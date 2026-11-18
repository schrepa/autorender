mod bc5;
mod bc7;
mod chunks;
mod config;
mod encode;
mod entry;
mod fault;
mod fragment;
mod geometry;
mod ktx2;
mod maps;
mod markers;
mod mips;
mod motion;
mod order;
mod part;
mod picture;
mod probe;
mod rig;
mod scene;
mod sets;
mod source;
mod uris;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde_json::json;

use crate::fault::Fault;

fn print_faults(faults: &[Fault]) {
    for fault in faults {
        println!("{}", fault.json());
    }
}

fn folder(path: &Path) -> PathBuf {
    path.parent().map(Path::to_path_buf).unwrap_or_default()
}

fn cook_all(cook_path: &Path) -> ExitCode {
    let finished = cook_steps(cook_path);
    println!(
        "{}",
        json!({"cook": cook_path.display().to_string(), "finished": if finished { "success" } else { "failure: the faults above say why" }})
    );
    if finished {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn cook_steps(cook_path: &Path) -> bool {
    let cook = match config::read(cook_path) {
        Ok(cook) => cook,
        Err(faults) => {
            print_faults(&faults);
            return false;
        }
    };
    let base = folder(cook_path);
    let missing = config::missing_baked_pictures(&cook, &base);
    if !missing.is_empty() {
        print_faults(&missing);
        return false;
    }
    let mut written: BTreeMap<String, (String, Vec<u8>)> = BTreeMap::new();
    let mut ready = Vec::new();
    let mut failed = false;
    for entry in &cook.entries {
        println!("{}", json!({"entry": entry.name(), "started": true}));
        let report = entry::run(&cook, &base, entry, &base.join(entry.source()));
        let mut faults = report.faults;
        let mut files = Vec::new();
        for (file, bytes) in report.files {
            match written.get(&file) {
                Some((_, held)) if *held == bytes => {}
                Some((earlier, _)) => faults.push(Fault::new(
                    "FileTwice",
                    &cook_path.display().to_string(),
                    file.clone(),
                    format!(
                        "written with different bytes by entries `{earlier}` and `{}`",
                        entry.name()
                    ),
                )),
                None => {
                    written.insert(file.clone(), (entry.name().to_owned(), bytes.clone()));
                    files.push((file, bytes));
                }
            }
        }
        println!(
            "{}",
            json!({"entry": entry.name(), "counts": report.counts})
        );
        for note in &report.notes {
            println!("{note}");
        }
        print_faults(&faults);
        failed |= !faults.is_empty();
        ready.push((entry.name().to_owned(), files));
    }
    if failed {
        return false;
    }
    let output = base.join(&cook.output);
    if let Err(error) = std::fs::create_dir_all(&output) {
        print_faults(&[Fault::new(
            "OutputUnwritable",
            &output.display().to_string(),
            "folder",
            error,
        )]);
        return false;
    }
    for (name, files) in ready {
        for (file, bytes) in &files {
            let path = output.join(file);
            if let Err(error) = std::fs::write(&path, bytes) {
                print_faults(&[Fault::new(
                    "OutputUnwritable",
                    &path.display().to_string(),
                    "file",
                    error,
                )]);
                return false;
            }
        }
        let names: Vec<&String> = files.iter().map(|(file, _)| file).collect();
        println!("{}", json!({"entry": name, "written": names}));
    }
    for part in &cook.scene_parts {
        let present = part.needs.iter().all(|name| {
            cook.entries.iter().any(|entry| match entry {
                config::Entry::Picture(picture) => {
                    picture.name == *name && base.join(&picture.source).is_file()
                }
                config::Entry::Character(_)
                | config::Entry::Form(_)
                | config::Entry::Clip(_)
                | config::Entry::Static(_)
                | config::Entry::Map(_) => false,
            })
        });
        match part::apply(&base, part, present) {
            Ok(line) => println!("{line}"),
            Err(fault) => {
                print_faults(&[fault]);
                return false;
            }
        }
    }
    for variant in &cook.scenes {
        match scene::write(&base, variant) {
            Ok(swapped) => println!(
                "{}",
                json!({"scene": variant.to, "from": variant.from, "textures_swapped": swapped})
            ),
            Err(fault) => {
                print_faults(&[fault]);
                return false;
            }
        }
    }
    shipped_uris(&cook, &base)
}

fn shipped_uris(cook: &config::Cook, base: &Path) -> bool {
    let mut shipped: BTreeMap<PathBuf, Vec<String>> = BTreeMap::new();
    for variant in &cook.scenes {
        for scene in [&variant.from, &variant.to] {
            let path = base.join(scene);
            let name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            shipped.entry(folder(&path)).or_default().push(name);
        }
    }
    let mut missing = false;
    for (at, scenes) in &shipped {
        let (line, faults) = uris::check(at, scenes);
        println!("{line}");
        print_faults(&faults);
        missing |= !faults.is_empty();
    }
    !missing
}

fn check(glb: &Path, cook_path: &Path, name: &str) -> ExitCode {
    let cook = match config::read(cook_path) {
        Ok(cook) => cook,
        Err(faults) => {
            print_faults(&faults);
            return ExitCode::FAILURE;
        }
    };
    let Some(entry) = cook.entries.iter().find(|entry| entry.name() == name) else {
        print_faults(&[Fault::new(
            "EntryMissing",
            &cook_path.display().to_string(),
            "entries",
            name,
        )]);
        return ExitCode::FAILURE;
    };
    let report = entry::run(&cook, &folder(cook_path), entry, glb);
    println!(
        "{}",
        json!({"entry": name, "file": glb.display().to_string(), "counts": report.counts})
    );
    for note in &report.notes {
        println!("{note}");
    }
    print_faults(&report.faults);
    if report.faults.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn answer(result: Result<serde_json::Value, Fault>) -> ExitCode {
    match result {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(fault) => {
            print_faults(&[fault]);
            ExitCode::FAILURE
        }
    }
}

fn usage() -> ExitCode {
    print_faults(&[Fault::new(
        "Usage",
        "glb_cook",
        "arguments",
        "glb_cook <cook.json> | glb_cook --check <file.glb> --budgets <cook.json> <entry name> | glb_cook --pose <file.clip> <file.skel> <seconds> <joint name> | glb_cook --blend <a.clip> <seconds a> <b.clip> <seconds b> <file.skel> <mask joint> <feather levels> <joint name>",
    )]);
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let words: Vec<&str> = arguments.iter().map(String::as_str).collect();
    match words.as_slice() {
        ["--check", glb, "--budgets", cook, name] => check(Path::new(glb), Path::new(cook), name),
        ["--pose", clip, skeleton, seconds, joint] => {
            answer(probe::pose(clip, skeleton, seconds, joint))
        }
        ["--blend", rest @ ..] => match <&[&str; 8]>::try_from(rest) {
            Ok(words) => answer(probe::mixed(words)),
            Err(_) => usage(),
        },
        [cook] if !cook.starts_with("--") => cook_all(Path::new(cook)),
        _ => usage(),
    }
}
