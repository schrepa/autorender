mod bc5;
mod bc7;
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
mod picture;
mod probe;
mod rig;
mod source;

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
    let cook = match config::read(cook_path) {
        Ok(cook) => cook,
        Err(faults) => {
            print_faults(&faults);
            return ExitCode::FAILURE;
        }
    };
    let base = folder(cook_path);
    let mut written: BTreeMap<String, String> = BTreeMap::new();
    let mut ready = Vec::new();
    let mut failed = false;
    for entry in &cook.entries {
        let report = entry::run(&cook, &base, entry, &base.join(entry.source()));
        let mut faults = report.faults;
        for (file, _) in &report.files {
            if let Some(earlier) = written.insert(file.clone(), entry.name().to_owned()) {
                faults.push(Fault::new(
                    "FileTwice",
                    &cook_path.display().to_string(),
                    file.clone(),
                    format!("written by entries `{earlier}` and `{}`", entry.name()),
                ));
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
        ready.push((entry.name().to_owned(), report.files));
    }
    if failed {
        return ExitCode::FAILURE;
    }
    let output = base.join(&cook.output);
    if let Err(error) = std::fs::create_dir_all(&output) {
        print_faults(&[Fault::new(
            "OutputUnwritable",
            &output.display().to_string(),
            "folder",
            error,
        )]);
        return ExitCode::FAILURE;
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
                return ExitCode::FAILURE;
            }
        }
        let names: Vec<&String> = files.iter().map(|(file, _)| file).collect();
        println!("{}", json!({"entry": name, "written": names}));
    }
    ExitCode::SUCCESS
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
