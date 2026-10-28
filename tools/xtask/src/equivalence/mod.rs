mod browser;
mod judge;
mod matrix;
mod session;
mod sides;

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

use serde_json::{Value, json};

use crate::Failure;
use crate::bench;
use crate::heavy::HeavyRun;
use crate::measure::{self, git};

use judge::{Judged, Verdict, compare_digests, digest_parts, hashes, judge, run_problems};
use matrix::{Cell, MATRIX, read_matrix};
use session::run_cell;
use sides::{Side, commit_side, prune, working_tree};

pub(crate) use judge::decode;
pub(crate) use matrix::check;

const FOLDER: &str = "target/equivalence";

const REPEAT: usize = 3;

const KEEP: usize = 4;

const USAGE: &str = "usage: cargo xtask equivalence [--base REV] [--candidate REV] [--tier smoke|full] [--cells NAME,...] [--aa] [--repeat N] [--keep N] [--gpu discrete|integrated] [--browser | --both [--hosts]]";

struct Options {
    base: Option<String>,
    candidate: Option<String>,
    tier: String,
    cells: Option<Vec<String>>,
    aa: bool,
    repeat: Option<usize>,
    keep: usize,
    gpu: String,
    native: bool,
    browser: bool,
    hosts: bool,
}

fn options(arguments: &[String]) -> Result<Options, String> {
    let mut options = Options {
        base: None,
        candidate: None,
        tier: "smoke".into(),
        cells: None,
        aa: false,
        repeat: None,
        keep: KEEP,
        gpu: "discrete".into(),
        native: true,
        browser: false,
        hosts: false,
    };
    let mut given = arguments.iter();
    while let Some(name) = given.next() {
        match name.as_str() {
            "--aa" => {
                options.aa = true;
                continue;
            }
            "--browser" => {
                (options.native, options.browser) = (false, true);
                continue;
            }
            "--both" => {
                (options.native, options.browser) = (true, true);
                continue;
            }
            "--hosts" => {
                options.hosts = true;
                continue;
            }
            _ => {}
        }
        let value = given
            .next()
            .ok_or_else(|| format!("{name} takes a value\n{USAGE}"))?;
        let count = || {
            value
                .parse::<usize>()
                .ok()
                .filter(|count| *count > 0)
                .ok_or_else(|| format!("{name} takes a count above 0\n{USAGE}"))
        };
        match name.as_str() {
            "--base" => options.base = Some(value.clone()),
            "--candidate" => options.candidate = Some(value.clone()),
            "--tier" if ["smoke", "full"].contains(&value.as_str()) => {
                options.tier = value.clone();
            }
            "--cells" => {
                options.cells = Some(value.split(',').map(str::to_owned).collect());
            }
            "--repeat" => options.repeat = Some(count()?),
            "--keep" => options.keep = count()?,
            "--gpu" if ["discrete", "integrated"].contains(&value.as_str()) => {
                options.gpu = value.clone();
            }
            _ => return Err(format!("unknown option {name} {value}\n{USAGE}")),
        }
    }
    if options.aa && options.candidate.is_some() {
        return Err(format!(
            "--aa runs the base against itself: no --candidate\n{USAGE}"
        ));
    }
    if options.hosts && !(options.native && options.browser) {
        return Err(format!(
            "--hosts compares the native and browser legs: with --both\n{USAGE}"
        ));
    }
    Ok(options)
}

pub(crate) fn tiers(root: &Path) -> Result<Vec<(String, String, String)>, String> {
    let (cells, _) = matrix::read_matrix(root)?;
    Ok(cells
        .into_iter()
        .map(|cell| (cell.name, cell.tier, cell.project))
        .collect())
}

fn default_base(root: &Path, rev: &str) -> Result<String, String> {
    git(root, &["merge-base", rev, "origin/main"]).map_err(|error| {
        format!("no merge-base of {rev} with origin/main ({error}): name one with --base")
    })
}

pub fn equivalence(root: &Path, arguments: &[String]) -> Result<(), Failure> {
    let options = options(arguments)?;
    let cells = selected(root, &options)?;
    let projects: BTreeSet<&str> = cells.iter().map(|cell| cell.project.as_str()).collect();
    let projects: Vec<&str> = projects.into_iter().collect();
    let base_rev = match &options.base {
        Some(base) => base.clone(),
        None => default_base(root, options.candidate.as_deref().unwrap_or("HEAD"))?,
    };
    let base = commit_side(root, "base", &base_rev, &projects)?;
    let candidate = match (&options.candidate, options.aa) {
        (_, true) => None,
        (Some(rev), false) => Some(commit_side(root, "candidate", rev, &projects)?),
        (None, false) => Some(working_tree(root, &projects)?),
    };
    let using: Vec<&str> = [&base]
        .into_iter()
        .chain(candidate.as_ref())
        .filter_map(|side| side.sha.as_deref())
        .collect();
    prune(root, options.keep, &using);
    let catalog = bench::catalog(root)?;
    let stamp = measure::iso(SystemTime::now()).replace([':', '.'], "-");
    let folder = root.join(FOLDER).join("runs").join(&stamp);
    fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
    let second = Second {
        side: candidate.as_ref().unwrap_or(&base),
        role: if options.aa {
            "base-again"
        } else {
            "candidate"
        },
    };
    eprintln!(
        "Equivalence of base {} and {} {}: {} cells.",
        base.label,
        second.role,
        second.side.label,
        cells.len()
    );
    let started = Instant::now();
    let mut legs = Vec::new();
    if options.native {
        let judged = cells
            .iter()
            .map(|cell| {
                judged(
                    cell,
                    &base,
                    &second,
                    &options,
                    &catalog,
                    &folder,
                    &Leg::Native,
                )
            })
            .collect();
        legs.push(("native", judged));
    }
    if options.browser {
        let base_page = browser::serve(root, &base, "tree", &folder)?;
        let second_page = match &candidate {
            None => None,
            Some(side) => Some(browser::serve(root, side, "tree-candidate", &folder)?),
        };
        let leg = Leg::Browser {
            root,
            base: &base_page,
            second: second_page.as_ref().unwrap_or(&base_page),
            size: browser::size(root)?,
        };
        let judged = cells
            .iter()
            .map(|cell| judged(cell, &base, &second, &options, &catalog, &folder, &leg))
            .collect();
        legs.push(("browser", judged));
    }
    let took = started.elapsed();
    let hosts = if options.hosts {
        hosts(&legs)
    } else {
        Vec::new()
    };
    let report = report(&options, &base, second.side, &legs, &hosts, took);
    fs::write(
        folder.join("report.json"),
        serde_json::to_string_pretty(&report).expect("reports serialize") + "\n",
    )
    .map_err(|error| error.to_string())?;
    for (leg, results) in &legs {
        if legs.len() > 1 {
            println!(
                "\n{} leg:",
                if *leg == "native" {
                    "Native"
                } else {
                    "Browser"
                }
            );
        }
        print_summary(&options, results);
    }
    print_hosts(&hosts);
    println!(
        "{} cells in {:.0} s; report and captures in {}.",
        cells.len(),
        took.as_secs_f64(),
        folder.display()
    );
    let all: Vec<&Judged> = legs.iter().flat_map(|(_, results)| results).collect();
    if options.aa {
        return aa_outcome(&all);
    }
    outcome(&all)
}

enum Leg<'a> {
    Native,
    Browser {
        root: &'a Path,
        base: &'a browser::Page,
        second: &'a browser::Page,
        size: [u32; 2],
    },
}

fn hosts(legs: &[(&str, Vec<Judged>)]) -> Vec<Value> {
    let leg = |name: &str| {
        legs.iter()
            .find(|(leg, _)| *leg == name)
            .map(|(_, judged)| judged)
    };
    let (Some(native), Some(browser)) = (leg("native"), leg("browser")) else {
        return Vec::new();
    };
    let mut measured = Vec::new();
    for (ours, theirs) in native.iter().zip(browser) {
        for key in ours.cell.keys() {
            let image = |runs: &[session::Run]| {
                runs.iter().find_map(|run| match run.shots.get(&key) {
                    Some(session::Shot::Taken { path, .. }) => Some(path.clone()),
                    _ => None,
                })
            };
            let (Some(a), Some(b)) = (image(&ours.candidate), image(&theirs.candidate)) else {
                continue;
            };
            let found = judge::pixel_difference(&a, &b);
            measured.push(json!({
                "cell": ours.cell.name,
                "capture": key,
                "native": a.display().to_string(),
                "browser": b.display().to_string(),
                "pixels": found.as_ref().map(|found| found.0).ok(),
                "total": found.as_ref().map(|found| found.1).ok(),
                "largest": found.as_ref().map(|found| found.2).ok(),
                "error": found.err(),
            }));
        }
    }
    measured
}

fn print_hosts(hosts: &[Value]) {
    if hosts.is_empty() {
        return;
    }
    println!("\nNative against Chrome, the candidate's images (measurements, not judged):");
    for entry in hosts {
        match entry["error"].as_str() {
            Some(error) => println!(
                "  {:22} {:24} {error}",
                entry["cell"].as_str().unwrap_or_default(),
                entry["capture"].as_str().unwrap_or_default()
            ),
            None => println!(
                "  {:22} {:24} {} of {} pixels differ (largest {})",
                entry["cell"].as_str().unwrap_or_default(),
                entry["capture"].as_str().unwrap_or_default(),
                entry["pixels"],
                entry["total"],
                entry["largest"]
            ),
        }
    }
}

fn selected(root: &Path, options: &Options) -> Result<Vec<Cell>, String> {
    let (cells, problems) = read_matrix(root)?;
    if !problems.is_empty() {
        return Err(format!(
            "{MATRIX} is not a valid matrix:\n  {}",
            problems.join("\n  ")
        ));
    }
    match &options.cells {
        Some(names) => {
            if let Some(name) = names
                .iter()
                .find(|name| !cells.iter().any(|cell| cell.name == **name))
            {
                return Err(format!("{MATRIX} has no cell {name}"));
            }
            Ok(cells
                .into_iter()
                .filter(|cell| names.contains(&cell.name))
                .collect())
        }
        None => Ok(cells
            .into_iter()
            .filter(|cell| options.tier == "full" || cell.tier == "smoke")
            .collect()),
    }
}

struct Second<'a> {
    side: &'a Side,
    role: &'static str,
}

fn judged<'a>(
    cell: &'a Cell,
    base: &Side,
    second: &Second,
    options: &Options,
    catalog: &[Value],
    folder: &Path,
    leg: &Leg,
) -> Judged<'a> {
    let _heavy = match HeavyRun::hold(&format!("cargo xtask equivalence: {}", cell.name)) {
        Ok(held) => held,
        Err(error) => return Judged::failed(cell, error),
    };
    let rounds = if options.aa {
        options.repeat.unwrap_or(1)
    } else if cell.unstable.is_empty() {
        1
    } else {
        options.repeat.unwrap_or(REPEAT)
    };
    let folder = folder.join(&cell.name);
    let (mut base_runs, mut second_runs) = (Vec::new(), Vec::new());
    let run_rounds = |base_runs: &mut Vec<_>, second_runs: &mut Vec<_>| {
        for _ in 0..rounds {
            let index = base_runs.len() + 1;
            match leg {
                Leg::Native => {
                    eprintln!("{}: base run {index}.", cell.name);
                    let at = folder.join(format!("base-{index}"));
                    base_runs.push(run_cell(cell, base, &at, &options.gpu));
                    eprintln!("{}: {} run {index}.", cell.name, second.role);
                    let at = folder.join(format!("{}-{index}", second.role));
                    second_runs.push(run_cell(cell, second.side, &at, &options.gpu));
                }
                Leg::Browser {
                    root,
                    base: base_page,
                    second: second_page,
                    size,
                } => {
                    let runs = [
                        (base, *base_page, "base", index),
                        (second.side, *second_page, second.role, index),
                    ];
                    let mut both = browser::round(root, cell, &runs, *size, &folder).into_iter();
                    base_runs.extend(both.next());
                    second_runs.extend(both.next());
                }
            }
        }
    };
    run_rounds(&mut base_runs, &mut second_runs);
    let (mut verdicts, mut changed) = judge(cell, &base_runs, &second_runs, catalog, &folder);
    let unmatched = verdicts.iter().any(|(key, verdict)| {
        cell.unstable.contains_key(key) && matches!(verdict, Verdict::Different(_))
    });
    if unmatched && !options.aa {
        eprintln!(
            "{}: an unstable capture matched no base image; running more.",
            cell.name
        );
        run_rounds(&mut base_runs, &mut second_runs);
        (verdicts, changed) = judge(cell, &base_runs, &second_runs, catalog, &folder);
    }
    Judged {
        cell,
        verdicts,
        changed,
        digests: compare_digests(cell, &base_runs, &second_runs),
        base: base_runs,
        candidate: second_runs,
    }
}

fn report(
    options: &Options,
    base: &Side,
    second: &Side,
    legs: &[(&str, Vec<Judged>)],
    hosts: &[Value],
    took: Duration,
) -> Value {
    let legs: serde_json::Map<String, Value> = legs
        .iter()
        .map(|(leg, results)| {
            let cells: Vec<Value> = results.iter().map(cell_report).collect();
            ((*leg).to_owned(), json!({ "cells": cells }))
        })
        .collect();
    let side = |side: &Side| {
        let programs: serde_json::Map<String, Value> = side
            .programs
            .iter()
            .map(|(project, program)| (project.clone(), program.display().to_string().into()))
            .collect();
        json!({
            "label": side.label,
            "sha": side.sha,
            "programs": programs,
        })
    };
    json!({
        "kind": if options.aa { "aa" } else { "equivalence" },
        "base": side(base),
        "candidate": side(second),
        "tier": options.tier,
        "gpu": options.gpu,
        "seconds": took.as_secs_f64(),
        "legs": legs,
        "hosts": hosts,
    })
}

fn cell_report(judged: &Judged) -> Value {
    let cell = judged.cell;
    let captures: Vec<Value> = judged
        .verdicts
        .iter()
        .map(|(key, verdict)| {
            let (result, detail) = match verdict {
                Verdict::Equal => ("equal", Value::Null),
                Verdict::WithinVariation {
                    pixels,
                    largest,
                    spread,
                } => (
                    "within-variation",
                    json!({ "pixels": pixels, "largest": largest, "base_spread": { "pixels": spread.0, "largest": spread.1 } }),
                ),
                Verdict::EqualUnstable { distinct } => {
                    ("equal-unstable", json!({ "base_distinct": distinct }))
                }
                Verdict::Different(what) => ("different", json!(what)),
                Verdict::Unavailable(why) => ("unavailable", json!(why)),
                Verdict::Unchecked(why) => ("unchecked", json!(why)),
            };
            json!({
                "capture": key,
                "result": result,
                "detail": detail,
                "unstable": cell.unstable.get(key),
                "base": hashes(&judged.base, key),
                "candidate": hashes(&judged.candidate, key),
            })
        })
        .collect();
    let work: Vec<Value> = judged
        .changed
        .iter()
        .map(|change| {
            json!({
                "capture": change.key,
                "name": change.name,
                "rule": change.rule,
                "base": change.base,
                "candidate": change.candidate,
                "fails": change.fails,
            })
        })
        .collect();
    let digests: Vec<Value> = judged
        .digests
        .iter()
        .map(|(view, verdict)| {
            let of = |runs: &[session::Run]| -> Vec<Value> {
                runs.iter()
                    .map(|run| {
                        run.digests
                            .get(view)
                            .map_or(Value::Null, |digest| digest["hash"].clone())
                    })
                    .collect()
            };
            json!({
                "view": view,
                "result": verdict,
                "base": of(&judged.base),
                "candidate": of(&judged.candidate),
            })
        })
        .collect();
    let problems: Vec<String> = run_problems("base", &judged.base)
        .into_iter()
        .chain(run_problems("candidate", &judged.candidate))
        .collect();
    json!({
        "name": cell.name,
        "captures": captures,
        "work": work,
        "digests": digests,
        "problems": problems,
    })
}

fn print_summary(options: &Options, results: &[Judged]) {
    println!("\n{:22} {:12} {:12} result", "cell", "view", "capture");
    for Judged { cell, verdicts, .. } in results {
        for (key, verdict) in verdicts {
            let (view, capture) = key.split_once(' ').unwrap_or((key, ""));
            let result = match verdict {
                Verdict::Equal => "equal".to_owned(),
                Verdict::WithinVariation {
                    pixels,
                    largest,
                    spread,
                } => format!(
                    "equal within the base's own variation: {pixels} pixels up to {largest} levels from its nearest image (the base's images differ by up to {} pixels, {} levels)",
                    spread.0, spread.1
                ),
                Verdict::EqualUnstable { distinct } => format!(
                    "equal: matches one of the base's {distinct} distinct images (unstable)"
                ),
                Verdict::Different(what) if options.aa => format!("varies: {what}"),
                Verdict::Different(what) => format!("DIFFERENT: {what}"),
                Verdict::Unavailable(why) => format!("unavailable on both sides: {why}"),
                Verdict::Unchecked(why) => format!("not checked: {why}"),
            };
            println!("{:22} {:12} {:12} {result}", cell.name, view, capture);
        }
    }
    let mut shown = false;
    for Judged {
        cell,
        changed,
        digests,
        ..
    } in results
    {
        for change in changed {
            if !shown {
                println!("\nWork counters that changed (rule; base -> candidate):");
                shown = true;
            }
            println!(
                "  {:22} {:24} {:30} {} -> {} ({}{})",
                cell.name,
                change.key,
                change.name,
                change.base,
                change.candidate,
                change.rule,
                if change.fails { ", FAILS" } else { "" }
            );
        }
        for (view, verdict) in digests {
            if verdict.starts_with("different") {
                println!("  {:22} digest of {view}: {verdict}", cell.name);
            }
        }
    }
}

fn outcome(results: &[&Judged]) -> Result<(), Failure> {
    let (mut different, mut unchecked) = (0, 0);
    for Judged {
        verdicts,
        changed,
        digests,
        ..
    } in results.iter().copied()
    {
        for (_, verdict) in verdicts {
            match verdict {
                Verdict::Different(_) => different += 1,
                Verdict::Unchecked(_) => unchecked += 1,
                _ => {}
            }
        }
        different += changed.iter().filter(|change| change.fails).count();
        different += digests
            .iter()
            .filter(|(_, verdict)| verdict.starts_with("different"))
            .count();
    }
    if different > 0 {
        return Err(Failure::found(format!(
            "the candidate differs from the base: {different} difference(s)"
        )));
    }
    if unchecked > 0 {
        return Err(format!("{unchecked} capture(s) could not be checked").into());
    }
    println!("The candidate is equivalent to the base.");
    Ok(())
}

fn aa_outcome(results: &[&Judged]) -> Result<(), Failure> {
    let mut varied = Vec::new();
    let mut steady = Vec::new();
    let mut unchecked = 0;
    for Judged {
        cell,
        verdicts,
        base,
        candidate,
        ..
    } in results.iter().copied()
    {
        for (key, verdict) in verdicts {
            let declared = cell.unstable.contains_key(key);
            let mut all = hashes(base, key);
            all.extend(hashes(candidate, key));
            let distinct: BTreeSet<&str> = all.into_iter().collect();
            match verdict {
                Verdict::Unchecked(_) => unchecked += 1,
                Verdict::Unavailable(_) => {}
                _ if distinct.len() > 1 && !declared => {
                    varied.push(format!(
                        "{} {key} ({} distinct images)",
                        cell.name,
                        distinct.len()
                    ));
                }
                _ if distinct.len() == 1 && declared => {
                    steady.push(format!("{} {key}", cell.name));
                }
                _ => {}
            }
        }
        let runs: Vec<&session::Run> = base.iter().chain(candidate).collect();
        let views: BTreeSet<&String> = runs.iter().flat_map(|run| run.digests.keys()).collect();
        for view in views {
            let digests: Vec<&Value> = runs
                .iter()
                .filter_map(|run| run.digests.get(view))
                .filter(|digest| !digest.is_null())
                .collect();
            let mut parts = BTreeSet::new();
            for pair in digests.windows(2) {
                parts.extend(digest_parts(pair[0], pair[1]));
            }
            for part in &parts {
                let key = format!("{view} digest:{part}");
                if !cell.unstable.contains_key(&key) {
                    varied.push(format!("{} {key}", cell.name));
                }
            }
            let prefix = format!("{view} digest:");
            for key in cell.unstable.keys() {
                if key
                    .strip_prefix(&prefix)
                    .is_some_and(|part| !parts.contains(part) && digests.len() > 1)
                {
                    steady.push(format!("{} {key}", cell.name));
                }
            }
        }
    }
    if !steady.is_empty() {
        println!(
            "Declared unstable but the same each run here: {}.",
            steady.join(", ")
        );
    }
    if !varied.is_empty() {
        return Err(Failure::found(format!(
            "what the matrix calls stable varied between base runs (fix it, or give each a reason under unstable): {}",
            varied.join(", ")
        )));
    }
    if unchecked > 0 {
        return Err(format!("{unchecked} capture(s) could not be checked").into());
    }
    println!("Every capture and digest part the matrix calls stable reproduced exactly.");
    Ok(())
}
