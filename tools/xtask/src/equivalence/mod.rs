//! `cargo xtask equivalence`: whether a change leaves what the engine draws
//! and does as it was, on the native host. A base (the merge-base with
//! origin/main unless `--base` names a commit) and a candidate (the working
//! tree, uncommitted changes included, unless `--candidate` names a commit)
//! each build the Stress Lab's native host with its development tools; then
//! every cell of `bench/equivalence.json` of the tier asked for runs on
//! both, one after the other, in one invocation on this machine. A cell
//! launches its workload (a bench scene) or authored scene with its
//! switches, scene settings and generated files from each side's own tree,
//! applies its environment, steps its ticks, and for each view sets it,
//! renders until it settles (nothing loading or compiling, the work
//! counters unchanged frame to frame) and a few frames more, so time-sliced
//! work completes its rounds, then captures it as the cell says: the exact
//! state, a CPU or GPU-presented fraction of a tick, or ordinary display
//! frames, with each frame's work counters and the engine's state digest
//! where the host answers it.
//!
//! Captures compare by SHA-256, then pixel by pixel within the cell's
//! tolerance (0 unless it says). Work counters whose rule in
//! `bench/metrics.json` is `equal`, `not-above` or `false` fail on a change;
//! `shown` ones are listed. An unstable capture carries its reason in the
//! matrix and passes when a candidate run matches one of the base's over
//! `--repeat` runs, or lies no farther from the nearest than the base's own
//! images lie from each other; `--aa` runs the base against itself to find
//! them.
//! Exit 0: equal; 3: different; 1: could not check.

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

pub(crate) use matrix::check;

/// The folders under which builds, the base's checkout and runs go.
const FOLDER: &str = "target/equivalence";

/// Runs of each side for a cell with unstable captures, unless `--repeat`.
const REPEAT: usize = 3;

/// Builds of commits kept, unless `--keep`.
const KEEP: usize = 4;

const USAGE: &str = "usage: cargo xtask equivalence [--base REV] [--candidate REV] [--tier smoke|full] [--cells NAME,...] [--aa] [--repeat N] [--keep N] [--gpu discrete|integrated]";

/// What the command line asks for.
struct Options {
    base: Option<String>,
    candidate: Option<String>,
    tier: String,
    cells: Option<Vec<String>>,
    aa: bool,
    /// Runs of each side for a cell with unstable captures; with `--aa`,
    /// pairs of base runs (one unless given).
    repeat: Option<usize>,
    keep: usize,
    gpu: String,
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
    };
    let mut given = arguments.iter();
    while let Some(name) = given.next() {
        if name == "--aa" {
            options.aa = true;
            continue;
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
    Ok(options)
}

/// The merge-base of `rev` with origin/main: the default base.
fn default_base(root: &Path, rev: &str) -> Result<String, String> {
    git(root, &["merge-base", rev, "origin/main"]).map_err(|error| {
        format!("no merge-base of {rev} with origin/main ({error}): name one with --base")
    })
}

pub fn equivalence(root: &Path, arguments: &[String]) -> Result<(), Failure> {
    let options = options(arguments)?;
    let cells = selected(root, &options)?;
    let base_rev = match &options.base {
        Some(base) => base.clone(),
        None => default_base(root, options.candidate.as_deref().unwrap_or("HEAD"))?,
    };
    let base = commit_side(root, "base", &base_rev)?;
    let candidate = match (&options.candidate, options.aa) {
        (_, true) => None,
        (Some(rev), false) => Some(commit_side(root, "candidate", rev)?),
        (None, false) => Some(working_tree(root)?),
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
    let heavy = HeavyRun::hold(&format!("cargo xtask equivalence ({} cells)", cells.len()))?;
    let results: Vec<Judged> = cells
        .iter()
        .map(|cell| judged(cell, &base, &second, &options, &catalog, &folder))
        .collect();
    drop(heavy);
    let took = started.elapsed();
    let report = report(&options, &base, second.side, &results, took);
    fs::write(
        folder.join("report.json"),
        serde_json::to_string_pretty(&report).expect("reports serialize") + "\n",
    )
    .map_err(|error| error.to_string())?;
    print_summary(&options, &results);
    println!(
        "{} cells in {:.0} s; report and captures in {}.",
        results.len(),
        took.as_secs_f64(),
        folder.display()
    );
    if options.aa {
        return aa_outcome(&results);
    }
    outcome(&results)
}

/// The cells the options select, from a valid matrix: those `--cells`
/// names, else the tier's.
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

/// The side runs alternate with the base's: the candidate, or with `--aa`
/// the base again, and what reports and folders call it.
struct Second<'a> {
    side: &'a Side,
    role: &'static str,
}

/// `cell` run on the base and the second side in rounds of one run each
/// (more for unstable captures, and once more where one matched no base
/// image), and judged.
fn judged<'a>(
    cell: &'a Cell,
    base: &Side,
    second: &Second,
    options: &Options,
    catalog: &[Value],
    folder: &Path,
) -> Judged<'a> {
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
            eprintln!("{}: base run {index}.", cell.name);
            let at = folder.join(format!("base-{index}"));
            base_runs.push(run_cell(cell, base, &at, &options.gpu));
            eprintln!("{}: {} run {index}.", cell.name, second.role);
            let at = folder.join(format!("{}-{index}", second.role));
            second_runs.push(run_cell(cell, second.side, &at, &options.gpu));
        }
    };
    run_rounds(&mut base_runs, &mut second_runs);
    let (mut verdicts, mut changed) = judge(cell, &base_runs, &second_runs, catalog, &folder);
    // An unstable capture that matched none of the base's images may have
    // missed the base's less frequent ones: one more round of runs before it
    // counts as different.
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

/// The run as JSON: the sides, each capture's verdict and hashes, the
/// counters that changed, and the digests.
fn report(
    options: &Options,
    base: &Side,
    second: &Side,
    results: &[Judged],
    took: Duration,
) -> Value {
    let cells: Vec<Value> = results.iter().map(cell_report).collect();
    let side = |side: &Side| {
        json!({
            "label": side.label,
            "sha": side.sha,
            "program": side.program.display().to_string(),
        })
    };
    json!({
        "kind": if options.aa { "aa" } else { "equivalence" },
        "base": side(base),
        "candidate": side(second),
        "tier": options.tier,
        "gpu": options.gpu,
        "seconds": took.as_secs_f64(),
        "cells": cells,
    })
}

/// One judged cell as JSON.
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

/// Prints each capture's verdict, then the work counters that changed.
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

/// The comparison's exit: 0 when every capture, judged counter and digest
/// is equal; 3 when one differs; 1 when one could not be checked.
fn outcome(results: &[Judged]) -> Result<(), Failure> {
    let (mut different, mut unchecked) = (0, 0);
    for Judged {
        verdicts,
        changed,
        digests,
        ..
    } in results
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

/// The A/A run's exit: 0 when every capture the matrix calls stable gave
/// the same image each run; 3 naming those that varied, which need a fix
/// or a reason in the matrix.
fn aa_outcome(results: &[Judged]) -> Result<(), Failure> {
    let mut varied = Vec::new();
    let mut steady = Vec::new();
    let mut unchecked = 0;
    for Judged {
        cell,
        verdicts,
        base,
        candidate,
        ..
    } in results
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
        // Digest parts that differ between any two runs of the base.
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
