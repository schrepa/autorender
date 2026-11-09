use std::collections::BTreeSet;
use std::path::Path;
use std::time::{Duration, Instant};

use engine_command_line::{Command, CommandOption};

use crate::Repository;

const USAGE: &str = "usage: cargo xtask lint [--base REV] [--all]";

pub struct Stage<C: ?Sized + 'static> {
    pub name: &'static str,
    pub needs: &'static [&'static str],
    pub covers: fn(&C) -> bool,
    pub run: fn(&C) -> Result<(), String>,
}

pub fn command() -> Command {
    Command::new(
        "lint",
        [
            CommandOption::taking("--base", "REV").defaults_to("HEAD"),
            CommandOption::flag("--all"),
        ],
    )
}

pub struct Changes {
    pub every: bool,
    paths: BTreeSet<String>,
    pub affected: BTreeSet<String>,
}

impl Changes {
    pub fn read(repository: &Repository, arguments: &[String]) -> Result<Self, String> {
        let mut base = "HEAD";
        let mut all = false;
        let mut given = arguments.iter();
        while let Some(argument) = given.next() {
            match argument.as_str() {
                "--base" => base = given.next().map(String::as_str).ok_or(USAGE)?,
                "--all" => all = true,
                _ => return Err(format!("unknown option {argument}\n{USAGE}")),
            }
        }
        let own = || -> BTreeSet<String> {
            repository
                .own()
                .map(|package| package.name.clone())
                .collect()
        };
        if all {
            return Ok(Self {
                every: true,
                paths: BTreeSet::new(),
                affected: own(),
            });
        }
        let root = repository.root.as_path();
        let base = git(root, &["merge-base", "HEAD", base]).map_err(|error| {
            format!("no merge-base of HEAD with {base} ({error}): name another with --base")
        })?;
        let mut paths: BTreeSet<String> = git(
            root,
            &["diff", "--name-only", "--no-renames", "--relative", &base],
        )?
        .lines()
        .map(str::to_owned)
        .collect();
        paths.extend(
            git(root, &["ls-files", "--others", "--exclude-standard"])?
                .lines()
                .map(str::to_owned),
        );
        let mut touched = BTreeSet::new();
        let mut outside = Vec::new();
        for path in &paths {
            match owner(repository, path) {
                Some(package) => {
                    touched.insert(package.to_owned());
                }
                None => outside.push(path.as_str()),
            }
        }
        let every = !outside.is_empty();
        let affected = if every {
            own()
        } else {
            dependents(repository, &touched)
        };
        let record = serde_json::json!({
            "event": "changes",
            "files": paths.len(),
            "base": &base[..base.len().min(12)],
            "crates": touched,
            "affected": affected.len(),
        });
        println!("{record}");
        if every {
            let record = serde_json::json!({
                "event": "paths_outside_crates",
                "paths": outside,
            });
            println!("{record}");
        }
        Ok(Self {
            every,
            paths,
            affected,
        })
    }

    pub fn affects(&self, package: &str) -> bool {
        self.affected.contains(package)
    }

    pub fn any(&self, covered: impl Fn(&str) -> bool) -> bool {
        self.every || self.paths.iter().any(|path| covered(path))
    }
}

fn git(root: &Path, arguments: &[&str]) -> Result<String, String> {
    let output = std::process::Command::new("git")
        .args(arguments)
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot run git: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed: {}",
            arguments.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn owner<'a>(repository: &'a Repository, path: &str) -> Option<&'a str> {
    repository
        .own()
        .filter_map(|package| Some((package.path.as_deref()?, package.name.as_str())))
        .filter(|(folder, _)| {
            path.strip_prefix(folder)
                .is_some_and(|rest| rest.starts_with('/'))
        })
        .max_by_key(|(folder, _)| folder.len())
        .map(|(_, name)| name)
}

fn dependents(repository: &Repository, touched: &BTreeSet<String>) -> BTreeSet<String> {
    let mut affected = touched.clone();
    loop {
        let reached: Vec<String> = repository
            .own()
            .filter(|package| !affected.contains(&package.name))
            .filter(|package| {
                package
                    .dependencies
                    .iter()
                    .any(|dependency| affected.contains(&dependency.package))
            })
            .map(|package| package.name.clone())
            .collect();
        if reached.is_empty() {
            return affected;
        }
        affected.extend(reached);
    }
}

enum Outcome {
    Passed,
    Unchanged,
    Failed(String),
    Skipped(String),
}

pub fn run<C: ?Sized + 'static>(context: &C, stages: &[Stage<C>]) -> Result<(), String> {
    let mut results: Vec<(&str, Outcome, Duration)> = Vec::with_capacity(stages.len());
    for stage in stages {
        if !(stage.covers)(context) {
            results.push((stage.name, Outcome::Unchanged, Duration::ZERO));
            continue;
        }
        let blocked: Vec<&str> = stage
            .needs
            .iter()
            .copied()
            .filter(|need| {
                !results
                    .iter()
                    .any(|(name, outcome, _)| name == need && matches!(outcome, Outcome::Passed))
            })
            .collect();
        if !blocked.is_empty() {
            results.push((
                stage.name,
                Outcome::Skipped(blocked.join(", ")),
                Duration::ZERO,
            ));
            continue;
        }
        println!(
            "{}",
            serde_json::json!({"event": "stage_started", "stage": stage.name})
        );
        let start = Instant::now();
        let outcome = match (stage.run)(context) {
            Ok(()) => Outcome::Passed,
            Err(error) => {
                let record = serde_json::json!({
                    "event": "stage_failed",
                    "stage": stage.name,
                    "error": error,
                });
                eprintln!("{record}");
                Outcome::Failed(error)
            }
        };
        results.push((stage.name, outcome, start.elapsed()));
    }
    for (name, outcome, time) in &results {
        let result = match outcome {
            Outcome::Passed => "ok",
            Outcome::Unchanged => "unchanged",
            Outcome::Failed(_) => "failed",
            Outcome::Skipped(_) => "skipped",
        };
        let record = serde_json::json!({
            "event": "stage_result",
            "stage": name,
            "result": result,
            "seconds": time.as_secs_f64(),
        });
        println!("{record}");
    }
    let total: Duration = results.iter().map(|(_, _, time)| *time).sum();
    println!(
        "{}",
        serde_json::json!({"event": "stages_total", "seconds": total.as_secs_f64()})
    );
    let problems: Vec<String> = results
        .iter()
        .filter_map(|(name, outcome, _)| match outcome {
            Outcome::Passed | Outcome::Unchanged => None,
            Outcome::Failed(error) => Some(format!("{name} failed: {error}")),
            Outcome::Skipped(blocked) => Some(format!("{name} skipped: needs {blocked}")),
        })
        .collect();
    if problems.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{} of {} stages did not pass:\n{}",
        problems.len(),
        results.len(),
        problems.join("\n")
    ))
}
