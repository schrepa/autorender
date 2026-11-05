use std::collections::BTreeSet;
use std::path::Path;
use std::time::{Duration, Instant};

use engine_checks::Repository;
use engine_checks::tools::clippy_free;

use crate::Failure;
use crate::measure::git;
use crate::process::{Client, clients};
use crate::smoke::PLAYER;

const USAGE: &str =
    "usage: cargo xtask verify [--level quick|pr|merge] [--base REV] [--plan] [--json]";

const TOOLS: &str = "xtask";

const CLIPPY: [&str; 5] = ["--all-targets", "--locked", "--", "-D", "warnings"];

const QUICK_STAGES: &[&str] = &[
    "kinds",
    "policy",
    "comments",
    "fmt",
    "web layers",
    "web errors",
    "web parameters",
    "scene sources",
    "bench",
    "equivalence matrix",
    "structure",
    "install",
];

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Level {
    Quick,
    Pr,
    Merge,
}

impl Level {
    const fn name(self) -> &'static str {
        match self {
            Self::Quick => "quick",
            Self::Pr => "pr",
            Self::Merge => "merge",
        }
    }
}

#[derive(Default)]
struct Selection {
    everything: bool,
    outside: Vec<String>,
    touched: BTreeSet<String>,
    affected: BTreeSet<String>,
}

impl Selection {
    fn of(repository: &Repository, changed: &BTreeSet<String>) -> Self {
        let mut selection = Self::default();
        for path in changed {
            match owner(repository, path) {
                Some(package) => {
                    selection.touched.insert(package.to_owned());
                }
                None => selection.outside.push(path.clone()),
            }
        }
        selection.everything = !selection.outside.is_empty();
        selection.affected = if selection.everything {
            repository
                .own()
                .map(|package| package.name.clone())
                .collect()
        } else {
            dependents(repository, &selection.touched)
        };
        selection
    }

    fn affects(&self, package: &str) -> bool {
        self.affected.contains(package)
    }
}

fn project(folder: &str, within: &BTreeSet<String>) -> bool {
    clients()
        .iter()
        .any(|client| client.folder == folder && within.contains(&client.package))
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

struct Step<'a> {
    name: String,
    run: Box<dyn Fn() -> Result<(), Failure> + 'a>,
}

impl<'a> Step<'a> {
    fn new(name: impl Into<String>, run: impl Fn() -> Result<(), Failure> + 'a) -> Self {
        Self {
            name: name.into(),
            run: Box::new(run),
        }
    }
}

pub fn verify(root: &Path, arguments: &[String]) -> Result<(), Failure> {
    let mut level = Level::Pr;
    let mut base = None;
    let mut plan_only = false;
    let mut json = false;
    let mut given = arguments.iter();
    while let Some(argument) = given.next() {
        match argument.as_str() {
            "--level" => {
                level = match given.next().map(String::as_str) {
                    Some("quick") => Level::Quick,
                    Some("pr") => Level::Pr,
                    Some("merge") => Level::Merge,
                    _ => return Err(USAGE.into()),
                };
            }
            "--base" => base = Some(given.next().ok_or(USAGE)?.clone()),
            "--plan" => plan_only = true,
            "--json" => json = true,
            _ => return Err(format!("unknown option {argument}\n{USAGE}").into()),
        }
    }
    let base = git(
        root,
        &[
            "merge-base",
            "HEAD",
            base.as_deref().unwrap_or("origin/main"),
        ],
    )
    .map_err(|error| {
        format!("no merge-base of HEAD with the base ({error}): name one with --base")
    })?;
    let mut changed: BTreeSet<String> = git(root, &["diff", "--name-only", &base])?
        .lines()
        .map(str::to_owned)
        .collect();
    changed.extend(
        git(root, &["ls-files", "--others", "--exclude-standard"])?
            .lines()
            .map(str::to_owned),
    );
    let selection = Selection::of(crate::rules::repository(root)?, &changed);
    let record = serde_json::json!({
        "event": "changes",
        "files": changed.len(),
        "base": &base[..base.len().min(12)],
        "crates": selection.touched,
        "affected": selection.affected.len(),
    });
    println!("{record}");
    if !selection.outside.is_empty() {
        let record = serde_json::json!({
            "event": "paths_outside_crates",
            "paths": selection.outside,
        });
        println!("{record}");
    }
    let steps = steps(root, level, &base, &selection)?;
    let names: Vec<&str> = steps.iter().map(|step| step.name.as_str()).collect();
    println!(
        "{}",
        serde_json::json!({"event": "steps_planned", "steps": names})
    );
    if plan_only {
        return Ok(());
    }
    let ran = run_steps(&steps);
    if json {
        println!(
            "{}",
            serde_json::json!({ "verify": {
                "level": level.name(),
                "base": base,
                "steps": ran.steps,
            }})
        );
    }
    ran.outcome
}

fn quick<'a>(root: &'a Path, selection: &Selection) -> Result<Vec<Step<'a>>, String> {
    let mut steps = Vec::new();
    for stage in crate::lint::LINT
        .iter()
        .filter(|stage| QUICK_STAGES.contains(&stage.name))
    {
        steps.push(Step::new(stage.name, move || Ok((stage.run)(root)?)));
    }
    let free: Vec<&'static str> = crate::rules::warnings_allowed(root)?
        .into_iter()
        .filter(|package| selection.affects(package))
        .collect();
    if selection.affected.len() > free.len() {
        let mut arguments = vec!["clippy".to_owned()];
        if selection.everything {
            arguments.push("--workspace".into());
            for package in &free {
                arguments.extend(["--exclude".to_owned(), (*package).to_owned()]);
            }
        } else {
            for package in &selection.affected {
                if !free.contains(&package.as_str()) {
                    arguments.extend(["-p".to_owned(), package.clone()]);
                }
            }
        }
        arguments.extend(CLIPPY.map(str::to_owned));
        let name = format!("cargo {}", arguments.join(" "));
        steps.push(Step::new(name, move || {
            let arguments: Vec<&str> = arguments.iter().map(String::as_str).collect();
            Ok(crate::process::run(root, "cargo", &arguments)?)
        }));
    }
    if !free.is_empty() {
        let name = format!("cargo clippy {} --all-targets", free.join(" "));
        steps.push(Step::new(name, move || {
            Ok(clippy_free(root, &free, &["--all-targets"])?)
        }));
    }
    if selection.affects(TOOLS) {
        steps.push(Step::new("pure (every corpus)", move || {
            crate::pure::pure(root, &[])
        }));
    }
    Ok(steps)
}

fn smokes<'a>(root: &'a Path, merge: bool, selection: &Selection) -> Vec<Step<'a>> {
    let wanted: Vec<&Client> = clients()
        .iter()
        .filter(|client| client.native.is_some())
        .filter(|client| merge || selection.affects(&client.package))
        .collect();
    let mut runs: Vec<Vec<String>> = Vec::new();
    if wanted.iter().any(|client| client.folder == PLAYER) {
        runs.push(Vec::new());
    }
    if merge {
        runs.push(vec!["--native-profile".to_owned(), "dev".to_owned()]);
        runs.push(vec![
            "meshes".to_owned(),
            "shadow_tiles_settle".to_owned(),
            "--browser".to_owned(),
        ]);
    }
    for client in wanted.iter().filter(|client| client.folder != PLAYER) {
        runs.push(vec!["--project".to_owned(), client.folder.clone()]);
    }
    runs.into_iter()
        .map(|options| {
            let name = format!("smoke {}", options.join(" ")).trim_end().to_owned();
            Step::new(name, move || crate::smoke::smoke(root, &options))
        })
        .collect()
}

fn steps<'a>(
    root: &'a Path,
    level: Level,
    base: &'a str,
    selection: &Selection,
) -> Result<Vec<Step<'a>>, String> {
    if level == Level::Quick {
        return quick(root, selection);
    }
    let merge = level == Level::Merge;
    let tools = selection.affects(TOOLS);
    let mut steps = vec![Step::new("lint", move || {
        Ok(engine_checks::stages::run(root, crate::lint::LINT)?)
    })];
    steps.extend(smokes(root, merge, selection));
    if merge || tools {
        steps.push(Step::new("pure --wasm terrain tables", move || {
            crate::pure::pure(
                root,
                &[
                    "--wasm".to_owned(),
                    "terrain".to_owned(),
                    "tables".to_owned(),
                ],
            )
        }));
    }
    let tiers = crate::equivalence::tiers(root)?;
    if merge {
        let options = vec![
            "--base".to_owned(),
            base.to_owned(),
            "--tier".into(),
            "full".into(),
        ];
        steps.push(Step::new(
            format!("equivalence {}", options.join(" ")),
            move || crate::equivalence::equivalence(root, &options),
        ));
    } else {
        let cells: BTreeSet<&str> = tiers
            .iter()
            .filter(|(_, tier, folder)| {
                project(folder, &selection.touched)
                    || (tier == "smoke" && project(folder, &selection.affected))
            })
            .map(|(name, _, _)| name.as_str())
            .collect();
        if !cells.is_empty() {
            let options = vec![
                "--base".to_owned(),
                base.to_owned(),
                "--cells".into(),
                cells.into_iter().collect::<Vec<_>>().join(","),
            ];
            steps.push(Step::new(
                format!("equivalence {}", options.join(" ")),
                move || crate::equivalence::equivalence(root, &options),
            ));
        }
    }
    let browser = clients()
        .iter()
        .any(|client| selection.affects(&client.package));
    if merge && browser {
        let options = vec!["--base".to_owned(), base.to_owned(), "--browser".into()];
        steps.push(Step::new(
            format!("equivalence {}", options.join(" ")),
            move || crate::equivalence::equivalence(root, &options),
        ));
    }
    if tools {
        steps.push(Step::new("selfcheck", move || {
            crate::selfcheck::selfcheck(root, &[])
        }));
    }
    Ok(steps)
}

fn failures(message: &str) -> serde_json::Value {
    let errors: Option<Vec<serde_json::Value>> = message
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str::<serde_json::Value>(line)
                .ok()
                .filter(|error| error.get("code").is_some() && error.get("path").is_some())
        })
        .collect();
    match errors {
        Some(errors) if !errors.is_empty() => serde_json::Value::Array(errors),
        _ => serde_json::Value::String(message.to_owned()),
    }
}

struct Ran {
    steps: Vec<serde_json::Value>,
    outcome: Result<(), Failure>,
}

fn run_steps(steps: &[Step]) -> Ran {
    let mut results: Vec<(&str, Option<u8>, Duration)> = Vec::new();
    let mut records = Vec::new();
    for step in steps {
        println!(
            "{}",
            serde_json::json!({"event": "step_started", "step": step.name})
        );
        let started = Instant::now();
        let (code, failure) = match (step.run)() {
            Ok(()) => (None, None),
            Err(failure) => {
                let record = serde_json::json!({
                    "event": "step_failed",
                    "step": step.name,
                    "error": failure.message,
                });
                eprintln!("{record}");
                (Some(failure.code), Some(failure.message))
            }
        };
        let took = started.elapsed();
        records.push(serde_json::json!({
            "step": step.name,
            "ok": code.is_none(),
            "exit": code,
            "seconds": took.as_secs_f64(),
            "failure": failure.as_deref().map(failures),
        }));
        results.push((&step.name, code, took));
    }
    Ran {
        steps: records,
        outcome: tabled(&results),
    }
}

fn tabled(results: &[(&str, Option<u8>, Duration)]) -> Result<(), Failure> {
    for (name, code, took) in results {
        let record = serde_json::json!({
            "event": "step_result",
            "step": name,
            "ok": code.is_none(),
            "exit": code,
            "seconds": took.as_secs_f64(),
        });
        println!("{record}");
    }
    let failed: Vec<&str> = results
        .iter()
        .filter(|(_, code, _)| code.is_some())
        .map(|(name, _, _)| *name)
        .collect();
    if failed.is_empty() {
        return Ok(());
    }
    let message = format!("verify: {} did not pass", failed.join("; "));
    if results.iter().any(|(_, code, _)| *code == Some(3)) {
        Err(Failure::found(message))
    } else {
        Err(message.into())
    }
}
