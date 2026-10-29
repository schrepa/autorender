mod map;

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::Failure;
use crate::measure::git;
use map::Check;

const USAGE: &str = "usage: cargo xtask verify [--level quick|pr|merge] [--base REV] [--plan]";

const QUICK_STAGES: &[&str] = &[
    "policy",
    "comments",
    "fmt",
    "web layers",
    "web errors",
    "web parameters",
    "scene sources",
    "generated files",
    "bench",
    "equivalence matrix",
    "coverage",
    "path map",
    "dependencies",
    "faces",
    "schedule ambiguities",
    "exports",
    "structure",
    "examples are leaves",
];

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Level {
    Quick,
    Pr,
    Merge,
}

#[derive(Default)]
struct Selection {
    everything: bool,
    corpora: BTreeSet<&'static str>,
    pure_wasm: bool,
    shaders: bool,
    smokes: BTreeSet<String>,
    all_smokes: bool,
    projects: BTreeSet<&'static str>,
    equivalence: bool,
    browser: bool,
    self_checks: BTreeSet<&'static str>,
    packages: BTreeSet<String>,
    unmapped: Vec<String>,
}

impl Selection {
    fn add(&mut self, root: &Path, smokes: &[String], path: &str) {
        let Some(checks) = map::checks_of(path) else {
            self.unmapped.push(path.to_owned());
            self.everything = true;
            return;
        };
        for check in checks {
            match check {
                Check::Everything => self.everything = true,
                Check::Pure(corpus) => {
                    self.corpora.insert(corpus);
                }
                Check::PureWasm => self.pure_wasm = true,
                Check::Shaders => self.shaders = true,
                Check::Smokes(topic) => {
                    self.smokes.extend(
                        smokes
                            .iter()
                            .filter(|smoke| map::covers(topic, smoke))
                            .cloned(),
                    );
                }
                Check::ItsSmoke => {
                    let name = path
                        .rsplit('/')
                        .next()
                        .and_then(|file| file.strip_suffix("-smoke.mjs"))
                        .filter(|name| smokes.iter().any(|smoke| smoke == name));
                    self.smokes.extend(name.map(str::to_owned));
                }
                Check::AllSmokes => self.all_smokes = true,
                Check::Project => self.projects.extend(map::project_of(path)),
                Check::Equivalence => self.equivalence = true,
                Check::Browser => self.browser = true,
                Check::Tools(kind) => {
                    self.self_checks.insert(kind);
                }
            }
        }
        if let Some(package) = package_of(root, path) {
            self.packages.insert(package);
        }
    }
}

fn package_of(root: &Path, path: &str) -> Option<String> {
    let mut folder = Path::new(path).parent();
    while let Some(at) = folder {
        if let Ok(manifest) = fs::read_to_string(root.join(at).join("Cargo.toml"))
            && let Some((_, package)) = manifest.split_once("[package]")
        {
            return package.lines().find_map(|line| {
                let value = line.trim().strip_prefix("name")?.trim().strip_prefix('=')?;
                Some(value.trim().trim_matches('"').to_owned())
            });
        }
        folder = at.parent().filter(|parent| !parent.as_os_str().is_empty());
    }
    None
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
    let smokes = smoke_names(root)?;
    let mut selection = Selection::default();
    for path in &changed {
        selection.add(root, &smokes, path);
    }
    println!(
        "{} files changed since {}.",
        changed.len(),
        &base[..base.len().min(12)]
    );
    if !selection.unmapped.is_empty() {
        println!(
            "No rule of the path map covers {}: everything runs.",
            selection.unmapped.join(", ")
        );
    }
    let steps = steps(root, level, &base, &selection)?;
    println!("Steps:");
    for step in &steps {
        println!("  {}", step.name);
    }
    if plan_only {
        return Ok(());
    }
    run_steps(&steps)
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn steps<'a>(
    root: &'a Path,
    level: Level,
    base: &'a str,
    selection: &Selection,
) -> Result<Vec<Step<'a>>, String> {
    let mut steps = Vec::new();
    let everything = selection.everything;
    if level == Level::Quick {
        for stage in crate::lint::LINT
            .iter()
            .filter(|stage| QUICK_STAGES.contains(&stage.name))
        {
            steps.push(Step::new(stage.name, move || Ok((stage.run)(root)?)));
        }
        if everything || selection.shaders {
            steps.push(Step::new("gpu", move || {
                Ok(engine_checks::gpu::check(crate::rules::repository(root)?)?)
            }));
        }
        if everything || !selection.packages.is_empty() {
            let mut arguments = vec!["clippy".to_owned()];
            if everything {
                arguments.push("--workspace".into());
            } else {
                for package in &selection.packages {
                    arguments.extend(["-p".to_owned(), package.clone()]);
                }
            }
            arguments
                .extend(["--all-targets", "--locked", "--", "-D", "warnings"].map(str::to_owned));
            let name = format!("cargo {}", arguments.join(" "));
            steps.push(Step::new(name, move || {
                let arguments: Vec<&str> = arguments.iter().map(String::as_str).collect();
                Ok(crate::process::run(root, "cargo", &arguments)?)
            }));
        }
        let corpora: Vec<String> = if everything {
            Vec::new()
        } else {
            selection
                .corpora
                .iter()
                .map(|corpus| (*corpus).to_owned())
                .collect()
        };
        if everything || !corpora.is_empty() {
            let name = if corpora.is_empty() {
                "pure (every corpus)".to_owned()
            } else {
                format!("pure {}", corpora.join(" "))
            };
            steps.push(Step::new(name, move || crate::pure::pure(root, &corpora)));
        }
        return Ok(steps);
    }
    steps.push(Step::new("lint", move || {
        Ok(engine_checks::stages::run(root, crate::lint::LINT)?)
    }));
    let smokes: Vec<String> = if level == Level::Merge || everything || selection.all_smokes {
        vec!["--all".into()]
    } else {
        selection.smokes.iter().cloned().collect()
    };
    if !smokes.is_empty() {
        let mut options = vec!["--serve".to_owned(), "--keep-going".to_owned()];
        options.extend(smokes);
        steps.push(Step::new(
            format!("smoke {}", options.join(" ")),
            move || crate::smoke::smoke(root, &options),
        ));
    }
    if level == Level::Merge {
        let mut options = vec!["--native-profile".to_owned(), "dev".to_owned()];
        options.extend(
            smoke_names(root)?
                .into_iter()
                .filter(|smoke| map::covers("native", smoke)),
        );
        if options.len() > 2 {
            steps.push(Step::new(
                format!("smoke {}", options.join(" ")),
                move || crate::smoke::smoke(root, &options),
            ));
        }
    }
    let every_project = level == Level::Merge || everything;
    for client in crate::process::clients() {
        let folder = client.folder.as_str();
        let wanted = every_project || selection.projects.contains(folder);
        if !wanted || crate::smoke::checks(root, folder)?.is_empty() {
            continue;
        }
        let options = vec![
            "--serve".to_owned(),
            "--keep-going".to_owned(),
            "--project".to_owned(),
            folder.to_owned(),
            "--all".to_owned(),
        ];
        steps.push(Step::new(
            format!("smoke {}", options.join(" ")),
            move || crate::smoke::smoke(root, &options),
        ));
    }
    if level == Level::Merge || everything || selection.pure_wasm {
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
    let project_cells: BTreeSet<&str> = tiers
        .iter()
        .filter(|(_, _, project)| selection.projects.contains(project.as_str()))
        .map(|(name, _, _)| name.as_str())
        .collect();
    let wanted_equivalence = everything || selection.equivalence || !project_cells.is_empty();
    if level == Level::Merge {
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
    } else if wanted_equivalence {
        let mut cells: BTreeSet<&str> = tiers
            .iter()
            .filter(|(_, tier, _)| tier == "smoke")
            .map(|(name, _, _)| name.as_str())
            .collect();
        cells.extend(project_cells);
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
    if level == Level::Merge && (everything || selection.browser) {
        let options = vec!["--base".to_owned(), base.to_owned(), "--browser".into()];
        steps.push(Step::new(
            format!("equivalence {}", options.join(" ")),
            move || crate::equivalence::equivalence(root, &options),
        ));
    }
    let self_checks: Vec<String> = if everything {
        Vec::new()
    } else {
        selection
            .self_checks
            .iter()
            .map(|kind| (*kind).to_owned())
            .collect()
    };
    if everything || !self_checks.is_empty() {
        let name = format!("selfcheck {}", self_checks.join(" "));
        steps.push(Step::new(name.trim_end().to_owned(), move || {
            crate::selfcheck::selfcheck(root, &self_checks)
        }));
    }
    Ok(steps)
}

fn run_steps(steps: &[Step]) -> Result<(), Failure> {
    let mut results: Vec<(&str, Option<u8>, Duration)> = Vec::new();
    for step in steps {
        println!("\n== verify: {} ==", step.name);
        let started = Instant::now();
        let code = match (step.run)() {
            Ok(()) => None,
            Err(failure) => {
                eprintln!("{} failed: {}", step.name, failure.message);
                Some(failure.code)
            }
        };
        results.push((&step.name, code, started.elapsed()));
    }
    let width = results
        .iter()
        .map(|(name, _, _)| name.len())
        .max()
        .unwrap_or(4)
        .max(4);
    println!("\n{:width$}  result   seconds", "step");
    for (name, code, took) in &results {
        let result = match code {
            None => "ok".to_owned(),
            Some(code) => format!("exit {code}"),
        };
        println!("{name:width$}  {result:7}  {:7.1}", took.as_secs_f64());
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

fn smoke_names(root: &Path) -> Result<Vec<String>, String> {
    Ok(crate::smoke::scripts(root)?
        .into_iter()
        .map(|(name, _)| name)
        .collect())
}

pub(crate) fn check(root: &Path) -> Result<(), String> {
    let mut problems = Vec::new();
    let tracked = git(root, &["ls-files"])?;
    let unmapped: Vec<&str> = tracked
        .lines()
        .filter(|path| map::checks_of(path).is_none())
        .collect();
    if !unmapped.is_empty() {
        problems.push(format!("no rule covers {}", unmapped.join(", ")));
    }
    let smokes = smoke_names(root)?;
    let corpora: Vec<&str> = crate::pure::CORPORA
        .iter()
        .map(|corpus| corpus.name)
        .collect();
    let mut topics = BTreeSet::new();
    for rule in map::MAP {
        for check in rule.checks {
            match check {
                Check::Smokes(topic) => {
                    topics.insert(*topic);
                }
                Check::Pure(corpus) if !corpora.contains(corpus) => {
                    problems.push(format!(
                        "a rule names the pure corpus {corpus}, which does not exist"
                    ));
                }
                Check::Tools(kind) if !crate::selfcheck::KINDS.contains(kind) => {
                    problems.push(format!(
                        "a rule names the self-check {kind}, which does not exist"
                    ));
                }
                _ => {}
            }
        }
    }
    let uncovered: Vec<&str> = smokes
        .iter()
        .filter(|smoke| !topics.iter().any(|topic| map::covers(topic, smoke)))
        .map(String::as_str)
        .collect();
    if !uncovered.is_empty() {
        problems.push(format!(
            "no rule's topic covers the smoke checks {}",
            uncovered.join(", ")
        ));
    }
    if problems.is_empty() {
        println!(
            "The path map covers {} tracked files and calls for {} smoke checks.",
            tracked.lines().count(),
            smokes.len()
        );
        return Ok(());
    }
    Err(format!(
        "the path map (tools/xtask/src/verify/map.rs) is incomplete:\n  {}",
        problems.join("\n  ")
    ))
}
