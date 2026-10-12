//! `cargo xtask verify`: the checks a change calls for, chosen by the
//! paths it touches ([`map::MAP`]), at one of three levels:
//!
//! - `quick` (under 2 minutes): formatting, the drift checks, the shaders
//!   where a rule calls for them, Clippy over the changed packages, and the
//!   pure corpora the rules call for;
//! - `pr` (under 10 minutes, the default): all of lint, the smoke checks the
//!   rules call for against a server of this checkout, the terrain corpus's
//!   jobs in Wasm where called for, equivalence's smoke tier and the cells
//!   the rules name, natively, where any rule calls for equivalence, and
//!   the factory tools' self-checks where called for;
//! - `merge` (about 30-45 minutes): all of lint, every smoke check, the
//!   native smoke again with its program built in the dev profile (where
//!   debug assertions check the engine's invariants, such as the terrain
//!   store's), the terrain corpus's jobs in Wasm, equivalence's full tier
//!   natively and, where a rule calls for it, its browser leg, and the
//!   self-checks called for.
//!
//! The change is the working tree, untracked files included, against the
//! base: its merge-base with origin/main, or with `--base REV`, which
//! equivalence compares with too. A changed file no rule covers calls for
//! everything. `--plan` prints the steps without running them. Every step
//! runs whatever failed before it; the run ends with each step's result and
//! time, and exits 3 when a step found a fault, 1 when one could not check.

mod map;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::Failure;
use crate::measure::git;
use map::Check;

const USAGE: &str = "usage: cargo xtask verify [--level quick|pr|merge] [--base REV] [--plan]";

/// The lint stages `quick` runs: the policy, the web host's text checks and
/// the drift checks, all of them seconds.
const QUICK_STAGES: &[&str] = &[
    "policy",
    "fmt",
    "web layers",
    "web errors",
    "web parameters",
    "scene sources",
    "generated files",
    "bench",
    "equivalence matrix",
    "coverage",
    "smoke descriptions",
    "path map",
    "features",
    "schedule ambiguities",
    "exports",
    "structure",
    "content locality",
];

/// How much a run checks.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Level {
    Quick,
    Pr,
    Merge,
}

/// What the changed files call for, gathered from the rules they match.
#[derive(Default)]
struct Selection {
    everything: bool,
    corpora: BTreeSet<&'static str>,
    pure_wasm: bool,
    shaders: bool,
    smokes: BTreeSet<String>,
    all_smokes: bool,
    cells: BTreeSet<&'static str>,
    equivalence: bool,
    browser: bool,
    /// The self-checks of the factory's tools called for.
    self_checks: BTreeSet<&'static str>,
    /// The Cargo packages holding changed files, for Clippy.
    packages: BTreeSet<String>,
    /// Changed files no rule covers: they call for everything.
    unmapped: Vec<String>,
}

impl Selection {
    /// Adds what `path`'s change calls for.
    fn add(&mut self, root: &Path, path: &str) {
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
                Check::Smoke(name) => {
                    self.smokes.insert(name.to_owned());
                }
                Check::ItsSmoke => {
                    let name = path
                        .rsplit('/')
                        .next()
                        .and_then(|file| file.strip_suffix("-smoke.mjs"));
                    self.smokes.extend(name.map(str::to_owned));
                }
                Check::AllSmokes => self.all_smokes = true,
                Check::Cells(cells) => self.cells.extend(cells),
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

/// The Cargo package whose folder holds `path`: the nearest folder above it
/// with a manifest naming a package.
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

/// One step of a run: its name and what it does.
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
    // The change is what this checkout did since it parted from the base.
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
    let mut selection = Selection::default();
    for path in &changed {
        selection.add(root, path);
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

/// The steps of `level` for what `selection` calls for, comparing with
/// `base`.
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
            steps.push(Step::new("gpu", move || Ok(crate::gpu::check(root)?)));
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
        Ok(crate::lint::run_stages(root, crate::lint::LINT)?)
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
    // The engine's debug assertions run nowhere else with a GPU: the
    // native smoke's program built in the dev profile.
    if level == Level::Merge {
        let options = vec![
            "--native-profile".to_owned(),
            "dev".to_owned(),
            "native".to_owned(),
        ];
        steps.push(Step::new(
            format!("smoke {}", options.join(" ")),
            move || crate::smoke::smoke(root, &options),
        ));
    }
    if level == Level::Merge || everything || selection.pure_wasm {
        steps.push(Step::new("pure --wasm terrain", move || {
            crate::pure::pure(root, &["--wasm".to_owned(), "terrain".to_owned()])
        }));
    }
    let tiers = crate::equivalence::tiers(root)?;
    let wanted_equivalence = everything || selection.equivalence || !selection.cells.is_empty();
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
            .filter(|(_, tier)| tier == "smoke")
            .map(|(name, _)| name.as_str())
            .collect();
        cells.extend(selection.cells.iter().copied());
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

/// Runs every step, each whatever failed before it, then lists each
/// result and time: fails with 3 where a step found a fault, else 1 where
/// one could not check.
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

/// Fails unless every tracked file matches a rule of the path map, and
/// every smoke check, equivalence cell and pure corpus the rules name
/// exists and every smoke check and cell beyond equivalence's smoke tier
/// is named by some rule: a file or a check nobody placed.
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
    let smokes: BTreeSet<String> = crate::smoke::scripts(root)?
        .into_iter()
        .map(|(name, _, _)| name)
        .collect();
    let tiers: BTreeMap<String, String> = crate::equivalence::tiers(root)?.into_iter().collect();
    let corpora: Vec<&str> = crate::pure::CORPORA
        .iter()
        .map(|corpus| corpus.name)
        .collect();
    let mut named_smokes = BTreeSet::new();
    let mut named_cells = BTreeSet::new();
    for rule in map::MAP {
        for check in rule.checks {
            match check {
                Check::Smoke(name) => {
                    if !smokes.contains(*name) {
                        problems.push(format!(
                            "a rule names the smoke check {name}, which does not exist"
                        ));
                    }
                    named_smokes.insert((*name).to_owned());
                }
                Check::Cells(cells) => {
                    for cell in *cells {
                        if !tiers.contains_key(*cell) {
                            problems.push(format!(
                                "a rule names the equivalence cell {cell}, which does not exist"
                            ));
                        }
                        named_cells.insert((*cell).to_owned());
                    }
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
    let unnamed: Vec<&str> = smokes
        .iter()
        .filter(|smoke| !named_smokes.contains(*smoke))
        .map(String::as_str)
        .collect();
    if !unnamed.is_empty() {
        problems.push(format!(
            "no rule calls for the smoke checks {}",
            unnamed.join(", ")
        ));
    }
    let unnamed: Vec<&str> = tiers
        .iter()
        .filter(|(cell, tier)| *tier != "smoke" && !named_cells.contains(*cell))
        .map(|(cell, _)| cell.as_str())
        .collect();
    if !unnamed.is_empty() {
        problems.push(format!(
            "no rule calls for the equivalence cells {}",
            unnamed.join(", ")
        ));
    }
    if problems.is_empty() {
        println!(
            "The path map covers {} tracked files and calls for {} smoke checks and {} cells.",
            tracked.lines().count(),
            smokes.len(),
            tiers.len()
        );
        return Ok(());
    }
    Err(format!(
        "the path map (tools/xtask/src/verify/map.rs) is incomplete:\n  {}",
        problems.join("\n  ")
    ))
}
