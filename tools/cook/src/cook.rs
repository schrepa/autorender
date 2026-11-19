use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::fs;
use std::num::NonZero;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use serde_json::{Value, json};

use crate::Refusal;
use crate::plan::{Entry, Graph, Plan, Wanted};
use crate::step::{Resolved, Step};
use crate::store::{
    Kept, Made, Record, Sources, Store, canonical, local, made, removed, seen, sha,
};

pub(crate) struct Fault(String, Refusal);

pub(crate) fn fault(step: &str, code: &'static str, file: &str, detail: impl Display) -> Fault {
    Fault(step.to_owned(), Refusal::new(code, file, detail))
}

type Known = BTreeMap<String, Kept>;

struct Job {
    dir: PathBuf,
    resolved: Vec<Resolved>,
}

#[derive(Default)]
struct Tally {
    ran: usize,
    reused: usize,
    stale: Vec<String>,
}

struct Cook<'a> {
    plan: &'a Plan,
    graph: &'a Graph,
    store: &'a Store,
    seen: &'a Sources,
    last: &'a Record,
    verify: bool,
}

fn say(line: &Value) {
    println!("{line}");
}

fn faulted(faults: Vec<Fault>) -> Result<(), Vec<Fault>> {
    faults.is_empty().then_some(()).ok_or(faults)
}

pub(crate) fn main(plan: Plan) -> ExitCode {
    let started = Instant::now();
    say(&json!({"run": "started", "steps": plan.steps.len()}));
    let mut tally = Tally::default();
    let faults = cook(plan, &mut tally).err().unwrap_or_default();
    for Fault(step, no) in &faults {
        say(&json!({"error": no.code, "step": step, "file": no.file, "detail": no.detail}));
    }
    let Tally { ran, reused, stale } = tally;
    let (count, seconds) = (faults.len(), started.elapsed().as_secs_f64());
    let finished = json!({"run": "finished", "ran": ran, "reused": reused, "faults": count, "stale": stale, "seconds": seconds});
    say(&finished);
    ExitCode::from(u8::from(count > 0))
}

fn arguments(plan: &mut Plan) -> Result<(Vec<String>, bool), Vec<Fault>> {
    let (mut arguments, mut targets, mut verify) = (std::env::args().skip(1), Vec::new(), false);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--verify" => verify = true,
            "--out" => match arguments.next() {
                Some(out) => plan.out = PathBuf::from(out),
                None => return Err(vec![fault("", "Flag", "--out", "names no folder")]),
            },
            flag if flag.starts_with("--") => return Err(vec![fault("", "Flag", flag, "")]),
            _ => targets.push(argument),
        }
    }
    Ok((targets, verify))
}

fn cook(mut plan: Plan, tally: &mut Tally) -> Result<(), Vec<Fault>> {
    let one = |fault: Fault| vec![fault];
    let (targets, verify) = arguments(&mut plan)?;
    let graph = plan.graph()?;
    let chosen = graph.chosen(&targets, !verify)?;
    let (store, _lock) = Store::open(&plan.out).map_err(one)?;
    let last = store.record().map_err(one)?;
    faulted(store.unowned(&last))?;
    let seen = sources(&plan, &last.sources)?;
    let (plan, graph, store) = (&plan, &graph, &store);
    let cook = Cook {
        plan,
        graph,
        store,
        seen: &seen,
        last: &last,
        verify,
    };
    let (done, faults) = cook.all(&chosen, tally);
    faulted(faults)?;
    let now = sources(plan, &seen)?;
    let changed = seen
        .iter()
        .filter(|(name, fact)| now[*name].sha256 != fact.sha256);
    faulted(
        changed
            .map(|(name, _)| fault("", "SourceChanged", name, ""))
            .collect(),
    )?;
    if verify {
        return Ok(());
    }
    let (sources, steps) = (now, assemble(plan, &last, done)?);
    let next = Record { sources, steps };
    tally.stale = cook.stale(&chosen, &next);
    let changes = store.publish(&last, &next);
    say(&json!({"files": changes.map_err(|e| one(fault("", "Publish", "", e)))?}));
    Ok(())
}

fn sources(plan: &Plan, last: &Sources) -> Result<Sources, Vec<Fault>> {
    let out = fs::canonicalize(&plan.out).map_err(|e| vec![fault("", "Output", "", e)])?;
    let (mut found, mut faults) = (BTreeMap::new(), Vec::new());
    for entry in &plan.steps {
        let step = &entry.name;
        for wanted in &entry.inputs {
            let Wanted::Source(name) = wanted else {
                continue;
            };
            let path = local(&plan.sources, name);
            if fs::canonicalize(&path).is_ok_and(|real| real.starts_with(&out)) {
                faults.push(fault(step, "SourceInOutput", name, ""));
            } else if !found.contains_key(name) {
                match seen(&path, last.get(name)) {
                    Ok(fact) => drop(found.insert(name.clone(), fact)),
                    Err(e) => faults.push(fault(step, "MissingSource", name, e)),
                }
            }
        }
    }
    faulted(faults).map(|()| found)
}

fn key(kept: &Kept) -> String {
    sha(json!([kept.kind, kept.version, kept.params, kept.inputs])
        .to_string()
        .as_bytes())
}

fn reason(verify: bool, last: Option<&Kept>, next: &Kept) -> String {
    if verify {
        return "verify".to_owned();
    }
    let Some(last) = last.filter(|last| last.kind == next.kind) else {
        return "new".to_owned();
    };
    if last.version != next.version {
        return "version".to_owned();
    }
    if last.params != next.params {
        return "params".to_owned();
    }
    let (before, after) = (&last.inputs, &next.inputs);
    let at = (0..before.len().max(after.len())).find(|&at| before.get(at) != after.get(at));
    let input = at.and_then(|at| after.get(at).or(before.get(at)));
    input.map_or_else(|| "new".to_owned(), |(name, _)| name.clone())
}

fn assemble(plan: &Plan, last: &Record, mut done: Known) -> Result<Known, Vec<Fault>> {
    let mut steps = BTreeMap::new();
    for name in plan.steps.iter().map(|entry| &entry.name) {
        if let Some(kept) = done.remove(name).or_else(|| last.steps.get(name).cloned()) {
            steps.insert(name.clone(), kept);
        }
    }
    let (mut writers, mut items, mut faults) = (BTreeMap::new(), BTreeSet::new(), Vec::new());
    for (step, kept) in &steps {
        for (uri, _) in kept.made.files.iter().filter(|(_, fact)| fact.published) {
            if let Some(other) = writers.insert(uri, step) {
                faults.push(fault(step, "TwoWriters", uri, other));
            }
        }
        for (section, ids) in &kept.made.items {
            for id in ids.keys().filter(|id| !items.insert((section, *id))) {
                faults.push(fault(step, "DuplicateItem", id, section));
            }
        }
    }
    faulted(faults).map(|()| steps)
}

impl Cook<'_> {
    fn all(&self, chosen: &BTreeSet<usize>, tally: &mut Tally) -> (Known, Vec<Fault>) {
        let pool = thread::available_parallelism().map_or(1, NonZero::get);
        let mut waiting = self.graph.order.clone();
        waiting.retain(|at| chosen.contains(at));
        let (mut done, mut faults, mut running) = (BTreeMap::new(), Vec::new(), 0);
        let (send, receive) = mpsc::channel();
        thread::scope(|scope| {
            loop {
                let mut place = 0;
                while place < waiting.len() && running < pool {
                    let known = if self.verify { &self.last.steps } else { &done };
                    if !self.ready(waiting[place], known) {
                        place += 1;
                        continue;
                    }
                    let (at, started) = (waiting.remove(place), Instant::now());
                    let entry = &self.plan.steps[at];
                    match self.prepare(entry, known) {
                        Err(fault) => faults.push(fault),
                        Ok((kept, None)) => {
                            tally.reused += 1;
                            let seconds = started.elapsed().as_secs_f64();
                            say(&json!({"reused": entry.name, "seconds": seconds}));
                            done.insert(entry.name.clone(), kept);
                        }
                        Ok((kept, Some(job))) => {
                            running += 1;
                            let send = send.clone();
                            scope.spawn(move || {
                                let made = self.make(entry, kept, &job);
                                drop(send.send((at, started, made)));
                            });
                        }
                    }
                }
                if running == 0 {
                    break;
                }
                let Ok((at, started, made)) = receive.recv() else {
                    break;
                };
                running -= 1;
                let name = &self.plan.steps[at].name;
                let (kept, notes) = match made {
                    Err(fault) => {
                        faults.push(fault);
                        continue;
                    }
                    Ok(made) => made,
                };
                tally.ran += 1;
                let last = self.last.steps.get(name);
                let why = reason(self.verify, last, &kept);
                let seconds = started.elapsed().as_secs_f64();
                say(&json!({"ran": name, "reason": why, "notes": notes, "seconds": seconds}));
                done.insert(name.clone(), kept);
            }
        });
        (done, faults)
    }

    fn ready(&self, at: usize, known: &Known) -> bool {
        let made = |&need: &usize| known.contains_key(&self.plan.steps[need].name);
        self.verify || self.graph.needs[at].iter().all(made)
    }

    fn identity(&self, entry: &Entry, known: &Known) -> Result<(Kept, Vec<Resolved>), Fault> {
        let (name, mut resolved) = (&entry.name, Vec::new());
        let mut push = |name: &String, sha256: &String, path| {
            let (name, sha256) = (name.clone(), sha256.clone());
            resolved.push(Resolved { name, sha256, path });
        };
        for wanted in &entry.inputs {
            let (step, only) = match wanted {
                Wanted::Source(file) => {
                    let sha256 = &self.seen[file].sha256;
                    push(file, sha256, local(&self.plan.sources, file));
                    continue;
                }
                Wanted::File(step, file) => (step, Some(file)),
                Wanted::Files(step) => (step, None),
            };
            let kept = known
                .get(step)
                .ok_or_else(|| fault(name, "NotCooked", step, ""))?;
            let dir = self.store.staged(step, &kept.key, false).join("files");
            let mut picked: Vec<_> = kept.made.files.iter().collect();
            picked.retain(|(file, _)| only.is_none_or(|only| only == *file));
            if let (Some(file), true) = (only, picked.is_empty()) {
                return Err(fault(name, "UnknownFile", file, step));
            }
            for (file, fact) in picked {
                push(file, &fact.sha256, local(&dir, file));
            }
        }
        let mut names = BTreeSet::new();
        if let Some(twice) = resolved.iter().find(|input| !names.insert(&input.name)) {
            return Err(fault(name, "InputTwice", &twice.name, ""));
        }
        let mut kept = Kept {
            kind: entry.kind.name.to_owned(),
            version: entry.kind.version,
            params: sha(canonical(&entry.params).to_string().as_bytes()),
            inputs: resolved
                .iter()
                .map(|input| (input.name.clone(), input.sha256.clone()))
                .collect(),
            key: String::new(),
            made: Made::default(),
        };
        kept.key = key(&kept);
        Ok((kept, resolved))
    }

    fn prepare(&self, entry: &Entry, known: &Known) -> Result<(Kept, Option<Job>), Fault> {
        let (mut kept, resolved) = self.identity(entry, known)?;
        let dir = self.store.staged(&entry.name, &kept.key, self.verify);
        if !self.verify
            && let Some(made) = made(&dir)
        {
            kept.made = made;
            return Ok((kept, None));
        }
        Ok((kept, Some(Job { dir, resolved })))
    }

    fn make(&self, entry: &Entry, mut kept: Kept, job: &Job) -> Result<(Kept, Vec<Value>), Fault> {
        let (dir, resolved) = (&job.dir, &job.resolved);
        let fail = |refusal: Refusal| Fault(entry.name.clone(), refusal);
        let stage = |error: std::io::Error| fail(Refusal::new("Stage", &entry.name, error));
        removed(fs::remove_dir_all(dir)).map_err(stage)?;
        fs::create_dir_all(dir).map_err(stage)?;
        let read = |input: &Resolved| {
            fs::read(&input.path).map_err(|e| Refusal::new("Read", &input.name, e))
        };
        let bytes = resolved.iter().map(read).collect::<Result<Vec<_>, _>>();
        let bytes = bytes.map_err(fail)?;
        let mut step = Step::new(entry, resolved, &bytes, dir);
        let ran = (entry.kind.run)(&mut step);
        let (made, notes, refused) = step.finish();
        ran.and(refused.map_or(Ok(()), Err)).map_err(fail)?;
        kept.made = made;
        if self.verify {
            self.compare(&entry.name, &kept).map_err(fail)?;
        }
        let record = serde_json::to_vec(&kept.made).map_err(|e| stage(e.into()))?;
        fs::write(dir.join("made"), record).map_err(stage)?;
        Ok((kept, notes))
    }

    fn compare(&self, name: &str, fresh: &Kept) -> Result<(), Refusal> {
        let kept = self.last.steps.get(name);
        match kept.filter(|kept| kept.key == fresh.key) {
            None => Err(Refusal::new("Stale", name, "cook before verifying")),
            Some(kept) if kept.made != fresh.made => {
                Err(Refusal::new("VersionNotRaised", name, ""))
            }
            Some(_) => Ok(()),
        }
    }

    fn stale(&self, chosen: &BTreeSet<usize>, next: &Record) -> Vec<String> {
        let mut stale = BTreeSet::new();
        for &at in self.graph.order.iter().filter(|at| !chosen.contains(at)) {
            let entry = &self.plan.steps[at];
            let now = self.identity(entry, &next.steps).ok();
            let kept = next.steps.get(&entry.name);
            let current = kept
                .zip(now)
                .is_some_and(|(kept, (now, _))| kept.key == now.key);
            if !current || self.graph.needs[at].iter().any(|need| stale.contains(need)) {
                stale.insert(at);
            }
        }
        let names = stale.into_iter().map(|at| self.plan.steps[at].name.clone());
        names.collect()
    }
}
