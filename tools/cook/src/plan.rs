use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::Kind;
use crate::cook::{Fault, fault};

pub struct Plan {
    pub(crate) sources: PathBuf,
    pub(crate) out: PathBuf,
    pub(crate) steps: Vec<Entry>,
}

pub(crate) struct Entry {
    pub(crate) kind: &'static Kind,
    pub(crate) name: String,
    pub(crate) params: Value,
    pub(crate) inputs: Vec<Wanted>,
}

pub(crate) enum Wanted {
    Source(String),
    File(String, String),
    Files(String),
}

pub struct Declared<'a> {
    entry: &'a mut Entry,
}

pub(crate) struct Graph {
    pub(crate) order: Vec<usize>,
    pub(crate) needs: Vec<Vec<usize>>,
    index: BTreeMap<String, usize>,
}

impl Plan {
    pub fn new(sources: &Path, out: &Path) -> Self {
        let (sources, out) = (sources.to_owned(), out.to_owned());
        Self {
            sources,
            out,
            steps: Vec::new(),
        }
    }

    pub fn step(&mut self, kind: &'static Kind, name: &str, params: Value) -> Declared<'_> {
        let (name, inputs) = (name.to_owned(), Vec::new());
        self.steps.push(Entry {
            kind,
            name,
            params,
            inputs,
        });
        let at = self.steps.len() - 1;
        let entry = &mut self.steps[at];
        Declared { entry }
    }

    pub(crate) fn graph(&self) -> Result<Graph, Vec<Fault>> {
        let (mut index, mut faults) = (BTreeMap::new(), Vec::new());
        for (at, name) in self.steps.iter().map(|entry| &entry.name).enumerate() {
            if index.insert(name.clone(), at).is_some() {
                faults.push(fault(name, "DuplicateStep", name, "two steps"));
            }
        }
        let mut needs = vec![Vec::new(); self.steps.len()];
        for (at, entry) in self.steps.iter().enumerate() {
            let name = &entry.name;
            for step in entry.inputs.iter().filter_map(Wanted::step) {
                match index.get(step) {
                    None => faults.push(fault(name, "UnknownStep", step, "no such step")),
                    Some(&from) if from == at => faults.push(fault(name, "OwnInput", step, "")),
                    Some(&from) => needs[at].push(from),
                }
            }
        }
        let order = order(&needs).map_err(|cycle| {
            let names: Vec<&str> = cycle
                .iter()
                .map(|&at| self.steps[at].name.as_str())
                .collect();
            fault(names[0], "Cycle", names[0], names.join(" -> "))
        });
        match order {
            Ok(order) if faults.is_empty() => Ok(Graph {
                order,
                needs,
                index,
            }),
            Ok(_) => Err(faults),
            Err(cycle) => Err(faults.into_iter().chain([cycle]).collect()),
        }
    }
}

fn order(needs: &[Vec<usize>]) -> Result<Vec<usize>, Vec<usize>> {
    let mut placed = vec![false; needs.len()];
    let mut order = Vec::new();
    while order.len() < needs.len() {
        let before = order.len();
        for at in 0..needs.len() {
            if !placed[at] && needs[at].iter().all(|&need| placed[need]) {
                placed[at] = true;
                order.push(at);
            }
        }
        if order.len() > before {
            continue;
        }
        let mut path = vec![(0..needs.len()).find(|&at| !placed[at]).unwrap_or(0)];
        while let Some(&next) = path
            .last()
            .and_then(|&at| needs[at].iter().find(|&&n| !placed[n]))
        {
            if let Some(start) = path.iter().position(|&at| at == next) {
                path.drain(..start);
                path.push(next);
                break;
            }
            path.push(next);
        }
        return Err(path);
    }
    Ok(order)
}

impl Graph {
    pub(crate) fn chosen(
        &self,
        targets: &[String],
        closed: bool,
    ) -> Result<BTreeSet<usize>, Vec<Fault>> {
        if targets.is_empty() {
            return Ok((0..self.needs.len()).collect());
        }
        let (mut todo, mut faults) = (Vec::new(), Vec::new());
        for target in targets {
            match self.index.get(target) {
                Some(&at) => todo.push(at),
                None => faults.push(fault("", "UnknownStep", target, "no such step")),
            }
        }
        let mut chosen = BTreeSet::new();
        while let Some(at) = todo.pop() {
            if chosen.insert(at) && closed {
                todo.extend(&self.needs[at]);
            }
        }
        faults.is_empty().then_some(chosen).ok_or(faults)
    }
}

impl Wanted {
    fn step(&self) -> Option<&str> {
        match self {
            Self::Source(_) => None,
            Self::File(step, _) | Self::Files(step) => Some(step),
        }
    }
}

impl Declared<'_> {
    pub fn source(&mut self, path: &str) -> &mut Self {
        self.entry.inputs.push(Wanted::Source(path.to_owned()));
        self
    }

    pub fn output(&mut self, step: &str, file: &str) -> &mut Self {
        let (step, file) = (step.to_owned(), file.to_owned());
        self.entry.inputs.push(Wanted::File(step, file));
        self
    }

    pub fn outputs(&mut self, step: &str) -> &mut Self {
        self.entry.inputs.push(Wanted::Files(step.to_owned()));
        self
    }
}
