use std::collections::{BTreeMap, BTreeSet};

use engine_render::linker::unmangle;
use naga::Span;

use crate::gpu::{Checked, plain, wgsl_spans};

const AHEAD: usize = 8;

type Kept = Vec<(String, usize)>;

fn normal(token: &str) -> Option<String> {
    match token {
        "," | ";" | "(" | ")" | ":" => None,
        _ if token.starts_with(|first: char| first.is_ascii_digit())
            || (token.starts_with('.') && token.len() > 1) =>
        {
            Some("#".to_owned())
        }
        _ => Some(plain(token)),
    }
}

fn index(text: &str) -> (Kept, Vec<(String, usize, usize)>) {
    let raw = wgsl_spans(text);
    let mut kept = Vec::new();
    let mut functions = Vec::new();
    let mut depth = 0usize;
    let mut open: Option<(String, usize)> = None;
    for (index, (token, at)) in raw.iter().enumerate() {
        if depth == 0
            && token == "fn"
            && open.is_none()
            && let Some((name, _)) = raw.get(index + 1)
        {
            open = Some((name.clone(), kept.len()));
        }
        if let Some(normalized) = normal(token) {
            kept.push((normalized, *at));
        }
        match token.as_str() {
            "{" => depth += 1,
            "}" => {
                depth = depth.saturating_sub(1);
                if depth == 0
                    && let Some((name, start)) = open.take()
                {
                    functions.push((name, start, kept.len()));
                }
            }
            _ => {}
        }
    }
    (kept, functions)
}

fn align(composed: &[&str], source: &[&str]) -> Vec<usize> {
    let mut map = Vec::with_capacity(composed.len());
    let last = source.len().saturating_sub(1);
    let (mut i, mut j) = (0, 0);
    while i < composed.len() {
        if j >= source.len() {
            map.push(last);
            i += 1;
            continue;
        }
        if composed[i] == source[j] || composed.get(i + 1) == source.get(j + 1) {
            map.push(j);
            i += 1;
            j += 1;
            continue;
        }
        let skip = (1..=AHEAD).find_map(|ahead| {
            if composed.get(i + ahead) == Some(&source[j]) {
                Some((ahead, 0))
            } else if source.get(j + ahead) == Some(&composed[i]) {
                Some((0, ahead))
            } else {
                None
            }
        });
        match skip {
            Some((ahead, 0)) => {
                map.extend(std::iter::repeat_n(j, ahead));
                i += ahead;
            }
            Some((_, ahead)) => j += ahead,
            None => {
                map.push(j);
                i += 1;
                j += 1;
            }
        }
    }
    map
}

struct Source {
    kept: Kept,
    functions: BTreeMap<String, (usize, usize)>,
    lines: Vec<usize>,
}

impl Source {
    fn line(&self, offset: usize) -> usize {
        self.lines.partition_point(|&start| start <= offset)
    }
}

pub(super) struct Sources {
    modules: BTreeMap<String, Source>,
}

impl Sources {
    pub(super) fn new(modules: &[Checked]) -> Self {
        let modules = modules
            .iter()
            .map(|checked| {
                let (kept, functions) = index(&checked.source);
                let lines = std::iter::once(0)
                    .chain(checked.source.match_indices('\n').map(|(at, _)| at + 1))
                    .collect();
                let functions = functions
                    .into_iter()
                    .map(|(name, start, end)| (name, (start, end)))
                    .collect();
                (
                    checked.name.clone(),
                    Source {
                        kept,
                        functions,
                        lines,
                    },
                )
            })
            .collect();
        Self { modules }
    }
}

struct Mapped {
    start: usize,
    end: usize,
    home: String,
    starts: Vec<usize>,
    lines: Vec<usize>,
}

pub(super) struct Layout {
    module: String,
    functions: Vec<Mapped>,
}

impl Layout {
    pub(super) fn new(checked: &Checked, sources: &Sources) -> Self {
        let (kept, functions) = index(&checked.text);
        let mut mapped = Vec::new();
        for (name, start, end) in functions {
            if start >= end {
                continue;
            }
            let (home, item) = unmangle(&name).unwrap_or_else(|| (checked.name.clone(), name));
            let Some(source) = sources.modules.get(&home) else {
                continue;
            };
            let Some(&(from, to)) = source.functions.get(&item).filter(|(from, to)| from < to)
            else {
                continue;
            };
            let composed: Vec<&str> = kept[start..end]
                .iter()
                .map(|(token, _)| token.as_str())
                .collect();
            let original: Vec<&str> = source.kept[from..to]
                .iter()
                .map(|(token, _)| token.as_str())
                .collect();
            let lines = align(&composed, &original)
                .into_iter()
                .map(|at| source.line(source.kept[from + at].1))
                .collect();
            mapped.push(Mapped {
                start: kept[start].1,
                end: kept[end - 1].1 + 1,
                home,
                starts: kept[start..end].iter().map(|(_, at)| *at).collect(),
                lines,
            });
        }
        Self {
            module: checked.name.clone(),
            functions: mapped,
        }
    }

    fn locate(&self, offset: usize) -> Option<(&str, usize)> {
        let function = self
            .functions
            .iter()
            .find(|function| function.start <= offset && offset < function.end)?;
        let index = function
            .starts
            .partition_point(|&start| start <= offset)
            .checked_sub(1)?;
        Some((function.home.as_str(), function.lines[index]))
    }

    pub(super) fn at(&self, span: Span) -> (String, usize) {
        span.to_range()
            .and_then(|range| self.locate(range.start))
            .map_or_else(
                || (self.module.clone(), 0),
                |(home, line)| (home.to_owned(), line),
            )
    }

    pub(super) fn marked(&self, span: Span, marks: &BTreeSet<(String, usize)>) -> bool {
        let Some(range) = span.to_range() else {
            return false;
        };
        let Some((home, first)) = self.locate(range.start) else {
            return false;
        };
        let last = self
            .locate(range.end.max(range.start + 1) - 1)
            .filter(|(other, _)| *other == home)
            .map_or(first, |(_, line)| line.max(first));
        marks
            .range((home.to_owned(), first)..=(home.to_owned(), last))
            .next()
            .is_some()
    }
}

pub(super) struct Marks {
    pub lines: BTreeSet<(String, usize)>,
    pub unexplained: Vec<String>,
}

pub(super) fn marks(modules: &[Checked], mark: &str, says: &str) -> Marks {
    let mut lines = BTreeSet::new();
    let mut unexplained = Vec::new();
    for checked in modules {
        for (index, line) in checked.source.lines().enumerate() {
            let Some(found) = line.find(mark) else {
                continue;
            };
            lines.insert((checked.name.clone(), index + 1));
            if line[found + mark.len()..].trim().is_empty() {
                unexplained.push(format!(
                    "{}:{}: marks a line `{mark}` without saying {says}",
                    checked.name,
                    index + 1
                ));
            }
        }
    }
    unexplained.sort();
    unexplained.dedup();
    Marks { lines, unexplained }
}

pub(super) fn unmangled(module: &naga::Module) -> naga::Module {
    let mut module = module.clone();
    for (_, function) in module.functions.iter_mut() {
        if let Some(name) = function.name.as_mut() {
            *name = plain(name);
        }
    }
    for (_, global) in module.global_variables.iter_mut() {
        if let Some(name) = global.name.as_mut() {
            *name = plain(name);
        }
    }
    module
}
