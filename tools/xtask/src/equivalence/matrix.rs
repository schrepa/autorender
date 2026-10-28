use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::Value;

use crate::measure::read_json;

pub(super) const MATRIX: &str = "bench/equivalence.json";

const CAPTURES: &str = "exact, cpu:ALPHA, gpu:ALPHA or display:FRAMES";

pub(super) enum Workload {
    Scene(String),
    Source(String),
}

pub(super) struct View {
    pub(super) name: String,
    pub(super) camera: Option<Value>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Capture {
    Exact,
    Cpu(f64),
    Gpu(f64),
    Display(u32),
}

impl Capture {
    fn parse(text: &str) -> Option<Self> {
        let alpha = |value: &str| {
            value
                .parse::<f64>()
                .ok()
                .filter(|alpha| (0.0..=1.0).contains(alpha))
        };
        match text.split_once(':') {
            None if text == "exact" => Some(Self::Exact),
            Some(("cpu", value)) => alpha(value).map(Self::Cpu),
            Some(("gpu", value)) => alpha(value).map(Self::Gpu),
            Some(("display", value)) => value
                .parse()
                .ok()
                .filter(|frames| *frames > 0)
                .map(Self::Display),
            _ => None,
        }
    }

    pub(super) fn label(self) -> String {
        match self {
            Self::Exact => "exact".into(),
            Self::Cpu(alpha) => format!("cpu:{alpha}"),
            Self::Gpu(alpha) => format!("gpu:{alpha}"),
            Self::Display(frames) => format!("display:{frames}"),
        }
    }
}

pub(super) struct Cell {
    pub(super) name: String,
    pub(super) tier: String,
    pub(super) workload: Workload,
    pub(super) options: Vec<(String, String)>,
    pub(super) environment: Option<Value>,
    pub(super) ticks: u32,
    pub(super) views: Vec<View>,
    pub(super) captures: Vec<Capture>,
    pub(super) tolerance: u8,
    pub(super) unstable: BTreeMap<String, String>,
}

impl Cell {
    pub(super) fn keys(&self) -> Vec<String> {
        self.views
            .iter()
            .flat_map(|view| {
                self.captures
                    .iter()
                    .map(move |capture| format!("{} {}", view.name, capture.label()))
            })
            .collect()
    }
}

const FIELDS: &[&str] = &[
    "name",
    "tier",
    "note",
    "scene",
    "source",
    "options",
    "environment",
    "ticks",
    "views",
    "captures",
    "tolerance",
    "unstable",
];

struct Known<'a> {
    root: &'a Path,
    views: Vec<String>,
    switches: Vec<Value>,
}

pub(super) fn read_matrix(root: &Path) -> Result<(Vec<Cell>, Vec<String>), String> {
    let matrix = read_json(&root.join(MATRIX))?;
    let known = Known {
        root,
        views: crate::bench::stress_choices(root, "views")?,
        switches: read_json(&root.join("bench/options.json"))?["options"]
            .as_array()
            .cloned()
            .unwrap_or_default(),
    };
    let mut problems = Vec::new();
    if matrix["note"].as_str().is_none_or(str::is_empty) {
        problems.push(format!("{MATRIX}: a note says what it is"));
    }
    for field in matrix
        .as_object()
        .into_iter()
        .flat_map(|fields| fields.keys())
    {
        if !["note", "not_toggled", "cells"].contains(&field.as_str()) {
            problems.push(format!(
                "{MATRIX}: {field} is not a field (note, not_toggled, cells)"
            ));
        }
    }
    let mut names = BTreeSet::new();
    let mut cells = Vec::new();
    for (index, entry) in matrix["cells"].as_array().into_iter().flatten().enumerate() {
        let mut found = Vec::new();
        let cell = cell(entry, &known, &mut found);
        if !names.insert(cell.name.clone()) {
            found.push("a name the matrix gives no other cell".into());
        }
        let at = format!("{MATRIX} cells[{index}] {}", cell.name);
        problems.extend(found.into_iter().map(|what| format!("{at}: {what}")));
        cells.push(cell);
    }
    if cells.is_empty() {
        problems.push(format!("{MATRIX}: cells lists the matrix's cells"));
    }
    problems.extend(section_problems(&matrix, &cells)?);
    Ok((cells, problems))
}

fn section_problems(matrix: &Value, cells: &[Cell]) -> Result<Vec<String>, String> {
    let schema: Value = serde_json::from_str(&engine_scene::SceneDocument::schema_json())
        .map_err(|error| format!("the scene schema is not JSON: {error}"))?;
    let environment: BTreeSet<&str> = schema["$defs"]["SceneEnvironment"]["properties"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(key, _)| key.as_str())
        .collect();
    let mut problems = Vec::new();
    let mut toggled = BTreeSet::new();
    for cell in cells {
        for key in cell
            .environment
            .as_ref()
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
            .map(|(key, _)| key.as_str())
        {
            if !environment.contains(key) {
                problems.push(format!(
                    "{MATRIX} {}: environment names {key}, which scene environments do not hold",
                    cell.name
                ));
            }
            toggled.insert(key);
        }
    }
    let sections: BTreeSet<&str> = crate::coverage::environment_sections()
        .iter()
        .map(|(key, _)| *key)
        .collect();
    let not_toggled = matrix["not_toggled"].as_object();
    for (key, reason) in not_toggled.into_iter().flatten() {
        if !sections.contains(key.as_str()) {
            problems.push(format!(
                "{MATRIX}: not_toggled names {key}, which is no environment section"
            ));
        } else if toggled.contains(key.as_str()) {
            problems.push(format!(
                "{MATRIX}: not_toggled names {key}, which a cell's environment switches"
            ));
        }
        if reason
            .as_str()
            .is_none_or(|reason| reason.trim().is_empty())
        {
            problems.push(format!(
                "{MATRIX}: not_toggled {key} carries a written reason"
            ));
        }
    }
    for section in sections {
        if !toggled.contains(section)
            && !not_toggled.is_some_and(|listed| listed.contains_key(section))
        {
            problems.push(format!(
                "{MATRIX}: no cell's environment switches the {section} section: add a variant, or list it under not_toggled with why"
            ));
        }
    }
    Ok(problems)
}

fn cell(entry: &Value, known: &Known, problems: &mut Vec<String>) -> Cell {
    let name = entry["name"].as_str().unwrap_or_default().to_owned();
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        problems.push("a name of lowercase letters, digits and dashes".into());
    }
    for field in entry
        .as_object()
        .into_iter()
        .flat_map(|fields| fields.keys())
    {
        if !FIELDS.contains(&field.as_str()) {
            problems.push(format!("{field} is not a field ({})", FIELDS.join(", ")));
        }
    }
    let tier = entry["tier"].as_str().unwrap_or_default().to_owned();
    if !["smoke", "full"].contains(&tier.as_str()) {
        problems.push("tier is smoke or full".into());
    }
    if entry["note"].as_str().is_none_or(str::is_empty) {
        problems.push("a note says what the cell covers".into());
    }
    let workload = workload(entry, known.root, problems);
    let environment = match &entry["environment"] {
        Value::Null => None,
        Value::Object(sections) if !sections.is_empty() => Some(entry["environment"].clone()),
        _ => {
            problems.push("environment holds sections by name".into());
            None
        }
    };
    let views = views(entry, &workload, known, problems);
    let cell = Cell {
        name,
        tier,
        options: options(entry, &known.switches, problems),
        environment,
        ticks: number(entry, "ticks", "a count", problems),
        captures: captures(entry, problems),
        tolerance: number(
            entry,
            "tolerance",
            "a channel difference from 0 to 255",
            problems,
        ),
        unstable: entry["unstable"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(key, reason)| (key.clone(), reason.as_str().unwrap_or_default().to_owned()))
            .collect(),
        workload,
        views,
    };
    let keys = cell.keys();
    for (key, reason) in &cell.unstable {
        let digest_part = key.split_once(" digest:").is_some_and(|(view, part)| {
            cell.views.iter().any(|known| known.name == view)
                && (part == "entities"
                    || part
                        .strip_prefix("components.")
                        .is_some_and(|name| !name.is_empty())
                    || part
                        .strip_prefix("assets.")
                        .is_some_and(|kind| !kind.is_empty()))
        });
        if !keys.contains(key) && !digest_part {
            problems.push(format!(
                "unstable names {key}, which is no capture of the cell (VIEW CAPTURE: {}) or part of a view's digest (VIEW digest:entities, VIEW digest:components.NAME, VIEW digest:assets.KIND)",
                keys.join(", ")
            ));
        }
        if reason.trim().is_empty() {
            problems.push(format!(
                "unstable capture {key} carries a written reason why it varies"
            ));
        }
    }
    cell
}

fn workload(entry: &Value, root: &Path, problems: &mut Vec<String>) -> Workload {
    match (entry["scene"].as_str(), entry["source"].as_str()) {
        (Some(scene), None) => {
            if !root.join(format!("bench/scenes/{scene}.json")).is_file() {
                problems.push(format!("bench/scenes/{scene}.json does not exist"));
            }
            Workload::Scene(scene.to_owned())
        }
        (None, Some(source)) => {
            if !(source.starts_with("bench/") || source.starts_with("web/scripts/fixtures/"))
                || !root.join(source).is_file()
            {
                problems.push(format!(
                    "source {source} is a scene file under bench/ or web/scripts/fixtures/"
                ));
            }
            Workload::Source(source.to_owned())
        }
        _ => {
            problems.push("names a bench scene (scene) or an authored scene file (source)".into());
            Workload::Scene(String::new())
        }
    }
}

fn options(entry: &Value, switches: &[Value], problems: &mut Vec<String>) -> Vec<(String, String)> {
    let mut options = Vec::new();
    for (option, value) in entry["options"].as_object().into_iter().flatten() {
        let text = match value {
            Value::String(text) => text.clone(),
            Value::Bool(true) => "on".into(),
            Value::Bool(false) => "off".into(),
            other => other.to_string(),
        };
        match switches.iter().find(|switch| switch["name"] == *option) {
            Some(switch)
                if !switch["values"]
                    .as_array()
                    .is_some_and(|values| values.iter().any(|known| *known == *text)) =>
            {
                problems.push(format!(
                    "option {option} is one of bench/options.json's values, not {text}"
                ));
            }
            Some(_) => {}
            None if option == "config" && value.is_object() => {}
            None => problems.push(format!(
                "option {option} is a switch of bench/options.json, or config"
            )),
        }
        options.push((option.clone(), text));
    }
    options
}

fn views(
    entry: &Value,
    workload: &Workload,
    known: &Known,
    problems: &mut Vec<String>,
) -> Vec<View> {
    let mut views = Vec::new();
    for view in entry["views"].as_array().into_iter().flatten() {
        match view {
            Value::String(name) => {
                if name != "scene"
                    && (matches!(workload, Workload::Source(_)) || !known.views.contains(name))
                {
                    problems.push(format!(
                        "view {name} is scene, or for bench scenes one of {}",
                        known.views.join(", ")
                    ));
                }
                views.push(View {
                    name: name.clone(),
                    camera: None,
                });
            }
            Value::Object(fields)
                if fields.get("name").and_then(Value::as_str).is_some()
                    && fields.get("camera").is_some_and(Value::is_object) =>
            {
                views.push(View {
                    name: fields["name"].as_str().unwrap_or_default().to_owned(),
                    camera: Some(fields["camera"].clone()),
                });
            }
            _ => {
                problems.push("a view is scene, a Stress Lab view's name, or {name, camera}".into())
            }
        }
    }
    let names: BTreeSet<&str> = views.iter().map(|view| view.name.as_str()).collect();
    if views.is_empty() || names.len() != views.len() {
        problems.push("views lists at least one view, each named once".into());
    }
    views
}

fn captures(entry: &Value, problems: &mut Vec<String>) -> Vec<Capture> {
    let mut captures = Vec::new();
    for capture in entry["captures"].as_array().into_iter().flatten() {
        match capture.as_str().and_then(Capture::parse) {
            Some(parsed) if !captures.contains(&parsed) => captures.push(parsed),
            _ => problems.push(format!("capture {capture} is one of {CAPTURES}, once")),
        }
    }
    if captures.is_empty() {
        problems.push("captures lists at least one capture".into());
    }
    captures
}

fn number<T: TryFrom<u64> + Default>(
    entry: &Value,
    field: &str,
    what: &str,
    problems: &mut Vec<String>,
) -> T {
    match &entry[field] {
        Value::Null => T::default(),
        value => value
            .as_u64()
            .and_then(|number| T::try_from(number).ok())
            .unwrap_or_else(|| {
                problems.push(format!("{field} is {what}"));
                T::default()
            }),
    }
}

pub(crate) fn check(root: &Path) -> Result<(), String> {
    let (cells, problems) = read_matrix(root)?;
    if !problems.is_empty() {
        return Err(format!(
            "{MATRIX} is not a valid matrix:\n  {}",
            problems.join("\n  ")
        ));
    }
    let captures: usize = cells.iter().map(|cell| cell.keys().len()).sum();
    println!(
        "{MATRIX} holds {} cells ({} smoke) of {captures} captures.",
        cells.len(),
        cells.iter().filter(|cell| cell.tier == "smoke").count()
    );
    Ok(())
}
