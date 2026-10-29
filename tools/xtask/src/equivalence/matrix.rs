use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::Value;

use crate::measure::read_json;

pub(super) const MATRIX: &str = "bench/equivalence.json";

const CAPTURES: &str = "exact, gpu:ALPHA or display:FRAMES";

pub(super) struct View {
    pub(super) name: String,
    pub(super) camera: Option<Value>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Capture {
    Exact,
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
            Self::Gpu(alpha) => format!("gpu:{alpha}"),
            Self::Display(frames) => format!("display:{frames}"),
        }
    }
}

pub(super) struct Cell {
    pub(super) name: String,
    pub(super) tier: String,
    pub(super) scene: String,
    pub(super) project: String,
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
    "scene",
    "options",
    "environment",
    "ticks",
    "views",
    "captures",
    "tolerance",
    "unstable",
];

pub(super) fn read_matrix(root: &Path) -> Result<(Vec<Cell>, Vec<String>), String> {
    let matrix = read_json(&root.join(MATRIX))?;
    let environment = environment_sections()?;
    let mut problems = Vec::new();
    for field in matrix
        .as_object()
        .into_iter()
        .flat_map(|fields| fields.keys())
    {
        if field != "cells" {
            problems.push(format!("{MATRIX}: {field} is not a field (cells)"));
        }
    }
    let mut names = BTreeSet::new();
    let mut cells = Vec::new();
    for (index, entry) in matrix["cells"].as_array().into_iter().flatten().enumerate() {
        let mut found = Vec::new();
        let cell = cell(entry, root, &environment, &mut found);
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
    Ok((cells, problems))
}

fn environment_sections() -> Result<BTreeSet<String>, String> {
    let engine = crate::bricks::standard_engine();
    let registry = engine
        .app
        .world()
        .resource::<engine::bricks::BrickRegistry>();
    let schema: Value = serde_json::from_str(&engine_scene::SceneDocument::schema_json(registry))
        .map_err(|error| format!("the scene schema is not JSON: {error}"))?;
    Ok(schema["$defs"]["SceneEnvironment"]["properties"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(key, _)| key.clone())
        .collect())
}

fn cell(
    entry: &Value,
    root: &Path,
    sections: &BTreeSet<String>,
    problems: &mut Vec<String>,
) -> Cell {
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
    let (scene, project) = scene(entry, root, problems);
    let environment = match &entry["environment"] {
        Value::Null => None,
        Value::Object(fields) if !fields.is_empty() => {
            for key in fields.keys().filter(|key| !sections.contains(*key)) {
                problems.push(format!(
                    "environment names {key}, which scene environments do not hold"
                ));
            }
            Some(entry["environment"].clone())
        }
        _ => {
            problems.push("environment holds sections by name".into());
            None
        }
    };
    let views = views(entry, problems);
    let cell = Cell {
        name,
        tier,
        scene,
        project,
        options: options(entry, problems),
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

fn scene(entry: &Value, root: &Path, problems: &mut Vec<String>) -> (String, String) {
    let Some(scene) = entry["scene"].as_str() else {
        problems.push("scene names a bench scene (bench/scenes/NAME.json)".into());
        return (String::new(), String::new());
    };
    match crate::bench::Scene::read(root, scene) {
        Ok(read) => (scene.to_owned(), read.project),
        Err(error) => {
            problems.push(error);
            (scene.to_owned(), String::new())
        }
    }
}

fn options(entry: &Value, problems: &mut Vec<String>) -> Vec<(String, String)> {
    let mut options = Vec::new();
    for (option, value) in entry["options"].as_object().into_iter().flatten() {
        let text = engine_checks::bench::switch(value);
        if let Some(values) = engine_checks::bench::setting_values(option)
            && !values.contains(&text.as_str())
        {
            problems.push(format!(
                "option {option} is one of {}, not {text}",
                values.join(", ")
            ));
        }
        options.push((option.clone(), text));
    }
    options
}

fn views(entry: &Value, problems: &mut Vec<String>) -> Vec<View> {
    let mut views = Vec::new();
    for view in entry["views"].as_array().into_iter().flatten() {
        match view {
            Value::String(name) if !name.is_empty() => {
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
            _ => problems.push("a view is scene, the game's view name, or {name, camera}".into()),
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
    let record = serde_json::json!({
        "event": "matrix_holds",
        "file": MATRIX,
        "cells": cells.len(),
        "smoke_cells": cells.iter().filter(|cell| cell.tier == "smoke").count(),
        "captures": captures,
    });
    println!("{record}");
    Ok(())
}
