use std::fs;
use std::path::Path;

use engine_checks::bench::switch;
use serde_json::{Map, Value};

pub fn bench(root: &Path, options: &[String]) -> Result<(), String> {
    match options.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        ["check"] => engine_checks::bench::check(crate::rules::repository(root)?),
        _ => Err("usage: cargo xtask bench check".into()),
    }
}

pub struct Scene {
    pub project: String,
    pub options: Map<String, Value>,
}

impl Scene {
    pub fn read(root: &Path, name: &str) -> Result<Self, String> {
        let path = format!("bench/scenes/{name}.json");
        let text = fs::read_to_string(root.join(&path))
            .map_err(|error| format!("cannot read {path}: {error}"))?;
        Self::parse(&text, &path)
    }

    pub fn at(root: &Path, sha: &str, name: &str) -> Result<Self, String> {
        let path = format!("bench/scenes/{name}.json");
        let text = crate::measure::git(root, &["show", &format!("{sha}:{path}")])
            .map_err(|_| format!("{path} is not at {sha}"))?;
        Self::parse(&text, &format!("{sha}:{path}"))
    }

    fn parse(text: &str, at: &str) -> Result<Self, String> {
        let file: Value = serde_json::from_str(text).map_err(|error| format!("{at}: {error}"))?;
        let project = file["project"]
            .as_str()
            .filter(|project| !project.is_empty())
            .ok_or_else(|| format!("{at}: project names a game crate's folder"))?;
        let options = file["options"]
            .as_object()
            .cloned()
            .ok_or_else(|| format!("{at}: options is an object of options by name"))?;
        Ok(Self {
            project: project.to_owned(),
            options,
        })
    }
}

pub fn arguments(options: &Map<String, Value>) -> Vec<String> {
    options
        .iter()
        .flat_map(|(option, value)| [format!("--{option}"), switch(value)])
        .collect()
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
