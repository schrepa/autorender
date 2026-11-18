use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use serde_json::{Value, json};

use crate::fault::Fault;

fn strings<'a>(value: &'a Value, out: &mut Vec<&'a str>) {
    match value {
        Value::String(text) => out.push(text),
        Value::Array(items) => items.iter().for_each(|item| strings(item, out)),
        Value::Object(fields) => fields.values().for_each(|item| strings(item, out)),
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn named_path(text: &str) -> bool {
    let Some((stem, extension)) = text.rsplit_once('.') else {
        return false;
    };
    text.contains('/')
        && !text.contains("://")
        && !text.starts_with(['/', '$', '.'])
        && !stem.ends_with('/')
        && (1..=5).contains(&extension.len())
        && extension.bytes().all(|byte| byte.is_ascii_alphanumeric())
}

fn exact(folder: &Path, uri: &str) -> Result<(), String> {
    let mut at = folder.join(".");
    for part in Path::new(uri).components() {
        let Component::Normal(part) = part else {
            return Err("names a folder outside the scene's own".to_owned());
        };
        let wanted = part.to_string_lossy();
        let listed: BTreeSet<String> = std::fs::read_dir(&at)
            .map_err(|_| format!("`{}` is not a folder", at.display()))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        if !listed.contains(wanted.as_ref()) {
            let other = listed
                .iter()
                .find(|name| name.eq_ignore_ascii_case(&wanted));
            return Err(match other {
                Some(name) => format!("`{wanted}` is on disk as `{name}` (case differs)"),
                None => format!("`{wanted}` is not in `{}`", at.display()),
            });
        }
        at.push(part);
    }
    if at.is_file() {
        Ok(())
    } else {
        Err("names a folder, not a file".to_owned())
    }
}

pub fn check(folder: &Path, scenes: &[String]) -> (Value, Vec<Fault>) {
    let mut faults = Vec::new();
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    let mut queue: Vec<String> = scenes.to_vec();
    let mut read = BTreeSet::new();
    while let Some(file) = queue.pop() {
        if !read.insert(file.clone()) {
            continue;
        }
        let path = folder.join(&file);
        let shown = path.display().to_string();
        let value: Value = match std::fs::read_to_string(&path)
            .map_err(|error| error.to_string())
            .and_then(|text| serde_json::from_str(&text).map_err(|error| error.to_string()))
        {
            Ok(value) => value,
            Err(error) => {
                faults.push(Fault::new("UriUnreadable", &shown, "file", error));
                continue;
            }
        };
        let mut found = Vec::new();
        strings(&value, &mut found);
        for uri in found.into_iter().filter(|text| named_path(text)) {
            if seen.contains_key(uri) {
                continue;
            }
            seen.insert(uri.to_owned(), file.clone());
            match exact(folder, uri) {
                Ok(()) if uri.ends_with(".json") => queue.push(uri.to_owned()),
                Ok(()) => {}
                Err(detail) => faults.push(Fault::new("UriMissing", &shown, uri, detail)),
            }
        }
    }
    let line = json!({
        "uris": {
            "files_read": read.len(),
            "uris_named": seen.len(),
            "missing": faults.len(),
        }
    });
    (line, faults)
}
