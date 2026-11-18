use std::path::Path;

use serde_json::{Map, Value, json};

use crate::config::ScenePart;
use crate::fault::Fault;

const LISTS: [&str; 4] = ["textures", "materials", "meshes", "entities"];

type Spans = (usize, usize, Vec<(usize, usize)>);

fn top_level_array(text: &str, key: &str) -> Option<Spans> {
    let bytes = text.as_bytes();
    let quoted = format!("\"{key}\"");
    let (mut depth, mut at, mut in_string, mut escaped) = (0usize, 0usize, false, false);
    let mut found = None;
    while at < bytes.len() {
        let byte = bytes[at];
        if in_string {
            match (escaped, byte) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => in_string = false,
                (false, _) => {}
            }
            at += 1;
            continue;
        }
        match byte {
            b'"' if depth == 1 && text[at..].starts_with(&quoted) => {
                let after = text[at + quoted.len()..].trim_start();
                if let Some(value) = after.strip_prefix(':') {
                    let open = text.len() - value.trim_start().len();
                    if bytes.get(open) == Some(&b'[') {
                        found = Some(open);
                        break;
                    }
                }
                in_string = true;
            }
            b'"' => in_string = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
        at += 1;
    }
    let open = found?;
    let (mut depth, mut at, mut in_string, mut escaped) = (0usize, open, false, false);
    let mut items = Vec::new();
    let mut start = None;
    while at < bytes.len() {
        let byte = bytes[at];
        if in_string {
            match (escaped, byte) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => in_string = false,
                (false, _) => {}
            }
            at += 1;
            continue;
        }
        match byte {
            b'"' => {
                in_string = true;
                if depth == 1 && start.is_none() {
                    start = Some(at);
                }
            }
            b'[' | b'{' => {
                if depth == 1 && start.is_none() {
                    start = Some(at);
                }
                depth += 1;
            }
            b']' | b'}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(first) = start {
                        items.push((first, text[..at].trim_end().len()));
                    }
                    return Some((open, at, items));
                }
            }
            b',' if depth == 1 => {
                if let Some(first) = start.take() {
                    items.push((first, text[..at].trim_end().len()));
                }
            }
            byte if depth == 1 && start.is_none() && !byte.is_ascii_whitespace() => {
                start = Some(at);
            }
            _ => {}
        }
        at += 1;
    }
    None
}

fn merged(held: &Value, made: &Value) -> Value {
    match (held, made) {
        (Value::Object(held), Value::Object(made)) => {
            let mut out: Map<String, Value> = held.clone();
            for (key, value) in made {
                let next = out
                    .get(key)
                    .map_or_else(|| value.clone(), |old| merged(old, value));
                out.insert(key.clone(), next);
            }
            Value::Object(out)
        }
        (_, made) => made.clone(),
    }
}

fn pretty(value: &Value, newline: &str) -> String {
    serde_json::to_string_pretty(value)
        .expect("values_serialize")
        .replace('\n', &format!("{newline}    "))
}

pub fn apply(base: &Path, part: &ScenePart, present: bool) -> Result<Value, Fault> {
    let path = base.join(&part.scene);
    let file = path.display().to_string();
    let text = std::fs::read_to_string(&path)
        .map_err(|error| Fault::new("SceneUnreadable", &file, "file", error))?;
    let mut expected: Value = serde_json::from_str(&text)
        .map_err(|error| Fault::new("SceneInvalid", &file, "file", error))?;
    let mut written = text.clone();
    let newline = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let mut changed = Vec::new();
    for list in LISTS {
        let made: &[Value] = match list {
            "textures" => &part.textures,
            "materials" => &part.materials,
            "meshes" => &part.meshes,
            _ => &part.entities,
        };
        let ids: Vec<&str> = made.iter().filter_map(|item| item["id"].as_str()).collect();
        let held = expected[list].as_array().cloned().unwrap_or_default();
        let (open, close, spans) = top_level_array(&written, list).ok_or_else(|| {
            Fault::new(
                "SceneInvalid",
                &file,
                list,
                "the scene has no such top-level array",
            )
        })?;
        if spans.len() != held.len() {
            return Err(Fault::new(
                "SceneInvalid",
                &file,
                list,
                "the array did not split into its items",
            ));
        }
        let mut items: Vec<(Value, String)> = Vec::new();
        for (value, (start, end)) in held.iter().zip(&spans) {
            let id = value["id"].as_str().unwrap_or_default();
            if !ids.contains(&id) {
                items.push((value.clone(), written[*start..*end].to_owned()));
                continue;
            }
            let Some(next) = made.iter().find(|item| item["id"] == value["id"]) else {
                continue;
            };
            if !present {
                changed.push(format!("{list}.{id} removed"));
                continue;
            }
            let joined = merged(value, next);
            if joined == *value {
                items.push((value.clone(), written[*start..*end].to_owned()));
            } else {
                changed.push(format!("{list}.{id} updated"));
                let shown = pretty(&joined, newline);
                items.push((joined, shown));
            }
        }
        if present {
            for next in made {
                if !held.iter().any(|value| value["id"] == next["id"]) {
                    changed.push(format!(
                        "{list}.{} added",
                        next["id"].as_str().unwrap_or_default()
                    ));
                    items.push((next.clone(), pretty(next, newline)));
                }
            }
        }
        let body = if items.is_empty() {
            String::new()
        } else {
            let joined: Vec<&str> = items.iter().map(|(_, shown)| shown.as_str()).collect();
            format!(
                "{newline}    {}{newline}  ",
                joined.join(&format!(",{newline}    "))
            )
        };
        expected[list] = Value::Array(items.into_iter().map(|(value, _)| value).collect());
        written = format!("{}{body}{}", &written[..=open], &written[close..]);
    }
    let read: Value = serde_json::from_str(&written)
        .map_err(|error| Fault::new("SceneInvalid", &file, "written", error))?;
    if read != expected {
        return Err(Fault::new(
            "ScenePartSpread",
            &file,
            &part.id,
            "the written scene reads back differently",
        ));
    }
    if written != text {
        std::fs::write(&path, &written)
            .map_err(|error| Fault::new("OutputUnwritable", &file, "file", error))?;
    }
    Ok(json!({"scene_part": part.id, "scene": part.scene, "in_scene": present, "changes": changed}))
}
