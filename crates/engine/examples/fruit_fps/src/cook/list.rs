use std::fs;
use std::path::Path;
use std::process::ExitCode;

use cook::{Kind, Plan, Refusal};
use cook_model::{CLIP, FORM, MATERIAL_MAPS, MODEL};
use cook_texture::PICTURE;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::frame::FRAME;
use crate::level::LEVEL;
use crate::sets::VISIBLE_SETS;
use crate::sound::{SOUND_BANK, bank_sources};

const LIST: &str = "cook.json";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct List {
    tolerance: Value,
    keys_per_second: Value,
    variants: Map<String, Value>,
    entries: Vec<Entry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    kind: String,
    name: String,
    sources: Vec<String>,
    params: Map<String, Value>,
}

pub fn cook() -> ExitCode {
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
    match planned(&assets) {
        Ok(plan) => cook::main(plan),
        Err(no) => {
            let line = json!({"error": no.code, "step": "", "file": no.file, "detail": no.detail});
            println!("{line}");
            ExitCode::FAILURE
        }
    }
}

fn planned(assets: &Path) -> Result<Plan, Refusal> {
    let refuse = |detail: String| Refusal::new("CookList", LIST, detail);
    let bytes = fs::read(assets.join(LIST)).map_err(|error| refuse(error.to_string()))?;
    let list: List = serde_json::from_slice(&bytes).map_err(|error| refuse(error.to_string()))?;
    let mut plan = Plan::new(assets, &assets.join("cooked"));
    for entry in &list.entries {
        declare(&mut plan, assets, &list, entry)
            .map_err(|detail| refuse(format!("entries.{}: {detail}", entry.name)))?;
    }
    Ok(plan)
}

fn shared(params: &mut Map<String, Value>, values: &[(&str, &Value)]) {
    for &(key, value) in values {
        params.entry(key).or_insert_with(|| value.clone());
    }
}

fn character<'a>(list: &'a List, name: Option<&Value>) -> Result<(&'a Value, &'a str), String> {
    let wanted = name.and_then(Value::as_str).unwrap_or_default();
    let owner = list.entries.iter().find(|entry| {
        entry.kind == "model"
            && entry.name == wanted
            && entry.params.get("kind").and_then(Value::as_str) == Some("character")
    });
    match owner.map(|entry| (entry.params.get("scale"), entry.sources.as_slice())) {
        Some((Some(scale), [source])) => Ok((scale, source.as_str())),
        _ => Err(format!("`{wanted}` is not a character entry with a scale and one source")),
    }
}

fn sets(plan: &mut Plan, entry: &Entry, params: Map<String, Value>) -> Result<(), String> {
    let [data] = entry.sources.as_slice() else {
        return Err("the visible sets read one data file".to_owned());
    };
    let level = params.get("level").and_then(Value::as_str).ok_or("no `level`")?;
    let (occluders, frame) = (format!("{level}.occluders.bin"), format!("{}_frame", entry.name));
    let level = level.to_owned();
    plan.step(&FRAME, &frame, json!({})).source(data);
    plan.step(&VISIBLE_SETS, &entry.name, Value::Object(params))
        .output(&level, &occluders)
        .output(&frame, "frame.json");
    Ok(())
}

fn declare(plan: &mut Plan, assets: &Path, list: &List, entry: &Entry) -> Result<(), String> {
    let mut params = entry.params.clone();
    for key in ["textures", "encodings"] {
        if let Some(Value::String(set)) = params.get(key) {
            let held = list.variants.get(set).ok_or_else(|| format!("no variant set `{set}`"))?;
            params.insert(key.to_owned(), held.clone());
        }
    }
    let mut sources = entry.sources.clone();
    let tolerance = ("tolerance", &list.tolerance);
    let keys = ("keys_per_second", &list.keys_per_second);
    let kind: &'static Kind = match entry.kind.as_str() {
        "picture" => &PICTURE,
        "material_maps" => &MATERIAL_MAPS,
        "model" => {
            if params.get("kind").and_then(Value::as_str) == Some("character") {
                shared(&mut params, &[keys]);
            }
            if let Some(owner) = params.get("maps_of")
                && !list.entries.iter().any(|held| {
                    held.kind == "material_maps" && owner.as_str() == Some(held.name.as_str())
                })
            {
                return Err(format!("`maps_of` {owner} names no `material_maps` entry"));
            }
            shared(&mut params, &[tolerance]);
            &MODEL
        }
        "form" => {
            let (scale, owner) = character(list, params.get("form_of"))?;
            sources.push(owner.to_owned());
            params.insert("scale".to_owned(), scale.clone());
            shared(&mut params, &[tolerance]);
            &FORM
        }
        "clip" => {
            let scale = match params.get("rig_from") {
                None => json!(1.0),
                Some(name) => {
                    let (scale, owner) = character(list, Some(name))?;
                    sources.push(owner.to_owned());
                    scale.clone()
                }
            };
            params.insert("scale".to_owned(), scale);
            shared(&mut params, &[tolerance, keys]);
            &CLIP
        }
        "level" => {
            shared(&mut params, &[tolerance]);
            &LEVEL
        }
        "sound_bank" => {
            let [table] = entry.sources.as_slice() else {
                return Err("sound_bank reads one sound table source".to_owned());
            };
            sources.extend(bank_sources(assets, table, &params)?);
            &SOUND_BANK
        }
        "visible_sets" => return sets(plan, entry, params),
        other => return Err(format!("no kind `{other}`")),
    };
    let mut declared = plan.step(kind, &entry.name, Value::Object(params));
    for source in &sources {
        declared.source(source);
    }
    Ok(())
}
