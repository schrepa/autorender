use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use brick_sound::SoundBank;
use cook::{Kind, Refusal, Step};
use serde::Deserialize;
use serde_json::json;

pub static SOUND_BANK: Kind = Kind {
    name: "sound_bank",
    version: 1,
    run: sound_bank,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SoundBankParams {
    groups: Vec<String>,
}

#[derive(Deserialize)]
struct TableFile {
    sounds: BTreeMap<String, TableSound>,
}

#[derive(Deserialize)]
struct TableSound {
    group: String,
    files: Vec<TableSoundFile>,
}

#[derive(Deserialize)]
struct TableSoundFile {
    file: String,
}

fn source_for(file: &str) -> String {
    file.strip_prefix("assets/")
        .map_or_else(|| format!("../{file}"), str::to_owned)
}

fn selected_files(bytes: &[u8], groups: &[String]) -> Result<Vec<String>, String> {
    if groups.is_empty() {
        return Err("groups must list at least one sound group".to_owned());
    }
    let table: TableFile =
        serde_json::from_slice(bytes).map_err(|error| format!("bad sound table: {error}"))?;
    let wanted: BTreeSet<&str> = groups.iter().map(String::as_str).collect();
    let mut matched_groups = BTreeSet::new();
    let mut files = BTreeSet::new();
    for sound in table.sounds.values() {
        if wanted.contains(sound.group.as_str()) {
            matched_groups.insert(sound.group.as_str());
            for entry in &sound.files {
                files.insert(entry.file.clone());
            }
        }
    }
    if let Some(missing) = wanted
        .iter()
        .find(|group| !matched_groups.contains(**group))
    {
        return Err(format!("sound group `{missing}` matched no sounds"));
    }
    Ok(files.into_iter().collect())
}

pub fn bank_sources(
    assets: &Path,
    table_source: &str,
    params: &serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<String>, String> {
    let params: SoundBankParams = serde_json::from_value(serde_json::Value::Object(params.clone()))
        .map_err(|error| format!("bad sound_bank params: {error}"))?;
    let bytes = fs::read(assets.join(table_source))
        .map_err(|error| format!("cannot read {table_source}: {error}"))?;
    let files = selected_files(&bytes, &params.groups)?;
    Ok(files.iter().map(|file| source_for(file)).collect())
}

fn sound_bank(step: &mut Step<'_>) -> Result<(), Refusal> {
    let name = step.name().to_owned();
    let params: SoundBankParams = step.params()?;
    let Some(table_input) = step.inputs().first() else {
        return Err(Refusal::new("SoundBankInputs", &name, "sound table source required"));
    };
    let files = selected_files(table_input.bytes, &params.groups)
        .map_err(|detail| Refusal::new("SoundBankGroups", table_input.name, detail))?;
    let mut entries = Vec::with_capacity(files.len());
    for file in &files {
        let source_name = source_for(file);
        let input = step.input(&source_name)?;
        entries.push((file.as_str(), input.bytes));
    }
    let packed = SoundBank::write(&entries)
        .map_err(|detail| Refusal::new("SoundBankEncode", &name, detail))?;
    let output = format!("{name}.esb");
    SoundBank::read(Arc::from(packed.as_slice()), &output)
        .map_err(|error| Refusal::new("SoundBankReadBack", &output, error))?;
    step.note(json!({
        "sound_bank": {
            "groups": params.groups,
            "files": entries.len(),
            "bytes": packed.len(),
        }
    }));
    step.write(&output, packed);
    Ok(())
}
