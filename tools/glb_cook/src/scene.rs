use std::path::Path;

use serde_json::Value;

use crate::config::SceneVariant;
use crate::fault::Fault;

fn swapped(uri: &str, variant: &SceneVariant) -> Option<String> {
    let ending = format!(".{}.ktx2", variant.replaces);
    uri.strip_suffix(&ending)
        .map(|stem| format!("{stem}.{}.ktx2", variant.variant))
}

pub fn write(base: &Path, variant: &SceneVariant) -> Result<usize, Fault> {
    let from = base.join(&variant.from);
    let to = base.join(&variant.to);
    let file = from.display().to_string();
    if from.parent() != to.parent() {
        return Err(Fault::new(
            "SceneFolder",
            &file,
            "to",
            "the written scene must sit beside its source so every uri resolves the same",
        ));
    }
    let text = std::fs::read_to_string(&from)
        .map_err(|error| Fault::new("SceneUnreadable", &file, "file", error))?;
    let mut expected: Value = serde_json::from_str(&text)
        .map_err(|error| Fault::new("SceneInvalid", &file, "file", error))?;
    let folder = from.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut swaps = Vec::new();
    for (index, texture) in expected
        .get_mut("textures")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let Some(uri) = texture.get("uri").and_then(Value::as_str) else {
            continue;
        };
        let Some(made) = swapped(uri, variant) else {
            continue;
        };
        if !folder.join(&made).is_file() {
            return Err(Fault::new(
                "SceneVariantMissing",
                &file,
                format!("textures[{index}].uri"),
                format!("`{uri}` becomes `{made}`, which was not cooked"),
            ));
        }
        swaps.push((uri.to_owned(), made.clone()));
        texture["uri"] = Value::String(made);
    }
    if swaps.is_empty() {
        return Err(Fault::new(
            "SceneVariantNone",
            &file,
            "textures",
            format!("no texture uri ends in `.{}.ktx2`", variant.replaces),
        ));
    }
    let mut written = text.clone();
    for (uri, made) in &swaps {
        let quoted = Value::String(uri.clone()).to_string();
        written = written.replace(&quoted, &Value::String(made.clone()).to_string());
    }
    let read: Value = serde_json::from_str(&written)
        .map_err(|error| Fault::new("SceneInvalid", &to.display().to_string(), "file", error))?;
    if read != expected {
        return Err(Fault::new(
            "SceneVariantSpread",
            &file,
            "file",
            "a swapped uri also appears outside `textures`; only texture uris may change",
        ));
    }
    std::fs::write(&to, written).map_err(|error| {
        Fault::new("OutputUnwritable", &to.display().to_string(), "file", error)
    })?;
    Ok(swaps.len())
}
