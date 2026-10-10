//! What the smokes check of each engine feature: every environment
//! section, inspector and development op, whether frame stats report the
//! section, and which smokes name each. `cargo xtask schema` writes it as
//! web/src/generated/coverage.json; lint keeps it current, so a gap shows
//! where agents and reviewers look.

use std::path::Path;

use serde_json::{Value, json};

/// Each environment section's key and settings type, in authored order.
macro_rules! section_names {
    ($($key:ident: $settings:ident { since: $since:literal, opt_in: $opt_in:literal, feature: $feature:ident }),* $(,)?) => {
        &[$((stringify!($key), stringify!($settings))),*]
    };
}

pub fn coverage(root: &Path) -> Result<String, String> {
    let sections: &[(&str, &str)] = engine::environment_sections!(section_names);
    let smokes: Vec<(String, String)> = crate::smoke::scripts(root)?
        .into_iter()
        .map(|(name, _, source)| (name, source))
        .collect();
    // Smokes naming `word` whole, or `"word"` quoted.
    let naming = |word: &str, quoted: bool| -> Vec<String> {
        smokes
            .iter()
            .filter(|(_, text)| {
                if quoted {
                    return text.contains(&format!("\"{word}\""));
                }
                text.match_indices(word).any(|(at, _)| {
                    let before = text[..at].chars().next_back();
                    let after = text[at + word.len()..].chars().next();
                    let part = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
                    !part(before) && !part(after)
                })
            })
            .map(|(name, _)| name.clone())
            .collect()
    };
    let stats: Value =
        serde_json::from_str(engine_dev::FRAME_STATS_SCHEMA).expect("generated schema is JSON");
    let reported = |key: &str| stats["properties"].get(key).is_some();
    // The protocol's own description of its types.
    let mut engine = engine::Engine::default();
    let tools = engine_dev::DevTools::default();
    let described: Value = serde_json::from_str(&tools.execute(
        &mut engine,
        engine_dev::Context::default(),
        &json!({"version": engine_dev::PROTOCOL_VERSION, "op": "describe"}).to_string(),
    ))
    .expect("replies are JSON");
    let section_types: Vec<&str> = sections.iter().map(|(_, name)| *name).collect();
    let inspectors: Vec<Value> = described["result"]["types"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|ty| ["component", "resource"].contains(&ty["kind"].as_str().unwrap_or_default()))
        .filter(|ty| !section_types.contains(&ty["name"].as_str().unwrap_or_default()))
        .map(|ty| {
            let name = ty["name"].as_str().unwrap_or_default();
            json!({"inspector": name, "kind": ty["kind"], "smokes": naming(name, true)})
        })
        .collect();
    let features: Vec<Value> = sections
        .iter()
        .map(|&(key, name)| {
            json!({
                "section": key,
                "inspector": name,
                "stats": reported(key),
                "smokes": naming(key, false),
            })
        })
        .collect();
    let protocol: Value =
        serde_json::from_str(engine_dev::PROTOCOL_SUMMARY).expect("the summary is JSON");
    let ops: Vec<Value> = ["queries", "controls"]
        .iter()
        .flat_map(|kind| protocol[kind].as_array().cloned().unwrap_or_default())
        .map(|op| {
            let op = op.as_str().unwrap_or_default().to_owned();
            json!({"op": op, "smokes": naming(&op, true)})
        })
        .collect();
    Ok(serde_json::to_string_pretty(&json!({
        "sections": features,
        "inspectors": inspectors,
        "ops": ops,
    }))
    .expect("coverage serializes")
        + "\n")
}

/// Fails unless some smoke checks every inspector, environment section and
/// protocol op: what the engine offers agents is exercised.
pub fn check(root: &Path) -> Result<(), String> {
    let coverage: Value = serde_json::from_str(&coverage(root)?).expect("coverage is JSON");
    let mut unchecked = Vec::new();
    for (list, key) in [
        ("inspectors", "inspector"),
        ("sections", "section"),
        ("ops", "op"),
    ] {
        for entry in coverage[list].as_array().into_iter().flatten() {
            if entry["smokes"].as_array().is_none_or(Vec::is_empty) {
                unchecked.push(format!("{key} {}", entry[key].as_str().unwrap_or_default()));
            }
        }
    }
    if unchecked.is_empty() {
        return Ok(());
    }
    Err(format!(
        "no smoke (web/scripts/*-smoke.mjs) checks {}: add one that does",
        unchecked.join(", ")
    ))
}
