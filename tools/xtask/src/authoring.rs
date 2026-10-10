use std::{
    fs,
    path::{Path, PathBuf},
};

use engine_dev::authoring::{read_scene_source, validate_scene};

pub fn validate_sources(root: &Path) -> Result<(), String> {
    let paths = scene_paths(root)?;
    if paths.is_empty() {
        return Err("no scene sources found in web/public/scenes/*.scene.json".to_owned());
    }
    for path in &paths {
        let display = path.strip_prefix(root).unwrap_or(path).display();
        // The same bounded reader and validation as `cargo xtask scene validate`.
        let report = read_scene_source(path)
            .and_then(|source| validate_scene(&source))
            .map_err(|error| format!("{display}: {error}"))?;
        println!(
            "Validated {display}: scene={} revision={} authored_entities={} prefabs={} expanded_entities={}",
            report.id, report.revision, report.entities, report.prefabs, report.expanded_entities
        );
    }
    println!("Validated {} scene source(s).", paths.len());
    Ok(())
}

fn scene_paths(root: &Path) -> Result<Vec<PathBuf>, String> {
    let directory = root.join("web/public/scenes");
    let mut paths = Vec::new();
    let entries = fs::read_dir(&directory)
        .map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("cannot read scene directory entry: {error}"))?;
        let path = entry.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".scene.json"))
            && path.is_file()
        {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

/// The files the web host, scripts and the Stress Lab read that Rust
/// generates: the development protocol's summary, the browser clients, the
/// frame stats' types, the Stress Lab page's types and catalog, and its
/// inspectors' schemas.
fn generated() -> [(&'static str, String); 5] {
    let clients = serde_json::to_string_pretty(&crate::process::clients_json())
        .expect("the client list serializes")
        + "\n";
    let stats: serde_json::Value =
        serde_json::from_str(engine_dev::FRAME_STATS_SCHEMA).expect("generated schema is JSON");
    [
        (
            "web/src/generated/protocol.json",
            engine_dev::PROTOCOL_SUMMARY.to_owned(),
        ),
        ("web/src/generated/clients.json", clients),
        (
            "web/src/generated/stats.ts",
            crate::typescript::interfaces(&stats, "FrameStats", "engine_core::FrameStats"),
        ),
        ("web/src/generated/stress.ts", stress_page()),
        (
            "apps/stress-lab/src/generated/inspectors.json",
            stress_inspectors(),
        ),
    ]
}

/// The Stress Lab page's types (its snapshot, configuration and scenes),
/// its configuration's limits and its catalog of choices.
fn stress_page() -> String {
    let snapshot = serde_json::to_value(schemars::schema_for!(stress_lab::StressSnapshot))
        .expect("schemas serialize");
    let mut page = crate::typescript::interfaces(&snapshot, "StressSnapshot", "stress_lab");
    let config = &snapshot["$defs"]["StressConfig"]["properties"];
    let limits: serde_json::Map<String, serde_json::Value> = config
        .as_object()
        .expect("the configuration is an object")
        .iter()
        .filter_map(|(field, property)| {
            let limits: serde_json::Map<String, serde_json::Value> = ["minimum", "maximum", "enum"]
                .into_iter()
                .filter_map(|key| Some((key.to_owned(), property.get(key)?.clone())))
                .collect();
            (!limits.is_empty()).then(|| (field.clone(), limits.into()))
        })
        .collect();
    page.push_str(
        "\n/** Each configuration field's bounds or values, as validation keeps them. */\n",
    );
    page.push_str(&format!(
        "export const STRESS_CONFIG_LIMITS = {} as const;\n",
        serde_json::to_string_pretty(&limits).expect("limits serialize")
    ));
    page.push_str(
        "\n/** The scenes, looks, effects, views and mesh batches the page offers, and the\n * per-frame sample's field names in order. */\n",
    );
    page.push_str(&format!(
        "export const STRESS_CATALOG = {} as const;\n",
        serde_json::to_string_pretty(&stress_lab::catalog()).expect("the catalog serializes")
    ));
    page.push_str("\n/** A field of the per-frame sample. */\nexport type StressSampleField = (typeof STRESS_CATALOG.sample)[number];\n");
    page
}

/// The schemas of the Stress Lab's inspectors, by name.
fn stress_inspectors() -> String {
    let schemas = serde_json::json!({
        "StressConfig": schemars::schema_for!(stress_lab::StressConfig),
        "StressMetrics": schemars::schema_for!(stress_lab::StressMetrics),
        "PlayerWalk": schemars::schema_for!(stress_lab::PlayerWalk),
    });
    serde_json::to_string_pretty(&schemas).expect("schemas serialize") + "\n"
}

/// Writes the generated files the web host and scripts read.
pub fn write_generated(root: &Path) -> Result<(), String> {
    for (path, contents) in generated() {
        if let Some(directory) = Path::new(path).parent() {
            fs::create_dir_all(root.join(directory))
                .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
        }
        fs::write(root.join(path), contents)
            .map_err(|error| format!("cannot write {path}: {error}"))?;
        println!("Generated {path}.");
    }
    Ok(())
}

pub fn check_generated(root: &Path) -> Result<(), String> {
    for (path, contents) in generated() {
        let saved = fs::read_to_string(root.join(path))
            .map_err(|error| format!("cannot read {path}: {error}; run `cargo xtask schema`"))?;
        if saved.replace("\r\n", "\n") != contents {
            return Err(format!("{path} is out of date; run `cargo xtask schema`"));
        }
        println!("Generated file matches {path}.");
    }
    Ok(())
}
