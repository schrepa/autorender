use std::fmt::Write as _;
use std::path::PathBuf;

use serde_json::Value;

fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default());
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap_or_default());
    let listed = manifest.join("../../bricks.json");
    println!("cargo:rerun-if-changed={}", listed.display());
    let catalog: Value = std::fs::read_to_string(&listed)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(Value::Null);
    let bricks = catalog["bricks"].as_array().cloned().unwrap_or_default();
    let mut written: Vec<(String, Vec<String>)> = bricks
        .iter()
        .filter_map(|brick| brick["name"].as_str())
        .map(|name| (name.to_owned(), Vec::new()))
        .collect();
    for read in bricks
        .iter()
        .flat_map(|brick| brick["reads"].as_array().into_iter().flatten())
        .filter(|read| read["kind"] == "target")
    {
        let Some(target) = read["place"].as_str() else {
            continue;
        };
        for provider in strings(&read["providers"]) {
            if let Some((_, targets)) = written.iter_mut().find(|(name, _)| *name == provider)
                && !targets.iter().any(|known| known == target)
            {
                targets.push(target.to_owned());
            }
        }
    }
    let mut source = String::from("pub(crate) static STANDARD: &[Standard] = &[\n");
    for (name, targets) in &written {
        let Some(brick) = bricks.iter().find(|brick| brick["name"] == name.as_str()) else {
            continue;
        };
        let mut places = strings(&brick["fills"]);
        places.extend(strings(&brick["joins"]));
        let _ = writeln!(
            source,
            "    Standard {{ package: {name:?}, places: &{places:?}, targets: &{targets:?} }},"
        );
    }
    source.push_str("];\n");
    std::fs::write(out.join("standard.rs"), source).expect("OUT_DIR is writable");
}
