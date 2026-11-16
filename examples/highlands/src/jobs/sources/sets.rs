pub(crate) const BARKS: [&str; 3] = ["spruce", "pine", "birch"];

pub(crate) const ATLASES: [&str; 3] = ["spruce_twigs", "pine_twigs", "undergrowth"];

pub(crate) const LEAVES: &str = "birch_leaves";

pub(crate) const BARK_PARTS: [&str; 3] = ["albedo", "normal", "orm"];

pub(crate) const ATLAS_PARTS: [&str; 2] = ["albedo", "normal"];

pub(crate) fn map_files(folder: &str, name: &str, parts: &[&str]) -> Vec<String> {
    parts
        .iter()
        .map(|part| format!("highlands/{folder}/{name}_{part}.png"))
        .collect()
}

pub(crate) fn cells_file(name: &str) -> String {
    format!("highlands/plants/{name}.json")
}

pub(crate) fn spray_files(name: &str) -> Vec<String> {
    let mut files = map_files("plants", name, &ATLAS_PARTS);
    files.push(cells_file(name));
    files
}
