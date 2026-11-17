use std::fmt::Write;
use std::fs;
use std::path::PathBuf;

#[path = "src/jobs/sources/sets.rs"]
mod sets;

const FOLDERS: [&str; 3] = ["ground", "plants", "bark"];

const PNG_HEADER: &[u8; 8] = b"\x89PNG\r\n\x1a\n";

fn main() {
    engine_sources::emit();
    source_files();
}

fn source_files() {
    let manifest =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let public = manifest.join("../../web/public");
    let mut files = Vec::new();
    for folder in FOLDERS {
        let dir = public.join("highlands").join(folder);
        println!("cargo:rerun-if-changed={}", dir.display());
        let mut names: Vec<String> = fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|entry| entry.path().is_file())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        for name in names {
            let bytes = fs::read(dir.join(&name)).expect("source files are readable");
            files.push((format!("highlands/{folder}/{name}"), bytes));
        }
    }
    let mut code = String::from("pub(crate) const SOURCE_FILES: &[SourceFile] = &[\n");
    for (path, bytes) in &files {
        let [width, height] = png_size(bytes).unwrap_or_default();
        let digest = engine_codec::content_hash(bytes);
        writeln!(
            code,
            "    SourceFile {{ path: {path:?}, digest: {digest:#034x}, size: [{width}, {height}] }},"
        )
        .expect("string_writes");
    }
    code.push_str("];\n\npub(crate) const SET_DIGESTS: &[(&str, u64)] = &[\n");
    for set in photograph_sets() {
        let parts: Option<Vec<&[u8]>> = set
            .iter()
            .map(|path| {
                files
                    .iter()
                    .find(|(file, _)| file == path)
                    .map(|(_, bytes)| &bytes[..])
            })
            .collect();
        let Some(parts) = parts else {
            continue;
        };
        let digest = parts
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |hash, bytes| mixed(hash, bytes));
        writeln!(code, "    ({:?}, {digest:#018x}),", set.join("|")).expect("string_writes");
    }
    code.push_str("];\n");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    fs::write(out.join("source_files.rs"), code).expect("OUT_DIR is writable");
}

fn photograph_sets() -> Vec<Vec<String>> {
    let barks = sets::BARKS.map(|name| sets::map_files("bark", name, &sets::BARK_PARTS));
    let atlases = sets::ATLASES.map(|name| sets::map_files("plants", name, &sets::ATLAS_PARTS));
    barks
        .into_iter()
        .chain(atlases)
        .chain([sets::spray_files(sets::LEAVES)])
        .collect()
}

fn mixed(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn png_size(bytes: &[u8]) -> Option<[u32; 2]> {
    let header = bytes.get(..24)?;
    if &header[..8] != PNG_HEADER {
        return None;
    }
    let word = |at: usize| {
        u32::from_be_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]])
    };
    Some([word(16), word(20)])
}
