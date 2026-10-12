//! The structure ratchet: no source file longer than [`MAX_FILE_LINES`]
//! lines and no struct with more than [`MAX_STRUCT_FIELDS`] fields, but
//! those [`LONG_FILES`] and [`WIDE_STRUCTS`] list, each at most its size
//! when listed. A listed one that grows fails; one that is no longer over
//! the limit fails until its entry goes, so the lists only shrink.

use std::path::Path;
use std::process::Command;

/// The most lines a source file has (Rust, WGSL, TypeScript, scripts).
const MAX_FILE_LINES: usize = 1500;
/// The most fields a struct has.
const MAX_STRUCT_FIELDS: usize = 30;

/// The files longer than [`MAX_FILE_LINES`]: each one's most lines.
const LONG_FILES: &[(&str, usize)] = &[
    ("apps/stress-lab/src/city/mod.rs", 2083),
    ("apps/stress-lab/src/highlands/mod.rs", 2085),
    ("apps/stress-lab/src/metro/mod.rs", 3557),
    ("crates/engine_assets/src/simplify.rs", 2383),
    ("crates/engine_core/src/hierarchy.rs", 1714),
    ("crates/engine_core/src/retained_render.rs", 1669),
    ("crates/engine_render/src/meshes/terrain.wgsl", 1843),
    ("crates/engine_render/src/terrain.rs", 2338),
    ("crates/engine_terrain/src/layers.rs", 1603),
    ("tools/assetc/src/bin/terrain_materials.rs", 1530),
];

/// The structs with more than [`MAX_STRUCT_FIELDS`] fields, by file and
/// name: each one's most fields.
const WIDE_STRUCTS: &[(&str, &str, usize)] = &[
    (
        "crates/engine_core/src/features/terrain/residency.rs",
        "TerrainState",
        42,
    ),
    ("crates/engine_core/src/frame_stats.rs", "RenderStats", 42),
    (
        "crates/engine_core/src/interpolation/gpu.rs",
        "GpuInputs",
        35,
    ),
    (
        "crates/engine_core/src/retained_render.rs",
        "RetainedRender",
        37,
    ),
    ("crates/engine_render/src/clouds.rs", "CloudPass", 48),
    ("crates/engine_render/src/renderer.rs", "Renderer", 50),
    ("crates/engine_render/src/scatter.rs", "ScatterPass", 40),
    ("crates/engine_render/src/terrain.rs", "TerrainPass", 65),
    (
        "crates/engine_render/src/terrain_vt.rs",
        "VirtualTexture",
        32,
    ),
    ("crates/engine_terrain/src/tile.rs", "Generator", 33),
];

/// Checks every source file git would commit against the limits and lists.
pub fn check(root: &Path) -> Result<(), String> {
    let mut problems = Vec::new();
    let mut long = Vec::new();
    let mut wide = Vec::new();
    for path in sources(root)? {
        let text = std::fs::read_to_string(root.join(&path))
            .map_err(|error| format!("cannot read {path}: {error}"))?;
        let lines = text.lines().count();
        if lines > MAX_FILE_LINES {
            long.push(path.clone());
            match LONG_FILES.iter().find(|(file, _)| *file == path) {
                Some(&(_, most)) if lines > most => problems.push(format!(
                    "{path}: {lines} lines, more than its {most}: it may only shrink (split it)"
                )),
                Some(_) => {}
                None => problems.push(format!(
                    "{path}: {lines} lines, more than {MAX_FILE_LINES}: split it"
                )),
            }
        }
        if !path.ends_with(".rs") {
            continue;
        }
        for (name, fields) in structs(&code(&text)) {
            if fields <= MAX_STRUCT_FIELDS {
                continue;
            }
            wide.push((path.clone(), name.clone()));
            match WIDE_STRUCTS
                .iter()
                .find(|(file, wide, _)| *file == path && *wide == name)
            {
                Some(&(_, _, most)) if fields > most => problems.push(format!(
                    "{path}: struct {name} has {fields} fields, more than its {most}: it may only shrink"
                )),
                Some(_) => {}
                None => problems.push(format!(
                    "{path}: struct {name} has {fields} fields, more than {MAX_STRUCT_FIELDS}: split it"
                )),
            }
        }
    }
    for (path, _) in LONG_FILES {
        if !long.iter().any(|file| file == path) {
            problems.push(format!(
                "{path} is listed in LONG_FILES (tools/xtask/src/structure.rs) but is no longer \
                 over {MAX_FILE_LINES} lines: remove its entry"
            ));
        }
    }
    for (path, name, _) in WIDE_STRUCTS {
        if !wide.iter().any(|(file, wide)| file == path && wide == name) {
            problems.push(format!(
                "{path}: struct {name} is listed in WIDE_STRUCTS (tools/xtask/src/structure.rs) \
                 but no longer has over {MAX_STRUCT_FIELDS} fields: remove its entry"
            ));
        }
    }
    if problems.is_empty() {
        println!(
            "No source grew: {} long files and {} wide structs, none over its size.",
            LONG_FILES.len(),
            WIDE_STRUCTS.len()
        );
        return Ok(());
    }
    Err(problems.join("\n"))
}

/// The sources git would commit, `/`-separated: Rust, WGSL, TypeScript
/// and scripts, but vendored and generated files.
pub(crate) fn sources(root: &Path) -> Result<Vec<String>, String> {
    let output = Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot run git ls-files: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git ls-files failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let mut files: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .split('\0')
        .filter(|path| {
            [".rs", ".wgsl", ".ts", ".mjs"]
                .iter()
                .any(|extension| path.ends_with(extension))
                && !path.starts_with("vendor/")
                && !path.contains("/generated/")
                && root.join(path).is_file()
        })
        .map(str::to_owned)
        .collect();
    files.sort();
    files.dedup();
    Ok(files)
}

/// `text` with its comments and its string and character literals blanked
/// out, lines kept, so only code is left to read.
fn code(text: &str) -> String {
    let characters: Vec<char> = text.chars().collect();
    let blank = |from: &[char]| -> String {
        from.iter()
            .map(|&character| if character == '\n' { '\n' } else { ' ' })
            .collect()
    };
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    while at < characters.len() {
        let rest = &characters[at..];
        let end = match rest {
            ['/', '/', ..] => rest.iter().position(|&c| c == '\n').unwrap_or(rest.len()),
            ['/', '*', ..] => {
                let mut depth = 0;
                let mut index = 0;
                while index < rest.len() {
                    match &rest[index..] {
                        ['/', '*', ..] => {
                            depth += 1;
                            index += 2;
                        }
                        ['*', '/', ..] => {
                            depth -= 1;
                            index += 2;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => index += 1,
                    }
                }
                index
            }
            ['r', '#' | '"', ..] if !word_before(&characters, at) => {
                let hashes = rest[1..].iter().take_while(|&&c| c == '#').count();
                if rest.get(1 + hashes) == Some(&'"') {
                    let close: Vec<char> = std::iter::once('"')
                        .chain(std::iter::repeat_n('#', hashes))
                        .collect();
                    let start = 2 + hashes;
                    rest[start..]
                        .windows(close.len())
                        .position(|window| window == close.as_slice())
                        .map_or(rest.len(), |found| start + found + close.len())
                } else {
                    0
                }
            }
            ['"', ..] => {
                let mut index = 1;
                while index < rest.len() && rest[index] != '"' {
                    index += if rest[index] == '\\' { 2 } else { 1 };
                }
                (index + 1).min(rest.len())
            }
            ['\'', '\\', ..] => rest[2..]
                .iter()
                .position(|&c| c == '\'')
                .map_or(rest.len(), |found| found + 3),
            ['\'', _, '\'', ..] => 3,
            _ => 0,
        };
        if end == 0 {
            out.push(characters[at]);
            at += 1;
        } else {
            out.push_str(&blank(&rest[..end]));
            at += end;
        }
    }
    out
}

/// Whether a word character comes before `at`.
fn word_before(characters: &[char], at: usize) -> bool {
    at > 0 && (characters[at - 1].is_alphanumeric() || characters[at - 1] == '_')
}

/// Each struct `code` declares, by name, with its fields: a named struct's
/// (as rustfmt lays them out, one `name:` each), a tuple struct's, none for
/// a unit struct.
fn structs(code: &str) -> Vec<(String, usize)> {
    let characters: Vec<char> = code.chars().collect();
    let word_at = |at: usize| -> String {
        characters[at..]
            .iter()
            .take_while(|c| c.is_alphanumeric() || **c == '_')
            .collect()
    };
    let mut found = Vec::new();
    let mut at = 0;
    while at < characters.len() {
        if !(characters[at].is_alphabetic() || characters[at] == '_')
            || word_before(&characters, at)
        {
            at += 1;
            continue;
        }
        let word = word_at(at);
        at += word.chars().count();
        if word != "struct" {
            continue;
        }
        let mut index = at;
        while index < characters.len() && characters[index].is_whitespace() {
            index += 1;
        }
        let name = word_at(index);
        if name.is_empty() {
            continue;
        }
        index += name.chars().count();
        // Generics, where `->` is no closing bracket.
        let mut angle = 0_i32;
        while index < characters.len() {
            match characters[index] {
                '-' if characters.get(index + 1) == Some(&'>') => index += 1,
                '<' => angle += 1,
                '>' => angle -= 1,
                '{' | '(' | ';' if angle == 0 => break,
                _ => {}
            }
            index += 1;
        }
        let fields = match characters.get(index) {
            Some('{') => named_fields(&characters, index),
            Some('(') => tuple_fields(&characters, index),
            _ => 0,
        };
        found.push((name, fields));
        at = index;
    }
    found
}

/// The fields of the named struct whose body opens at `open`: each word
/// directly inside it followed by a single `:`.
fn named_fields(characters: &[char], open: usize) -> usize {
    let mut depth = 0;
    let mut fields = 0;
    let mut index = open;
    while index < characters.len() {
        let character = characters[index];
        match character {
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            _ if depth == 1
                && (character.is_alphabetic() || character == '_')
                && !word_before(characters, index) =>
            {
                let word: String = characters[index..]
                    .iter()
                    .take_while(|c| c.is_alphanumeric() || **c == '_')
                    .collect();
                let mut after = index + word.chars().count();
                while after < characters.len() && characters[after].is_whitespace() {
                    after += 1;
                }
                if word != "pub"
                    && characters.get(after) == Some(&':')
                    && characters.get(after + 1) != Some(&':')
                {
                    fields += 1;
                }
                index += word.chars().count();
                continue;
            }
            _ => {}
        }
        index += 1;
    }
    fields
}

/// The fields of the tuple struct whose fields open at `open`: its
/// top-level parts between commas.
fn tuple_fields(characters: &[char], open: usize) -> usize {
    let mut depth = 0;
    let mut parts = 0;
    let mut filled = false;
    let mut index = open;
    while index < characters.len() {
        match characters[index] {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            ',' if depth == 1 => {
                parts += usize::from(filled);
                filled = false;
            }
            character if !character.is_whitespace() => filled = true,
            _ => {}
        }
        index += 1;
    }
    parts + usize::from(filled)
}
