//! Reads of the camera's eye on the CPU, which `cargo xtask gpu` holds to a
//! list: every decision of detail by distance (windows, levels, thinning,
//! tiers and budgets' cuts) reads the render snapshot's `detail_eye`, and
//! only the files [`ALLOWED`] lists read its `camera_position`, for
//! transforms, culling, optics, sky and fog.

use std::path::{Path, PathBuf};

/// The files reading the render snapshot's `camera_position`, each with
/// what it reads it for.
const ALLOWED: &[(&str, &str)] = &[
    (
        "crates/engine_core/src/retained_render.rs",
        "copies a frame's whole snapshot",
    ),
    (
        "crates/engine_render/src/post.rs",
        "screen effects' rays from the eye (optics)",
    ),
    (
        "crates/engine_render/src/renderer/frame/prepare.rs",
        "the camera's uniform (transforms), haze, the sea's veil, the clouds and the sky",
    ),
    (
        "crates/engine_render/src/renderer/terrain_draws.rs",
        "terrain's eye-relative frame and culling, and water's view",
    ),
    (
        "crates/engine_render/src/renderer/virtual_draws.rs",
        "virtual geometry's normal cones, tested from the eye (culling)",
    ),
    (
        "crates/engine_render/src/temporal.rs",
        "temporally antialiased frames' camera uniform (transforms)",
    ),
];

/// The folders whose Rust files are read.
const FOLDERS: [&str; 2] = ["crates", "apps"];

/// The snapshot's field of the camera's eye.
const FIELD: &str = "camera_position";

/// Every read of `camera_position` outside [`ALLOWED`], and every listed
/// file that reads it no more; returns how many files read it.
pub(super) fn check(root: &Path) -> Result<usize, String> {
    let mut files = Vec::new();
    for folder in FOLDERS {
        walk(&root.join(folder), &mut files);
    }
    files.sort();
    let mut problems = Vec::new();
    let mut readers = Vec::new();
    for file in files {
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        let lines = reads(&text);
        if lines.is_empty() {
            continue;
        }
        let path = file
            .strip_prefix(root)
            .unwrap_or(&file)
            .to_string_lossy()
            .replace('\\', "/");
        if !ALLOWED.iter().any(|&(allowed, _)| allowed == path) {
            for line in lines {
                problems.push(format!(
                    "{path}:{line}: reads the render snapshot's `{FIELD}`; decisions of detail by \
                     distance read `detail_eye` (only the files tools/xtask/src/gpu/eye_reads.rs \
                     lists read the camera's eye, for transforms, culling, optics, sky and fog)"
                ));
            }
        }
        readers.push(path);
    }
    for &(allowed, _) in ALLOWED {
        if !readers.iter().any(|reader| reader == allowed) {
            problems.push(format!(
                "tools/xtask/src/gpu/eye_reads.rs lists {allowed}, which reads no `{FIELD}`: \
                 remove it"
            ));
        }
    }
    if problems.is_empty() {
        Ok(readers.len())
    } else {
        Err(problems.join("\n"))
    }
}

/// The lines (from 1) of Rust `text` that read `camera_position`: the word
/// in code, but where a field declares or initializes it (`camera_position:`)
/// or an assignment writes it (`camera_position =`).
fn reads(text: &str) -> Vec<usize> {
    let word = |c: char| c.is_alphanumeric() || c == '_';
    let mut lines = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let code = line.split("//").next().unwrap_or_default();
        let read = code.match_indices(FIELD).any(|(at, _)| {
            let rest = &code[at + FIELD.len()..];
            let whole = !code[..at].chars().next_back().is_some_and(word)
                && !rest.chars().next().is_some_and(word);
            let after = rest.trim_start();
            let written = (after.starts_with(':') && !after.starts_with("::"))
                || (after.starts_with('=') && !after.starts_with("=="));
            whole && !written
        });
        if read {
            lines.push(index + 1);
        }
    }
    lines
}

/// The Rust files under `folder`, but build output.
fn walk(folder: &Path, files: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(folder).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name != "target") {
                walk(&path, files);
            }
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
}
