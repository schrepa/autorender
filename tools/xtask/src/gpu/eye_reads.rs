use std::path::{Path, PathBuf};

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
        "the camera's uniform (transforms), haze, the clouds and the sky",
    ),
    (
        "crates/engine_render/src/renderer/terrain_draws.rs",
        "terrain's eye-relative frame and culling, and water's view",
    ),
    (
        "crates/engine_render/src/temporal.rs",
        "temporally antialiased frames' camera uniform (transforms)",
    ),
];

const FIELD: &str = "camera_position";

pub(super) fn check(root: &Path) -> Result<usize, String> {
    let mut files = Vec::new();
    walk(&root.join("crates"), &mut files);
    for client in crate::process::clients() {
        if !client.folder.starts_with("crates/") {
            walk(&root.join(&client.folder), &mut files);
        }
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
