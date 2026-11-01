use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

const HASHED: &[(&str, &str)] = &[
    ("LODBUILD", "engine_lodbuild"),
    ("TEXTURES", "engine_textures"),
];

fn main() {
    let manifest = PathBuf::from(var("CARGO_MANIFEST_DIR"));
    let root = workspace_root(&manifest);
    let workspace =
        fs::read_to_string(root.join("Cargo.toml")).expect("the workspace manifest reads");
    let lock = root.join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", root.join("Cargo.toml").display());
    println!("cargo:rerun-if-changed={}", lock.display());
    let mut code = String::new();
    for &(constant, package) in HASHED {
        let folder = root.join(dependency_path(&workspace, package));
        let hash = digest(package, &root, &folder, &lock);
        code.push_str(&format!(
            "pub(crate) const {constant}: [u8; 32] = {hash:?};\n"
        ));
    }
    fs::write(PathBuf::from(var("OUT_DIR")).join("sources.rs"), code)
        .expect("OUT_DIR is writable");
}

fn var(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is set by cargo"))
}

fn workspace_root(start: &Path) -> PathBuf {
    let mut at = Some(start);
    while let Some(dir) = at {
        let workspace = fs::read_to_string(dir.join("Cargo.toml"))
            .is_ok_and(|manifest| manifest.lines().any(|line| line.trim() == "[workspace]"));
        if workspace {
            return dir.to_path_buf();
        }
        at = dir.parent();
    }
    start.to_path_buf()
}

fn dependency_path(workspace: &str, package: &str) -> String {
    workspace
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix(package)?
                .trim_start()
                .strip_prefix('=')
        })
        .find_map(|rest| {
            rest.split("path = \"")
                .nth(1)?
                .split('"')
                .next()
                .map(str::to_owned)
        })
        .unwrap_or_else(|| panic!("the workspace names no path for {package}"))
}

fn digest(package: &str, root: &Path, folder: &Path, lock: &Path) -> [u8; 32] {
    let mut files = Vec::new();
    list(&folder.join("src"), &mut files);
    files.push(folder.join("Cargo.toml"));
    files.sort();
    println!("cargo:rerun-if-changed={}", folder.join("src").display());
    println!("cargo:rerun-if-changed={}", folder.join("Cargo.toml").display());
    let mut hash = Sha256::new();
    hash.update(package.as_bytes());
    hash.update([0]);
    for file in &files {
        hash.update(relative(root, file).as_bytes());
        hash.update([0]);
        hash.update(text_bytes(file));
        hash.update([0]);
    }
    hash.update(text_bytes(lock));
    hash.finalize().into()
}

fn list(folder: &Path, files: &mut Vec<PathBuf>) {
    let Ok(read) = fs::read_dir(folder) else {
        return;
    };
    for entry in read.flatten() {
        let path = entry.path();
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        if path.is_dir() {
            list(&path, files);
        } else {
            files.push(path);
        }
    }
}

fn relative(root: &Path, file: &Path) -> String {
    file.strip_prefix(root)
        .unwrap_or(file)
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

fn text_bytes(file: &Path) -> Vec<u8> {
    let bytes = fs::read(file).unwrap_or_default();
    let mut out = Vec::with_capacity(bytes.len());
    for (at, byte) in bytes.iter().enumerate() {
        if *byte == b'\r' && bytes.get(at + 1) == Some(&b'\n') {
            continue;
        }
        out.push(*byte);
    }
    out
}
