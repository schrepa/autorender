//! Exact file identities and publication, shared by preparation and reports.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

/// Canonical paths passed to Cargo and native tools must not use Windows'
/// verbatim prefix: MSVC's response-file handling does not accept it reliably.
pub fn canonical(path: &Path) -> Result<PathBuf, String> {
    let path = path.canonicalize().map_err(|e| e.to_string())?;
    #[cfg(windows)]
    {
        let text = path.to_string_lossy();
        if let Some(unc) = text.strip_prefix("\\\\?\\UNC\\") {
            return Ok(PathBuf::from(format!("\\\\{unc}")));
        }
        if let Some(normal) = text.strip_prefix("\\\\?\\") {
            return Ok(PathBuf::from(normal));
        }
    }
    Ok(path)
}

/// SHA-256 of bytes, without text normalization.
pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// SHA-256 of a file without loading large artifacts into memory.
pub fn hash(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Unique within and across process lifetimes, without an external service.
pub fn unique() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    format!(
        "{time:x}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

/// Parse a JSON file, naming it in errors.
pub fn json_file(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?)
        .map_err(|error| format!("{}: {error}", path.display()))
}

/// Publish an entire JSON document. A crash cannot leave a truncated report.
pub fn atomic_json(path: &Path, value: &Value) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let temporary = parent.join(format!(".{}.tmp", unique()));
    let result = (|| {
        use std::io::Write;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| e.to_string())?;
        file.write_all(&serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        drop(file);
        fs::rename(&temporary, path).map_err(|e| format!("publishing {}: {e}", path.display()))
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}

/// Every ordinary file, in stable order. Managed stores never follow links.
pub fn list(root: &Path) -> Result<Vec<PathBuf>, String> {
    list_except(root, &[])
}

fn list_except(root: &Path, mutable: &[&Path]) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    if !root.exists() {
        return Ok(files);
    }
    visit(root, root, &mut files, mutable)?;
    files.sort();
    Ok(files)
}

fn visit(
    root: &Path,
    folder: &Path,
    files: &mut Vec<PathBuf>,
    mutable: &[&Path],
) -> Result<(), String> {
    reject_link(folder)?;
    for entry in fs::read_dir(folder).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let kind = entry.file_type().map_err(|e| e.to_string())?;
        if kind.is_symlink() {
            return Err(format!(
                "managed output contains a link: {}",
                entry.path().display()
            ));
        }
        if kind.is_dir() {
            reject_link(&entry.path())?;
            if !mutable
                .iter()
                .any(|directory| entry.path() == root.join(directory))
            {
                visit(root, &entry.path(), files, mutable)?;
            }
        } else if kind.is_file() {
            files.push(
                entry
                    .path()
                    .strip_prefix(root)
                    .map_err(|e| e.to_string())?
                    .to_path_buf(),
            );
        } else {
            return Err(format!("unsupported output: {}", entry.path().display()));
        }
    }
    Ok(())
}

/// Junctions are reparse points too; never traverse one during collection.
pub fn reject_link(path: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let linked = metadata.file_type().is_symlink();
    #[cfg(windows)]
    let linked = {
        use std::os::windows::fs::MetadataExt;
        linked || metadata.file_attributes() & 0x400 != 0
    };
    if linked {
        return Err(format!("managed path is a link: {}", path.display()));
    }
    Ok(())
}

/// Validate a manifest path before joining it to an owned root.
pub fn relative(name: &str) -> Result<&Path, String> {
    let path = Path::new(name);
    if name.is_empty()
        || path
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(format!("not a relative artifact path: {name}"));
    }
    Ok(path)
}

/// Exact manifest of all files under an artifact directory.
pub fn manifest(root: &Path) -> Result<Value, String> {
    manifest_except(root, &[])
}

/// Prepared files remain exact; explicitly declared runtime directories are
/// mutable state, validated at their boundary and counted by storage GC.
pub fn manifest_except(root: &Path, mutable: &[&Path]) -> Result<Value, String> {
    let mut entries = Vec::new();
    for path in list_except(root, mutable)? {
        let full = root.join(&path);
        entries.push(json!({"path": path.to_string_lossy().replace('\\', "/"), "bytes": fs::metadata(&full).map_err(|e| e.to_string())?.len(), "sha256": hash(&full)?}));
    }
    Ok(Value::Array(entries))
}

/// Check completeness, extra files, lengths and bytes before admitting reuse.
pub fn verify(root: &Path, expected: &Value) -> Result<bool, String> {
    Ok(manifest(root)? == *expected)
}

/// Materialize results, so eviction of disposable state cannot break them.
pub fn copy_tree(from: &Path, into: &Path) -> Result<(), String> {
    fs::create_dir_all(into).map_err(|e| e.to_string())?;
    for path in list(from)? {
        let target = into.join(&path);
        fs::create_dir_all(target.parent().expect("file parent")).map_err(|e| e.to_string())?;
        fs::copy(from.join(path), target).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Logical bytes including incomplete/locked entries; failures are errors.
pub fn bytes(root: &Path) -> Result<u64, String> {
    // Active compilers and atomic publication can rename files during a scan.
    // Retry a failed snapshot rather than reporting missing paths as free space.
    let mut last = None;
    for _ in 0..20 {
        match snapshot_bytes(root) {
            Ok(bytes) => return Ok(bytes),
            Err(error) => last = Some(error),
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    Err(format!(
        "cannot account {} while files change: {}",
        root.display(),
        last.unwrap_or_default()
    ))
}

fn snapshot_bytes(root: &Path) -> Result<u64, String> {
    list(root)?.into_iter().try_fold(0u64, |size, file| {
        size.checked_add(
            fs::metadata(root.join(file))
                .map_err(|e| e.to_string())?
                .len(),
        )
        .ok_or_else(|| "cache size overflow".into())
    })
}
