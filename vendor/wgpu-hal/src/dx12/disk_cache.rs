//! Engine patch (see VENDORED.md in this crate's folder): DXC's output kept
//! on disk, so a later process skips compiling shaders an earlier one
//! compiled. A file holds its whole key (the DXC build, the compile's
//! arguments and the HLSL source) before the DXIL, and is used only when that
//! key matches byte for byte; the file name is only a hash of it. Files are
//! written whole under a temporary name and renamed into place, so processes
//! may share the folder, and the least recently used go beyond a byte budget.
//!
//! `ENGINE_DX12_SHADER_CACHE` names the folder, or `off` turns the cache
//! off; by default it is `wgpu-dx12-shader-cache` in the system's temporary
//! folder.

use alloc::{format, vec::Vec};
use core::hash::{Hash, Hasher};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::SystemTime,
};

/// What a file starts with: the format, changed whenever the key changes.
const MAGIC: &[u8] = b"engine-dxil-cache 1\n";
/// The bytes of files the folder keeps, the most recently used first.
const BUDGET: u64 = 512 * 1024 * 1024;

/// Whether this process has trimmed the folder to its budget.
static TRIMMED: AtomicBool = AtomicBool::new(false);
/// Temporary files' numbers in this process.
static WRITES: AtomicU64 = AtomicU64::new(0);

/// The cache's folder, or none when it is off.
fn folder() -> Option<PathBuf> {
    match std::env::var_os("ENGINE_DX12_SHADER_CACHE") {
        Some(value) if value == "off" => None,
        Some(value) if !value.is_empty() => Some(PathBuf::from(value)),
        _ => Some(std::env::temp_dir().join("wgpu-dx12-shader-cache")),
    }
}

/// One compile's entry: its file and its whole key.
pub(super) struct DiskKey {
    path: PathBuf,
    key: Vec<u8>,
}

impl DiskKey {
    /// The entry of DXC build `compiler` compiling `source`'s `entry_point`
    /// for `full_stage` with the other arguments `arguments` describes;
    /// none when the cache is off.
    pub(super) fn new(
        compiler: &str,
        arguments: &str,
        source_name: Option<&str>,
        entry_point: &str,
        full_stage: &str,
        source: &str,
    ) -> Option<Self> {
        let folder = folder()?;
        let mut key = Vec::with_capacity(source.len() + 256);
        for part in [
            compiler,
            arguments,
            source_name.unwrap_or(""),
            entry_point,
            full_stage,
        ] {
            key.extend_from_slice(part.as_bytes());
            key.push(0);
        }
        key.extend_from_slice(source.as_bytes());
        let mut hasher = std::hash::DefaultHasher::new();
        key.hash(&mut hasher);
        let first = hasher.finish();
        source.len().hash(&mut hasher);
        let name = format!("{first:016x}{:016x}.dxil", hasher.finish());
        Some(Self {
            path: folder.join(name),
            key,
        })
    }

    /// The DXIL kept for this key, if a file holds exactly it; marks the file
    /// used.
    pub(super) fn load(&self) -> Option<Vec<u8>> {
        let bytes = fs::read(&self.path).ok()?;
        let rest = bytes.strip_prefix(MAGIC)?;
        let (length, rest) = rest.split_first_chunk::<8>()?;
        let length = usize::try_from(u64::from_le_bytes(*length)).ok()?;
        if rest.len() < length || rest[..length] != self.key[..] {
            return None;
        }
        let dxil = &rest[length..];
        // A DXIL container starts with its magic.
        if !dxil.starts_with(b"DXBC") {
            return None;
        }
        if let Ok(file) = fs::File::options().write(true).open(&self.path) {
            let _ = file.set_modified(SystemTime::now());
        }
        Some(dxil.to_vec())
    }

    /// Keeps `dxil` for this key, trimming the folder to its budget the first
    /// time in this process. Failures only cost later processes the compile.
    pub(super) fn store(&self, dxil: &[u8]) {
        let Some(folder) = self.path.parent() else {
            return;
        };
        if fs::create_dir_all(folder).is_err() {
            return;
        }
        if !TRIMMED.swap(true, Ordering::Relaxed) {
            trim(folder);
        }
        let temporary = self.path.with_extension(format!(
            "{}-{}.part",
            std::process::id(),
            WRITES.fetch_add(1, Ordering::Relaxed)
        ));
        let written = fs::File::create(&temporary).and_then(|mut file| {
            file.write_all(MAGIC)?;
            file.write_all(&(self.key.len() as u64).to_le_bytes())?;
            file.write_all(&self.key)?;
            file.write_all(dxil)
        });
        if written.is_err() || fs::rename(&temporary, &self.path).is_err() {
            let _ = fs::remove_file(&temporary);
        }
    }
}

/// Removes the least recently used files of `folder` beyond the budget, and
/// temporary files a stopped process left a day ago or more.
fn trim(folder: &std::path::Path) {
    let Ok(entries) = fs::read_dir(folder) else {
        return;
    };
    let now = SystemTime::now();
    let mut files: Vec<(SystemTime, u64, PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let modified = metadata.modified().unwrap_or(now);
        let path = entry.path();
        match path.extension().and_then(|extension| extension.to_str()) {
            Some("dxil") => files.push((modified, metadata.len(), path)),
            Some("part")
                if now
                    .duration_since(modified)
                    .is_ok_and(|age| age.as_secs() > 24 * 60 * 60) =>
            {
                let _ = fs::remove_file(path);
            }
            _ => {}
        }
    }
    files.sort_by(|a, b| b.0.cmp(&a.0));
    let mut kept = 0u64;
    for (_, length, path) in files {
        kept += length;
        if kept > BUDGET {
            let _ = fs::remove_file(path);
        }
    }
}

/// The bytes of a compiled shader.
pub(super) fn bytes(shader: &super::CompiledShader) -> &[u8] {
    use super::CompiledShader;
    match shader {
        CompiledShader::Dxc(blob) => unsafe {
            core::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize())
        },
        CompiledShader::Fxc(blob) => unsafe {
            core::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize())
        },
        CompiledShader::Precompiled(bytes) => bytes,
    }
}
