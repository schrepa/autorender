//! Content-addressed wasm-bindgen output. The input module, actual generator
//! executable and every option identify a generation independently of a branch.

use crate::{cache::Store, files, jobs};
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};

/// Generate or materialize a complete binding set into the caller's directory.
pub fn generate(
    root: &Path,
    artifact: &Path,
    name: &str,
    extra: &[&str],
    into: &Path,
) -> Result<Value, String> {
    let executable = format!("wasm-bindgen{}", std::env::consts::EXE_SUFFIX);
    let program = std::env::var_os("PATH")
        .and_then(|path| {
            std::env::split_paths(&path)
                .map(|p| p.join(&executable))
                .find(|p| p.is_file())
        })
        .ok_or("wasm-bindgen is not on PATH")?;
    let identity = json!({"version": 1, "wasm_sha256": files::hash(artifact)?, "generator_sha256": files::hash(&program)?, "target": "web", "name": name, "options": extra});
    let key = identity.to_string();
    let store = Store::open(root)?;
    let (entry, hit) = match store.read("bindings", &key)? {
        Some(entry) => (entry, true),
        None => {
            let entry = store.entry("bindings", &key)?;
            if !entry.valid()? {
                let staging = entry.staging()?;
                let generated = staging.join("bindings");
                let mut command = Command::new(program);
                command
                    .current_dir(root)
                    .arg(artifact)
                    .args(["--target", "web", "--out-dir"])
                    .arg(&generated)
                    .args(["--out-name", name])
                    .args(extra);
                let run = jobs::run(&mut command, &staging.join("logs"), &Default::default())?;
                drop(command);
                if !run.status.success() {
                    return Err(format!("wasm-bindgen failed: {}", run.stderr.display()));
                }
                for suffix in [".js", "_bg.wasm", ".d.ts", "_bg.wasm.d.ts"] {
                    if !generated.join(format!("{name}{suffix}")).is_file() {
                        return Err(format!("wasm-bindgen omitted {name}{suffix}"));
                    }
                }
                files::atomic_json(&staging.join("identity.json"), &identity)?;
                entry.publish()?;
            }
            (entry, false)
        }
    };
    fs::create_dir_all(into).map_err(|e| e.to_string())?;
    files::copy_tree(&entry.data().join("bindings"), into)?;
    Ok(json!({"identity": identity, "cache_hit": hit, "lock_ms": entry.wait_ms}))
}
