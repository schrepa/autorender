//! `cargo xtask new app NAME`: a new game from the engine's app template
//! (`crates/engine/examples/app_template.rs`, which lint compiles), its page
//! and its entry, registered by its manifest.

use std::{fs, path::Path};

use crate::process::{clients, run};

pub fn new(root: &Path, options: &[String]) -> Result<(), String> {
    let [kind, name] = options else {
        return Err("usage: cargo xtask new app NAME".into());
    };
    if kind != "app" {
        return Err(format!(
            "cannot make a new {kind:?}; usage: cargo xtask new app NAME"
        ));
    }
    let valid = name.split('-').all(|word| {
        !word.is_empty()
            && word
                .chars()
                .all(|character| character.is_ascii_lowercase() || character.is_ascii_digit())
    }) && name.starts_with(|character: char| character.is_ascii_lowercase());
    if !valid || name.starts_with("engine") || name == "xtask" {
        return Err(format!(
            "{name:?} is not an app name: lowercase words of letters and digits joined by \
             dashes, not starting with `engine`"
        ));
    }
    let wasm = name.replace('-', "_");
    let pascal: String = name
        .split('-')
        .map(|word| word[..1].to_ascii_uppercase() + &word[1..])
        .collect();
    let crate_dir = root.join("apps").join(name);
    let page = format!("{name}.html");
    for taken in [
        crate_dir.clone(),
        root.join("web").join(&page),
        root.join("web/src").join(name),
    ] {
        if taken.exists() {
            return Err(format!("{} already exists", taken.display()));
        }
    }
    if clients()
        .iter()
        .any(|client| client.game.as_deref() == Some(name.as_str()) || client.wasm == wasm)
    {
        return Err(format!("a client named {name} already exists"));
    }

    let template = fs::read_to_string(root.join("crates/engine/examples/app_template.rs"))
        .map_err(|error| format!("cannot read the app template: {error}"))?;
    // The template's own header describes the template.
    let body = template
        .lines()
        .skip_while(|line| line.starts_with("//!"))
        .collect::<Vec<_>>()
        .join("\n");
    let source = format!(
        "//! {pascal}: a game the engine runs on its page, `web/{page}`.\n{}\n",
        body.replace("Template", &pascal)
    );
    let manifest = format!(
        r#"[package]
name = "{name}"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
publish.workspace = true

[lib]
crate-type = ["cdylib", "rlib"]

# A browser game: xtask builds it and serves {page}, which tools select as
# `{name}`.
[package.metadata.engine]
page = "{page}"
game = "{name}"

[dependencies]
engine.workspace = true

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen.workspace = true
wasm-bindgen-futures.workspace = true
web-sys.workspace = true

[lints]
workspace = true
"#
    );
    let html = format!(
        r#"<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{pascal}</title>
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
  </head>
  <body>
    <script type="module" src="/src/{name}/main.ts"></script>
  </body>
</html>
"#
    );
    let entry = format!(
        r#"// {pascal} on the engine's own page: its canvas and statistics.
import {{ loadGame }} from "../engine/bindings";
import {{ startGame }} from "../engine/host";

startGame({{
  async create(canvas) {{
    const {{ default: init, {pascal}Game }} = await loadGame("{wasm}", () => import("../../pkg/{wasm}"));
    await init();
    return {pascal}Game.create(canvas);
  }},
}});
"#
    );
    let files = [
        (crate_dir.join("Cargo.toml"), manifest),
        (crate_dir.join("src/lib.rs"), source),
        (root.join("web").join(&page), html),
        (root.join("web/src").join(name).join("main.ts"), entry),
    ];
    for (path, contents) in &files {
        fs::create_dir_all(path.parent().expect("files have folders"))
            .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
        fs::write(path, contents)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
        println!(
            "Created {}.",
            path.strip_prefix(root).unwrap_or(path).display()
        );
    }
    // Adds the app to Cargo.lock, checks it builds and registers it with
    // the web host.
    run(root, "cargo", &["check", "-p", name])?;
    run(root, "cargo", &["xtask", "schema"])?;
    println!("Run `cargo xtask dev` and open http://127.0.0.1:5173/{page}.");
    Ok(())
}
