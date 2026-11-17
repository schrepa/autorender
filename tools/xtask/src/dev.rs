use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use serde_json::json;

use crate::pipeline::{build_dev, install_web_dependencies};
use crate::process::{clients, slashed};
use crate::serve::Server;

pub const PORT: u16 = 5173;

const USAGE: &str = "usage: cargo xtask dev [--game NAME] [--port N]";
const TREES: [&str; 3] = ["crates", "tools", "vendor"];
const FILES: [&str; 4] = [
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/config.toml",
];
const SKIPPED: [&str; 2] = ["target", "node_modules"];
const LOG: &str = "artifacts/dev/vite.log";
const POLL: Duration = Duration::from_millis(250);
const VITE_START: Duration = Duration::from_secs(60);
const REPLY: Duration = Duration::from_secs(5);
const LISTED: usize = 10;

type Sources = BTreeMap<PathBuf, (Option<SystemTime>, u64)>;

pub fn dev(root: &Path, arguments: &[String]) -> Result<(), String> {
    let (game, port) = options(arguments)?;
    build_dev(root, game)?;
    install_web_dependencies(root)?;
    let server = Server::vite(root, port, &root.join(LOG), VITE_START)?;
    let pages: Vec<_> = clients()
        .iter()
        .filter(|client| game.is_none_or(|game| client.game.as_deref() == Some(game)))
        .filter_map(|client| client.page.as_ref())
        .map(|page| format!("{}/{page}", server.url))
        .collect();
    println!(
        "{}",
        json!({"event": "serving", "pages": pages, "log": LOG})
    );
    let mut seen = sources(root);
    loop {
        thread::sleep(POLL);
        let mut now = sources(root);
        if now == seen {
            continue;
        }
        loop {
            thread::sleep(POLL);
            let settled = sources(root);
            if settled == now {
                break;
            }
            now = settled;
        }
        let changed = changed(root, &seen, &now);
        seen = now;
        let rebuild = changed.iter().any(|path| !path.ends_with(".scene.json"));
        let listed = &changed[..changed.len().min(LISTED)];
        println!(
            "{}",
            json!({"event": "changed", "files": listed, "rebuild": rebuild})
        );
        if !rebuild {
            post(
                port,
                "/_engine/scene",
                &json!({"files": changed}).to_string(),
            )?;
            println!("{}", json!({"event": "scene_sent"}));
        } else if build(root, game) {
            post(port, "/_engine/reload", "")?;
            println!("{}", json!({"event": "reloaded"}));
        }
    }
}

fn options(arguments: &[String]) -> Result<(Option<&str>, u16), String> {
    let (mut game, mut port) = (None, PORT);
    for pair in arguments.chunks(2) {
        match pair {
            [option, value] if option == "--game" => game = Some(value.as_str()),
            [option, value] if option == "--port" => {
                port = value
                    .parse::<u16>()
                    .ok()
                    .filter(|port| *port != 0)
                    .ok_or("dev --port requires an integer from 1 to 65535")?;
            }
            _ => return Err(USAGE.to_owned()),
        }
    }
    Ok((game, port))
}

fn build(root: &Path, game: Option<&str>) -> bool {
    let started = Instant::now();
    let built = build_dev(root, game);
    let elapsed_ms = started.elapsed().as_millis();
    match &built {
        Ok(()) => println!("{}", json!({"event": "built", "elapsed_ms": elapsed_ms})),
        Err(error) => println!("{}", json!({"event": "build_failed", "error": error})),
    }
    built.is_ok()
}

fn post(port: u16, path: &str, body: &str) -> Result<(), String> {
    let failed = |error: std::io::Error| format!("cannot post {path} to Vite: {error}");
    let mut stream = TcpStream::connect(("127.0.0.1", port)).map_err(failed)?;
    stream.set_read_timeout(Some(REPLY)).map_err(failed)?;
    write!(
        stream,
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .map_err(failed)?;
    let mut reply = String::new();
    stream.read_to_string(&mut reply).map_err(failed)?;
    let status = reply.lines().next().unwrap_or_default();
    if !status.starts_with("HTTP/1.1 200") {
        return Err(format!("Vite refused {path}: {status}"));
    }
    Ok(())
}

fn sources(root: &Path) -> Sources {
    let mut found = Sources::new();
    for tree in TREES {
        walk(&root.join(tree), &mut found);
    }
    for file in FILES {
        let path = root.join(file);
        if let Ok(metadata) = fs::metadata(&path) {
            found.insert(path, (metadata.modified().ok(), metadata.len()));
        }
    }
    found
}

fn walk(folder: &Path, found: &mut Sources) {
    let Ok(entries) = fs::read_dir(folder) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') || SKIPPED.iter().any(|skip| name == *skip) {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if metadata.is_dir() {
            walk(&entry.path(), found);
        } else {
            found.insert(entry.path(), (metadata.modified().ok(), metadata.len()));
        }
    }
}

fn changed(root: &Path, before: &Sources, after: &Sources) -> Vec<String> {
    let edited = after
        .iter()
        .filter(|(path, stamp)| before.get(*path) != Some(*stamp))
        .map(|(path, _)| path);
    let removed = before.keys().filter(|path| !after.contains_key(*path));
    edited
        .chain(removed)
        .map(|path| slashed(path.strip_prefix(root).unwrap_or(path)))
        .collect()
}
