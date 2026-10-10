//! The engine factory's checks: smoke scripts (`web/scripts/NAME.mjs`) run
//! against a development server of one commit, with the scripts of the
//! same or another commit, every attempt kept. A check holds when its
//! first attempt does what it must (exit 0, or 3 for an assertion where it
//! must fail); one that does not runs again, up to `ATTEMPTS` in all, so a
//! failure followed by a pass shows as flaky. `cargo xtask checks` runs a
//! check task; measure tasks run theirs after their workloads.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::bench::hex;
use crate::measure::{checkout, commit_id, git, iso, output, read_json};

/// Attempts a check gets in all.
const ATTEMPTS: usize = 3;
/// How long a development server may take to build and start.
const SERVER_START: Duration = Duration::from_secs(30 * 60);

/// What a check must do: exit 0, or 3 (a failed assertion).
fn holds(expect: &str, exit_code: i32) -> bool {
    match expect {
        "fail" => exit_code == 3,
        _ => exit_code == 0,
    }
}

/// A free port on this machine, which the operating system picks.
fn free_port() -> Result<u16, String> {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .map(|address| address.port())
        .map_err(|error| format!("no free port: {error}"))
}

/// A development server of one commit, stopped with its processes when
/// dropped.
struct Server {
    child: Child,
    url: String,
    build: Value,
}

impl Drop for Server {
    fn drop(&mut self) {
        let pid = self.child.id().to_string();
        // Cargo starts npm, which starts Vite: the whole tree.
        if cfg!(windows) {
            let _ = Command::new("taskkill")
                .args(["/T", "/F", "/PID", &pid])
                .output();
        } else {
            let _ = Command::new("kill")
                .args(["-TERM", "--", &format!("-{pid}")])
                .output();
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Starts `cargo xtask dev` in `source` on a free port and waits until it
/// serves: its build of the page logged into `log`.
fn start_server(source: &Path, sha: &str, log: &Path) -> Result<Server, String> {
    let port = free_port()?;
    let file = fs::File::create(log).map_err(|error| error.to_string())?;
    let errors = file.try_clone().map_err(|error| error.to_string())?;
    let mut command = Command::new("cargo");
    command
        .args(["xtask", "dev", "--port", &port.to_string()])
        .current_dir(source)
        .stdout(file)
        .stderr(errors);
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut command, 0);
    let started = Instant::now();
    let child = command
        .spawn()
        .map_err(|error| format!("cannot start the development server: {error}"))?;
    let mut server = Server {
        child,
        url: format!("http://127.0.0.1:{port}"),
        build: Value::Null,
    };
    loop {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            break;
        }
        if let Some(status) = server.child.try_wait().map_err(|error| error.to_string())? {
            return Err(format!(
                "the development server of {sha} stopped ({status}); see {}",
                log.display()
            ));
        }
        if started.elapsed() > SERVER_START {
            return Err(format!(
                "the development server of {sha} did not start within {} minutes",
                SERVER_START.as_secs() / 60
            ));
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    server.build = web_build(source, sha, started.elapsed())?;
    Ok(server)
}

/// What a development server serves: the commit's page and its Wasm
/// clients with their development tools.
fn web_build(source: &Path, sha: &str, took: Duration) -> Result<Value, String> {
    let mut wasm: Vec<PathBuf> = fs::read_dir(source.join("web/pkg-dev"))
        .map_err(|error| format!("cannot read web/pkg-dev: {error}"))?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "wasm")
        })
        .collect();
    wasm.sort();
    let mut hash = Sha256::new();
    for path in &wasm {
        let bytes = fs::read(path).map_err(|error| error.to_string())?;
        hash.update(path.file_name().unwrap_or_default().as_encoded_bytes());
        hash.update([0]);
        hash.update(&bytes);
    }
    let wasm_bindgen = output(source, "wasm-bindgen", &["--version"]).unwrap_or_default();
    Ok(json!({
        "sha": sha,
        "profile": "release",
        "target": "web-wasm",
        "features": ["engine/dev"],
        "toolchain": {
            "rustc": output(source, "rustc", &["-V"])?,
            "cargo": output(source, "cargo", &["-V"])?,
            "node": output(source, "node", &["--version"])?,
            "wasm-bindgen": wasm_bindgen,
        },
        "env": {"RUSTFLAGS": std::env::var("RUSTFLAGS").unwrap_or_default()},
        "artifact_hash": hex(&hash.finalize()),
        "build_ms": took.as_secs_f64() * 1000.0,
    }))
}

/// A build that did not finish: its commit, and as its artifact the log
/// that says why.
fn failed_build(source: &Path, sha: &str, log: &Path) -> Result<Value, String> {
    Ok(json!({
        "sha": sha,
        "profile": "release",
        "target": "web-wasm",
        "features": ["engine/dev"],
        "toolchain": {
            "rustc": output(source, "rustc", &["-V"])?,
            "cargo": output(source, "cargo", &["-V"])?,
        },
        "env": {"RUSTFLAGS": std::env::var("RUSTFLAGS").unwrap_or_default()},
        "artifact_hash": crate::measure::file_hash(log)?,
        "build_ms": 0,
    }))
}

/// The web folder whose scripts run: `scripts_sha`'s, its dependencies
/// installed, or the served commit's own.
fn scripts_folder(
    root: &Path,
    source: &Path,
    scripts_sha: Option<&str>,
) -> Result<PathBuf, String> {
    let Some(sha) = scripts_sha else {
        return Ok(source.join("web"));
    };
    let scripts = checkout(root, sha, None)?;
    crate::pipeline::install_web_dependencies(&scripts)?;
    Ok(scripts.join("web"))
}

/// Runs `checks` (the task's: name, sha, scripts_sha, expect), each
/// commit's server started once, into `folder`; stops adding attempts at
/// `deadline`. The checks' entries, and whether the deadline cut any short.
pub fn run(
    root: &Path,
    checks: &[Value],
    folder: &Path,
    deadline: Option<SystemTime>,
) -> Result<(Vec<Value>, bool), String> {
    fs::create_dir_all(folder).map_err(|error| error.to_string())?;
    let mut by_commit: BTreeMap<&str, Vec<&Value>> = BTreeMap::new();
    for check in checks {
        let sha = commit_id(check["sha"].as_str(), "a check's sha")?;
        if !check["scripts_sha"].is_null() {
            commit_id(check["scripts_sha"].as_str(), "a check's scripts_sha")?;
        }
        by_commit.entry(sha).or_default().push(check);
    }
    let (mut entries, mut incomplete) = (Vec::new(), false);
    for (sha, checks) in by_commit {
        let source = checkout(root, sha, None)?;
        eprintln!("Starting the development server of {sha}.");
        let log = folder.join(format!("server-{}.log", &sha[..12]));
        let server = start_server(&source, sha, &log);
        for check in checks {
            let name = check["name"].as_str().ok_or("a check has a name")?;
            let expect = check["expect"].as_str().unwrap_or("pass");
            let scripts_sha = check["scripts_sha"].as_str();
            let mut attempts = Vec::new();
            let build = match &server {
                Ok(server) => server.build.clone(),
                Err(_) => failed_build(&source, sha, &log)?,
            };
            let web = match &server {
                Ok(_) => Some(scripts_folder(root, &source, scripts_sha)?),
                Err(_) => None,
            };
            while attempts.len() < ATTEMPTS {
                if !attempts.is_empty()
                    && deadline.is_some_and(|deadline| SystemTime::now() > deadline)
                {
                    incomplete = true;
                    break;
                }
                let started_at = iso(SystemTime::now());
                let started = Instant::now();
                let attempt_log =
                    folder.join(format!("{name}-{}-{}.log", &sha[..12], attempts.len() + 1));
                let exit_code = match (&server, &web) {
                    (Ok(server), Some(web)) => attempt(web, name, &server.url, &attempt_log)?,
                    (Err(error), _) => {
                        fs::write(&attempt_log, format!("{error}\n"))
                            .map_err(|error| error.to_string())?;
                        -1
                    }
                    (Ok(_), None) => unreachable!("a served commit has its scripts"),
                };
                eprintln!(
                    "Check {name} on {}: attempt {} exited {exit_code}.",
                    &sha[..12],
                    attempts.len() + 1
                );
                attempts.push(json!({
                    "started_at": started_at,
                    "duration_ms": started.elapsed().as_secs_f64() * 1000.0,
                    "exit_code": exit_code,
                    "log": attempt_log.display().to_string(),
                }));
                // Nothing to run again without a server.
                if holds(expect, exit_code) || server.is_err() {
                    break;
                }
            }
            entries.push(json!({
                "name": name,
                "sha": sha,
                "scripts_sha": scripts_sha,
                "expect": expect,
                "build": build,
                "attempts": attempts,
            }));
        }
        drop(server);
    }
    Ok((entries, incomplete))
}

/// One attempt of check `name` from `web` against `url`: its exit code,
/// its output into `log`.
fn attempt(web: &Path, name: &str, url: &str, log: &Path) -> Result<i32, String> {
    let script = format!("scripts/{name}.mjs");
    if !web.join(&script).is_file() {
        fs::write(log, format!("{script} does not exist\n")).map_err(|error| error.to_string())?;
        return Ok(-1);
    }
    let mut child = Command::new("node")
        .arg(&script)
        .env("DEMO_URL", url)
        .current_dir(web)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot run node: {error}"))?;
    let mut errors = child.stderr.take().expect("piped");
    let stderr = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = errors.read_to_string(&mut text);
        text
    });
    let mut stdout = String::new();
    let _ = child
        .stdout
        .take()
        .expect("piped")
        .read_to_string(&mut stdout);
    let status = child.wait().map_err(|error| error.to_string())?;
    let stderr = stderr.join().unwrap_or_default();
    fs::write(log, format!("{stdout}\n{stderr}")).map_err(|error| error.to_string())?;
    Ok(status.code().unwrap_or(-1))
}

/// `cargo xtask checks`: a check task (its pull request merged into main,
/// `sha`, and main, `main_sha`) as the factory's check report.
pub fn check_task(root: &Path, arguments: &[String]) -> Result<(), String> {
    let usage = "usage: cargo xtask checks --task TASK.json --profile PROFILE.json --out REPORT.json [--formats FORMATS.json]";
    let mut found: BTreeMap<&str, PathBuf> = BTreeMap::new();
    let mut given = arguments.iter();
    while let Some(name) = given.next() {
        let key = match name.as_str() {
            "--task" | "--profile" | "--out" | "--formats" => &name[2..],
            _ => return Err(format!("unknown option {name}\n{usage}")),
        };
        let value = given
            .next()
            .ok_or_else(|| format!("{name} takes a path\n{usage}"))?;
        found.insert(key, PathBuf::from(value));
    }
    let [Some(task), Some(profile), Some(out)] =
        ["task", "profile", "out"].map(|key| found.remove(key))
    else {
        return Err(usage.into());
    };
    let task = read_json(&task)?;
    let spec = if task["spec"].is_object() {
        &task["spec"]
    } else {
        &task
    };
    let id = task["id"].as_u64().ok_or("the task has no id")?;
    let profile_start = read_json(&profile)?;
    let tools_sha = git(root, &["rev-parse", "HEAD"])?;
    if let Some(wanted) = spec["tools_sha"]
        .as_str()
        .filter(|wanted| *wanted != tools_sha)
    {
        return Err(format!(
            "the tools come from {wanted}; this checkout is {tools_sha} (git checkout --detach {wanted})"
        ));
    }
    let setup = match &spec["setup"] {
        Value::String(id) => id.clone(),
        setup => setup["id"].as_str().unwrap_or("unknown").to_owned(),
    };
    let deadline = spec["deadline"]
        .as_str()
        .and_then(crate::measure::parse_iso);
    let folder = root
        .join("target")
        .join("measure")
        .join("checks")
        .join(id.to_string());
    let checks = spec["checks"].as_array().cloned().unwrap_or_default();
    let (checks, incomplete) = run(root, &checks, &folder, deadline)?;
    let report = json!({
        "kind": "check",
        "task": id,
        "tools_sha": tools_sha,
        "profile_start": profile_start,
        "profile_end": read_json(&profile)?,
        "setup": setup,
        "sha": spec["sha"],
        "main_sha": spec["main_sha"],
        "checks": checks,
        "incomplete": incomplete,
    });
    fs::write(
        &out,
        serde_json::to_string_pretty(&report).expect("reports serialize") + "\n",
    )
    .map_err(|error| format!("cannot write {}: {error}", out.display()))?;
    for check in report["checks"].as_array().into_iter().flatten() {
        let codes: Vec<String> = check["attempts"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|attempt| attempt["exit_code"].to_string())
            .collect();
        println!(
            "{} ({}): exited {}",
            check["name"].as_str().unwrap_or_default(),
            check["expect"].as_str().unwrap_or_default(),
            codes.join(", then ")
        );
    }
    println!("Wrote {}.", out.display());
    match found.remove("formats") {
        Some(formats) => crate::measure::check_format(&report, &formats, "check"),
        None => Ok(()),
    }
}
