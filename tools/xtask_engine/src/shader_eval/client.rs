use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntryMetrics {
    pub runs_per_frame: f64,
    pub ops_per_run: f64,
    pub samples_per_run: f64,
    pub m_ops_per_frame: f64,
    pub m_samples_per_frame: f64,
}

#[derive(Clone, Debug, Default)]
pub struct WorkSample {
    pub frames: u64,
    pub entries: BTreeMap<String, EntryMetrics>,
}

#[derive(Clone, Debug, Default)]
pub struct PassSample {
    pub frames: u64,
    pub passes_ms: BTreeMap<String, f64>,
    pub unavailable: Option<String>,
}

pub struct SpawnConfig<'a> {
    pub root: &'a Path,
    pub host: &'a str,
    pub game: &'a str,
    pub build: &'a str,
    pub size: (u32, u32),
    pub window: bool,
    pub run_name: &'a str,
}

pub struct AgentClient {
    writer: TcpStream,
    reader: BufReader<TcpStream>,
    folder: PathBuf,
    adapter: Value,
    owned: Option<Child>,
}

impl Drop for AgentClient {
    fn drop(&mut self) {
        let Some(mut child) = self.owned.take() else {
            return;
        };
        let close = json!({"verb": "call", "name": "host.close", "args": {}});
        let _ = writeln!(self.writer, "{close}");
        let _ = self.writer.flush();
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if matches!(child.try_wait(), Ok(Some(_))) {
                return;
            }
            thread::sleep(Duration::from_millis(25));
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

impl AgentClient {
    pub fn spawn(config: SpawnConfig<'_>) -> Result<Self, String> {
        let mut command = xtask_command(config.root);
        command
            .args([
                "agent",
                "start",
                "--host",
                config.host,
                "--game",
                config.game,
                "--build",
                config.build,
                "--size",
                &format!("{}x{}", config.size.0, config.size.1),
                "--run",
                config.run_name,
                "--deadline-ms",
                "60000",
                "--reply-limit",
                "4194304",
            ])
            .current_dir(config.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if config.window {
            command.arg("--window");
        }
        let mut child = command
            .spawn()
            .map_err(|error| format!("cannot start `xtask agent start`: {error}"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "missing agent stdout pipe".to_owned())?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| "missing agent stderr pipe".to_owned())?;
        let logs = Arc::new(Mutex::new(Vec::<String>::new()));
        let (ready_tx, ready_rx) = mpsc::channel::<Value>();
        spawn_stdout_reader(stdout, ready_tx, Arc::clone(&logs));
        spawn_stderr_reader(stderr, Arc::clone(&logs));
        let session = match ready_rx.recv() {
            Ok(session) => session,
            Err(_) => {
                let status = child.wait().ok();
                let tail = recent_logs(&logs);
                return Err(format!(
                    "agent session exited before becoming ready (status {status:?}):\n{tail}"
                ));
            }
        };
        Self::connect(config.root, &session, Some(child))
    }

    pub fn attach(root: &Path, run_name: &str) -> Result<Self, String> {
        let file = root.join("artifacts/runs").join(run_name).join("session.json");
        let text = fs::read_to_string(&file)
            .map_err(|error| format!("cannot read {}: {error}", file.display()))?;
        let session: Value = serde_json::from_str(&text)
            .map_err(|error| format!("invalid {}: {error}", file.display()))?;
        Self::connect(root, &session, None)
    }

    fn connect(root: &Path, session: &Value, owned: Option<Child>) -> Result<Self, String> {
        let port = session["port"]
            .as_u64()
            .and_then(|port| u16::try_from(port).ok())
            .ok_or_else(|| format!("session missing valid port: {session}"))?;
        let folder = session["folder"]
            .as_str()
            .map(|path| {
                let candidate = PathBuf::from(path);
                if candidate.is_absolute() {
                    candidate
                } else {
                    root.join(candidate)
                }
            })
            .ok_or_else(|| format!("session missing folder: {session}"))?;
        let adapter = if !session["adapter"].is_null() {
            session["adapter"].clone()
        } else {
            session["identity"]["adapter"].clone()
        };
        let stream = TcpStream::connect(("127.0.0.1", port))
            .map_err(|error| format!("cannot connect to agent on port {port}: {error}"))?;
        let reader_stream = stream
            .try_clone()
            .map_err(|error| format!("cannot clone TCP stream: {error}"))?;
        Ok(Self {
            writer: stream,
            reader: BufReader::new(reader_stream),
            folder,
            adapter,
            owned,
        })
    }

    pub fn adapter(&self) -> &Value {
        &self.adapter
    }

    pub fn send(&mut self, request: &Value) -> Result<Value, String> {
        let line = serde_json::to_string(request).map_err(|error| error.to_string())?;
        writeln!(self.writer, "{line}").map_err(|error| format!("write to agent failed: {error}"))?;
        self.writer
            .flush()
            .map_err(|error| format!("flush to agent failed: {error}"))?;
        let mut raw = String::new();
        let read = self
            .reader
            .read_line(&mut raw)
            .map_err(|error| format!("read from agent failed: {error}"))?;
        if read == 0 {
            return Err(format!("agent closed connection while answering {line}"));
        }
        let reply: Value = serde_json::from_str(raw.trim())
            .map_err(|error| format!("invalid agent JSON reply ({error}): {raw}"))?;
        if reply["ok"] != true {
            return Err(format!("agent refused {line}: {}", reply["error"]));
        }
        if reply.get("value").is_some() {
            return Ok(reply["value"].clone());
        }
        if let Some(file) = reply["file"].as_str() {
            let text = fs::read_to_string(file)
                .map_err(|error| format!("cannot read spilled reply {file}: {error}"))?;
            let spilled: Value = serde_json::from_str(&text)
                .map_err(|error| format!("invalid spilled reply {file}: {error}"))?;
            return Ok(spilled["value"].clone());
        }
        Ok(Value::Null)
    }

    pub fn set_instruments(&mut self, instruments: &[&str]) -> Result<Value, String> {
        self.send(&json!({
            "verb": "set",
            "name": "instruments",
            "args": { "value": instruments },
        }))
    }

    pub fn step_settled(&mut self, frames: u32, ticks_per_frame: u32) -> Result<Value, String> {
        self.send(&json!({
            "verb": "call",
            "name": "step",
            "args": { "until": "settled", "frames": frames, "ticks_per_frame": ticks_per_frame },
            "deadline_ms": 60000,
        }))
    }

    pub fn step_frames(&mut self, frames: u32, ticks_per_frame: u32) -> Result<Value, String> {
        self.send(&json!({
            "verb": "call",
            "name": "step",
            "args": { "frames": frames, "ticks_per_frame": ticks_per_frame },
            "deadline_ms": 60000,
        }))
    }

    pub fn reload_shaders(&mut self) -> Result<Value, String> {
        let reply = self.send(&json!({
            "verb": "call",
            "name": "gpu.shaders.reload",
            "args": {},
            "deadline_ms": 60000,
        }))?;
        let _ = self.step_settled(240, 0)?;
        Ok(reply)
    }

    pub fn capture_to(
        &mut self,
        (width, height): (u32, u32),
        destination: &Path,
    ) -> Result<(u32, u32, String), String> {
        let reply = self.send(&json!({
            "verb": "call",
            "name": "capture",
            "args": { "width": width, "height": height },
            "deadline_ms": 30000,
        }))?;
        if let Some(reason) = reply["unavailable"].as_str() {
            return Err(format!("capture unavailable: {reason}"));
        }
        let file = reply["path"]
            .as_str()
            .ok_or_else(|| format!("capture reply missing path: {reply}"))?;
        let captured_width = reply["width"].as_u64().unwrap_or(u64::from(width)) as u32;
        let captured_height = reply["height"].as_u64().unwrap_or(u64::from(height)) as u32;
        let sha256 = reply["sha256"].as_str().unwrap_or_default().to_owned();
        let source = self.folder.join(file);
        wait_for_png(&source, Duration::from_secs(3))?;
        fs::copy(&source, destination).map_err(|error| {
            format!(
                "cannot copy {} to {}: {error}",
                source.display(),
                destination.display()
            )
        })?;
        Ok((captured_width, captured_height, sha256))
    }

    pub fn measure_work(&mut self, frames: u32) -> Result<WorkSample, String> {
        self.set_instruments(&["gpu.pipelines.work"])?;
        self.step_settled(240, 0)?;
        self.step_frames(5, 0)?;
        let f0 = self.wait_work_ready(None)?;
        let runs0 = self.read_totals("gpu.pipelines.work.runs")?;
        let ops0 = self.read_totals("gpu.pipelines.work.operations")?;
        let samples0 = self.read_totals("gpu.pipelines.work.samples")?;
        self.step_frames(frames, 0)?;
        let f1 = self.wait_work_ready(Some(f0 + u64::from(frames)))?;
        let runs1 = self.read_totals("gpu.pipelines.work.runs")?;
        let ops1 = self.read_totals("gpu.pipelines.work.operations")?;
        let samples1 = self.read_totals("gpu.pipelines.work.samples")?;
        self.set_instruments(&[])?;
        let measured = f1.saturating_sub(f0).max(1);
        let mut entries = BTreeMap::new();
        for (name, after_runs) in runs1 {
            let delta_runs = after_runs.saturating_sub(runs0.get(&name).copied().unwrap_or(0));
            if delta_runs == 0 {
                continue;
            }
            let delta_ops = ops1
                .get(&name)
                .copied()
                .unwrap_or(0)
                .saturating_sub(ops0.get(&name).copied().unwrap_or(0));
            let delta_samples = samples1
                .get(&name)
                .copied()
                .unwrap_or(0)
                .saturating_sub(samples0.get(&name).copied().unwrap_or(0));
            let runs_f = delta_runs as f64;
            let frames_f = measured as f64;
            entries.insert(
                name,
                EntryMetrics {
                    runs_per_frame: runs_f / frames_f,
                    ops_per_run: delta_ops as f64 / runs_f,
                    samples_per_run: delta_samples as f64 / runs_f,
                    m_ops_per_frame: (delta_ops as f64 / frames_f) / 1.0e6,
                    m_samples_per_frame: (delta_samples as f64 / frames_f) / 1.0e6,
                },
            );
        }
        Ok(WorkSample {
            frames: measured,
            entries,
        })
    }

    fn wait_work_ready(&mut self, min_frame: Option<u64>) -> Result<u64, String> {
        for _ in 0..40 {
            let state = self.send(&json!({
                "verb": "read",
                "name": "gpu.pipelines.work",
                "args": {},
            }))?;
            if let Some(through) = state["through_frame"].as_u64() {
                let mut compiling = false;
                let mut max_from = 0_u64;
                if let Some(entries) = state["entries"].as_object() {
                    for info in entries.values() {
                        if info["unavailable"].as_str() == Some("compiling") {
                            compiling = true;
                        }
                        if let Some(from) = info["from_frame"].as_u64() {
                            max_from = max_from.max(from);
                        }
                    }
                }
                let reached = min_frame.is_none_or(|target| through >= target);
                if !compiling && through >= max_from && reached {
                    return Ok(through);
                }
            }
            self.step_frames(1, 0)?;
        }
        Err("timed out waiting for gpu.pipelines.work readback".to_owned())
    }

    fn read_totals(&mut self, name: &str) -> Result<BTreeMap<String, u64>, String> {
        let value = self.send(&json!({
            "verb": "read",
            "name": name,
            "args": {},
        }))?;
        let mut out = BTreeMap::new();
        if let Some(map) = value.as_object() {
            for (entry, data) in map {
                if let Some(total) = data["total"].as_u64() {
                    out.insert(entry.clone(), total);
                }
            }
        }
        Ok(out)
    }

    pub fn measure_passes(&mut self, frames: u32) -> Result<PassSample, String> {
        if let Err(error) = self.set_instruments(&["gpu.passes.split"]) {
            return Ok(PassSample {
                frames: 0,
                passes_ms: BTreeMap::new(),
                unavailable: Some(error),
            });
        }
        self.step_frames(15, 0)?;
        if !self.wait_passes_ready()? {
            self.set_instruments(&[])?;
            return Ok(PassSample {
                frames: 0,
                passes_ms: BTreeMap::new(),
                unavailable: Some("gpu.passes.split produced no timed frames".to_owned()),
            });
        }
        let time0 = self.read_u64_map("gpu.passes.time")?;
        let timed0 = self.read_u64_map("gpu.passes.timed")?;
        let start_max = timed0.values().copied().max().unwrap_or(0);
        self.step_frames(frames, 0)?;
        let target = start_max + u64::from(frames);
        for _ in 0..30 {
            let timed_now = self.read_u64_map("gpu.passes.timed")?;
            if timed_now.values().copied().max().unwrap_or(0) >= target {
                break;
            }
            self.step_frames(1, 0)?;
        }
        let time1 = self.read_u64_map("gpu.passes.time")?;
        let timed1 = self.read_u64_map("gpu.passes.timed")?;
        self.set_instruments(&[])?;
        let mut passes_ms = BTreeMap::new();
        let mut max_measured = 0_u64;
        for (span, after_ns) in time1 {
            let delta_frames = timed1
                .get(&span)
                .copied()
                .unwrap_or(0)
                .saturating_sub(timed0.get(&span).copied().unwrap_or(0));
            if delta_frames == 0 {
                continue;
            }
            let delta_ns = after_ns.saturating_sub(time0.get(&span).copied().unwrap_or(0));
            max_measured = max_measured.max(delta_frames);
            passes_ms.insert(span, (delta_ns as f64 / delta_frames as f64) / 1.0e6);
        }
        Ok(PassSample {
            frames: max_measured,
            passes_ms,
            unavailable: None,
        })
    }

    fn wait_passes_ready(&mut self) -> Result<bool, String> {
        for _ in 0..25 {
            let groups = self.send(&json!({
                "verb": "read",
                "name": "gpu.passes.groups",
                "args": {},
            }))?;
            if groups.get("unavailable").is_some() {
                return Ok(false);
            }
            if groups["through_frame"].as_u64().is_some() {
                return Ok(true);
            }
            self.step_frames(1, 0)?;
        }
        Ok(false)
    }

    fn read_u64_map(&mut self, name: &str) -> Result<BTreeMap<String, u64>, String> {
        let value = self.send(&json!({
            "verb": "read",
            "name": name,
            "args": {},
        }))?;
        let mut out = BTreeMap::new();
        if let Some(map) = value.as_object() {
            for (key, item) in map {
                if let Some(number) = item.as_u64() {
                    out.insert(key.clone(), number);
                }
            }
        }
        Ok(out)
    }
}

fn xtask_command(root: &Path) -> Command {
    let exe_name = if cfg!(windows) { "xtask.exe" } else { "xtask" };
    if let Ok(current) = std::env::current_exe()
        && let Some(folder) = current.parent()
    {
        let sibling = folder.join(exe_name);
        if sibling.is_file() {
            return Command::new(sibling);
        }
    }
    let debug_bin = root.join("target/debug").join(exe_name);
    if debug_bin.is_file() {
        return Command::new(debug_bin);
    }
    let mut cargo = Command::new("cargo");
    cargo.args(["run", "-p", "xtask", "--"]);
    cargo
}

fn spawn_stdout_reader(
    stdout: std::process::ChildStdout,
    ready_tx: mpsc::Sender<Value>,
    logs: Arc<Mutex<Vec<String>>>,
) {
    thread::spawn(move || {
        let mut sent = false;
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if !sent
                && let Ok(parsed) = serde_json::from_str::<Value>(&line)
                && parsed["event"].as_str() == Some("ready")
            {
                let _ = ready_tx.send(parsed["session"].clone());
                sent = true;
            }
            push_log(&logs, line);
        }
    });
}

fn spawn_stderr_reader(stderr: std::process::ChildStderr, logs: Arc<Mutex<Vec<String>>>) {
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            push_log(&logs, line);
        }
    });
}

fn push_log(logs: &Arc<Mutex<Vec<String>>>, line: String) {
    if let Ok(mut guard) = logs.lock() {
        if guard.len() >= 80 {
            guard.remove(0);
        }
        guard.push(line);
    }
}

fn recent_logs(logs: &Arc<Mutex<Vec<String>>>) -> String {
    logs.lock()
        .map(|guard| guard.join("\n"))
        .unwrap_or_default()
}

fn wait_for_png(path: &Path, timeout: Duration) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    loop {
        if path.is_file() && super::image::read_png(path).is_ok() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "captured image {} did not arrive within {} ms",
                path.display(),
                timeout.as_millis()
            ));
        }
        thread::sleep(Duration::from_millis(25));
    }
}
