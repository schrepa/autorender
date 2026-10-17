//! Native mark timing with durable raw output and an owned process deadline.
//! Marks are timestamped in the reader as they arrive, before parsing or I/O.

use engine_profiler::load::{Load, Monitor};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant, SystemTime},
};

pub(crate) struct HostRun {
    pub marks: BTreeMap<String, Instant>,
    pub report: Value,
    pub load: Load,
    pub stdout: String,
    pub stderr: String,
    pub success: bool,
}

pub(crate) fn run_host(
    root: &Path,
    program: &Path,
    arguments: &[String],
    environment: &[(&str, String)],
    raw: &Path,
    deadline: Option<SystemTime>,
) -> Result<HostRun, String> {
    fs::create_dir_all(raw).map_err(|e| e.to_string())?;
    let mut out_file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(raw.join("stdout.log"))
        .map_err(|e| e.to_string())?;
    let mut err_file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(raw.join("stderr.log"))
        .map_err(|e| e.to_string())?;
    let mut command = Command::new(program);
    command
        .args(arguments)
        .envs(environment.iter().map(|(name, value)| (*name, value)))
        .current_dir(root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    xtask::jobs::owned(&mut command);
    let mut child = command.spawn().map_err(|e| e.to_string())?;
    let pid = child.id();
    let mut errors = child.stderr.take().expect("piped stderr");
    let stderr = std::thread::spawn(move || {
        std::io::copy(&mut errors, &mut err_file).map_err(|e| e.to_string())
    });
    let stdout = child.stdout.take().expect("piped stdout");
    let reader = std::thread::spawn(move || read_marks(stdout, Monitor::new(pid), &mut out_file));
    let start = Instant::now();
    let timeout = deadline
        .map(|end| end.duration_since(SystemTime::now()).unwrap_or_default())
        .unwrap_or(Duration::from_secs(30 * 60));
    let mut stopped = None;
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            break Some(status);
        }
        let cancelled =
            std::env::var_os("ENGINE_CANCEL_FILE").is_some_and(|path| Path::new(&path).exists());
        if cancelled || start.elapsed() >= timeout {
            xtask::jobs::stop(&mut child)?;
            stopped = Some(if cancelled {
                "cancelled"
            } else {
                "deadline exceeded"
            });
            break None;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let mut run = reader.join().map_err(|_| "mark reader panicked")??;
    stderr.join().map_err(|_| "stderr reader panicked")??;
    if let Some(reason) = stopped {
        return Err(format!(
            "native run {reason}; raw output in {}",
            raw.display()
        ));
    }
    run.stderr = fs::read_to_string(raw.join("stderr.log")).map_err(|e| e.to_string())?;
    run.success = status.is_some_and(|status| status.success());
    Ok(run)
}

fn read_marks(
    stdout: std::process::ChildStdout,
    mut monitor: Monitor,
    raw: &mut fs::File,
) -> Result<HostRun, String> {
    let mut run = HostRun {
        marks: BTreeMap::new(),
        report: Value::Null,
        load: Load::default(),
        stdout: String::new(),
        stderr: String::new(),
        success: false,
    };
    let mut recorded = Vec::new();
    for line in BufReader::new(stdout).lines() {
        let at = Instant::now();
        let line = line.map_err(|e| e.to_string())?;
        // Small mark lines remain in memory until the timed interval ends.
        // Disk writes never enter the serialized/pipelined timing interval.
        recorded.extend_from_slice(line.as_bytes());
        recorded.push(b'\n');
        if let Ok(value) = serde_json::from_str::<Value>(&line) {
            if let Some(name) = value["mark"].as_str() {
                match name {
                    "serialized-start" => monitor.start(),
                    "pipelined-end" => {
                        run.load = monitor.end();
                        raw.write_all(&recorded).map_err(|e| e.to_string())?;
                        raw.flush().map_err(|e| e.to_string())?;
                        recorded.clear();
                    }
                    _ => {}
                }
                run.marks.insert(name.to_owned(), at);
                continue;
            }
            if value["report"].is_object() {
                run.report = value["report"].clone();
                continue;
            }
        }
        run.stdout.push_str(&line);
        run.stdout.push('\n');
    }
    raw.write_all(&recorded).map_err(|e| e.to_string())?;
    raw.sync_all().map_err(|e| e.to_string())?;
    Ok(run)
}
