use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

pub struct Server {
    child: Child,
    pub url: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        if cfg!(windows) {
            let _ = Command::new("taskkill")
                .args(["/T", "/F", "/PID", &self.child.id().to_string()])
                .output();
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Server {
    pub fn vite(root: &Path, port: u16, log: &Path, limit: Duration) -> Result<Self, String> {
        if let Some(folder) = log.parent() {
            fs::create_dir_all(folder).map_err(|error| error.to_string())?;
        }
        let file = fs::File::create(log).map_err(|error| error.to_string())?;
        let errors = file.try_clone().map_err(|error| error.to_string())?;
        let started = Instant::now();
        let child = Command::new("node")
            .args([
                "node_modules/vite/bin/vite.js",
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--strictPort",
            ])
            .current_dir(root.join("web"))
            .stdin(Stdio::null())
            .stdout(file)
            .stderr(errors)
            .spawn()
            .map_err(|error| format!("cannot start Vite: {error}"))?;
        let mut server = Self {
            child,
            url: format!("http://127.0.0.1:{port}"),
        };
        loop {
            if TcpStream::connect(("127.0.0.1", port)).is_ok() && owns(server.child.id(), port)? {
                return Ok(server);
            }
            if let Some(status) = server.child.try_wait().map_err(|error| error.to_string())? {
                return Err(format!("Vite stopped ({status}); see {}", log.display()));
            }
            if started.elapsed() > limit {
                return Err(format!(
                    "Vite did not start within {} s; see {}",
                    limit.as_secs(),
                    log.display()
                ));
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }
}

fn owns(child: u32, port: u16) -> Result<bool, String> {
    let listening = listeners(port)?;
    if listening.contains(&child) {
        return Ok(true);
    }
    if listening.is_empty() {
        return Ok(false);
    }
    let parents = parents()?;
    Ok(listening
        .into_iter()
        .any(|pid| descends(pid, child, &parents)))
}

fn descends(mut pid: u32, ancestor: u32, parents: &BTreeMap<u32, u32>) -> bool {
    let mut seen = BTreeSet::new();
    while seen.insert(pid) {
        if pid == ancestor {
            return true;
        }
        match parents.get(&pid) {
            Some(&parent) => pid = parent,
            None => return false,
        }
    }
    false
}

fn listed(program: &str, arguments: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(arguments)
        .stdin(Stdio::null())
        .output()
        .map_err(|error| {
            format!("cannot run {program} to find who listens on the port: {error}")
        })?;
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(windows)]
fn listeners(port: u16) -> Result<BTreeSet<u32>, String> {
    let suffix = format!(":{port}");
    Ok(listed("netstat", &["-ano", "-p", "TCP"])?
        .lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let local = fields.get(1)?;
            let pid = fields.last()?.parse::<u32>().ok()?;
            (fields.first() == Some(&"TCP") && local.ends_with(&suffix) && pid != 0).then_some(pid)
        })
        .collect())
}

#[cfg(target_os = "linux")]
fn listeners(port: u16) -> Result<BTreeSet<u32>, String> {
    let suffix = format!(":{port}");
    Ok(listed("ss", &["-H", "-l", "-t", "-n", "-p"])?
        .lines()
        .filter(|line| {
            line.split_whitespace()
                .nth(3)
                .is_some_and(|local| local.ends_with(&suffix))
        })
        .flat_map(|line| line.split("pid=").skip(1))
        .filter_map(|rest| {
            rest.split(|character: char| !character.is_ascii_digit())
                .next()?
                .parse()
                .ok()
        })
        .collect())
}

#[cfg(all(unix, not(target_os = "linux")))]
fn listeners(port: u16) -> Result<BTreeSet<u32>, String> {
    let filter = format!("-iTCP:{port}");
    Ok(listed("lsof", &["-nP", &filter, "-sTCP:LISTEN", "-t"])?
        .lines()
        .filter_map(|line| line.trim().parse().ok())
        .collect())
}

#[cfg(windows)]
fn parents() -> Result<BTreeMap<u32, u32>, String> {
    let table = listed(
        "powershell",
        &[
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_Process | ForEach-Object { '{0} {1}' -f $_.ProcessId, $_.ParentProcessId }",
        ],
    )?;
    Ok(pairs(&table))
}

#[cfg(unix)]
fn parents() -> Result<BTreeMap<u32, u32>, String> {
    Ok(pairs(&listed("ps", &["-A", "-o", "pid=", "-o", "ppid="])?))
}

fn pairs(table: &str) -> BTreeMap<u32, u32> {
    table
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace().map(str::parse::<u32>);
            Some((fields.next()?.ok()?, fields.next()?.ok()?))
        })
        .collect()
}
