//! Owned subprocesses with durable raw output and a finite outer deadline.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};

/// Raw files are opened before launch and survive failure or cancellation.
pub struct Output {
    pub status: ExitStatus,
    pub stdout: PathBuf,
    pub stderr: PathBuf,
    pub elapsed_ms: u128,
}

/// A cancellation marker is caller owned. Its existence cancels the job.
pub struct Limits {
    pub timeout: Duration,
    pub cancel: Option<PathBuf>,
    /// Stop a runaway logger; polling permits a small in-progress overshoot.
    pub max_log_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30 * 60),
            cancel: std::env::var_os("ENGINE_CANCEL_FILE").map(PathBuf::from),
            max_log_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Configure an independently killable tree and suppress helper windows.
pub fn owned(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
}

/// Terminate only the tree started for this operation.
pub fn stop(child: &mut Child) -> Result<(), String> {
    #[cfg(windows)]
    let mut killer = {
        let mut c = Command::new("taskkill");
        c.args(["/PID", &child.id().to_string(), "/T", "/F"]);
        c
    };
    #[cfg(unix)]
    let mut killer = {
        let mut c = Command::new("kill");
        c.args(["-KILL", "--", &format!("-{}", child.id())]);
        c
    };
    owned(&mut killer);
    let result = killer.stdout(Stdio::null()).stderr(Stdio::null()).status();
    if !result.is_ok_and(|status| status.success()) {
        child.kill().map_err(|e| e.to_string())?;
    }
    child.wait().map_err(|e| e.to_string())?;
    Ok(())
}

/// Capture a fresh attempt. Never reuse an earlier stdout/stderr file.
pub fn run(command: &mut Command, folder: &Path, limits: &Limits) -> Result<Output, String> {
    fs::create_dir_all(folder).map_err(|e| e.to_string())?;
    let stdout = folder.join("stdout.log");
    let stderr = folder.join("stderr.log");
    let open = |path: &Path| {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| format!("{}: {e}", path.display()))
    };
    command
        .stdout(Stdio::from(open(&stdout)?))
        .stderr(Stdio::from(open(&stderr)?));
    owned(command);
    let started = Instant::now();
    let spawned = command.spawn();
    // Command itself owns copies of the redirected handles. Close them even
    // on launch failure, so Windows can publish or collect the attempt folder.
    command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    let mut child = spawned.map_err(|e| format!("cannot start {command:?}: {e}"))?;
    loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            return Ok(Output {
                status,
                stdout,
                stderr,
                elapsed_ms: started.elapsed().as_millis(),
            });
        }
        let cancelled = limits.cancel.as_ref().is_some_and(|path| path.exists());
        let oversized = [&stdout, &stderr]
            .iter()
            .any(|path| fs::metadata(path).is_ok_and(|m| m.len() > limits.max_log_bytes));
        if cancelled || oversized || started.elapsed() >= limits.timeout {
            stop(&mut child)?;
            return Err(format!(
                "{} after {} ms; raw output in {}",
                if cancelled {
                    "cancelled"
                } else if oversized {
                    "log byte limit exceeded"
                } else {
                    "deadline exceeded"
                },
                started.elapsed().as_millis(),
                folder.display()
            ));
        }
        thread::sleep(Duration::from_millis(50));
    }
}
