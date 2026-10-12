//! Development servers tools start for themselves: a commit's own
//! `cargo xtask dev` (checks), or this checkout's page served by Vite
//! (`cargo xtask smoke --serve`), each on a free loopback port, never the
//! ports people's own servers use, and stopped with every process it
//! started.

use std::fs;
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// The ports people's own development servers use (Vite's default and the
/// next two), which tools never take.
const PEOPLES_PORTS: std::ops::RangeInclusive<u16> = 5173..=5175;

/// A free port on this machine, which the operating system picks, never
/// one of [`PEOPLES_PORTS`].
pub fn free_port() -> Result<u16, String> {
    for _ in 0..16 {
        let port = TcpListener::bind("127.0.0.1:0")
            .and_then(|listener| listener.local_addr())
            .map(|address| address.port())
            .map_err(|error| format!("no free port: {error}"))?;
        if !PEOPLES_PORTS.contains(&port) {
            return Ok(port);
        }
    }
    Err("no free port outside 5173-5175".into())
}

/// A running development server, stopped with its processes when dropped.
pub struct Server {
    child: Child,
    pub url: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        let pid = self.child.id().to_string();
        // Cargo starts npm, which starts Vite, which starts esbuild: the
        // whole tree.
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

impl Server {
    /// Starts `command`, a server of `what` that listens on loopback `port`,
    /// its output into `log`, and waits until it accepts connections, at
    /// most `limit`.
    pub fn start(
        mut command: Command,
        port: u16,
        log: &Path,
        what: &str,
        limit: Duration,
    ) -> Result<Self, String> {
        if let Some(folder) = log.parent() {
            fs::create_dir_all(folder).map_err(|error| error.to_string())?;
        }
        let file = fs::File::create(log).map_err(|error| error.to_string())?;
        let errors = file.try_clone().map_err(|error| error.to_string())?;
        command.stdin(Stdio::null()).stdout(file).stderr(errors);
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut command, 0);
        let started = Instant::now();
        let child = command
            .spawn()
            .map_err(|error| format!("cannot start the development server of {what}: {error}"))?;
        let mut server = Self {
            child,
            url: format!("http://127.0.0.1:{port}"),
        };
        loop {
            if TcpStream::connect(("127.0.0.1", port)).is_ok() {
                return Ok(server);
            }
            if let Some(status) = server.child.try_wait().map_err(|error| error.to_string())? {
                return Err(format!(
                    "the development server of {what} stopped ({status}); see {}",
                    log.display()
                ));
            }
            if started.elapsed() > limit {
                return Err(format!(
                    "the development server of {what} did not start within {} s; see {}",
                    limit.as_secs(),
                    log.display()
                ));
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    /// This checkout's page with its development tools, as `cargo xtask
    /// dev` serves it, its output into `log`: its clients built first.
    pub fn page(root: &Path, log: &Path) -> Result<Self, String> {
        crate::pipeline::prepare_dev(root)?;
        let port = free_port()?;
        let mut command = Command::new("node");
        command
            .args([
                "node_modules/vite/bin/vite.js",
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--strictPort",
            ])
            .current_dir(root.join("web"));
        let server = Self::start(
            command,
            port,
            log,
            "this checkout",
            Duration::from_secs(120),
        )?;
        eprintln!("Serving this checkout's page at {}.", server.url);
        Ok(server)
    }
}
