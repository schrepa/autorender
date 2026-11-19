use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
#[cfg(feature = "profiling")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, TryRecvError, channel};
use std::time::Duration;

use engine::host::{AdapterStatus, Shot};
use engine::scene::{SceneFiles, SceneKinds};
use engine::{Error, ErrorCode, Expected, Game, GameHost};
use engine_registry::ReachMut;
use serde_json::json;

use crate::{file_error, files};

const RUNS: &str = "artifacts/runs";

const DEEP: &str = "deep";

#[cfg(feature = "profiling")]
static DEEP_WANTED: AtomicBool = AtomicBool::new(false);
static INSTRUMENTS: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    engine::host::instruments()
        .iter()
        .copied()
        .chain([DEEP])
        .collect()
});

pub struct Session {
    lines: Option<Receiver<String>>,
    run: Option<PathBuf>,
    #[cfg(feature = "profiling")]
    deep: Option<(engine_profiler::DeepCapture, PathBuf, String)>,
}

pub struct Start<'a> {
    pub requests: bool,
    pub size: (u32, u32),
    pub run: Option<&'a str>,
}

impl Session {
    pub fn open<G: Game>(host: &mut GameHost<G>, start: Start<'_>) -> engine::Result<Self> {
        #[cfg(feature = "profiling")]
        engine_profiler::deep_setting(host.engine_mut().app.world_mut());
        host.engine_mut()
            .app
            .world_mut()
            .insert_resource(engine::dev::Instruments {
                names: &INSTRUMENTS,
                switch: instrument,
            });
        let run = match start.run {
            Some(name) => {
                let folder = run_folder("--run", name)?;
                std::fs::create_dir_all(&folder)
                    .map_err(|error| file_error("--run", &folder, &error))?;
                Some(folder)
            }
            None => None,
        };
        let lines = start.requests.then(|| {
            let ready = json!({
                "ready": true,
                "size": [start.size.0, start.size.1],
                "adapter": AdapterStatus::of(host.adapter()),
                "run": start.run,
            });
            say(&ready.to_string());
            read_lines()
        });
        Ok(Self {
            lines,
            run,
            #[cfg(feature = "profiling")]
            deep: None,
        })
    }

    pub fn receive<G: Game>(&mut self, host: &mut GameHost<G>, wait: Duration) -> bool {
        let Some(lines) = &self.lines else {
            return true;
        };
        let first = match lines.recv_timeout(wait) {
            Ok(line) => line,
            Err(RecvTimeoutError::Timeout) => return true,
            Err(RecvTimeoutError::Disconnected) => return false,
        };
        host.dev_push(&first);
        loop {
            match lines.try_recv() {
                Ok(line) => {
                    host.dev_push(&line);
                }
                Err(TryRecvError::Empty) => return true,
                Err(TryRecvError::Disconnected) => return false,
            }
        }
    }

    pub fn after<G: Game>(&mut self, host: &mut GameHost<G>) -> engine::Result<()> {
        let take = host.dev_take();
        for (_, line) in take.replies {
            say(&line);
        }
        for shot in take.pictures {
            if let Err(error) = self.picture(&shot) {
                let record =
                    json!({"event": "picture_not_written", "file": shot.file, "error": error});
                eprintln!("{record}");
            }
        }
        if take.reload {
            match read_source(host) {
                Ok((source, files)) => host.dev_reloaded(&source, &files),
                Err(error) => host.dev_reload_failed(error),
            }
        }
        #[cfg(feature = "profiling")]
        if let Err(error) = self.follow_deep(host) {
            let record = json!({"event": "deep_capture_failed", "error": error});
            eprintln!("{record}");
        }
        Ok(())
    }

    #[cfg(feature = "profiling")]
    fn follow_deep<G: Game>(&mut self, host: &GameHost<G>) -> engine::Result<()> {
        match (
            self.deep.take(),
            DEEP_WANTED.load(Ordering::Relaxed),
            self.run.clone(),
        ) {
            (None, true, Some(folder)) => {
                let sizes = match engine_profiler::deep_sizes(host.engine().app.world()) {
                    Ok(sizes) => sizes,
                    Err(reason) => {
                        DEEP_WANTED.store(false, Ordering::Relaxed);
                        return Err(
                            Error::new(ErrorCode::ProfilingUnavailable, "$.deep").got(reason)
                        );
                    }
                };
                let threads = engine_profiler::sampler::SampledThread::current("Main thread")
                    .into_iter()
                    .chain(engine_profiler::sampler::announced_threads())
                    .collect();
                let stem = format!("deep-{}", host.frame_times().frame);
                let capture = engine_profiler::DeepCapture::start(threads, sizes);
                self.deep = Some((capture, folder, stem));
            }
            (Some(running), true, Some(_)) => self.deep = Some(running),
            (Some((capture, folder, stem)), false, _)
            | (Some((capture, folder, stem)), true, None) => {
                capture.finish(&folder, &stem)?;
            }
            (None, _, _) => {}
        }
        Ok(())
    }

    fn picture(&self, shot: &Shot) -> engine::Result<()> {
        let folder = self.run.as_ref().ok_or_else(|| {
            Error::new(ErrorCode::MissingField, "--run").expected(Expected::Present)
        })?;
        let path = folder.join(&shot.file);
        let failed = |_| {
            Error::new(ErrorCode::SourceIo, "$")
                .expected(Expected::Writable)
                .got(path.display().to_string())
        };
        let file = std::fs::File::create(&path).map_err(|error| file_error("$", &path, &error))?;
        let mut encoder = png::Encoder::new(BufWriter::new(file), shot.width, shot.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(failed)?;
        writer.write_image_data(&shot.rgba).map_err(failed)?;
        writer.finish().map_err(failed)
    }
}

pub fn follow_source<G: Game>(host: &mut GameHost<G>) {
    if let Some(Err(error)) = host.follow_source() {
        let record = json!({"event": "source_changed", "error": error});
        eprintln!("{record}");
    }
}

fn read_source<G: Game>(host: &mut GameHost<G>) -> engine::Result<(String, SceneFiles)> {
    let Some(path) = host.source().map(Path::to_owned) else {
        return Err(Error::new(ErrorCode::UnsupportedCommand, "$.name").got("reload"));
    };
    host.watch_source(&path);
    let source =
        std::fs::read_to_string(&path).map_err(|error| file_error("$.source", &path, &error))?;
    let kinds = SceneKinds::of(host.engine().app.world());
    let files = files::scene_files(&path, &source, &kinds)?;
    Ok((source, files))
}

fn run_folder(path: &str, name: &str) -> engine::Result<PathBuf> {
    let plain = !name.is_empty()
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
    if !plain {
        return Err(Error::new(ErrorCode::InvalidName, path).got(name));
    }
    Ok(Path::new(RUNS).join(name))
}

fn instrument(
    reach: &mut ReachMut<'_>,
    name: &str,
    on: bool,
) -> Result<&'static str, &'static str> {
    if name == DEEP {
        return deep(reach, on);
    }
    engine::host::instrument(reach, name, on)
}

#[cfg(feature = "profiling")]
fn deep(reach: &mut ReachMut<'_>, on: bool) -> Result<&'static str, &'static str> {
    if on {
        engine_profiler::deep_sizes(reach.world)?;
    }
    DEEP_WANTED.store(on, Ordering::Relaxed);
    Ok("samples every thread's stack and the heap's allocation stacks, which slows the frame")
}

#[cfg(not(feature = "profiling"))]
fn deep(_: &mut ReachMut<'_>, _: bool) -> Result<&'static str, &'static str> {
    Err("deep_capture_needs_native_profiling_build")
}

fn say(line: &str) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{line}").and_then(|()| out.flush());
}

fn read_lines() -> Receiver<String> {
    let (send, lines) = channel();
    std::thread::Builder::new()
        .name("engine-dev-input".into())
        .spawn(move || {
            engine_profile::announce_thread(format_args!("Dev input"));
            for line in std::io::stdin().lock().lines() {
                let Ok(line) = line else { break };
                if !line.trim().is_empty() && send.send(line).is_err() {
                    break;
                }
            }
        })
        .expect("thread_starts");
    lines
}
