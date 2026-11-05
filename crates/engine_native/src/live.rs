use std::path::PathBuf;
use std::time::Duration;

use engine::host::{FramePhase, median, now_ms};
use engine::{Game, GameHost};
use engine_profiler::DeepCapture;
use engine_profiler::sampler::SampledThread;
use serde_json::{Value, json};
use winit::event::WindowEvent;

const HITCH_OVER_MEDIAN: f64 = 1.5;
const INTERVAL: usize = 1;
const COLUMNS: [&str; 19] = [
    "at_ms",
    "interval_ms",
    "busy_ms",
    "simulation_ms",
    "extraction_ms",
    "render_ms",
    "gpu_ms",
    "fixed_steps",
    "dropped_seconds",
    "cut_seconds",
    "upload_bytes",
    "draw_calls",
    "pipelines_created",
    "render_width",
    "scale",
    "compiling",
    "motion",
    "mark",
    "heap_live_bytes",
];

pub(crate) struct LiveProfile {
    capture: DeepCapture,
    folder: PathBuf,
    started: f64,
    frames: Vec<[f64; COLUMNS.len()]>,
    events: Vec<Value>,
    before: Option<f64>,
    phase: Option<FramePhase>,
    motion: f64,
}

impl LiveProfile {
    pub(crate) fn start<G: Game>(host: &mut GameHost<G>, folder: PathBuf) -> engine::Result<Self> {
        let threads = SampledThread::current("Main thread").into_iter().collect();
        let capture = DeepCapture::start(host, threads, engine::profiling::MAX_EVENTS)?;
        Ok(Self {
            capture,
            folder,
            started: now_ms(),
            frames: Vec::new(),
            events: Vec::new(),
            before: None,
            phase: None,
            motion: 0.0,
        })
    }

    fn event(&mut self, kind: &str, detail: &str) {
        self.events
            .push(json!([rounded(now_ms() - self.started), kind, detail]));
    }

    #[expect(clippy::wildcard_enum_match_arm, reason = "winit's other events")]
    pub(crate) fn noted(&mut self, event: &WindowEvent) {
        let (kind, detail) = match event {
            WindowEvent::Focused(focused) => ("focus", focused.to_string()),
            WindowEvent::Resized(size) => ("resize", format!("{}x{}", size.width, size.height)),
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                ("scale", scale_factor.to_string())
            }
            WindowEvent::KeyboardInput { event, .. } if !event.repeat => {
                ("key", format!("{:?} {:?}", event.physical_key, event.state))
            }
            WindowEvent::MouseInput { state, button, .. } => {
                ("button", format!("{button:?} {state:?}"))
            }
            WindowEvent::MouseWheel { delta, .. } => ("wheel", format!("{delta:?}")),
            WindowEvent::CursorLeft { .. } => ("cursor_left", String::new()),
            WindowEvent::CloseRequested => ("close", String::new()),
            _ => return,
        };
        self.event(kind, &detail);
    }

    pub(crate) fn motion(&mut self, (dx, dy): (f64, f64)) {
        self.motion += dx.abs() + dy.abs();
    }

    pub(crate) fn frame<G: Game>(&mut self, phase: FramePhase, busy: Duration, host: &GameHost<G>) {
        if self.phase != Some(phase) {
            self.phase = Some(phase);
            self.event("phase", &format!("{phase:?}"));
        }
        if phase != FramePhase::Playing {
            self.before = None;
            return;
        }
        let now = now_ms();
        let interval = self.before.map_or(0.0, |before| now - before);
        self.before = Some(now);
        let stats = host.frame_times();
        let render = &stats.render;
        self.frames.push([
            now - self.started,
            interval,
            busy.as_secs_f64() * 1000.0,
            stats.simulation_ms,
            stats.extraction_ms,
            stats.render_ms,
            f64::from(stats.resolution.gpu_ms.unwrap_or(-1.0)),
            stats.fixed_steps as f64,
            stats.dropped_seconds,
            stats.cut_seconds,
            (render.upload_bytes + render.mesh_upload_bytes + render.texture_upload_bytes) as f64,
            render.draw_calls as f64,
            render.pipelines_created as f64,
            stats.resolution.render_width as f64,
            stats.dynamic_resolution.scale as f64,
            f64::from(u8::from(stats.pipelines_compiling)),
            std::mem::take(&mut self.motion),
            engine::profiling::frame_mark() as f64,
            engine::profiling::allocation_snapshot().live_bytes as f64,
        ]);
    }

    pub(crate) fn finish<G: Game>(self, host: &mut GameHost<G>) -> engine::Result<()> {
        let intervals: Vec<f64> = self.frames.iter().map(|frame| frame[INTERVAL]).collect();
        let limit = median(&intervals).unwrap_or(0.0) * HITCH_OVER_MEDIAN;
        let hitches: Vec<Value> = self
            .frames
            .iter()
            .filter(|frame| frame[INTERVAL] > limit)
            .map(row)
            .collect();
        let frames: Vec<Value> = self.frames.iter().map(row).collect();
        let play = json!({
            "columns": COLUMNS,
            "median_interval_ms": median(&intervals),
            "hitch_over_ms": limit,
            "hitches": hitches,
            "events": self.events,
            "frames": frames,
        });
        let context = json!({
            "program": crate::options::program(),
            "arguments": std::env::args().skip(1).collect::<Vec<_>>(),
            "window": true,
            "frames": self.frames.len(),
        });
        let files = self
            .capture
            .finish(host, &self.folder, "profile", context, play)?;
        let report = json!({
            "summary": files.summary.display().to_string(),
            "firefox": files.firefox.display().to_string(),
            "report": files.report.display().to_string(),
            "complete": files.complete,
        });
        println!("{:#}", json!({ "profile": report }));
        Ok(())
    }
}

fn row(frame: &[f64; COLUMNS.len()]) -> Value {
    Value::Array(frame.iter().map(|&value| json!(rounded(value))).collect())
}

fn rounded(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}
