mod client;
mod image;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use engine_command_line::{Command, CommandOption};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use client::{AgentClient, EntryMetrics, PassSample, SpawnConfig, WorkSample};
use image::VisualDiff;

const DEFAULT_FRAMES: u32 = 60;
const DEFAULT_WORK_FRAMES: u32 = 15;
const DEFAULT_MAX_CHANNEL_DIFF: u8 = 2;
const DEFAULT_MAX_MEAN_DIFF: f64 = 0.05;

pub fn command() -> Command {
    Command::new(
        "shaders eval",
        [
            CommandOption::taking("--suite", "FILE").required(),
            CommandOption::taking("--out", "DIR").required(),
            CommandOption::taking("--baseline", "DIR"),
            CommandOption::taking("--host", "native|browser").defaults_to("native"),
            CommandOption::taking("--build", "dev|profiling").defaults_to("profiling"),
            CommandOption::taking("--size", "WxH").in_unit("px"),
            CommandOption::taking("--frames", "N").defaults_to(DEFAULT_FRAMES),
            CommandOption::taking("--work-frames", "N").defaults_to(DEFAULT_WORK_FRAMES),
            CommandOption::taking("--max-channel-diff", "N").defaults_to(DEFAULT_MAX_CHANNEL_DIFF),
            CommandOption::taking("--max-mean-diff", "F").defaults_to("0.05"),
            CommandOption::taking("--view", "NAMES"),
            CommandOption::taking("--run", "NAME"),
            CommandOption::flag("--window"),
            CommandOption::flag("--rga"),
            CommandOption::flag("--no-work"),
            CommandOption::flag("--no-passes"),
        ],
    )
}

#[derive(Debug)]
struct Options {
    suite: PathBuf,
    out: PathBuf,
    baseline: Option<PathBuf>,
    host: String,
    build: String,
    size: Option<(u32, u32)>,
    frames: u32,
    work_frames: u32,
    max_channel_diff: u8,
    max_mean_diff: f64,
    views: Vec<String>,
    run: Option<String>,
    window: bool,
    rga: bool,
    measure_work: bool,
    measure_passes: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Suite {
    game: String,
    size: [u32; 2],
    capture_size: Option<[u32; 2]>,
    #[serde(default)]
    setup: Vec<Value>,
    views: Vec<SuiteView>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SuiteView {
    name: String,
    #[serde(default)]
    setup: Vec<Value>,
    max_channel_diff: Option<u8>,
    max_mean_diff: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EntryDelta {
    before_ops_per_run: f64,
    after_ops_per_run: f64,
    delta_ops_per_run: f64,
    pct_ops_per_run: f64,
    before_m_ops_per_frame: f64,
    after_m_ops_per_frame: f64,
    delta_m_ops_per_frame: f64,
    pct_m_ops_per_frame: f64,
    before_samples_per_run: f64,
    after_samples_per_run: f64,
    delta_samples_per_run: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PassDelta {
    before_ms: f64,
    after_ms: f64,
    delta_ms: f64,
    pct: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StaticDelta {
    module: String,
    entry_point: String,
    before_expressions: u64,
    after_expressions: u64,
    delta_expressions: i64,
    before_samples: u64,
    after_samples: u64,
    delta_samples: i64,
    before_rdna3_vgprs: Option<f64>,
    after_rdna3_vgprs: Option<f64>,
    before_rdna3_instructions: Option<u64>,
    after_rdna3_instructions: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ViewReport {
    name: String,
    picture: String,
    width: u32,
    height: u32,
    sha256: String,
    visual: Option<VisualDiff>,
    work_frames: u64,
    entries: BTreeMap<String, EntryMetrics>,
    entry_deltas: BTreeMap<String, EntryDelta>,
    pass_frames: u64,
    passes_ms: BTreeMap<String, f64>,
    pass_deltas_ms: BTreeMap<String, PassDelta>,
    passes_unavailable: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EvalReport {
    suite: String,
    game: String,
    host: String,
    build: String,
    size: [u32; 2],
    capture_size: [u32; 2],
    adapter: Value,
    baseline: Option<String>,
    passed: bool,
    views: Vec<ViewReport>,
    static_report: Option<String>,
    static_deltas: Vec<StaticDelta>,
}

pub fn run(root: &Path, arguments: &[String]) -> Result<(), String> {
    let options = parse_options(arguments)?;
    let suite_path = resolve(root, &options.suite);
    let suite_text = fs::read_to_string(&suite_path)
        .map_err(|error| format!("cannot read {}: {error}", suite_path.display()))?;
    let suite: Suite = serde_json::from_str(&suite_text)
        .map_err(|error| format!("invalid {}: {error}", suite_path.display()))?;
    if suite.views.is_empty() {
        return Err(format!("{} defines no views", suite_path.display()));
    }
    let selected = select_views(&suite.views, &options.views)?;
    let out = resolve(root, &options.out);
    fs::create_dir_all(&out).map_err(|error| format!("cannot create {}: {error}", out.display()))?;
    let baseline_dir = options.baseline.as_deref().map(|path| resolve(root, path));
    let baseline_report = baseline_dir.as_deref().map(load_baseline).transpose()?;
    let size = options.size.unwrap_or((suite.size[0], suite.size[1]));
    let capture_size = options
        .size
        .or_else(|| suite.capture_size.map(|pair| (pair[0], pair[1])))
        .unwrap_or(size);
    let mut client = open_client(root, &options, &suite, size)?;
    if options.run.is_some() {
        let reloaded = client.reload_shaders()?;
        println!("{}", json!({"event": "shaders_reloaded", "result": reloaded}));
    } else {
        for command in &suite.setup {
            let _ = client.send(command)?;
        }
    }
    let mut views = Vec::with_capacity(selected.len());
    let mut failed_views = Vec::new();
    for view in selected {
        let baseline_view = baseline_report
            .as_ref()
            .and_then(|report| report.views.iter().find(|entry| entry.name == view.name));
        let report = eval_view(
            &mut client,
            &options,
            &out,
            baseline_dir.as_deref(),
            baseline_view,
            view,
            capture_size,
        )?;
        if report.visual.as_ref().is_some_and(|visual| !visual.passed) {
            failed_views.push(report.name.clone());
        }
        print_view_events(&report);
        views.push(report);
    }
    let (static_report, static_deltas) = if options.rga {
        run_static(root, &out, baseline_dir.as_deref())?
    } else {
        (None, Vec::new())
    };
    let passed = failed_views.is_empty();
    let report = EvalReport {
        suite: options.suite.display().to_string(),
        game: suite.game,
        host: options.host,
        build: options.build,
        size: [size.0, size.1],
        capture_size: [capture_size.0, capture_size.1],
        adapter: client.adapter().clone(),
        baseline: options.baseline.map(|path| path.display().to_string()),
        passed,
        views,
        static_report,
        static_deltas,
    };
    let file = out.join("eval.json");
    let json_text = serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
    fs::write(&file, json_text)
        .map_err(|error| format!("cannot write {}: {error}", file.display()))?;
    println!(
        "{}",
        json!({
            "event": "eval_done",
            "passed": passed,
            "views": report.views.len(),
            "failed_views": failed_views,
            "file": file.display().to_string(),
        })
    );
    if passed {
        Ok(())
    } else {
        Err(format!(
            "visual regression gate failed for: {}",
            failed_views.join(", ")
        ))
    }
}

fn open_client(
    root: &Path,
    options: &Options,
    suite: &Suite,
    size: (u32, u32),
) -> Result<AgentClient, String> {
    if let Some(run_name) = options.run.as_deref() {
        return AgentClient::attach(root, run_name);
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    let run_name = format!("shader-eval-{}-{stamp}", std::process::id());
    AgentClient::spawn(SpawnConfig {
        root,
        host: &options.host,
        game: &suite.game,
        build: &options.build,
        size,
        window: options.window,
        run_name: &run_name,
    })
}

fn eval_view(
    client: &mut AgentClient,
    options: &Options,
    out: &Path,
    baseline_dir: Option<&Path>,
    baseline_view: Option<&ViewReport>,
    view: &SuiteView,
    capture_size: (u32, u32),
) -> Result<ViewReport, String> {
    let _ = client.set_instruments(&[])?;
    for command in &view.setup {
        let _ = client.send(command)?;
    }
    let picture_name = format!("{}.png", view.name);
    let picture_path = out.join(&picture_name);
    let (width, height, sha256) = client.capture_to(capture_size, &picture_path)?;
    let visual = match baseline_dir {
        Some(baseline_dir) => {
            let baseline_path = baseline_dir.join(&picture_name);
            let before = image::read_png(&baseline_path)?;
            let after = image::read_png(&picture_path)?;
            let channel_limit = view.max_channel_diff.unwrap_or(options.max_channel_diff);
            let mean_limit = view.max_mean_diff.unwrap_or(options.max_mean_diff);
            let (mut diff, diff_image) = image::compare(&before, &after, channel_limit, mean_limit)?;
            if let Some(diff_picture) = diff_image {
                let diff_name = format!("{}.diff.png", view.name);
                image::write_png(&out.join(&diff_name), &diff_picture)?;
                diff.diff_picture = Some(diff_name);
            }
            Some(diff)
        }
        None => None,
    };
    let work = if options.measure_work {
        client.measure_work(options.work_frames)?
    } else {
        WorkSample::default()
    };
    let passes = if options.measure_passes {
        client.measure_passes(options.frames)?
    } else {
        PassSample::default()
    };
    let entry_deltas = baseline_view
        .map(|before| compute_entry_deltas(&before.entries, &work.entries))
        .unwrap_or_default();
    let pass_deltas_ms = baseline_view
        .map(|before| compute_pass_deltas(&before.passes_ms, &passes.passes_ms))
        .unwrap_or_default();
    Ok(ViewReport {
        name: view.name.clone(),
        picture: picture_name,
        width,
        height,
        sha256,
        visual,
        work_frames: work.frames,
        entries: work.entries,
        entry_deltas,
        pass_frames: passes.frames,
        passes_ms: passes.passes_ms,
        pass_deltas_ms,
        passes_unavailable: passes.unavailable,
    })
}

fn parse_options(arguments: &[String]) -> Result<Options, String> {
    let mut suite = None;
    let mut out = None;
    let mut baseline = None;
    let mut host = "native".to_owned();
    let mut build = "profiling".to_owned();
    let mut size = None;
    let mut frames = DEFAULT_FRAMES;
    let mut work_frames = DEFAULT_WORK_FRAMES;
    let mut max_channel_diff = DEFAULT_MAX_CHANNEL_DIFF;
    let mut max_mean_diff = DEFAULT_MAX_MEAN_DIFF;
    let mut views = Vec::new();
    let mut run = None;
    let mut window = false;
    let mut rga = false;
    let mut measure_work = true;
    let mut measure_passes = true;
    let mut given = arguments.iter();
    while let Some(flag) = given.next() {
        match flag.as_str() {
            "--window" => window = true,
            "--rga" => rga = true,
            "--no-work" => measure_work = false,
            "--no-passes" => measure_passes = false,
            _ => {
                let value = given
                    .next()
                    .ok_or_else(|| format!("{flag} takes a value; run `cargo xtask help`"))?;
                match flag.as_str() {
                    "--suite" => suite = Some(PathBuf::from(value)),
                    "--out" => out = Some(PathBuf::from(value)),
                    "--baseline" => baseline = Some(PathBuf::from(value)),
                    "--host" if value == "native" || value == "browser" => host = value.clone(),
                    "--build" if value == "dev" || value == "profiling" => build = value.clone(),
                    "--size" => size = Some(parse_size(value)?),
                    "--frames" => frames = parse_u32("--frames", value, 1)?,
                    "--work-frames" => work_frames = parse_u32("--work-frames", value, 1)?,
                    "--max-channel-diff" => {
                        max_channel_diff = value
                            .parse::<u8>()
                            .map_err(|_| format!("--max-channel-diff takes 0..=255, got {value:?}"))?;
                    }
                    "--max-mean-diff" => {
                        max_mean_diff = value
                            .parse::<f64>()
                            .ok()
                            .filter(|number| number.is_finite() && *number >= 0.0)
                            .ok_or_else(|| {
                                format!("--max-mean-diff takes a non-negative number, got {value:?}")
                            })?;
                    }
                    "--view" => {
                        for name in value.split(',').map(str::trim).filter(|name| !name.is_empty()) {
                            views.push(name.to_owned());
                        }
                    }
                    "--run" => run = Some(value.clone()),
                    _ => return Err(format!("unknown option {flag:?} for `shaders eval`")),
                }
            }
        }
    }
    let suite = suite.ok_or("shaders eval requires --suite FILE")?;
    let out = out.ok_or("shaders eval requires --out DIR")?;
    Ok(Options {
        suite,
        out,
        baseline,
        host,
        build,
        size,
        frames,
        work_frames,
        max_channel_diff,
        max_mean_diff,
        views,
        run,
        window,
        rga,
        measure_work,
        measure_passes,
    })
}

fn parse_size(text: &str) -> Result<(u32, u32), String> {
    text.split_once('x')
        .and_then(|(width, height)| Some((width.parse::<u32>().ok()?, height.parse::<u32>().ok()?)))
        .filter(|&(width, height)| width > 0 && height > 0)
        .ok_or_else(|| format!("--size takes WxH, got {text:?}"))
}

fn parse_u32(flag: &str, text: &str, min: u32) -> Result<u32, String> {
    text.parse::<u32>()
        .ok()
        .filter(|&value| value >= min)
        .ok_or_else(|| format!("{flag} takes an integer >= {min}, got {text:?}"))
}

fn resolve(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn select_views<'a>(views: &'a [SuiteView], filter: &[String]) -> Result<Vec<&'a SuiteView>, String> {
    if filter.is_empty() {
        return Ok(views.iter().collect());
    }
    let mut selected = Vec::new();
    for wanted in filter {
        let found = views
            .iter()
            .find(|view| view.name == *wanted)
            .ok_or_else(|| {
                let known: Vec<&str> = views.iter().map(|view| view.name.as_str()).collect();
                format!(
                    "unknown --view {wanted:?}; suite defines: {}",
                    known.join(", ")
                )
            })?;
        if !selected.iter().any(|existing: &&SuiteView| existing.name == found.name) {
            selected.push(found);
        }
    }
    Ok(selected)
}

fn load_baseline(folder: &Path) -> Result<EvalReport, String> {
    let path = folder.join("eval.json");
    let text = fs::read_to_string(&path)
        .map_err(|error| format!("cannot read baseline {}: {error}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|error| format!("invalid baseline {}: {error}", path.display()))
}

fn pct_change(before: f64, after: f64) -> f64 {
    if before.abs() < 1.0e-9 {
        0.0
    } else {
        100.0 * (after - before) / before
    }
}

fn compute_entry_deltas(
    before: &BTreeMap<String, EntryMetrics>,
    after: &BTreeMap<String, EntryMetrics>,
) -> BTreeMap<String, EntryDelta> {
    let mut deltas = BTreeMap::new();
    for (name, after_metrics) in after {
        let Some(before_metrics) = before.get(name) else {
            continue;
        };
        deltas.insert(
            name.clone(),
            EntryDelta {
                before_ops_per_run: before_metrics.ops_per_run,
                after_ops_per_run: after_metrics.ops_per_run,
                delta_ops_per_run: after_metrics.ops_per_run - before_metrics.ops_per_run,
                pct_ops_per_run: pct_change(before_metrics.ops_per_run, after_metrics.ops_per_run),
                before_m_ops_per_frame: before_metrics.m_ops_per_frame,
                after_m_ops_per_frame: after_metrics.m_ops_per_frame,
                delta_m_ops_per_frame: after_metrics.m_ops_per_frame
                    - before_metrics.m_ops_per_frame,
                pct_m_ops_per_frame: pct_change(
                    before_metrics.m_ops_per_frame,
                    after_metrics.m_ops_per_frame,
                ),
                before_samples_per_run: before_metrics.samples_per_run,
                after_samples_per_run: after_metrics.samples_per_run,
                delta_samples_per_run: after_metrics.samples_per_run
                    - before_metrics.samples_per_run,
            },
        );
    }
    deltas
}

fn compute_pass_deltas(
    before: &BTreeMap<String, f64>,
    after: &BTreeMap<String, f64>,
) -> BTreeMap<String, PassDelta> {
    let mut deltas = BTreeMap::new();
    for (span, &after_ms) in after {
        let Some(&before_ms) = before.get(span) else {
            continue;
        };
        deltas.insert(
            span.clone(),
            PassDelta {
                before_ms,
                after_ms,
                delta_ms: after_ms - before_ms,
                pct: pct_change(before_ms, after_ms),
            },
        );
    }
    deltas
}

fn run_static(
    root: &Path,
    out: &Path,
    baseline_dir: Option<&Path>,
) -> Result<(Option<String>, Vec<StaticDelta>), String> {
    let folder = out.join("static");
    let (after_report, path) = crate::shader_report::generate(root, &folder, true)?;
    let deltas = match baseline_dir
        .map(|dir| dir.join("static/report.json"))
        .filter(|path| path.is_file())
    {
        Some(before_path) => {
            let text = fs::read_to_string(&before_path)
                .map_err(|error| format!("cannot read {}: {error}", before_path.display()))?;
            let before_report: Value = serde_json::from_str(&text)
                .map_err(|error| format!("invalid {}: {error}", before_path.display()))?;
            compute_static_deltas(&before_report, &after_report)
        }
        None => Vec::new(),
    };
    Ok((Some(path.display().to_string()), deltas))
}

fn index_static_entries(report: &Value) -> BTreeMap<(String, String), &Value> {
    let mut map = BTreeMap::new();
    for entry in report["entries"].as_array().into_iter().flatten() {
        if let (Some(module), Some(name)) = (
            entry["module"].as_str(),
            entry["entry_point"].as_str(),
        ) {
            map.insert((module.to_owned(), name.to_owned()), entry);
        }
    }
    map
}

fn compute_static_deltas(before: &Value, after: &Value) -> Vec<StaticDelta> {
    let before_map = index_static_entries(before);
    let after_map = index_static_entries(after);
    let mut deltas = Vec::new();
    for ((module, entry_point), after_entry) in after_map {
        let Some(before_entry) = before_map.get(&(module.clone(), entry_point.clone())) else {
            continue;
        };
        let before_expr = before_entry["naga"]["expressions"].as_u64().unwrap_or(0);
        let after_expr = after_entry["naga"]["expressions"].as_u64().unwrap_or(0);
        let before_samples = before_entry["naga"]["texture_samples"].as_u64().unwrap_or(0);
        let after_samples = after_entry["naga"]["texture_samples"].as_u64().unwrap_or(0);
        let before_vgprs = before_entry["amd"]["gfx1100"]["resources"]["used_vgprs"].as_f64();
        let after_vgprs = after_entry["amd"]["gfx1100"]["resources"]["used_vgprs"].as_f64();
        let before_inst = before_entry["amd"]["gfx1100"]["instructions"].as_u64();
        let after_inst = after_entry["amd"]["gfx1100"]["instructions"].as_u64();
        let changed = before_expr != after_expr
            || before_samples != after_samples
            || before_vgprs != after_vgprs
            || before_inst != after_inst;
        if changed {
            deltas.push(StaticDelta {
                module,
                entry_point,
                before_expressions: before_expr,
                after_expressions: after_expr,
                delta_expressions: after_expr as i64 - before_expr as i64,
                before_samples,
                after_samples,
                delta_samples: after_samples as i64 - before_samples as i64,
                before_rdna3_vgprs: before_vgprs,
                after_rdna3_vgprs: after_vgprs,
                before_rdna3_instructions: before_inst,
                after_rdna3_instructions: after_inst,
            });
        }
    }
    deltas
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn total_gpu_ms(passes_ms: &BTreeMap<String, f64>) -> Option<f64> {
    if let Some(&busy) = passes_ms.get("busy") {
        return Some(busy);
    }
    let mut sum = 0.0;
    let mut found = false;
    for (name, &ms) in passes_ms {
        if name.starts_with("batch_") {
            sum += ms;
            found = true;
        }
    }
    found.then_some(sum)
}

fn print_view_events(report: &ViewReport) {
    let total_m_ops: f64 = report
        .entries
        .values()
        .map(|entry| entry.m_ops_per_frame)
        .sum();
    println!(
        "{}",
        json!({
            "event": "eval_view",
            "view": report.name,
            "visual_passed": report.visual.as_ref().map(|visual| visual.passed),
            "max_channel_diff": report.visual.as_ref().map(|visual| visual.max_channel_diff),
            "mean_abs_diff": report.visual.as_ref().map(|visual| round2(visual.mean_abs_diff)),
            "total_m_ops_per_frame": round2(total_m_ops),
            "gpu_ms": total_gpu_ms(&report.passes_ms).map(round2),
        })
    );
    for (entry, delta) in &report.entry_deltas {
        if delta.delta_ops_per_run.abs() < 0.05 && delta.delta_samples_per_run.abs() < 0.005 {
            continue;
        }
        println!(
            "{}",
            json!({
                "event": "entry_delta",
                "view": report.name,
                "entry": entry,
                "ops_per_run": [
                    round2(delta.before_ops_per_run),
                    round2(delta.after_ops_per_run),
                    round2(delta.pct_ops_per_run),
                ],
                "m_ops_per_frame": [
                    round2(delta.before_m_ops_per_frame),
                    round2(delta.after_m_ops_per_frame),
                    round2(delta.pct_m_ops_per_frame),
                ],
                "samples_per_run": [
                    round2(delta.before_samples_per_run),
                    round2(delta.after_samples_per_run),
                ],
            })
        );
    }
    for (span, delta) in &report.pass_deltas_ms {
        println!(
            "{}",
            json!({
                "event": "pass_delta",
                "view": report.name,
                "pass": span,
                "ms": [round2(delta.before_ms), round2(delta.after_ms), round2(delta.pct)],
            })
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_eval_options_with_defaults() -> Result<(), String> {
        let args = vec![
            "--suite".into(),
            "suite.json".into(),
            "--out".into(),
            "out".into(),
            "--view".into(),
            "a,b".into(),
            "--rga".into(),
            "--window".into(),
        ];
        let parsed = parse_options(&args)?;
        assert_eq!(parsed.host, "native", "default host should be native");
        assert_eq!(parsed.build, "profiling", "default build should be profiling");
        assert_eq!(parsed.frames, DEFAULT_FRAMES, "default pass frames should apply");
        assert_eq!(parsed.views, vec!["a", "b"], "comma-separated views should split");
        assert!(parsed.rga, "--rga flag should set rga");
        assert!(parsed.window, "--window flag should set window");
        Ok(())
    }

    #[test]
    fn computes_entry_and_pass_deltas() -> Result<(), String> {
        let mut before = BTreeMap::new();
        before.insert(
            "fs_level".into(),
            EntryMetrics {
                runs_per_frame: 1_000_000.0,
                ops_per_run: 100.0,
                samples_per_run: 2.0,
                m_ops_per_frame: 100.0,
                m_samples_per_frame: 2.0,
            },
        );
        let mut after = BTreeMap::new();
        after.insert(
            "fs_level".into(),
            EntryMetrics {
                runs_per_frame: 1_000_000.0,
                ops_per_run: 75.0,
                samples_per_run: 1.5,
                m_ops_per_frame: 75.0,
                m_samples_per_frame: 1.5,
            },
        );
        let deltas = compute_entry_deltas(&before, &after);
        let level = deltas
            .get("fs_level")
            .ok_or_else(|| "missing fs_level delta".to_owned())?;
        assert!(
            (level.pct_ops_per_run + 25.0).abs() < 1.0e-6,
            "25% op reduction should compute -25.0%"
        );
        assert!(
            (level.delta_samples_per_run + 0.5).abs() < 1.0e-6,
            "sample delta should be -0.5"
        );
        let mut p_before = BTreeMap::new();
        p_before.insert("opaque_scene".into(), 4.0);
        let mut p_after = BTreeMap::new();
        p_after.insert("opaque_scene".into(), 3.0);
        let p_deltas = compute_pass_deltas(&p_before, &p_after);
        let opaque = p_deltas
            .get("opaque_scene")
            .ok_or_else(|| "missing opaque_scene pass delta".to_owned())?;
        assert!(
            (opaque.pct + 25.0).abs() < 1.0e-6,
            "pass time reduction should compute -25.0%"
        );
        Ok(())
    }
}
