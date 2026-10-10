//! What a machine can tell a profile beyond what every machine gives
//! without setup: its operating system, CPU and GPU, and the deeper layers
//! (system-wide CPU sampling, exact instruction counts, GPU per-shader-line
//! costs and hardware counters) with whether their tools are installed and
//! how to set them up. None of these run by default; profiles recommend
//! them, the GPU vendor's tools and hardware counters highly.

use serde_json::{Value, json};

/// The machine: operating system, architecture, CPU, logical processors
/// and the GPU the game drew on.
pub(crate) fn machine(adapter: &engine::AdapterInfo) -> Value {
    json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "cpu": cpu_name(),
        "logical_processors": std::thread::available_parallelism().map_or(0, usize::from),
        "gpu": {
            "name": adapter.name,
            "vendor": vendor(adapter),
            "type": adapter.device_type,
            "backend": adapter.backend,
            "driver": adapter.driver,
        },
        "linux_perf_event_paranoid": perf_event_paranoid(),
    })
}

/// The GPU's vendor, by its PCI vendor ID or name.
fn vendor(adapter: &engine::AdapterInfo) -> &'static str {
    match adapter.vendor {
        0x10de => "nvidia",
        0x1002 | 0x1022 => "amd",
        0x8086 => "intel",
        0x106b => "apple",
        0x5143 => "qualcomm",
        0x13b5 => "arm",
        _ if adapter.name.starts_with("Apple") => "apple",
        _ => "other",
    }
}

fn cpu_name() -> Option<String> {
    if cfg!(windows) {
        return std::env::var("PROCESSOR_IDENTIFIER").ok();
    }
    if cfg!(target_os = "linux") {
        let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        return info
            .lines()
            .find(|line| line.starts_with("model name"))
            .and_then(|line| line.split_once(':'))
            .map(|(_, name)| name.trim().to_owned());
    }
    if cfg!(target_vendor = "apple") {
        let output = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()?;
        return Some(String::from_utf8_lossy(&output.stdout).trim().to_owned());
    }
    None
}

fn perf_event_paranoid() -> Option<i32> {
    std::fs::read_to_string("/proc/sys/kernel/perf_event_paranoid")
        .ok()?
        .trim()
        .parse()
        .ok()
}

/// A program found on the path, or at one of `paths` (Windows install
/// folders, where `*` matches a version folder).
pub(crate) fn installed(program: &str, paths: &[&str]) -> Option<String> {
    let executable = if cfg!(windows) {
        format!("{program}.exe")
    } else {
        program.to_owned()
    };
    let on_path = std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|folder| folder.join(&executable))
            .find(|file| file.is_file())
    });
    if let Some(file) = on_path {
        return Some(file.display().to_string());
    }
    paths.iter().find_map(|pattern| {
        let Some((head, rest)) = pattern.split_once('*') else {
            let file = std::path::Path::new(pattern);
            return file.is_file().then(|| file.display().to_string());
        };
        // The folder holding the version folders, and their names' prefix.
        let split = head.rfind(['/', '\\'])?;
        let (folder, prefix) = (&head[..split], &head[split + 1..]);
        let mut versions: Vec<_> = std::fs::read_dir(folder)
            .ok()?
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().starts_with(prefix))
            .map(|entry| entry.path().join(rest.trim_start_matches(['/', '\\'])))
            .filter(|file| file.is_file())
            .collect();
        versions.sort();
        versions.pop().map(|file| file.display().to_string())
    })
}

/// One recommended layer: what it adds, the tool, whether it is installed
/// here, and how to set it up.
fn layer(what: &str, tool: &str, found: Option<String>, setup: &str, adds: &str) -> Value {
    json!({
        "layer": what,
        "priority": "recommended",
        "tool": tool,
        "installed": found,
        "setup": setup,
        "adds": adds,
    })
}

/// `layer`, highly recommended: what this GPU's own counters and tools
/// tell.
fn highly(mut layer: Value) -> Value {
    layer["priority"] = "highly recommended".into();
    layer
}

/// The deeper layers this machine could add, beyond the default profile's,
/// each with its tool and setup; the ones its GPU and operating system
/// support.
pub(crate) fn recommended(adapter: &engine::AdapterInfo, shallow_stacks: bool) -> Vec<Value> {
    let mut layers = Vec::new();
    let os = std::env::consts::OS;
    if os == "linux" && shallow_stacks {
        layers.push(layer(
            "complete CPU stacks",
            "frame pointers",
            None,
            "build with RUSTFLAGS=-Cforce-frame-pointers=yes (the agent's native profiling builds do)",
            "the in-process sampler follows frame pointers on Linux; without them stacks stop early",
        ));
    }
    match os {
        "windows" => layers.push(layer(
            "system-wide CPU sampling with kernel stacks",
            "samply (or Windows Performance Recorder)",
            installed("samply", &[]).or_else(|| installed("wpr", &[])),
            "install samply (cargo install samply) and run `samply record PROGRAM ...` (any native game's program); Windows asks for admin once per recording (ETW)",
            "kernel and driver stacks, other processes, context switches",
        )),
        "linux" => layers.push(layer(
            "system-wide CPU sampling and hardware counters",
            "samply or perf",
            installed("samply", &[]).or_else(|| installed("perf", &[])),
            "allow it once: `sudo sysctl kernel.perf_event_paranoid=1` (Ubuntu and Debian default to 4 and 3), or grant CAP_PERFMON; then `samply record` or `perf record -g`",
            "kernel stacks, cache misses, branch mispredictions per line",
        )),
        "macos" => layers.push(layer(
            "system-wide CPU sampling and counters",
            "Instruments (xctrace) or samply",
            installed("xctrace", &[]).or_else(|| installed("samply", &[])),
            "install Xcode, or samply (`samply record` launches programs without setup)",
            "kernel stacks, CPU counters, Processor Trace on M4 and later",
        )),
        _ => {}
    }
    layers.push(layer(
        "exact instruction counts",
        if os == "linux" {
            "Valgrind (callgrind)"
        } else {
            "DynamoRIO (drcachesim)"
        },
        installed("valgrind", &[]).or_else(|| installed("drrun", &[])),
        "run a short offscreen capture under the tool; 20 to 50 times slower, so only a few frames",
        "every instruction executed and every cache miss, deterministic",
    ));
    // Vulkan's performance queries: Intel's drivers, and Mesa's on Linux.
    let counted = match vendor(adapter) {
        "intel" => os == "windows" || os == "linux",
        "amd" | "qualcomm" => os == "linux",
        _ => false,
    };
    if counted {
        layers.push(highly(layer(
            "GPU hardware counters per pass, in process",
            "the GPU driver's Vulkan performance queries",
            Some("with the driver".into()),
            "none: profile with `--counters on` (or a comma-separated list of the driver's counter names); the game then draws on Vulkan, and counting perturbs timings, so time without it",
            "each pass's shader-core activity, stalls and occupancy, sampler and memory traffic, depth-test rejects and pixels",
        )));
    }
    if vendor(adapter) == "nvidia" {
        layers.push(highly(layer(
            "GPU cost per shader line, unit throughputs per pass",
            "Nsight Graphics (GPU Trace with its real-time shader profiler; GPUs from 2020 on)",
            installed("ngfx", &[r"C:\Program Files\NVIDIA Corporation\Nsight Graphics */host/windows-desktop-nomad-x64/ngfx.exe"]),
            "install Nsight Graphics and have an administrator allow GPU performance counters for all users once (NVIDIA Control Panel: Desktop > Enable Developer Settings, then Developer > Manage GPU Performance Counters; Linux: the nvidia module's NVreg_RestrictProfilingToAdminUsers=0); then run in a window with `--vendor nsight --vendor-dir DIR`",
            "per-shader-line warp stalls, SM, cache and memory throughputs per pass",
        )));
    }
    if os == "windows" {
        layers.push(highly(layer(
            "GPU hardware counters per draw, replayed from a frame-exact capture",
            "PIX (its counter plugins: Intel's, AMD's, NVIDIA's and Qualcomm's GPUs)",
            installed("pixtool", &[r"C:\Program Files\Microsoft PIX\*/pixtool.exe"]),
            "install PIX and turn on Windows Developer Mode, which replays need; then `--offscreen N --vendor pix --vendor-dir DIR` on D3D12 (dxcompiler.dll beside the program); NVIDIA's counters also need the performance-counter permission Nsight needs",
            "each draw's and render-graph node's shader-core activity, stalls, occupancy, sampler and memory counters, and the capture, replayable in PIX",
        )));
    }
    match vendor(adapter) {
        "amd" => layers.push(highly(layer(
            "GPU cost per shader instruction",
            "Radeon GPU Profiler (instruction timing; RDNA 1 to 4)",
            installed("RadeonDeveloperPanelCLI", &[]).or_else(|| installed("RadeonGPUProfiler", &[])),
            "install the Radeon Developer Tool Suite and capture with `RadeonDeveloperPanelCLI --rgp-instruction-tracing`; on Linux with RADV, MESA_VK_TRACE=rgp",
            "per-instruction latency, wave occupancy, cache counters per pass",
        ))),
        "apple" => layers.push(highly(layer(
            "GPU cost per shader line",
            "Xcode (Metal frame capture, shader profiler)",
            installed("xcrun", &[]),
            "install Xcode; capture with MTL_CAPTURE_ENABLED=1 and `--gpu-capture frame.gputrace`, then open it in Xcode (per-line costs; heat maps on M3 and later)",
            "per-shader-line cost and instruction counts",
        ))),
        _ => {}
    }
    if os != "macos" {
        layers.push(layer(
            "frame-exact capture of every command (Vulkan, D3D12)",
            "RenderDoc",
            installed(
                "renderdoccmd",
                &[r"C:\Program Files\RenderDoc\renderdoccmd.exe"],
            ),
            "install RenderDoc; run `renderdoccmd capture -w PROGRAM --offscreen 30 --gpu-capture frame` (Vulkan: WGPU_BACKEND=vulkan)",
            "every command, resource and shader of one frame, with pass names",
        ));
    }
    layers.push(layer(
        "shader machine code and register pressure, any machine",
        "Radeon GPU Analyzer (offline, no GPU needed)",
        installed("rga", &[]),
        "`cargo xtask gpu --report DIR` compiles every shader with it (downloaded on first use)",
        "AMD ISA, VGPR use and occupancy for every shader entry point",
    ));
    layers
}
