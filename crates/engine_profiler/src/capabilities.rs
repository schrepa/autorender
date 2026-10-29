use serde_json::{Value, json};

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

fn vendor(adapter: &engine::AdapterInfo) -> &'static str {
    match engine::host::vendor(adapter) {
        "" => "other",
        vendor => vendor,
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

fn layer(name: &str, tools: &[&str], found: Option<String>, commands: &[&str]) -> Value {
    json!({
        "layer": name,
        "priority": "recommended",
        "tools": tools,
        "installed": found,
        "commands": commands,
    })
}

fn highly(mut layer: Value) -> Value {
    layer["priority"] = "highly_recommended".into();
    layer
}

pub(crate) fn recommended(adapter: &engine::AdapterInfo, shallow_stacks: bool) -> Vec<Value> {
    let mut layers = Vec::new();
    let os = std::env::consts::OS;
    if os == "linux" && shallow_stacks {
        layers.push(layer(
            "cpu_stacks_complete",
            &[],
            None,
            &["RUSTFLAGS=-Cforce-frame-pointers=yes"],
        ));
    }
    match os {
        "windows" => layers.push(layer(
            "cpu_sampling_system",
            &["samply", "wpr"],
            installed("samply", &[]).or_else(|| installed("wpr", &[])),
            &["cargo install samply", "samply record PROGRAM"],
        )),
        "linux" => layers.push(layer(
            "cpu_sampling_system",
            &["samply", "perf"],
            installed("samply", &[]).or_else(|| installed("perf", &[])),
            &[
                "sudo sysctl kernel.perf_event_paranoid=1",
                "samply record PROGRAM",
                "perf record -g PROGRAM",
            ],
        )),
        "macos" => layers.push(layer(
            "cpu_sampling_system",
            &["xctrace", "samply"],
            installed("xctrace", &[]).or_else(|| installed("samply", &[])),
            &["samply record PROGRAM"],
        )),
        _ => {}
    }
    layers.push(layer(
        "instruction_counts",
        if os == "linux" {
            &["valgrind"]
        } else {
            &["drrun"]
        },
        installed("valgrind", &[]).or_else(|| installed("drrun", &[])),
        &[],
    ));
    let counted = match vendor(adapter) {
        "intel" => os == "windows" || os == "linux",
        "amd" | "qualcomm" => os == "linux",
        _ => false,
    };
    if counted {
        layers.push(highly(layer(
            "gpu_counters_per_pass",
            &["vulkan_performance_query"],
            Some("driver".into()),
            &["--counters on"],
        )));
    }
    if vendor(adapter) == "nvidia" {
        layers.push(highly(layer(
            "gpu_cost_per_shader_line",
            &["ngfx"],
            installed("ngfx", &[r"C:\Program Files\NVIDIA Corporation\Nsight Graphics */host/windows-desktop-nomad-x64/ngfx.exe"]),
            &["--vendor nsight --vendor-dir DIR"],
        )));
    }
    if os == "windows" {
        layers.push(highly(layer(
            "gpu_counters_per_draw",
            &["pixtool"],
            installed("pixtool", &[r"C:\Program Files\Microsoft PIX\*/pixtool.exe"]),
            &["--offscreen N --vendor pix --vendor-dir DIR"],
        )));
    }
    match vendor(adapter) {
        "amd" => layers.push(highly(layer(
            "gpu_cost_per_instruction",
            &["RadeonDeveloperPanelCLI", "RadeonGPUProfiler"],
            installed("RadeonDeveloperPanelCLI", &[]).or_else(|| installed("RadeonGPUProfiler", &[])),
            &[
                "RadeonDeveloperPanelCLI --rgp-instruction-tracing",
                "MESA_VK_TRACE=rgp",
            ],
        ))),
        "apple" => layers.push(highly(layer(
            "gpu_cost_per_shader_line",
            &["xcrun"],
            installed("xcrun", &[]),
            &["MTL_CAPTURE_ENABLED=1 PROGRAM --gpu-capture frame.gputrace"],
        ))),
        _ => {}
    }
    if os != "macos" {
        layers.push(layer(
            "frame_capture",
            &["renderdoccmd"],
            installed(
                "renderdoccmd",
                &[r"C:\Program Files\RenderDoc\renderdoccmd.exe"],
            ),
            &["renderdoccmd capture -w PROGRAM --offscreen 30 --gpu-capture frame"],
        ));
    }
    layers.push(layer(
        "shader_machine_code",
        &["rga"],
        installed("rga", &[]),
        &["cargo xtask gpu --report DIR"],
    ));
    layers
}
