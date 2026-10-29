mod agent;
mod agent_lifecycle;
mod frame_loop;
mod gpu_presentation;
mod host_parity;
mod input;
mod native;
mod plugins;
mod requests;

use engine_core::smoke::{SmokeEntry, SmokeResult, SmokeSession};

fn core_smoke(
    name: &'static str,
    gpu: bool,
    run: fn(&mut dyn SmokeSession) -> SmokeResult,
) -> SmokeEntry {
    SmokeEntry {
        name,
        brick: engine_core::CORE_ROW,
        gpu,
        run,
    }
}

pub fn core() -> Vec<SmokeEntry> {
    vec![
        core_smoke("frame_loop", true, frame_loop::frame_loop),
        core_smoke("input_keys", false, input::input_keys),
        core_smoke("input_camera", true, input::input_camera),
        core_smoke("agent", true, agent::agent),
        core_smoke("agent_lifecycle", true, agent_lifecycle::agent_lifecycle),
        core_smoke("host_parity", true, host_parity::host_parity),
        core_smoke("host_images", true, host_parity::host_images),
        core_smoke("host_faults", true, host_parity::host_faults),
        core_smoke("native", true, native::native),
        core_smoke("plugins", true, plugins::plugins),
        core_smoke("gpu_presentation", true, gpu_presentation::gpu_presentation),
    ]
}
