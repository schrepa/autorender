use super::AdapterStatus;

pub const KNOWN_DIFFERENCES: &[&str] = &[
    "shader_toolchain",
    "presentation_pacing",
    "readback_latency",
    "range_read_io",
    "codegen",
    "timer_resolution",
    "device_limits",
];

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct HostEnvironment {
    pub canvas_width: u32,
    pub canvas_height: u32,
    pub device_pixel_ratio: f64,
    pub user_agent: String,
    pub adapter: AdapterStatus,
    pub known_differences: &'static [&'static str],
}
