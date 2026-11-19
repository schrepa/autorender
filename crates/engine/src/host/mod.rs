mod adapter;
mod frame_loop;
mod input;
pub mod policy;
mod scene_load;
mod settings;

pub use crate::app_host::now_ms;
#[cfg(feature = "dev")]
pub use crate::dev_host::{DevTake, Shot, instrument, instruments};
pub use adapter::AdapterStatus;
pub use engine_panic::{PANIC_GLOBAL, PANIC_PREFIX, PanicReport};
#[cfg(feature = "dev")]
pub(crate) use frame_loop::in_flight;
pub use frame_loop::{FrameLoop, FramePhase, HostPause};
pub use input::{DeviceGatherer, InputSource};
pub use scene_load::{FileRead, SceneLoad};
pub use settings::{HostSettings, OVERLAY, Settings, SettingsLog, TypedSettings};

#[cfg(all(feature = "schema", not(target_arch = "wasm32")))]
pub fn typescript() -> String {
    use crate::tooling::{constant, declarations};
    let schema =
        |schema: schemars::Schema| serde_json::to_value(schema).expect("schema_serializes");
    let mut output = declarations(&schema(schemars::schema_for!(PanicReport)), "PanicReport")
        .trim_start()
        .to_owned();
    for (name, value) in [
        ("PANIC_PREFIX", PANIC_PREFIX),
        ("PANIC_GLOBAL", PANIC_GLOBAL),
    ] {
        output.push_str(&constant(name, &value.into()));
    }
    for (name, value) in policy::constants() {
        output.push_str(&constant(name, &value));
    }
    let phases: serde_json::Map<String, serde_json::Value> = FramePhase::ALL
        .iter()
        .map(|&(phase, name)| (name.to_owned(), (phase as u32).into()))
        .collect();
    output.push_str(&constant("FRAME_PHASES", &phases.into()));
    let pauses: serde_json::Map<String, serde_json::Value> = HostPause::ALL
        .iter()
        .map(|&(pause, name)| (name.to_owned(), (pause as u32).into()))
        .collect();
    output.push_str(&constant("HOST_PAUSES", &pauses.into()));
    let cursors: serde_json::Map<String, serde_json::Value> = engine_core::CursorMode::ALL
        .iter()
        .map(|&(mode, name)| (name.to_owned(), (mode as u32).into()))
        .collect();
    output.push_str(&constant("CURSOR_MODES", &cursors.into()));
    let touches: serde_json::Map<String, serde_json::Value> = engine_core::TouchPhase::ALL
        .iter()
        .map(|&(phase, name)| (name.to_owned(), (phase as u32).into()))
        .collect();
    output.push_str(&constant("TOUCH_PHASES", &touches.into()));
    output
}
