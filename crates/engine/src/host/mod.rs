mod adapter;
mod frame_loop;
mod input;
mod settings;

pub use crate::app_host::now_ms;
#[cfg(feature = "dev")]
pub use crate::dev_host::{DevTake, Shot, instrument, instruments, kept_entry};
pub use adapter::AdapterStatus;
pub use engine_panic::{PANIC_GLOBAL, PANIC_PREFIX, PanicReport};
#[cfg(feature = "dev")]
pub(crate) use frame_loop::in_flight;
pub use frame_loop::{FrameLoop, FramePhase};
pub use input::{DeviceGatherer, InputSource};
pub use settings::{HostSettings, OVERLAY, Settings, SettingsLog, TypedSettings};

pub const WHEEL_PIXELS: f32 = 100.0;

pub const SOUND_ORDERS: [(&str, &[&str]); 6] = [
    ("open", &["latency_given", "latency_seconds"]),
    ("decode", &["sound", "length"]),
    (
        "start",
        &["voice", "sound", "gain", "pan", "rate", "looping", "offset_seconds"],
    ),
    ("tune", &["voice", "gain", "pan", "rate", "seconds"]),
    ("stop", &["voice", "seconds"]),
    ("master", &["gain", "seconds"]),
];

#[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
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
    output.push_str(&constant("WHEEL_PIXELS", &serde_json::json!(WHEEL_PIXELS)));
    let phases: serde_json::Map<String, serde_json::Value> = FramePhase::ALL
        .iter()
        .map(|&(phase, name)| (name.to_owned(), (phase as u32).into()))
        .collect();
    output.push_str(&constant("FRAME_PHASES", &phases.into()));
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
    let orders: serde_json::Map<String, serde_json::Value> = SOUND_ORDERS
        .iter()
        .enumerate()
        .map(|(code, (name, fields))| {
            let mut layout = serde_json::json!({"code": code, "size": fields.len() + 1});
            for (at, field) in fields.iter().enumerate() {
                layout[*field] = (at + 1).into();
            }
            ((*name).to_owned(), layout)
        })
        .collect();
    output.push_str(&constant("SOUND_ORDERS", &orders.into()));
    output
}
