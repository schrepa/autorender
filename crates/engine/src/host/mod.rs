mod adapter;
mod environment;
mod frame_loop;
mod input;
mod jobs;
mod level;
mod pipeline_list;
pub mod policy;
mod resolution;
mod scene_load;
mod settings;
mod views;

pub use crate::app_host::{TerrainClock, now_ms};
pub use adapter::{AdapterStatus, vendor};
pub use engine_core::{
    Distribution, GAMEPLAY_JOBS, IDLE_JOBS, JobId, JobRef, JobRequest, JobShare, MISSING_RECORD,
    RecordFetch, RecordKey, RecordParity, URGENT_JOBS, VIEW_JOBS, content_hash, job_class,
    job_priority, median,
};
pub use engine_panic::{PANIC_GLOBAL, PANIC_PREFIX, PanicReport};
pub use engine_registry::JobList;
pub use environment::{HostEnvironment, KNOWN_DIFFERENCES};
pub use frame_loop::{FrameLoop, FramePhase};
pub use input::{FrameInput, InputGatherer, KEYS, MENU_KEYS, MOVE_KEYS, MenuKey, MoveDirection};
pub(crate) use jobs::{Finished, JobScheduler};
pub use jobs::{JobMessage, JobOrder};
pub(crate) use level::{LevelLoad, LevelStep};
pub(crate) use pipeline_list::PipelineList;
pub(crate) use resolution::DynamicScale;
pub use resolution::{render_size, view_size};
pub use scene_load::{
    FileRead, IMAGE_FORMATS, Image, ImageFormat, SceneLoad, decode_image, image_format,
};
pub use settings::{
    HostSettings, OVERLAY, SETTINGS, Setting, apply_setting, has_setting, reset_setting,
    run_operation,
};
pub use views::{show_look, show_view};

#[cfg(all(feature = "schema", not(target_arch = "wasm32")))]
pub fn typescript() -> String {
    use crate::tooling::{constant, declarations};
    let schema =
        |schema: schemars::Schema| serde_json::to_value(schema).expect("schema_serializes");
    let mut output = declarations(&schema(schemars::schema_for!(Distribution)), "Distribution")
        .trim_start()
        .to_owned();
    output.push_str(&declarations(
        &schema(schemars::schema_for!(PanicReport)),
        "PanicReport",
    ));
    output.push_str(&declarations(
        &schema(schemars::schema_for!(AdapterStatus)),
        "AdapterStatus",
    ));
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
    let cursors: serde_json::Map<String, serde_json::Value> = engine_core::CursorMode::ALL
        .iter()
        .map(|&(mode, name)| (name.to_owned(), (mode as u32).into()))
        .collect();
    output.push_str(&constant("CURSOR_MODES", &cursors.into()));
    let formats: Vec<serde_json::Value> = IMAGE_FORMATS
        .iter()
        .map(|format| format.name.into())
        .collect();
    output.push_str(&constant("IMAGE_FORMATS", &formats.into()));
    let settings: serde_json::Map<String, serde_json::Value> = SETTINGS
        .iter()
        .map(|setting| (setting.name.to_owned(), setting.accepted().into()))
        .collect();
    output.push_str(&constant("HOST_SETTINGS", &settings.into()));
    output.push_str(&constant("HOST_OVERLAY", &OVERLAY.into()));
    output
}
