use engine_core::smoke::{Checked, ask, control, environment, inspect, number, query};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

const TONE: &str = "sounds/tone.wav";
const BROKEN: &str = "sounds/broken.wav";
const BROKEN_BYTES: &[u8] = b"RIFFbroken";
const DESCRIBED: [&str; 5] = [
    "sound",
    "audio",
    "audio_stats",
    "audio_device",
    "audio_voices",
];
const SAMPLE_RATE: u32 = 8000;
const SAMPLES: u32 = 256;
const BYTES_PER_SAMPLE: u16 = 2;
const BITS_PER_SAMPLE: u16 = 16;
const CHANNELS: u16 = 1;
const PCM: u16 = 1;
const FORMAT_BYTES: u32 = 16;
const HEADER_BYTES: u32 = 36;

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.file(TONE, &wav());
    session.file(BROKEN, BROKEN_BYTES);
    play(session, &empty())?;
    for name in DESCRIBED {
        ask(session, "describe", json!({"name": name}))?;
    }
    quiet(session)?;
    settings(session)?;
    without_device(session)?;
    with_device(session)?;
    play(session, &empty())?;
    session.step(1);
    let stats = inspect(session, "audio_stats")?;
    smoke_eq!(stats["voices"], json!(0), "despawn_stops_voices");
    Ok(())
}

fn play(session: &mut dyn SmokeSession, document: &Value) -> Checked<()> {
    session.source(&document.to_string());
    control(session, "pause", json!({}))?;
    Ok(())
}

fn quiet(session: &mut dyn SmokeSession) -> Checked<()> {
    session.step(1);
    let stats = inspect(session, "audio_stats")?;
    smoke_eq!(stats["device"], json!(false), "no_sounds_no_device");
    smoke_eq!(stats["sounds_loaded"], json!(0), "no_sounds_none_loaded");
    smoke_eq!(stats["voices"], json!(0), "no_sounds_no_voices");
    Ok(())
}

fn settings(session: &mut dyn SmokeSession) -> Checked<()> {
    environment(session, json!({"merge": true, "audio": {"effects": 0.5}}))?;
    let settings = inspect(session, "audio")?;
    smoke_eq!(settings["effects"], json!(0.5), "bus_volume_applies");
    Ok(())
}

fn without_device(session: &mut dyn SmokeSession) -> Checked<()> {
    control(session, "audio_device", json!({"available": false}))?;
    play(session, &sounding())?;
    session.step(1);
    let stats = inspect(session, "audio_stats")?;
    smoke_eq!(stats["device"], json!(false), "missing_device_reported");
    smoke_eq!(stats["voices"], json!(0), "missing_device_plays_nothing");
    let counted = number(&stats["sounds_loaded"])
        + number(&stats["decoding"])
        + number(&stats["decode_failures"]);
    smoke_eq!(counted, 2.0, "every_file_counted");
    smoke_eq!(
        stats["latest_failure"].is_string(),
        true,
        "failure_reported"
    );
    if cfg!(not(target_arch = "wasm32")) {
        smoke_eq!(stats["sounds_loaded"], json!(1), "valid_file_decodes");
        smoke_eq!(stats["decode_failures"], json!(1), "broken_file_fails");
    }
    let voices = query(session, "audio_voices", json!({}))?;
    smoke_eq!(voices["voices"], json!([]), "missing_device_lists_nothing");
    Ok(())
}

fn with_device(session: &mut dyn SmokeSession) -> Checked<()> {
    control(session, "audio_device", json!({"available": true}))?;
    session.step(1);
    let stats = inspect(session, "audio_stats")?;
    smoke_eq!(
        stats["device"] == true,
        number(&stats["voices"]) > 0.0,
        "voices_follow_device"
    );
    Ok(())
}

fn wav() -> Vec<u8> {
    let frame = u32::from(BYTES_PER_SAMPLE);
    let data = SAMPLES * frame;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(HEADER_BYTES + data).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&FORMAT_BYTES.to_le_bytes());
    bytes.extend_from_slice(&PCM.to_le_bytes());
    bytes.extend_from_slice(&CHANNELS.to_le_bytes());
    bytes.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    bytes.extend_from_slice(&(SAMPLE_RATE * frame).to_le_bytes());
    bytes.extend_from_slice(&BYTES_PER_SAMPLE.to_le_bytes());
    bytes.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data.to_le_bytes());
    bytes.resize(bytes.len() + data as usize, 0);
    bytes
}

fn sounding() -> Value {
    json!({
        "id": "audio-fixture",
        "revision": 1,
        "materials": [],
        "meshes": [],
        "entities": [
            {"id": "tone", "sound": {"file": TONE, "bus": "ambient", "looping": true}},
            {"id": "near", "sound": {"file": TONE, "looping": true, "spatial": true, "radius": 5}},
            {"id": "broken", "sound": {"file": BROKEN}},
        ],
    })
}

fn empty() -> Value {
    json!({
        "id": "audio-empty",
        "revision": 1,
        "materials": [],
        "meshes": [],
        "entities": [],
    })
}
