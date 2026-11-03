use engine_core::{Error, ErrorCode, InvalidSetting, Rule, SceneCx, Setting, valid_uri};
use engine_places::audio::SoundSource;
use serde_json::{Value, json};

pub(crate) const MIN_PITCH: f32 = 0.0625;
pub(crate) const MAX_PITCH: f32 = 16.0;
pub(crate) const MIN_RADIUS: f32 = 0.125;
pub(crate) const MAX_RADIUS: f32 = 1_000_000.0;

const VOLUME: Rule = Rule::range(0.0, 1.0);
const PITCH: Rule = Rule::range(MIN_PITCH as f64, MAX_PITCH as f64);
const RADIUS: Rule = Rule::range(MIN_RADIUS as f64, MAX_RADIUS as f64);
const RULES: [(&str, Rule); 3] = [("volume", VOLUME), ("pitch", PITCH), ("radius", RADIUS)];

fn ruled(source: &SoundSource) -> Result<(), InvalidSetting> {
    source.volume.check("volume", VOLUME)?;
    source.pitch.check("pitch", PITCH)?;
    source.radius.check("radius", RADIUS)
}

pub(crate) fn check_sound(source: &SoundSource, path: &str) -> Result<(), Error> {
    valid_uri(&source.file, &format!("{path}.file"))?;
    ruled(source).map_err(|invalid| invalid.error(ErrorCode::InvalidValue, path))
}

pub(crate) fn sound_files(source: &SoundSource) -> Vec<String> {
    vec![source.file.clone()]
}

pub(crate) fn prepare_sound(
    mut source: SoundSource,
    cx: &SceneCx<'_>,
    path: &str,
) -> Result<SoundSource, Error> {
    source.bytes = cx.buffer(&source.file, &format!("{path}.file"))?;
    Ok(source)
}

pub(crate) fn sound_schema() -> Value {
    let mut schema = schemars::schema_for!(SoundSource);
    for (field, rule) in RULES {
        engine_core::state(&mut schema, field, rule);
    }
    serde_json::to_value(schema).unwrap_or(Value::Null)
}

pub(crate) fn sound_example() -> Value {
    json!({"file": "sounds/wind.ogg", "bus": "ambient", "looping": true})
}
